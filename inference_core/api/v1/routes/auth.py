"""
Authentication Endpoints

FastAPI endpoints for user authentication, registration,
and session management.
"""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.config import Settings, get_settings
from inference_core.core.dependecies import get_current_active_user, get_db
from inference_core.core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
)
from inference_core.schemas.auth import (
    AccessToken,
    EmailVerificationConfirm,
    EmailVerificationRequest,
    LoginRequest,
    PasswordChange,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshIntrospection,
    RegisterRequest,
    UserProfile,
    UserProfileUpdate,
)
from inference_core.schemas.common import SuccessResponse
from inference_core.services.auth_service import AuthService
from inference_core.services.refresh_session_store import RefreshSessionStore

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()
logger = logging.getLogger(__name__)


@router.post(
    "/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED
)
async def register(
    user_data: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserProfile:
    """
    Register a new user

    Args:
        user_data: User registration data
        db: Database session

    Returns:
        Created user profile

    Raises:
        HTTPException: If username or email already exists
    """
    auth_service = AuthService(db)

    # Check if user already exists
    existing_user = await auth_service.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    existing_user = await auth_service.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
        )

    # Create new user
    user = await auth_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        is_active=settings.auth_register_default_active,  # Use config setting
    )

    # Send verification email if enabled
    if settings.auth_send_verification_email_on_register:
        try:
            verification_token = auth_service.create_email_verification_token(
                str(user.id)
            )
            await auth_service.send_verification_email(user, verification_token)
        except Exception as e:
            # Log error but don't fail registration
            logger.error(f"Failed to send verification email during registration: {e}")

    return UserProfile.model_validate(user)


@router.post("/login", response_model=AccessToken)
async def login(
    login_data: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AccessToken:
    """
    Login user and return access token (refresh token set as HttpOnly cookie)

    Args:
        login_data: Login credentials
        response: FastAPI response object for setting cookies
        db: Database session

    Returns:
        Access token only (refresh token in cookie)

    Raises:
        HTTPException: If credentials are invalid
    """
    auth_service = AuthService(db)

    # Authenticate user
    user = await auth_service.authenticate_user(
        login_data.username, login_data.password
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active (existing behavior, now configurable)
    if settings.auth_login_require_active and not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Check if user email is verified (new behavior)
    if settings.auth_login_require_verified and not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email not verified"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    # Create refresh token and register session in Redis
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    try:
        payload = jwt.decode(
            refresh_token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        jti = payload.get("jti")
        exp = payload.get("exp")
        if jti and exp:
            store = RefreshSessionStore()
            await store.add(jti=jti, user_id=str(user.id), exp=int(exp))
    except Exception:
        # If Redis unavailable, we still return tokens (session-less fallback)
        pass

    # Set refresh token as HttpOnly cookie
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=refresh_token,
        max_age=settings.refresh_cookie_max_age,
        path=settings.refresh_cookie_path,
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite=settings.refresh_cookie_samesite,
        domain=settings.refresh_cookie_domain,
    )

    return AccessToken(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: dict = Depends(get_current_active_user),
) -> UserProfile:
    """
    Get current user profile

    Args:
        current_user: Current authenticated user

    Returns:
        User profile
    """
    return UserProfile(**current_user)


@router.put("/me", response_model=UserProfile)
async def update_profile(
    profile_data: UserProfileUpdate,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    """
    Update current user profile

    Args:
        profile_data: Profile update data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated user profile
    """
    auth_service = AuthService(db)

    update_dict = profile_data.model_dump(exclude_unset=True)
    user = await auth_service.update_user_profile(
        user_id=current_user["id"], **update_dict
    )

    return UserProfile.model_validate(user)


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """
    Change user password

    Args:
        password_data: Password change data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Success response

    Raises:
        HTTPException: If current password is incorrect
    """
    auth_service = AuthService(db)

    # Verify current password
    user = await auth_service.get_user_by_id(current_user["id"])
    if not user or not verify_password(
        password_data.current_password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect current password"
        )

    # Update password
    await auth_service.update_user_password(
        user_id=current_user["id"], new_password=password_data.new_password
    )

    return SuccessResponse(
        success=True,
        message="Password changed successfully",
        timestamp=str(datetime.now(UTC).isoformat()),
    )


@router.post("/forgot-password", response_model=SuccessResponse)
async def forgot_password(
    reset_data: PasswordResetRequest, db: AsyncSession = Depends(get_db)
) -> SuccessResponse:
    """
    Request password reset

    Args:
        reset_data: Password reset request data
        db: Database session

    Returns:
        Success response
    """
    auth_service = AuthService(db)

    # Generate reset token and send email
    await auth_service.request_password_reset(reset_data.email)

    return SuccessResponse(
        success=True,
        message="Password reset instructions sent to email",
        timestamp=str(datetime.now(UTC).isoformat()),
    )


@router.post("/reset-password", response_model=SuccessResponse)
async def reset_password(
    reset_data: PasswordResetConfirm, db: AsyncSession = Depends(get_db)
) -> SuccessResponse:
    """
    Reset password with token

    Args:
        reset_data: Password reset confirmation data
        db: Database session

    Returns:
        Success response

    Raises:
        HTTPException: If token is invalid or expired
    """
    auth_service = AuthService(db)

    # Reset password
    success = await auth_service.reset_password(
        token=reset_data.token, new_password=reset_data.new_password
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    return SuccessResponse(
        success=True,
        message="Password reset successfully",
        timestamp=str(datetime.now(UTC).isoformat()),
    )


@router.post("/refresh", response_model=AccessToken)
async def refresh_tokens(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> AccessToken:
    """
    Exchange a refresh token (from cookie) for a new access token and rotated refresh token.
    """
    store = RefreshSessionStore()

    # Extract refresh token from cookie
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found in cookie",
        )

    # Decode and validate that session exists
    try:
        decoded = jwt.decode(
            refresh_token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        if decoded.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token type"
            )
        jti = decoded.get("jti")
        sub = decoded.get("sub")
        if not jti or not sub:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malformed refresh token",
            )
        if not await store.exists(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked"
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    # Rotate: revoke old, issue new refresh, store session
    await store.revoke(jti)
    access_token = create_access_token(data={"sub": str(sub)})
    new_refresh = create_refresh_token(data={"sub": str(sub)})
    try:
        decoded_new = jwt.decode(
            new_refresh, settings.secret_key, algorithms=[settings.algorithm]
        )
        new_jti = decoded_new.get("jti")
        new_exp = decoded_new.get("exp")
        if new_jti and new_exp:
            await store.add(jti=new_jti, user_id=str(sub), exp=int(new_exp))
    except Exception:
        pass

    # Set new refresh token as HttpOnly cookie
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=new_refresh,
        max_age=settings.refresh_cookie_max_age,
        path=settings.refresh_cookie_path,
        httponly=True,
        secure=settings.refresh_cookie_secure,
        samesite=settings.refresh_cookie_samesite,
        domain=settings.refresh_cookie_domain,
    )

    return AccessToken(access_token=access_token, token_type="bearer")


@router.get("/introspect", response_model=RefreshIntrospection)
async def introspect_refresh_token(
    request: Request, settings: Settings = Depends(get_settings)
) -> RefreshIntrospection:
    """Validate refresh token cookie WITHOUT rotating it.

    Returns a minimal payload indicating whether the refresh token session is
    active. Does NOT issue new tokens or revoke the existing one – safe for
    high-frequency middleware / SSR checks.
    """
    store = RefreshSessionStore()

    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if not refresh_token:
        return RefreshIntrospection(active=False)

    try:
        decoded = jwt.decode(
            refresh_token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        if decoded.get("type") != "refresh":
            return RefreshIntrospection(active=False)
        jti = decoded.get("jti")
        sub = decoded.get("sub")
        exp = decoded.get("exp")
        if not jti or not sub or not exp:
            return RefreshIntrospection(active=False)
        if not await store.exists(jti):
            return RefreshIntrospection(active=False)
        return RefreshIntrospection(active=True, user_id=str(sub), expires_at=int(exp))
    except Exception:
        return RefreshIntrospection(active=False)


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> SuccessResponse:
    """
    Logout current user by revoking refresh token session and clearing cookie

    Returns:
        Success response
    """
    store = RefreshSessionStore()

    # Extract refresh token from cookie
    refresh_token = request.cookies.get(settings.refresh_cookie_name)

    # Best-effort revoke of refresh token if present
    if refresh_token:
        try:
            decoded = jwt.decode(
                refresh_token, settings.secret_key, algorithms=[settings.algorithm]
            )
            if decoded.get("type") == "refresh" and decoded.get("jti"):
                await store.revoke(decoded["jti"])
        except Exception:
            pass

    # Clear the refresh token cookie
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=settings.refresh_cookie_path,
        domain=settings.refresh_cookie_domain,
    )

    return SuccessResponse(
        success=True,
        message="Logged out successfully",
        timestamp=str(datetime.now(UTC).isoformat()),
    )


@router.post("/verify-email/request", response_model=SuccessResponse)
async def request_verification_email(
    request_data: EmailVerificationRequest, db: AsyncSession = Depends(get_db)
) -> SuccessResponse:
    """
    Request email verification for user

    Args:
        request_data: Email verification request data
        db: Database session

    Returns:
        Success response
    """
    auth_service = AuthService(db)

    # Send verification email
    await auth_service.request_verification_email(request_data.email)

    return SuccessResponse(
        success=True,
        message="Verification email sent if account exists",
        timestamp=str(datetime.now(UTC).isoformat()),
    )


@router.post("/verify-email", response_model=SuccessResponse)
async def verify_email(
    verification_data: EmailVerificationConfirm, db: AsyncSession = Depends(get_db)
) -> SuccessResponse:
    """
    Verify user email with token

    Args:
        verification_data: Email verification confirmation data
        db: Database session

    Returns:
        Success response

    Raises:
        HTTPException: If token is invalid or expired
    """
    auth_service = AuthService(db)

    # Verify email
    success = await auth_service.verify_email_with_token(verification_data.token)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )

    return SuccessResponse(
        success=True,
        message="Email verified successfully",
        timestamp=str(datetime.now(UTC).isoformat()),
    )
