"""
Authentication Endpoints

FastAPI endpoints for user authentication, registration,
and session management.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.dependecies import get_current_active_user, get_db
from app.core.security import create_access_token, verify_password
from app.schemas.auth import (
    LoginRequest,
    PasswordChange,
    PasswordResetConfirm,
    PasswordResetRequest,
    RegisterRequest,
    Token,
    UserProfile,
    UserProfileUpdate,
)
from app.schemas.common import SuccessResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


@router.post(
    "/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED
)
async def register(
    user_data: RegisterRequest, db: AsyncSession = Depends(get_db)
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
    )

    return UserProfile.model_validate(user)


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest, db: AsyncSession = Depends(get_db)) -> Token:
    """
    Login user and return access tokens

    Args:
        login_data: Login credentials
        db: Database session

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If credentials are invalid
    """
    auth_service = AuthService(db)
    settings = get_settings()

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

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    # Create refresh token
    refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)
    refresh_token = create_access_token(
        data={"sub": str(user.id), "type": "refresh"},
        expires_delta=refresh_token_expires,
    )

    return Token(
        access_token=access_token, refresh_token=refresh_token, token_type="bearer"
    )


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


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    current_user: dict = Depends(get_current_active_user),
) -> SuccessResponse:
    """
    Logout current user

    Args:
        current_user: Current authenticated user

    Returns:
        Success response
    """
    # In a real implementation, you might want to blacklist the token
    # or store logout information

    return SuccessResponse(
        success=True,
        message="Logged out successfully",
        timestamp=str(datetime.now(UTC).isoformat()),
    )
