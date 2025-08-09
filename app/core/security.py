"""
Security and Authentication Module

Handles JWT tokens, password hashing, and authentication dependencies.
"""

import secrets
from datetime import UTC, datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import ValidationError

from app.core.config import get_settings

# Import TokenData after schemas are available
try:
    from app.schemas.auth import TokenData
except ImportError:
    # Fallback TokenData definition if schemas not available
    from pydantic import BaseModel

    class TokenData(BaseModel):
        user_id: str


# Security instances
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Centralized security management"""

    def __init__(self):
        self.settings = get_settings()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """
        Hash a password

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token

        Args:
            data: Data to encode in token
            expires_delta: Token expiration time

        Returns:
            JWT token string
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(
                minutes=self.settings.access_token_expire_minutes
            )

        # mark as access token and include random jti for potential future denylist
        to_encode.update(
            {
                "exp": expire,
                "type": "access",
                "jti": secrets.token_urlsafe(16),
            }
        )
        encoded_jwt = jwt.encode(
            to_encode, self.settings.secret_key, algorithm=self.settings.algorithm
        )
        return encoded_jwt

    def create_refresh_token(self, data: dict) -> str:
        """
        Create JWT refresh token

        Args:
            data: Data to encode in token

        Returns:
            JWT refresh token string
        """
        to_encode = data.copy()
        expire = datetime.now(UTC) + timedelta(
            days=self.settings.refresh_token_expire_days
        )
        # add type and jti for tracking in Redis
        to_encode.update(
            {
                "exp": expire,
                "type": "refresh",
                "jti": secrets.token_urlsafe(24),
            }
        )

        encoded_jwt = jwt.encode(
            to_encode, self.settings.secret_key, algorithm=self.settings.algorithm
        )
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            Decoded token data or None if invalid
        """
        try:
            payload = jwt.decode(
                token, self.settings.secret_key, algorithms=[self.settings.algorithm]
            )
            # Enforce access tokens only for auth-protected dependencies
            if payload.get("type") != "access":
                return None
            user_id: str = payload.get("sub")
            if user_id is None:
                return None

            token_data = TokenData(user_id=user_id)
            return token_data

        except (JWTError, ValidationError):
            return None

    def generate_password_reset_token(self, email: str) -> str:
        """
        Generate password reset token

        Args:
            email: User email

        Returns:
            Password reset token
        """
        delta = timedelta(hours=1)  # Reset token expires in 1 hour
        now = datetime.now(UTC)
        expires = now + delta
        # Use integer numeric dates for compatibility
        exp = int(expires.timestamp())
        nbf = int(now.timestamp())
        encoded_jwt = jwt.encode(
            {"exp": exp, "nbf": nbf, "sub": email, "type": "reset"},
            self.settings.secret_key,
            algorithm=self.settings.algorithm,
        )
        return encoded_jwt

    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """
        Verify password reset token

        Args:
            token: Password reset token

        Returns:
            Email if token is valid, None otherwise
        """
        try:
            decoded_token = jwt.decode(
                token, self.settings.secret_key, algorithms=[self.settings.algorithm]
            )

            if decoded_token.get("type") != "reset":
                return None

            return decoded_token["sub"]
        except JWTError:
            return None

    @staticmethod
    def generate_random_string(length: int = 32) -> str:
        """
        Generate a random string

        Args:
            length: Length of string to generate

        Returns:
            Random string
        """
        # URL-safe random string
        return secrets.token_urlsafe(length)


# Global security manager instance
security_manager = SecurityManager()


# Dependency functions
async def get_current_user_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    Dependency to get current user from JWT token

    Args:
        credentials: HTTP Authorization credentials

    Returns:
        Token data

    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token_data = security_manager.verify_token(credentials.credentials)
        if token_data is None:
            raise credentials_exception
        return token_data
    except JWTError:
        raise credentials_exception


async def get_current_user_id(
    token_data: TokenData = Depends(get_current_user_token),
) -> str:
    """
    Dependency to get current user ID

    Args:
        token_data: Token data from JWT

    Returns:
        User ID
    """
    return token_data.user_id


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Convenience function to create access token

    Args:
        data: Data to encode
        expires_delta: Expiration time

    Returns:
        JWT token
    """
    return security_manager.create_access_token(data, expires_delta)


def create_refresh_token(data: dict) -> str:
    """
    Convenience function to create refresh token
    """
    return security_manager.create_refresh_token(data)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Convenience function to verify password

    Args:
        plain_password: Plain password
        hashed_password: Hashed password

    Returns:
        True if password matches
    """
    return security_manager.verify_password(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Convenience function to hash password

    Args:
        password: Plain password

    Returns:
        Hashed password
    """
    return security_manager.get_password_hash(password)
