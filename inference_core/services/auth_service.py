"""
Authentication Service

Business logic for user authentication, registration, and session management.
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.core.security import (
    get_password_hash,
    security_manager,
    verify_password,
)
from inference_core.database.sql.models.user import User
from inference_core.schemas.auth import RegisterRequest


class AuthService:
    """Authentication service for user management"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID

        Args:
            user_id: User ID

        Returns:
            User instance or None
        """
        result = await self.db.execute(
            select(User).where(User.id == user_id, User.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email

        Args:
            email: User email

        Returns:
            User instance or None
        """
        result = await self.db.execute(
            select(User).where(User.email == email, User.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username

        Args:
            username: Username

        Returns:
            User instance or None
        """
        result = await self.db.execute(
            select(User).where(User.username == username, User.is_deleted == False)
        )
        return result.scalar_one_or_none()

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username/email and password

        Args:
            username: Username or email
            password: Password

        Returns:
            User instance if authenticated, None otherwise
        """
        # Try to find user by username or email
        user = await self.get_user_by_username(username)
        if not user:
            user = await self.get_user_by_email(username)

        if not user:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        return user

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        is_active: bool = True,
        is_superuser: bool = False,
        is_verified: bool = False,
    ) -> User:
        """
        Create new user

        Args:
            username: Username
            email: Email
            password: Password
            first_name: First name
            last_name: Last name
            is_active: Whether user is active
            is_superuser: Whether user is superuser
            is_verified: Whether email is verified

        Returns:
            Created user instance
        """
        hashed_password = get_password_hash(password)

        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            is_active=is_active,
            is_superuser=is_superuser,
            is_verified=is_verified,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def update_user_profile(
        self,
        user_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Optional[User]:
        """
        Update user profile

        Args:
            user_id: User ID
            first_name: New first name
            last_name: New last name
            email: New email

        Returns:
            Updated user instance or None
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return None

        if first_name is not None:
            user.first_name = first_name
        if last_name is not None:
            user.last_name = last_name
        if email is not None:
            user.email = email

        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def update_user_password(self, user_id: str, new_password: str) -> bool:
        """
        Update user password

        Args:
            user_id: User ID
            new_password: New password

        Returns:
            True if updated successfully
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.hashed_password = get_password_hash(new_password)
        await self.db.commit()

        return True

    async def request_password_reset(self, email: str) -> bool:
        """
        Request password reset

        Args:
            email: User email

        Returns:
            True if reset token generated
        """
        user = await self.get_user_by_email(email)
        if not user:
            # Don't reveal if email exists
            return True

        # Generate reset token
        reset_token = security_manager.generate_password_reset_token(email)

        # In a real application, send email with reset link
        # For now, just log the token (remove in production)
        print(f"Password reset token for {email}: {reset_token}")

        return True

    async def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset password with token

        Args:
            token: Reset token
            new_password: New password

        Returns:
            True if password reset successfully
        """
        email = security_manager.verify_password_reset_token(token)
        if not email:
            return False

        user = await self.get_user_by_email(email)
        if not user:
            return False

        user.hashed_password = get_password_hash(new_password)
        await self.db.commit()

        return True

    async def deactivate_user(self, user_id: str) -> bool:
        """
        Deactivate user

        Args:
            user_id: User ID

        Returns:
            True if deactivated successfully
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.is_active = False
        await self.db.commit()

        return True

    async def activate_user(self, user_id: str) -> bool:
        """
        Activate user

        Args:
            user_id: User ID

        Returns:
            True if activated successfully
        """
        user = await self.get_user_by_id(user_id)
        if not user:
            return False

        user.is_active = True
        await self.db.commit()

        return True
