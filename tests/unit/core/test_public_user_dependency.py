"""
Tests for ``get_current_user_or_public`` dependency.

WHY: This dependency gates agent-instance endpoints. It must:
  * fall back to the seeded public user ONLY when access mode is 'public',
  * still reject malformed/invalid JWTs even in public mode,
  * reject unauthenticated callers in user/superuser modes,
  * fail closed (503) if the public user row is missing,
  * return the authenticated user when a valid JWT is present.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from inference_core.core.dependecies import get_current_user_or_public
from inference_core.core.public_user import PUBLIC_USER_ID


def _mock_settings(mode: str) -> MagicMock:
    s = MagicMock()
    s.llm_api_access_mode = mode
    return s


def _credentials(token: str = "some-token") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _public_user_payload() -> dict:
    return {
        "id": str(PUBLIC_USER_ID),
        "username": "public",
        "email": "public@inference-core.local",
        "first_name": "Public",
        "last_name": "User",
        "is_active": True,
        "is_superuser": False,
        "is_verified": True,
        "created_at": None,
        "updated_at": None,
        "is_public_anonymous": True,
    }


class TestPublicModeNoCredentials:
    @pytest.mark.asyncio
    async def test_returns_public_user_when_seeded(self):
        db = MagicMock()
        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("public"),
        ), patch(
            "inference_core.core.public_user.get_public_user_dict",
            new=AsyncMock(return_value=_public_user_payload()),
        ):
            user = await get_current_user_or_public(credentials=None, db=db)

        assert user["id"] == str(PUBLIC_USER_ID)
        assert user["is_public_anonymous"] is True

    @pytest.mark.asyncio
    async def test_raises_503_when_public_user_missing(self):
        db = MagicMock()
        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("public"),
        ), patch(
            "inference_core.core.public_user.get_public_user_dict",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(credentials=None, db=db)

        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_raises_503_when_public_user_inactive(self):
        db = MagicMock()
        payload = _public_user_payload()
        payload["is_active"] = False
        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("public"),
        ), patch(
            "inference_core.core.public_user.get_public_user_dict",
            new=AsyncMock(return_value=payload),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(credentials=None, db=db)

        assert exc.value.status_code == 503


class TestNonPublicModesRejectAnonymous:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["user", "superuser"])
    async def test_no_credentials_raises_401(self, mode: str):
        db = MagicMock()
        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings(mode),
        ):
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(credentials=None, db=db)

        assert exc.value.status_code == 401


class TestInvalidTokenAlwaysRejected:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["public", "user", "superuser"])
    async def test_invalid_token_raises_401(self, mode: str):
        db = MagicMock()
        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings(mode),
        ), patch("inference_core.core.dependecies.security_manager") as sec:
            sec.verify_token.return_value = None
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(credentials=_credentials("bad"), db=db)

        assert exc.value.status_code == 401


class TestValidTokenReturnsRealUser:
    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self):
        user_uuid = uuid.uuid4()

        # Mock ORM user row
        user_row = MagicMock()
        user_row.id = user_uuid
        user_row.username = "alice"
        user_row.email = "alice@example.com"
        user_row.first_name = "Alice"
        user_row.last_name = "A"
        user_row.is_active = True
        user_row.is_superuser = False
        user_row.is_verified = True
        user_row.created_at = None
        user_row.updated_at = None

        # Mock DB execute → result.scalar_one_or_none()
        result = MagicMock()
        result.scalar_one_or_none.return_value = user_row
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        token_data = MagicMock()
        token_data.user_id = str(user_uuid)

        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("public"),
        ), patch("inference_core.core.dependecies.security_manager") as sec:
            sec.verify_token.return_value = token_data
            user = await get_current_user_or_public(
                credentials=_credentials("good"), db=db
            )

        assert user["id"] == str(user_uuid)
        assert user["username"] == "alice"
        # Authenticated users are NEVER flagged as anonymous.
        assert "is_public_anonymous" not in user

    @pytest.mark.asyncio
    async def test_valid_token_but_inactive_user_raises_400(self):
        user_uuid = uuid.uuid4()
        user_row = MagicMock()
        user_row.id = user_uuid
        user_row.is_active = False

        result = MagicMock()
        result.scalar_one_or_none.return_value = user_row
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        token_data = MagicMock()
        token_data.user_id = str(user_uuid)

        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("public"),
        ), patch("inference_core.core.dependecies.security_manager") as sec:
            sec.verify_token.return_value = token_data
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(
                    credentials=_credentials("good"), db=db
                )

        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_token_but_user_not_found_raises_404(self):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        token_data = MagicMock()
        token_data.user_id = str(uuid.uuid4())

        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("public"),
        ), patch("inference_core.core.dependecies.security_manager") as sec:
            sec.verify_token.return_value = token_data
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(
                    credentials=_credentials("good"), db=db
                )

        assert exc.value.status_code == 404


class TestPublicUserPasswordLocked:
    """Sanity check: the locked hash must never validate via bcrypt."""

    def test_locked_hash_rejected_by_security_manager(self):
        from inference_core.core.public_user import PUBLIC_USER_LOCKED_PASSWORD_HASH
        from inference_core.core.security import security_manager

        # Any plain text, including the sentinel string itself, must fail.
        assert (
            security_manager.verify_password(
                "!locked!public-user-no-login!",
                PUBLIC_USER_LOCKED_PASSWORD_HASH,
            )
            is False
        )
        assert (
            security_manager.verify_password(
                "anything", PUBLIC_USER_LOCKED_PASSWORD_HASH
            )
            is False
        )


def _user_row(
    user_uuid: uuid.UUID, *, is_active: bool = True, is_superuser: bool = False
) -> MagicMock:
    row = MagicMock()
    row.id = user_uuid
    row.username = "u"
    row.email = "u@example.com"
    row.first_name = "U"
    row.last_name = "U"
    row.is_active = is_active
    row.is_superuser = is_superuser
    row.is_verified = True
    row.created_at = None
    row.updated_at = None
    return row


class TestSuperuserModeEnforcement:
    """In mode=='superuser' a valid but non-superuser JWT must be rejected."""

    @pytest.mark.asyncio
    async def test_superuser_mode_rejects_regular_user(self):
        user_uuid = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = _user_row(
            user_uuid, is_superuser=False
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        token_data = MagicMock()
        token_data.user_id = str(user_uuid)

        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("superuser"),
        ), patch(
            "inference_core.core.dependecies.security_manager"
        ) as sec:
            sec.verify_token.return_value = token_data
            with pytest.raises(HTTPException) as exc:
                await get_current_user_or_public(
                    credentials=_credentials("good"), db=db
                )

        assert exc.value.status_code == 403

    @pytest.mark.asyncio
    async def test_superuser_mode_accepts_superuser(self):
        user_uuid = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = _user_row(
            user_uuid, is_superuser=True
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        token_data = MagicMock()
        token_data.user_id = str(user_uuid)

        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("superuser"),
        ), patch(
            "inference_core.core.dependecies.security_manager"
        ) as sec:
            sec.verify_token.return_value = token_data
            user = await get_current_user_or_public(
                credentials=_credentials("good"), db=db
            )

        assert user["is_superuser"] is True

    @pytest.mark.asyncio
    async def test_user_mode_accepts_regular_user(self):
        user_uuid = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = _user_row(
            user_uuid, is_superuser=False
        )
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)

        token_data = MagicMock()
        token_data.user_id = str(user_uuid)

        with patch(
            "inference_core.core.dependecies.get_settings",
            return_value=_mock_settings("user"),
        ), patch(
            "inference_core.core.dependecies.security_manager"
        ) as sec:
            sec.verify_token.return_value = token_data
            user = await get_current_user_or_public(
                credentials=_credentials("good"), db=db
            )

        assert user["is_superuser"] is False
