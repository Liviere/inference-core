"""``auth_verify_email_request_endpoint_enabled`` switches the public route off."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from inference_core.api.v1.routes import auth
from inference_core.core.config import Settings
from inference_core.schemas.auth import EmailVerificationRequest

REQUEST = EmailVerificationRequest(email="someone@example.com")


def test_on_by_default():
    field = Settings.model_fields["auth_verify_email_request_endpoint_enabled"]
    assert field.default is True


@pytest.mark.asyncio
async def test_switched_off_it_answers_404_and_sends_nothing():
    service = SimpleNamespace(request_verification_email=AsyncMock())
    settings = SimpleNamespace(auth_verify_email_request_endpoint_enabled=False)
    with patch.object(auth, "AuthService", return_value=service):
        with pytest.raises(HTTPException) as refusal:
            await auth.request_verification_email(
                REQUEST, db=object(), settings=settings
            )
    assert refusal.value.status_code == 404
    service.request_verification_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_switched_on_it_sends():
    service = SimpleNamespace(request_verification_email=AsyncMock())
    settings = SimpleNamespace(auth_verify_email_request_endpoint_enabled=True)
    with patch.object(auth, "AuthService", return_value=service):
        answer = await auth.request_verification_email(
            REQUEST, db=object(), settings=settings
        )
    assert answer.success is True
    service.request_verification_email.assert_awaited_once_with("someone@example.com")
