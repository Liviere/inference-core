"""
Refresh token session store (Redis)

Stores refresh token sessions keyed by a token ID (jti), with TTL matching token expiry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from jose import jwt

from inference_core.core.config import get_settings
from inference_core.core.redis_client import get_redis


class RefreshSessionStore:
    def __init__(self):
        self.settings = get_settings()
        self.redis = get_redis()
        self.prefix = self.settings.redis_refresh_prefix

    def _key(self, jti: str) -> str:
        return f"{self.prefix}{jti}"

    async def add(self, jti: str, user_id: str, exp: int) -> None:
        """Register a refresh token session with TTL until exp."""
        ttl = max(0, exp - int(datetime.now(timezone.utc).timestamp()))
        try:
            # Store minimal payload to validate rotation or revoke
            await self.redis.hset(self._key(jti), mapping={"sub": user_id, "exp": exp})
            if ttl > 0:
                await self.redis.expire(self._key(jti), ttl)
        except Exception:
            # Best-effort only; ignore Redis failures
            return

    async def exists(self, jti: str) -> bool:
        try:
            return await self.redis.exists(self._key(jti)) == 1
        except Exception:
            # If Redis unavailable, treat as not existing to be safe
            return False

    async def revoke(self, jti: str) -> None:
        try:
            await self.redis.delete(self._key(jti))
        except Exception:
            return

    async def get_subject(self, jti: str) -> Optional[str]:
        try:
            data = await self.redis.hgetall(self._key(jti))
            return data.get("sub") if data else None
        except Exception:
            return None

    async def decode_and_validate_refresh(self, token: str) -> dict:
        """Decode refresh token and ensure session exists in Redis."""
        payload = jwt.decode(
            token,
            self.settings.secret_key,
            algorithms=[self.settings.algorithm],
        )
        if payload.get("type") != "refresh":
            raise ValueError("Not a refresh token")
        jti = payload.get("jti")
        sub = payload.get("sub")
        exp = payload.get("exp")
        if not jti or not sub or not exp:
            raise ValueError("Malformed refresh token")
        if not await self.exists(jti):
            raise ValueError("Refresh session not found or revoked")
        return payload
