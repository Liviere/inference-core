"""
LLM Usage Logging Module

Handles usage normalization, pricing calculation, and persistent logging
for all LLM interactions (standard, streaming, Celery).
"""

import logging
import time
import uuid
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from inference_core.celery.async_utils import run_async_safely
from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.llm_request_log import LLMRequestLog
from inference_core.database.sql.models.pricing_snapshot import LLMPricingSnapshot
from inference_core.llm.config import PricingConfig, UsageLoggingConfig

logger = logging.getLogger(__name__)

# In-memory cache for pricing snapshot ids to minimize DB lookups during a process lifetime
_PRICING_SNAPSHOT_CACHE: dict[str, uuid.UUID] = {}


def reset_usage_logging_cache() -> None:
    """Reset the in-memory pricing snapshot cache.

    Use this in Jupyter notebooks or multi-event-loop environments
    alongside reset_database_for_new_event_loop() when reinitializing
    connections.
    """
    global _PRICING_SNAPSHOT_CACHE
    _PRICING_SNAPSHOT_CACHE = {}
    logger.debug("Usage logging pricing snapshot cache cleared")


class UsageNormalizer:
    """Handles normalization of usage data from different providers"""

    @staticmethod
    def normalize_usage(
        raw_usage: Dict[str, Any], key_aliases: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Normalize usage data using key aliases

        Args:
            raw_usage: Raw usage data from provider
            key_aliases: Mapping of provider keys to normalized keys

        Returns:
            Normalized usage dictionary
        """
        normalized = {}

        # Apply aliases first
        for provider_key, normalized_key in key_aliases.items():
            if provider_key in raw_usage:
                normalized[normalized_key] = raw_usage[provider_key]

        # Copy remaining keys directly
        for key, value in raw_usage.items():
            if key not in key_aliases:
                normalized[key] = value

        return normalized


class PricingCalculator:
    """Handles cost calculations for LLM usage"""

    @staticmethod
    def compute_cost(
        normalized_usage: Dict[str, Any], pricing_config: PricingConfig
    ) -> Dict[str, Any]:
        """
        Compute costs from normalized usage data

        Args:
            normalized_usage: Normalized usage data
            pricing_config: Pricing configuration

        Returns:
            Cost calculation result
        """
        result = {
            "cost_input_usd": None,
            "cost_output_usd": None,
            "cost_extras_usd": 0.0,
            "cost_total_usd": 0.0,
            "extra_costs": {},
            "cost_estimated": False,
            "context_multiplier": 1.0,
        }

        # Extract core tokens
        input_tokens = normalized_usage.get("input_tokens", 0)
        output_tokens = normalized_usage.get("output_tokens", 0)

        # Determine context multiplier
        total_tokens = input_tokens + output_tokens
        context_multiplier = PricingCalculator._get_context_multiplier(
            total_tokens, pricing_config.context_tiers
        )
        result["context_multiplier"] = context_multiplier

        # Calculate core costs
        if input_tokens:
            result["cost_input_usd"] = PricingCalculator._calculate_dimension_cost(
                input_tokens, pricing_config.input.cost_per_1k, context_multiplier
            )

        if output_tokens:
            result["cost_output_usd"] = PricingCalculator._calculate_dimension_cost(
                output_tokens, pricing_config.output.cost_per_1k, context_multiplier
            )

        # Calculate extra dimension costs
        extra_costs_total = 0.0
        for dim_name, dim_pricing in pricing_config.extras.items():
            if dim_name in normalized_usage:
                tokens = normalized_usage[dim_name]
                if tokens:
                    cost = PricingCalculator._calculate_dimension_cost(
                        tokens, dim_pricing.cost_per_1k, context_multiplier
                    )
                    result["extra_costs"][dim_name] = cost
                    extra_costs_total += cost

        result["cost_extras_usd"] = extra_costs_total

        # Check for unpriced tokens
        known_dimensions = {"input_tokens", "output_tokens"} | set(
            pricing_config.extras.keys()
        )
        for key, value in normalized_usage.items():
            if key.endswith("_tokens") and key not in known_dimensions and value > 0:
                result["cost_estimated"] = True
                if pricing_config.extras_policy.passthrough_unpriced:
                    # Store in extra_tokens but don't add to cost
                    pass

        # Calculate total cost
        total = 0.0
        if result["cost_input_usd"] is not None:
            total += result["cost_input_usd"]
        if result["cost_output_usd"] is not None:
            total += result["cost_output_usd"]
        total += result["cost_extras_usd"]

        result["cost_total_usd"] = total

        # Apply rounding
        if pricing_config.rounding:
            decimals = pricing_config.rounding.decimals
            result = PricingCalculator._round_costs(result, decimals)

        return result

    @staticmethod
    def _calculate_dimension_cost(
        tokens: int, cost_per_1k: float, multiplier: float = 1.0
    ) -> float:
        """Calculate cost for a dimension"""
        if tokens <= 0:
            return 0.0
        return (tokens / 1000.0) * cost_per_1k * multiplier

    @staticmethod
    def _get_context_multiplier(total_tokens: int, context_tiers: list) -> float:
        """Get context multiplier based on total tokens"""
        if not context_tiers:
            return 1.0

        # Find the appropriate tier
        for tier in sorted(context_tiers, key=lambda t: t.max_context):
            if total_tokens <= tier.max_context:
                return tier.multiplier

        # If no tier matches, use the highest tier multiplier
        if context_tiers:
            return max(context_tiers, key=lambda t: t.max_context).multiplier

        return 1.0

    @staticmethod
    def _round_costs(cost_result: Dict[str, Any], decimals: int) -> Dict[str, Any]:
        """Round cost values to specified decimal places"""
        cost_fields = [
            "cost_input_usd",
            "cost_output_usd",
            "cost_extras_usd",
            "cost_total_usd",
        ]

        for field in cost_fields:
            if cost_result[field] is not None:
                decimal_value = Decimal(str(cost_result[field]))
                rounded = decimal_value.quantize(
                    Decimal("0." + "0" * decimals), rounding=ROUND_HALF_UP
                )
                cost_result[field] = float(rounded)

        # Round extra costs
        if cost_result["extra_costs"]:
            for dim_name, cost in cost_result["extra_costs"].items():
                decimal_value = Decimal(str(cost))
                rounded = decimal_value.quantize(
                    Decimal("0." + "0" * decimals), rounding=ROUND_HALF_UP
                )
                cost_result["extra_costs"][dim_name] = float(rounded)

        return cost_result


class UsageSession:
    """Manages accumulation and logging of usage data for a single request"""

    def __init__(
        self,
        task_type: str,
        request_mode: str,
        model_name: str,
        provider: str,
        pricing_config: Optional[PricingConfig] = None,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        logging_config: Optional[UsageLoggingConfig] = None,
    ):
        self.task_type = task_type
        self.request_mode = request_mode
        self.model_name = model_name
        self.provider = provider
        self.pricing_config = pricing_config
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.logging_config = logging_config or UsageLoggingConfig()

        self.start_time = time.monotonic()
        self.accumulated_usage: Dict[str, Any] = {}
        self.streamed = False
        self.partial = False

    def accumulate(self, usage_fragment: Dict[str, Any]):
        """Accumulate usage data from a fragment (for streaming)"""
        for key, value in usage_fragment.items():
            if isinstance(value, (int, float)):
                self.accumulated_usage[key] = self.accumulated_usage.get(key, 0) + value
            else:
                # For non-numeric values, take the latest
                self.accumulated_usage[key] = value

    async def finalize(
        self,
        success: bool,
        error: Optional[Exception] = None,
        final_usage: Optional[Dict[str, Any]] = None,
        streamed: bool = False,
        partial: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Finalize and persist usage log

        Args:
            success: Whether the request succeeded
            error: Optional exception if request failed
            final_usage: Final/override usage data
            streamed: Whether response was streamed
            partial: Whether response was partial/aborted
            details: Extended execution details (agent steps, tool calls, etc.)
        """
        if not self.logging_config.enabled:
            return

        try:
            # Calculate latency
            latency_ms = int((time.monotonic() - self.start_time) * 1000)

            # Merge usage data
            usage_raw = dict(self.accumulated_usage)
            if final_usage:
                usage_raw.update(final_usage)

            # Extract core tokens and extra tokens
            input_tokens = usage_raw.get("input_tokens")
            output_tokens = usage_raw.get("output_tokens")
            total_tokens = None

            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

            # Collect extra tokens (non-core dimensions)
            extra_tokens = {}
            core_keys = {"input_tokens", "output_tokens", "total_tokens"}
            for key, value in usage_raw.items():
                if key.endswith("_tokens") and key not in core_keys:
                    extra_tokens[key] = value

            # Normalize usage and calculate costs
            cost_result = {}
            pricing_snapshot_id = None

            if self.pricing_config:
                normalized_usage = UsageNormalizer.normalize_usage(
                    usage_raw, self.pricing_config.key_aliases
                )
                cost_result = PricingCalculator.compute_cost(
                    normalized_usage, self.pricing_config
                )

            # Obtain (or create) pricing snapshot row lazily
            if self.pricing_config:
                extras_pricing = {
                    k: {"cost_per_1k": v.cost_per_1k}
                    for k, v in self.pricing_config.extras.items()
                }
                snapshot_hash = LLMPricingSnapshot.compute_hash(
                    provider=self.provider,
                    model_name=self.model_name,
                    currency=self.pricing_config.currency,
                    input_cost_per_1k=self.pricing_config.input.cost_per_1k,
                    output_cost_per_1k=self.pricing_config.output.cost_per_1k,
                    extras=extras_pricing,
                )

                # Simple in-memory cache to avoid roundtrips
                snapshot_id = _PRICING_SNAPSHOT_CACHE.get(snapshot_hash)
                async with get_async_session() as snapshot_session:
                    if snapshot_id is None:
                        # Try to find existing
                        existing = await snapshot_session.execute(
                            select(LLMPricingSnapshot).where(
                                LLMPricingSnapshot.snapshot_hash == snapshot_hash
                            )
                        )
                        row = existing.scalars().first()
                        if row:
                            snapshot_id = row.id
                        else:
                            # Create new snapshot
                            new_snapshot = LLMPricingSnapshot(
                                snapshot_hash=snapshot_hash,
                                provider=self.provider,
                                model_name=self.model_name,
                                input_cost_per_1k=self.pricing_config.input.cost_per_1k,
                                output_cost_per_1k=self.pricing_config.output.cost_per_1k,
                                currency=self.pricing_config.currency,
                                extras=extras_pricing or None,
                            )
                            snapshot_session.add(new_snapshot)
                            try:
                                await snapshot_session.flush()
                                snapshot_id = new_snapshot.id
                            except IntegrityError:
                                # Race: someone else inserted concurrently; re-query
                                await snapshot_session.rollback()
                                retry = await snapshot_session.execute(
                                    select(LLMPricingSnapshot).where(
                                        LLMPricingSnapshot.snapshot_hash
                                        == snapshot_hash
                                    )
                                )
                                row2 = retry.scalars().first()
                                if row2:
                                    snapshot_id = row2.id
                                else:
                                    raise
                            await snapshot_session.commit()
                        if snapshot_id:
                            _PRICING_SNAPSHOT_CACHE[snapshot_hash] = snapshot_id
                pricing_snapshot_id = snapshot_id

            # Handle error information
            error_type = None
            error_message = None
            if error:
                error_type = type(error).__name__
                error_message = str(error)[:500]  # Truncate

            # Create log entry
            log_entry = LLMRequestLog(
                user_id=self.user_id,
                session_id=self.session_id,
                task_type=self.task_type,
                request_mode=self.request_mode,
                model_name=self.model_name,
                provider=self.provider,
                request_id=self.request_id,
                latency_ms=latency_ms,
                success=success,
                error_type=error_type,
                error_message=error_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                extra_tokens=extra_tokens if extra_tokens else None,
                usage_raw=usage_raw,
                pricing_snapshot_id=pricing_snapshot_id,
                cost_input_usd=cost_result.get("cost_input_usd"),
                cost_output_usd=cost_result.get("cost_output_usd"),
                cost_extras_usd=cost_result.get("cost_extras_usd"),
                cost_total_usd=cost_result.get("cost_total_usd"),
                extra_costs=(
                    cost_result.get("extra_costs")
                    if cost_result.get("extra_costs")
                    else None
                ),
                cost_estimated=cost_result.get("cost_estimated", False),
                context_multiplier=cost_result.get("context_multiplier"),
                streamed=streamed,
                partial=partial,
                details=details,
            )

            # Persist to database
            async with get_async_session() as session:
                session.add(log_entry)
                await session.commit()
                logger.debug(
                    f"Persisted usage log for {self.task_type} request: {log_entry.id} (pricing_snapshot_id={pricing_snapshot_id})"
                )

        except Exception as e:
            if self.logging_config.fail_open:
                logger.error(f"Usage logging failed, continuing: {e}")
            else:
                logger.error(f"Usage logging failed: {e}")
                raise

    def finalize_sync(
        self,
        success: bool,
        error: Optional[Exception] = None,
        final_usage: Optional[Dict[str, Any]] = None,
        streamed: bool = False,
        partial: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Synchronous wrapper for finalize() - uses run_async_safely().

        Uses run_async_safely() to reuse the Celery worker loop when available,
        avoiding creation of conflicting event loops. Falls back to creating
        a temporary loop in a thread if not in worker context.

        Use this method when calling from synchronous code (e.g., LangChain v1 middleware hooks).

        Args:
            success: Whether the request succeeded
            error: Optional exception if request failed
            final_usage: Final/override usage data
            streamed: Whether response was streamed
            partial: Whether response was partial/aborted
            details: Extended execution details (agent steps, tool calls, etc.)
        """
        try:
            run_async_safely(
                self.finalize(
                    success=success,
                    error=error,
                    final_usage=final_usage,
                    streamed=streamed,
                    partial=partial,
                    details=details,
                ),
                timeout=30.0,
            )
        except TimeoutError:
            logger.error("Usage logging finalization timed out after 30 seconds")
        except Exception as e:
            if self.logging_config.fail_open:
                logger.error(f"Usage logging sync finalization failed: {e}")
            else:
                raise


class UsageLogger:
    """Factory for creating usage sessions"""

    def __init__(self, logging_config: UsageLoggingConfig):
        self.logging_config = logging_config

    def start_session(
        self,
        task_type: str,
        request_mode: str,
        model_name: str,
        provider: str,
        pricing_config: Optional[PricingConfig] = None,
        user_id: Optional[uuid.UUID] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> UsageSession:
        """Start a new usage logging session"""
        return UsageSession(
            task_type=task_type,
            request_mode=request_mode,
            model_name=model_name,
            provider=provider,
            pricing_config=pricing_config,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            logging_config=self.logging_config,
        )
