"""
LLM Usage Service

Service for aggregating and querying LLM usage statistics and costs.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from inference_core.database.sql.connection import get_async_session
from inference_core.database.sql.models.llm_request_log import LLMRequestLog

logger = logging.getLogger(__name__)


class LLMUsageService:
    """Service for querying LLM usage statistics and costs"""
    
    @staticmethod
    async def get_usage_stats(
        user_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get aggregated usage statistics
        
        Args:
            user_id: Optional user ID to filter by
            days: Number of days to look back
            
        Returns:
            Dictionary with usage and cost statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with get_async_session() as session:
            # Build base query
            query = select(LLMRequestLog).where(
                LLMRequestLog.created_at >= cutoff_date
            )
            
            if user_id:
                query = query.where(LLMRequestLog.user_id == user_id)
            
            # Execute query to get raw data
            result = await session.execute(query)
            logs = result.scalars().all()
            
            # Aggregate statistics
            stats = {
                "usage": {
                    "total_requests": len(logs),
                    "successful_requests": sum(1 for log in logs if log.success),
                    "failed_requests": sum(1 for log in logs if not log.success),
                    "total_tokens": sum(log.total_tokens or 0 for log in logs if log.total_tokens),
                    "input_tokens": sum(log.input_tokens or 0 for log in logs if log.input_tokens),
                    "output_tokens": sum(log.output_tokens or 0 for log in logs if log.output_tokens),
                    "by_model": {},
                    "by_task_type": {},
                    "by_provider": {},
                },
                "cost": {
                    "currency": "USD",
                    "total": 0.0,
                    "estimated_portion": 0.0,
                    "by_model": {},
                    "by_task_type": {},
                    "by_provider": {},
                    "core_breakdown": {
                        "input": 0.0,
                        "output": 0.0,
                        "extras": 0.0,
                    },
                }
            }
            
            # Process each log
            for log in logs:
                model = log.model_name
                task_type = log.task_type
                provider = log.provider
                
                # Usage aggregation by dimensions
                if model not in stats["usage"]["by_model"]:
                    stats["usage"]["by_model"][model] = {
                        "requests": 0,
                        "tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                
                if task_type not in stats["usage"]["by_task_type"]:
                    stats["usage"]["by_task_type"][task_type] = {
                        "requests": 0,
                        "tokens": 0,
                    }
                
                if provider not in stats["usage"]["by_provider"]:
                    stats["usage"]["by_provider"][provider] = {
                        "requests": 0,
                        "tokens": 0,
                    }
                
                # Update usage counts
                stats["usage"]["by_model"][model]["requests"] += 1
                stats["usage"]["by_model"][model]["tokens"] += log.total_tokens or 0
                stats["usage"]["by_model"][model]["input_tokens"] += log.input_tokens or 0
                stats["usage"]["by_model"][model]["output_tokens"] += log.output_tokens or 0
                
                stats["usage"]["by_task_type"][task_type]["requests"] += 1
                stats["usage"]["by_task_type"][task_type]["tokens"] += log.total_tokens or 0
                
                stats["usage"]["by_provider"][provider]["requests"] += 1
                stats["usage"]["by_provider"][provider]["tokens"] += log.total_tokens or 0
                
                # Cost aggregation
                if log.cost_total_usd is not None:
                    cost = float(log.cost_total_usd)
                    stats["cost"]["total"] += cost
                    
                    # Track estimated costs
                    if log.cost_estimated:
                        stats["cost"]["estimated_portion"] += cost
                    
                    # By model
                    if model not in stats["cost"]["by_model"]:
                        stats["cost"]["by_model"][model] = 0.0
                    stats["cost"]["by_model"][model] += cost
                    
                    # By task type
                    if task_type not in stats["cost"]["by_task_type"]:
                        stats["cost"]["by_task_type"][task_type] = 0.0
                    stats["cost"]["by_task_type"][task_type] += cost
                    
                    # By provider
                    if provider not in stats["cost"]["by_provider"]:
                        stats["cost"]["by_provider"][provider] = 0.0
                    stats["cost"]["by_provider"][provider] += cost
                    
                    # Core breakdown
                    if log.cost_input_usd is not None:
                        stats["cost"]["core_breakdown"]["input"] += float(log.cost_input_usd)
                    if log.cost_output_usd is not None:
                        stats["cost"]["core_breakdown"]["output"] += float(log.cost_output_usd)
                    if log.cost_extras_usd is not None:
                        stats["cost"]["core_breakdown"]["extras"] += float(log.cost_extras_usd)
            
            # Round cost values to 6 decimal places
            stats["cost"]["total"] = round(stats["cost"]["total"], 6)
            stats["cost"]["estimated_portion"] = round(stats["cost"]["estimated_portion"], 6)
            
            for key in stats["cost"]["by_model"]:
                stats["cost"]["by_model"][key] = round(stats["cost"]["by_model"][key], 6)
            for key in stats["cost"]["by_task_type"]:
                stats["cost"]["by_task_type"][key] = round(stats["cost"]["by_task_type"][key], 6)
            for key in stats["cost"]["by_provider"]:
                stats["cost"]["by_provider"][key] = round(stats["cost"]["by_provider"][key], 6)
            
            for key in stats["cost"]["core_breakdown"]:
                stats["cost"]["core_breakdown"][key] = round(stats["cost"]["core_breakdown"][key], 6)
            
            return stats
    
    @staticmethod
    async def get_recent_logs(
        limit: int = 100,
        user_id: Optional[str] = None,
        task_type: Optional[str] = None,
        success: Optional[bool] = None
    ) -> list:
        """
        Get recent usage logs
        
        Args:
            limit: Maximum number of logs to return
            user_id: Optional user ID filter
            task_type: Optional task type filter
            success: Optional success status filter
            
        Returns:
            List of recent log entries
        """
        async with get_async_session() as session:
            query = select(LLMRequestLog).order_by(LLMRequestLog.created_at.desc()).limit(limit)
            
            if user_id:
                query = query.where(LLMRequestLog.user_id == user_id)
            if task_type:
                query = query.where(LLMRequestLog.task_type == task_type)
            if success is not None:
                query = query.where(LLMRequestLog.success == success)
            
            result = await session.execute(query)
            logs = result.scalars().all()
            
            return [
                {
                    "id": str(log.id),
                    "created_at": log.created_at.isoformat(),
                    "task_type": log.task_type,
                    "model_name": log.model_name,
                    "provider": log.provider,
                    "success": log.success,
                    "input_tokens": log.input_tokens,
                    "output_tokens": log.output_tokens,
                    "total_tokens": log.total_tokens,
                    "cost_total_usd": float(log.cost_total_usd) if log.cost_total_usd else None,
                    "cost_estimated": log.cost_estimated,
                    "latency_ms": log.latency_ms,
                    "streamed": log.streamed,
                    "partial": log.partial,
                }
                for log in logs
            ]


# Global service instance
llm_usage_service = LLMUsageService()


def get_llm_usage_service() -> LLMUsageService:
    """Get global LLM usage service instance"""
    return llm_usage_service