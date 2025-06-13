"""
Monitoring and Analytics Package for RAG CMS
Provides comprehensive metrics collection, alert management, and performance reporting.
"""

from .prompt_analytics import (
    PromptAnalytics,
    QueryMetrics,
    AlertThreshold,
    track_query_metrics,
    get_analytics_instance
)

__all__ = [
    "PromptAnalytics",
    "QueryMetrics", 
    "AlertThreshold",
    "track_query_metrics",
    "get_analytics_instance"
] 