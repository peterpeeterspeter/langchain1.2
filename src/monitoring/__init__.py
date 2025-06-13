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

from .performance_profiler import (
    PerformanceProfiler,
    TimingRecord,
    PerformanceSnapshot,
    profile_operation
)

__all__ = [
    # Prompt Analytics
    "PromptAnalytics",
    "QueryMetrics", 
    "AlertThreshold",
    "track_query_metrics",
    "get_analytics_instance",
    
    # Performance Profiler
    "PerformanceProfiler",
    "TimingRecord",
    "PerformanceSnapshot",
    "profile_operation"
] 