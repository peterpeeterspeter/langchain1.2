# Prompt Analytics System

## Overview

The Prompt Analytics System (`src/monitoring/prompt_analytics.py`) provides real-time, comprehensive metric collection and intelligent alert management for the Universal RAG CMS. It is designed to offer deep insights into query performance, response quality, and system health, enabling proactive issue detection and continuous optimization.

Key features include:

-   **Buffered Metrics Collection**: Efficiently gathers various performance and quality metrics with intelligent buffering and flushing to the database.
-   **Multi-Dimensional Analytics**: Tracks and analyzes data across several categories: query classification, response performance, content quality, caching efficiency, and error rates.
-   **Real-Time Alert Management**: Configurable alert thresholds and cooldown periods for critical system events.
-   **Performance Reporting**: Generates historical performance reports and trend analysis.
-   **Integration Capabilities**: Seamlessly integrates with the `IntegratedRAGChain` and other system components.

## Core Class: `PromptAnalytics`

The `PromptAnalytics` class is the central engine for collecting, processing, and reporting metrics. It interacts with Supabase for persistent storage and retrieval of analytics data.

```python
from supabase import Client
from datetime import timedelta, datetime
from typing import Dict, Any, Optional, List
from enum import Enum

# Assuming these are defined elsewhere or passed in
# from src.config.prompt_config import PerformanceConfig

class PromptAnalytics:
    def __init__(
        self,
        supabase_client: Client,
        config: Optional[Any] = None, # Can be PerformanceConfig or similar
        buffer_size: int = 50,
        flush_interval_minutes: int = 5
    ):
        self.client = supabase_client
        self.buffer = [] # In-memory buffer for metrics
        self.buffer_size = buffer_size
        self.flush_interval = timedelta(minutes=flush_interval_minutes)
        self.last_flush_time = datetime.utcnow()
        self.alert_thresholds = {} # Configurable alerts
        self.active_alerts = {}    # Currently active alerts
        self.config = config
        # ... (other internal attributes and setup)

    async def track_query_start(self, query: str, user_context: Optional[Dict] = None) -> str:
        # Initiates tracking for a new query, returning a unique query ID.
        pass

    async def track_query_completion(
        self,
        query_id: str,
        metrics: Dict[str, Any],
        success: bool = True,
        error_details: Optional[str] = None
    ):
        # Records completion metrics for a query, processes data, and triggers alerts.
        pass

    async def flush_metrics_buffer(self):
        # Flushes buffered metrics to the Supabase database.
        pass

    async def get_real_time_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        # Retrieves aggregated metrics for a recent time window.
        pass

    async def set_alert_threshold(
        self,
        metric_type: str,
        warning_threshold: float,
        critical_threshold: float,
        cooldown_minutes: int = 15
    ):
        # Configures an alert threshold for a specific metric type.
        pass

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        # Retrieves a list of currently active (unacknowledged) alerts.
        pass

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        # Marks an alert as acknowledged in the database.
        pass

    async def get_performance_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        # Generates a comprehensive performance report for a given period.
        pass

    async def get_quality_metrics(self, days: int = 7) -> Dict[str, Any]:
        # Retrieves aggregated quality metrics (e.g., confidence scores, source quality).
        pass

    async def get_confidence_trends(self, days: int = 7) -> Dict[str, Any]:
        # Analyzes trends in confidence scores over time.
        pass
```

## Metrics Collection and Analysis

The `PromptAnalytics` system collects various metrics, which are then analyzed to provide actionable insights. Key metric categories include:

-   **Query Metrics**: Tracks individual query IDs, processing times, and associated metadata.
-   **Classification Metrics**: Records accuracy and confidence of query type classification.
-   **Performance Metrics**: Captures response times, token usage, and latency breakdown.
-   **Quality Metrics**: Logs confidence scores (overall and by factor), source quality, and response validation results.
-   **Cache Metrics**: Monitors cache hit/miss rates, effective TTLs, and cache value distribution.
-   **Error Metrics**: Records errors, their types, and frequency.

Metrics are stored in a Supabase table (`query_metrics` as defined in `src/monitoring/prompt_analytics.py` and `src/monitoring/README.md`) for historical analysis.

### Data Models for Metrics

Metrics are structured using internal data models (e.g., `QueryMetrics`, `PerformanceMetrics`, `QualityMetrics`) to ensure consistency and facilitate analysis. These models are defined within `prompt_analytics.py`.

## Alert Configuration and Management

Proactive alerting is a critical feature, allowing the system to notify administrators of potential issues based on predefined thresholds.

### Setting Alert Thresholds

Alerts can be configured for various metrics:

```python
from src.monitoring import PromptAnalytics

# Assuming analytics manager is initialized
# analytics = PromptAnalytics(supabase_client)

async def setup_alerts_example():
    # Set warning/critical thresholds for error rate
    await analytics.set_alert_threshold(
        metric_type="error_rate",
        warning_threshold=0.05,    # 5% error rate
        critical_threshold=0.10,   # 10% error rate
        cooldown_minutes=30
    )

    # Set warning/critical thresholds for average response time
    await analytics.set_alert_threshold(
        metric_type="avg_response_time",
        warning_threshold=1500,    # 1.5 seconds
        critical_threshold=3000,   # 3 seconds
        cooldown_minutes=60
    )

    # Set warning/critical thresholds for confidence score
    await analytics.set_alert_threshold(
        metric_type="confidence_score",
        warning_threshold=0.70,    # Below 0.70 is warning
        critical_threshold=0.50,   # Below 0.50 is critical
        cooldown_minutes=15
    )

# await setup_alerts_example()
```

### Managing Active Alerts

Administrators can retrieve and acknowledge alerts through the API or direct calls to `PromptAnalytics`:

```python
async def manage_alerts_example():
    active_alerts = await analytics.get_active_alerts()
    print("Active Alerts:")
    for alert in active_alerts:
        print(f"- {alert['severity']}: {alert['message']} (ID: {alert['id']})")
    
    if active_alerts:
        # Acknowledge the first critical alert
        critical_alert = next((a for a in active_alerts if a['severity'] == 'critical'), None)
        if critical_alert:
            await analytics.acknowledge_alert(critical_alert['id'], "admin_user")
            print(f"Alert {critical_alert['id']} acknowledged.")

# await manage_alerts_example()
```

## Dashboard Setup and Customization (Conceptual)

While this system directly provides the data, setting up dashboards typically involves external tools (e.g., Grafana, PowerBI) that connect to your Supabase instance.

To visualize the collected metrics:

1.  **Connect your Dashboard Tool**: Point your BI or monitoring dashboard tool to your Supabase Postgres database.
2.  **Query `query_metrics` and `alert_instances` tables**: These tables (defined in `src/monitoring/README.md`) hold all the raw and alert data.
3.  **Create Visualizations**: Design charts for:
    *   **Response Time Trends**: Average, P95, P99 latency over time.
    *   **Error Rate**: Percentage of failed queries.
    *   **Confidence Score Distribution**: Histogram or average confidence over time.
    *   **Cache Hit Rate**: Percentage of queries served from cache.
    *   **Alert Status**: Active, acknowledged, and resolved alerts.
    *   **Query Type Breakdown**: Distribution of queries by type.

## Performance Troubleshooting Playbook

Use the Prompt Analytics system to diagnose and resolve performance issues:

1.  **Identify Anomalies**: Monitor real-time metrics for sudden spikes in response time or error rates, or drops in confidence scores.
2.  **Generate Performance Reports**: Use `get_performance_report()` to analyze historical trends and identify periods of degradation.
3.  **Pinpoint Bottlenecks**: Cross-reference with `PerformanceProfiler` reports (from `src/monitoring/performance_profiler.py` and `PERFORMANCE_PROFILER.md`) to identify specific operations (retrieval, LLM generation, embedding) causing slowdowns.
4.  **Analyze Quality Metrics**: Investigate `get_quality_metrics()` and `get_confidence_trends()` to see if performance issues correlate with a decrease in response quality.
5.  **Review Logs**: Dive into detailed logs (via the `RAGPipelineLogger` in `src/utils/integration_helpers.py`) for specific query IDs to understand the context of failures or slow responses.
6.  **Implement Optimizations**: Apply changes based on findings (e.g., optimize prompts, improve retrieval, adjust cache settings). The system's configurations (`PromptOptimizationConfig`) can be updated dynamically.
7.  **Monitor Impact**: Observe metrics post-optimization to confirm improvements.

## Optimization Recommendation Interpretation

The analytics system, especially when combined with the Performance Profiler, can provide optimization recommendations. Interpret these as follows:

-   **High Impact / Critical Severity**: These indicate issues that significantly degrade user experience or system stability. Prioritize immediate action.
-   **Common Bottlenecks**: Look for operations that consistently appear in performance reports as slow (e.g., `llm_generation`, `retrieval`). These are prime candidates for optimization.
-   **Confidence Score Drops**: A decline in confidence often points to issues with content quality, source relevance, or LLM generation. Review the `confidence_breakdown` for specifics.
-   **Cache Misses**: High cache miss rates suggest either ineffective caching strategies or frequently changing data, requiring adjustments to `CacheConfig` or `IntelligentCache` settings.

This documentation aims to guide users through effectively leveraging the Prompt Analytics System for system monitoring, alerting, and performance troubleshooting. 