# Monitoring and Analytics System

Comprehensive monitoring and analytics infrastructure for RAG CMS, providing real-time metrics collection, alert management, performance profiling, and optimization recommendations.

## Overview

The monitoring system consists of two main components:

1. **Prompt Analytics** (`prompt_analytics.py`) - Real-time metrics collection and alert management
2. **Performance Profiler** (`performance_profiler.py`) - Timing analysis and bottleneck identification

Together, these systems provide complete observability for your RAG pipeline operations.

## Components

### 1. Prompt Analytics System

Real-time monitoring with buffered metrics collection and intelligent alerting.

**Key Features:**
- Buffered metrics collection with automatic flushing
- Multi-dimensional analytics (classification, performance, quality, cache, errors)
- Real-time alert management with cooldown periods
- Performance reporting with trend analysis
- Bottleneck identification and optimization recommendations

**Core Classes:**
- `PromptAnalytics` - Main analytics engine
- `QueryMetrics` - Structured metric data model
- `AlertThreshold` - Configurable alert thresholds
- `AlertInstance` - Alert state management

### 2. Performance Profiler System

Advanced performance profiling with nested operation timing and optimization suggestions.

**Key Features:**
- Nested operation profiling with thread-local storage
- Context managers and decorators for flexible profiling
- Bottleneck detection with configurable thresholds
- Operation-specific optimization recommendations
- Performance impact scoring and trend analysis
- Historical data analysis and reporting

**Core Classes:**
- `PerformanceProfiler` - Main profiling engine
- `TimingRecord` - Individual operation timing data
- `PerformanceSnapshot` - System performance snapshots

## Quick Start

### Basic Usage

```python
from src.monitoring import PromptAnalytics, PerformanceProfiler
from supabase import create_client

# Initialize systems
client = create_client(url, key)
analytics = PromptAnalytics(client)
profiler = PerformanceProfiler(client)

# Monitor and profile an operation
async def monitored_operation():
    # Start analytics tracking
    query_id = analytics.track_query_start()
    
    # Profile the operation
    async with profiler.profile("rag_query", query_id=query_id):
        # Your RAG pipeline operations
        result = await process_rag_query()
        
        # Track completion
        analytics.track_query_completion(
            classification_accuracy=0.85,
            response_quality=0.92,
            cache_hit=True
        )
        
        return result
```

### Advanced Monitoring

```python
# Set up custom alert thresholds
await analytics.set_alert_threshold(
    metric_type="error_rate",
    warning_threshold=0.05,
    critical_threshold=0.10
)

# Configure profiler thresholds
profiler.bottleneck_thresholds.update({
    "retrieval": 500,  # ms
    "llm_generation": 2000,
    "embedding_generation": 200
})

# Get comprehensive reports
analytics_report = await analytics.get_performance_report()
profiler_report = await profiler.get_optimization_report()
```

## Integration Examples

### With Configuration System

```python
from src.config.enhanced_config import EnhancedConfigManager

config_manager = EnhancedConfigManager()
config = await config_manager.get_configuration()

# Configure monitoring based on settings
analytics = PromptAnalytics(
    client,
    buffer_size=config.monitoring.get("buffer_size", 50),
    flush_interval=config.monitoring.get("flush_interval", 30)
)

profiler = PerformanceProfiler(
    client,
    enable_profiling=config.performance.get("enable_profiling", True)
)
```

### With Enhanced Confidence Scoring

```python
from src.chains.enhanced_confidence_scoring_system import EnhancedConfidenceScorer

async def enhanced_rag_pipeline(query: str):
    query_id = analytics.track_query_start()
    
    async with profiler.profile("enhanced_rag_query", query_id=query_id):
        # Process with confidence scoring
        scorer = EnhancedConfidenceScorer()
        
        async with profiler.profile("retrieval"):
            documents = await retrieve_documents(query)
        
        async with profiler.profile("confidence_scoring"):
            scored_response = await scorer.score_response(
                query, documents, response
            )
        
        # Track enhanced metrics
        analytics.track_query_completion(
            classification_accuracy=scored_response.classification_score,
            response_quality=scored_response.overall_confidence,
            confidence_factors=scored_response.confidence_factors
        )
        
        return scored_response
```

## Database Schema

### Analytics Tables

```sql
-- Metrics storage
CREATE TABLE query_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    value FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Alert management
CREATE TABLE alert_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ
);
```

### Performance Profiler Tables

```sql
-- Performance profiles
CREATE TABLE performance_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id TEXT NOT NULL,
    profile_data JSONB NOT NULL,
    total_duration_ms FLOAT NOT NULL,
    bottleneck_operations JSONB DEFAULT '[]',
    optimization_suggestions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bottleneck statistics
CREATE TABLE performance_bottlenecks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type TEXT NOT NULL,
    avg_duration_ms FLOAT NOT NULL,
    p95_duration_ms FLOAT NOT NULL,
    p99_duration_ms FLOAT NOT NULL,
    occurrence_count INTEGER NOT NULL,
    impact_score FLOAT NOT NULL,
    suggested_optimizations JSONB DEFAULT '[]',
    detected_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Monitoring Strategies

### 1. Real-Time Monitoring

Monitor critical metrics in real-time with immediate alerting:

```python
# Set up critical alerts
await analytics.set_alert_threshold("error_rate", 0.01, 0.05)
await analytics.set_alert_threshold("avg_response_time", 1000, 2000)
await analytics.set_alert_threshold("confidence_score", 0.7, 0.5)

# Enable real-time profiling for bottleneck detection
profiler.enabled = True
```

### 2. Performance Optimization

Use profiling data to optimize performance:

```python
# Get optimization recommendations
report = await profiler.get_optimization_report(hours=24)

for priority in report['optimization_priorities']:
    if priority['priority'] == 'critical':
        print(f"Urgent: {priority['operation']} - {priority['timeline']}")
        print(f"Expected improvement: {priority['expected_improvement']}")
```

### 3. Quality Monitoring

Track response quality and confidence over time:

```python
# Analyze quality trends
quality_metrics = await analytics.get_quality_metrics(days=7)
confidence_trends = await analytics.get_confidence_trends(days=7)

# Alert on quality degradation
if quality_metrics['avg_confidence'] < 0.8:
    await analytics.trigger_quality_alert()
```

## Best Practices

### 1. Metric Collection
- Use descriptive operation names for better analysis
- Include relevant metadata in tracking calls
- Balance monitoring frequency with performance impact
- Implement sampling for high-frequency operations

### 2. Alert Management
- Set appropriate thresholds based on baseline performance
- Use alert cooldowns to prevent spam
- Implement escalation procedures for critical alerts
- Regularly review and adjust alert thresholds

### 3. Performance Profiling
- Profile critical paths and suspected bottlenecks
- Use nested profiling for complex operations
- Review optimization reports regularly
- Prioritize high-impact, low-effort optimizations

### 4. Data Analysis
- Combine metrics from both systems for comprehensive insights
- Use historical data for trend analysis
- Implement automated anomaly detection
- Create custom dashboards for key stakeholders

## API Reference

### Prompt Analytics

- `PromptAnalytics(client, buffer_size=50, flush_interval=30)`
- `track_query_start() -> str`
- `track_query_completion(**metrics)`
- `set_alert_threshold(metric_type, warning, critical)`
- `get_performance_report(hours=24) -> Dict`
- `get_real_time_metrics() -> Dict`

### Performance Profiler

- `PerformanceProfiler(client, enable_profiling=True)`
- `async profile(operation_name, **metadata)`
- `profile_async(func)` / `profile_sync(func)`
- `capture_performance_snapshot() -> PerformanceSnapshot`
- `get_optimization_report(hours=24) -> Dict`

### Utility Functions

- `track_query_metrics(analytics, metrics_dict)`
- `get_analytics_instance() -> PromptAnalytics`
- `profile_operation(profiler)` - Decorator factory

## Migration Files

Database migrations are provided for setting up the required tables:

- `migrations/002_create_monitoring_tables.sql` - Analytics tables
- `migrations/003_create_performance_profiler_tables.sql` - Profiler tables

## Documentation

Detailed documentation is available:

- `PERFORMANCE_PROFILER.md` - Comprehensive profiler documentation
- Implementation examples and best practices
- Troubleshooting guides and performance considerations

## Future Enhancements

Planned improvements include:

- **Machine Learning Integration**: Predictive performance modeling
- **Distributed Monitoring**: Multi-service monitoring and tracing
- **Visual Dashboards**: Interactive performance and metrics visualization
- **Automated Optimization**: Self-healing performance improvements
- **Cost Monitoring**: AI API usage and cost tracking
- **User Experience Metrics**: End-user satisfaction tracking

The monitoring system provides comprehensive observability for your RAG pipeline, enabling data-driven optimization and proactive issue resolution. 