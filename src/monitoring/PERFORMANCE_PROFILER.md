# Performance Profiler System

The Performance Profiler provides comprehensive timing analysis, bottleneck identification, and optimization recommendations for RAG pipeline operations.

## Overview

The Performance Profiler is designed to help identify performance bottlenecks in your RAG system and provide actionable optimization recommendations. It tracks operation timing, analyzes performance patterns, and generates detailed reports.

## Key Features

### 🎯 Core Profiling Capabilities
- **Nested Operation Profiling**: Track complex operation hierarchies with parent-child relationships
- **Thread-Safe Execution**: Uses thread-local storage for concurrent profiling
- **Async/Sync Support**: Profile both synchronous and asynchronous operations
- **Context Managers**: Clean profiling with automatic timing and cleanup

### 🔍 Bottleneck Detection
- **Configurable Thresholds**: Set custom timing thresholds for different operation types
- **Recursive Analysis**: Identify bottlenecks at any level of the operation hierarchy
- **Impact Scoring**: Calculate performance impact scores (0-100) based on frequency, duration, and variance
- **Pattern Recognition**: Detect recurring performance issues across operations

### 🚀 Optimization Engine
- **Operation-Specific Suggestions**: Tailored recommendations for different operation types
- **Priority Classification**: Critical, high, medium, and low priority optimizations
- **Effort Estimation**: Estimate implementation effort for each optimization
- **Performance Impact Prediction**: Forecast expected improvements

### 📊 Analytics & Reporting
- **Performance Snapshots**: System-wide performance monitoring
- **Trend Analysis**: Historical performance trend detection
- **Optimization Reports**: Comprehensive analysis with actionable insights
- **Supabase Integration**: Persistent storage for historical analysis

## Quick Start

### Basic Usage

```python
from src.monitoring import PerformanceProfiler
from supabase import create_client

# Initialize profiler
client = create_client(url, key)
profiler = PerformanceProfiler(client, enable_profiling=True)

# Profile an operation using context manager
async def process_query():
    async with profiler.profile("rag_query", query_id="123") as record:
        # Your operation code here
        await embedding_generation()
        await vector_search()
        await llm_generation()
```

### Decorator Usage

```python
# Profile async functions
@profiler.profile_async
async def embedding_generation(text: str):
    # Embedding logic
    return embeddings

# Profile sync functions  
@profiler.profile_sync
def post_process_results(results):
    # Post-processing logic
    return processed_results

# Using decorator factory for flexibility
@profile_operation(profiler)
async def complex_operation():
    # This will automatically be profiled
    pass
```

### Nested Profiling

```python
async def complex_rag_query():
    async with profiler.profile("rag_query") as root:
        
        # Classification step
        async with profiler.profile("query_classification"):
            query_type = await classify_query(query)
        
        # Retrieval with sub-operations
        async with profiler.profile("retrieval") as retrieval:
            async with profiler.profile("embedding_generation"):
                embeddings = await generate_embeddings(query)
            
            async with profiler.profile("vector_search"):
                docs = await search_vectors(embeddings)
            
            async with profiler.profile("reranking"):
                ranked_docs = await rerank_documents(docs, query)
        
        # LLM generation
        async with profiler.profile("llm_generation"):
            response = await generate_response(query, ranked_docs)
        
        return response
```

## Configuration

### Bottleneck Thresholds

```python
profiler = PerformanceProfiler(client)

# Customize thresholds (in milliseconds)
profiler.bottleneck_thresholds = {
    "retrieval": 500,
    "embedding_generation": 200,
    "llm_generation": 2000,
    "cache_lookup": 50,
    "database_query": 100,
    "api_call": 1000
}
```

### Enable/Disable Profiling

```python
# Disable profiling for production if needed
profiler = PerformanceProfiler(client, enable_profiling=False)

# Or toggle at runtime
profiler.enabled = False
```

## Data Models

### TimingRecord

Records timing information for a single operation:

```python
@dataclass
class TimingRecord:
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['TimingRecord'] = field(default_factory=list)
    error: Optional[str] = None
```

### PerformanceSnapshot

System performance at a point in time:

```python
@dataclass  
class PerformanceSnapshot:
    timestamp: datetime
    active_operations: int
    memory_usage_mb: float
    cpu_percent: float
    pending_tasks: int
    cache_size: int
    avg_response_time_ms: float
```

## Analytics & Reporting

### Performance Snapshots

```python
# Capture current system performance
snapshot = await profiler.capture_performance_snapshot()
print(f"Memory usage: {snapshot.memory_usage_mb:.1f}MB")
print(f"Avg response time: {snapshot.avg_response_time_ms:.1f}ms")
```

### Optimization Reports

```python
# Generate comprehensive optimization report
report = await profiler.get_optimization_report(hours=24)

print(f"Total queries analyzed: {report['summary']['total_profiled_queries']}")
print(f"Average duration: {report['summary']['avg_query_duration_ms']:.1f}ms")

# Top bottlenecks
for bottleneck in report['top_bottlenecks']:
    print(f"Operation: {bottleneck['operation']}")
    print(f"Impact Score: {bottleneck['impact_score']:.1f}")
    print(f"Optimizations: {bottleneck['optimizations']}")
```

### Optimization Priorities

```python
# Get prioritized optimization recommendations
priorities = report['optimization_priorities']

for priority in priorities:
    print(f"Operation: {priority['operation']}")
    print(f"Priority: {priority['priority']}")
    print(f"Timeline: {priority['timeline']}")
    print(f"Expected Improvement: {priority['expected_improvement']}")
    print(f"Effort: {priority['effort_estimate']}")
```

### Performance Baseline Establishment

Establishing a performance baseline is crucial for effectively measuring the impact of optimizations and identifying deviations. The Performance Profiler assists in this by allowing you to capture snapshots over time and analyze aggregated metrics.

To establish a baseline:
1.  **Monitor During Normal Operations**: Run the profiler in a production-like environment during typical usage periods to gather a representative dataset.
2.  **Capture Snapshots**: Use `profiler.capture_performance_snapshot()` regularly to collect system-wide metrics like memory usage, CPU, active operations, and average response times.
3.  **Analyze Historical Data**: Utilize `profiler.get_optimization_report()` and `PromptAnalytics` reports (from `src/monitoring/PROMPT_ANALYTICS.md`) over a specified period (e.g., 24 hours, 7 days) to get baseline average durations, P95/P99 latencies, and error rates for key operations.
4.  **Define Key Performance Indicators (KPIs)**: Based on the collected data, establish acceptable ranges for:
    *   Overall RAG query response time.
    *   Individual operation durations (retrieval, LLM generation).
    *   Cache hit rates.
    *   Error rates.
5.  **Document Baseline**: Record these KPIs and their ranges. Any future deviations outside these ranges should trigger alerts or further investigation.

### Performance Trend Analysis

Understanding performance trends helps in predicting future issues, validating optimizations, and capacity planning. The `PerformanceProfiler` and `PromptAnalytics` systems work together to provide this capability.

To analyze trends:
1.  **Time-Series Data**: The profiler stores `performance_profiles` and `performance_bottlenecks` in Supabase, capturing `created_at` timestamps. This allows for time-series analysis.
2.  **Aggregate Metrics**: Use `PromptAnalytics.get_performance_report()` and `get_quality_metrics()` to fetch historical data and observe trends in:
    *   **Average Response Times**: Identify gradual increases, indicating system load or inefficiencies.
    *   **Error Rate Trends**: Detect rising error rates that might signal instability.
    *   **Cache Hit Rate Evolution**: Observe if caching strategies are becoming more or less effective.
    *   **Confidence Score Trends**: A declining trend in confidence scores could indicate issues with content freshness or LLM output quality.
3.  **Visualization**: Connect your Supabase database to external dashboarding tools (e.g., Grafana, PowerBI) to visualize these trends over time. This allows for easy identification of anomalies and long-term performance shifts.
4.  **Post-Optimization Review**: After deploying an optimization, monitor the relevant metrics for a period to confirm the expected improvement and ensure no regressions are introduced.

By consistently establishing baselines and analyzing trends, you can maintain a high-performing and reliable RAG CMS.

## Optimization Suggestions

The profiler provides operation-specific optimization suggestions:

### Retrieval Operations
- Implement query result caching with semantic similarity
- Use hybrid search combining dense and sparse retrievers  
- Optimize chunk size and overlap parameters
- Enable parallel chunk retrieval

### Embedding Operations
- Batch process multiple texts in single API call
- Cache frequently used embeddings
- Consider using a local embedding model for lower latency

### LLM Operations
- Implement response streaming for better UX
- Use smaller models for simple queries
- Enable prompt caching for repeated patterns
- Optimize token usage in prompts

### Cache Operations
- Optimize cache key generation
- Consider in-memory caching for frequently accessed data
- Implement cache warming strategies

### Database Operations
- Add appropriate indexes for common queries
- Optimize query structure and joins
- Consider read replicas for high-traffic operations

## Database Schema

The profiler uses two main tables in Supabase:

### performance_profiles
```sql
CREATE TABLE performance_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id TEXT NOT NULL,
    profile_data JSONB NOT NULL,
    total_duration_ms FLOAT NOT NULL,
    bottleneck_operations JSONB DEFAULT '[]',
    optimization_suggestions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### performance_bottlenecks
```sql
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

## Best Practices

### 1. Strategic Profiling
- Profile critical paths and suspected bottlenecks
- Use descriptive operation names for better analysis
- Include relevant metadata in profiling context

### 2. Performance Impact
- Monitor the profiler's own performance impact
- Disable profiling in production if overhead is significant
- Use sampling for high-frequency operations

### 3. Analysis & Action
- Review optimization reports regularly
- Prioritize high-impact, low-effort optimizations
- Track performance trends over time

### 4. Integration
- Combine with alert system for proactive monitoring
- Use with A/B testing for optimization validation
- Integrate with CI/CD for performance regression detection

## Integration with Other Systems

### Monitoring System Integration
```python
from src.monitoring import PromptAnalytics, PerformanceProfiler

# Use both systems together
analytics = PromptAnalytics(client)
profiler = PerformanceProfiler(client)

async def monitored_and_profiled_operation():
    async with profiler.profile("complex_operation") as profile:
        # Track metrics
        analytics.track_query_start()
        
        # Your operation
        result = await perform_operation()
        
        # Track completion
        analytics.track_query_completion(success=True)
        
        return result
```

### Configuration System Integration
```python
from src.config.enhanced_config import EnhancedConfigManager

# Use configuration-driven profiling
config_manager = EnhancedConfigManager()
config = await config_manager.get_configuration()

# Enable profiling based on configuration
enable_profiling = config.performance_monitoring.get("enable_profiling", True)
profiler = PerformanceProfiler(client, enable_profiling=enable_profiling)
```

## Troubleshooting

### Common Issues

**1. High Memory Usage**
- Reduce operation history buffer size
- Implement more aggressive data cleanup
- Monitor snapshot history size

**2. Performance Overhead**
- Disable profiling for high-frequency operations
- Use sampling instead of full profiling
- Optimize metadata collection

**3. Missing Profiles**
- Verify Supabase connection and permissions
- Check operation naming consistency
- Ensure proper exception handling

### Debug Mode
```python
# Enable verbose logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

profiler = PerformanceProfiler(client, enable_profiling=True)
```

## Performance Considerations

The profiler is designed to be lightweight, but consider these factors:

- **Memory Usage**: Operation history is limited to 1000 entries per operation type
- **Storage**: Profiles are stored in Supabase for historical analysis
- **CPU Overhead**: Minimal impact, but avoid profiling micro-operations
- **Network**: Batch database operations for efficiency

## Future Enhancements

Planned improvements include:

- **Machine Learning**: Predictive performance modeling
- **Distributed Profiling**: Multi-service performance tracking
- **Real-time Alerts**: Performance threshold breach notifications
- **Visual Dashboards**: Interactive performance visualization
- **Automated Optimization**: Self-healing performance improvements

## API Reference

See the main PerformanceProfiler class documentation for detailed API information:

- `PerformanceProfiler.__init__(client, enable_profiling=True)`
- `async PerformanceProfiler.profile(operation_name, **metadata)`
- `PerformanceProfiler.profile_async(func)` 
- `PerformanceProfiler.profile_sync(func)`
- `async PerformanceProfiler.capture_performance_snapshot()`
- `async PerformanceProfiler.get_optimization_report(hours=24)`
- `profile_operation(profiler)` - Decorator factory

The Performance Profiler system provides comprehensive insights into your RAG pipeline performance, enabling data-driven optimization decisions and continuous performance improvement. 