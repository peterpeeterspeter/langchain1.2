# Feature Flags and A/B Testing Infrastructure

## Overview

The Feature Flags and A/B Testing Infrastructure provides sophisticated capabilities for feature rollouts, experimentation, and data-driven decision making in the RAG CMS system. This system enables:

- **Feature Flags**: Enable/disable features with fine-grained control
- **Gradual Rollouts**: Roll out features to a percentage of users
- **A/B Testing**: Run controlled experiments with statistical analysis
- **User Segmentation**: Deterministic and random user targeting
- **Analytics**: Comprehensive metrics and statistical significance testing

## Quick Start

### Basic Feature Flag Usage

```python
from src.config import FeatureFlagManager, FeatureFlag, FeatureStatus
from supabase import create_client

# Initialize
supabase = create_client(url, key)
manager = FeatureFlagManager(supabase)

# Create a feature flag
flag = FeatureFlag(
    name="enhanced_search",
    description="Enhanced search with semantic similarity",
    status=FeatureStatus.GRADUAL_ROLLOUT,
    rollout_percentage=25.0
)

await manager.create_feature_flag(flag)

# Check if feature is enabled for a user
user_context = {"user_id": "user123"}
if await manager.is_feature_enabled("enhanced_search", user_context):
    # Use enhanced search
    pass
```

### A/B Testing

```python
from src.config import FeatureVariant

# Create A/B test variants
variants = [
    FeatureVariant(
        name="control",
        weight=50.0,
        config_overrides={"model": "gpt-3.5"}
    ),
    FeatureVariant(
        name="treatment", 
        weight=50.0,
        config_overrides={"model": "gpt-4"}
    )
]

# Create A/B test flag
ab_flag = FeatureFlag(
    name="model_comparison",
    description="Compare GPT-3.5 vs GPT-4",
    status=FeatureStatus.AB_TEST,
    rollout_percentage=100.0,
    variants=variants
)

await manager.create_feature_flag(ab_flag)

# Get variant for user
variant = await manager.get_variant("model_comparison", user_context)
if variant:
    model = variant.config_overrides["model"]
    # Use the assigned model
```

### Using the Decorator

```python
from src.config import feature_flag

class RAGService:
    def __init__(self, feature_flag_manager):
        self.feature_flag_manager = feature_flag_manager
    
    @feature_flag("enhanced_search", default=False)
    async def search(self, query, user_context=None):
        # This method only runs if enhanced_search is enabled
        return enhanced_search_results(query)
```

## Core Components

### Feature Flag Statuses

```python
class FeatureStatus(str, Enum):
    DISABLED = "disabled"           # Feature is off for everyone
    ENABLED = "enabled"             # Feature is on for everyone  
    GRADUAL_ROLLOUT = "gradual_rollout"  # Roll out to percentage of users
    AB_TEST = "ab_test"             # A/B test with variants
    CANARY = "canary"               # Canary deployment
```

### Segmentation Strategies

#### Hash-Based Segmentation (Default)
```python
# Deterministic segmentation - same user always gets same result
strategy = HashBasedSegmentation(salt="rag-cms-features")

# Consistent assignment
user_context = {"user_id": "user123"}
enabled = strategy.should_enable_feature(user_context, 50.0)
# Always returns same result for user123
```

#### Random Segmentation
```python
# Random assignment for each request
strategy = RandomSegmentation()

# Non-deterministic
enabled = strategy.should_enable_feature(user_context, 50.0)
# May return different results each time
```

### Feature Flag Configuration

```python
@dataclass
class FeatureFlag:
    name: str                                    # Unique identifier
    description: str                             # Human-readable description
    status: FeatureStatus                        # Current status
    rollout_percentage: float = 0.0              # 0-100% rollout
    segments: Dict[str, Any] = {}                # Segmentation config
    variants: List[FeatureVariant] = []          # A/B test variants
    metadata: Dict[str, Any] = {}                # Additional metadata
    created_at: datetime                         # Creation timestamp
    updated_at: datetime                         # Last update
    expires_at: Optional[datetime] = None        # Optional expiration
```

### A/B Test Variants

```python
@dataclass
class FeatureVariant:
    name: str                                    # Variant identifier (e.g., "control", "treatment")
    weight: float                                # Percentage allocation (0-100)
    config_overrides: Dict[str, Any] = {}        # Configuration overrides
    metadata: Dict[str, Any] = {}                # Additional metadata
```

## Advanced Usage

### Experiment Management

```python
# Create an experiment
experiment_id = await manager.create_experiment(
    feature_flag_name="model_comparison",
    experiment_name="GPT Model Performance Test",
    hypothesis="GPT-4 will provide 20% better quality scores",
    success_metrics=["quality_score", "user_satisfaction", "response_time"],
    duration_days=14
)

# Track metrics
await manager.track_metric(
    experiment_id=experiment_id,
    variant_name="treatment",
    metric_type="quality_score", 
    metric_value=0.85,
    user_context=user_context,
    metadata={"query_type": "technical"}
)

# Get experiment results with statistical analysis
results = await manager.get_experiment_results(experiment_id)
print(f"P-value: {results['statistical_analysis']['p_value']}")
print(f"Significant: {results['statistical_analysis']['is_significant']}")
```

### Metric Types

The system tracks several built-in metric types:

- **impression**: User saw the feature
- **conversion**: User completed desired action
- **error**: Error occurred during feature use
- **response_time_ms**: Response time in milliseconds
- **quality_score**: Quality score (0-1 scale)
- **custom metrics**: Any additional metrics you define

### Statistical Analysis

The system automatically calculates:

- **Conversion rates** for each variant
- **Statistical significance** (p-values, confidence intervals)
- **Effect size** and relative improvement
- **Recommendations** based on results

```python
{
    "control_conversion_rate": 0.12,
    "treatment_conversion_rate": 0.15,
    "relative_improvement": 25.0,  # 25% improvement
    "p_value": 0.032,
    "is_significant": true,
    "confidence_interval_95": {
        "lower": 0.005,
        "upper": 0.055
    },
    "recommendation": "Treatment variant shows statistically significant improvement. Consider full rollout."
}
```

## Experiment Design Best Practices

To ensure your A/B tests yield reliable and actionable insights, follow these best practices:

1.  **Clear Hypothesis**: Define a specific, measurable hypothesis (e.g., "Changing prompt X will increase user engagement by Y%").
2.  **Define Success Metrics**: Clearly identify the primary and secondary metrics that will determine the experiment's success (e.g., `conversion`, `quality_score`, `response_time_ms`). Ensure metrics are measurable and relevant to your hypothesis.
3.  **Isolate Variables**: Test only one significant change at a time to accurately attribute impact.
4.  **Control Group**: Always include a control group (the existing experience) to compare against your new variants.
5.  **Sufficient Sample Size**: Ensure you have enough users or queries in each variant to achieve statistical significance. The duration of the experiment (`duration_days`) should be long enough to gather sufficient data.
6.  **Randomization**: Use effective user segmentation (hash-based is recommended for consistency) to ensure users are randomly and deterministically assigned to variants.
7.  **Monitor During Experiment**: Continuously monitor key performance indicators and error rates for all variants to detect any negative impact early.
8.  **Avoid Peeking**: Do not analyze results prematurely. Wait until the predefined `duration_days` or sufficient sample size is reached to avoid false positives.
9.  **Iterate**: A/B testing is an iterative process. Learn from each experiment and use the insights to inform subsequent tests.

## Rollout Strategy Recommendations

Choosing the right feature flag status and rollout strategy is crucial for minimizing risk and effectively managing new feature deployments.

### When to Use Each `FeatureStatus`:

-   **`DISABLED`**: 
    *   **Use Case**: Initial development, features not ready for any exposure, or temporarily disabling a problematic feature.
    *   **Recommendation**: Default status for new, incomplete features.

-   **`CANARY`**: 
    *   **Use Case**: Deploying a new, potentially high-risk feature to a very small, specific group (e.g., internal QA team, developers).
    *   **Recommendation**: Ideal for early-stage testing in production environments to catch critical issues before wider release. Typically used with 1-5% `rollout_percentage`.

-   **`GRADUAL_ROLLOUT`**: 
    *   **Use Case**: Incrementally exposing a feature to a larger percentage of your user base over time.
    *   **Recommendation**: Start with a small percentage (e.g., 5-10%) and gradually increase it (e.g., 25%, 50%, 75%, 100%) while continuously monitoring performance and user feedback. This is the safest way to roll out most new features.

-   **`AB_TEST`**: 
    *   **Use Case**: When you have multiple variants of a feature (including a control) and want to scientifically determine which performs best against specific metrics.
    *   **Recommendation**: Ensure clear hypotheses, sufficient sample sizes, and run the experiment for a predefined duration. Leverage the system's statistical analysis capabilities.

-   **`ENABLED`**: 
    *   **Use Case**: When a feature has been fully tested, validated, and proven beneficial through a gradual rollout or A/B test, and is ready for all users.
    *   **Recommendation**: Only switch to `ENABLED` once confidence in the feature's stability and positive impact is high.

### Best Practices for Rollouts:

-   **Monitor Closely**: During any rollout (especially `GRADUAL_ROLLOUT` and `CANARY`), closely monitor key metrics, error rates, and user feedback. Be prepared to quickly `DISABLED` or `rollback` the feature if negative impacts are detected.
-   **Define Rollback Plan**: Always have a clear plan for reverting to a previous state if issues arise. The configuration management system facilitates this with versioning and rollback capabilities.
-   **Communicate Changes**: Inform relevant stakeholders (product, engineering, support) about rollout plans and progress.

This comprehensive approach to experiment design and rollout strategies ensures robust feature management and data-driven product development.

## Feature Flag Lifecycle

### 1. Development Phase
```python
# Start with disabled flag during development
flag = FeatureFlag(
    name="new_feature",
    status=FeatureStatus.DISABLED,
    description="New experimental feature"
)
```

### 2. Internal Testing
```python
# Enable for internal team only
await manager.update_feature_flag("new_feature", {
    "status": FeatureStatus.GRADUAL_ROLLOUT,
    "rollout_percentage": 5.0  # 5% rollout for internal testing
})
```

### 3. Gradual Rollout
```python
# Gradually increase rollout
await manager.update_feature_flag("new_feature", {
    "rollout_percentage": 25.0  # 25% of users
})

# Monitor metrics and increase
await manager.update_feature_flag("new_feature", {
    "rollout_percentage": 50.0  # 50% of users
})
```

### 4. A/B Testing
```python
# Convert to A/B test to validate improvements
variants = [
    FeatureVariant(name="control", weight=50.0),
    FeatureVariant(name="treatment", weight=50.0, 
                  config_overrides={"enhanced": True})
]

await manager.update_feature_flag("new_feature", {
    "status": FeatureStatus.AB_TEST,
    "variants": [manager._variant_to_dict(v) for v in variants],
    "rollout_percentage": 100.0
})
```

### 5. Full Rollout
```python
# After successful A/B test, enable for everyone
await manager.update_feature_flag("new_feature", {
    "status": FeatureStatus.ENABLED,
    "rollout_percentage": 100.0
})
```

### 6. Cleanup
```python
# Remove flag after feature is stable
# (Or keep disabled for emergency rollback)
await manager.update_feature_flag("new_feature", {
    "status": FeatureStatus.DISABLED
})
```

## Integration with RAG CMS

### Configuration Integration

```python
from src.config import ConfigurationManager, FeatureFlagManager

class RAGService:
    def __init__(self, supabase_client):
        self.config_manager = ConfigurationManager(supabase_client)
        self.feature_flag_manager = FeatureFlagManager(supabase_client)
    
    async def query(self, user_query, user_context=None):
        # Get base configuration
        config = await self.config_manager.get_active_config()
        
        # Check feature flags
        if await self.feature_flag_manager.is_feature_enabled(
            "enhanced_search", user_context
        ):
            # Use enhanced search algorithm
            return await self.enhanced_search(user_query, config)
        else:
            # Use standard search
            return await self.standard_search(user_query, config)
```

### Monitoring Integration

```python
from src.monitoring import PerformanceProfiler

class RAGService:
    async def query_with_monitoring(self, user_query, user_context=None):
        # Get A/B test variant
        variant = await self.feature_flag_manager.get_variant(
            "search_algorithm_test", user_context
        )
        
        # Track experiment metrics
        if variant:
            with PerformanceProfiler() as profiler:
                result = await self.search_with_variant(user_query, variant)
                
                # Track performance metrics
                await self.feature_flag_manager.track_metric(
                    experiment_id="search_experiment_123",
                    variant_name=variant.name,
                    metric_type="response_time_ms",
                    metric_value=profiler.total_duration * 1000,
                    user_context=user_context
                )
                
                return result
```

## Best Practices

### 1. Feature Flag Naming
- Use descriptive, hierarchical names: `search.enhanced_algorithm`
- Include team/component: `recommendations.collaborative_filtering`
- Version for major changes: `model.gpt4_v2`

### 2. Gradual Rollouts
- Start small (1-5%) for high-risk features
- Monitor key metrics at each stage
- Have rollback plan ready
- Increase rollout gradually: 5% → 25% → 50% → 100%

### 3. A/B Testing
- Define hypothesis upfront
- Choose appropriate sample size
- Run tests for sufficient duration (typically 1-2 weeks)
- Monitor both success and guardrail metrics
- Don't peek at results too early

### 4. User Context
- Include relevant user identifiers (`user_id`, `session_id`)
- Add contextual information (`user_type`, `subscription_tier`)
- Keep context consistent across requests

### 5. Cleanup
- Remove unused feature flags regularly
- Archive completed experiments
- Document learnings and decisions

## Error Handling

### Graceful Degradation
```python
async def search_with_fallback(self, query, user_context=None):
    try:
        if await self.feature_flag_manager.is_feature_enabled(
            "enhanced_search", user_context
        ):
            return await self.enhanced_search(query)
    except Exception as e:
        # Log error but don't fail the request
        logger.error(f"Feature flag check failed: {e}")
    
    # Fallback to standard search
    return await self.standard_search(query)
```

### Cache Handling
```python
# Feature flags are cached for 5 minutes by default
# Force refresh if needed
flag = await manager.get_feature_flag("feature_name")
# Or clear cache
manager._flag_cache.clear()
```

## Database Schema

### Tables Created

The migration `004_create_feature_flags_tables.sql` creates:

1. **feature_flags**: Main feature flag configuration
2. **ab_test_experiments**: Experiment metadata and configuration  
3. **ab_test_metrics**: Individual metric events for analysis

### Key Indexes

- `idx_feature_flags_name`: Fast lookup by name
- `idx_feature_flags_status`: Filter by status
- `idx_metrics_experiment`: Aggregate metrics by experiment
- `idx_metrics_time`: Time-based queries for analytics

## Performance Considerations

### Caching
- Feature flags cached for 5 minutes
- Cache invalidated on updates
- Consider longer cache for stable flags

### Database Performance
- Indexes optimized for common queries
- Metrics table may grow large - consider partitioning
- Use connection pooling for high traffic

### Statistical Calculations
- Uses SciPy when available for accurate calculations
- Falls back to approximations if SciPy unavailable
- Calculations only performed for experiments with sufficient data (>30 samples per variant)

## Troubleshooting

### Common Issues

1. **Feature flag not updating**
   - Check cache TTL (5 minutes default)
   - Verify database update succeeded
   - Clear cache if needed

2. **Inconsistent user assignment**
   - Ensure user_id is consistent
   - Check segmentation strategy configuration
   - Verify hash salt is stable

3. **Statistical analysis not appearing**
   - Ensure sufficient sample size (>30 per variant)
   - Check that experiment has 2 variants
   - Verify metric data is being tracked

4. **SciPy import errors**
   - System falls back to approximations automatically
   - Install SciPy for full statistical capabilities
   - Check `SCIPY_AVAILABLE` flag in results

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check feature flag state
flag = await manager.get_feature_flag("feature_name")
print(f"Flag status: {flag.status if flag else 'Not found'}")

# Test segmentation
strategy = manager._get_segmentation_strategy(flag)
result = strategy.should_enable_feature(user_context, flag.rollout_percentage)
print(f"User enabled: {result}")

# Check experiment results
results = await manager.get_experiment_results(experiment_id)
print(json.dumps(results, indent=2, default=str))
```

This comprehensive feature flag and A/B testing system provides the foundation for data-driven feature development and experimentation in the RAG CMS platform. 