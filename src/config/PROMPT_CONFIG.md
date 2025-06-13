# Prompt Optimization and Configuration System

## Overview

The Prompt Optimization and Configuration System provides dynamic, real-time management of RAG pipeline parameters, enabling live updates without service restarts. It leverages Pydantic models for robust validation and integrates with Supabase for persistent storage and version control.

Key features include:

-   **Runtime Configuration**: Adjust RAG pipeline parameters (e.g., temperature, max tokens, system prompts) on the fly.
-   **Version Control & Rollback**: Maintain a history of configurations and revert to previous stable versions.
-   **Validation**: Ensure new configurations are valid before deployment to prevent errors.
-   **Query-Type Specific Settings**: Define unique caching and behavior rules per query type.
-   **Performance & Feature Flag Integration**: Configure monitoring thresholds and default feature flag states.

## Core Components

### `PromptOptimizationConfig`

The main Pydantic model for the entire RAG pipeline configuration. It aggregates various sub-configurations into a single, versioned entity.

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from src.config.prompt_config import (
    QueryClassificationConfig, ContextFormattingConfig,
    CacheConfig, PerformanceConfig, BasicFeatureFlags
)

class PromptOptimizationConfig(BaseModel):
    query_classification: QueryClassificationConfig = Field(default_factory=QueryClassificationConfig)
    context_formatting: ContextFormattingConfig = Field(default_factory=ContextFormattingConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    feature_flags: BasicFeatureFlags = Field(default_factory=BasicFeatureFlags)

    version: str = Field(default="1.0.0")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = Field(default=None)
    change_notes: Optional[str] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptOptimizationConfig':
        return cls.model_validate(data)

    def get_hash(self) -> str:
        # Generates a unique hash for the configuration, excluding volatile metadata
        # for consistent change detection.
        pass
```

### `ConfigurationManager`

Responsible for interacting with Supabase to store, retrieve, and manage configuration versions. It includes caching mechanisms to optimize performance.

```python
from supabase import Client
from src.config.prompt_config import PromptOptimizationConfig
from datetime import timedelta

class ConfigurationManager:
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_name = "prompt_configurations"
        self.cache_duration = timedelta(minutes=5) # Config caching
        # ... (other internal attributes)

    async def get_active_config(self, force_refresh: bool = False) -> PromptOptimizationConfig:
        # Fetches the currently active configuration, utilizing a cache.
        pass

    async def save_config(
        self,
        config: PromptOptimizationConfig,
        updated_by: str,
        change_notes: Optional[str] = None
    ) -> str:
        # Saves a new configuration, deactivating the previous active one.
        # Returns the ID of the new configuration.
        pass

    async def validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        # Validates a given configuration dictionary against the Pydantic model.
        # Returns a dictionary indicating validity and potential errors.
        pass

    async def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        # Retrieves a list of past configuration versions.
        pass

    async def rollback_config(self, version_id: str, updated_by: str) -> PromptOptimizationConfig:
        # Reverts to a previous configuration version by ID.
        pass
```

## Sub-Configurations

The `PromptOptimizationConfig` is composed of several specialized configuration models:

### `QueryClassificationConfig`

Defines settings for classifying incoming user queries.

```python
from pydantic import BaseModel, Field, validator
from src.config.prompt_config import QueryType

class QueryClassificationConfig(BaseModel):
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    fallback_type: QueryType = Field(default=QueryType.GENERAL)
    enable_multi_classification: bool = Field(default=False)
    max_classification_attempts: int = Field(default=2, ge=1, le=5)

    @validator('confidence_threshold')
    def validate_threshold(cls, v):
        # Ensure threshold is within optimal performance range
        pass
```

### `ContextFormattingConfig`

Controls how retrieved documents are formatted and presented to the LLM.

```python
from pydantic import BaseModel, Field, validator

class ContextFormattingConfig(BaseModel):
    max_context_length: int = Field(default=3000, ge=500, le=10000)
    quality_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    freshness_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = Field(default=True)
    max_chunks_per_source: int = Field(default=3, ge=1, le=10)
    chunk_overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.5)

    @validator('relevance_weight')
    def validate_weights(cls, v, values):
        # Ensure freshness and relevance weights sum to 1.0
        pass
```

### `CacheConfig`

Manages caching behavior, including time-to-live (TTL) settings for different query types.

```python
from pydantic import BaseModel, Field
from src.config.prompt_config import QueryType

class CacheConfig(BaseModel):
    casino_review_ttl: int = Field(default=24, description="Hours to cache casino reviews")
    news_ttl: int = Field(default=2, description="Hours to cache news content")
    # ... (other TTL fields for different query types)
    general_ttl: int = Field(default=6, description="Hours to cache general content")

    def get_ttl(self, query_type: QueryType) -> int:
        # Returns the appropriate TTL for a given query type.
        pass
```

### `PerformanceConfig`

Configures performance monitoring and alerting thresholds.

```python
from pydantic import BaseModel, Field

class PerformanceConfig(BaseModel):
    enable_monitoring: bool = Field(default=True)
    enable_profiling: bool = Field(default=False)
    response_time_warning_ms: int = Field(default=2000, ge=100, le=10000)
    response_time_critical_ms: int = Field(default=5000, ge=1000, le=30000)
    error_rate_warning_percent: float = Field(default=5.0, ge=0.1, le=20.0)
    error_rate_critical_percent: float = Field(default=10.0, ge=1.0, le=30.0)
    min_samples_for_alerts: int = Field(default=100, ge=10, le=1000)
    alert_cooldown_minutes: int = Field(default=15, ge=5, le=60)
```

### `BasicFeatureFlags`

Provides basic, legacy feature flag settings that can be overridden or complemented by the dedicated `FeatureFlagManager`.

```python
from pydantic import BaseModel, Field, validator

class BasicFeatureFlags(BaseModel):
    enable_contextual_retrieval: bool = Field(default=True)
    enable_hybrid_search: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=False)
    enable_response_caching: bool = Field(default=True)
    enable_semantic_cache: bool = Field(default=True)
    enable_auto_retry: bool = Field(default=True)
    enable_cost_optimization: bool = Field(default=True)
    ab_test_percentage: float = Field(default=100.0, ge=0.0, le=100.0)

    @validator('ab_test_percentage')
    def validate_percentage(cls, v):
        # Rounds percentage to 2 decimal places.
        pass
```

## Quick Start & Usage Examples

### Initializing and Accessing Configuration

```python
from src.config.prompt_config import ConfigurationManager, PromptOptimizationConfig
from supabase import create_client # Assuming supabase client is set up

# Initialize Supabase client
# url = "YOUR_SUPABASE_URL"
# key = "YOUR_SUPABASE_KEY"
# supabase = create_client(url, key)

# manager = ConfigurationManager(supabase)

async def get_current_config_example():
    active_config = await manager.get_active_config()
    print(f"Active System Prompt: {active_config.context_formatting.system_prompt}")
    print(f"Cache TTL for News: {active_config.cache_config.get_ttl(QueryType.NEWS)} hours")

# await get_current_config_example()
```

### Updating Configuration

```python
from src.config.prompt_config import PromptOptimizationConfig

async def update_config_example():
    current_config = await manager.get_active_config()
    
    # Modify a setting
    current_config.cache_config.news_ttl = 4 # Update news TTL to 4 hours
    current_config.performance.response_time_warning_ms = 1500 # Adjust warning threshold

    # Save the updated configuration
    config_id = await manager.save_config(
        current_config,
        updated_by="admin_user",
        change_notes="Adjusted news cache TTL and response time warning"
    )
    print(f"Configuration updated with ID: {config_id}")

# await update_config_example()
```

### Validating Configuration

```python
async def validate_config_example():
    invalid_config_data = {
        "cache_config": {"news_ttl": -5} # Invalid TTL
    }
    validation_result = await manager.validate_config(invalid_config_data)
    print(f"Validation Result: {validation_result['valid']}")
    if not validation_result['valid']:
        print(f"Errors: {validation_result['details']}")

# await validate_config_example()
```

### Configuration History and Rollback

```python
async def history_and_rollback_example():
    history = await manager.get_config_history(limit=5)
    print("Configuration History:")
    for entry in history:
        print(f"- Version: {entry['version']}, Updated By: {entry['updated_by']}, Created: {entry['created_at']}")
    
    if history:
        # Rollback to the second last version (example)
        old_version_id = history[1]['id']
        rolled_back_config = await manager.rollback_config(old_version_id, "admin_user")
        print(f"Rolled back to version: {rolled_back_config.version}")

# await history_and_rollback_example()
```

## Best Practices for Configuration Management

-   **Version Control**: Always use clear, descriptive `change_notes` when saving configurations.
-   **Validation**: Utilize `validate_config` before deploying any new configuration to production.
-   **Atomic Updates**: The system ensures only one configuration is active at a time.
-   **Security**: Implement robust RLS (Row Level Security) policies on your `prompt_configurations` Supabase table to control access.
-   **Monitoring Integration**: Link configuration changes to performance metrics to observe impact.

## Deployment Considerations

-   Ensure your `SUPABASE_URL` and `SUPABASE_KEY` environment variables are correctly set for the `ConfigurationManager` to connect to your Supabase project.
-   The `prompt_configurations` table and its indexes should be created as part of your database migration process (refer to `migrations/` for schema setup). 