"""
Production Configuration and Settings for Contextual Retrieval System

This module provides comprehensive configuration management for the Universal RAG CMS
contextual retrieval system with production-ready features.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic.env_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseModel, Field, validator
    from pydantic import BaseSettings
    
    def root_validator(func):
        return validator('*', pre=True, allow_reuse=True)(func)

# Core enums
class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse" 
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    MULTI_QUERY = "multi_query"

class CacheStrategy(Enum):
    """Cache optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"

class PerformanceProfile(Enum):
    """Performance optimization profiles."""
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    RESOURCE_OPTIMIZED = "resource_optimized"

# Component configurations
class ContextualEmbeddingConfig(BaseModel):
    """Configuration for contextual embedding system."""
    
    context_window_size: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of sentences before/after for context"
    )
    include_document_title: bool = Field(
        default=True,
        description="Include document title in contextualized text"
    )
    include_section_headers: bool = Field(
        default=True,
        description="Include section headers in context"
    )
    max_context_length: int = Field(
        default=512,
        ge=100,
        le=2048,
        description="Maximum context length in tokens"
    )
    overlap_strategy: str = Field(
        default="sliding_window",
        description="Strategy for handling context overlaps"
    )

class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search system."""
    
    dense_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for dense vector search"
    )
    sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for sparse BM25 search"
    )
    enable_reranking: bool = Field(
        default=True,
        description="Enable cross-encoder reranking"
    )
    rerank_top_k: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of documents to rerank"
    )
    bm25_k1: float = Field(
        default=1.2,
        ge=0.1,
        le=3.0,
        description="BM25 k1 parameter"
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b parameter"
    )
    
    @root_validator
    def validate_weights(cls, values):
        """Validate that weights sum to 1.0."""
        if isinstance(values, dict):
            dense_weight = values.get('dense_weight', 0.7)
            sparse_weight = values.get('sparse_weight', 0.3)
            total = dense_weight + sparse_weight
            
            if abs(total - 1.0) > 0.01:
                raise ValueError(f'Dense and sparse weights must sum to 1.0, got {total}')
        
        return values

class MultiQueryConfig(BaseModel):
    """Configuration for multi-query retrieval."""
    
    max_query_variations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of query variations"
    )
    query_expansion_model: str = Field(
        default="gpt-4.1-mini",
        description="Model for query expansion"
    )
    enable_parallel_search: bool = Field(
        default=True,
        description="Enable parallel processing"
    )
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum worker threads"
    )

class SelfQueryConfig(BaseModel):
    """Configuration for self-query metadata filtering."""
    
    enable_metadata_filtering: bool = Field(
        default=True,
        description="Enable automatic metadata filtering"
    )
    filter_extraction_model: str = Field(
        default="gpt-4.1-mini",
        description="Model for filter extraction"
    )
    max_filters_per_query: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum filters per query"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for filters"
    )
    supported_filter_types: List[str] = Field(
        default=["category", "date_range", "author", "content_type", "domain"],
        description="Supported filter types"
    )

class MMRConfig(BaseModel):
    """Configuration for Maximal Marginal Relevance."""
    
    lambda_param: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Balance between relevance and diversity"
    )
    mmr_k: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Documents to fetch before MMR"
    )
    diversity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum diversity score"
    )
    enable_metadata_diversity: bool = Field(
        default=True,
        description="Consider metadata in diversity"
    )

class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""
    
    max_response_time_ms: int = Field(
        default=2000,
        ge=100,
        le=30000,
        description="Maximum response time in ms"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable intelligent caching"
    )
    cache_strategy: CacheStrategy = Field(
        default=CacheStrategy.ADAPTIVE,
        description="Caching strategy"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Cache TTL in hours"
    )
    enable_request_batching: bool = Field(
        default=True,
        description="Enable request batching"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for requests"
    )
    connection_pool_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Database connection pool size"
    )

class MonitoringConfig(BaseModel):
    """Configuration for monitoring and logging."""
    
    enable_metrics_collection: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to retain metrics"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    enable_query_logging: bool = Field(
        default=True,
        description="Enable query logging"
    )
    enable_performance_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    alert_on_slow_queries: bool = Field(
        default=True,
        description="Alert on slow queries"
    )
    slow_query_threshold_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Slow query threshold in ms"
    )

class APIConfig(BaseModel):
    """Configuration for API endpoints."""
    
    enable_api: bool = Field(
        default=True,
        description="Enable REST API"
    )
    host: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    port: int = Field(
        default=8000,
        ge=1000,
        le=65535,
        description="API port"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS origins"
    )
    rate_limit_requests_per_minute: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Rate limit per minute"
    )
    enable_api_key_auth: bool = Field(
        default=False,
        description="Enable API key auth"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )

class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    
    supabase_url: str = Field(
        ...,
        description="Supabase project URL"
    )
    supabase_key: str = Field(
        ...,
        description="Supabase service role key"
    )
    enable_row_level_security: bool = Field(
        default=True,
        description="Enable RLS"
    )
    connection_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Connection timeout"
    )
    query_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Query timeout"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max retry attempts"
    )

# Main configuration class
class RetrievalSettings(BaseSettings):
    """Main configuration class for contextual retrieval system."""
    
    # Environment settings
    environment: DeploymentEnvironment = Field(
        default=DeploymentEnvironment.DEVELOPMENT,
        description="Deployment environment"
    )
    performance_profile: PerformanceProfile = Field(
        default=PerformanceProfile.BALANCED,
        description="Performance profile"
    )
    
    # Component configurations
    contextual_embedding: ContextualEmbeddingConfig = Field(
        default_factory=ContextualEmbeddingConfig,
        description="Contextual embedding config"
    )
    hybrid_search: HybridSearchConfig = Field(
        default_factory=HybridSearchConfig,
        description="Hybrid search config"
    )
    multi_query: MultiQueryConfig = Field(
        default_factory=MultiQueryConfig,
        description="Multi-query config"
    )
    self_query: SelfQueryConfig = Field(
        default_factory=SelfQueryConfig,
        description="Self-query config"
    )
    mmr: MMRConfig = Field(
        default_factory=MMRConfig,
        description="MMR config"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance config"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring config"
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API config"
    )
    database: DatabaseConfig = Field(
        default_factory=lambda: DatabaseConfig(
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", "")
        ),
        description="Database config"
    )
    
    # API keys
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key"
    )
    
    # Feature flags
    enable_contextual_retrieval: bool = Field(
        default=True,
        description="Enable contextual retrieval"
    )
    enable_task2_integration: bool = Field(
        default=True,
        description="Enable Task 2 integration"
    )
    enable_experimental_features: bool = Field(
        default=False,
        description="Enable experimental features"
    )
    
    class Config:
        env_prefix = "RETRIEVAL_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def apply_performance_profile(self):
        """Apply performance profile optimizations to configuration."""
        if self.performance_profile == PerformanceProfile.SPEED_OPTIMIZED:
            # Optimize for speed
            self.performance.max_response_time_ms = 1000
            self.performance.cache_strategy = CacheStrategy.AGGRESSIVE
            self.hybrid_search.rerank_top_k = 20
            self.multi_query.max_query_variations = 2
            self.contextual_embedding.context_window_size = 1
            
        elif self.performance_profile == PerformanceProfile.QUALITY_OPTIMIZED:
            # Optimize for quality
            self.performance.max_response_time_ms = 5000
            self.performance.cache_strategy = CacheStrategy.CONSERVATIVE
            self.hybrid_search.rerank_top_k = 100
            self.multi_query.max_query_variations = 5
            self.contextual_embedding.context_window_size = 3
            
        elif self.performance_profile == PerformanceProfile.RESOURCE_OPTIMIZED:
            # Minimize resource usage
            self.performance.connection_pool_size = 10
            self.performance.batch_size = 5
            self.multi_query.max_workers = 2
            self.hybrid_search.rerank_top_k = 30
    
    def get_query_type_config(self, query_type: str) -> Dict[str, Any]:
        """Get optimized configuration for specific query type."""
        base_config = self.dict()
        
        # Query-type specific optimizations
        optimizations = {
            "factual": {
                "hybrid_search.dense_weight": 0.8,
                "hybrid_search.sparse_weight": 0.2,
                "mmr.lambda_param": 0.8,
                "self_query.enable_metadata_filtering": True
            },
            "comparison": {
                "hybrid_search.dense_weight": 0.6,
                "hybrid_search.sparse_weight": 0.4,
                "mmr.lambda_param": 0.6,
                "multi_query.max_query_variations": 4
            },
            "tutorial": {
                "contextual_embedding.context_window_size": 3,
                "mmr.lambda_param": 0.7,
                "performance.cache_ttl_hours": 48
            },
            "news": {
                "performance.cache_ttl_hours": 2,
                "hybrid_search.enable_reranking": True,
                "mmr.lambda_param": 0.9
            }
        }
        
        if query_type in optimizations:
            for key, value in optimizations[query_type].items():
                keys = key.split('.')
                config_section = base_config
                for k in keys[:-1]:
                    config_section = config_section[k]
                config_section[keys[-1]] = value
        
        return base_config

# Configuration management utilities
class ConfigurationManager:
    """Manages configuration loading, validation, and hot-reload capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/retrieval_settings.json"
        self.config_cache = {}
        self.last_reload = datetime.now()
        self.reload_interval = timedelta(minutes=5)
        
    def load_config(
        self, 
        environment: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> RetrievalSettings:
        """Load configuration with optional environment and overrides."""
        
        # Start with default configuration
        config_data = {}
        
        # Load from file if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
        
        # Apply environment-specific overrides
        if environment:
            env_config_path = f"config/retrieval_settings_{environment}.json"
            if os.path.exists(env_config_path):
                with open(env_config_path, 'r') as f:
                    env_config = json.load(f)
                    config_data.update(env_config)
        
        # Apply runtime overrides
        if config_overrides:
            config_data.update(config_overrides)
        
        # Create settings instance
        settings = RetrievalSettings(**config_data)
        
        # Apply performance profile optimizations
        settings.apply_performance_profile()
        
        # Cache the configuration
        cache_key = f"{environment}_{hash(str(config_overrides))}"
        self.config_cache[cache_key] = {
            'config': settings,
            'loaded_at': datetime.now()
        }
        
        return settings
    
    def reload_config_if_needed(self, current_config: RetrievalSettings) -> RetrievalSettings:
        """Reload configuration if the reload interval has passed."""
        
        if datetime.now() - self.last_reload > self.reload_interval:
            try:
                # Check if config file has been modified
                if os.path.exists(self.config_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(self.config_path))
                    if file_mtime > self.last_reload:
                        # Reload configuration
                        new_config = self.load_config(
                            environment=current_config.environment.value
                        )
                        logging.info("Configuration reloaded due to file changes")
                        return new_config
                
                self.last_reload = datetime.now()
            except Exception as e:
                logging.error(f"Failed to reload configuration: {e}")
        
        return current_config
    
    def validate_config(self, config: RetrievalSettings) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check API keys in production
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.openai_api_key:
                issues.append("OpenAI API key required for production")
            if not config.database.supabase_url:
                issues.append("Supabase URL required for production")
            if not config.database.supabase_key:
                issues.append("Supabase key required for production")
        
        # Validate performance settings
        if config.performance.max_response_time_ms < 500:
            issues.append("Response time threshold too low, may cause timeouts")
        
        # Validate hybrid search weights
        total_weight = config.hybrid_search.dense_weight + config.hybrid_search.sparse_weight
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Hybrid search weights must sum to 1.0, got {total_weight}")
        
        # Validate resource limits
        if config.performance.connection_pool_size < 5:
            issues.append("Connection pool size too small for production use")
        
        return issues
    
    def export_config(self, config: RetrievalSettings, output_path: str):
        """Export configuration to JSON file."""
        config_dict = config.dict()
        
        # Remove sensitive information
        sensitive_keys = ['openai_api_key', 'anthropic_api_key', 'supabase_key']
        for key in sensitive_keys:
            if key in config_dict:
                config_dict[key] = "***REDACTED***"
            # Also check nested configs
            for section in config_dict.values():
                if isinstance(section, dict) and key in section:
                    section[key] = "***REDACTED***"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logging.info(f"Configuration exported to {output_path}")

# Factory function for easy configuration creation
def create_retrieval_settings(
    environment: str = "development",
    preset: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> RetrievalSettings:
    """
    Factory function to create retrieval settings with presets and overrides.
    
    Args:
        environment: Target environment (development, staging, production)
        preset: Configuration preset (speed_optimized, quality_optimized, etc.)
        config_overrides: Additional configuration overrides
    
    Returns:
        Configured RetrievalSettings instance
    """
    
    # Start with environment preset
    config_data = {"environment": environment}
    
    # Apply preset if specified
    if preset == "speed_optimized":
        config_data["performance_profile"] = "speed_optimized"
    elif preset == "quality_optimized":
        config_data["performance_profile"] = "quality_optimized"
    elif preset == "resource_optimized":
        config_data["performance_profile"] = "resource_optimized"
    
    # Apply custom overrides
    if config_overrides:
        config_data.update(config_overrides)
    
    # Create and return settings
    settings = RetrievalSettings(**config_data)
    settings.apply_performance_profile()
    
    return settings

# Global configuration manager instance
config_manager = ConfigurationManager()

# Usage examples
if __name__ == "__main__":
    # Example: Load development configuration
    dev_config = create_retrieval_settings("development")
    print(f"Development config loaded: {dev_config.environment}")
    
    # Example: Load production with quality optimization
    prod_config = create_retrieval_settings(
        environment="production",
        preset="quality_optimized",
        config_overrides={
            "api": {"port": 8080},
            "performance": {"max_response_time_ms": 3000}
        }
    )
    print(f"Production config: {prod_config.performance_profile}")
    
    # Example: Validate configuration
    issues = config_manager.validate_config(prod_config)
    if issues:
        print(f"Configuration issues: {issues}")
    else:
        print("Configuration is valid") 