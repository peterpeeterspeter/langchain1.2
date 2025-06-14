"""
Task 3.7: Retrieval Configuration Management

This module provides configuration management for the contextual retrieval system,
enabling performance optimization through environment-based settings.
"""

from pydantic import BaseSettings, Field
from typing import Optional
from enum import Enum


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    MULTI_QUERY = "multi_query"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    QUALITY_FOCUSED = "quality_focused"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class RetrievalSettings(BaseSettings):
    """Settings for contextual retrieval system."""
    
    # Hybrid search weights
    dense_weight: float = Field(0.7, env="RETRIEVAL_DENSE_WEIGHT")
    sparse_weight: float = Field(0.3, env="RETRIEVAL_SPARSE_WEIGHT")
    
    # MMR settings
    mmr_lambda: float = Field(0.7, env="RETRIEVAL_MMR_LAMBDA")
    mmr_k: int = Field(20, env="RETRIEVAL_MMR_K")
    
    # Contextual settings
    context_window_size: int = Field(2, env="RETRIEVAL_CONTEXT_WINDOW")
    include_document_title: bool = Field(True, env="RETRIEVAL_INCLUDE_TITLE")
    include_section_headers: bool = Field(True, env="RETRIEVAL_INCLUDE_HEADERS")
    
    # Multi-query settings
    max_query_variations: int = Field(3, env="RETRIEVAL_MAX_QUERY_VARIATIONS")
    query_expansion_model: str = Field("gpt-4", env="RETRIEVAL_QUERY_MODEL")
    
    # Performance settings
    enable_caching: bool = Field(True, env="RETRIEVAL_ENABLE_CACHE")
    cache_ttl_hours: int = Field(24, env="RETRIEVAL_CACHE_TTL_HOURS")
    parallel_retrieval: bool = Field(True, env="RETRIEVAL_ENABLE_PARALLEL")
    max_workers: int = Field(4, env="RETRIEVAL_MAX_WORKERS")
    
    # Search thresholds
    similarity_threshold: float = Field(0.7, env="RETRIEVAL_SIMILARITY_THRESHOLD")
    default_k: int = Field(10, env="RETRIEVAL_DEFAULT_K")
    
    # Optimization settings
    optimization_strategy: OptimizationStrategy = Field(
        OptimizationStrategy.BALANCED, 
        env="RETRIEVAL_OPTIMIZATION_STRATEGY"
    )
    enable_performance_monitoring: bool = Field(True, env="RETRIEVAL_ENABLE_MONITORING")
    
    # Performance targets
    target_latency_ms: float = Field(500.0, env="RETRIEVAL_TARGET_LATENCY_MS")
    target_throughput_qps: float = Field(10.0, env="RETRIEVAL_TARGET_THROUGHPUT_QPS")
    target_cache_hit_rate: float = Field(0.7, env="RETRIEVAL_TARGET_CACHE_HIT_RATE")
    target_quality_score: float = Field(0.8, env="RETRIEVAL_TARGET_QUALITY_SCORE")
    
    # Resource limits
    max_concurrent_queries: int = Field(20, env="RETRIEVAL_MAX_CONCURRENT_QUERIES")
    max_memory_usage_mb: float = Field(1024.0, env="RETRIEVAL_MAX_MEMORY_MB")
    max_api_calls_per_minute: int = Field(100, env="RETRIEVAL_MAX_API_CALLS_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
retrieval_settings = RetrievalSettings()


def get_optimized_config_for_strategy(strategy: OptimizationStrategy) -> dict:
    """Get optimized configuration for a specific strategy."""
    
    base_config = {
        "dense_weight": retrieval_settings.dense_weight,
        "sparse_weight": retrieval_settings.sparse_weight,
        "mmr_lambda": retrieval_settings.mmr_lambda,
        "context_window_size": retrieval_settings.context_window_size,
        "max_query_variations": retrieval_settings.max_query_variations,
        "default_k": retrieval_settings.default_k
    }
    
    if strategy == OptimizationStrategy.LATENCY_FOCUSED:
        return {
            **base_config,
            "dense_weight": 0.9,
            "sparse_weight": 0.1,
            "context_window_size": 1,
            "max_query_variations": 1,
            "default_k": 5
        }
    
    elif strategy == OptimizationStrategy.QUALITY_FOCUSED:
        return {
            **base_config,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "context_window_size": 4,
            "max_query_variations": 5,
            "default_k": 20,
            "mmr_lambda": 0.9
        }
    
    elif strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
        return {
            **base_config,
            "dense_weight": 0.8,
            "sparse_weight": 0.2,
            "context_window_size": 1,
            "max_query_variations": 1,
            "default_k": 3
        }
    
    else:  # BALANCED or ADAPTIVE
        return base_config


__all__ = [
    'RetrievalSettings',
    'RetrievalStrategy', 
    'OptimizationStrategy',
    'retrieval_settings',
    'get_optimized_config_for_strategy'
] 