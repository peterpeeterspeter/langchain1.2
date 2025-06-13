"""Configuration package for RAG CMS."""

from .prompt_config import (
    QueryType,
    CacheConfig,
    QueryClassificationConfig,
    ContextFormattingConfig,
    PerformanceConfig,
    FeatureFlags,
    PromptOptimizationConfig,
    ConfigurationManager,
    get_config_manager
)

__all__ = [
    'QueryType',
    'CacheConfig',
    'QueryClassificationConfig',
    'ContextFormattingConfig',
    'PerformanceConfig',
    'FeatureFlags',
    'PromptOptimizationConfig',
    'ConfigurationManager',
    'get_config_manager'
]

# Configuration package
# Database and storage configuration 