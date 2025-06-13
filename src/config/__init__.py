"""Configuration package for RAG CMS."""

from .prompt_config import (
    QueryType,
    CacheConfig,
    QueryClassificationConfig,
    ContextFormattingConfig,
    PerformanceConfig,
    BasicFeatureFlags,
    PromptOptimizationConfig,
    ConfigurationManager,
    get_config_manager
)

from .feature_flags import (
    FeatureStatus,
    SegmentationType,
    FeatureFlag,
    FeatureVariant,
    ExperimentMetrics,
    SegmentationStrategy,
    HashBasedSegmentation,
    RandomSegmentation,
    FeatureFlagManager,
    feature_flag
)

__all__ = [
    'QueryType',
    'CacheConfig',
    'QueryClassificationConfig',
    'ContextFormattingConfig',
    'PerformanceConfig',
    'BasicFeatureFlags',
    'PromptOptimizationConfig',
    'ConfigurationManager',
    'get_config_manager',
    # Advanced Feature Flags
    'FeatureStatus',
    'SegmentationType',
    'FeatureFlag',
    'FeatureVariant',
    'ExperimentMetrics',
    'SegmentationStrategy',
    'HashBasedSegmentation',
    'RandomSegmentation',
    'FeatureFlagManager',
    'feature_flag'
]

# Configuration package
# Database and storage configuration 