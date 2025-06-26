"""
Content Type Configuration Module.

This module defines configuration systems for content-type specific cache settings.
It provides configurations for different content types (news, reviews, regulatory)
including distance thresholds, TTL values, embedding model parameters, and other
cache-specific settings.

The configuration system is implemented using dataclasses and Pydantic models
for type safety and validation. It supports:
- Distance threshold optimization per content type
- TTL configuration in seconds
- Embedding model parameter configuration
- Cache size and performance settings
- Content-specific semantic similarity parameters
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

from .exceptions import CacheConfigurationException


class ContentType(str, Enum):
    """Enumeration of supported content types."""
    NEWS = "news"
    REVIEWS = "reviews"
    REGULATORY = "regulatory"


class EmbeddingModel(str, Enum):
    """Enumeration of supported embedding models."""
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"


@dataclass
class CachePerformanceSettings:
    """Performance settings for cache operations."""
    max_cache_size: int = 10000
    max_memory_usage_mb: int = 512
    cleanup_threshold: float = 0.8
    batch_size: int = 100


class ContentTypeConfig(BaseModel):
    """Configuration for a specific content type."""
    
    content_type: ContentType
    distance_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=3600, ge=60)  # TTL in seconds
    embedding_model: EmbeddingModel = EmbeddingModel.OPENAI_ADA_002
    embedding_dimensions: int = Field(default=1536, gt=0)
    max_query_length: int = Field(default=8000, gt=0)
    cache_prefix: str = Field(default="cache")
    performance_settings: CachePerformanceSettings = Field(default_factory=CachePerformanceSettings)
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('cache_prefix')
    def validate_cache_prefix(cls, v):
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Cache prefix must be alphanumeric with optional underscores or hyphens")
        return v
    
    @validator('ttl_seconds')
    def validate_ttl_seconds(cls, v):
        if v <= 0:
            raise ValueError("TTL must be positive integer representing seconds")
        return v


class ContentTypeConfigFactory:
    """Factory class for creating content-type specific configurations."""
    
    _default_configs: Dict[ContentType, Dict[str, Any]] = {
        ContentType.NEWS: {
            "distance_threshold": 0.15,
            "ttl_seconds": 1800,  # 30 minutes - news changes frequently
            "embedding_model": EmbeddingModel.OPENAI_3_SMALL,
            "cache_prefix": "news_cache",
            "performance_settings": CachePerformanceSettings(
                max_cache_size=5000,
                max_memory_usage_mb=256,
                cleanup_threshold=0.75
            )
        },
        ContentType.REVIEWS: {
            "distance_threshold": 0.2,
            "ttl_seconds": 7200,  # 2 hours - reviews are semi-static
            "embedding_model": EmbeddingModel.OPENAI_ADA_002,
            "cache_prefix": "reviews_cache",
            "performance_settings": CachePerformanceSettings(
                max_cache_size=15000,
                max_memory_usage_mb=512,
                cleanup_threshold=0.8
            )
        },
        ContentType.REGULATORY: {
            "distance_threshold": 0.1,
            "ttl_seconds": 86400,  # 24 hours - regulatory info is stable
            "embedding_model": EmbeddingModel.OPENAI_3_LARGE,
            "cache_prefix": "regulatory_cache",
            "performance_settings": CachePerformanceSettings(
                max_cache_size=20000,
                max_memory_usage_mb=1024,
                cleanup_threshold=0.85
            )
        }
    }
    
    @classmethod
    def create_config(cls, content_type: ContentType, **overrides) -> ContentTypeConfig:
        """Create a configuration for the specified content type with optional overrides."""
        try:
            default_config = cls._default_configs[content_type].copy()
            default_config.update(overrides)
            default_config['content_type'] = content_type
            
            return ContentTypeConfig(**default_config)
        except KeyError:
            raise CacheConfigurationException(
                f"No default configuration found for content type: {content_type}",
                config_key=f"content_type.{content_type}"
            )
        except Exception as e:
            raise CacheConfigurationException(
                f"Failed to create configuration for content type {content_type}: {str(e)}",
                config_key=f"content_type.{content_type}",
                details={"overrides": overrides}
            )
    
    @classmethod
    def get_all_configs(cls) -> Dict[ContentType, ContentTypeConfig]:
        """Get configurations for all supported content types."""
        configs = {}
        for content_type in ContentType:
            configs[content_type] = cls.create_config(content_type)
        return configs
    
    @classmethod
    def update_default_config(cls, content_type: ContentType, **updates) -> None:
        """Update the default configuration for a content type."""
        if content_type not in cls._default_configs:
            raise CacheConfigurationException(
                f"Cannot update unknown content type: {content_type}",
                config_key=f"content_type.{content_type}"
            )
        
        cls._default_configs[content_type].update(updates)


class ConfigurationManager:
    """Manager class for handling content type configurations."""
    
    def __init__(self):
        self._configs: Dict[ContentType, ContentTypeConfig] = {}
        self._load_default_configs()
    
    def _load_default_configs(self) -> None:
        """Load default configurations for all content types."""
        self._configs = ContentTypeConfigFactory.get_all_configs()
    
    def get_config(self, content_type: ContentType) -> ContentTypeConfig:
        """Get configuration for a specific content type."""
        if content_type not in self._configs:
            raise CacheConfigurationException(
                f"Configuration not found for content type: {content_type}",
                config_key=f"content_type.{content_type}"
            )
        return self._configs[content_type]
    
    def update_config(self, content_type: ContentType, **updates) -> None:
        """Update configuration for a specific content type."""
        try:
            current_config = self.get_config(content_type)
            updated_data = current_config.dict()
            updated_data.update(updates)
            self._configs[content_type] = ContentTypeConfig(**updated_data)
        except Exception as e:
            raise CacheConfigurationException(
                f"Failed to update configuration for content type {content_type}: {str(e)}",
                config_key=f"content_type.{content_type}",
                details={"updates": updates}
            )
    
    def get_all_configs(self) -> Dict[ContentType, ContentTypeConfig]:
        """Get all content type configurations."""
        return self._configs.copy()
    
    def reload_configs(self) -> None:
        """Reload configurations from defaults."""
        self._load_default_configs() 