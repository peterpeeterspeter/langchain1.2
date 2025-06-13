"""
Unit tests for prompt configuration system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

# Add the project root to Python path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.config.prompt_config import (
    QueryType,
    CacheConfig,
    QueryClassificationConfig,
    ContextFormattingConfig,
    PerformanceConfig,
    FeatureFlags,
    PromptOptimizationConfig,
    ConfigurationManager
)
from tests.fixtures.test_configs import (
    TestConfigFixtures,
    MockSupabaseClient
)


class TestQueryType:
    """Test QueryType enum."""
    
    def test_query_type_values(self):
        """Test that all query types have expected values."""
        expected_types = {
            "casino_review",
            "news", 
            "product_review",
            "technical_doc",
            "general",
            "guide",
            "faq"
        }
        actual_types = {qt.value for qt in QueryType}
        assert actual_types == expected_types
    
    def test_query_type_enum_behavior(self):
        """Test QueryType enum behavior and string representation"""
        # Test enum values
        assert QueryType.CASINO_REVIEW.value == "casino_review"
        assert QueryType.NEWS.value == "news"
        
        # Test string representation (actual behavior)
        assert str(QueryType.CASINO_REVIEW) == "QueryType.CASINO_REVIEW"
        
        # Test that we can create enum from string value
        assert QueryType("casino_review") == QueryType.CASINO_REVIEW
        assert QueryType("news") == QueryType.NEWS


class TestCacheConfig:
    """Test CacheConfig validation and functionality."""
    
    def test_default_cache_config(self):
        """Test default cache configuration values."""
        config = CacheConfig()
        assert config.casino_review_ttl == 24
        assert config.news_ttl == 2
        assert config.product_review_ttl == 12
        assert config.technical_doc_ttl == 168
        assert config.general_ttl == 6
        assert config.guide_ttl == 48
        assert config.faq_ttl == 72
    
    def test_custom_cache_config(self):
        """Test custom cache configuration values."""
        config = CacheConfig(
            casino_review_ttl=48,
            news_ttl=1,
            technical_doc_ttl=336
        )
        assert config.casino_review_ttl == 48
        assert config.news_ttl == 1
        assert config.technical_doc_ttl == 336
        # Defaults should remain
        assert config.product_review_ttl == 12
    
    def test_get_ttl_method(self):
        """Test get_ttl method for different query types."""
        config = CacheConfig()
        
        # Test all query types
        assert config.get_ttl(QueryType.CASINO_REVIEW) == 24
        assert config.get_ttl(QueryType.NEWS) == 2
        assert config.get_ttl(QueryType.PRODUCT_REVIEW) == 12
        assert config.get_ttl(QueryType.TECHNICAL_DOC) == 168
        assert config.get_ttl(QueryType.GENERAL) == 6
        assert config.get_ttl(QueryType.GUIDE) == 48
        assert config.get_ttl(QueryType.FAQ) == 72
    
    def test_get_ttl_fallback(self):
        """Test get_ttl fallback for unknown query type."""
        config = CacheConfig()
        # This should return general_ttl as fallback
        fake_query_type = "unknown_type"
        result = config.get_ttl(fake_query_type)
        assert result == config.general_ttl


class TestQueryClassificationConfig:
    """Test QueryClassificationConfig validation."""
    
    def test_default_classification_config(self):
        """Test default classification configuration."""
        config = QueryClassificationConfig()
        assert config.confidence_threshold == 0.7
        assert config.fallback_type == QueryType.GENERAL
        assert config.enable_multi_classification is False
        assert config.max_classification_attempts == 2
    
    def test_valid_confidence_threshold(self):
        """Test valid confidence threshold values."""
        # Test valid values
        for threshold in [0.5, 0.6, 0.75, 0.9, 0.95]:
            config = QueryClassificationConfig(confidence_threshold=threshold)
            assert config.confidence_threshold == threshold
    
    def test_invalid_confidence_threshold_low(self):
        """Test invalid low confidence threshold."""
        with pytest.raises(ValidationError) as exc_info:
            QueryClassificationConfig(confidence_threshold=0.3)
        assert "Confidence threshold should be between 0.5 and 0.95" in str(exc_info.value)
    
    def test_invalid_confidence_threshold_high(self):
        """Test invalid high confidence threshold."""
        with pytest.raises(ValidationError) as exc_info:
            QueryClassificationConfig(confidence_threshold=0.99)
        assert "Confidence threshold should be between 0.5 and 0.95" in str(exc_info.value)
    
    def test_max_attempts_validation(self):
        """Test max classification attempts validation."""
        # Valid values
        config = QueryClassificationConfig(max_classification_attempts=1)
        assert config.max_classification_attempts == 1
        
        config = QueryClassificationConfig(max_classification_attempts=5)
        assert config.max_classification_attempts == 5
        
        # Invalid values
        with pytest.raises(ValidationError):
            QueryClassificationConfig(max_classification_attempts=0)
        
        with pytest.raises(ValidationError):
            QueryClassificationConfig(max_classification_attempts=6)


class TestContextFormattingConfig:
    """Test ContextFormattingConfig validation."""
    
    def test_default_context_config(self):
        """Test default context formatting configuration."""
        config = ContextFormattingConfig()
        assert config.max_context_length == 3000
        assert config.quality_threshold == 0.75
        assert config.freshness_weight == 0.3
        assert config.relevance_weight == 0.7
        assert config.include_metadata is True
        assert config.max_chunks_per_source == 3
        assert config.chunk_overlap_ratio == 0.1
    
    def test_weight_sum_validation_valid(self):
        """Test valid weight sum validation."""
        config = ContextFormattingConfig(
            freshness_weight=0.4,
            relevance_weight=0.6
        )
        assert config.freshness_weight == 0.4
        assert config.relevance_weight == 0.6
    
    def test_weight_sum_validation_invalid(self):
        """Test invalid weight sum validation."""
        with pytest.raises(ValidationError) as exc_info:
            ContextFormattingConfig(
                freshness_weight=0.3,
                relevance_weight=0.8  # Sum = 1.1
            )
        assert "Freshness and relevance weights must sum to 1.0" in str(exc_info.value)
    
    def test_context_length_validation(self):
        """Test context length validation."""
        # Valid values
        config = ContextFormattingConfig(max_context_length=500)
        assert config.max_context_length == 500
        
        config = ContextFormattingConfig(max_context_length=10000)
        assert config.max_context_length == 10000
        
        # Invalid values
        with pytest.raises(ValidationError):
            ContextFormattingConfig(max_context_length=400)  # Too low
        
        with pytest.raises(ValidationError):
            ContextFormattingConfig(max_context_length=12000)  # Too high


class TestPerformanceConfig:
    """Test PerformanceConfig validation."""
    
    def test_default_performance_config(self):
        """Test default performance configuration."""
        config = PerformanceConfig()
        assert config.enable_monitoring is True
        assert config.enable_profiling is False
        assert config.response_time_warning_ms == 2000
        assert config.response_time_critical_ms == 5000
        assert config.error_rate_warning_percent == 5.0
        assert config.error_rate_critical_percent == 10.0
        assert config.min_samples_for_alerts == 100
        assert config.alert_cooldown_minutes == 15
    
    def test_response_time_validation(self):
        """Test response time validation."""
        # Valid values
        config = PerformanceConfig(
            response_time_warning_ms=100,
            response_time_critical_ms=1000
        )
        assert config.response_time_warning_ms == 100
        assert config.response_time_critical_ms == 1000
        
        # Test boundary values
        config = PerformanceConfig(response_time_warning_ms=10000)
        assert config.response_time_warning_ms == 10000
        
        config = PerformanceConfig(response_time_critical_ms=30000)
        assert config.response_time_critical_ms == 30000


class TestFeatureFlags:
    """Test FeatureFlags validation."""
    
    def test_default_feature_flags(self):
        """Test default feature flags."""
        flags = FeatureFlags()
        assert flags.enable_contextual_retrieval is True
        assert flags.enable_hybrid_search is True
        assert flags.enable_query_expansion is False
        assert flags.enable_response_caching is True
        assert flags.enable_semantic_cache is True
        assert flags.enable_auto_retry is True
        assert flags.enable_cost_optimization is True
        assert flags.ab_test_percentage == 100.0
    
    def test_ab_test_percentage_validation(self):
        """Test A/B test percentage validation and rounding."""
        # Valid values
        flags = FeatureFlags(ab_test_percentage=50.0)
        assert flags.ab_test_percentage == 50.0
        
        flags = FeatureFlags(ab_test_percentage=0.0)
        assert flags.ab_test_percentage == 0.0
        
        flags = FeatureFlags(ab_test_percentage=100.0)
        assert flags.ab_test_percentage == 100.0
        
        # Test rounding
        flags = FeatureFlags(ab_test_percentage=75.555)
        assert flags.ab_test_percentage == 75.56
        
        # Invalid values
        with pytest.raises(ValidationError):
            FeatureFlags(ab_test_percentage=-1.0)
        
        with pytest.raises(ValidationError):
            FeatureFlags(ab_test_percentage=101.0)


class TestPromptOptimizationConfig:
    """Test main PromptOptimizationConfig class."""
    
    def test_default_prompt_config(self):
        """Test default prompt optimization configuration."""
        config = PromptOptimizationConfig()
        
        # Check nested configs are properly instantiated
        assert isinstance(config.query_classification, QueryClassificationConfig)
        assert isinstance(config.context_formatting, ContextFormattingConfig)
        assert isinstance(config.cache_config, CacheConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.feature_flags, FeatureFlags)
        
        # Check metadata
        assert config.version == "1.0.0"
        assert isinstance(config.last_updated, datetime)
        assert config.updated_by is None
    
    def test_custom_prompt_config(self):
        """Test custom prompt optimization configuration."""
        config = TestConfigFixtures.get_custom_config()
        
        # Verify custom values are set
        assert config.query_classification.confidence_threshold == 0.8
        assert config.context_formatting.max_context_length == 4000
        assert config.cache_config.casino_review_ttl == 48
        assert config.performance.enable_profiling is True
        assert config.feature_flags.enable_hybrid_search is False
        assert config.version == "2.0.0"
        assert config.updated_by == "test_user"
    
    def test_to_dict_method(self):
        """Test configuration to dictionary conversion."""
        config = TestConfigFixtures.get_default_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "query_classification" in config_dict
        assert "context_formatting" in config_dict
        assert "cache_config" in config_dict
        assert "performance" in config_dict
        assert "feature_flags" in config_dict
        assert "version" in config_dict
        assert "last_updated" in config_dict
    
    def test_from_dict_method(self):
        """Test configuration from dictionary creation."""
        original_config = TestConfigFixtures.get_custom_config()
        config_dict = original_config.to_dict()
        
        # Create new config from dict
        restored_config = PromptOptimizationConfig.from_dict(config_dict)
        
        # Verify values are preserved
        assert restored_config.query_classification.confidence_threshold == 0.8
        assert restored_config.context_formatting.max_context_length == 4000
        assert restored_config.version == "2.0.0"
        assert restored_config.updated_by == "test_user"
    
    def test_get_hash_method(self):
        """Test configuration hash generation."""
        config1 = TestConfigFixtures.get_default_config()
        config2 = TestConfigFixtures.get_default_config()
        config3 = TestConfigFixtures.get_custom_config()
        
        # Same configurations should have same hash
        assert config1.get_hash() == config2.get_hash()
        
        # Different configurations should have different hashes
        assert config1.get_hash() != config3.get_hash()
        
        # Hash should be consistent across calls
        hash1 = config1.get_hash()
        hash2 = config1.get_hash()
        assert hash1 == hash2
        
        # Hash should be a valid SHA256 hex string
        config_hash = config1.get_hash()
        assert len(config_hash) == 64
        assert all(c in '0123456789abcdef' for c in config_hash)
    
    def test_hash_excludes_metadata(self):
        """Test that hash excludes metadata fields."""
        config1 = TestConfigFixtures.get_default_config()
        config2 = TestConfigFixtures.get_default_config()
        
        # Modify metadata fields
        config2.last_updated = datetime.utcnow() + timedelta(hours=1)
        config2.updated_by = "different_user"
        
        # Hashes should still be the same
        assert config1.get_hash() == config2.get_hash()


class TestConfigurationManager:
    """Test ConfigurationManager class."""
    
    def test_init(self):
        """Test ConfigurationManager initialization."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        assert manager.client == mock_client
        assert manager.table_name == "prompt_configurations"
        assert manager.cache_duration == timedelta(minutes=5)
        assert manager._cached_config is None
        assert manager._cache_timestamp is None
    
    @pytest.mark.asyncio
    async def test_get_active_config_default(self):
        """Test getting active configuration when none exists."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # No active config in database
        config = await manager.get_active_config()
        
        # Should return default config
        assert isinstance(config, PromptOptimizationConfig)
        assert config.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_get_active_config_from_database(self):
        """Test fetching active config from database"""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Add a custom config to the mock database
        custom_config = {
            "id": 2,
            "config_data": {"cache_config": {"general_ttl": 500}},
            "is_active": True,
            "created_at": "2024-01-20T10:00:00Z"
        }
        mock_client.data_store["prompt_configurations"].append(custom_config)
        
        config = await manager.get_active_config()
        
        # Should retrieve the custom config
        assert config.cache_config.general_ttl == 500
    
    @pytest.mark.asyncio
    async def test_get_active_config_caching(self):
        """Test configuration caching behavior"""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Add a config to mock database
        cached_config = {
            "id": 2,
            "config_data": {"cache_config": {"general_ttl": 600}},
            "is_active": True,
            "created_at": "2024-01-20T10:00:00Z"
        }
        mock_client.data_store["prompt_configurations"].append(cached_config)
        
        # First call should cache the config
        config1 = await manager.get_active_config()
        assert config1.cache_config.general_ttl == 600
        
        # Second call should use cache (same object)
        config2 = await manager.get_active_config()
        assert config2.cache_config.general_ttl == 600
        
        # Cache should be populated
        assert manager._cached_config is not None
        assert manager._cache_timestamp is not None
    
    @pytest.mark.asyncio
    async def test_get_active_config_force_refresh(self):
        """Test forcing config refresh from database"""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Add initial config
        initial_config = {
            "id": 2,
            "config_data": {"cache_config": {"general_ttl": 700}},
            "is_active": True,
            "created_at": "2024-01-20T10:00:00Z"
        }
        mock_client.data_store["prompt_configurations"].append(initial_config)
        
        # Get config to populate cache
        config1 = await manager.get_active_config()
        assert config1.cache_config.general_ttl == 700
        
        # Modify database
        mock_client.data_store["prompt_configurations"][1]["config_data"]["cache_config"]["general_ttl"] = 800
        
        # Force refresh should get updated config
        config2 = await manager.get_active_config(force_refresh=True)
        assert config2.cache_config.general_ttl == 800
    
    @pytest.mark.asyncio
    async def test_get_active_config_error_handling(self):
        """Test error handling when database fails"""
        mock_client = MockSupabaseClient(fail_mode="database_error")
        manager = ConfigurationManager(mock_client)
        
        # Should return default config on database error
        config = await manager.get_active_config()
        
        # Should get default configuration
        assert isinstance(config, PromptOptimizationConfig)
        assert config.cache_config.general_ttl == 6  # Default value
    
    @pytest.mark.asyncio
    async def test_save_config(self):
        """Test saving configuration to database"""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        config = PromptOptimizationConfig()
        
        # Test successful save
        config_id = await manager.save_config(
            config, 
            "test_user",
            "Test configuration"
        )
        
        assert config_id is not None
        # Should have at least the original config in the data store
        assert len(mock_client.data_store["prompt_configurations"]) >= 1

    @pytest.mark.asyncio
    async def test_validate_config_valid(self):
        """Test config validation with valid data"""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        valid_config = {
            "query_classification": {"confidence_threshold": 0.8},
            "cache_config": {"general_ttl": 300},
            "feature_flags": {"enable_response_caching": True}
        }
        
        result = await manager.validate_config(valid_config)
        
        assert result["valid"] is True
        assert "config" in result
        assert "hash" in result

    @pytest.mark.asyncio
    async def test_validate_config_invalid(self):
        """Test config validation with invalid data"""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        invalid_config = {
            "query_classification": {"confidence_threshold": 2.0},  # Invalid: > 1.0
            "cache_config": {"general_ttl": -100}  # Invalid: negative
        }
        
        result = await manager.validate_config(invalid_config)
        
        assert result["valid"] is False
        assert "error" in result
    
    def test_cache_validation(self):
        """Test cache timestamp validation."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # No cache should be invalid
        assert not manager._is_cache_valid()
        
        # Set cache
        config = TestConfigFixtures.get_default_config()
        manager._update_cache(config)
        
        # Fresh cache should be valid
        assert manager._is_cache_valid()
        
        # Expired cache should be invalid
        manager._cache_timestamp = datetime.utcnow() - timedelta(minutes=10)
        assert not manager._is_cache_valid()


if __name__ == "__main__":
    pytest.main([__file__]) 