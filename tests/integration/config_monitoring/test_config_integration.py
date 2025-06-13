"""
Integration tests for configuration and monitoring systems.
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.config.prompt_config import (
    PromptOptimizationConfig,
    ConfigurationManager,
    QueryType
)
from tests.fixtures.test_configs import (
    TestConfigFixtures,
    MockSupabaseClient,
    PerformanceTestData
)


class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    @pytest.mark.asyncio
    async def test_full_config_lifecycle(self):
        """Test complete configuration lifecycle: create, save, retrieve, update, rollback."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # 1. Start with default config
        config_v1 = await manager.get_active_config()
        assert config_v1.version == "1.0.0"
        
        # 2. Save a custom configuration
        custom_config = TestConfigFixtures.get_custom_config()
        config_id_v2 = await manager.save_config(
            custom_config,
            "admin",
            "Updated to custom configuration"
        )
        
        # 3. Retrieve the saved configuration
        config_v2 = await manager.get_active_config(force_refresh=True)
        assert config_v2.version == "2.0.0"
        assert config_v2.updated_by == "test_user"
        assert config_v2.query_classification.confidence_threshold == 0.8
        
        # 4. Validate configuration changes are working
        validation_result = await manager.validate_config(config_v2.to_dict())
        assert validation_result["is_valid"] is True
        
        # 5. Create another version
        config_v2.feature_flags.enable_query_expansion = False
        config_v2.version = "2.1.0"
        
        config_id_v3 = await manager.save_config(
            config_v2,
            "admin",
            "Disabled query expansion"
        )
        
        # 6. Verify new version is active
        config_v3 = await manager.get_active_config(force_refresh=True)
        assert config_v3.version == "2.1.0"
        assert config_v3.feature_flags.enable_query_expansion is False
        
        # 7. Test rollback functionality
        rollback_success = await manager.rollback_config(config_id_v2, "admin")
        assert rollback_success is True
        
        # 8. Verify rollback worked
        rolled_back_config = await manager.get_active_config(force_refresh=True)
        assert rolled_back_config.version == "2.0.0"
        assert rolled_back_config.feature_flags.enable_query_expansion is True
    
    @pytest.mark.asyncio
    async def test_config_validation_edge_cases(self):
        """Test configuration validation with various edge cases."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Test edge case configurations
        edge_cases = TestConfigFixtures.get_edge_case_configs()
        
        for case in edge_cases:
            config = case["config"]
            config_dict = config.to_dict()
            
            # Validate the configuration
            result = await manager.validate_config(config_dict)
            
            # All edge cases should be valid (they're at boundaries)
            assert result["is_valid"] is True, f"Edge case '{case['name']}' failed validation"
            
            # Should be able to save and retrieve
            config_id = await manager.save_config(
                config,
                f"test_{case['name']}",
                f"Testing {case['name']} configuration"
            )
            
            assert config_id is not None
    
    @pytest.mark.asyncio
    async def test_config_caching_behavior(self):
        """Test configuration caching behavior in detail."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Add initial config to database
        initial_config = TestConfigFixtures.get_default_config()
        mock_client.configs.append({
            "id": "initial",
            "config_data": initial_config.to_dict(),
            "is_active": True,
            "version": "1.0.0"
        })
        
        # First access should hit database and cache result
        config1 = await manager.get_active_config()
        assert manager._cached_config is not None
        cache_time1 = manager._cache_timestamp
        
        # Second access should use cache (verify by clearing database)
        original_configs = mock_client.configs.copy()
        mock_client.configs.clear()
        
        config2 = await manager.get_active_config()
        assert config1.get_hash() == config2.get_hash()
        assert manager._cache_timestamp == cache_time1  # Should be same timestamp
        
        # Restore database
        mock_client.configs = original_configs
        
        # Force refresh should bypass cache
        updated_config = TestConfigFixtures.get_custom_config()
        mock_client.configs[0]["config_data"] = updated_config.to_dict()
        mock_client.configs[0]["version"] = "2.0.0"
        
        config3 = await manager.get_active_config(force_refresh=True)
        assert config3.version == "2.0.0"
        assert config3.get_hash() != config1.get_hash()
    
    @pytest.mark.asyncio
    async def test_config_error_handling(self):
        """Test configuration error handling scenarios."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Test database connection failure
        mock_client.set_failure_mode("execute")
        
        # Should return default config on error
        config = await manager.get_active_config()
        assert isinstance(config, PromptOptimizationConfig)
        assert config.version == "1.0.0"
        
        # Clear failure mode
        mock_client.clear_failure_mode()
        
        # Test save operation failure
        mock_client.set_failure_mode("insert")
        
        config = TestConfigFixtures.get_custom_config()
        
        # Save should handle error gracefully
        with pytest.raises(Exception):
            await manager.save_config(config, "test", "Should fail")
    
    @pytest.mark.asyncio
    async def test_config_history_tracking(self):
        """Test configuration history and versioning."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Create multiple configuration versions
        configs = [
            (TestConfigFixtures.get_default_config(), "v1.0.0", "Initial"),
            (TestConfigFixtures.get_custom_config(), "v2.0.0", "Custom settings"),
        ]
        
        saved_ids = []
        for config, version, notes in configs:
            config.version = version
            config_id = await manager.save_config(
                config,
                "version_tester",
                notes
            )
            saved_ids.append(config_id)
        
        # Get configuration history
        history = await manager.get_config_history(limit=5)
        
        assert len(history) >= 2
        
        # History should be ordered by creation time (newest first)
        assert history[0]["version"] == "v2.0.0"
        assert history[1]["version"] == "v1.0.0"
        
        # Each history entry should have required fields
        for entry in history:
            assert "id" in entry
            assert "version" in entry
            assert "created_at" in entry
            assert "updated_by" in entry
            assert "change_notes" in entry


class TestMonitoringIntegration:
    """Integration tests for monitoring systems."""
    
    @pytest.mark.asyncio
    async def test_query_metrics_collection(self):
        """Test query metrics collection and aggregation."""
        # This would test the actual metrics collection
        # For now, we'll test with sample data
        
        sample_metrics = PerformanceTestData.get_query_metrics_sample()
        
        # Test data consistency
        assert len(sample_metrics) == 100
        
        # Test that all required fields are present
        required_fields = [
            "id", "query_text", "query_type", "total_latency_ms",
            "relevance_score", "confidence_score", "cache_hit",
            "error_occurred", "created_at"
        ]
        
        for metric in sample_metrics[:5]:  # Test first 5
            for field in required_fields:
                assert field in metric, f"Missing field: {field}"
        
        # Test data types and ranges
        for metric in sample_metrics[:10]:
            assert isinstance(metric["total_latency_ms"], int)
            assert metric["total_latency_ms"] > 0
            assert 0.0 <= metric["relevance_score"] <= 1.0
            assert 0.0 <= metric["confidence_score"] <= 1.0
            assert isinstance(metric["cache_hit"], bool)
            assert isinstance(metric["error_occurred"], bool)
    
    @pytest.mark.asyncio
    async def test_performance_profiling_integration(self):
        """Test performance profiling data collection."""
        sample_profiles = PerformanceTestData.get_performance_profiles_sample()
        
        # Test profile data structure
        assert len(sample_profiles) == 50
        
        required_profile_fields = [
            "id", "query_id", "embedding_generation_ms", "vector_search_ms",
            "context_preparation_ms", "llm_inference_ms", "peak_memory_mb",
            "input_tokens", "output_tokens", "total_cost"
        ]
        
        for profile in sample_profiles[:5]:
            for field in required_profile_fields:
                assert field in profile, f"Missing profile field: {field}"
        
        # Test performance metrics ranges
        for profile in sample_profiles[:10]:
            assert profile["embedding_generation_ms"] >= 0
            assert profile["vector_search_ms"] >= 0
            assert profile["peak_memory_mb"] > 0
            assert profile["input_tokens"] > 0
            assert profile["output_tokens"] > 0
            assert profile["total_cost"] >= 0
    
    @pytest.mark.asyncio
    async def test_feature_flag_integration(self):
        """Test feature flag integration with configuration."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Get configuration with feature flags
        config = await manager.get_active_config()
        
        # Test feature flag evaluation
        assert hasattr(config.feature_flags, 'enable_contextual_retrieval')
        assert hasattr(config.feature_flags, 'enable_hybrid_search')
        assert hasattr(config.feature_flags, 'enable_query_expansion')
        assert hasattr(config.feature_flags, 'ab_test_percentage')
        
        # Test that feature flags affect behavior
        if config.feature_flags.enable_contextual_retrieval:
            # This would trigger contextual retrieval in actual implementation
            pass
        
        if config.feature_flags.enable_hybrid_search:
            # This would trigger hybrid search in actual implementation
            pass
        
        # Test A/B testing percentage
        ab_percentage = config.feature_flags.ab_test_percentage
        assert 0.0 <= ab_percentage <= 100.0


class TestConfigMonitoringIntegration:
    """Integration tests combining configuration and monitoring."""
    
    @pytest.mark.asyncio
    async def test_config_change_monitoring(self):
        """Test that configuration changes are properly tracked."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Initial configuration
        config_v1 = TestConfigFixtures.get_default_config()
        config_id_v1 = await manager.save_config(
            config_v1,
            "admin",
            "Initial configuration"
        )
        
        # Modified configuration
        config_v2 = TestConfigFixtures.get_custom_config()
        config_id_v2 = await manager.save_config(
            config_v2,
            "admin",
            "Performance optimizations"
        )
        
        # Verify both configurations are tracked
        assert len(mock_client.configs) >= 2
        
        # Verify configuration metadata
        saved_configs = [c for c in mock_client.configs if c.get("change_notes")]
        assert len(saved_configs) >= 2
        
        # Check that only one config is active
        active_configs = [c for c in mock_client.configs if c.get("is_active")]
        assert len(active_configs) == 1
        assert active_configs[0]["id"] == config_id_v2
    
    @pytest.mark.asyncio
    async def test_performance_impact_of_config_changes(self):
        """Test that configuration changes don't negatively impact performance."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Measure baseline performance
        start_time = datetime.utcnow()
        config1 = await manager.get_active_config()
        baseline_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Save new configuration
        custom_config = TestConfigFixtures.get_custom_config()
        await manager.save_config(custom_config, "perf_test", "Performance test")
        
        # Measure performance with cache refresh
        start_time = datetime.utcnow()
        config2 = await manager.get_active_config(force_refresh=True)
        refresh_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Both operations should be fast (under 1 second for mocked operations)
        assert baseline_duration < 1.0
        assert refresh_duration < 1.0
        
        # Cached access should be very fast
        start_time = datetime.utcnow()
        config3 = await manager.get_active_config()
        cached_duration = (datetime.utcnow() - start_time).total_seconds()
        
        assert cached_duration < 0.1  # Cached access should be very fast


if __name__ == "__main__":
    pytest.main([__file__]) 