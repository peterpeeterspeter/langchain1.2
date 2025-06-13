"""
Integration tests for IntegratedRAGChain
Tests the integration of monitoring, configuration, and feature flags with the base RAG chain.
"""

import asyncio
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import uuid

# Import the integrated chain
from src.chains.integrated_rag_chain import (
    IntegratedRAGChain, 
    create_integrated_rag_chain
)
from src.utils.integration_helpers import (
    quick_setup_integrated_rag,
    IntegrationSetup,
    IntegrationHealthChecker,
    ConfigurationValidator
)

class TestIntegratedRAGChain:
    """Test suite for IntegratedRAGChain integration."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = {
            'data': [{'id': 'test-id'}]
        }
        return mock_client
    
    @pytest.fixture
    def integration_config(self):
        """Sample integration configuration."""
        return {
            'supabase_url': 'https://test.supabase.co',
            'supabase_key': 'test-key',
            'enable_monitoring': True,
            'enable_profiling': False,
            'enable_feature_flags': True,
            'cache_ttl_hours': 24
        }
    
    def test_integration_chain_initialization(self, mock_supabase_client):
        """Test that IntegratedRAGChain initializes properly with all managers."""
        
        chain = IntegratedRAGChain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client,
            enable_monitoring=True,
            enable_profiling=False,
            enable_feature_flags=True,
            enable_configuration=True
        )
        
        # Verify initialization
        assert chain.enable_monitoring == True
        assert chain.enable_profiling == False
        assert chain.enable_feature_flags == True
        assert chain.enable_configuration == True
        assert chain.supabase_client == mock_supabase_client
        
        # Verify managers are initialized or None based on settings
        assert chain.analytics is not None  # Should be initialized with monitoring enabled
        assert chain.profiler is None  # Should be None with profiling disabled
        assert chain.feature_flags is not None  # Should be initialized
        assert chain.config_manager is not None  # Should be initialized
        assert chain.logger is not None  # Should always be initialized
        assert chain.pipeline_logger is not None  # Should always be initialized
    
    def test_integration_chain_without_supabase(self):
        """Test IntegratedRAGChain works without Supabase client (graceful degradation)."""
        
        chain = IntegratedRAGChain(
            model_name="gpt-4",
            supabase_client=None,
            enable_monitoring=True,
            enable_profiling=True,
            enable_feature_flags=True,
            enable_configuration=True
        )
        
        # Verify graceful degradation
        assert chain.analytics is None
        assert chain.profiler is None
        assert chain.feature_flags is None
        assert chain.config_manager is None
        assert chain.logger is not None  # Should still work without Supabase
        assert chain.pipeline_logger is not None
    
    @pytest.mark.asyncio
    async def test_ainvoke_with_monitoring(self, mock_supabase_client):
        """Test that ainvoke works with monitoring enabled."""
        
        # Mock the parent class methods
        with patch.object(IntegratedRAGChain.__bases__[0], 'ainvoke') as mock_parent_ainvoke:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.answer = "Test answer"
            mock_response.confidence_score = 0.85
            mock_response.cached = False
            mock_response.sources = [{'title': 'test source', 'similarity_score': 0.9}]
            mock_response.metadata = {'test': 'metadata'}
            mock_response.token_usage = {'total_tokens': 100}
            mock_response.query_analysis = {'query_type': 'general'}
            mock_parent_ainvoke.return_value = mock_response
            
            # Create chain with mocked managers
            chain = IntegratedRAGChain(
                model_name="gpt-4",
                supabase_client=mock_supabase_client,
                enable_monitoring=True,
                enable_profiling=False,
                enable_feature_flags=True
            )
            
            # Mock the analytics and feature flags
            chain.analytics = AsyncMock()
            chain.feature_flags = AsyncMock()
            chain.feature_flags.is_feature_enabled.return_value = True
            chain.feature_flags.get_variant.return_value = None
            
            # Test ainvoke
            query = "What is the best online casino?"
            user_context = {"user_id": "test_user"}
            
            response = await chain.ainvoke(query, user_context=user_context)
            
            # Verify response contains monitoring metadata
            assert 'query_id' in response.metadata
            assert 'monitoring_enabled' in response.metadata
            assert 'total_pipeline_time_ms' in response.metadata
            assert response.metadata['monitoring_enabled'] == True
            
            # Verify parent was called
            mock_parent_ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_feature_flag_integration(self, mock_supabase_client):
        """Test feature flag integration in the chain."""
        
        chain = IntegratedRAGChain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client,
            enable_feature_flags=True
        )
        
        # Mock feature flags manager
        chain.feature_flags = AsyncMock()
        chain.feature_flags.is_feature_enabled.return_value = True
        
        # Test check_feature_flag method
        user_context = {"user_id": "test_user"}
        result = await chain.check_feature_flag("enable_hybrid_search", user_context)
        
        assert result == True
        chain.feature_flags.is_feature_enabled.assert_called_once_with(
            "enable_hybrid_search", user_context
        )
    
    @pytest.mark.asyncio
    async def test_configuration_reload(self, mock_supabase_client):
        """Test configuration reload functionality."""
        
        chain = IntegratedRAGChain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client,
            enable_configuration=True
        )
        
        # Mock config manager
        chain.config_manager = AsyncMock()
        mock_config = MagicMock()
        mock_config.version = "v1.0"
        chain.config_manager.get_active_config.return_value = mock_config
        
        # Test reload
        result = await chain.reload_configuration()
        
        assert result == True
        chain.config_manager.get_active_config.assert_called_once_with(force_refresh=True)
    
    def test_monitoring_stats(self, mock_supabase_client):
        """Test monitoring statistics retrieval."""
        
        chain = IntegratedRAGChain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client,
            enable_monitoring=True
        )
        
        # Mock analytics
        chain.analytics = AsyncMock()
        
        # Test get_monitoring_stats
        stats = chain.get_monitoring_stats()
        
        assert 'monitoring_enabled' in stats
        assert 'profiling_enabled' in stats
        assert 'feature_flags_enabled' in stats
        assert stats['monitoring_enabled'] == True

    def test_factory_function(self, mock_supabase_client):
        """Test the factory function for creating integrated chains."""
        
        # Test with all features enabled
        chain = create_integrated_rag_chain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client,
            enable_all_features=True
        )
        
        assert isinstance(chain, IntegratedRAGChain)
        assert chain.enable_monitoring == True
        assert chain.enable_profiling == True
        assert chain.enable_feature_flags == True
        assert chain.enable_configuration == True
        
        # Test with minimal features
        chain_minimal = create_integrated_rag_chain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client,
            enable_all_features=False,
            enable_monitoring=False,
            enable_profiling=False
        )
        
        assert chain_minimal.enable_monitoring == False
        assert chain_minimal.enable_profiling == False


class TestIntegrationHelpers:
    """Test suite for integration helper functions."""
    
    def test_configuration_validator(self):
        """Test configuration validation."""
        
        # Valid configuration
        valid_config = {
            'supabase_url': 'https://test.supabase.co',
            'supabase_key': 'test-key',
            'enable_monitoring': True,
            'enable_profiling': False,
            'enable_feature_flags': True
        }
        
        is_valid, issues = ConfigurationValidator.validate_integration_config(valid_config)
        assert is_valid == True
        assert len(issues) == 0
        
        # Invalid configuration
        invalid_config = {
            'supabase_url': 'invalid-url',  # Invalid URL
            'enable_monitoring': 'yes',  # Should be boolean
            # Missing required fields
        }
        
        is_valid, issues = ConfigurationValidator.validate_integration_config(invalid_config)
        assert is_valid == False
        assert len(issues) > 0
        assert any('Missing required field' in issue for issue in issues)
        assert any('must be a boolean' in issue for issue in issues)
    
    def test_runtime_environment_validation(self):
        """Test runtime environment validation."""
        
        # Mock missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            is_valid, issues = ConfigurationValidator.validate_runtime_environment()
            assert is_valid == False
            assert any('Missing environment variable' in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_health_checker(self):
        """Test health checker functionality."""
        
        # Mock managers
        managers = {
            'supabase_client': MagicMock(),
            'config_manager': AsyncMock(),
            'analytics': AsyncMock(),
            'feature_flags': AsyncMock()
        }
        
        # Setup successful responses
        managers['supabase_client'].table.return_value.select.return_value.limit.return_value.execute.return_value = {
            'data': [{'id': 'test'}]
        }
        
        mock_config = MagicMock()
        mock_config.version = "v1.0"
        managers['config_manager'].get_active_config.return_value = mock_config
        
        managers['analytics'].get_real_time_metrics.return_value = {'test': 'metrics'}
        managers['feature_flags'].is_feature_enabled.return_value = False
        
        # Test health check
        health_status = await IntegrationHealthChecker.check_all_systems(managers)
        
        assert health_status['overall_health'] == 'healthy'
        assert 'supabase' in health_status['systems']
        assert 'configuration' in health_status['systems']
        assert 'analytics' in health_status['systems']
        assert 'feature_flags' in health_status['systems']
        
        assert health_status['systems']['supabase']['status'] == 'healthy'
        assert health_status['systems']['configuration']['status'] == 'healthy'
        assert health_status['systems']['analytics']['status'] == 'healthy'
        assert health_status['systems']['feature_flags']['status'] == 'healthy'
    
    @pytest.mark.asyncio 
    async def test_integration_setup(self):
        """Test integration setup utilities."""
        
        # Test default configuration creation
        config = await IntegrationSetup.create_default_configuration()
        
        assert config.query_classification.confidence_threshold == 0.75
        assert config.context_formatting.max_context_length == 4000
        assert config.cache_config.casino_review_ttl == 48
        assert config.cache_config.news_ttl == 2
        assert config.performance.response_time_warning_ms == 2000
        assert config.feature_flags.enable_contextual_retrieval == True
        assert config.feature_flags.enable_hybrid_search == False


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_monitored_rag_chain_alias(self, mock_supabase_client):
        """Test that MonitoredUniversalRAGChain alias works."""
        
        from src.chains.integrated_rag_chain import MonitoredUniversalRAGChain
        
        chain = MonitoredUniversalRAGChain(
            model_name="gpt-4",
            supabase_client=mock_supabase_client
        )
        
        assert isinstance(chain, IntegratedRAGChain)
    
    def test_import_structure(self):
        """Test that all expected imports work."""
        
        # Test importing from chains package
        from src.chains import (
            IntegratedRAGChain,
            create_integrated_rag_chain,
            MonitoredUniversalRAGChain
        )
        
        assert IntegratedRAGChain is not None
        assert create_integrated_rag_chain is not None
        assert MonitoredUniversalRAGChain is not None
        
        # Test importing integration helpers
        from src.utils.integration_helpers import (
            quick_setup_integrated_rag,
            IntegrationSetup,
            IntegrationHealthChecker
        )
        
        assert quick_setup_integrated_rag is not None
        assert IntegrationSetup is not None
        assert IntegrationHealthChecker is not None


# Example usage test (not run in pytest)
async def example_integration_usage():
    """Example of how to use the integrated system."""
    
    # Quick setup for testing
    try:
        managers = await quick_setup_integrated_rag(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key"
        )
        
        # Create integrated chain
        chain = create_integrated_rag_chain(
            model_name="gpt-4",
            supabase_client=managers['supabase_client'],
            enable_all_features=True
        )
        
        # Example query with monitoring
        response = await chain.ainvoke(
            "What are the best online casinos?",
            user_context={"user_id": "test_user_123"}
        )
        
        print(f"Response: {response.answer[:100]}...")
        print(f"Query ID: {response.metadata.get('query_id')}")
        print(f"Monitoring enabled: {response.metadata.get('monitoring_enabled')}")
        
        # Check feature flag
        hybrid_enabled = await chain.check_feature_flag("enable_hybrid_search")
        print(f"Hybrid search enabled: {hybrid_enabled}")
        
        # Get monitoring stats
        stats = chain.get_monitoring_stats()
        print(f"Monitoring stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run example usage (not in pytest)
    result = asyncio.run(example_integration_usage())
    print(f"Integration test result: {result}") 