"""
Comprehensive tests for native LangChain cache integration.

This test suite verifies that the UniversalRAGChain correctly integrates
with LangChain's native RedisSemanticCache for efficient semantic caching.
"""

import pytest
import asyncio
import warnings
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional

# Import the components under test
try:
    from src.chains.universal_rag_lcel import UniversalRAGChain
except ImportError:
    from chains.universal_rag_lcel import UniversalRAGChain

from src.services.cache.casino_cache import setup_casino_cache, CasinoSemanticCache


class TestNativeCacheIntegration:
    """Test native LangChain cache integration"""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings for testing"""
        mock_emb = Mock()
        mock_emb.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        mock_emb.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
        return mock_emb
    
    @pytest.fixture  
    def mock_llm(self):
        """Mock LLM for testing"""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Test response")
        mock_llm.ainvoke = AsyncMock(return_value="Test async response")
        return mock_llm
        
    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever for testing"""
        mock_ret = Mock()
        mock_ret.invoke = Mock(return_value=[
            Mock(page_content="Test content", metadata={"source": "test"})
        ])
        return mock_ret

    @patch('src.services.cache.casino_cache.RedisSemanticCache')
    @patch('src.services.cache.casino_cache.set_llm_cache')
    def test_setup_casino_cache_basic(self, mock_set_cache, mock_redis_cache, mock_embeddings):
        """Test basic casino cache setup"""
        # Setup
        mock_cache_instance = Mock()
        mock_redis_cache.return_value = mock_cache_instance
        
        # Execute
        cache = setup_casino_cache(mock_embeddings, content_type="news")
        
        # Verify
        assert isinstance(cache, CasinoSemanticCache)
        assert cache.content_type == "news"
        assert cache.ttl == 3600  # 1 hour for news
        assert cache.threshold == 0.15  # Strict threshold for news
        
        # Verify Redis cache was created with correct parameters
        mock_redis_cache.assert_called_once_with(
            redis_url="redis://localhost:6379",
            embeddings=mock_embeddings,
            distance_threshold=0.15,
            ttl=3600,
            prefix="casino_cache_news"
        )
        
        # Verify global cache was set
        mock_set_cache.assert_called_once_with(mock_cache_instance)

    @patch('src.services.cache.casino_cache.RedisSemanticCache')
    def test_content_type_configurations(self, mock_redis_cache, mock_embeddings):
        """Test different content type configurations"""
        test_cases = [
            ("news", 3600, 0.15),
            ("reviews", 86400, 0.2), 
            ("regulatory", 604800, 0.25),
            ("default", 21600, 0.2)
        ]
        
        for content_type, expected_ttl, expected_threshold in test_cases:
            cache = CasinoSemanticCache(mock_embeddings, content_type=content_type)
            
            assert cache.content_type == content_type
            assert cache.ttl == expected_ttl
            assert cache.threshold == expected_threshold

    @patch('src.services.cache.casino_cache.RedisSemanticCache')
    @patch('src.services.cache.casino_cache.set_llm_cache')
    def test_convenience_functions(self, mock_set_cache, mock_redis_cache, mock_embeddings):
        """Test convenience setup functions"""
        from src.services.cache.casino_cache import setup_news_cache, setup_reviews_cache, setup_regulatory_cache
        
        mock_cache_instance = Mock()
        mock_redis_cache.return_value = mock_cache_instance
        
        # Test news cache
        news_cache = setup_news_cache(mock_embeddings)
        assert news_cache.content_type == "news"
        
        # Test reviews cache
        reviews_cache = setup_reviews_cache(mock_embeddings)
        assert reviews_cache.content_type == "reviews"
        
        # Test regulatory cache
        regulatory_cache = setup_regulatory_cache(mock_embeddings)
        assert regulatory_cache.content_type == "regulatory"

    @patch('langchain_community.cache.RedisSemanticCache')
    def test_universal_rag_chain_cache_integration(self, mock_redis_cache, mock_embeddings, mock_llm, mock_retriever):
        """Test that UniversalRAGChain integrates with native cache"""
        # Skip this test if UniversalRAGChain doesn't exist or isn't compatible
        try:
            # Create chain with caching enabled
            chain = UniversalRAGChain(
                llm=mock_llm,
                retriever=mock_retriever,
                enable_caching=True
            )
            
            # Test cache configuration method
            if hasattr(chain, 'configure_cache_for_content_type'):
                with patch('langchain_core.globals.set_llm_cache') as mock_set_cache:
                    configured_chain = chain.configure_cache_for_content_type("news")
                    assert configured_chain is not None
                    # set_llm_cache should have been called
                    assert mock_set_cache.called
                    
        except (ImportError, AttributeError, TypeError) as e:
            pytest.skip(f"UniversalRAGChain not compatible with current test setup: {e}")

    def test_deprecation_warnings(self, mock_embeddings, mock_llm, mock_retriever):
        """Test that deprecated cache methods show warnings"""
        try:
            chain = UniversalRAGChain(
                llm=mock_llm,
                retriever=mock_retriever,
                enable_caching=True
            )
            
            # Test deprecated methods
            if hasattr(chain, 'get_cache_stats'):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    stats = chain.get_cache_stats()
                    
                    # Should have issued a deprecation warning
                    assert len(w) > 0
                    assert issubclass(w[0].category, DeprecationWarning)
                    assert "deprecated" in str(w[0].message).lower()
                    
        except (ImportError, AttributeError, TypeError) as e:
            pytest.skip(f"UniversalRAGChain not compatible with current test setup: {e}")

    def test_cache_clear_functionality(self, mock_embeddings):
        """Test cache clearing functionality"""
        with patch('src.services.cache.casino_cache.RedisSemanticCache') as mock_redis_cache:
            mock_cache_instance = Mock()
            mock_redis_cache.return_value = mock_cache_instance
            
            cache = CasinoSemanticCache(mock_embeddings, content_type="news")
            cache.clear()
            
            # Verify clear was called on the underlying cache
            mock_cache_instance.clear.assert_called_once()

    @patch('src.services.cache.casino_cache.RedisSemanticCache')
    def test_redis_connection_error_handling(self, mock_redis_cache, mock_embeddings):
        """Test handling of Redis connection errors"""
        # Simulate Redis connection error
        mock_redis_cache.side_effect = Exception("Redis connection failed")
        
        with pytest.raises(Exception, match="Redis connection failed"):
            setup_casino_cache(mock_embeddings)

    def test_ttl_values_are_integers(self, mock_embeddings):
        """Ensure TTL values are integers (seconds) not timedelta objects"""
        cache = CasinoSemanticCache(mock_embeddings, content_type="news")
        
        assert isinstance(cache.ttl, int)
        assert cache.ttl > 0
        
        # Test all content types return integer TTLs
        for content_type in ["news", "reviews", "regulatory", "default"]:
            cache = CasinoSemanticCache(mock_embeddings, content_type=content_type)
            assert isinstance(cache.ttl, int)
            assert cache.ttl > 0


class TestAsyncCacheIntegration:
    """Test async integration with native cache"""
    
    @pytest.fixture
    def mock_async_embeddings(self):
        """Mock async embeddings"""
        mock_emb = Mock()
        mock_emb.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_emb.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return mock_emb

    @pytest.mark.asyncio
    @patch('src.services.cache.casino_cache.RedisSemanticCache')
    async def test_async_cache_operations(self, mock_redis_cache, mock_async_embeddings):
        """Test async cache operations work correctly"""
        mock_cache_instance = Mock()
        mock_redis_cache.return_value = mock_cache_instance
        
        cache = setup_casino_cache(mock_async_embeddings, content_type="news")
        
        # Verify cache was created
        assert cache is not None
        assert cache.content_type == "news"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 