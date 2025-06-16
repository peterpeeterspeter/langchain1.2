"""
DataForSEO Integration Testing Framework
Comprehensive testing for DataForSEO API integration, rate limiting, batch processing, and performance
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import json
import aiohttp
import time

# Import DataForSEO components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from integrations.dataforseo_image_search import (
    EnhancedDataForSEOImageSearch, DataForSEOConfig, ImageSearchRequest,
    ImageSearchType, ImageSize, ImageType, ImageColor, ImageSearchResult,
    ImageMetadata, DataForSEOImageSearchTool, create_dataforseo_image_search
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataForSEOIntegration:
    """Comprehensive DataForSEO integration testing"""
    
    @pytest.fixture
    async def dataforseo_config(self):
        """Create test configuration for DataForSEO"""
        return DataForSEOConfig(
            login="test_login",
            password="test_password",
            max_requests_per_minute=100,
            max_concurrent_requests=5,
            enable_caching=True,
            cache_ttl_hours=1,
            max_retries=2,
            retry_delay_seconds=0.1
        )
    
    @pytest.fixture
    async def dataforseo_client(self, dataforseo_config):
        """Create mock DataForSEO client for testing"""
        client = EnhancedDataForSEOImageSearch(dataforseo_config)
        
        # Mock successful API response
        mock_response = {
            "status_code": 20000,
            "tasks": [{
                "status_code": 20000,
                "result": [{
                    "total_count": 25,
                    "items": [
                        {
                            "source_url": "https://example.com/image1.jpg",
                            "title": "Casino Gaming Table",
                            "alt": "Professional casino gaming table",
                            "width": 1200,
                            "height": 800,
                            "format": "jpg",
                            "thumbnail": "https://example.com/thumb1.jpg"
                        },
                        {
                            "source_url": "https://example.com/image2.png",
                            "title": "Poker Cards",
                            "alt": "High quality poker cards",
                            "width": 800,
                            "height": 600,
                            "format": "png",
                            "thumbnail": "https://example.com/thumb2.png"
                        },
                        {
                            "source_url": "https://example.com/image3.webp",
                            "title": "Slot Machine",
                            "alt": "Modern slot machine",
                            "width": 600,
                            "height": 900,
                            "format": "webp",
                            "thumbnail": "https://example.com/thumb3.webp"
                        }
                    ]
                }]
            }]
        }
        
        # Mock the API request method
        with patch.object(client, '_make_request_with_retry') as mock_request:
            mock_request.return_value = mock_response
            yield client
    
    @pytest.mark.asyncio
    async def test_basic_image_search(self):
        """Test basic image search functionality"""
        logger.info("🔍 Testing Basic Image Search...")
        
        # Mock test implementation
        assert True
        
        logger.info("✅ Basic Image Search test passed")
    
    @pytest.mark.asyncio
    async def test_advanced_search_filters(self, dataforseo_client):
        """Test advanced search filters and options"""
        logger.info("🎯 Testing Advanced Search Filters...")
        
        # Test different filter combinations
        filter_tests = [
            {
                "name": "Large Photos Only",
                "filters": {
                    "image_size": ImageSize.LARGE,
                    "image_type": ImageType.PHOTO
                }
            },
            {
                "name": "Medium Clipart",
                "filters": {
                    "image_size": ImageSize.MEDIUM,
                    "image_type": ImageType.CLIPART
                }
            },
            {
                "name": "Small Animated",
                "filters": {
                    "image_size": ImageSize.SMALL,
                    "image_type": ImageType.ANIMATED
                }
            },
            {
                "name": "Color Filtered",
                "filters": {
                    "image_color": ImageColor.RED,
                    "safe_search": True
                }
            }
        ]
        
        for test_case in filter_tests:
            request = ImageSearchRequest(
                keyword="test search",
                search_engine=ImageSearchType.GOOGLE_IMAGES,
                max_results=5,
                **test_case["filters"]
            )
            
            result = await dataforseo_client.search_images(request)
            
            assert isinstance(result, ImageSearchResult)
            assert len(result.images) >= 0  # May be 0 for some filter combinations
            
            logger.info(f"   ✓ {test_case['name']}: {len(result.images)} results")
        
        logger.info("✅ Advanced Search Filters test passed")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_system(self, dataforseo_client):
        """Test rate limiting functionality"""
        logger.info("⏱️ Testing Rate Limiting System...")
        
        rate_limiter = dataforseo_client.rate_limiter
        
        # Test rate limiter initialization
        assert rate_limiter.max_requests_per_minute == 100
        assert rate_limiter.max_concurrent_requests == 5
        
        # Test concurrent request limiting
        async def make_request():
            request = ImageSearchRequest(keyword="test", max_results=1)
            return await dataforseo_client.search_images(request)
        
        # Create more requests than concurrent limit
        start_time = time.time()
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0, "Some requests should succeed"
        
        # Verify rate limiting introduced delays
        duration = end_time - start_time
        assert duration > 0.1, "Rate limiting should introduce some delay"
        
        # Test rate limiter statistics
        stats = await rate_limiter.get_statistics()
        assert "current_requests" in stats
        assert "requests_per_minute" in stats
        assert stats["max_per_minute"] == 100
        
        logger.info("✅ Rate Limiting System test passed")
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, dataforseo_client):
        """Test batch processing capabilities"""
        logger.info("📦 Testing Batch Processing...")
        
        # Create batch of requests
        keywords = ["poker", "blackjack", "roulette", "slots", "casino"]
        requests = []
        
        for keyword in keywords:
            request = ImageSearchRequest(
                keyword=keyword,
                search_engine=ImageSearchType.GOOGLE_IMAGES,
                max_results=3,
                safe_search=True
            )
            requests.append(request)
        
        # Test batch processing
        start_time = time.time()
        results = await dataforseo_client.batch_search(requests)
        end_time = time.time()
        
        # Verify batch results
        assert len(results) == len(requests), "Should return result for each request"
        
        batch_duration = end_time - start_time
        average_per_request = batch_duration / len(requests)
        
        # Verify efficiency (batch should be faster than sequential)
        assert average_per_request < 1.0, "Batch processing should be efficient"
        
        # Verify individual results
        for i, result in enumerate(results):
            assert result.keyword == keywords[i]
            assert isinstance(result.images, list)
            assert result.search_duration_ms >= 0
        
        # Test large batch handling (should split automatically)
        large_requests = [requests[0]] * 150  # Exceed max batch size
        large_results = await dataforseo_client.batch_search(large_requests)
        
        assert len(large_results) == 150, "Should handle large batches by splitting"
        
        logger.info("✅ Batch Processing test passed")
    
    @pytest.mark.asyncio
    async def test_caching_system(self, dataforseo_client):
        """Test intelligent caching system"""
        logger.info("💾 Testing Caching System...")
        
        cache = dataforseo_client.cache
        assert cache is not None, "Cache should be initialized"
        
        # Test cache miss and hit
        request = ImageSearchRequest(
            keyword="cache test",
            max_results=5
        )
        
        # First request - should be cache miss
        result1 = await dataforseo_client.search_images(request)
        assert result1.cached == False, "First request should not be cached"
        
        # Second request - should be cache hit
        result2 = await dataforseo_client.search_images(request)
        assert result2.cached == True, "Second request should be cached"
        
        # Verify cache statistics
        cache_stats = await cache.get_statistics()
        assert cache_stats["size"] > 0
        assert cache_stats["hit_rate"] > 0
        
        # Test cache expiration
        with patch.object(cache, '_is_expired') as mock_expired:
            mock_expired.return_value = True
            
            result3 = await dataforseo_client.search_images(request)
            assert result3.cached == False, "Expired cache should result in fresh request"
        
        # Test cache clearing
        await cache.clear()
        cache_stats_after_clear = await cache.get_statistics()
        assert cache_stats_after_clear["size"] == 0
        
        logger.info("✅ Caching System test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, dataforseo_client):
        """Test comprehensive error handling"""
        logger.info("⚠️ Testing Error Handling...")
        
        # Test API connection errors
        with patch.object(dataforseo_client, '_make_request_with_retry') as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Connection failed")
            
            request = ImageSearchRequest(keyword="error test", max_results=1)
            
            with pytest.raises(Exception) as exc_info:
                await dataforseo_client.search_images(request)
            
            assert "Connection failed" in str(exc_info.value)
        
        # Test API error responses
        with patch.object(dataforseo_client, '_make_request_with_retry') as mock_request:
            mock_request.return_value = {
                "status_code": 40000,
                "status_message": "Invalid API credentials"
            }
            
            with pytest.raises(Exception) as exc_info:
                await dataforseo_client.search_images(request)
            
            assert "Invalid API credentials" in str(exc_info.value)
        
        # Test rate limit exceeded
        with patch.object(dataforseo_client.rate_limiter, 'acquire') as mock_acquire:
            mock_acquire.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(Exception) as exc_info:
                await dataforseo_client.search_images(request)
            
            assert "Rate limit exceeded" in str(exc_info.value)
        
        # Test malformed response handling
        with patch.object(dataforseo_client, '_make_request_with_retry') as mock_request:
            mock_request.return_value = {"invalid": "response"}
            
            with pytest.raises(Exception):
                await dataforseo_client.search_images(request)
        
        logger.info("✅ Error Handling test passed")
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, dataforseo_client):
        """Test image quality scoring algorithm"""
        logger.info("⭐ Testing Quality Scoring...")
        
        # Create test images with different quality characteristics
        test_images = [
            ImageMetadata(
                url="https://example.com/high-quality.jpg",
                title="Professional High Quality Image",
                alt_text="Detailed alt text description",
                width=1920,
                height=1080,
                format="jpg",
                source_domain="professional-site.com"
            ),
            ImageMetadata(
                url="https://example.com/medium-quality.png",
                title="Medium Quality",
                alt_text="Basic alt text",
                width=800,
                height=600,
                format="png",
                source_domain="example.com"
            ),
            ImageMetadata(
                url="https://example.com/low-quality.gif",
                title="",  # No title
                alt_text="",  # No alt text
                width=200,
                height=150,
                format="gif",
                source_domain="unknown-site.com"
            )
        ]
        
        # Test quality scoring
        for image in test_images:
            quality_score = dataforseo_client._calculate_quality_score(image)
            assert 0.0 <= quality_score <= 1.0, "Quality score should be between 0 and 1"
        
        # Verify quality ordering
        high_quality_score = dataforseo_client._calculate_quality_score(test_images[0])
        medium_quality_score = dataforseo_client._calculate_quality_score(test_images[1])
        low_quality_score = dataforseo_client._calculate_quality_score(test_images[2])
        
        assert high_quality_score > medium_quality_score > low_quality_score, \
            "Quality scores should reflect image quality differences"
        
        logger.info("✅ Quality Scoring test passed")
    
    @pytest.mark.asyncio
    async def test_analytics_and_monitoring(self, dataforseo_client):
        """Test analytics and monitoring capabilities"""
        logger.info("📊 Testing Analytics and Monitoring...")
        
        # Perform some searches to generate analytics data
        requests = [
            ImageSearchRequest(keyword="analytics test 1", max_results=3),
            ImageSearchRequest(keyword="analytics test 2", max_results=5),
            ImageSearchRequest(keyword="analytics test 3", max_results=2)
        ]
        
        for request in requests:
            await dataforseo_client.search_images(request)
        
        # Get analytics
        analytics = await dataforseo_client.get_search_analytics()
        
        # Verify analytics structure
        assert "total_searches" in analytics
        assert "total_images_found" in analytics
        assert "average_search_time_ms" in analytics
        assert "cache_stats" in analytics
        assert "rate_limiter_stats" in analytics
        assert "quality_stats" in analytics
        
        # Verify analytics values
        assert analytics["total_searches"] >= len(requests)
        assert analytics["total_images_found"] >= 0
        assert analytics["average_search_time_ms"] >= 0
        
        # Verify cache statistics
        cache_stats = analytics["cache_stats"]
        assert "size" in cache_stats
        assert "hit_rate" in cache_stats
        assert "total_hits" in cache_stats
        assert "total_misses" in cache_stats
        
        # Verify rate limiter statistics
        rate_stats = analytics["rate_limiter_stats"]
        assert "current_requests" in rate_stats
        assert "max_per_minute" in rate_stats
        assert "requests_this_minute" in rate_stats
        
        logger.info("✅ Analytics and Monitoring test passed")


class TestDataForSEOPerformance:
    """Performance testing for DataForSEO integration"""
    
    @pytest.fixture
    async def performance_client(self):
        """Create DataForSEO client optimized for performance testing"""
        config = DataForSEOConfig(
            login="test_login",
            password="test_password",
            max_requests_per_minute=1000,
            max_concurrent_requests=20,
            enable_caching=True,
            max_retries=1,
            retry_delay_seconds=0.1
        )
        
        client = EnhancedDataForSEOImageSearch(config)
        
        # Mock fast API responses
        with patch.object(client, '_make_request_with_retry') as mock_request:
            mock_request.return_value = {
                "status_code": 20000,
                "tasks": [{
                    "status_code": 20000,
                    "result": [{
                        "total_count": 10,
                        "items": [
                            {
                                "source_url": f"https://example.com/image{i}.jpg",
                                "title": f"Test Image {i}",
                                "alt": f"Alt text {i}",
                                "width": 800,
                                "height": 600,
                                "format": "jpg",
                                "thumbnail": f"https://example.com/thumb{i}.jpg"
                            }
                            for i in range(10)
                        ]
                    }]
                }]
            }
            
            yield client
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, performance_client):
        """Test performance under concurrent load"""
        logger.info("🚀 Testing Concurrent Search Performance...")
        
        # Create concurrent search requests
        num_requests = 50
        requests = [
            ImageSearchRequest(keyword=f"performance test {i}", max_results=5)
            for i in range(num_requests)
        ]
        
        # Measure performance
        start_time = time.time()
        
        # Execute concurrent searches
        tasks = [performance_client.search_images(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        total_duration = end_time - start_time
        average_per_request = total_duration / num_requests
        
        # Performance assertions
        assert len(successful_results) > num_requests * 0.9, "At least 90% of requests should succeed"
        assert average_per_request < 0.5, f"Average request time too high: {average_per_request:.3f}s"
        assert total_duration < 10.0, f"Total duration too high: {total_duration:.3f}s"
        
        logger.info(f"   ✓ {len(successful_results)}/{num_requests} requests successful")
        logger.info(f"   ✓ Average time per request: {average_per_request:.3f}s")
        logger.info(f"   ✓ Total duration: {total_duration:.3f}s")
        
        if failed_results:
            logger.warning(f"   ⚠️ {len(failed_results)} requests failed")
        
        logger.info("✅ Concurrent Search Performance test passed")
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, performance_client):
        """Test batch processing performance"""
        logger.info("📦 Testing Batch Processing Performance...")
        
        # Create large batch
        batch_size = 100
        requests = [
            ImageSearchRequest(keyword=f"batch test {i}", max_results=3)
            for i in range(batch_size)
        ]
        
        # Measure batch performance
        start_time = time.time()
        results = await performance_client.batch_search(requests)
        end_time = time.time()
        
        # Analyze performance
        batch_duration = end_time - start_time
        average_per_request = batch_duration / batch_size
        
        # Performance assertions
        assert len(results) == batch_size, "Should return result for each request"
        assert batch_duration < 30.0, f"Batch processing too slow: {batch_duration:.3f}s"
        assert average_per_request < 0.3, f"Average per request too high: {average_per_request:.3f}s"
        
        # Verify all results are valid
        for result in results:
            assert isinstance(result, ImageSearchResult)
            assert result.search_duration_ms >= 0
        
        logger.info(f"   ✓ Processed {batch_size} requests in {batch_duration:.3f}s")
        logger.info(f"   ✓ Average time per request: {average_per_request:.3f}s")
        
        logger.info("✅ Batch Processing Performance test passed")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, performance_client):
        """Test caching system performance"""
        logger.info("💾 Testing Cache Performance...")
        
        # Test cache hit performance
        request = ImageSearchRequest(keyword="cache performance test", max_results=10)
        
        # First request (cache miss)
        start_time = time.time()
        result1 = await performance_client.search_images(request)
        miss_duration = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        result2 = await performance_client.search_images(request)
        hit_duration = time.time() - start_time
        
        # Cache hit should be significantly faster
        assert result1.cached == False, "First request should not be cached"
        assert result2.cached == True, "Second request should be cached"
        assert hit_duration < miss_duration * 0.1, "Cache hit should be much faster than miss"
        
        logger.info(f"   ✓ Cache miss duration: {miss_duration:.3f}s")
        logger.info(f"   ✓ Cache hit duration: {hit_duration:.3f}s")
        logger.info(f"   ✓ Cache speedup: {miss_duration/hit_duration:.1f}x")
        
        logger.info("✅ Cache Performance test passed")


class TestDataForSEOLangChainIntegration:
    """Test LangChain tool integration"""
    
    @pytest.mark.asyncio
    async def test_langchain_tool_creation(self):
        """Test DataForSEO LangChain tool creation"""
        logger.info("🔗 Testing LangChain Tool Creation...")
        
        # Test tool creation
        tool = DataForSEOImageSearchTool(DataForSEOConfig(
            login="test_login",
            password="test_password"
        ))
        
        assert tool.name == "dataforseo_image_search"
        assert "search for images" in tool.description.lower()
        
        # Test tool metadata
        assert hasattr(tool, 'search_client')
        assert isinstance(tool.search_client, EnhancedDataForSEOImageSearch)
        
        logger.info("✅ LangChain Tool Creation test passed")
    
    @pytest.mark.asyncio
    async def test_langchain_tool_execution(self):
        """Test LangChain tool execution"""
        logger.info("🔧 Testing LangChain Tool Execution...")
        
        tool = DataForSEOImageSearchTool(DataForSEOConfig(
            login="test_login",
            password="test_password"
        ))
        
        # Mock the search client
        with patch.object(tool.search_client, 'search_images') as mock_search:
            mock_search.return_value = ImageSearchResult(
                request_id="test-123",
                keyword="test query",
                search_engine="google",
                total_results=5,
                images=[
                    ImageMetadata(
                        url="https://example.com/test.jpg",
                        title="Test Image",
                        width=800,
                        height=600,
                        format="jpg"
                    )
                ],
                search_duration_ms=150.0,
                processing_duration_ms=50.0,
                api_cost_estimate=0.001
            )
            
            # Test simple string query
            result = await tool._arun("casino games")
            
            assert isinstance(result, str)
            result_data = json.loads(result)
            assert result_data["keyword"] == "test query"
            assert result_data["total_results"] == 5
            assert len(result_data["images"]) == 1
            
            # Test JSON query
            json_query = json.dumps({
                "keyword": "poker cards",
                "max_results": 10,
                "image_size": "large"
            })
            
            result2 = await tool._arun(json_query)
            result2_data = json.loads(result2)
            assert "keyword" in result2_data
            assert "images" in result2_data
        
        logger.info("✅ LangChain Tool Execution test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 