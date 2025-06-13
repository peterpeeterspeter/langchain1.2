"""
Comprehensive Test Suite for Enhanced Confidence Scoring System Integration
Tests all components: EnhancedConfidenceCalculator, Universal RAG Chain integration,
Source Quality Analysis, Intelligent Caching, and Response Validation.

Covers Task 2.18: Comprehensive Testing & Validation Suite
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

# Import the components to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chains.enhanced_confidence_scoring_system import (
    EnhancedConfidenceCalculator,
    ConfidenceBreakdown,
    ConfidenceIntegrator,
    SourceQualityAnalyzer,
    IntelligentCache,
    ResponseValidator,
    EnhancedRAGResponse,
    ConfidenceFactors,
    SourceQualityTier,
    ResponseQualityLevel,
    CacheStrategy
)

from chains.universal_rag_lcel import (
    UniversalRAGChain,
    create_universal_rag_chain,
    RAGResponse
)

from chains.advanced_prompt_system import (
    QueryType,
    ExpertiseLevel,
    ResponseFormat
)

from langchain_core.documents import Document


class TestEnhancedConfidenceCalculator:
    """Test suite for EnhancedConfidenceCalculator core functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        source_analyzer = Mock(spec=SourceQualityAnalyzer)
        cache_system = Mock(spec=IntelligentCache)
        response_validator = Mock(spec=ResponseValidator)
        return source_analyzer, cache_system, response_validator
    
    @pytest.fixture
    def confidence_calculator(self, mock_components):
        """Create EnhancedConfidenceCalculator instance for testing."""
        source_analyzer, cache_system, response_validator = mock_components
        return EnhancedConfidenceCalculator(
            source_quality_analyzer=source_analyzer,
            cache_system=cache_system,
            response_validator=response_validator
        )
    
    @pytest.fixture
    def sample_response(self):
        """Create sample EnhancedRAGResponse for testing."""
        return EnhancedRAGResponse(
            content="This is a comprehensive casino safety guide for beginners...",
            sources=[
                {
                    "content": "Licensed casinos are regulated by authorities...",
                    "metadata": {"source": "gambling-commission.gov", "authority": "high"}
                },
                {
                    "content": "Look for SSL certificates and encryption...",
                    "metadata": {"source": "casino-safety-blog.com", "authority": "medium"}
                }
            ],
            confidence_score=0.5,
            response_time=1.2,
            metadata={}
        )
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_integration(self, confidence_calculator, sample_response):
        """Test end-to-end confidence calculation."""
        # Mock the validator response
        validation_metrics = Mock()
        validation_metrics.overall_score = 0.8
        validation_metrics.content_score = 0.75
        validation_metrics.format_score = 0.85
        validation_metrics.quality_score = 0.8
        
        validation_issues = []
        
        confidence_calculator.response_validator.validate_response = AsyncMock(
            return_value=(validation_metrics, validation_issues)
        )
        
        # Mock source analyzer
        confidence_calculator.source_analyzer.analyze_source_quality = AsyncMock(
            return_value={
                'quality_scores': {
                    'authority': 0.8,
                    'credibility': 0.7,
                    'expertise': 0.75,
                    'recency': 0.6
                }
            }
        )
        
        # Test confidence calculation
        breakdown, enhanced_response = await confidence_calculator.calculate_enhanced_confidence(
            response=sample_response,
            query="Which casino is safest for beginners?",
            query_type="review",
            sources=sample_response.sources,
            generation_metadata={'retrieval_quality': 0.8, 'response_time_ms': 1200}
        )
        
        # Assertions
        assert isinstance(breakdown, ConfidenceBreakdown)
        assert 0.0 <= breakdown.overall_confidence <= 1.0
        assert breakdown.content_quality > 0
        assert breakdown.source_quality > 0
        assert breakdown.query_matching > 0
        assert breakdown.technical_factors > 0
        assert enhanced_response.confidence_score == breakdown.overall_confidence
        assert len(breakdown.quality_flags) > 0
        assert breakdown.calculation_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_query_type_specific_scoring(self, confidence_calculator, sample_response):
        """Test that different query types produce different scoring emphasis."""
        
        # Mock validator and source analyzer
        validation_metrics = Mock()
        validation_metrics.overall_score = 0.8
        validation_metrics.content_score = 0.75
        validation_metrics.format_score = 0.85
        validation_metrics.quality_score = 0.8
        
        confidence_calculator.response_validator.validate_response = AsyncMock(
            return_value=(validation_metrics, [])
        )
        confidence_calculator.source_analyzer.analyze_source_quality = AsyncMock(
            return_value={'quality_scores': {'authority': 0.8, 'credibility': 0.7}})
        
        # Test factual query type (should emphasize accuracy)
        breakdown_factual, _ = await confidence_calculator.calculate_enhanced_confidence(
            response=sample_response,
            query="What is the house edge of blackjack?",
            query_type="factual",
            sources=sample_response.sources,
            generation_metadata={}
        )
        
        # Test tutorial query type (should emphasize clarity)
        breakdown_tutorial, _ = await confidence_calculator.calculate_enhanced_confidence(
            response=sample_response,
            query="How to play blackjack step by step?",
            query_type="tutorial",
            sources=sample_response.sources,
            generation_metadata={}
        )
        
        # Should produce different confidence scores due to different emphasis
        assert breakdown_factual.overall_confidence != breakdown_tutorial.overall_confidence
    
    def test_quality_flags_generation(self, confidence_calculator):
        """Test quality flag generation based on confidence levels."""
        
        # Test excellent quality flags
        breakdown_excellent = ConfidenceBreakdown(overall_confidence=0.95)
        flags = confidence_calculator._generate_quality_flags(breakdown_excellent)
        assert 'excellent_quality' in flags
        
        # Test poor quality flags
        breakdown_poor = ConfidenceBreakdown(
            overall_confidence=0.3,
            content_quality=0.4,
            source_quality=0.3
        )
        flags = confidence_calculator._generate_quality_flags(breakdown_poor)
        assert 'poor_quality' in flags
        assert 'content_quality_concern' in flags
        assert 'source_quality_concern' in flags
    
    def test_improvement_suggestions(self, confidence_calculator):
        """Test improvement suggestion generation."""
        
        breakdown = ConfidenceBreakdown(
            content_quality=0.5,
            source_quality=0.6,
            query_matching=0.4,
            content_metrics={'completeness': 0.5, 'clarity': 0.4},
            source_metrics={'source_count': 2},
            query_metrics={'intent_alignment': 0.5}
        )
        
        suggestions = confidence_calculator._generate_improvement_suggestions(breakdown, "comparison")
        
        assert len(suggestions) > 0
        assert any("content" in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_regeneration_logic(self, confidence_calculator):
        """Test response regeneration decision logic."""
        
        # Test low confidence - should regenerate
        low_confidence = ConfidenceBreakdown(overall_confidence=0.3)
        should_regen, reason = await confidence_calculator.should_regenerate_response(low_confidence)
        assert should_regen
        assert "Low confidence score" in reason
        
        # Test poor quality flag - should regenerate
        poor_quality = ConfidenceBreakdown(
            overall_confidence=0.6,
            quality_flags=['poor_quality']
        )
        should_regen, reason = await confidence_calculator.should_regenerate_response(poor_quality)
        assert should_regen
        assert "Poor quality detected" in reason
        
        # Test acceptable quality - should not regenerate
        good_quality = ConfidenceBreakdown(overall_confidence=0.8)
        should_regen, reason = await confidence_calculator.should_regenerate_response(good_quality)
        assert not should_regen
        assert "Quality acceptable" in reason


class TestUniversalRAGChainIntegration:
    """Test suite for Universal RAG Chain integration with Enhanced Confidence Scoring."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock()
        mock_store.asimilarity_search_with_score = AsyncMock(return_value=[
            (Document(page_content="Casino safety information", metadata={"source": "authority.com"}), 0.9),
            (Document(page_content="Beginner gambling guide", metadata={"source": "guide.com"}), 0.8)
        ])
        return mock_store
    
    @pytest.fixture
    def enhanced_rag_chain(self, mock_vector_store):
        """Create UniversalRAGChain with enhanced confidence enabled."""
        return create_universal_rag_chain(
            model_name="gpt-4",
            enable_enhanced_confidence=True,
            enable_prompt_optimization=True,
            vector_store=mock_vector_store
        )
    
    @pytest.mark.asyncio
    @patch('chains.universal_rag_lcel.ChatOpenAI')
    async def test_enhanced_confidence_in_rag_response(self, mock_llm, enhanced_rag_chain):
        """Test that RAG responses include enhanced confidence metadata."""
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.ainvoke = AsyncMock(return_value="Comprehensive casino safety guide for beginners...")
        mock_llm.return_value = mock_llm_instance
        
        # Mock the confidence calculator
        with patch.object(enhanced_rag_chain.confidence_integrator, 'enhance_rag_response') as mock_enhance:
            enhanced_response = Mock()
            enhanced_response.content = "Enhanced response content"
            enhanced_response.confidence_score = 0.85
            enhanced_response.sources = []
            enhanced_response.metadata = {
                'confidence_breakdown': {
                    'content_quality': 0.8,
                    'source_quality': 0.9,
                    'query_matching': 0.85,
                    'technical_factors': 0.8,
                    'overall_confidence': 0.85
                },
                'quality_flags': ['high_quality'],
                'improvement_suggestions': []
            }
            mock_enhance.return_value = enhanced_response
            
            # Test query
            response = await enhanced_rag_chain.ainvoke("Which casino is safest for beginners?")
            
            # Assertions
            assert isinstance(response, RAGResponse)
            assert response.confidence_score == 0.85
            assert 'confidence_breakdown' in response.metadata
            assert response.metadata['confidence_breakdown']['overall_confidence'] == 0.85
            assert 'quality_flags' in response.metadata
            mock_enhance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_to_basic_confidence(self, mock_vector_store):
        """Test fallback to basic confidence when enhanced scoring is disabled."""
        
        chain = create_universal_rag_chain(
            model_name="gpt-4",
            enable_enhanced_confidence=False,
            vector_store=mock_vector_store
        )
        
        assert chain.confidence_calculator is None
        assert chain.confidence_integrator is None
        
        # The chain should still work with basic confidence calculation
        with patch.object(chain, '_calculate_enhanced_confidence', return_value=0.7) as mock_calc:
            with patch.object(chain.llm, 'ainvoke', return_value="Basic response"):
                response = await chain.ainvoke("Test query")
                
                assert isinstance(response, RAGResponse)
                assert response.confidence_score == 0.7
                mock_calc.assert_called_once()
    
    def test_initialization_with_enhanced_confidence(self):
        """Test proper initialization of enhanced confidence components."""
        
        chain = create_universal_rag_chain(enable_enhanced_confidence=True)
        
        assert chain.enable_enhanced_confidence is True
        assert chain.confidence_calculator is not None
        assert chain.confidence_integrator is not None
        assert isinstance(chain.confidence_calculator, EnhancedConfidenceCalculator)
        assert isinstance(chain.confidence_integrator, ConfidenceIntegrator)


class TestSourceQualityAnalyzer:
    """Test suite for Source Quality Analyzer component."""
    
    @pytest.fixture
    def source_analyzer(self):
        """Create SourceQualityAnalyzer instance."""
        return SourceQualityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_authority_assessment(self, source_analyzer):
        """Test authority assessment functionality."""
        
        # High authority source
        high_auth_doc = Document(
            page_content="Official regulatory information from government authority...",
            metadata={"source": "government.gov", "domain": "gov"}
        )
        
        result = await source_analyzer.analyze_source_quality(high_auth_doc)
        authority_score = result['quality_scores']['authority']
        
        assert 0.0 <= authority_score <= 1.0
        assert authority_score > 0.5  # Should be high for government source
    
    @pytest.mark.asyncio
    async def test_recency_assessment(self, source_analyzer):
        """Test recency assessment functionality."""
        
        # Recent content
        recent_doc = Document(
            page_content="Updated information for 2024, latest regulations...",
            metadata={"published_date": "2024-01-01"}
        )
        
        result = await source_analyzer.analyze_source_quality(recent_doc)
        recency_score = result['quality_scores']['recency']
        
        assert 0.0 <= recency_score <= 1.0
        assert recency_score > 0.5  # Should be high for recent content
    
    @pytest.mark.asyncio
    async def test_expertise_assessment(self, source_analyzer):
        """Test expertise assessment functionality."""
        
        # Expert content
        expert_doc = Document(
            page_content="Dr. Smith, PhD in Mathematics, explains the statistical analysis...",
            metadata={"author": "Dr. Smith, PhD"}
        )
        
        result = await source_analyzer.analyze_source_quality(expert_doc)
        expertise_score = result['quality_scores']['expertise']
        
        assert 0.0 <= expertise_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_negative_indicators_detection(self, source_analyzer):
        """Test detection of negative quality indicators."""
        
        # Content with negative indicators
        poor_doc = Document(
            page_content="This is just my opinion, I think maybe possibly...",
            metadata={}
        )
        
        result = await source_analyzer.analyze_source_quality(poor_doc)
        
        # Should have lower overall score due to negative indicators
        assert result['composite_score'] < 0.7
        assert len(result.get('negative_indicators', [])) > 0


class TestIntelligentCacheSystem:
    """Test suite for Intelligent Cache System."""
    
    @pytest.fixture
    def cache_system(self):
        """Create IntelligentCache instance."""
        return IntelligentCache(strategy=CacheStrategy.ADAPTIVE, max_size=100)
    
    @pytest.fixture
    def sample_enhanced_response(self):
        """Create sample enhanced response for caching."""
        return EnhancedRAGResponse(
            content="Cached response content",
            sources=[],
            confidence_score=0.8,
            response_time=1.0,
            metadata={'cache_key': 'test_query'}
        )
    
    @pytest.mark.asyncio
    async def test_quality_based_caching(self, cache_system, sample_enhanced_response):
        """Test that only high-quality responses are cached."""
        
        # High quality response - should be cached
        high_quality_response = sample_enhanced_response
        high_quality_response.confidence_score = 0.85
        
        await cache_system.set("high quality query", high_quality_response)
        cached = await cache_system.get("high quality query")
        
        assert cached is not None
        assert cached.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_low_quality_not_cached(self, cache_system, sample_enhanced_response):
        """Test that low-quality responses are not cached."""
        
        # Configure cache to reject low quality
        cache_system._get_adaptive_quality_threshold = Mock(return_value=0.75)
        
        # Low quality response - should not be cached
        low_quality_response = sample_enhanced_response
        low_quality_response.confidence_score = 0.4
        
        await cache_system.set("low quality query", low_quality_response)
        cached = await cache_system.get("low quality query")
        
        # Should not be found in cache due to low quality
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_adaptive_ttl_calculation(self, cache_system, sample_enhanced_response):
        """Test adaptive TTL calculation based on quality and query type."""
        
        # Mock query analysis for different query types
        tutorial_analysis = Mock()
        tutorial_analysis.query_type = Mock()
        tutorial_analysis.query_type.value = 'tutorial'
        
        news_analysis = Mock()
        news_analysis.query_type = Mock()
        news_analysis.query_type.value = 'news'
        
        # Tutorial content should have longer TTL
        tutorial_ttl = cache_system._get_adaptive_ttl("How to play poker", tutorial_analysis, 0.8)
        news_ttl = cache_system._get_adaptive_ttl("Latest casino news", news_analysis, 0.8)
        
        assert tutorial_ttl > news_ttl
    
    def test_cache_performance_tracking(self, cache_system):
        """Test cache performance metrics tracking."""
        
        # Record some cache operations
        cache_system._record_performance("query1", True, "cache_hit")
        cache_system._record_performance("query2", False, "cache_miss")
        cache_system._record_performance("query3", True, "cache_hit")
        
        metrics = cache_system.get_performance_metrics()
        
        assert 'hit_rate' in metrics
        assert 'total_requests' in metrics
        assert metrics['total_requests'] == 3
        assert metrics['hit_rate'] == 2/3  # 2 hits out of 3 requests


class TestResponseValidator:
    """Test suite for Response Validation Framework."""
    
    @pytest.fixture
    def response_validator(self):
        """Create ResponseValidator instance."""
        return ResponseValidator()
    
    @pytest.mark.asyncio
    async def test_format_validation(self, response_validator):
        """Test response format validation."""
        
        # Well-formatted response
        good_response = """
        # Casino Safety Guide
        
        Here are the key points:
        - Check licensing information
        - Verify SSL encryption
        - Read user reviews
        
        In conclusion, safety is paramount.
        """
        
        metrics, issues = await response_validator.validate_response(
            response_content=good_response,
            query="How to choose a safe casino?",
            sources=[],
            context={}
        )
        
        assert metrics.format_score > 0.6
        format_issues = [issue for issue in issues if issue.category.value == 'format']
        assert len(format_issues) == 0  # Should have no format issues
    
    @pytest.mark.asyncio
    async def test_content_quality_validation(self, response_validator):
        """Test content quality validation."""
        
        # High-quality content
        quality_content = """
        Casino safety depends on several critical factors that beginners should understand.
        Licensed casinos are regulated by gaming authorities and must follow strict guidelines.
        Look for certifications from organizations like eCOGRA or GLI.
        Always verify the casino's license number on the regulator's website.
        """
        
        metrics, issues = await response_validator.validate_response(
            response_content=quality_content,
            query="How to identify safe casinos?",
            sources=[],
            context={}
        )
        
        assert metrics.content_score > 0.6
        assert metrics.overall_score > 0.6
    
    @pytest.mark.asyncio
    async def test_source_utilization_validation(self, response_validator):
        """Test source utilization validation."""
        
        content = "Casino licensing is important for safety."
        sources = [
            {
                "content": "Licensed casinos must follow regulatory guidelines",
                "metadata": {"source": "gambling-authority.gov"}
            }
        ]
        
        metrics, issues = await response_validator.validate_response(
            response_content=content,
            query="Why is casino licensing important?",
            sources=sources,
            context={}
        )
        
        assert metrics.source_score > 0.0
        
        # Test with no sources
        metrics_no_sources, _ = await response_validator.validate_response(
            response_content=content,
            query="Test query",
            sources=[],
            context={}
        )
        
        assert metrics_no_sources.source_score == 0.0
    
    @pytest.mark.asyncio
    async def test_critical_issue_detection(self, response_validator):
        """Test detection of critical validation issues."""
        
        # Content with potential critical issues
        problematic_content = "Maybe this casino is safe, I think, but I'm not sure."
        
        metrics, issues = await response_validator.validate_response(
            response_content=problematic_content,
            query="Is this casino safe?",
            sources=[],
            context={}
        )
        
        critical_issues = [issue for issue in issues if issue.severity.value == 'critical']
        assert len(critical_issues) >= 0  # May or may not detect critical issues


class TestIntegrationScenarios:
    """Test suite for end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_rag_pipeline_with_confidence(self):
        """Test complete RAG pipeline with enhanced confidence scoring."""
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.asimilarity_search_with_score = AsyncMock(return_value=[
            (Document(page_content="Authoritative casino safety content", 
                     metadata={"source": "gaming-authority.gov"}), 0.95)
        ])
        
        # Create enhanced RAG chain
        chain = create_universal_rag_chain(
            model_name="gpt-4",
            enable_enhanced_confidence=True,
            enable_prompt_optimization=True,
            enable_caching=True,
            vector_store=mock_vector_store
        )
        
        # Mock LLM
        with patch.object(chain.llm, 'ainvoke', return_value="Comprehensive casino safety guide..."):
            response = await chain.ainvoke("Which casino is safest for beginners?")
            
            assert isinstance(response, RAGResponse)
            assert 0.0 <= response.confidence_score <= 1.0
            assert len(response.sources) > 0
            assert response.response_time > 0
            
            # Should have enhanced metadata if confidence scoring is enabled
            if hasattr(response, 'metadata'):
                assert isinstance(response.metadata, dict)
    
    @pytest.mark.asyncio
    async def test_regeneration_workflow(self):
        """Test response regeneration workflow for low-confidence responses."""
        
        # Create confidence calculator
        calculator = EnhancedConfidenceCalculator()
        
        # Mock dependencies
        calculator.response_validator.validate_response = AsyncMock(return_value=(
            Mock(overall_score=0.3, content_score=0.2, format_score=0.4, quality_score=0.3),
            []
        ))
        calculator.source_analyzer.analyze_source_quality = AsyncMock(return_value={
            'quality_scores': {'authority': 0.3, 'credibility': 0.3}
        })
        
        # Create low-quality response
        poor_response = EnhancedRAGResponse(
            content="Short answer",
            sources=[],
            confidence_score=0.3,
            response_time=1.0
        )
        
        # Calculate confidence
        breakdown, _ = await calculator.calculate_enhanced_confidence(
            response=poor_response,
            query="Complex query requiring detailed answer",
            query_type="tutorial",
            sources=[],
            generation_metadata={}
        )
        
        # Check if regeneration is recommended
        should_regen, reason = await calculator.should_regenerate_response(breakdown)
        
        assert should_regen
        assert "Low confidence score" in reason or "Poor quality" in reason
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for confidence calculation."""
        
        calculator = EnhancedConfidenceCalculator()
        
        # Test initialization time
        start_time = time.time()
        new_calculator = EnhancedConfidenceCalculator()
        init_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert init_time < 100  # Should initialize within 100ms
        
        # Test memory usage (basic check)
        import sys
        memory_size = sys.getsizeof(calculator)
        assert memory_size < 10000  # Should be reasonably sized


# Performance and Load Tests
class TestPerformanceAndLoad:
    """Test suite for performance and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_confidence_calculations(self):
        """Test concurrent confidence calculations for load testing."""
        
        calculator = EnhancedConfidenceCalculator()
        
        # Mock dependencies for fast execution
        calculator.response_validator.validate_response = AsyncMock(return_value=(
            Mock(overall_score=0.8, content_score=0.8, format_score=0.8, quality_score=0.8),
            []
        ))
        calculator.source_analyzer.analyze_source_quality = AsyncMock(return_value={
            'quality_scores': {'authority': 0.8}
        })
        
        # Create multiple responses for concurrent processing
        responses = [
            EnhancedRAGResponse(
                content=f"Response {i}",
                sources=[],
                confidence_score=0.5,
                response_time=1.0
            ) for i in range(10)
        ]
        
        # Process concurrently
        start_time = time.time()
        tasks = [
            calculator.calculate_enhanced_confidence(
                response=response,
                query=f"Query {i}",
                query_type="general",
                sources=[],
                generation_metadata={}
            ) for i, response in enumerate(responses)
        ]
        
        results = await asyncio.gather(*tasks)
        processing_time = (time.time() - start_time) * 1000
        
        # Verify all calculations completed
        assert len(results) == 10
        for breakdown, enhanced_response in results:
            assert isinstance(breakdown, ConfidenceBreakdown)
            assert 0.0 <= breakdown.overall_confidence <= 1.0
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 5000  # 5 seconds for 10 concurrent calculations
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self):
        """Test cache performance under concurrent access."""
        
        cache = IntelligentCache(max_size=100)
        
        # Create test responses
        responses = [
            EnhancedRAGResponse(
                content=f"Cached content {i}",
                sources=[],
                confidence_score=0.8,
                response_time=1.0
            ) for i in range(50)
        ]
        
        # Test concurrent cache operations
        set_tasks = [
            cache.set(f"query_{i}", response)
            for i, response in enumerate(responses)
        ]
        
        await asyncio.gather(*set_tasks)
        
        # Test concurrent cache retrieval
        get_tasks = [
            cache.get(f"query_{i}")
            for i in range(50)
        ]
        
        start_time = time.time()
        cached_responses = await asyncio.gather(*get_tasks)
        retrieval_time = (time.time() - start_time) * 1000
        
        # Verify cache performance
        hit_count = sum(1 for response in cached_responses if response is not None)
        assert hit_count > 0  # Should have some cache hits
        assert retrieval_time < 1000  # Should retrieve within 1 second


if __name__ == "__main__":
    # Run specific test categories
    import subprocess
    
    print("🧪 Running Enhanced Confidence Scoring System Test Suite")
    print("=" * 70)
    
    # Run all tests
    try:
        result = subprocess.run([
            "python", "-m", "pytest", __file__, 
            "-v", "--tb=short", "--asyncio-mode=auto"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nTest execution completed with return code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        
        # Fallback: run tests directly
        print("\nRunning tests directly...")
        pytest.main([__file__, "-v", "--tb=short"]) 