"""
Comprehensive Test Suite for Enhanced RAG System (Task 2)
Testing confidence scoring, caching, monitoring, A/B testing, and integration components.

Tests cover:
- Enhanced confidence scoring algorithms
- Intelligent caching with multiple strategies  
- Source quality analysis and validation
- Response validation and enhancement
- Performance monitoring and analytics
- Feature flags and A/B testing framework
- Configuration management
- Integration with main RAG chain
"""

import pytest
import asyncio
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from chains.enhanced_confidence_scoring_system import (
        EnhancedConfidenceCalculator,
        ConfidenceBreakdown,
        ConfidenceFactors,
        ConfidenceIntegrator,
        SourceQualityAnalyzer,
        IntelligentCache,
        ResponseValidator,
        EnhancedRAGResponse,
        SourceQualityTier,
        ResponseQualityLevel,
        CacheStrategy,
        UniversalRAGEnhancementSystem
    )
    ENHANCED_CONFIDENCE_AVAILABLE = True
except ImportError:
    ENHANCED_CONFIDENCE_AVAILABLE = False

try:
    from chains.universal_rag_lcel import (
        UniversalRAGChain,
        create_universal_rag_chain,
        RAGResponse
    )
    RAG_CHAIN_AVAILABLE = True
except ImportError:
    RAG_CHAIN_AVAILABLE = False

try:
    from config.prompt_config import (
        PromptOptimizationConfig,
        CacheConfig,
        QueryType,
        ExpertiseLevel,
        ResponseFormat
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from monitoring.prompt_analytics import (
        PromptAnalytics,
        AlertManager,
        MetricType
    )
    from monitoring.performance_profiler import PerformanceProfiler
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from config.feature_flags import (
        FeatureFlagManager,
        FeatureFlag,
        ExperimentManager,
        UserSegmentationStrategy
    )
    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False

from langchain_core.documents import Document


# ============================================================================
# ENHANCED CONFIDENCE SCORING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.confidence
class TestEnhancedConfidenceScoring:
    """Test enhanced confidence scoring algorithms and components."""
    
    @pytest.fixture
    def sample_rag_response(self):
        """Sample RAG response for testing."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        return EnhancedRAGResponse(
            content="This is a comprehensive casino safety guide for beginners. Licensed casinos are regulated by gaming authorities and use SSL encryption for secure transactions.",
            sources=[
                {
                    "content": "Licensed casinos are regulated by gaming authorities...",
                    "metadata": {"source": "gaming-commission.gov", "authority": "high", "recency": "2024-01-15"},
                    "quality_score": 0.85
                },
                {
                    "content": "SSL encryption ensures secure financial transactions...",
                    "metadata": {"source": "casino-security.org", "authority": "medium", "recency": "2024-02-01"},
                    "quality_score": 0.72
                }
            ],
            confidence_score=0.5,  # Initial score before enhancement
            response_time=1.5,
            cached=False,
            metadata={"query_type": "review", "expertise_level": "beginner"}
        )
    
    @pytest.fixture
    def mock_confidence_calculator(self):
        """Mock confidence calculator with dependencies."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        # Mock components
        source_analyzer = Mock(spec=SourceQualityAnalyzer)
        cache_system = Mock(spec=IntelligentCache)
        response_validator = Mock(spec=ResponseValidator)
        
        # Create calculator
        calculator = EnhancedConfidenceCalculator(
            source_quality_analyzer=source_analyzer,
            cache_system=cache_system,
            response_validator=response_validator
        )
        
        return calculator, source_analyzer, cache_system, response_validator
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_accuracy(self, mock_confidence_calculator, sample_rag_response):
        """Test accuracy of confidence calculation algorithm."""
        calculator, source_analyzer, cache_system, response_validator = mock_confidence_calculator
        
        # Mock validator response
        validation_metrics = Mock()
        validation_metrics.overall_score = 0.82
        validation_metrics.content_score = 0.78
        validation_metrics.format_score = 0.87
        validation_metrics.quality_score = 0.81
        
        response_validator.validate_response = AsyncMock(
            return_value=(validation_metrics, [])
        )
        
        # Mock source analyzer
        source_analyzer.analyze_source_quality = AsyncMock(
            return_value={
                'quality_scores': {
                    'authority': 0.85,
                    'credibility': 0.78,
                    'expertise': 0.82,
                    'recency': 0.75,
                    'detail': 0.80,
                    'objectivity': 0.77,
                    'transparency': 0.73,
                    'citation': 0.71
                },
                'tier': SourceQualityTier.HIGH,
                'penalties': []
            }
        )
        
        # Test confidence calculation
        breakdown, enhanced_response = await calculator.calculate_enhanced_confidence(
            response=sample_rag_response,
            query="Which casino is safest for beginners?",
            query_type="review",
            sources=sample_rag_response.sources,
            generation_metadata={'retrieval_quality': 0.8, 'response_time_ms': 1500}
        )
        
        # Assertions for confidence calculation accuracy
        assert isinstance(breakdown, ConfidenceBreakdown)
        assert 0.0 <= breakdown.overall_confidence <= 1.0
        assert breakdown.overall_confidence > sample_rag_response.confidence_score  # Should be enhanced
        
        # Test confidence factor components
        assert breakdown.content_quality > 0.5
        assert breakdown.source_quality > 0.5
        assert breakdown.query_matching > 0.5
        assert breakdown.technical_factors > 0.5
        
        # Test weighted calculation logic
        expected_confidence = (
            breakdown.content_quality * 0.35 +
            breakdown.source_quality * 0.25 +
            breakdown.query_matching * 0.20 +
            breakdown.technical_factors * 0.20
        )
        assert abs(breakdown.overall_confidence - expected_confidence) < 0.01
        
        # Test response enhancement
        assert enhanced_response.confidence_score == breakdown.overall_confidence
        assert len(breakdown.quality_flags) > 0
        assert breakdown.calculation_time_ms > 0
    
    def test_confidence_factors_calculation(self, mock_confidence_calculator):
        """Test individual confidence factor calculations."""
        calculator, _, _, _ = mock_confidence_calculator
        
        # Test excellent content factors
        excellent_factors = ConfidenceFactors(
            completeness=0.95,
            relevance=0.92,
            accuracy_indicators=0.90,
            source_reliability=0.88,
            source_coverage=0.91,
            source_consistency=0.89,
            intent_alignment=0.93,
            expertise_match=0.87,
            format_appropriateness=0.94,
            retrieval_quality=0.86,
            generation_stability=0.92,
            optimization_effectiveness=0.85
        )
        
        content_quality = calculator._calculate_content_quality(excellent_factors)
        source_quality = calculator._calculate_source_quality(excellent_factors)
        query_matching = calculator._calculate_query_matching(excellent_factors)
        technical_factors = calculator._calculate_technical_factors(excellent_factors)
        
        # Assertions for excellent quality
        assert content_quality > 0.85
        assert source_quality > 0.85
        assert query_matching > 0.85
        assert technical_factors > 0.85
        
        # Test poor content factors
        poor_factors = ConfidenceFactors(
            completeness=0.3,
            relevance=0.4,
            accuracy_indicators=0.2,
            source_reliability=0.35,
            source_coverage=0.25,
            source_consistency=0.3,
            intent_alignment=0.4,
            expertise_match=0.2,
            format_appropriateness=0.45,
            retrieval_quality=0.3,
            generation_stability=0.35,
            optimization_effectiveness=0.25
        )
        
        poor_content_quality = calculator._calculate_content_quality(poor_factors)
        poor_source_quality = calculator._calculate_source_quality(poor_factors)
        poor_query_matching = calculator._calculate_query_matching(poor_factors)
        poor_technical_factors = calculator._calculate_technical_factors(poor_factors)
        
        # Assertions for poor quality
        assert poor_content_quality < 0.5
        assert poor_source_quality < 0.5
        assert poor_query_matching < 0.5
        assert poor_technical_factors < 0.5
    
    def test_quality_flag_generation(self, mock_confidence_calculator):
        """Test quality flag generation based on confidence levels."""
        calculator, _, _, _ = mock_confidence_calculator
        
        # Test excellent quality flags
        excellent_breakdown = ConfidenceBreakdown(
            overall_confidence=0.95,
            content_quality=0.92,
            source_quality=0.94,
            query_matching=0.96,
            technical_factors=0.93,
            calculation_time_ms=450
        )
        
        excellent_flags = calculator._generate_quality_flags(excellent_breakdown)
        assert 'excellent_quality' in excellent_flags
        assert 'high_confidence' in excellent_flags
        assert len(excellent_flags) >= 2
        
        # Test concerning quality flags
        concerning_breakdown = ConfidenceBreakdown(
            overall_confidence=0.35,
            content_quality=0.4,
            source_quality=0.3,
            query_matching=0.45,
            technical_factors=0.25,
            calculation_time_ms=1200
        )
        
        concerning_flags = calculator._generate_quality_flags(concerning_breakdown)
        assert 'poor_quality' in concerning_flags
        assert 'content_quality_concern' in concerning_flags
        assert 'source_quality_concern' in concerning_flags
        assert 'technical_concern' in concerning_flags
        assert len(concerning_flags) >= 4
    
    def test_improvement_suggestions(self, mock_confidence_calculator):
        """Test improvement suggestion generation."""
        calculator, _, _, _ = mock_confidence_calculator
        
        # Test suggestions for low scores
        low_score_breakdown = ConfidenceBreakdown(
            overall_confidence=0.4,
            content_quality=0.3,
            source_quality=0.5,
            query_matching=0.6,
            technical_factors=0.2,
            calculation_time_ms=1800
        )
        
        suggestions = calculator._generate_improvement_suggestions(low_score_breakdown)
        
        # Should suggest improvements for low-scoring areas
        assert len(suggestions) > 0
        assert any('content quality' in suggestion.lower() for suggestion in suggestions)
        assert any('technical factors' in suggestion.lower() for suggestion in suggestions)
        assert any('performance' in suggestion.lower() for suggestion in suggestions)
        
        # Test fewer suggestions for high scores
        high_score_breakdown = ConfidenceBreakdown(
            overall_confidence=0.88,
            content_quality=0.85,
            source_quality=0.90,
            query_matching=0.87,
            technical_factors=0.89,
            calculation_time_ms=320
        )
        
        high_suggestions = calculator._generate_improvement_suggestions(high_score_breakdown)
        assert len(high_suggestions) <= len(suggestions)  # Fewer suggestions for high quality
    
    @pytest.mark.asyncio
    async def test_regeneration_threshold_logic(self, mock_confidence_calculator, sample_rag_response):
        """Test response regeneration threshold logic."""
        calculator, source_analyzer, cache_system, response_validator = mock_confidence_calculator
        
        # Mock low-quality validation
        low_quality_metrics = Mock()
        low_quality_metrics.overall_score = 0.25
        low_quality_metrics.content_score = 0.3
        low_quality_metrics.format_score = 0.2
        low_quality_metrics.quality_score = 0.25
        
        response_validator.validate_response = AsyncMock(
            return_value=(low_quality_metrics, ['critical_error', 'poor_formatting'])
        )
        
        source_analyzer.analyze_source_quality = AsyncMock(
            return_value={
                'quality_scores': {'authority': 0.3, 'credibility': 0.2},
                'tier': SourceQualityTier.LOW,
                'penalties': ['unreliable_source']
            }
        )
        
        # Test low-quality response
        breakdown, enhanced_response = await calculator.calculate_enhanced_confidence(
            response=sample_rag_response,
            query="Test query",
            query_type="factual",
            sources=sample_rag_response.sources,
            generation_metadata={}
        )
        
        # Should recommend regeneration for very low confidence
        assert breakdown.overall_confidence < 0.5
        assert 'regeneration_recommended' in breakdown.quality_flags
        assert len(breakdown.improvement_suggestions) > 2


# ============================================================================
# SOURCE QUALITY ANALYSIS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.confidence
class TestSourceQualityAnalyzer:
    """Test source quality analysis algorithms."""
    
    @pytest.fixture
    def source_analyzer(self):
        """Create source analyzer instance."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        return SourceQualityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_authority_assessment(self, source_analyzer):
        """Test authority scoring algorithm."""
        # Test high-authority source
        high_authority_source = {
            "content": "Academic research on gambling regulation",
            "metadata": {
                "source": "harvard.edu",
                "domain_authority": 95,
                "academic": True,
                "citation_count": 150
            }
        }
        
        authority_score = await source_analyzer._assess_authority(high_authority_source)
        assert authority_score > 0.8
        
        # Test low-authority source
        low_authority_source = {
            "content": "Random blog post about casinos",
            "metadata": {
                "source": "random-blog.com",
                "domain_authority": 15,
                "academic": False
            }
        }
        
        low_authority_score = await source_analyzer._assess_authority(low_authority_source)
        assert low_authority_score < 0.5
        assert authority_score > low_authority_score
    
    @pytest.mark.asyncio
    async def test_recency_assessment(self, source_analyzer):
        """Test recency scoring algorithm."""
        # Test recent source
        recent_source = {
            "content": "Latest gambling regulations",
            "metadata": {
                "published_date": "2024-01-15",
                "last_updated": "2024-01-20"
            }
        }
        
        recency_score = await source_analyzer._assess_recency(recent_source)
        assert recency_score > 0.8
        
        # Test old source
        old_source = {
            "content": "Outdated gambling information",
            "metadata": {
                "published_date": "2010-01-15",
                "last_updated": "2010-01-15"
            }
        }
        
        old_recency_score = await source_analyzer._assess_recency(old_source)
        assert old_recency_score < 0.4
        assert recency_score > old_recency_score
    
    @pytest.mark.asyncio
    async def test_expertise_detection(self, source_analyzer):
        """Test expertise detection algorithm."""
        # Test expert-authored content
        expert_source = {
            "content": "Professional analysis of casino mathematics and probability theory",
            "metadata": {
                "author": "Dr. John Smith, PhD Mathematics",
                "credentials": ["PhD", "Professor"],
                "experience_years": 15,
                "specialization": "gambling mathematics"
            }
        }
        
        expertise_score = await source_analyzer._assess_expertise(expert_source)
        assert expertise_score > 0.7
        
        # Test non-expert content
        amateur_source = {
            "content": "My personal experience at casinos",
            "metadata": {
                "author": "Anonymous User",
                "credentials": [],
                "experience_years": 0
            }
        }
        
        amateur_expertise_score = await source_analyzer._assess_expertise(amateur_source)
        assert amateur_expertise_score < 0.5
        assert expertise_score > amateur_expertise_score
    
    @pytest.mark.asyncio
    async def test_negative_indicators_detection(self, source_analyzer):
        """Test detection of negative quality indicators."""
        # Test source with negative indicators
        problematic_source = {
            "content": "CLICK HERE FOR AMAZING CASINO BONUSES!!! 100% GUARANTEED WINS!!!",
            "metadata": {
                "source": "spam-casino.com",
                "ads_present": True,
                "misleading_claims": True,
                "affiliate_heavy": True
            }
        }
        
        analysis = await source_analyzer.analyze_source_quality([problematic_source])
        
        assert len(analysis['penalties']) > 0
        assert analysis['tier'] in [SourceQualityTier.LOW, SourceQualityTier.VERY_LOW]
        assert any('misleading' in penalty.lower() for penalty in analysis['penalties'])
        
        # Test clean source
        clean_source = {
            "content": "Objective analysis of casino safety measures",
            "metadata": {
                "source": "regulatory-authority.gov",
                "ads_present": False,
                "misleading_claims": False,
                "objective_tone": True
            }
        }
        
        clean_analysis = await source_analyzer.analyze_source_quality([clean_source])
        
        assert len(clean_analysis['penalties']) == 0
        assert clean_analysis['tier'] in [SourceQualityTier.HIGH, SourceQualityTier.VERY_HIGH]
    
    @pytest.mark.asyncio
    async def test_comprehensive_quality_scoring(self, source_analyzer):
        """Test comprehensive source quality scoring."""
        # Test mixed-quality sources
        sources = [
            {
                "content": "Academic research on gambling addiction",
                "metadata": {
                    "source": "nih.gov",
                    "published_date": "2024-01-01",
                    "author": "Dr. Sarah Johnson",
                    "peer_reviewed": True
                }
            },
            {
                "content": "Casino promotional material",
                "metadata": {
                    "source": "casino-promo.com",
                    "published_date": "2024-02-01",
                    "promotional": True,
                    "bias_detected": True
                }
            },
            {
                "content": "Neutral casino review",
                "metadata": {
                    "source": "casino-reviews.org",
                    "published_date": "2023-12-15",
                    "balanced_perspective": True
                }
            }
        ]
        
        analysis = await source_analyzer.analyze_source_quality(sources)
        
        # Should have quality scores for all indicators
        quality_scores = analysis['quality_scores']
        required_indicators = ['authority', 'credibility', 'expertise', 'recency', 'detail', 'objectivity', 'transparency', 'citation']
        
        for indicator in required_indicators:
            assert indicator in quality_scores
            assert 0.0 <= quality_scores[indicator] <= 1.0
        
        # Should have overall tier assessment
        assert analysis['tier'] in [tier for tier in SourceQualityTier]
        
        # Should have penalty detection
        assert 'penalties' in analysis
        assert isinstance(analysis['penalties'], list)


# ============================================================================
# INTELLIGENT CACHE SYSTEM TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.cache
class TestIntelligentCacheSystem:
    """Test intelligent caching with multiple strategies."""
    
    @pytest.fixture
    def cache_configs(self):
        """Different cache configuration setups."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        return {
            'conservative': IntelligentCache(strategy=CacheStrategy.CONSERVATIVE, max_size=50),
            'balanced': IntelligentCache(strategy=CacheStrategy.BALANCED, max_size=100),
            'aggressive': IntelligentCache(strategy=CacheStrategy.AGGRESSIVE, max_size=200),
            'adaptive': IntelligentCache(strategy=CacheStrategy.ADAPTIVE, max_size=150)
        }
    
    @pytest.fixture
    def sample_responses(self):
        """Sample responses with different quality levels."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        return {
            'high_quality': EnhancedRAGResponse(
                content="Excellent detailed response about casino safety",
                sources=[{"content": "Gov source", "metadata": {"authority": "high"}}],
                confidence_score=0.92,
                response_time=0.8,
                cached=False,
                metadata={}
            ),
            'medium_quality': EnhancedRAGResponse(
                content="Good response about casino safety",
                sources=[{"content": "Blog source", "metadata": {"authority": "medium"}}],
                confidence_score=0.75,
                response_time=1.2,
                cached=False,
                metadata={}
            ),
            'low_quality': EnhancedRAGResponse(
                content="Basic response about casinos",
                sources=[{"content": "Weak source", "metadata": {"authority": "low"}}],
                confidence_score=0.45,
                response_time=2.1,
                cached=False,
                metadata={}
            )
        }
    
    @pytest.mark.asyncio
    async def test_quality_based_caching(self, cache_configs, sample_responses):
        """Test caching decisions based on response quality."""
        conservative_cache = cache_configs['conservative']
        
        # High quality should be cached
        high_quality_key = "high_quality_query"
        await conservative_cache.set(high_quality_key, sample_responses['high_quality'])
        
        cached_high = await conservative_cache.get(high_quality_key)
        assert cached_high is not None
        assert cached_high.confidence_score == 0.92
        
        # Low quality should not be cached in conservative mode
        low_quality_key = "low_quality_query"
        should_cache = conservative_cache._should_cache_response(sample_responses['low_quality'])
        assert not should_cache
        
        # Medium quality caching depends on strategy
        aggressive_cache = cache_configs['aggressive']
        medium_should_cache = aggressive_cache._should_cache_response(sample_responses['medium_quality'])
        assert medium_should_cache  # Aggressive should cache medium quality
    
    @pytest.mark.asyncio
    async def test_adaptive_ttl_calculation(self, cache_configs, sample_responses):
        """Test adaptive TTL calculation based on content quality."""
        adaptive_cache = cache_configs['adaptive']
        
        # High quality should get longer TTL
        high_ttl = adaptive_cache._calculate_adaptive_ttl(sample_responses['high_quality'])
        low_ttl = adaptive_cache._calculate_adaptive_ttl(sample_responses['low_quality'])
        
        assert high_ttl > low_ttl
        assert high_ttl >= 3600  # At least 1 hour for high quality
        assert low_ttl <= 1800   # Max 30 minutes for low quality
        
        # Test TTL factors
        # Recency should affect TTL
        recent_response = EnhancedRAGResponse(
            content="Recent information",
            sources=[{"metadata": {"published_date": "2024-01-15"}}],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        old_response = EnhancedRAGResponse(
            content="Old information",
            sources=[{"metadata": {"published_date": "2020-01-15"}}],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        recent_ttl = adaptive_cache._calculate_adaptive_ttl(recent_response)
        old_ttl = adaptive_cache._calculate_adaptive_ttl(old_response)
        
        assert recent_ttl > old_ttl
    
    def test_cache_performance_tracking(self, cache_configs):
        """Test cache performance metrics tracking."""
        balanced_cache = cache_configs['balanced']
        
        # Simulate cache operations
        balanced_cache._record_hit("query1")
        balanced_cache._record_hit("query2")
        balanced_cache._record_miss("query3")
        balanced_cache._record_miss("query4")
        balanced_cache._record_hit("query1")  # Another hit
        
        # Test metrics calculation
        metrics = balanced_cache.get_performance_metrics()
        
        assert 'hit_rate' in metrics
        assert 'miss_rate' in metrics
        assert 'total_requests' in metrics
        assert 'cache_size' in metrics
        
        # Hit rate should be 3/5 = 0.6
        assert abs(metrics['hit_rate'] - 0.6) < 0.01
        assert abs(metrics['miss_rate'] - 0.4) < 0.01
        assert metrics['total_requests'] == 5
    
    @pytest.mark.asyncio
    async def test_cache_strategy_behaviors(self, cache_configs, sample_responses):
        """Test different caching strategy behaviors."""
        conservative = cache_configs['conservative']
        aggressive = cache_configs['aggressive']
        
        medium_response = sample_responses['medium_quality']
        
        # Conservative should be selective
        conservative_should_cache = conservative._should_cache_response(medium_response)
        
        # Aggressive should cache more liberally
        aggressive_should_cache = aggressive._should_cache_response(medium_response)
        
        # Aggressive should be more likely to cache than conservative
        if not conservative_should_cache:
            assert aggressive_should_cache or aggressive_should_cache == conservative_should_cache
        
        # Test TTL differences
        conservative_ttl = conservative._calculate_adaptive_ttl(medium_response)
        aggressive_ttl = aggressive._calculate_adaptive_ttl(medium_response)
        
        # Aggressive might cache longer for marginal content
        assert aggressive_ttl >= conservative_ttl
    
    @pytest.mark.asyncio
    async def test_cache_eviction_policies(self, cache_configs, sample_responses):
        """Test cache eviction when size limits are reached."""
        small_cache = IntelligentCache(strategy=CacheStrategy.BALANCED, max_size=2)
        
        # Fill cache to capacity
        await small_cache.set("key1", sample_responses['high_quality'])
        await small_cache.set("key2", sample_responses['medium_quality'])
        
        # Adding third item should evict based on policy
        await small_cache.set("key3", sample_responses['high_quality'])
        
        # Should still have high quality items
        key1_cached = await small_cache.get("key1")
        key3_cached = await small_cache.get("key3")
        
        assert key1_cached is not None or key3_cached is not None  # At least one high quality retained
        
        # Medium quality more likely to be evicted
        key2_cached = await small_cache.get("key2")
        if key1_cached and key3_cached:
            assert key2_cached is None  # Medium quality evicted


# ============================================================================
# RESPONSE VALIDATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.validation
class TestResponseValidator:
    """Test response validation and enhancement algorithms."""
    
    @pytest.fixture
    def response_validator(self):
        """Create response validator instance."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        return ResponseValidator()
    
    @pytest.mark.asyncio
    async def test_format_validation(self, response_validator):
        """Test response format validation."""
        # Test well-formatted response
        good_response = EnhancedRAGResponse(
            content="This is a well-structured response about casino safety. It provides clear information with proper formatting and logical flow.",
            sources=[{"content": "Source content", "metadata": {}}],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        metrics, issues = await response_validator.validate_response(
            good_response, "What makes a casino safe?", "review"
        )
        
        assert metrics.format_score > 0.7
        assert len([issue for issue in issues if 'format' in issue.lower()]) == 0
        
        # Test poorly formatted response
        bad_response = EnhancedRAGResponse(
            content="casino safe yes good",  # Poor formatting, fragments
            sources=[{"content": "Source", "metadata": {}}],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        bad_metrics, bad_issues = await response_validator.validate_response(
            bad_response, "What makes a casino safe?", "review"
        )
        
        assert bad_metrics.format_score < 0.5
        assert len([issue for issue in bad_issues if 'format' in issue.lower()]) > 0
    
    @pytest.mark.asyncio
    async def test_content_quality_validation(self, response_validator):
        """Test content quality validation."""
        # Test comprehensive, relevant content
        comprehensive_response = EnhancedRAGResponse(
            content="Casino safety depends on several key factors: 1) Licensing and regulation by recognized authorities, 2) Use of SSL encryption for data protection, 3) Independent auditing of games for fairness, 4) Responsible gambling tools and policies, 5) Transparent terms and conditions. Licensed casinos must comply with strict security standards...",
            sources=[
                {"content": "Regulatory information", "metadata": {"authority": "high"}},
                {"content": "Security standards", "metadata": {"authority": "high"}}
            ],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        metrics, issues = await response_validator.validate_response(
            comprehensive_response, "What makes a casino safe?", "review"
        )
        
        assert metrics.content_score > 0.8
        assert metrics.quality_score > 0.8
        assert len([issue for issue in issues if 'incomplete' in issue.lower()]) == 0
        
        # Test incomplete, low-quality content
        incomplete_response = EnhancedRAGResponse(
            content="Casinos are safe sometimes.",  # Vague, incomplete
            sources=[{"content": "Weak source", "metadata": {"authority": "low"}}],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        incomplete_metrics, incomplete_issues = await response_validator.validate_response(
            incomplete_response, "What makes a casino safe?", "review"
        )
        
        assert incomplete_metrics.content_score < 0.5
        assert len([issue for issue in incomplete_issues if any(term in issue.lower() for term in ['incomplete', 'vague', 'brief'])]) > 0
    
    @pytest.mark.asyncio
    async def test_source_utilization_validation(self, response_validator):
        """Test validation of source utilization in response."""
        # Test good source utilization
        well_sourced_response = EnhancedRAGResponse(
            content="Licensed casinos are regulated by gaming authorities (as confirmed by regulatory documentation). They use SSL encryption for secure transactions (verified by security audits). Independent testing ensures game fairness (per third-party audit reports).",
            sources=[
                {"content": "Gaming authorities regulate licensed casinos...", "metadata": {"source": "gaming-commission.gov"}},
                {"content": "SSL encryption secures transactions...", "metadata": {"source": "security-audit.org"}},
                {"content": "Independent testing ensures fairness...", "metadata": {"source": "audit-reports.com"}}
            ],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        metrics, issues = await response_validator.validate_response(
            well_sourced_response, "How are casinos regulated?", "factual"
        )
        
        assert metrics.overall_score > 0.8
        source_issues = [issue for issue in issues if 'source' in issue.lower()]
        assert len(source_issues) == 0
        
        # Test poor source utilization
        poorly_sourced_response = EnhancedRAGResponse(
            content="Casinos have regulations and security measures. They are generally safe.",  # Doesn't reference sources
            sources=[
                {"content": "Detailed regulatory framework...", "metadata": {"source": "gov.org"}},
                {"content": "Comprehensive security measures...", "metadata": {"source": "security.org"}}
            ],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        poor_metrics, poor_issues = await response_validator.validate_response(
            poorly_sourced_response, "How are casinos regulated?", "factual"
        )
        
        assert poor_metrics.overall_score < 0.6
        source_utilization_issues = [issue for issue in poor_issues if 'source' in issue.lower()]
        assert len(source_utilization_issues) > 0
    
    @pytest.mark.asyncio
    async def test_critical_issue_detection(self, response_validator):
        """Test detection of critical response issues."""
        # Test response with critical issues
        critical_response = EnhancedRAGResponse(
            content="All casinos are 100% safe and you will never lose money. Guaranteed wins every time!",  # Misleading claims
            sources=[{"content": "Promotional material", "metadata": {"source": "casino-ads.com"}}],
            confidence_score=0.8,
            response_time=1.0,
            cached=False,
            metadata={}
        )
        
        metrics, issues = await response_validator.validate_response(
            critical_response, "Are casinos safe?", "review"
        )
        
        assert metrics.overall_score < 0.3
        critical_issues = [issue for issue in issues if any(term in issue.lower() for term in ['misleading', 'false', 'critical'])]
        assert len(critical_issues) > 0
        
        # Should flag for regeneration
        regeneration_recommended = any('regenerat' in issue.lower() for issue in issues)
        assert regeneration_recommended


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.confidence
class TestEnhancedRAGIntegration:
    """Integration tests for enhanced RAG system components."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated enhancement system."""
        if not ENHANCED_CONFIDENCE_AVAILABLE:
            pytest.skip("Enhanced confidence system not available")
        
        return UniversalRAGEnhancementSystem()
    
    @pytest.mark.asyncio
    async def test_end_to_end_enhancement_pipeline(self, integrated_system):
        """Test complete enhancement pipeline."""
        # Mock initial RAG response
        initial_response = EnhancedRAGResponse(
            content="Casino safety involves licensing and security measures.",
            sources=[
                {"content": "Licensed casinos follow regulations", "metadata": {"source": "regulator.gov"}},
                {"content": "Security measures protect players", "metadata": {"source": "casino-security.org"}}
            ],
            confidence_score=0.5,  # Initial low confidence
            response_time=1.8,
            cached=False,
            metadata={}
        )
        
        # Process through enhancement pipeline
        enhanced_response = await integrated_system.enhance_response(
            response=initial_response,
            query="What makes casinos safe for players?",
            query_type="review",
            user_context={"expertise_level": "beginner"}
        )
        
        # Assertions for enhancement
        assert enhanced_response.confidence_score > initial_response.confidence_score
        assert hasattr(enhanced_response, 'confidence_breakdown')
        assert hasattr(enhanced_response, 'quality_flags')
        assert hasattr(enhanced_response, 'improvement_suggestions')
        
        # Should have enhanced metadata
        assert 'enhancement_applied' in enhanced_response.metadata
        assert 'processing_time_ms' in enhanced_response.metadata
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, integrated_system):
        """Test caching integration in enhancement pipeline."""
        query = "How do I verify casino licensing?"
        
        # First request should not be cached
        response1 = await integrated_system.get_enhanced_response(query, "factual")
        assert not response1.cached
        
        # Second identical request should be cached (if quality is sufficient)
        response2 = await integrated_system.get_enhanced_response(query, "factual")
        
        if response1.confidence_score > 0.7:  # High quality gets cached
            assert response2.cached
            assert response2.content == response1.content
        
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, integrated_system):
        """Test performance monitoring during enhancement."""
        # Process multiple responses to generate metrics
        queries = [
            ("Are online casinos safe?", "review"),
            ("What is the house edge in blackjack?", "factual"),
            ("How to choose a casino bonus?", "tutorial")
        ]
        
        responses = []
        for query, query_type in queries:
            response = await integrated_system.get_enhanced_response(query, query_type)
            responses.append(response)
        
        # Get performance metrics
        metrics = integrated_system.get_performance_metrics()
        
        assert 'total_requests' in metrics
        assert 'average_confidence' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'average_response_time' in metrics
        
        assert metrics['total_requests'] >= len(queries)
        assert 0.0 <= metrics['average_confidence'] <= 1.0
        assert 0.0 <= metrics['cache_hit_rate'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 