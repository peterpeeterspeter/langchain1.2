"""
Comprehensive Testing Framework for Content Processing Pipeline
Task 10.5: Production-ready test suite for Enhanced FTI Pipeline components

Features:
- Unit tests for content type detection
- Adaptive chunking strategy validation  
- Pipeline orchestration testing
- Feature/Training/Inference testing
- Performance benchmarks and quality validation
- Integration with Tasks 1-3
"""

import pytest
import asyncio
import time
import logging
import json
import tempfile
import psutil
import hashlib
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

# Test configuration and utilities
from tests.conftest import *

# Import content processing pipeline components
try:
    from src.pipelines.enhanced_fti_pipeline import (
        EnhancedFTIOrchestrator, EnhancedFTIConfig, PipelineStage,
        ContentType, QueryType, EnhancedFeatureData, TrainingMetrics,
        EnhancedInferenceResult
    )
    from src.pipelines.content_type_detector import (
        ContentTypeDetector, ContentType as DetectorContentType,
        ContentTypeResult
    )
    from src.pipelines.adaptive_chunking import (
        AdaptiveChunkingSystem, ChunkingStrategy, ProcessedChunk,
        ChunkMetadata, FixedSizeChunker, SemanticChunker,
        StructuralChunker, AdaptiveChunker
    )
    CONTENT_PIPELINE_AVAILABLE = True
except ImportError as e:
    CONTENT_PIPELINE_AVAILABLE = False
    pytest.skip(f"Content processing pipeline components not available: {e}", allow_module_level=True)


# ============================================================================
# PERFORMANCE BENCHMARKS AND THRESHOLDS  
# ============================================================================

PIPELINE_PERFORMANCE_THRESHOLDS = {
    'max_content_detection_time_ms': 1000,
    'max_chunking_time_ms': 2000,
    'max_feature_extraction_time_ms': 3000,
    'max_inference_time_ms': 5000,
    'min_content_type_accuracy': 0.85,
    'min_chunking_quality_score': 0.7,
    'max_memory_usage_mb': 200,
    'min_throughput_docs_per_sec': 2.0
}

QUALITY_THRESHOLDS = {
    'min_content_detection_confidence': 0.8,
    'min_chunk_coherence_score': 0.6,
    'min_semantic_similarity': 0.7,
    'max_chunk_size_variance': 0.3,
    'min_pipeline_success_rate': 0.95
}


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

@dataclass
class TestContent:
    """Enhanced test content with metadata for pipeline testing."""
    content: str
    expected_type: ContentType
    expected_chunks: int
    quality_indicators: Dict[str, Any]
    processing_complexity: str
    language: str = "en"

class ContentProcessingTestDataGenerator:
    """Generate realistic test data for content processing pipeline testing."""
    
    @staticmethod
    def create_diverse_content_samples(count: int = 50) -> List[TestContent]:
        """Generate diverse content samples for comprehensive testing."""
        content_templates = {
            ContentType.CASINO_REVIEW: {
                'templates': [
                    """# {casino_name} Casino Review 2024
                    
                    ## Overview
                    {casino_name} is a {adjective} online casino established in {year}. Licensed by {regulator}, 
                    this casino offers {game_count}+ games from {providers}.
                    
                    ## Games & Software
                    The casino features an extensive collection of:
                    - Slot machines ({slot_count}+ titles)
                    - Table games (Blackjack, Roulette, Baccarat)
                    - Live dealer games
                    - Progressive jackpots
                    
                    ## Bonuses & Promotions
                    Welcome Package: {bonus_amount} + {free_spins} free spins
                    Wagering requirements: {wagering}x
                    
                    ## Payment Methods
                    Deposits: {deposit_methods}
                    Withdrawals: {withdrawal_methods}
                    Processing time: {processing_time}
                    
                    ## Verdict
                    Rating: {rating}/5
                    {verdict_text}
                    """,
                ],
                'variables': {
                    'casino_name': ['Golden Palace', 'Royal Flush', 'Lucky Star', 'Diamond Casino'],
                    'adjective': ['reputable', 'established', 'innovative', 'player-friendly'],
                    'year': ['2018', '2019', '2020', '2021'],
                    'regulator': ['Malta Gaming Authority', 'UK Gambling Commission', 'Curacao eGaming'],
                    'game_count': ['500', '800', '1200', '1500'],
                    'providers': ['NetEnt, Microgaming', 'Evolution Gaming, Pragmatic Play'],
                    'slot_count': ['300', '500', '700', '900'],
                    'bonus_amount': ['100% up to $500', '200% up to $1000', '50% up to $2000'],
                    'free_spins': ['50', '100', '200', '300'],
                    'wagering': ['35', '40', '25', '45'],
                    'deposit_methods': ['Credit cards, e-wallets', 'Crypto, bank transfer'],
                    'withdrawal_methods': ['E-wallets, bank transfer', 'Crypto, credit cards'],
                    'processing_time': ['24-48 hours', '1-3 business days', 'Instant'],
                    'rating': ['4.2', '4.5', '3.8', '4.1'],
                    'verdict_text': ['Excellent choice for serious players', 'Good for casual gaming', 'Recommended for high rollers']
                },
                'expected_chunks': 8,
                'complexity': 'medium'
            },
            
            ContentType.TECHNICAL_DOCS: {
                'templates': [
                    """# API Documentation - {api_name}
                    
                    ## Introduction
                    The {api_name} provides {functionality} for {use_case}.
                    Base URL: https://api.{domain}.com/v{version}
                    
                    ## Authentication
                    ```http
                    Authorization: Bearer YOUR_API_KEY
                    ```
                    
                    ## Endpoints
                    
                    ### GET /{endpoint}
                    Retrieves {resource_type} data.
                    
                    #### Parameters
                    - `limit` (integer): Maximum number of results (default: 10)
                    - `offset` (integer): Number of results to skip (default: 0)
                    - `filter` (string): Filter criteria
                    
                    #### Response
                    ```json
                    {{
                        "data": [{{"id": 1, "name": "example"}}],
                        "total": 100,
                        "limit": 10,
                        "offset": 0
                    }}
                    ```
                    
                    ### POST /{endpoint}
                    Creates a new {resource_type}.
                    
                    #### Request Body
                    ```json
                    {{
                        "name": "string",
                        "description": "string",
                        "active": boolean
                    }}
                    ```
                    
                    ## Error Handling
                    The API returns standard HTTP status codes:
                    - 200: Success
                    - 400: Bad Request
                    - 401: Unauthorized
                    - 404: Not Found
                    - 500: Internal Server Error
                    
                    ## Rate Limiting
                    Rate limit: {rate_limit} requests per minute.
                    """,
                ],
                'variables': {
                    'api_name': ['UserAPI', 'PaymentAPI', 'GameAPI', 'CasinoAPI'],
                    'functionality': ['user management', 'payment processing', 'game integration'],
                    'use_case': ['mobile apps', 'web applications', 'third-party integration'],
                    'domain': ['example', 'api', 'service', 'platform'],
                    'version': ['1', '2', '3'],
                    'endpoint': ['users', 'payments', 'games', 'casinos'],
                    'resource_type': ['user', 'payment', 'game', 'casino'],
                    'rate_limit': ['100', '500', '1000', '60']
                },
                'expected_chunks': 12,
                'complexity': 'high'
            },
            
            ContentType.NEWS_ARTICLE: {
                'templates': [
                    """{headline}
                    
                    {city}, {date} - {lead_paragraph}
                    
                    {body_paragraph_1}
                    
                    {quote_paragraph}
                    
                    {body_paragraph_2}
                    
                    {conclusion_paragraph}
                    
                    This story is developing and will be updated as more information becomes available.
                    """,
                ],
                'variables': {
                    'headline': ['Online Gaming Industry Sees Record Growth', 'New Regulations Impact Casino Operations'],
                    'city': ['Las Vegas', 'London', 'Malta', 'Gibraltar'],
                    'date': ['January 15, 2024', 'February 20, 2024', 'March 10, 2024'],
                    'lead_paragraph': ['The online gaming sector reported unprecedented growth this quarter.', 'Industry leaders met to discuss new regulatory frameworks.'],
                    'body_paragraph_1': ['Market analysts attribute this growth to several factors.', 'The new regulations will take effect next month.'],
                    'quote_paragraph': ['"This represents a significant shift in the industry," said industry expert Jane Smith.', '"We are committed to responsible gaming," stated CEO John Doe.'],
                    'body_paragraph_2': ['Companies are adapting their strategies accordingly.', 'Operators are implementing new compliance measures.'],
                    'conclusion_paragraph': ['The impact of these changes will be closely monitored.', 'Industry stakeholders remain optimistic about the future.']
                },
                'expected_chunks': 4,
                'complexity': 'low'
            }
        }
        
        content_samples = []
        
        for i in range(count):
            content_type = list(content_templates.keys())[i % len(content_templates)]
            template_data = content_templates[content_type]
            
            # Select template and generate content
            template = np.random.choice(template_data['templates'])
            content = template
            
            # Replace variables
            for var_name, values in template_data['variables'].items():
                if f'{{{var_name}}}' in content:
                    content = content.replace(f'{{{var_name}}}', np.random.choice(values))
            
            # Generate quality indicators
            quality_indicators = {
                'word_count': len(content.split()),
                'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                'has_headings': '#' in content,
                'has_code_blocks': '```' in content,
                'has_lists': '-' in content or '•' in content,
                'complexity_score': np.random.uniform(0.5, 0.9)
            }
            
            content_samples.append(TestContent(
                content=content,
                expected_type=content_type,
                expected_chunks=template_data['expected_chunks'],
                quality_indicators=quality_indicators,
                processing_complexity=template_data['complexity']
            ))
        
        return content_samples
    
    @staticmethod
    def create_edge_case_content() -> List[TestContent]:
        """Generate edge case content for robust testing."""
        edge_cases = [
            TestContent(
                content="",
                expected_type=ContentType.GENERAL,
                expected_chunks=0,
                quality_indicators={'word_count': 0},
                processing_complexity='minimal'
            ),
            TestContent(
                content="A",
                expected_type=ContentType.GENERAL,
                expected_chunks=1,
                quality_indicators={'word_count': 1},
                processing_complexity='minimal'
            ),
            TestContent(
                content="A " * 10000,  # Very long repetitive content
                expected_type=ContentType.GENERAL,
                expected_chunks=20,
                quality_indicators={'word_count': 10000},
                processing_complexity='high'
            ),
            TestContent(
                content="🎰🎲🃏 Casino review with emojis and special characters: @#$%^&*()[]{}",
                expected_type=ContentType.CASINO_REVIEW,
                expected_chunks=1,
                quality_indicators={'word_count': 11, 'has_special_chars': True},
                processing_complexity='medium'
            )
        ]
        
        return edge_cases 


# ============================================================================
# PYTEST FIXTURES AND MOCKS
# ============================================================================

@pytest.fixture(scope="session")
def test_content_samples():
    """Generate test content samples for the session."""
    return ContentProcessingTestDataGenerator.create_diverse_content_samples(30)

@pytest.fixture(scope="session")
def edge_case_content():
    """Generate edge case content for testing."""
    return ContentProcessingTestDataGenerator.create_edge_case_content()

@pytest.fixture
def fti_config():
    """Create test configuration for Enhanced FTI Pipeline."""
    return EnhancedFTIConfig(
        supabase_url="http://localhost:54321",
        supabase_key="test_key",
        openai_api_key="test_openai_key",
        model_name="gpt-4o",
        temperature=0.1,
        chunk_size=800,
        chunk_overlap=150,
        dense_weight=0.7,
        sparse_weight=0.3,
        mmr_lambda=0.7,
        enable_caching=True,
        cache_ttl_hours=24,
        async_processing=True
    )

@pytest.fixture
def mock_embeddings():
    """Mock embeddings model for testing."""
    mock = Mock(spec=Embeddings)
    
    def create_embedding(text: str) -> List[float]:
        # Create deterministic embeddings based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        return np.random.normal(0, 1, 1536).tolist()
    
    mock.embed_documents = AsyncMock(side_effect=lambda texts: [create_embedding(t) for t in texts])
    mock.embed_query = AsyncMock(side_effect=create_embedding)
    
    return mock

@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    mock_client = Mock()
    
    # Mock vector store operations
    mock_table = Mock()
    mock_table.insert = AsyncMock(return_value=Mock(data=[{"id": "test_id"}]))
    mock_table.select = Mock(return_value=Mock(execute=AsyncMock(return_value=Mock(data=[]))))
    mock_table.update = AsyncMock(return_value=Mock(data=[]))
    mock_table.delete = AsyncMock(return_value=Mock())
    
    mock_client.table = Mock(return_value=mock_table)
    mock_client.storage = Mock()
    
    return mock_client

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    
    def generate_response(prompt: str) -> str:
        # Simple mock responses based on prompt content
        if "content type" in prompt.lower():
            return "CASINO_REVIEW"
        elif "query variations" in prompt.lower():
            return "What are the best casino bonuses?\nWhich casinos offer the highest bonuses?\nTop casino bonus offers"
        elif "metadata filters" in prompt.lower():
            return '{"category": "casino_review", "rating": ">4.0"}'
        else:
            return "Mock response for testing purposes."
    
    mock.invoke = AsyncMock(side_effect=lambda input_data: Mock(content=generate_response(str(input_data))))
    mock.ainvoke = AsyncMock(side_effect=lambda input_data: Mock(content=generate_response(str(input_data))))
    
    return mock

@pytest.fixture
async def content_type_detector():
    """Create configured content type detector for testing."""
    detector = ContentTypeDetector(cache_ttl=300)
    return detector

@pytest.fixture
async def adaptive_chunking_system(mock_supabase_client):
    """Create configured adaptive chunking system for testing."""
    system = AdaptiveChunkingSystem(supabase_client=mock_supabase_client)
    return system

@pytest.fixture
async def fti_orchestrator(fti_config, mock_embeddings, mock_supabase_client, mock_llm):
    """Create configured FTI orchestrator for testing."""
    with patch('src.pipelines.enhanced_fti_pipeline.OpenAIEmbeddings', return_value=mock_embeddings), \
         patch('src.pipelines.enhanced_fti_pipeline.ChatOpenAI', return_value=mock_llm), \
         patch('src.pipelines.enhanced_fti_pipeline.create_client', return_value=mock_supabase_client):
        
        orchestrator = EnhancedFTIOrchestrator(config=fti_config)
        await orchestrator.initialize()
        return orchestrator


# ============================================================================
# UNIT TESTS: CONTENT TYPE DETECTION
# ============================================================================

@pytest.mark.unit
@pytest.mark.content_pipeline
class TestContentTypeDetector:
    """Comprehensive unit tests for content type detection system."""
    
    async def test_casino_review_detection(self, content_type_detector, test_content_samples):
        """Test accurate detection of casino review content."""
        casino_samples = [sample for sample in test_content_samples 
                         if sample.expected_type == ContentType.CASINO_REVIEW]
        
        if not casino_samples:
            pytest.skip("No casino review samples available")
        
        correct_detections = 0
        total_detections = 0
        
        for sample in casino_samples[:5]:  # Test first 5 samples
            result = await content_type_detector.detect_content_type(
                content=sample.content,
                filename="test_review.html"
            )
            
            total_detections += 1
            if result.content_type.value == sample.expected_type.value:
                correct_detections += 1
            
            # Validate detection quality
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            assert result.processing_time >= 0
            assert isinstance(result.indicators, dict)
            assert isinstance(result.metadata, dict)
        
        # Check accuracy threshold
        accuracy = correct_detections / total_detections if total_detections > 0 else 0
        assert accuracy >= QUALITY_THRESHOLDS['min_content_detection_confidence']
    
    async def test_technical_documentation_detection(self, content_type_detector, test_content_samples):
        """Test accurate detection of technical documentation."""
        tech_samples = [sample for sample in test_content_samples 
                       if sample.expected_type == ContentType.TECHNICAL_DOCS]
        
        if not tech_samples:
            pytest.skip("No technical documentation samples available")
        
        for sample in tech_samples[:3]:
            result = await content_type_detector.detect_content_type(
                content=sample.content,
                filename="api_docs.md"
            )
            
            # Technical docs should have high confidence due to structure
            assert result.confidence >= 0.7
            
            # Should detect code blocks and technical indicators
            indicators = result.indicators
            assert 'structure' in indicators
            assert 'keywords' in indicators
            
            # Validate processing time
            assert result.processing_time < PIPELINE_PERFORMANCE_THRESHOLDS['max_content_detection_time_ms'] / 1000
    
    async def test_content_type_caching(self, content_type_detector):
        """Test caching mechanism for content type detection."""
        test_content = "This is a comprehensive casino review with detailed analysis."
        
        # First detection
        start_time = time.time()
        result1 = await content_type_detector.detect_content_type(content=test_content)
        first_detection_time = time.time() - start_time
        
        # Second detection (should be cached)
        start_time = time.time()
        result2 = await content_type_detector.detect_content_type(content=test_content)
        second_detection_time = time.time() - start_time
        
        # Results should be identical
        assert result1.content_type == result2.content_type
        assert result1.confidence == result2.confidence
        
        # Second detection should be faster (cached)
        assert second_detection_time < first_detection_time
        
        # Validate cache statistics
        cache_stats = content_type_detector.get_cache_stats()
        assert cache_stats['total_entries'] >= 1
        assert cache_stats['hit_rate'] >= 0.0
    
    async def test_edge_case_content_handling(self, content_type_detector, edge_case_content):
        """Test handling of edge case content."""
        for edge_case in edge_case_content:
            result = await content_type_detector.detect_content_type(
                content=edge_case.content
            )
            
            # Should handle edge cases gracefully
            assert result is not None
            assert result.content_type in ContentType
            assert 0.0 <= result.confidence <= 1.0
            assert result.processing_time >= 0
            
            # Empty content should have low confidence
            if not edge_case.content.strip():
                assert result.confidence <= 0.5
    
    @pytest.mark.performance
    async def test_content_detection_performance(self, content_type_detector, test_content_samples):
        """Test content type detection performance under load."""
        # Test processing speed
        start_time = time.time()
        
        tasks = []
        for sample in test_content_samples[:10]:
            task = content_type_detector.detect_content_type(content=sample.content)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Validate all results
        for result in results:
            assert result is not None
            assert result.processing_time < PIPELINE_PERFORMANCE_THRESHOLDS['max_content_detection_time_ms'] / 1000
        
        # Check throughput
        throughput = len(results) / total_time
        assert throughput >= 5.0  # At least 5 detections per second 


# ============================================================================
# UNIT TESTS: ADAPTIVE CHUNKING SYSTEM
# ============================================================================

@pytest.mark.unit
@pytest.mark.content_pipeline
class TestAdaptiveChunkingSystem:
    """Comprehensive unit tests for adaptive chunking system."""
    
    async def test_fixed_size_chunking(self, adaptive_chunking_system, test_content_samples):
        """Test fixed-size chunking strategy."""
        # Get a long content sample
        long_samples = [sample for sample in test_content_samples 
                       if sample.quality_indicators['word_count'] > 500]
        
        if not long_samples:
            pytest.skip("No long content samples available")
        
        sample = long_samples[0]
        
        # Process with fixed-size chunking
        chunks = await adaptive_chunking_system.process_content(
            content=sample.content,
            content_id="test_fixed_chunking"
        )
        
        # Validate chunking results
        assert len(chunks) > 0
        assert all(isinstance(chunk, ProcessedChunk) for chunk in chunks)
        
        # Check chunk metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_id is not None
            assert chunk.metadata.word_count > 0
            assert chunk.metadata.strategy_used in ChunkingStrategy
            assert chunk.metadata.semantic_coherence >= 0.0
            assert chunk.metadata.semantic_coherence <= 1.0
            
            # Check positional metadata
            if i > 0:
                assert chunk.metadata.start_position >= chunks[i-1].metadata.end_position
        
        # Validate chunk size consistency
        word_counts = [chunk.metadata.word_count for chunk in chunks]
        avg_word_count = np.mean(word_counts)
        variance = np.var(word_counts) / avg_word_count if avg_word_count > 0 else 0
        assert variance <= QUALITY_THRESHOLDS['max_chunk_size_variance']
    
    async def test_semantic_chunking(self, adaptive_chunking_system):
        """Test semantic chunking with coherence validation."""
        # Create content with clear semantic boundaries
        test_content = """
        # Introduction to Casino Gaming
        
        Casino gaming has evolved significantly over the past decades. Modern casinos offer
        a wide variety of entertainment options for players of all skill levels.
        
        # Popular Casino Games
        
        Slot machines remain the most popular choice among casino visitors. These games
        require no skill and offer instant gratification with colorful themes and sounds.
        
        Table games like blackjack and poker require strategy and skill. Professional
        players often spend years mastering these complex games.
        
        # Responsible Gaming
        
        It's important to always gamble responsibly. Set limits before you start playing
        and stick to them. Never gamble money you cannot afford to lose.
        """
        
        chunks = await adaptive_chunking_system.process_content(
            content=test_content,
            content_id="test_semantic_chunking"
        )
        
        # Should create chunks based on semantic boundaries
        assert len(chunks) >= 3  # At least one chunk per section
        
        # Check semantic coherence scores
        coherence_scores = [chunk.metadata.semantic_coherence for chunk in chunks]
        avg_coherence = np.mean(coherence_scores)
        assert avg_coherence >= QUALITY_THRESHOLDS['min_chunk_coherence_score']
        
        # Verify each chunk has reasonable content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 10
            assert chunk.metadata.sentence_count > 0
    
    async def test_adaptive_strategy_selection(self, adaptive_chunking_system, test_content_samples):
        """Test that adaptive chunking selects appropriate strategies."""
        strategy_selections = defaultdict(int)
        
        for sample in test_content_samples[:10]:
            chunks = await adaptive_chunking_system.process_content(
                content=sample.content,
                content_id=f"test_adaptive_{sample.expected_type.value}"
            )
            
            if chunks:
                strategy_used = chunks[0].metadata.strategy_used
                strategy_selections[strategy_used] += 1
        
        # Should use multiple strategies
        assert len(strategy_selections) >= 2
        
        # Structural chunking should be used for technical docs
        tech_samples = [sample for sample in test_content_samples 
                       if sample.expected_type == ContentType.TECHNICAL_DOCS]
        
        if tech_samples:
            sample = tech_samples[0]
            chunks = await adaptive_chunking_system.process_content(
                content=sample.content,
                content_id="test_structural"
            )
            
            # Technical docs should prefer structural chunking
            if chunks and '##' in sample.content:
                structural_chunks = [c for c in chunks 
                                   if c.metadata.strategy_used == ChunkingStrategy.STRUCTURAL]
                assert len(structural_chunks) > 0
    
    async def test_chunk_overlap_handling(self, adaptive_chunking_system):
        """Test proper handling of chunk overlap."""
        test_content = "This is a test document. " * 100  # Repetitive content for clear testing
        
        chunks = await adaptive_chunking_system.process_content(
            content=test_content,
            content_id="test_overlap"
        )
        
        if len(chunks) > 1:
            # Check overlap metadata
            for i in range(1, len(chunks)):
                current_chunk = chunks[i]
                previous_chunk = chunks[i-1]
                
                # Should have some overlap with previous chunk
                assert current_chunk.metadata.overlap_with_previous >= 0
                
                # Check that overlap is reasonable
                assert current_chunk.metadata.overlap_with_previous <= len(previous_chunk.content)
    
    async def test_importance_scoring(self, adaptive_chunking_system):
        """Test chunk importance scoring functionality."""
        # Content with varying importance indicators
        test_content = """
        # Introduction
        This is a basic introduction paragraph with general information.
        
        ## Important Section
        **This is a critical section** with emphasis and important details.
        Key points include:
        - First important point
        - Second critical detail
        
        ### API Example
        ```python
        def important_function():
            return "This is code"
        ```
        
        Conclusion paragraph with summary information.
        """
        
        chunks = await adaptive_chunking_system.process_content(
            content=test_content,
            content_id="test_importance"
        )
        
        # Should assign different importance scores
        importance_scores = [chunk.metadata.importance_score for chunk in chunks]
        
        # Should have variety in importance scores
        assert max(importance_scores) > min(importance_scores)
        
        # Headers and emphasized content should have higher importance
        for chunk in chunks:
            if '**' in chunk.content or '#' in chunk.content:
                assert chunk.metadata.importance_score >= 0.3
    
    @pytest.mark.performance
    async def test_chunking_performance(self, adaptive_chunking_system, test_content_samples):
        """Test chunking system performance under load."""
        start_time = time.time()
        
        # Process multiple content samples
        tasks = []
        for i, sample in enumerate(test_content_samples[:5]):
            task = adaptive_chunking_system.process_content(
                content=sample.content,
                content_id=f"perf_test_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Validate all results
        total_chunks = 0
        for chunks in results:
            assert chunks is not None
            total_chunks += len(chunks)
        
        # Check performance metrics
        performance_metrics = adaptive_chunking_system.get_performance_metrics()
        assert performance_metrics['total_processed'] >= 5
        assert performance_metrics['avg_processing_time'] < PIPELINE_PERFORMANCE_THRESHOLDS['max_chunking_time_ms'] / 1000
        
        # Check throughput
        throughput = total_chunks / total_time
        assert throughput >= 10.0  # At least 10 chunks per second


# ============================================================================
# INTEGRATION TESTS: ENHANCED FTI PIPELINE
# ============================================================================

@pytest.mark.integration
@pytest.mark.content_pipeline
class TestEnhancedFTIPipelineIntegration:
    """Integration tests for the complete Enhanced FTI Pipeline."""
    
    async def test_feature_extraction_pipeline(self, fti_orchestrator, test_content_samples):
        """Test the feature extraction stage of the FTI pipeline."""
        sample = test_content_samples[0]
        
        # Extract features
        feature_data = await fti_orchestrator.extract_features(
            content=sample.content,
            content_id="test_feature_extraction"
        )
        
        # Validate feature data structure
        assert isinstance(feature_data, EnhancedFeatureData)
        assert feature_data.content_id == "test_feature_extraction"
        assert len(feature_data.contextual_embeddings) > 0
        assert len(feature_data.contextual_chunks) > 0
        assert feature_data.content_type is not None
        assert isinstance(feature_data.metadata, dict)
        assert isinstance(feature_data.processing_metrics, dict)
        
        # Validate Task 3 integration flags
        assert isinstance(feature_data.hybrid_search_enabled, bool)
        assert isinstance(feature_data.mmr_applied, bool)
        assert feature_data.context_window_used > 0
        
        # Check processing metrics
        metrics = feature_data.processing_metrics
        assert 'processing_time' in metrics
        assert 'chunk_count' in metrics
        assert 'embedding_time' in metrics
    
    async def test_training_optimization_pipeline(self, fti_orchestrator):
        """Test the training/optimization stage of the FTI pipeline."""
        # Create sample training data
        training_samples = [
            ("What are the best casino bonuses?", "Comprehensive analysis of top casino bonuses..."),
            ("How to play blackjack?", "Complete blackjack strategy guide..."),
            ("Casino review methodology", "Our review process includes multiple factors...")
        ]
        
        # Run training optimization
        training_metrics = await fti_orchestrator.optimize_parameters(
            training_samples=training_samples,
            optimization_target="response_quality"
        )
        
        # Validate training results
        assert isinstance(training_metrics, TrainingMetrics)
        assert isinstance(training_metrics.prompt_optimization_results, dict)
        assert isinstance(training_metrics.parameter_optimization, dict)
        assert isinstance(training_metrics.best_configuration, dict)
        assert training_metrics.training_duration > 0
        
        # Check optimization results
        param_optimization = training_metrics.parameter_optimization
        assert 'dense_weight' in param_optimization
        assert 'mmr_lambda' in param_optimization
        assert 'context_window_size' in param_optimization
    
    async def test_inference_pipeline_end_to_end(self, fti_orchestrator):
        """Test complete inference pipeline with all integrations."""
        test_query = "What are the most trustworthy online casinos with the best bonuses?"
        
        # Run complete inference
        result = await fti_orchestrator.generate_response(query=test_query)
        
        # Validate inference result structure
        assert isinstance(result, EnhancedInferenceResult)
        assert result.content is not None
        assert len(result.content) > 50  # Should generate substantial content
        
        # Validate Task 2 integration (Enhanced RAG Response)
        if result.enhanced_response:
            assert result.enhanced_response.content == result.content
            assert result.enhanced_response.confidence_score >= 0.0
            assert result.enhanced_response.source_quality_score >= 0.0
        
        # Validate Task 3 integration (Contextual Sources)
        assert isinstance(result.contextual_sources, list)
        if result.contextual_sources:
            for source in result.contextual_sources:
                assert isinstance(source, Document)
                assert source.page_content is not None
        
        # Validate confidence scoring
        assert 0.0 <= result.confidence_score <= 1.0
        
        # Validate performance metrics
        assert isinstance(result.performance_metrics, dict)
        assert 'response_time' in result.performance_metrics
        assert 'retrieval_time' in result.performance_metrics
        
        # Check cache metadata if caching is enabled
        if result.cache_metadata:
            assert 'cache_hit' in result.cache_metadata
            assert 'cache_key' in result.cache_metadata
    
    async def test_pipeline_stage_switching(self, fti_orchestrator):
        """Test switching between different pipeline stages."""
        # Test each pipeline stage
        stages_to_test = [PipelineStage.FEATURE, PipelineStage.TRAINING, PipelineStage.INFERENCE]
        
        for stage in stages_to_test:
            fti_orchestrator.current_stage = stage
            
            if stage == PipelineStage.FEATURE:
                # Feature extraction should work
                result = await fti_orchestrator.extract_features(
                    content="Test content for feature extraction",
                    content_id="stage_test"
                )
                assert isinstance(result, EnhancedFeatureData)
            
            elif stage == PipelineStage.INFERENCE:
                # Inference should work
                result = await fti_orchestrator.generate_response(
                    query="Test query for inference"
                )
                assert isinstance(result, EnhancedInferenceResult)
    
    async def test_error_handling_and_fallbacks(self, fti_orchestrator):
        """Test error handling and fallback mechanisms."""
        # Test with problematic content
        problematic_queries = [
            "",  # Empty query
            "A" * 10000,  # Very long query
            "🎰🎲🃏" * 100,  # Special characters
            None  # None input
        ]
        
        for query in problematic_queries:
            try:
                if query is None:
                    # Should handle None gracefully
                    with pytest.raises((ValueError, TypeError)):
                        await fti_orchestrator.generate_response(query=query)
                else:
                    result = await fti_orchestrator.generate_response(query=query)
                    
                    # Should return valid result even for edge cases
                    assert isinstance(result, EnhancedInferenceResult)
                    assert result.content is not None
                    
            except Exception as e:
                # Should log errors gracefully
                assert "error" in str(e).lower() or "invalid" in str(e).lower()
    
    @pytest.mark.performance
    async def test_pipeline_performance_integration(self, fti_orchestrator, test_content_samples):
        """Test overall pipeline performance with realistic workload."""
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Process multiple samples through different stages
        tasks = []
        
        # Feature extraction tasks
        for i, sample in enumerate(test_content_samples[:3]):
            task = fti_orchestrator.extract_features(
                content=sample.content,
                content_id=f"perf_test_{i}"
            )
            tasks.append(task)
        
        # Inference tasks
        test_queries = [
            "What are the best casino games?",
            "How do casino bonuses work?",
            "What makes a casino trustworthy?"
        ]
        
        for query in test_queries:
            task = fti_orchestrator.generate_response(query=query)
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        assert memory_increase < PIPELINE_PERFORMANCE_THRESHOLDS['max_memory_usage_mb']
        
        # Validate all results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= len(tasks) * 0.8  # At least 80% success rate
        
        # Check overall throughput
        throughput = len(successful_results) / total_time
        assert throughput >= PIPELINE_PERFORMANCE_THRESHOLDS['min_throughput_docs_per_sec'] 


# ============================================================================
# QUALITY VALIDATION TESTS
# ============================================================================

@pytest.mark.quality
@pytest.mark.content_pipeline
class TestContentProcessingQuality:
    """Quality validation tests for content processing pipeline."""
    
    async def test_content_type_accuracy_validation(self, content_type_detector, test_content_samples):
        """Validate content type detection accuracy across different types."""
        type_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for sample in test_content_samples:
            result = await content_type_detector.detect_content_type(
                content=sample.content,
                filename=f"test_{sample.expected_type.value}.txt"
            )
            
            content_type = sample.expected_type
            type_accuracy[content_type]['total'] += 1
            
            if result.content_type.value == content_type.value:
                type_accuracy[content_type]['correct'] += 1
        
        # Calculate accuracy for each content type
        for content_type, stats in type_accuracy.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                # Should meet minimum accuracy threshold for each type
                assert accuracy >= QUALITY_THRESHOLDS['min_content_detection_confidence']
    
    async def test_chunking_quality_consistency(self, adaptive_chunking_system, test_content_samples):
        """Test consistency and quality of chunking across different content types."""
        quality_metrics = []
        
        for sample in test_content_samples[:10]:
            chunks = await adaptive_chunking_system.process_content(
                content=sample.content,
                content_id=f"quality_test_{sample.expected_type.value}"
            )
            
            if chunks:
                # Calculate quality metrics
                coherence_scores = [chunk.metadata.semantic_coherence for chunk in chunks]
                avg_coherence = np.mean(coherence_scores)
                
                word_counts = [chunk.metadata.word_count for chunk in chunks]
                size_consistency = 1 - (np.std(word_counts) / np.mean(word_counts)) if np.mean(word_counts) > 0 else 0
                
                quality_score = (avg_coherence + size_consistency) / 2
                quality_metrics.append(quality_score)
        
        # Overall quality should meet threshold
        if quality_metrics:
            overall_quality = np.mean(quality_metrics)
            assert overall_quality >= QUALITY_THRESHOLDS['min_chunking_quality_score']
    
    async def test_pipeline_integration_quality(self, fti_orchestrator):
        """Test quality of integrated pipeline components."""
        test_queries = [
            "What are the most reliable casino review sources?",
            "How do progressive jackpot slots work?",
            "What security measures do top casinos use?",
            "Compare different types of casino bonuses",
            "Explain responsible gambling practices"
        ]
        
        quality_scores = []
        
        for query in test_queries:
            result = await fti_orchestrator.generate_response(query=query)
            
            # Quality indicators
            content_length = len(result.content.split())
            confidence_score = result.confidence_score
            
            # Response should be substantial and confident
            length_quality = min(content_length / 100, 1.0)  # Scale to 0-1
            
            overall_quality = (length_quality + confidence_score) / 2
            quality_scores.append(overall_quality)
        
        # Average quality should meet threshold
        avg_quality = np.mean(quality_scores)
        assert avg_quality >= QUALITY_THRESHOLDS['min_chunking_quality_score']
    
    async def test_source_quality_integration(self, fti_orchestrator):
        """Test integration with source quality analysis from Task 2."""
        query = "Comprehensive analysis of online casino security features"
        
        result = await fti_orchestrator.generate_response(query=query)
        
        # Should integrate source quality analysis
        if result.enhanced_response:
            # Source quality should be analyzed
            assert result.enhanced_response.source_quality_score >= 0.0
            assert result.enhanced_response.source_quality_score <= 1.0
            
            # Should have source quality metadata
            metadata = result.enhanced_response.metadata
            if 'source_quality_analysis' in metadata:
                quality_analysis = metadata['source_quality_analysis']
                assert isinstance(quality_analysis, dict)
                assert 'overall_score' in quality_analysis


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

@pytest.mark.edge_cases
@pytest.mark.content_pipeline
class TestContentProcessingEdgeCases:
    """Test edge cases and error handling for content processing pipeline."""
    
    async def test_empty_content_handling(self, content_type_detector, adaptive_chunking_system):
        """Test handling of empty or minimal content."""
        edge_cases = ["", " ", "\n", "\t", "A", "🎰"]
        
        for content in edge_cases:
            # Content type detection should handle gracefully
            type_result = await content_type_detector.detect_content_type(content=content)
            assert type_result is not None
            assert type_result.content_type in ContentType
            
            # Chunking should handle gracefully
            chunks = await adaptive_chunking_system.process_content(
                content=content,
                content_id=f"edge_case_{hash(content)}"
            )
            assert isinstance(chunks, list)
            
            # For empty content, should return empty chunks or single minimal chunk
            if not content.strip():
                assert len(chunks) <= 1
    
    async def test_large_content_handling(self, adaptive_chunking_system, fti_orchestrator):
        """Test handling of very large content."""
        # Create large content (simulate a very long document)
        large_content = "This is a long casino review. " * 1000  # ~30k characters
        
        # Chunking should handle large content efficiently
        start_time = time.time()
        chunks = await adaptive_chunking_system.process_content(
            content=large_content,
            content_id="large_content_test"
        )
        processing_time = time.time() - start_time
        
        # Should process within reasonable time
        assert processing_time < 10.0  # 10 seconds max
        assert len(chunks) > 1  # Should be chunked
        
        # Each chunk should be reasonable size
        for chunk in chunks:
            assert chunk.metadata.word_count <= 1500  # Reasonable chunk size
    
    async def test_special_characters_handling(self, content_type_detector, adaptive_chunking_system):
        """Test handling of content with special characters and encoding."""
        special_content_samples = [
            "Casino review with émojis 🎰🎲 and spëcial chäracteŕs",
            "Русский текст о казино и игровых автоматах",
            "中文赌场评论内容",
            "Content with\ttabs\nand\r\nnewlines",
            "HTML <b>bold</b> and <i>italic</i> content",
            "JSON {\"rating\": 4.5, \"bonus\": \"$1000\"} embedded content"
        ]
        
        for content in special_content_samples:
            # Should detect content type without errors
            type_result = await content_type_detector.detect_content_type(content=content)
            assert type_result is not None
            
            # Should chunk without errors
            chunks = await adaptive_chunking_system.process_content(
                content=content,
                content_id=f"special_chars_{hash(content)}"
            )
            assert isinstance(chunks, list)
            
            # Content should be preserved
            if chunks:
                total_content = " ".join(chunk.content for chunk in chunks)
                # Basic content preservation check (allowing for some processing)
                assert len(total_content) > len(content) * 0.8
    
    async def test_concurrent_processing_safety(self, fti_orchestrator):
        """Test thread safety with concurrent processing."""
        test_queries = [
            f"Casino query {i}: What are the best bonus offers?"
            for i in range(20)
        ]
        
        # Process all queries concurrently
        tasks = [
            fti_orchestrator.generate_response(query=query)
            for query in test_queries
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Should handle concurrent processing without errors
        successful_results = [r for r in results if not isinstance(r, Exception)]
        error_results = [r for r in results if isinstance(r, Exception)]
        
        # Most results should be successful
        success_rate = len(successful_results) / len(results)
        assert success_rate >= QUALITY_THRESHOLDS['min_pipeline_success_rate']
        
        # Should complete in reasonable time
        assert total_time < 30.0  # 30 seconds max for 20 concurrent queries
    
    async def test_memory_cleanup_and_resource_management(self, fti_orchestrator, test_content_samples):
        """Test proper memory cleanup and resource management."""
        import gc
        
        # Baseline memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple large operations
        for i in range(10):
            sample = test_content_samples[i % len(test_content_samples)]
            
            # Feature extraction
            feature_data = await fti_orchestrator.extract_features(
                content=sample.content * 5,  # Make it larger
                content_id=f"memory_test_{i}"
            )
            
            # Inference
            result = await fti_orchestrator.generate_response(
                query=f"Test query {i} about casino features"
            )
            
            # Force garbage collection
            del feature_data, result
            gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < PIPELINE_PERFORMANCE_THRESHOLDS['max_memory_usage_mb'] * 2


# ============================================================================
# CLEANUP AND CONFIGURATION TESTS
# ============================================================================

@pytest.mark.content_pipeline
class TestContentProcessingCleanup:
    """Test cleanup and configuration validation."""
    
    def test_configuration_validation(self, fti_config):
        """Test FTI configuration validation."""
        # Required fields should be present
        assert fti_config.chunk_size > 0
        assert fti_config.chunk_overlap >= 0
        assert fti_config.chunk_overlap < fti_config.chunk_size
        
        # Weights should sum to 1.0
        assert abs((fti_config.dense_weight + fti_config.sparse_weight) - 1.0) < 0.01
        
        # Lambda should be in valid range
        assert 0.0 <= fti_config.mmr_lambda <= 1.0
        
        # Cache TTL should be positive
        assert fti_config.cache_ttl_hours > 0
    
    async def test_system_resource_cleanup(self, content_type_detector, adaptive_chunking_system):
        """Test proper cleanup of system resources."""
        # Use the systems
        await content_type_detector.detect_content_type("Test content")
        await adaptive_chunking_system.process_content("Test content", "cleanup_test")
        
        # Clear caches
        content_type_detector.clear_cache()
        adaptive_chunking_system.reset_metrics()
        
        # Verify cleanup
        cache_stats = content_type_detector.get_cache_stats()
        assert cache_stats['total_entries'] == 0
        
        performance_metrics = adaptive_chunking_system.get_performance_metrics()
        assert performance_metrics['total_processed'] == 0
    
    def test_performance_threshold_validation(self):
        """Validate that performance thresholds are reasonable."""
        # Time thresholds should be positive
        assert PIPELINE_PERFORMANCE_THRESHOLDS['max_content_detection_time_ms'] > 0
        assert PIPELINE_PERFORMANCE_THRESHOLDS['max_chunking_time_ms'] > 0
        assert PIPELINE_PERFORMANCE_THRESHOLDS['max_feature_extraction_time_ms'] > 0
        assert PIPELINE_PERFORMANCE_THRESHOLDS['max_inference_time_ms'] > 0
        
        # Quality thresholds should be in valid ranges
        assert 0.0 <= QUALITY_THRESHOLDS['min_content_detection_confidence'] <= 1.0
        assert 0.0 <= QUALITY_THRESHOLDS['min_chunk_coherence_score'] <= 1.0
        assert 0.0 <= QUALITY_THRESHOLDS['min_semantic_similarity'] <= 1.0
        assert 0.0 <= QUALITY_THRESHOLDS['min_pipeline_success_rate'] <= 1.0
        
        # Memory threshold should be reasonable
        assert PIPELINE_PERFORMANCE_THRESHOLDS['max_memory_usage_mb'] > 50  # At least 50MB
        assert PIPELINE_PERFORMANCE_THRESHOLDS['max_memory_usage_mb'] < 1000  # Less than 1GB


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest for content processing pipeline tests."""
    # Add custom markers
    config.addinivalue_line("markers", "content_pipeline: Content processing pipeline tests")
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "quality: Quality validation tests")
    config.addinivalue_line("markers", "edge_cases: Edge case tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for better organization."""
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to pipeline tests
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def log_test_info(request):
    """Automatically log test information."""
    test_name = request.node.name
    test_class = request.node.cls.__name__ if request.node.cls else "Global"
    
    logging.info(f"Starting test: {test_class}::{test_name}")
    
    def log_finish():
        logging.info(f"Finished test: {test_class}::{test_name}")
    
    request.addfinalizer(log_finish) 