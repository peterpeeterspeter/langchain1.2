"""
Task 3.8: Comprehensive Testing Framework
Universal RAG CMS - Contextual Retrieval System Tests

This module provides comprehensive unit and integration tests for all components
of the contextual retrieval system (Tasks 3.1-3.7).
"""

import pytest
import asyncio
import json
import time
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
import uuid

# Test fixtures and setup
try:
    from src.retrieval.contextual_retrieval import (
        ContextualRetrievalSystem, ContextualEmbeddingSystem, HybridSearchEngine,
        MultiQueryRetrieval, SelfQueryRetrieval, MaximalMarginalRelevance,
        RetrievalConfig, RetrievalStrategy, ContextualChunk
    )
    from src.optimization.retrieval_optimizer import (
        RetrievalOptimizer, OptimizationStrategy, PerformanceMetrics,
        ConnectionPoolManager, CacheOptimizer
    )
    from langchain_core.documents import Document
    CONTEXTUAL_RETRIEVAL_AVAILABLE = True
except ImportError:
    CONTEXTUAL_RETRIEVAL_AVAILABLE = False
    pytest.skip("Contextual retrieval system not available", allow_module_level=True)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="Bitcoin is a decentralized digital currency that operates without a central bank.",
            metadata={
                "title": "Introduction to Bitcoin",
                "source": "crypto-guide.com",
                "category": "cryptocurrency",
                "published_date": "2024-01-15"
            }
        ),
        Document(
            page_content="Poker is a card game that combines skill, strategy, and chance.",
            metadata={
                "title": "Poker Basics",
                "source": "poker-pro.net",
                "category": "games",
                "published_date": "2024-02-01"
            }
        ),
        Document(
            page_content="Machine learning algorithms can predict patterns in large datasets.",
            metadata={
                "title": "ML Predictions",
                "source": "ai-research.org",
                "category": "technology",
                "published_date": "2024-01-20"
            }
        )
    ]

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    mock_embeddings = Mock()
    mock_embeddings.embed_documents = AsyncMock(return_value=[
        [0.1] * 1536,  # Mock 1536-dimensional embeddings
        [0.2] * 1536,
        [0.3] * 1536
    ])
    mock_embeddings.embed_query = AsyncMock(return_value=[0.15] * 1536)
    return mock_embeddings

@pytest.fixture
def retrieval_config():
    """Default retrieval configuration for testing."""
    return RetrievalConfig(
        dense_weight=0.7,
        sparse_weight=0.3,
        context_window_size=2,
        mmr_lambda=0.7,
        max_query_variations=3,
        cache_ttl_hours=24,
        enable_caching=True,
        parallel_retrieval=True,
        max_workers=2
    )

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock_llm = Mock()
    mock_llm.ainvoke = AsyncMock(return_value="What are the best cryptocurrency investment strategies?")
    return mock_llm

@pytest.fixture
def mock_vectorstore():
    """Mock vector store for testing."""
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search_with_score = AsyncMock(return_value=[
        (Document(page_content="Test content 1", metadata={"id": "1"}), 0.9),
        (Document(page_content="Test content 2", metadata={"id": "2"}), 0.8),
        (Document(page_content="Test content 3", metadata={"id": "3"}), 0.7)
    ])
    return mock_vectorstore

@pytest.fixture
async def contextual_retrieval_system(retrieval_config, mock_embeddings, mock_llm, mock_vectorstore):
    """Create a contextual retrieval system for testing."""
    return ContextualRetrievalSystem(
        vectorstore=mock_vectorstore,
        embeddings=mock_embeddings,
        llm=mock_llm,
        config=retrieval_config
    )


# ============================================================================
# UNIT TESTS - TASK 3.1: CONTEXTUAL EMBEDDING SYSTEM
# ============================================================================

class TestContextualEmbeddingSystem:
    """Test suite for contextual embedding system."""
    
    def test_contextual_chunk_creation(self, retrieval_config):
        """Test contextual chunk creation with metadata."""
        embedding_system = ContextualEmbeddingSystem(retrieval_config)
        
        document = Document(
            page_content="Full document content with multiple sentences. This is the middle part. And this is the end.",
            metadata={"title": "Test Document", "source": "test.com"}
        )
        
        chunks = ["This is the middle part."]
        chunk_metadata = [{"chunk_index": 1}]
        
        contextual_chunks = embedding_system.create_contextual_chunks(
            document, chunks, chunk_metadata
        )
        
        assert len(contextual_chunks) == 1
        chunk = contextual_chunks[0]
        
        assert chunk.text == "This is the middle part."
        assert "Full document content" in chunk.context
        assert "Test Document" in chunk.full_text
        assert chunk.metadata["title"] == "Test Document"
        assert chunk.chunk_index == 1
    
    def test_context_window_extraction(self, retrieval_config):
        """Test context window extraction logic."""
        embedding_system = ContextualEmbeddingSystem(retrieval_config)
        
        sentences = [
            "First sentence.",
            "Second sentence.",
            "Target sentence.",
            "Fourth sentence.",
            "Fifth sentence."
        ]
        
        context = embedding_system._extract_context_window(sentences, 2, 2)
        
        assert "First sentence" in context
        assert "Second sentence" in context
        assert "Target sentence" in context
        assert "Fourth sentence" in context
        assert "Fifth sentence" in context
    
    @pytest.mark.asyncio
    async def test_contextual_embedding_generation(self, retrieval_config, mock_embeddings):
        """Test contextual embedding generation."""
        embedding_system = ContextualEmbeddingSystem(retrieval_config)
        embedding_system.embeddings = mock_embeddings
        
        contextual_chunk = ContextualChunk(
            text="Test chunk",
            context="Test context",
            chunk_index=0,
            document_id="test-doc",
            metadata={"title": "Test"}
        )
        
        await embedding_system.generate_contextual_embeddings([contextual_chunk])
        
        # Verify embeddings were called with full contextualized text
        mock_embeddings.embed_documents.assert_called_once()
        call_args = mock_embeddings.embed_documents.call_args[0][0]
        assert "Test chunk" in call_args[0]
        assert "Test context" in call_args[0]


# ============================================================================
# UNIT TESTS - TASK 3.2: HYBRID SEARCH ENGINE
# ============================================================================

class TestHybridSearchEngine:
    """Test suite for hybrid search engine."""
    
    @pytest.mark.asyncio
    async def test_dense_search(self, retrieval_config, mock_vectorstore, mock_embeddings):
        """Test dense vector search functionality."""
        hybrid_engine = HybridSearchEngine(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
            config=retrieval_config
        )
        
        query = "test query"
        results = await hybrid_engine.dense_search(query, k=5)
        
        assert len(results) == 3  # Based on mock data
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Verify vector store was called correctly
        mock_vectorstore.similarity_search_with_score.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sparse_search(self, retrieval_config, mock_vectorstore):
        """Test sparse (BM25) search functionality."""
        hybrid_engine = HybridSearchEngine(
            vectorstore=mock_vectorstore,
            embeddings=Mock(),
            config=retrieval_config
        )
        
        # Mock the BM25 search results
        with patch.object(hybrid_engine, '_bm25_search') as mock_bm25:
            mock_bm25.return_value = [
                (Document(page_content="BM25 result 1", metadata={"id": "bm25_1"}), 0.8),
                (Document(page_content="BM25 result 2", metadata={"id": "bm25_2"}), 0.6)
            ]
            
            query = "test sparse query"
            results = await hybrid_engine.sparse_search(query, k=5)
            
            assert len(results) == 2
            assert results[0][1] >= results[1][1]  # Results should be sorted by score
    
    @pytest.mark.asyncio
    async def test_hybrid_search_combination(self, retrieval_config, mock_vectorstore, mock_embeddings):
        """Test hybrid search combining dense and sparse results."""
        hybrid_engine = HybridSearchEngine(
            vectorstore=mock_vectorstore,
            embeddings=mock_embeddings,
            config=retrieval_config
        )
        
        # Mock both search methods
        with patch.object(hybrid_engine, 'dense_search') as mock_dense, \
             patch.object(hybrid_engine, 'sparse_search') as mock_sparse:
            
            mock_dense.return_value = [
                (Document(page_content="Dense result", metadata={"id": "dense_1"}), 0.9)
            ]
            mock_sparse.return_value = [
                (Document(page_content="Sparse result", metadata={"id": "sparse_1"}), 0.7)
            ]
            
            query = "hybrid test query"
            results = await hybrid_engine.hybrid_search(query, k=5)
            
            assert len(results) <= 5
            assert all(isinstance(doc, Document) for doc, _ in results)
            
            # Verify both search methods were called
            mock_dense.assert_called_once()
            mock_sparse.assert_called_once()
    
    def test_score_combination_logic(self, retrieval_config):
        """Test the logic for combining dense and sparse scores."""
        hybrid_engine = HybridSearchEngine(
            vectorstore=Mock(),
            embeddings=Mock(),
            config=retrieval_config
        )
        
        dense_score = 0.8
        sparse_score = 0.6
        
        combined_score = hybrid_engine._combine_scores(
            dense_score, sparse_score, 
            retrieval_config.dense_weight, retrieval_config.sparse_weight
        )
        
        expected_score = (0.8 * 0.7) + (0.6 * 0.3)
        assert abs(combined_score - expected_score) < 0.001


# ============================================================================
# UNIT TESTS - TASK 3.3: MULTI-QUERY RETRIEVAL
# ============================================================================

class TestMultiQueryRetrieval:
    """Test suite for multi-query retrieval system."""
    
    @pytest.mark.asyncio
    async def test_query_generation(self, retrieval_config, mock_llm):
        """Test query variation generation."""
        multi_query = MultiQueryRetrieval(llm=mock_llm, config=retrieval_config)
        
        # Mock LLM to return multiple query variations
        mock_llm.ainvoke.return_value = """
        1. What are the best cryptocurrency investment strategies?
        2. How to invest in digital currencies safely?
        3. Which crypto assets have the highest returns?
        """
        
        original_query = "best crypto investments"
        variations = await multi_query.generate_query_variations(original_query)
        
        assert len(variations) >= 1
        assert all(isinstance(variation, str) for variation in variations)
        assert original_query not in variations  # Should be different from original
    
    @pytest.mark.asyncio
    async def test_parallel_retrieval(self, retrieval_config, mock_llm):
        """Test parallel retrieval execution."""
        multi_query = MultiQueryRetrieval(llm=mock_llm, config=retrieval_config)
        
        # Mock retrieval method
        async def mock_retrieve(query):
            await asyncio.sleep(0.01)  # Simulate async work
            return [
                Document(page_content=f"Result for {query}", metadata={"query": query, "id": str(uuid.uuid4())})
            ]
        
        multi_query.base_retriever = Mock()
        multi_query.base_retriever.aget_relevant_documents = mock_retrieve
        
        queries = ["query1", "query2", "query3"]
        start_time = time.time()
        results = await multi_query.parallel_retrieve(queries)
        end_time = time.time()
        
        # Should be faster than sequential execution
        assert (end_time - start_time) < 0.1  # Should complete quickly with mocked async
        assert len(results) == 3
        assert all(isinstance(docs, list) for docs in results)
    
    @pytest.mark.asyncio
    async def test_result_deduplication(self, retrieval_config, mock_llm):
        """Test deduplication of retrieval results."""
        multi_query = MultiQueryRetrieval(llm=mock_llm, config=retrieval_config)
        
        # Create duplicate documents
        doc1 = Document(page_content="Unique content 1", metadata={"id": "1"})
        doc2 = Document(page_content="Unique content 2", metadata={"id": "2"})
        doc3 = Document(page_content="Unique content 1", metadata={"id": "1"})  # Duplicate
        
        all_results = [[doc1, doc2], [doc2, doc3], [doc1]]
        
        deduplicated = multi_query.deduplicate_results(all_results)
        
        # Should have only unique documents
        assert len(deduplicated) == 2
        ids = [doc.metadata.get("id") for doc in deduplicated]
        assert len(set(ids)) == 2  # All unique IDs


# ============================================================================
# UNIT TESTS - TASK 3.4: SELF-QUERY RETRIEVAL
# ============================================================================

class TestSelfQueryRetrieval:
    """Test suite for self-query metadata filtering."""
    
    def test_filter_pattern_matching(self, retrieval_config):
        """Test pattern matching for filter extraction."""
        self_query = SelfQueryRetrieval(llm=Mock(), config=retrieval_config)
        
        # Test various filter patterns
        test_cases = [
            ("recent casino reviews", {"created_at": {"gte": "recent"}}),
            ("high quality poker guides", {"quality_score": {"gte": "high"}}),
            ("tutorials from last month", {"created_at": {"gte": "last_month"}}),
            ("best rated slot games", {"rating": {"gte": "best"}})
        ]
        
        for query, expected_structure in test_cases:
            filters = self_query.extract_filters_from_query(query)
            
            # Check that filters were extracted
            assert len(filters) > 0, f"No filters extracted for query: {query}"
    
    @pytest.mark.asyncio
    async def test_llm_filter_extraction(self, retrieval_config, mock_llm):
        """Test LLM-based filter extraction."""
        self_query = SelfQueryRetrieval(llm=mock_llm, config=retrieval_config)
        
        # Mock LLM response with structured filter
        mock_llm.ainvoke.return_value = '{"content_type": "review", "rating": {"gte": 4}, "category": "casino"}'
        
        query = "Show me highly rated casino reviews"
        filters = await self_query.llm_extract_filters(query)
        
        assert isinstance(filters, dict)
        assert "content_type" in filters or "rating" in filters
    
    def test_filter_application(self, retrieval_config, sample_documents):
        """Test application of extracted filters to documents."""
        self_query = SelfQueryRetrieval(llm=Mock(), config=retrieval_config)
        
        # Test filter that should match some documents
        filters = {"category": "games"}
        
        filtered_docs = self_query.apply_filters(sample_documents, filters)
        
        # Should only return documents matching the filter
        assert len(filtered_docs) == 1
        assert filtered_docs[0].metadata["category"] == "games"
    
    def test_query_cleaning(self, retrieval_config):
        """Test removal of filter terms from query."""
        self_query = SelfQueryRetrieval(llm=Mock(), config=retrieval_config)
        
        original_query = "recent high quality casino reviews from last month"
        cleaned_query = self_query.clean_query(original_query)
        
        # Should remove filter terms
        assert "recent" not in cleaned_query.lower()
        assert "high quality" not in cleaned_query.lower()
        assert "last month" not in cleaned_query.lower()
        assert "casino reviews" in cleaned_query.lower()


# ============================================================================
# UNIT TESTS - TASK 3.5: MAXIMAL MARGINAL RELEVANCE
# ============================================================================

class TestMaximalMarginalRelevance:
    """Test suite for MMR implementation."""
    
    def test_mmr_calculation(self, retrieval_config):
        """Test MMR score calculation logic."""
        mmr = MaximalMarginalRelevance(config=retrieval_config)
        
        # Mock documents with embeddings
        docs = [
            Document(page_content="Doc 1", metadata={"embedding": [0.1] * 10}),
            Document(page_content="Doc 2", metadata={"embedding": [0.2] * 10}),
            Document(page_content="Doc 3", metadata={"embedding": [0.3] * 10})
        ]
        
        query_embedding = [0.15] * 10
        selected_docs = []
        
        mmr_score = mmr.calculate_mmr_score(
            candidate_doc=docs[0],
            query_embedding=query_embedding,
            selected_documents=selected_docs,
            lambda_param=0.7
        )
        
        # First document should have high MMR (no diversity penalty)
        assert mmr_score > 0.5
        
        # Add document to selected and test diversity penalty
        selected_docs.append(docs[0])
        mmr_score_2 = mmr.calculate_mmr_score(
            candidate_doc=docs[1],
            query_embedding=query_embedding,
            selected_documents=selected_docs,
            lambda_param=0.7
        )
        
        # Should consider diversity
        assert isinstance(mmr_score_2, float)
    
    def test_cosine_similarity(self, retrieval_config):
        """Test cosine similarity calculation."""
        mmr = MaximalMarginalRelevance(config=retrieval_config)
        
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]
        
        # Test orthogonal vectors
        sim1 = mmr.cosine_similarity(vec1, vec2)
        assert abs(sim1) < 0.001  # Should be ~0
        
        # Test identical vectors
        sim2 = mmr.cosine_similarity(vec1, vec3)
        assert abs(sim2 - 1.0) < 0.001  # Should be ~1
    
    @pytest.mark.asyncio
    async def test_mmr_selection(self, retrieval_config):
        """Test complete MMR document selection."""
        mmr = MaximalMarginalRelevance(config=retrieval_config)
        
        # Create diverse documents with different embeddings
        documents = []
        for i in range(5):
            embedding = [0.0] * 10
            embedding[i % 3] = 1.0  # Create some diversity
            doc = Document(
                page_content=f"Document {i}",
                metadata={"id": str(i), "embedding": embedding}
            )
            documents.append(doc)
        
        query_embedding = [0.5] * 10
        
        selected = await mmr.select_documents(
            documents=documents,
            query_embedding=query_embedding,
            k=3,
            lambda_param=0.7
        )
        
        assert len(selected) == 3
        assert all(isinstance(doc, Document) for doc in selected)
        
        # Check that diversity was considered (different embeddings selected)
        selected_embeddings = [doc.metadata["embedding"] for doc in selected]
        # Should not select identical embeddings if diversity is working
        assert len(set(str(emb) for emb in selected_embeddings)) > 1


# ============================================================================
# INTEGRATION TESTS - TASK 3.6: CONTEXTUAL RETRIEVAL SYSTEM
# ============================================================================

class TestContextualRetrievalSystemIntegration:
    """Integration tests for the complete contextual retrieval system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_retrieval(self, contextual_retrieval_system):
        """Test complete end-to-end retrieval pipeline."""
        query = "best cryptocurrency investment strategies"
        
        # Mock all the subsystems
        with patch.object(contextual_retrieval_system.contextual_embedding, 'get_contextual_embeddings') as mock_embed, \
             patch.object(contextual_retrieval_system.hybrid_search, 'hybrid_search') as mock_hybrid, \
             patch.object(contextual_retrieval_system.multi_query, 'retrieve_with_variations') as mock_multi, \
             patch.object(contextual_retrieval_system.self_query, 'retrieve_with_filters') as mock_self, \
             patch.object(contextual_retrieval_system.mmr, 'select_documents') as mock_mmr:
            
            # Setup mock returns
            mock_embed.return_value = [0.1] * 1536
            mock_hybrid.return_value = [
                (Document(page_content="Test result 1", metadata={"id": "1"}), 0.9),
                (Document(page_content="Test result 2", metadata={"id": "2"}), 0.8)
            ]
            mock_multi.return_value = [
                Document(page_content="Multi result", metadata={"id": "3"})
            ]
            mock_self.return_value = [
                Document(page_content="Self result", metadata={"id": "4"})
            ]
            mock_mmr.return_value = [
                Document(page_content="Final result", metadata={"id": "5"})
            ]
            
            # Execute retrieval
            results = await contextual_retrieval_system.aget_relevant_documents(query)
            
            # Verify results
            assert len(results) > 0
            assert all(isinstance(doc, Document) for doc in results)
            
            # Verify all components were called
            mock_embed.assert_called()
            mock_hybrid.assert_called()
    
    @pytest.mark.asyncio
    async def test_retrieval_with_different_strategies(self, contextual_retrieval_system):
        """Test retrieval with different strategies."""
        query = "poker strategy guide"
        
        strategies = [
            RetrievalStrategy.DENSE,
            RetrievalStrategy.SPARSE,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.MULTI_QUERY
        ]
        
        for strategy in strategies:
            contextual_retrieval_system.config.retrieval_strategy = strategy
            
            with patch.object(contextual_retrieval_system, '_execute_retrieval_strategy') as mock_execute:
                mock_execute.return_value = [
                    Document(page_content=f"Result for {strategy.value}", metadata={"strategy": strategy.value})
                ]
                
                results = await contextual_retrieval_system.aget_relevant_documents(query)
                
                assert len(results) > 0
                mock_execute.assert_called_with(query, strategy)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, contextual_retrieval_system):
        """Test error handling and fallback mechanisms."""
        query = "test query for error handling"
        
        # Test fallback when hybrid search fails
        with patch.object(contextual_retrieval_system.hybrid_search, 'hybrid_search') as mock_hybrid:
            mock_hybrid.side_effect = Exception("Hybrid search failed")
            
            # Should fallback to basic search
            with patch.object(contextual_retrieval_system.vectorstore, 'similarity_search') as mock_basic:
                mock_basic.return_value = [
                    Document(page_content="Fallback result", metadata={"fallback": True})
                ]
                
                results = await contextual_retrieval_system.aget_relevant_documents(query)
                
                assert len(results) > 0
                assert results[0].metadata.get("fallback") is True
    
    @pytest.mark.asyncio 
    async def test_performance_metrics_collection(self, contextual_retrieval_system):
        """Test performance metrics collection during retrieval."""
        query = "performance test query"
        
        # Enable performance tracking
        contextual_retrieval_system.config.track_performance = True
        
        with patch.object(contextual_retrieval_system, '_collect_performance_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "total_time_ms": 150,
                "cache_hit": False,
                "components_used": ["hybrid_search", "mmr"]
            }
            
            results = await contextual_retrieval_system.aget_relevant_documents(query)
            
            # Verify metrics were collected
            mock_metrics.assert_called()


# ============================================================================
# PERFORMANCE TESTS - TASK 3.7: OPTIMIZATION
# ============================================================================

class TestPerformanceOptimization:
    """Performance and optimization tests."""
    
    @pytest.mark.asyncio
    async def test_retrieval_performance_under_load(self, contextual_retrieval_system):
        """Test system performance under concurrent load."""
        queries = [f"test query {i}" for i in range(10)]
        
        # Mock the retrieval to return quickly
        with patch.object(contextual_retrieval_system, 'aget_relevant_documents') as mock_retrieve:
            mock_retrieve.return_value = [
                Document(page_content="Fast result", metadata={"id": "fast"})
            ]
            
            start_time = time.time()
            
            # Execute queries concurrently
            tasks = [contextual_retrieval_system.aget_relevant_documents(query) for query in queries]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            
            # Verify all queries completed
            assert len(results) == 10
            
            # Performance should be reasonable (concurrent execution)
            total_time = end_time - start_time
            assert total_time < 1.0  # Should complete quickly with mocks
    
    def test_memory_usage_optimization(self, contextual_retrieval_system):
        """Test memory usage stays within reasonable bounds."""
        import sys
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and process multiple large documents
        large_documents = []
        for i in range(100):
            doc = Document(
                page_content="Large document content " * 100,  # ~2.5KB per document
                metadata={"id": str(i), "size": "large"}
            )
            large_documents.append(doc)
        
        # Process documents through contextual embedding
        embedding_system = ContextualEmbeddingSystem(contextual_retrieval_system.config)
        
        chunks = ["Test chunk"] * len(large_documents)
        chunk_metadata = [{"chunk_index": i} for i in range(len(large_documents))]
        
        # This should not cause excessive memory usage
        contextual_chunks = embedding_system.create_contextual_chunks(
            large_documents[0], chunks, chunk_metadata
        )
        
        # Force garbage collection and check object count
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should be reasonable (not growing excessively)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold for object creation
    
    @pytest.mark.asyncio
    async def test_cache_performance_optimization(self, contextual_retrieval_system):
        """Test cache performance and hit rates."""
        # Enable caching
        contextual_retrieval_system.config.enable_caching = True
        
        query = "cached query test"
        
        # Mock cache system
        with patch.object(contextual_retrieval_system, 'cache_system') as mock_cache:
            mock_cache.get.return_value = None  # First call - cache miss
            mock_cache.set = AsyncMock()
            
            # First retrieval - should be cache miss
            with patch.object(contextual_retrieval_system, '_execute_retrieval') as mock_execute:
                mock_execute.return_value = [
                    Document(page_content="Cached result", metadata={"cached": False})
                ]
                
                result1 = await contextual_retrieval_system.aget_relevant_documents(query)
                
                # Should have called the actual retrieval
                mock_execute.assert_called_once()
                
                # Should have cached the result
                mock_cache.set.assert_called_once()
            
            # Second retrieval - should be cache hit
            mock_cache.get.return_value = [
                Document(page_content="Cached result", metadata={"cached": True})
            ]
            
            with patch.object(contextual_retrieval_system, '_execute_retrieval') as mock_execute2:
                result2 = await contextual_retrieval_system.aget_relevant_documents(query)
                
                # Should NOT have called actual retrieval
                mock_execute2.assert_not_called()
                
                # Should return cached result
                assert result2[0].metadata.get("cached") is True


# ============================================================================
# QUALITY VALIDATION TESTS
# ============================================================================

class TestQualityValidation:
    """Tests for result quality validation and metrics."""
    
    def test_relevance_scoring(self, sample_documents):
        """Test relevance scoring algorithms."""
        from src.optimization.retrieval_optimizer import PerformanceMetrics
        
        # Mock relevance calculation
        def calculate_relevance(query: str, documents: List[Document]) -> float:
            # Simple keyword matching for testing
            query_words = set(query.lower().split())
            relevant_docs = 0
            
            for doc in documents:
                doc_words = set(doc.page_content.lower().split())
                if query_words & doc_words:  # Intersection
                    relevant_docs += 1
            
            return relevant_docs / len(documents) if documents else 0.0
        
        # Test with relevant query
        relevant_query = "bitcoin cryptocurrency digital"
        relevance_score = calculate_relevance(relevant_query, sample_documents)
        assert relevance_score > 0.0
        
        # Test with irrelevant query
        irrelevant_query = "quantum physics molecular structure"
        irrelevance_score = calculate_relevance(irrelevant_query, sample_documents)
        assert irrelevance_score <= relevance_score
    
    def test_diversity_scoring(self, sample_documents):
        """Test diversity scoring for document sets."""
        
        def calculate_diversity(documents: List[Document]) -> float:
            if len(documents) <= 1:
                return 0.0
            
            # Simple category-based diversity
            categories = set()
            for doc in documents:
                category = doc.metadata.get("category", "unknown")
                categories.add(category)
            
            return len(categories) / len(documents)
        
        # Test with diverse documents
        diversity_score = calculate_diversity(sample_documents)
        assert diversity_score > 0.5  # Should be diverse
        
        # Test with similar documents
        similar_docs = [sample_documents[0]] * 3  # All same category
        similar_diversity = calculate_diversity(similar_docs)
        assert similar_diversity < diversity_score
    
    @pytest.mark.asyncio
    async def test_response_time_validation(self, contextual_retrieval_system):
        """Test response time validation and SLA compliance."""
        query = "response time test"
        max_response_time = 2.0  # 2 seconds SLA
        
        with patch.object(contextual_retrieval_system, '_execute_retrieval') as mock_execute:
            # Mock fast response
            async def fast_retrieval(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms
                return [Document(page_content="Fast result", metadata={"fast": True})]
            
            mock_execute.side_effect = fast_retrieval
            
            start_time = time.time()
            results = await contextual_retrieval_system.aget_relevant_documents(query)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should meet SLA
            assert response_time < max_response_time
            assert len(results) > 0


# ============================================================================
# INTEGRATION WITH TASK 2 COMPONENTS
# ============================================================================

class TestTask2Integration:
    """Test integration with existing Task 2 components."""
    
    @pytest.mark.asyncio
    async def test_confidence_scoring_integration(self, contextual_retrieval_system):
        """Test integration with enhanced confidence scoring system."""
        query = "integration test query"
        
        # Mock confidence calculator
        with patch('src.chains.enhanced_confidence_scoring_system.EnhancedConfidenceCalculator') as mock_calculator:
            mock_instance = mock_calculator.return_value
            mock_instance.calculate_enhanced_confidence = AsyncMock(return_value=(
                Mock(overall_confidence=0.85),  # ConfidenceBreakdown
                Mock(confidence_score=0.85)     # EnhancedRAGResponse
            ))
            
            # Test retrieval with confidence scoring
            results = await contextual_retrieval_system.aget_relevant_documents(query)
            
            # Should integrate confidence scores
            assert len(results) > 0
    
    @pytest.mark.asyncio 
    async def test_intelligent_cache_integration(self, contextual_retrieval_system):
        """Test integration with intelligent caching system."""
        from src.chains.enhanced_confidence_scoring_system import IntelligentCache, CacheStrategy
        
        # Create cache instance
        cache = IntelligentCache(strategy=CacheStrategy.ADAPTIVE)
        contextual_retrieval_system.cache_system = cache
        
        query = "cache integration test"
        
        # Test cache miss and set
        result1 = await cache.get(query)
        assert result1 is None  # Cache miss
        
        # Mock result to cache
        mock_response = Mock()
        mock_response.confidence_score = 0.8
        mock_response.content = "Cached content"
        
        await cache.set(query, mock_response)
        
        # Test cache hit
        result2 = await cache.get(query)
        assert result2 is not None  # Cache hit


# ============================================================================
# TEST UTILITIES AND HELPERS
# ============================================================================

class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_mock_documents(count: int, diverse: bool = True) -> List[Document]:
        """Create mock documents for testing."""
        documents = []
        
        categories = ["technology", "finance", "games", "science", "health"] if diverse else ["technology"]
        
        for i in range(count):
            category = categories[i % len(categories)]
            doc = Document(
                page_content=f"Test document {i} about {category}",
                metadata={
                    "id": str(i),
                    "category": category,
                    "title": f"Document {i}",
                    "source": f"source-{i}.com",
                    "published_date": "2024-01-01"
                }
            )
            documents.append(doc)
        
        return documents
    
    @staticmethod
    def assert_performance_metrics(metrics: PerformanceMetrics, max_time_ms: float = 1000):
        """Assert performance metrics meet requirements."""
        assert metrics.total_time_ms <= max_time_ms
        assert metrics.relevance_score >= 0.0
        assert metrics.relevance_score <= 1.0
        assert metrics.memory_usage_mb >= 0.0
        assert metrics.result_count >= 0


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance measurement."""
    
    @pytest.mark.asyncio
    async def test_retrieval_throughput_benchmark(self, contextual_retrieval_system):
        """Benchmark retrieval throughput."""
        queries = [f"benchmark query {i}" for i in range(50)]
        
        with patch.object(contextual_retrieval_system, 'aget_relevant_documents') as mock_retrieve:
            # Mock fast retrieval
            mock_retrieve.return_value = [
                Document(page_content="Benchmark result", metadata={"benchmark": True})
            ]
            
            start_time = time.time()
            
            # Execute all queries
            for query in queries:
                await contextual_retrieval_system.aget_relevant_documents(query)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate throughput
            throughput = len(queries) / total_time
            
            # Log benchmark results
            print(f"\nBenchmark Results:")
            print(f"Queries: {len(queries)}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Throughput: {throughput:.2f} queries/second")
            
            # Assert minimum performance
            assert throughput > 10  # At least 10 queries/second with mocks
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage patterns."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large number of contextual chunks
        config = RetrievalConfig()
        embedding_system = ContextualEmbeddingSystem(config)
        
        chunks = []
        for i in range(1000):
            chunk = ContextualChunk(
                text=f"Test chunk {i} with substantial content",
                context=f"Context for chunk {i}",
                chunk_index=i,
                document_id=f"doc-{i}",
                metadata={"test": True}
            )
            chunks.append(chunk)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Benchmark:")
        print(f"Initial Memory: {initial_memory:.2f} MB")
        print(f"Final Memory: {final_memory:.2f} MB")
        print(f"Memory Increase: {memory_increase:.2f} MB")
        print(f"Memory per chunk: {memory_increase / len(chunks) * 1024:.2f} KB")
        
        # Assert reasonable memory usage
        assert memory_increase < 100  # Less than 100MB for 1000 chunks


# ============================================================================
# RUN CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Configure pytest for comprehensive testing
    pytest.main([
        __file__,
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--durations=10",       # Show slowest 10 tests
        "--cov=src",           # Coverage for src directory
        "--cov-report=html",   # HTML coverage report
        "--cov-report=term",   # Terminal coverage report
        "-m", "not benchmark", # Skip benchmark tests by default
    ]) 