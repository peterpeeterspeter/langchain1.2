"""
Hybrid Search Infrastructure for Universal RAG CMS
=================================================

This module implements Task 3.2: Hybrid Search Infrastructure with BM25 Integration
Combines dense vector similarity with sparse BM25 keyword matching for optimal retrieval.

Key Features:
- Dense vector search using embeddings
- Sparse BM25 keyword matching 
- Score normalization and fusion algorithms
- Async parallel search execution
- Integration with contextual embedding system
- Performance optimization and caching
"""

import asyncio
import time
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import statistics
from collections import defaultdict, Counter

import numpy as np
from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
import rank_bm25

# Import contextual embedding system
from .contextual_embedding import ContextualChunk, ContextualEmbeddingSystem, RetrievalConfig

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search methods available."""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only" 
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class FusionMethod(Enum):
    """Methods for combining dense and sparse search results."""
    RECIPROCAL_RANK_FUSION = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"   # Weighted score combination
    CONVEX_COMBINATION = "convex"   # Convex combination of scores
    LEARNED_FUSION = "learned"      # Machine learned fusion


@dataclass
class SearchResult:
    """Individual search result with unified scoring."""
    document: Document
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rank_dense: int = -1
    rank_sparse: int = -1
    rank_hybrid: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class HybridSearchResults:
    """Complete hybrid search results with performance metrics."""
    results: List[SearchResult]
    total_results: int
    dense_search_time: float = 0.0
    sparse_search_time: float = 0.0
    fusion_time: float = 0.0
    total_time: float = 0.0
    search_metadata: Dict[str, Any] = field(default_factory=dict)


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search operations."""
    
    # Search weights and parameters
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for dense search (0.0-1.0)")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for sparse search (0.0-1.0)")
    
    # Result limits
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    dense_k: int = Field(default=50, ge=1, le=200, description="Number of dense results to retrieve")
    sparse_k: int = Field(default=50, ge=1, le=200, description="Number of sparse results to retrieve")
    
    # Fusion settings
    fusion_method: FusionMethod = Field(default=FusionMethod.RECIPROCAL_RANK_FUSION)
    rrf_k: int = Field(default=60, ge=1, description="RRF parameter k")
    
    # Performance settings  
    enable_parallel_search: bool = Field(default=True, description="Enable parallel dense/sparse search")
    cache_embeddings: bool = Field(default=True, description="Cache query embeddings")
    normalize_scores: bool = Field(default=True, description="Normalize scores before fusion")
    
    # BM25 parameters
    bm25_k1: float = Field(default=1.5, ge=0.0, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 b parameter")
    
    # Quality thresholds
    min_dense_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum dense score threshold")
    min_sparse_score: float = Field(default=0.0, ge=0.0, description="Minimum sparse score threshold")
    
    @validator('dense_weight', 'sparse_weight')
    def validate_weights(cls, v, values):
        """Ensure weights sum to 1.0 if both are provided."""
        if 'dense_weight' in values and 'sparse_weight' in values:
            total = values['dense_weight'] + v
            if not (0.95 <= total <= 1.05):  # Allow small tolerance
                raise ValueError("Dense weight + sparse weight must equal 1.0")
        return v


class BM25SearchEngine:
    """Optimized BM25 search engine with preprocessing and caching."""
    
    def __init__(self, documents: List[Document], config: HybridSearchConfig):
        """Initialize BM25 search engine with documents."""
        self.config = config
        self.documents = documents
        self.doc_texts = []
        self.tokenized_docs = []
        self.bm25 = None
        self._setup_bm25()
        
    def _setup_bm25(self):
        """Setup BM25 index with preprocessing."""
        logger.info(f"Setting up BM25 index for {len(self.documents)} documents")
        
        # Extract and preprocess text
        for doc in self.documents:
            text = doc.page_content
            self.doc_texts.append(text)
            
            # Simple tokenization (can be enhanced with better NLP)
            tokens = self._tokenize(text)
            self.tokenized_docs.append(tokens)
        
        # Initialize BM25 with custom parameters
        self.bm25 = rank_bm25.BM25Okapi(
            self.tokenized_docs,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )
        
        logger.info("BM25 index setup complete")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with preprocessing."""
        # Convert to lowercase and split on whitespace/punctuation
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search using BM25 and return results with scores."""
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Create results with scores
        doc_scores = [(self.documents[i], scores[i]) for i in range(len(scores))]
        
        # Sort by score (descending) and take top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:k]


class DenseSearchEngine:
    """Dense vector search engine with embedding caching."""
    
    def __init__(self, documents: List[Document], embeddings_model: OpenAIEmbeddings):
        """Initialize dense search with documents and embeddings."""
        self.documents = documents
        self.embeddings_model = embeddings_model
        self.doc_embeddings = None
        self.embedding_cache = {}
        self._setup_embeddings()
    
    async def _setup_embeddings(self):
        """Setup document embeddings (async for large collections)."""
        logger.info(f"Computing embeddings for {len(self.documents)} documents")
        
        # Extract text content
        texts = [doc.page_content for doc in self.documents]
        
        # Compute embeddings in batches for efficiency
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await self.embeddings_model.aembed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        self.doc_embeddings = np.array(all_embeddings)
        logger.info("Document embeddings computed")
    
    async def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search using dense vectors and return results with similarity scores."""
        if self.doc_embeddings is None:
            await self._setup_embeddings()
        
        # Check cache for query embedding
        if query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            query_embedding = await self.embeddings_model.aembed_query(query)
            self.embedding_cache[query] = query_embedding
        
        # Compute cosine similarities
        query_vec = np.array(query_embedding)
        similarities = np.dot(self.doc_embeddings, query_vec) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Create results with scores
        doc_scores = [(self.documents[i], similarities[i]) for i in range(len(similarities))]
        
        # Sort by similarity (descending) and take top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:k]


class ScoreFusion:
    """Advanced score fusion algorithms for hybrid search."""
    
    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]], 
        k: int = 60
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        
        # Create document to rank mappings
        dense_ranks = {id(doc): rank + 1 for rank, (doc, _) in enumerate(dense_results)}
        sparse_ranks = {id(doc): rank + 1 for rank, (doc, _) in enumerate(sparse_results)}
        
        # Create document to score mappings
        dense_scores = {id(doc): score for doc, score in dense_results}
        sparse_scores = {id(doc): score for doc, score in sparse_results}
        
        # Collect all unique documents
        all_docs = {}
        for doc, _ in dense_results + sparse_results:
            all_docs[id(doc)] = doc
        
        # Calculate RRF scores
        fusion_results = []
        for doc_id, doc in all_docs.items():
            dense_rank = dense_ranks.get(doc_id, len(dense_results) + 1)
            sparse_rank = sparse_ranks.get(doc_id, len(sparse_results) + 1)
            
            rrf_score = 1 / (k + dense_rank) + 1 / (k + sparse_rank)
            
            result = SearchResult(
                document=doc,
                dense_score=dense_scores.get(doc_id, 0.0),
                sparse_score=sparse_scores.get(doc_id, 0.0),
                hybrid_score=rrf_score,
                rank_dense=dense_rank,
                rank_sparse=sparse_rank
            )
            fusion_results.append(result)
        
        # Sort by RRF score (descending)
        fusion_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Update hybrid ranks
        for i, result in enumerate(fusion_results):
            result.rank_hybrid = i + 1
        
        return fusion_results
    
    @staticmethod
    def weighted_sum_fusion(
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        normalize: bool = True
    ) -> List[SearchResult]:
        """Weighted sum fusion with optional score normalization."""
        
        # Extract scores for normalization
        dense_scores_only = [score for _, score in dense_results]
        sparse_scores_only = [score for _, score in sparse_results]
        
        # Normalize scores if requested
        if normalize and dense_scores_only and sparse_scores_only:
            dense_max = max(dense_scores_only)
            dense_min = min(dense_scores_only)
            sparse_max = max(sparse_scores_only) 
            sparse_min = min(sparse_scores_only)
            
            # Min-max normalization
            def normalize_score(score, min_val, max_val):
                if max_val == min_val:
                    return 1.0
                return (score - min_val) / (max_val - min_val)
        else:
            normalize_score = lambda x, min_val, max_val: x
            dense_max = dense_min = sparse_max = sparse_min = 0
        
        # Create document mappings
        dense_docs = {id(doc): (doc, normalize_score(score, dense_min, dense_max)) 
                     for doc, score in dense_results}
        sparse_docs = {id(doc): (doc, normalize_score(score, sparse_min, sparse_max))
                      for doc, score in sparse_results}
        
        # Collect all unique documents
        all_doc_ids = set(dense_docs.keys()) | set(sparse_docs.keys())
        
        # Calculate weighted scores
        fusion_results = []
        for doc_id in all_doc_ids:
            dense_doc, dense_norm_score = dense_docs.get(doc_id, (None, 0.0))
            sparse_doc, sparse_norm_score = sparse_docs.get(doc_id, (None, 0.0))
            
            doc = dense_doc or sparse_doc
            weighted_score = dense_weight * dense_norm_score + sparse_weight * sparse_norm_score
            
            result = SearchResult(
                document=doc,
                dense_score=dense_norm_score if dense_doc else 0.0,
                sparse_score=sparse_norm_score if sparse_doc else 0.0,
                hybrid_score=weighted_score
            )
            fusion_results.append(result)
        
        # Sort by weighted score (descending)
        fusion_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return fusion_results


class HybridSearchEngine:
    """Main hybrid search engine combining dense and sparse search."""
    
    def __init__(
        self,
        documents: List[Document],
        embeddings_model: OpenAIEmbeddings,
        config: Optional[HybridSearchConfig] = None
    ):
        """Initialize hybrid search engine."""
        self.documents = documents
        self.embeddings_model = embeddings_model
        self.config = config or HybridSearchConfig()
        
        # Initialize search engines
        self.dense_engine = DenseSearchEngine(documents, embeddings_model)
        self.sparse_engine = BM25SearchEngine(documents, self.config)
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_dense_time': 0.0,
            'avg_sparse_time': 0.0,
            'avg_fusion_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info(f"HybridSearchEngine initialized with {len(documents)} documents")
    
    async def search(
        self, 
        query: str, 
        search_type: SearchType = SearchType.HYBRID,
        max_results: Optional[int] = None
    ) -> HybridSearchResults:
        """Perform hybrid search with configurable search type."""
        
        start_time = time.time()
        max_results = max_results or self.config.max_results
        
        try:
            if search_type == SearchType.DENSE_ONLY:
                return await self._dense_only_search(query, max_results)
            elif search_type == SearchType.SPARSE_ONLY:
                return await self._sparse_only_search(query, max_results)
            elif search_type == SearchType.HYBRID:
                return await self._hybrid_search(query, max_results)
            elif search_type == SearchType.ADAPTIVE:
                return await self._adaptive_search(query, max_results)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty results on error
            return HybridSearchResults(
                results=[],
                total_results=0,
                total_time=time.time() - start_time,
                search_metadata={'error': str(e)}
            )
    
    async def _hybrid_search(self, query: str, max_results: int) -> HybridSearchResults:
        """Perform full hybrid search with parallel execution."""
        
        search_start = time.time()
        
        # Parallel search execution if enabled
        if self.config.enable_parallel_search:
            dense_task = self._timed_dense_search(query, self.config.dense_k)
            sparse_task = self._timed_sparse_search(query, self.config.sparse_k)
            
            (dense_results, dense_time), (sparse_results, sparse_time) = await asyncio.gather(
                dense_task, sparse_task
            )
        else:
            # Sequential execution
            dense_results, dense_time = await self._timed_dense_search(query, self.config.dense_k)
            sparse_results, sparse_time = await self._timed_sparse_search(query, self.config.sparse_k)
        
        # Fusion phase
        fusion_start = time.time()
        fused_results = await self._fuse_results(dense_results, sparse_results)
        fusion_time = time.time() - fusion_start
        
        # Apply thresholds and limit results
        filtered_results = self._apply_thresholds(fused_results[:max_results])
        
        total_time = time.time() - search_start
        
        # Update statistics
        self._update_stats(dense_time, sparse_time, fusion_time)
        
        return HybridSearchResults(
            results=filtered_results,
            total_results=len(filtered_results),
            dense_search_time=dense_time,
            sparse_search_time=sparse_time,
            fusion_time=fusion_time,
            total_time=total_time,
            search_metadata={
                'query': query,
                'fusion_method': self.config.fusion_method.value,
                'dense_results': len(dense_results),
                'sparse_results': len(sparse_results)
            }
        )
    
    async def _dense_only_search(self, query: str, max_results: int) -> HybridSearchResults:
        """Perform dense-only search."""
        start_time = time.time()
        
        dense_results, dense_time = await self._timed_dense_search(query, max_results)
        
        # Convert to SearchResult format
        search_results = [
            SearchResult(
                document=doc,
                dense_score=score,
                hybrid_score=score,
                rank_dense=i + 1,
                rank_hybrid=i + 1
            )
            for i, (doc, score) in enumerate(dense_results)
        ]
        
        total_time = time.time() - start_time
        
        return HybridSearchResults(
            results=search_results[:max_results],
            total_results=len(search_results),
            dense_search_time=dense_time,
            total_time=total_time,
            search_metadata={'query': query, 'search_type': 'dense_only'}
        )
    
    async def _sparse_only_search(self, query: str, max_results: int) -> HybridSearchResults:
        """Perform sparse-only search."""
        start_time = time.time()
        
        sparse_results, sparse_time = await self._timed_sparse_search(query, max_results)
        
        # Convert to SearchResult format
        search_results = [
            SearchResult(
                document=doc,
                sparse_score=score,
                hybrid_score=score,
                rank_sparse=i + 1,
                rank_hybrid=i + 1
            )
            for i, (doc, score) in enumerate(sparse_results)
        ]
        
        total_time = time.time() - start_time
        
        return HybridSearchResults(
            results=search_results[:max_results],
            total_results=len(search_results),
            sparse_search_time=sparse_time,
            total_time=total_time,
            search_metadata={'query': query, 'search_type': 'sparse_only'}
        )
    
    async def _adaptive_search(self, query: str, max_results: int) -> HybridSearchResults:
        """Adaptive search that chooses optimal strategy based on query characteristics."""
        
        # Simple adaptive logic (can be enhanced with ML)
        query_length = len(query.split())
        has_keywords = any(word.isupper() or len(word) > 10 for word in query.split())
        
        if query_length <= 3 and has_keywords:
            # Short queries with keywords benefit from sparse search
            search_type = SearchType.SPARSE_ONLY
        elif query_length > 10:
            # Long queries benefit from dense search
            search_type = SearchType.DENSE_ONLY  
        else:
            # Default to hybrid for balanced queries
            search_type = SearchType.HYBRID
        
        result = await self.search(query, search_type, max_results)
        result.search_metadata['adaptive_choice'] = search_type.value
        
        return result
    
    async def _timed_dense_search(self, query: str, k: int) -> Tuple[List[Tuple[Document, float]], float]:
        """Timed dense search execution."""
        start_time = time.time()
        results = await self.dense_engine.search(query, k)
        elapsed = time.time() - start_time
        return results, elapsed
    
    async def _timed_sparse_search(self, query: str, k: int) -> Tuple[List[Tuple[Document, float]], float]:
        """Timed sparse search execution."""
        start_time = time.time()
        # Run sparse search in thread pool since it's CPU-bound
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.sparse_engine.search, query, k)
        elapsed = time.time() - start_time
        return results, elapsed
    
    async def _fuse_results(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]]
    ) -> List[SearchResult]:
        """Fuse dense and sparse results using configured method."""
        
        if self.config.fusion_method == FusionMethod.RECIPROCAL_RANK_FUSION:
            return ScoreFusion.reciprocal_rank_fusion(
                dense_results, sparse_results, self.config.rrf_k
            )
        elif self.config.fusion_method == FusionMethod.WEIGHTED_SUM:
            return ScoreFusion.weighted_sum_fusion(
                dense_results, sparse_results,
                self.config.dense_weight, self.config.sparse_weight,
                self.config.normalize_scores
            )
        else:
            # Default to RRF
            return ScoreFusion.reciprocal_rank_fusion(
                dense_results, sparse_results, self.config.rrf_k
            )
    
    def _apply_thresholds(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply quality thresholds to filter results."""
        filtered = []
        
        for result in results:
            if (result.dense_score >= self.config.min_dense_score and 
                result.sparse_score >= self.config.min_sparse_score):
                filtered.append(result)
        
        return filtered
    
    def _update_stats(self, dense_time: float, sparse_time: float, fusion_time: float):
        """Update performance statistics."""
        self.search_stats['total_searches'] += 1
        n = self.search_stats['total_searches']
        
        # Running average calculation
        self.search_stats['avg_dense_time'] = (
            (self.search_stats['avg_dense_time'] * (n - 1) + dense_time) / n
        )
        self.search_stats['avg_sparse_time'] = (
            (self.search_stats['avg_sparse_time'] * (n - 1) + sparse_time) / n
        )
        self.search_stats['avg_fusion_time'] = (
            (self.search_stats['avg_fusion_time'] * (n - 1) + fusion_time) / n
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_searches': self.search_stats['total_searches'],
            'average_times': {
                'dense_search_ms': self.search_stats['avg_dense_time'] * 1000,
                'sparse_search_ms': self.search_stats['avg_sparse_time'] * 1000,
                'fusion_ms': self.search_stats['avg_fusion_time'] * 1000,
                'total_avg_ms': (
                    self.search_stats['avg_dense_time'] +
                    self.search_stats['avg_sparse_time'] +
                    self.search_stats['avg_fusion_time']
                ) * 1000
            },
            'configuration': {
                'dense_weight': self.config.dense_weight,
                'sparse_weight': self.config.sparse_weight,
                'fusion_method': self.config.fusion_method.value,
                'parallel_search': self.config.enable_parallel_search
            }
        }


# Integration helper for contextual embedding system
class ContextualHybridSearch:
    """Hybrid search integrated with contextual embedding system."""
    
    def __init__(
        self,
        contextual_system: ContextualEmbeddingSystem,
        embeddings_model: OpenAIEmbeddings,
        hybrid_config: Optional[HybridSearchConfig] = None
    ):
        """Initialize contextual hybrid search."""
        self.contextual_system = contextual_system
        self.embeddings_model = embeddings_model
        self.hybrid_config = hybrid_config or HybridSearchConfig()
        self.hybrid_engine = None
        
    async def setup_search_index(self, documents: List[Document]) -> None:
        """Setup search index with contextual enhancement."""
        
        # Generate contextual chunks
        enhanced_documents = []
        for doc in documents:
            contextual_chunk = await self.contextual_system.create_contextual_chunk(
                document=doc,
                surrounding_content=[],  # Would be populated with actual surrounding content
                config=RetrievalConfig()
            )
            
            # Create enhanced document with contextual full text
            enhanced_doc = Document(
                page_content=contextual_chunk.full_text,
                metadata={
                    **doc.metadata,
                    'original_content': doc.page_content,
                    'contextual_metadata': contextual_chunk.metadata,
                    'context_type': contextual_chunk.context_type,
                    'quality_score': contextual_chunk.quality_score
                }
            )
            enhanced_documents.append(enhanced_doc)
        
        # Initialize hybrid search with enhanced documents
        self.hybrid_engine = HybridSearchEngine(
            enhanced_documents,
            self.embeddings_model,
            self.hybrid_config
        )
        
        logger.info(f"Contextual hybrid search index setup with {len(enhanced_documents)} enhanced documents")
    
    async def contextual_search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        max_results: int = 20
    ) -> HybridSearchResults:
        """Perform contextual hybrid search."""
        
        if not self.hybrid_engine:
            raise ValueError("Search index not initialized. Call setup_search_index() first.")
        
        # Perform hybrid search
        results = await self.hybrid_engine.search(query, search_type, max_results)
        
        # Enhance results with contextual metadata
        for result in results.results:
            result.metadata['contextual_search'] = True
            result.metadata['original_content'] = result.document.metadata.get('original_content')
            result.metadata['contextual_quality'] = result.document.metadata.get('quality_score', 0.0)
        
        results.search_metadata['contextual_enhancement'] = True
        
        return results 