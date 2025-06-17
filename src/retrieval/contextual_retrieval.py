# Task 3: Contextual Retrieval System Architecture
# src/retrieval/contextual_retrieval.py

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import asyncio
from datetime import datetime, timedelta
import hashlib
import logging
import time

# Import from your existing Task 2 implementation
try:
    from ..chains.enhanced_confidence_scoring_system import (
        SourceQualityAnalyzer, 
        IntelligentCache,
        EnhancedRAGResponse
    )
    TASK2_INTEGRATION_AVAILABLE = True
except ImportError:
    TASK2_INTEGRATION_AVAILABLE = False
    logging.warning("Task 2 components not available. Running in standalone mode.")

# Import existing Task 3 components
try:
    from .contextual_embedding import ContextualEmbeddingSystem, ContextualChunk, RetrievalConfig
    from .hybrid_search import HybridSearchEngine, HybridSearchConfig
    from .multi_query import MultiQueryRetriever, MultiQueryConfig
    from .self_query import SelfQueryRetriever, SelfQueryConfig
except ImportError:
    logging.warning("Some Task 3 components not available. Using fallback implementations.")

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enum for different retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    MULTI_QUERY = "multi_query"
    
    # Task 2 Integration settings
    enable_source_quality_analysis: bool = True
    enable_confidence_scoring: bool = True
    quality_threshold: float = 0.6
    
    # Self-query settings
    enable_metadata_filtering: bool = True
    filter_confidence_threshold: float = 0.7


class MaximalMarginalRelevance:
    """
    Subtask 3.5: Add Maximal Marginal Relevance (MMR)
    Selects diverse results while maintaining relevance.
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Handle zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(np.clip(cosine_sim, -1.0, 1.0))
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def apply_mmr(
        self,
        query_embedding: List[float],
        documents_with_embeddings: List[Tuple[Document, List[float]]],
        k: int = 10
    ) -> List[Document]:
        """Apply MMR to select diverse and relevant documents."""
        if not documents_with_embeddings:
            return []
        
        try:
            # Calculate relevance scores for all documents
            relevance_scores = []
            for doc, embedding in documents_with_embeddings:
                relevance = self.calculate_similarity(query_embedding, embedding)
                relevance_scores.append((doc, embedding, relevance))
            
            # Sort by relevance
            relevance_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Select documents using MMR
            selected = []
            selected_embeddings = []
            candidates = relevance_scores.copy()
            
            # Select the most relevant document first
            if candidates:
                best_doc, best_embedding, best_score = candidates.pop(0)
                selected.append(best_doc)
                selected_embeddings.append(best_embedding)
                
                # Add MMR metadata
                best_doc.metadata['mmr_position'] = 0
                best_doc.metadata['relevance_score'] = best_score
                best_doc.metadata['diversity_score'] = 1.0
            
            # Select remaining documents based on MMR score
            while len(selected) < k and candidates:
                mmr_scores = []
                
                for doc, embedding, relevance in candidates:
                    # Calculate maximum similarity to already selected documents
                    max_sim_to_selected = max(
                        self.calculate_similarity(embedding, sel_embedding)
                        for sel_embedding in selected_embeddings
                    ) if selected_embeddings else 0.0
                    
                    # Calculate MMR score
                    mmr_score = (
                        self.config.mmr_lambda * relevance - 
                        (1 - self.config.mmr_lambda) * max_sim_to_selected
                    )
                    mmr_scores.append((doc, embedding, mmr_score, relevance, max_sim_to_selected))
                
                # Select document with highest MMR score
                mmr_scores.sort(key=lambda x: x[2], reverse=True)
                best_doc, best_embedding, mmr_score, relevance, diversity = mmr_scores[0]
                
                # Add MMR metadata
                best_doc.metadata['mmr_position'] = len(selected)
                best_doc.metadata['mmr_score'] = mmr_score
                best_doc.metadata['relevance_score'] = relevance
                best_doc.metadata['diversity_score'] = 1.0 - diversity
                
                selected.append(best_doc)
                selected_embeddings.append(best_embedding)
                
                # Remove selected document from candidates
                candidates = [
                    (d, e, r) for d, e, r in candidates 
                    if d.page_content != best_doc.page_content
                ]
            
            self.logger.info(f"MMR selected {len(selected)} diverse documents from {len(documents_with_embeddings)} candidates")
            return selected
            
        except Exception as e:
            self.logger.error(f"Error in MMR application: {e}")
            # Fallback: return top documents by relevance
            return [doc for doc, _, _ in relevance_scores[:k]]


@dataclass
class RetrievalMetrics:
    """Metrics for contextual retrieval performance."""
    total_documents_retrieved: int = 0
    cache_hit_rate: float = 0.0
    average_response_time_ms: float = 0.0
    mmr_selections: int = 0
    quality_enhancements: int = 0
    confidence_scores: List[float] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = []


class ContextualRetrievalSystem(BaseRetriever):
    """
    Main contextual retrieval system integrating all Task 3 components.
    This is the unified system that brings together:
    - Task 3.1: Contextual Embedding System
    - Task 3.2: Hybrid Search Infrastructure  
    - Task 3.3: Multi-Query Retrieval
    - Task 3.4: Self-Query Metadata Filtering
    - Task 3.5: MMR & Task 2 Integration
    """
    
    def __init__(
        self,
        config: RetrievalConfig = None,
        supabase_client = None,
        embeddings_model = None,
        **kwargs
    ):
        """Initialize the contextual retrieval system."""
        # Initialize base retriever first
        super().__init__(**kwargs)
        
        # Store configuration and components
        self._config = config or RetrievalConfig()
        self._supabase_client = supabase_client
        self._embeddings_model = embeddings_model
        self._logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self._metrics = RetrievalMetrics()
        
        # Initialize components
        self._init_components()
        
        self._logger.info("Contextual Retrieval System initialized with all Task 3 components")
    
    @property
    def config(self) -> RetrievalConfig:
        """Get the configuration."""
        return self._config
    
    @property
    def supabase_client(self):
        """Get the Supabase client."""
        return self._supabase_client
    
    @property
    def embeddings_model(self):
        """Get the embeddings model."""
        return self._embeddings_model
    
    @property
    def logger(self):
        """Get the logger."""
        return self._logger
    
    @property
    def metrics(self) -> RetrievalMetrics:
        """Get the metrics."""
        return self._metrics
    
    def _init_components(self):
        """Initialize all retrieval components."""
        try:
            # Task 3.1: Contextual Embedding System
            self._contextual_embedding = ContextualEmbeddingSystem(
                config=self.config
            )
            
            # Task 3.2: Hybrid Search Infrastructure (with fallback defaults)
            try:
                # Use SupabaseVectorStore directly like the main chain does
                if self.supabase_client and self.embeddings_model:
                    from langchain_community.vectorstores import SupabaseVectorStore
                    
                    self._hybrid_search = SupabaseVectorStore(
                        client=self.supabase_client,
                        embedding=self.embeddings_model,
                        table_name="documents",
                        query_name="match_documents"
                    )
                else:
                    self._hybrid_search = None
                    
            except Exception as e:
                self.logger.warning(f"Hybrid search initialization failed: {e}. Using fallback.")
                self._hybrid_search = None
            
            # Task 3.3: Multi-Query Retrieval (with fallback defaults)
            max_variations = getattr(self.config, 'max_query_variations', 1)
            if max_variations > 1:
                try:
                    multi_query_config = MultiQueryConfig(
                        max_variations=max_variations,
                        llm_model=getattr(self.config, 'query_expansion_model', 'gpt-4')
                    )
                    self._multi_query = MultiQueryRetriever(
                        base_retriever=self._hybrid_search,
                        config=multi_query_config
                    )
                except Exception as e:
                    self.logger.warning(f"Multi-query initialization failed: {e}")
                    self._multi_query = None
            else:
                self._multi_query = None
            
            # Task 3.4: Self-Query Metadata Filtering (with fallback defaults)
            enable_filtering = getattr(self.config, 'enable_metadata_filtering', False)
            if enable_filtering:
                try:
                    self_query_config = SelfQueryConfig(
                        analysis_confidence_threshold=getattr(self.config, 'filter_confidence_threshold', 0.7)
                    )
                    self._self_query = SelfQueryRetriever(
                        hybrid_search_engine=self._hybrid_search,
                        multi_query_retriever=self._multi_query,
                        config=self_query_config
                    )
                except Exception as e:
                    self.logger.warning(f"Self-query initialization failed: {e}")
                    self._self_query = None
            else:
                self._self_query = None
            
            # Task 3.5: MMR & Task 2 Integration
            self._mmr = MaximalMarginalRelevance(self.config)
            
            # Task 2 Integration (if available)
            enable_quality_analysis = getattr(self.config, 'enable_source_quality_analysis', True)
            if TASK2_INTEGRATION_AVAILABLE and enable_quality_analysis:
                self._source_quality_analyzer = SourceQualityAnalyzer()
                if getattr(self.config, 'enable_caching', True):
                    self._intelligent_cache = IntelligentCache()
                else:
                    self._intelligent_cache = None
            else:
                self._source_quality_analyzer = None
                self._intelligent_cache = None
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    @property
    def contextual_embedding(self):
        """Get the contextual embedding system."""
        return self._contextual_embedding
    
    @property
    def hybrid_search(self):
        """Get the hybrid search engine."""
        return self._hybrid_search
    
    @property
    def multi_query(self):
        """Get the multi-query retriever."""
        return self._multi_query
    
    @property
    def self_query(self):
        """Get the self-query retriever."""
        return self._self_query
    
    @property
    def mmr(self):
        """Get the MMR instance."""
        return self._mmr
    
    @property
    def source_quality_analyzer(self):
        """Get the source quality analyzer."""
        return self._source_quality_analyzer
    
    @property
    def intelligent_cache(self):
        """Get the intelligent cache."""
        return self._intelligent_cache
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: int = 10,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> List[Document]:
        """Retrieve relevant documents using the full contextual retrieval pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Check cache first (if Task 2 integration available)
            if self.intelligent_cache:
                cached_result = await self.intelligent_cache.get(query)
                if cached_result:
                    self.metrics.cache_hit_rate += 1
                    self.logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached_result.sources[:k] if hasattr(cached_result, 'sources') else []
            
            # Step 2: Extract metadata filters using self-query (Task 3.4)
            filters = []
            enable_filtering = getattr(self.config, 'enable_metadata_filtering', False)
            mmr_k = getattr(self.config, 'mmr_k', 20)  # Default to 20 documents for MMR
            if self.self_query and enable_filtering:
                try:
                    self_query_results = await self.self_query.retrieve(query, max_results=mmr_k)
                    filters = self_query_results.applied_filters
                    self.logger.info(f"Extracted {len(filters)} metadata filters")
                except Exception as e:
                    self.logger.warning(f"Self-query filtering failed: {e}")
            
            # Step 3: Get query embedding for contextual search
            if self.embeddings_model:
                query_embedding = await self.embeddings_model.aembed_query(query)
            else:
                query_embedding = None
            
            # Step 4: Perform retrieval based on strategy
            if strategy == RetrievalStrategy.MULTI_QUERY and self.multi_query:
                # Multi-query expansion (Task 3.3)
                results = await self.multi_query.retrieve(query, max_results=mmr_k)
                documents_with_embeddings = [
                    (doc, doc.metadata.get('embedding', []))
                    for doc in results.combined_results
                ]
            elif strategy == RetrievalStrategy.CONTEXTUAL:
                # Contextual embedding search (Task 3.1)
                contextual_chunks = await self.contextual_embedding.create_contextual_chunks([query])
                documents_with_embeddings = []
                for chunk in contextual_chunks:
                    doc = Document(
                        page_content=chunk.enhanced_content,
                        metadata=chunk.metadata
                    )
                    documents_with_embeddings.append((doc, chunk.embedding))
            else:
                # Vector search using SupabaseVectorStore
                if self.hybrid_search:
                    try:
                        # Use the same direct RPC approach as the main chain
                        query_embedding = await self.embeddings_model.aembed_query(query)
                        
                        response = self.supabase_client.rpc(
                            'match_documents',
                            {
                                'query_embedding': query_embedding,
                                'match_threshold': 0.3,  # Our fixed threshold
                                'match_count': mmr_k
                            }
                        ).execute()
                        
                        documents_with_embeddings = []
                        if response.data:
                            for item in response.data:
                                # Extract metadata safely
                                item_metadata = item.get('metadata', {})
                                
                                doc = Document(
                                    page_content=item.get('content', ''),
                                    metadata={
                                        'id': item.get('id'),
                                        'similarity': float(item.get('similarity', 0.0)),
                                        # Get data from metadata if available
                                        'keyword': item_metadata.get('keyword', ''),
                                        'article_id': item_metadata.get('article_id', ''),
                                        'created_at': item_metadata.get('created_at', ''),
                                        # Add original metadata
                                        'original_metadata': item_metadata
                                    }
                                )
                                documents_with_embeddings.append((doc, []))  # No embedding needed
                        
                        self.logger.info(f"Retrieved {len(documents_with_embeddings)} documents via direct RPC")
                        
                    except Exception as e:
                        self.logger.error(f"Direct RPC search failed: {e}")
                        documents_with_embeddings = []
                else:
                    documents_with_embeddings = []
            
            # Step 5: Apply MMR for diversity (Task 3.5)
            if query_embedding and documents_with_embeddings:
                final_documents = await self.mmr.apply_mmr(
                    query_embedding=query_embedding,
                    documents_with_embeddings=documents_with_embeddings,
                    k=k
                )
                self.metrics.mmr_selections += 1
            else:
                # Fallback: take top documents
                final_documents = [doc for doc, _ in documents_with_embeddings[:k]]
            
            # Step 6: Enhance with source quality analysis (Task 2 integration)
            enable_source_quality_analysis = getattr(self.config, 'enable_source_quality_analysis', True)
            if self.source_quality_analyzer and enable_source_quality_analysis:
                try:
                    for doc in final_documents:
                        quality_score = await self.source_quality_analyzer.analyze_source_quality(
                            content=doc.page_content,
                            metadata=doc.metadata
                        )
                        doc.metadata['source_quality_score'] = quality_score
                        if quality_score >= self.config.quality_threshold:
                            self.metrics.quality_enhancements += 1
                except Exception as e:
                    self.logger.warning(f"Source quality analysis failed: {e}")
            
            # Step 7: Calculate confidence scores (Task 2 integration) 
            enable_confidence_scoring = getattr(self.config, 'enable_confidence_scoring', True)
            if enable_confidence_scoring and final_documents:
                # Simple confidence based on retrieval scores and quality
                for doc in final_documents:
                    relevance = doc.metadata.get('relevance_score', 0.5)
                    quality = doc.metadata.get('source_quality_score', 0.5)
                    diversity = doc.metadata.get('diversity_score', 0.5)
                    result_count_bonus = min(len(final_documents) / k, 1.0) * 0.1
                    
                    confidence = (relevance * 0.4 + quality * 0.3 + diversity * 0.2 + result_count_bonus)
                    doc.metadata['confidence_score'] = min(confidence, 1.0)
                    self.metrics.confidence_scores.append(confidence)
            
            # Step 8: Cache results (if Task 2 integration available)
            if self.intelligent_cache and final_documents:
                try:
                    # Create a simple response object for caching
                    cache_response = {
                        'sources': final_documents,
                        'metadata': {
                            'retrieval_strategy': strategy.value,
                            'num_results': len(final_documents),
                            'processing_time_ms': (time.time() - start_time) * 1000
                        }
                    }
                    await self.intelligent_cache.set(query, cache_response)
                except Exception as e:
                    self.logger.warning(f"Failed to cache results: {e}")
            
            # Update metrics
            self.metrics.total_documents_retrieved += len(final_documents)
            response_time = (time.time() - start_time) * 1000
            self.metrics.average_response_time_ms = (
                (self.metrics.average_response_time_ms + response_time) / 2
                if self.metrics.average_response_time_ms > 0
                else response_time
            )
            
            # Add retrieval metadata to documents
            for i, doc in enumerate(final_documents):
                doc.metadata.update({
                    'retrieval_timestamp': datetime.utcnow().isoformat(),
                    'retrieval_strategy': strategy.value,
                    'retrieval_rank': i,
                    'processing_time_ms': response_time
                })
            
            self.logger.info(f"Retrieved {len(final_documents)} documents in {response_time:.1f}ms")
            return final_documents
            
        except Exception as e:
            self.logger.error(f"Error in contextual retrieval: {e}")
            # Fallback to empty results - no fallback search needed
            self.logger.warning("Returning empty results due to retrieval failure")
            return []
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs
    ) -> List[Document]:
        """Synchronous version of document retrieval (required by BaseRetriever)."""
        # Use asyncio to run the async method
        import asyncio
        
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                task = asyncio.create_task(
                    self._aget_relevant_documents(query, run_manager=run_manager, **kwargs)
                )
                # This is a workaround for sync calls from async contexts
                return []  # Return empty for now, should use async version
            else:
                # We're not in an event loop, can run async directly
                return loop.run_until_complete(
                    self._aget_relevant_documents(query, run_manager=run_manager, **kwargs)
                )
        except Exception as e:
            self.logger.error(f"Error in synchronous retrieval: {e}")
            # Fallback to empty results
            return []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the retrieval system."""
        avg_confidence = (
            sum(self.metrics.confidence_scores) / len(self.metrics.confidence_scores)
            if self.metrics.confidence_scores
            else 0.0
        )
        
        return {
            'total_documents_retrieved': self.metrics.total_documents_retrieved,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'average_response_time_ms': self.metrics.average_response_time_ms,
            'mmr_selections': self.metrics.mmr_selections,
            'quality_enhancements': self.metrics.quality_enhancements,
            'average_confidence_score': avg_confidence,
            'task_2_integration_available': TASK2_INTEGRATION_AVAILABLE,
            'components_enabled': {
                'contextual_embedding': True,
                'hybrid_search': True,
                'multi_query': self.multi_query is not None,
                'self_query': self.self_query is not None,
                'mmr': True,
                'source_quality_analysis': self.source_quality_analyzer is not None,
                'intelligent_caching': self.intelligent_cache is not None
            }
        }


# Factory function for easy system creation
def create_contextual_retrieval_system(
    supabase_url: str = None,
    supabase_key: str = None,
    embeddings_model = None,
    config: RetrievalConfig = None,
    **kwargs
) -> ContextualRetrievalSystem:
    """
    Factory function to create a fully configured contextual retrieval system.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        embeddings_model: Embeddings model for vector search
        config: Configuration for the retrieval system
        **kwargs: Additional configuration options
        
    Returns:
        Fully configured ContextualRetrievalSystem
    """
    from supabase import create_client
    
    # Create Supabase client if credentials provided
    supabase_client = None
    if supabase_url and supabase_key:
        try:
            supabase_client = create_client(supabase_url, supabase_key)
        except Exception as e:
            logger.warning(f"Failed to create Supabase client: {e}")
    
    # Use default config if none provided
    if config is None:
        config = RetrievalConfig(**kwargs)
    
    # Create embeddings model if not provided
    if embeddings_model is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings_model = OpenAIEmbeddings()
        except ImportError:
            logger.warning("OpenAI embeddings not available. Some features may be limited.")
    
    return ContextualRetrievalSystem(
        config=config,
        supabase_client=supabase_client,
        embeddings_model=embeddings_model
    )
