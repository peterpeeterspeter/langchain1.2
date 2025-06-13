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
from .contextual_embedding import ContextualEmbeddingSystem, ContextualChunk
from .hybrid_search import HybridSearchEngine, HybridSearchConfig
from .multi_query import MultiQueryRetriever, MultiQueryConfig
from .self_query import SelfQueryRetriever, SelfQueryConfig

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enum for different retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    MULTI_QUERY = "multi_query"


@dataclass
class RetrievalConfig:
    """Configuration for the contextual retrieval system."""
    # Hybrid search weights
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    # Contextual embedding settings
    context_window_size: int = 2  # sentences before/after for context
    include_document_title: bool = True
    include_section_headers: bool = True
    
    # Multi-query settings
    max_query_variations: int = 3
    query_expansion_model: str = "gpt-4"
    
    # MMR settings
    mmr_lambda: float = 0.7  # Balance between relevance and diversity
    mmr_k: int = 20  # Fetch k documents before MMR
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    parallel_retrieval: bool = True
    max_workers: int = 4
    
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
