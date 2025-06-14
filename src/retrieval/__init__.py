"""
Contextual Retrieval System
==========================

This module implements advanced retrieval capabilities including:
- Contextual embedding with document structure awareness
- Hybrid search combining dense vector and sparse BM25
- Multi-query retrieval with LLM expansion
- Self-query metadata filtering
- Maximal Marginal Relevance for diversity
- Seamless integration with Task 2 confidence scoring

Task 3.1: Contextual Embedding System ✅
Task 3.2: Hybrid Search Infrastructure ✅
Task 3.3: Multi-Query Retrieval ✅
Task 3.4: Self-Query Metadata Filtering ✅
Task 3.5: MMR & Task 2 Integration ✅
"""

from .contextual_embedding import (
    ContextualChunk,
    ContextualEmbeddingSystem,
    RetrievalConfig,
    DocumentProcessor
)

from .hybrid_search import (
    HybridSearchEngine,
    ContextualHybridSearch,
    HybridSearchConfig,
    SearchType,
    FusionMethod,
    SearchResult,
    HybridSearchResults,
    BM25SearchEngine,
    DenseSearchEngine,
    ScoreFusion
)

from .multi_query import (
    MultiQueryRetriever,
    QueryExpander,
    ResultAggregator,
    MultiQueryConfig,
    QueryExpansionStrategy,
    QueryType,
    ExpandedQuery,
    MultiQueryResults
)

from .self_query import (
    SelfQueryRetriever,
    QueryAnalyzer,
    SelfQueryConfig,
    MetadataFilter,
    FilterOperator,
    FilterScope,
    QueryAnalysis,
    SelfQueryResults
)

from .contextual_retrieval import (
    ContextualRetrievalSystem,
    MaximalMarginalRelevance,
    RetrievalStrategy,
    create_contextual_retrieval_system
)

__all__ = [
    # Contextual Embedding (Task 3.1)
    "ContextualChunk",
    "ContextualEmbeddingSystem", 
    "RetrievalConfig",
    "DocumentProcessor",
    
    # Hybrid Search (Task 3.2)
    "HybridSearchEngine",
    "ContextualHybridSearch", 
    "HybridSearchConfig",
    "SearchType",
    "FusionMethod",
    "SearchResult",
    "HybridSearchResults",
    "BM25SearchEngine",
    "DenseSearchEngine",
    "ScoreFusion",
    
        # Multi-Query Retrieval (Task 3.3)
    "MultiQueryRetriever",
    "QueryExpander",
    "ResultAggregator", 
    "MultiQueryConfig",
    "QueryExpansionStrategy",
    "QueryType",
    "ExpandedQuery",
    "MultiQueryResults",
    
    # Self-Query Metadata Filtering (Task 3.4)
    "SelfQueryRetriever",
    "QueryAnalyzer",
    "SelfQueryConfig",
    "MetadataFilter",
    "FilterOperator",
    "FilterScope",
    "QueryAnalysis",
    "SelfQueryResults",
    
    # MMR & Task 2 Integration (Task 3.5)
    "ContextualRetrievalSystem",
    "MaximalMarginalRelevance",
    "RetrievalStrategy",
    "create_contextual_retrieval_system"
]

__version__ = "1.1.0" 