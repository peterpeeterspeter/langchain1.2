"""
Multi-Query Retrieval System for Universal RAG CMS
==================================================

This module implements Task 3.3: Multi-Query Retrieval with LLM Query Expansion
Generates query variations using LLM and processes them in parallel for comprehensive result coverage.

Key Features:
- LLM-powered query expansion and variation generation
- Parallel query processing for performance optimization
- Query type-aware expansion strategies
- Result aggregation and deduplication
- Integration with hybrid search infrastructure
- Performance monitoring and optimization
"""

import asyncio
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import statistics
from collections import defaultdict, Counter
import re

from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Import hybrid search components
from .hybrid_search import (
    HybridSearchEngine, ContextualHybridSearch, HybridSearchResults,
    SearchType, SearchResult, HybridSearchConfig
)

logger = logging.getLogger(__name__)


class QueryExpansionStrategy(Enum):
    """Strategies for query expansion."""
    SEMANTIC_EXPANSION = "semantic"      # Expand with semantically similar terms
    PERSPECTIVE_EXPANSION = "perspective" # Different viewpoints/perspectives
    SPECIFICITY_EXPANSION = "specificity" # More/less specific variations
    CONTEXTUAL_EXPANSION = "contextual"   # Context-aware variations
    DOMAIN_EXPANSION = "domain"          # Domain-specific variations
    COMPREHENSIVE = "comprehensive"      # All strategies combined


class QueryType(Enum):
    """Types of queries for targeted expansion."""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"
    REVIEW = "review"
    TROUBLESHOOTING = "troubleshooting"
    NEWS = "news"
    RESEARCH = "research"
    GENERAL = "general"


@dataclass
class ExpandedQuery:
    """Individual expanded query with metadata."""
    query_text: str
    expansion_strategy: QueryExpansionStrategy
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: float = 0.0


@dataclass
class MultiQueryResults:
    """Results from multi-query retrieval with aggregation metadata."""
    aggregated_results: List[SearchResult]
    original_query: str
    expanded_queries: List[ExpandedQuery]
    individual_results: Dict[str, HybridSearchResults]
    total_unique_documents: int
    expansion_time: float = 0.0
    search_time: float = 0.0
    aggregation_time: float = 0.0
    total_time: float = 0.0
    performance_metadata: Dict[str, Any] = field(default_factory=dict)


class MultiQueryConfig(BaseModel):
    """Configuration for multi-query retrieval."""
    
    # Query expansion settings
    num_expansions: int = Field(default=3, ge=1, le=10, description="Number of query expansions to generate")
    expansion_strategies: List[QueryExpansionStrategy] = Field(
        default=[QueryExpansionStrategy.SEMANTIC_EXPANSION, QueryExpansionStrategy.PERSPECTIVE_EXPANSION],
        description="List of expansion strategies to use"
    )
    
    # LLM settings
    llm_model: str = Field(default="gpt-4.1-mini", description="LLM model for query expansion")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Temperature for query generation")
    max_tokens: int = Field(default=500, ge=100, le=2000, description="Max tokens for LLM response")
    
    # Processing settings
    enable_parallel_search: bool = Field(default=True, description="Enable parallel query processing")
    max_concurrent_queries: int = Field(default=5, ge=1, le=20, description="Maximum concurrent query searches")
    
    # Result aggregation settings
    aggregation_method: str = Field(default="weighted_fusion", description="Method for aggregating results")
    deduplication_threshold: float = Field(default=0.9, ge=0.0, le=1.0, description="Similarity threshold for deduplication")
    max_final_results: int = Field(default=20, ge=1, le=100, description="Maximum final results to return")
    
    # Quality settings
    min_expansion_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence for query expansions")
    enable_query_validation: bool = Field(default=True, description="Enable validation of generated queries")
    
    # Performance settings
    search_timeout: float = Field(default=30.0, ge=5.0, description="Timeout for search operations in seconds")
    cache_expansions: bool = Field(default=True, description="Cache query expansions")


class QueryExpander:
    """LLM-powered query expansion system."""
    
    def __init__(self, config: MultiQueryConfig):
        """Initialize the query expander with LLM."""
        self.config = config
        
        # Initialize LLM
        if "gpt" in config.llm_model.lower():
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.max_tokens
            )
        elif "claude" in config.llm_model.lower():
            self.llm = ChatAnthropic(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.max_tokens
            )
        else:
            # Default to OpenAI
            self.llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=config.llm_temperature,
                max_tokens=config.max_tokens
            )
        
        # Expansion cache for performance
        self.expansion_cache = {}
        
        # Enhanced query expansion prompt
        self.expansion_template = """You are an expert query expansion specialist focused on creating diverse, high-quality search variations that improve retrieval effectiveness.

**Your Mission:**
Generate {num_queries} distinct but related search queries that capture different aspects, phrasings, and semantic variations of the original query while maintaining search intent.

**Query Expansion Strategies:**

1. **Semantic Variations:**
   - Use synonyms and related terminology
   - Include industry-specific language variations
   - Add colloquial and formal versions
   - Incorporate different expertise level language

2. **Structural Variations:**
   - Rephrase as questions vs. statements
   - Use different query lengths (short, medium, long)
   - Include specific vs. general formulations
   - Add temporal or contextual modifiers

3. **Intent Diversification:**
   - Include informational, navigational, and transactional variations
   - Add comparison and evaluation angles
   - Include troubleshooting and how-to perspectives
   - Consider different user expertise levels

4. **Semantic Enrichment:**
   - Add related concepts and entities
   - Include broader and narrower topic variations
   - Incorporate contextual industry terms
   - Consider geographical or demographic variations

**Quality Standards:**
- Each query must be distinct and non-redundant
- Maintain relevance to the original search intent
- Ensure natural language flow
- Optimize for different search engines and databases
- Balance specificity with discoverability

**Original Query:** {question}

**Instructions:**
Generate exactly {num_queries} diverse search variations that would help retrieve comprehensive, relevant information for the original query. Each variation should:
- Capture a different aspect or angle of the topic
- Use different vocabulary and phrasing
- Maintain relevance to the core search intent
- Be optimized for effective information retrieval

**Query Variations:**
1."""
    
    async def expand_query(
        self,
        query: str,
        strategy: QueryExpansionStrategy = QueryExpansionStrategy.SEMANTIC_EXPANSION,
        num_queries: int = 3
    ) -> List[str]:
        """Expand a query into multiple variations using the specified strategy"""
        try:
            cache_key = f"{query}:{strategy.value}:{num_queries}"
            if cache_key in self.expansion_cache:
                return self.expansion_cache[cache_key]
            
            # Use the improved template
            prompt = ChatPromptTemplate.from_template(self.expansion_template)
            
            # Create expansion chain
            expansion_chain = prompt | self.llm | StrOutputParser()
            
            # Generate expanded queries
            result = await expansion_chain.ainvoke({
                "question": query,
                "num_queries": num_queries
            })
            
            # Parse the result into individual queries
            lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
            queries = []
            
            for line in lines:
                # Remove numbering if present
                clean_query = re.sub(r'^\d+\.?\s*', '', line).strip()
                if clean_query and clean_query != query:
                    queries.append(clean_query)
            
            # Ensure we have the requested number of queries
            if len(queries) < num_queries:
                queries.extend([query] * (num_queries - len(queries)))
            
            # Cache the result
            self.expansion_cache[cache_key] = queries[:num_queries]
            
            return queries[:num_queries]
            
        except Exception as e:
            print(f"❌ Query expansion failed: {e}")
            return [query] * num_queries  # Return original query as fallback


class ResultAggregator:
    """Aggregates and deduplicates results from multiple queries."""
    
    def __init__(self, config: MultiQueryConfig):
        """Initialize result aggregator."""
        self.config = config
    
    def aggregate_results(
        self,
        original_query: str,
        expanded_queries: List[ExpandedQuery],
        search_results: Dict[str, HybridSearchResults]
    ) -> List[SearchResult]:
        """Aggregate results from multiple query searches."""
        
        start_time = time.time()
        
        # Collect all results with their source queries
        all_results = []
        query_weights = self._calculate_query_weights(original_query, expanded_queries)
        
        # Process original query results
        if original_query in search_results:
            original_results = search_results[original_query].results
            for result in original_results:
                result.metadata['source_query'] = original_query
                result.metadata['query_weight'] = query_weights.get(original_query, 1.0)
                result.metadata['is_original'] = True
                all_results.append(result)
        
        # Process expanded query results
        for expanded_query in expanded_queries:
            query_text = expanded_query.query_text
            if query_text in search_results:
                expanded_results = search_results[query_text].results
                weight = query_weights.get(query_text, expanded_query.confidence)
                
                for result in expanded_results:
                    result.metadata['source_query'] = query_text
                    result.metadata['query_weight'] = weight
                    result.metadata['expansion_strategy'] = expanded_query.expansion_strategy.value
                    result.metadata['expansion_confidence'] = expanded_query.confidence
                    result.metadata['is_original'] = False
                    all_results.append(result)
        
        # Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        # Aggregate scores
        aggregated_results = self._aggregate_scores(unique_results)
        
        # Sort by aggregated score
        aggregated_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Limit results
        final_results = aggregated_results[:self.config.max_final_results]
        
        aggregation_time = time.time() - start_time
        logger.info(f"Aggregated {len(all_results)} results into {len(final_results)} unique results in {aggregation_time:.2f}s")
        
        return final_results
    
    def _calculate_query_weights(self, original_query: str, expanded_queries: List[ExpandedQuery]) -> Dict[str, float]:
        """Calculate weights for different queries based on their characteristics."""
        
        weights = {original_query: 1.0}  # Original query gets full weight
        
        for expanded_query in expanded_queries:
            # Base weight from expansion confidence
            weight = expanded_query.confidence
            
            # Strategy-specific adjustments
            if expanded_query.expansion_strategy == QueryExpansionStrategy.SEMANTIC_EXPANSION:
                weight *= 0.9  # Slight reduction for semantic variations
            elif expanded_query.expansion_strategy == QueryExpansionStrategy.PERSPECTIVE_EXPANSION:
                weight *= 0.8  # More reduction for perspective changes
            elif expanded_query.expansion_strategy == QueryExpansionStrategy.DOMAIN_EXPANSION:
                weight *= 1.1  # Slight boost for domain-specific queries
            
            weights[expanded_query.query_text] = weight
        
        return weights
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate documents from results."""
        
        unique_results = []
        seen_docs = set()
        
        for result in results:
            # Create document identifier
            doc_id = self._get_document_id(result.document)
            
            if doc_id not in seen_docs:
                unique_results.append(result)
                seen_docs.add(doc_id)
            else:
                # If duplicate, find existing result and potentially update metadata
                for existing_result in unique_results:
                    if self._get_document_id(existing_result.document) == doc_id:
                        # Merge metadata from multiple queries
                        self._merge_result_metadata(existing_result, result)
                        break
        
        return unique_results
    
    def _get_document_id(self, document: Document) -> str:
        """Generate unique identifier for a document."""
        
        # Use content hash as primary identifier
        content_hash = hashlib.md5(document.page_content.encode()).hexdigest()
        
        # Include metadata if available
        metadata_str = ""
        if document.metadata:
            # Sort metadata for consistency
            sorted_metadata = sorted(document.metadata.items())
            metadata_str = json.dumps(sorted_metadata, sort_keys=True)
        
        combined = f"{content_hash}_{hashlib.md5(metadata_str.encode()).hexdigest()}"
        return combined
    
    def _merge_result_metadata(self, existing_result: SearchResult, new_result: SearchResult):
        """Merge metadata from duplicate results."""
        
        # Track all source queries
        if 'all_source_queries' not in existing_result.metadata:
            existing_result.metadata['all_source_queries'] = [existing_result.metadata.get('source_query', '')]
        
        new_source_query = new_result.metadata.get('source_query', '')
        if new_source_query not in existing_result.metadata['all_source_queries']:
            existing_result.metadata['all_source_queries'].append(new_source_query)
        
        # Update scores if new result has better scores
        if new_result.hybrid_score > existing_result.hybrid_score:
            existing_result.hybrid_score = new_result.hybrid_score
            existing_result.dense_score = max(existing_result.dense_score, new_result.dense_score)
            existing_result.sparse_score = max(existing_result.sparse_score, new_result.sparse_score)
        
        # Track query diversity
        existing_result.metadata['query_diversity'] = len(existing_result.metadata['all_source_queries'])
    
    def _aggregate_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Aggregate scores for results considering query weights and diversity."""
        
        for result in results:
            # Base score from hybrid search
            base_score = result.hybrid_score
            
            # Weight adjustment
            query_weight = result.metadata.get('query_weight', 1.0)
            weighted_score = base_score * query_weight
            
            # Diversity bonus for results found by multiple queries
            query_diversity = result.metadata.get('query_diversity', 1)
            diversity_bonus = min(0.2, (query_diversity - 1) * 0.05)  # Up to 20% bonus
            
            # Original query bonus
            original_bonus = 0.1 if result.metadata.get('is_original', False) else 0.0
            
            # Calculate final aggregated score
            final_score = weighted_score + diversity_bonus + original_bonus
            result.hybrid_score = min(1.0, final_score)  # Cap at 1.0
            
            # Update metadata
            result.metadata['aggregated_score'] = final_score
            result.metadata['diversity_bonus'] = diversity_bonus
            result.metadata['original_bonus'] = original_bonus
        
        return results


class MultiQueryRetriever:
    """Main multi-query retrieval system orchestrator."""
    
    def __init__(
        self,
        hybrid_search_engine: Union[HybridSearchEngine, ContextualHybridSearch],
        config: Optional[MultiQueryConfig] = None
    ):
        """Initialize multi-query retriever."""
        self.hybrid_search_engine = hybrid_search_engine
        self.config = config or MultiQueryConfig()
        
        # Initialize components
        self.query_expander = QueryExpander(self.config)
        self.result_aggregator = ResultAggregator(self.config)
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'avg_expansion_time': 0.0,
            'avg_search_time': 0.0,
            'avg_aggregation_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info("MultiQueryRetriever initialized")
    
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        search_type: SearchType = SearchType.HYBRID,
        max_results: Optional[int] = None
    ) -> MultiQueryResults:
        """Perform multi-query retrieval with expansion and aggregation."""
        
        start_time = time.time()
        max_results = max_results or self.config.max_final_results
        
        try:
            # Phase 1: Query Expansion
            expansion_start = time.time()
            expanded_queries = await self.query_expander.expand_query(query)
            expansion_time = time.time() - expansion_start
            
            # Phase 2: Parallel Search
            search_start = time.time()
            search_results = await self._perform_parallel_searches(
                query, expanded_queries, search_type, max_results
            )
            search_time = time.time() - search_start
            
            # Phase 3: Result Aggregation
            aggregation_start = time.time()
            aggregated_results = self.result_aggregator.aggregate_results(
                query, expanded_queries, search_results
            )
            aggregation_time = time.time() - aggregation_start
            
            total_time = time.time() - start_time
            
            # Update performance statistics
            self._update_performance_stats(expansion_time, search_time, aggregation_time)
            
            # Count unique documents
            unique_docs = len(set(self.result_aggregator._get_document_id(result.document) 
                                for result in aggregated_results))
            
            return MultiQueryResults(
                aggregated_results=aggregated_results,
                original_query=query,
                expanded_queries=expanded_queries,
                individual_results=search_results,
                total_unique_documents=unique_docs,
                expansion_time=expansion_time,
                search_time=search_time,
                aggregation_time=aggregation_time,
                total_time=total_time,
                performance_metadata={
                    'query_type': query_type.value,
                    'search_type': search_type.value,
                    'num_expansions': len(expanded_queries),
                    'total_searches': len(search_results),
                    'final_results': len(aggregated_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-query retrieval failed: {e}")
            
            # Return minimal results on error
            return MultiQueryResults(
                aggregated_results=[],
                original_query=query,
                expanded_queries=[],
                individual_results={},
                total_unique_documents=0,
                total_time=time.time() - start_time,
                performance_metadata={'error': str(e)}
            )
    
    async def _perform_parallel_searches(
        self,
        original_query: str,
        expanded_queries: List[ExpandedQuery],
        search_type: SearchType,
        max_results: int
    ) -> Dict[str, HybridSearchResults]:
        """Perform parallel searches for all queries."""
        
        # Prepare all queries to search
        all_queries = [original_query] + [eq.query_text for eq in expanded_queries]
        
        if self.config.enable_parallel_search:
            # Create semaphore to limit concurrent searches
            semaphore = asyncio.Semaphore(self.config.max_concurrent_queries)
            
            # Create search tasks
            search_tasks = [
                self._limited_search(semaphore, query, search_type, max_results)
                for query in all_queries
            ]
            
            # Execute searches with timeout
            try:
                search_results = await asyncio.wait_for(
                    asyncio.gather(*search_tasks, return_exceptions=True),
                    timeout=self.config.search_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Search timeout after {self.config.search_timeout}s")
                search_results = [None] * len(all_queries)
        else:
            # Sequential search
            search_results = []
            for query in all_queries:
                try:
                    result = await self.hybrid_search_engine.search(query, search_type, max_results)
                    search_results.append(result)
                except Exception as e:
                    logger.error(f"Search failed for query '{query}': {e}")
                    search_results.append(None)
        
        # Process results
        results_dict = {}
        for i, (query, result) in enumerate(zip(all_queries, search_results)):
            if result is not None and not isinstance(result, Exception):
                results_dict[query] = result
            else:
                logger.warning(f"No results for query: {query}")
        
        return results_dict
    
    async def _limited_search(
        self,
        semaphore: asyncio.Semaphore,
        query: str,
        search_type: SearchType,
        max_results: int
    ) -> HybridSearchResults:
        """Perform search with semaphore limiting."""
        async with semaphore:
            if hasattr(self.hybrid_search_engine, 'contextual_search'):
                # ContextualHybridSearch
                return await self.hybrid_search_engine.contextual_search(query, search_type, max_results)
            else:
                # Regular HybridSearchEngine
                return await self.hybrid_search_engine.search(query, search_type, max_results)
    
    def _update_performance_stats(self, expansion_time: float, search_time: float, aggregation_time: float):
        """Update running performance statistics."""
        self.performance_stats['total_queries'] += 1
        n = self.performance_stats['total_queries']
        
        # Running averages
        self.performance_stats['avg_expansion_time'] = (
            (self.performance_stats['avg_expansion_time'] * (n - 1) + expansion_time) / n
        )
        self.performance_stats['avg_search_time'] = (
            (self.performance_stats['avg_search_time'] * (n - 1) + search_time) / n
        )
        self.performance_stats['avg_aggregation_time'] = (
            (self.performance_stats['avg_aggregation_time'] * (n - 1) + aggregation_time) / n
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'total_queries': self.performance_stats['total_queries'],
            'average_times': {
                'expansion_ms': self.performance_stats['avg_expansion_time'] * 1000,
                'search_ms': self.performance_stats['avg_search_time'] * 1000,
                'aggregation_ms': self.performance_stats['avg_aggregation_time'] * 1000,
                'total_avg_ms': (
                    self.performance_stats['avg_expansion_time'] +
                    self.performance_stats['avg_search_time'] +
                    self.performance_stats['avg_aggregation_time']
                ) * 1000
            },
            'configuration': {
                'num_expansions': self.config.num_expansions,
                'parallel_search': self.config.enable_parallel_search,
                'max_concurrent': self.config.max_concurrent_queries,
                'llm_model': self.config.llm_model
            },
            'cache_performance': {
                'expansion_cache_size': len(self.query_expander.expansion_cache),
                'cache_hits': self.performance_stats['cache_hits']
            }
        }
    
    async def clear_caches(self):
        """Clear all caches to free memory."""
        self.query_expander.expansion_cache.clear()
        logger.info("Multi-query retrieval caches cleared") 