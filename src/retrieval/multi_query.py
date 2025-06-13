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
    llm_model: str = Field(default="gpt-3.5-turbo", description="LLM model for query expansion")
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
                model="gpt-3.5-turbo",
                temperature=config.llm_temperature,
                max_tokens=config.max_tokens
            )
        
        # Expansion cache for performance
        self.expansion_cache = {}
        
        # Expansion prompt templates
        self.expansion_prompts = {
            QueryExpansionStrategy.SEMANTIC_EXPANSION: ChatPromptTemplate.from_template(
                """Generate 3 semantically similar queries to find the same information as the original query.
                
Original Query: {query}

Focus on:
- Synonyms and related terms
- Alternative phrasings
- Different ways to express the same concept

Return only the queries, one per line, without numbering or explanation."""
            ),
            
            QueryExpansionStrategy.PERSPECTIVE_EXPANSION: ChatPromptTemplate.from_template(
                """Generate 3 queries that approach the same topic from different perspectives or angles.
                
Original Query: {query}

Focus on:
- Different viewpoints or approaches
- Various aspects of the topic
- Alternative framings of the question

Return only the queries, one per line, without numbering or explanation."""
            ),
            
            QueryExpansionStrategy.SPECIFICITY_EXPANSION: ChatPromptTemplate.from_template(
                """Generate 3 queries with different levels of specificity about the same topic.
                
Original Query: {query}

Generate:
- 1 more specific/detailed query
- 1 more general/broader query  
- 1 differently focused query

Return only the queries, one per line, without numbering or explanation."""
            ),
            
            QueryExpansionStrategy.CONTEXTUAL_EXPANSION: ChatPromptTemplate.from_template(
                """Generate 3 queries that add relevant context to better find comprehensive information.
                
Original Query: {query}

Focus on:
- Adding relevant background context
- Including related concepts
- Expanding with domain-specific terms

Return only the queries, one per line, without numbering or explanation."""
            ),
            
            QueryExpansionStrategy.DOMAIN_EXPANSION: ChatPromptTemplate.from_template(
                """Generate 3 domain-specific variations of the query using technical terminology.
                
Original Query: {query}

Focus on:
- Technical/professional terminology
- Industry-specific language
- Expert-level phrasing

Return only the queries, one per line, without numbering or explanation."""
            )
        }
    
    async def expand_query(
        self, 
        query: str, 
        query_type: QueryType = QueryType.GENERAL,
        strategies: Optional[List[QueryExpansionStrategy]] = None
    ) -> List[ExpandedQuery]:
        """Expand a query using specified strategies."""
        
        start_time = time.time()
        
        # Use configured strategies if none specified
        if strategies is None:
            strategies = self.config.expansion_strategies
        
        # Check cache first
        cache_key = self._get_cache_key(query, strategies)
        if self.config.cache_expansions and cache_key in self.expansion_cache:
            cached_expansions = self.expansion_cache[cache_key]
            # Update generation time
            for expansion in cached_expansions:
                expansion.generation_time = time.time() - start_time
            return cached_expansions
        
        expanded_queries = []
        
        # Process each expansion strategy
        for strategy in strategies:
            try:
                strategy_expansions = await self._expand_with_strategy(query, strategy, query_type)
                expanded_queries.extend(strategy_expansions)
            except Exception as e:
                logger.error(f"Error in expansion strategy {strategy}: {e}")
                continue
        
        # Filter by confidence threshold
        filtered_expansions = [
            exp for exp in expanded_queries 
            if exp.confidence >= self.config.min_expansion_confidence
        ]
        
        # Deduplicate expansions
        unique_expansions = self._deduplicate_expansions(filtered_expansions, query)
        
        # Update generation times
        generation_time = time.time() - start_time
        for expansion in unique_expansions:
            expansion.generation_time = generation_time
        
        # Cache results
        if self.config.cache_expansions:
            self.expansion_cache[cache_key] = unique_expansions
        
        logger.info(f"Generated {len(unique_expansions)} unique query expansions in {generation_time:.2f}s")
        return unique_expansions
    
    async def _expand_with_strategy(
        self, 
        query: str, 
        strategy: QueryExpansionStrategy,
        query_type: QueryType
    ) -> List[ExpandedQuery]:
        """Expand query using a specific strategy."""
        
        if strategy not in self.expansion_prompts:
            logger.warning(f"No prompt template for strategy: {strategy}")
            return []
        
        try:
            # Get the appropriate prompt template
            prompt_template = self.expansion_prompts[strategy]
            
            # Create chain
            chain = prompt_template | self.llm | StrOutputParser()
            
            # Generate expansions
            result = await chain.ainvoke({"query": query})
            
            # Parse the result into individual queries
            expanded_queries = []
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            
            for line in lines:
                # Clean up the line (remove numbers, bullets, etc.)
                cleaned_query = self._clean_expansion(line)
                
                if cleaned_query and cleaned_query.lower() != query.lower():
                    # Calculate confidence based on quality indicators
                    confidence = self._calculate_expansion_confidence(cleaned_query, query, strategy)
                    
                    expansion = ExpandedQuery(
                        query_text=cleaned_query,
                        expansion_strategy=strategy,
                        confidence=confidence,
                        metadata={
                            "original_query": query,
                            "query_type": query_type.value,
                            "strategy": strategy.value
                        }
                    )
                    expanded_queries.append(expansion)
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error generating expansions with strategy {strategy}: {e}")
            return []
    
    def _clean_expansion(self, expansion: str) -> str:
        """Clean and normalize expanded query."""
        import re
        
        # Remove numbering, bullets, quotes
        expansion = re.sub(r'^\d+[\.\)]\s*', '', expansion)
        expansion = re.sub(r'^[\-\*•]\s*', '', expansion)
        expansion = expansion.strip('"\'')
        expansion = expansion.strip()
        
        # Remove empty or very short expansions
        if len(expansion) < 10:
            return ""
        
        return expansion
    
    def _calculate_expansion_confidence(
        self, 
        expansion: str, 
        original: str, 
        strategy: QueryExpansionStrategy
    ) -> float:
        """Calculate confidence score for expanded query."""
        
        confidence = 0.5  # Base confidence
        
        # Length-based confidence
        if 10 <= len(expansion) <= 200:
            confidence += 0.2
        elif len(expansion) > 200:
            confidence -= 0.1
        
        # Similarity to original (not too similar, not too different)
        original_words = set(original.lower().split())
        expansion_words = set(expansion.lower().split())
        
        if original_words and expansion_words:
            overlap = len(original_words & expansion_words) / len(original_words | expansion_words)
            
            # Optimal overlap is 20-60%
            if 0.2 <= overlap <= 0.6:
                confidence += 0.2
            elif overlap > 0.8:  # Too similar
                confidence -= 0.3
            elif overlap < 0.1:  # Too different
                confidence -= 0.2
        
        # Strategy-specific adjustments
        if strategy == QueryExpansionStrategy.SEMANTIC_EXPANSION:
            # Semantic expansions should have moderate overlap
            if 0.3 <= overlap <= 0.7:
                confidence += 0.1
        elif strategy == QueryExpansionStrategy.PERSPECTIVE_EXPANSION:
            # Perspective expansions can have less overlap
            if overlap <= 0.5:
                confidence += 0.1
        
        # Quality indicators
        if any(word in expansion.lower() for word in ['what', 'how', 'why', 'when', 'where']):
            confidence += 0.1  # Question words indicate good query structure
        
        return max(0.0, min(1.0, confidence))
    
    def _deduplicate_expansions(self, expansions: List[ExpandedQuery], original_query: str) -> List[ExpandedQuery]:
        """Remove duplicate and very similar expansions."""
        
        unique_expansions = []
        seen_queries = {original_query.lower()}
        
        # Sort by confidence (highest first)
        expansions.sort(key=lambda x: x.confidence, reverse=True)
        
        for expansion in expansions:
            query_lower = expansion.query_text.lower()
            
            # Check for exact duplicates
            if query_lower in seen_queries:
                continue
            
            # Check for high similarity
            is_similar = False
            for seen_query in seen_queries:
                similarity = self._calculate_query_similarity(query_lower, seen_query)
                if similarity > self.config.deduplication_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_expansions.append(expansion)
                seen_queries.add(query_lower)
                
                # Limit number of expansions
                if len(unique_expansions) >= self.config.num_expansions:
                    break
        
        return unique_expansions
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_cache_key(self, query: str, strategies: List[QueryExpansionStrategy]) -> str:
        """Generate cache key for query expansion."""
        strategy_str = "_".join(sorted([s.value for s in strategies]))
        cache_input = f"{query}_{strategy_str}_{self.config.llm_temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()


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
            expanded_queries = await self.query_expander.expand_query(query, query_type)
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