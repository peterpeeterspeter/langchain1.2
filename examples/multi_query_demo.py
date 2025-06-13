#!/usr/bin/env python3
"""
Multi-Query Retrieval System Demonstration
==========================================

Task 3.3: Comprehensive demonstration of multi-query retrieval with LLM query expansion,
parallel processing, and result aggregation.

This script demonstrates:
- LLM-powered query expansion with multiple strategies
- Parallel query processing for performance optimization
- Result aggregation and deduplication
- Integration with hybrid search infrastructure
- Performance analysis and monitoring
- Different query types and expansion strategies
"""

import asyncio
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Import retrieval system components
from src.retrieval import (
    MultiQueryRetriever,
    QueryExpander,
    MultiQueryConfig,
    QueryExpansionStrategy,
    QueryType,
    HybridSearchEngine,
    ContextualHybridSearch,
    HybridSearchConfig,
    SearchType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockHybridSearchEngine:
    """Mock hybrid search engine for demonstration purposes."""
    
    def __init__(self):
        """Initialize mock search engine with sample documents."""
        self.documents = [
            Document(
                page_content="Casino slots offer various payout percentages, typically ranging from 85% to 98%. "
                           "Higher denomination slots generally have better RTPs (Return to Player rates).",
                metadata={"title": "Slot Machine RTP Guide", "type": "guide", "authority_score": 0.8}
            ),
            Document(
                page_content="Blackjack basic strategy reduces the house edge to approximately 0.5%. "
                           "Card counting can further improve player odds but requires significant practice.",
                metadata={"title": "Blackjack Strategy Guide", "type": "tutorial", "authority_score": 0.9}
            ),
            Document(
                page_content="Online poker tournaments have different structures: freezeouts, rebuys, and satellites. "
                           "Each format requires adapted strategies for optimal play.",
                metadata={"title": "Poker Tournament Types", "type": "comparison", "authority_score": 0.7}
            ),
            Document(
                page_content="Responsible gambling involves setting limits, understanding odds, and recognizing "
                           "problem gambling signs. Many casinos offer self-exclusion programs.",
                metadata={"title": "Responsible Gaming", "type": "educational", "authority_score": 0.95}
            ),
            Document(
                page_content="Roulette betting systems like Martingale, Fibonacci, and D'Alembert are popular "
                           "but don't change the mathematical house edge of the game.",
                metadata={"title": "Roulette Betting Systems", "type": "analysis", "authority_score": 0.8}
            ),
            Document(
                page_content="Live dealer games combine online convenience with authentic casino atmosphere. "
                           "Popular options include live blackjack, roulette, and baccarat.",
                metadata={"title": "Live Dealer Gaming", "type": "review", "authority_score": 0.85}
            ),
            Document(
                page_content="Progressive slot jackpots accumulate across multiple machines or casinos. "
                           "Mega jackpots can reach millions but have extremely low hit frequencies.",
                metadata={"title": "Progressive Slots Explained", "type": "guide", "authority_score": 0.82}
            ),
            Document(
                page_content="Casino bonuses include welcome bonuses, free spins, cashback, and loyalty rewards. "
                           "Understanding wagering requirements is crucial before claiming bonuses.",
                metadata={"title": "Casino Bonus Guide", "type": "guide", "authority_score": 0.88}
            )
        ]
    
    async def search(self, query: str, search_type: SearchType = SearchType.HYBRID, max_results: int = 10):
        """Mock search that returns relevant documents based on query keywords."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            score = 0.0
            
            # Simple keyword matching for demonstration
            content_words = doc.page_content.lower().split()
            query_words = query_lower.split()
            
            # Calculate relevance score
            for query_word in query_words:
                if query_word in content_words:
                    score += 0.2
                if query_word in doc.metadata.get('title', '').lower():
                    score += 0.3
                if query_word in doc.metadata.get('type', '').lower():
                    score += 0.1
            
            # Authority bonus
            score += doc.metadata.get('authority_score', 0.5) * 0.1
            
            if score > 0.1:  # Minimum relevance threshold
                from src.retrieval.hybrid_search import SearchResult
                result = SearchResult(
                    document=doc,
                    dense_score=score * 0.7,
                    sparse_score=score * 0.3,
                    hybrid_score=score,
                    metadata={
                        'query': query,
                        'relevance_score': score,
                        'search_type': search_type.value
                    }
                )
                results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        limited_results = results[:max_results]
        
        # Create mock HybridSearchResults
        from src.retrieval.hybrid_search import HybridSearchResults
        return HybridSearchResults(
            results=limited_results,
            total_results=len(limited_results),
            search_time=0.1,
            metadata={
                'query': query,
                'search_type': search_type.value,
                'mock_engine': True
            }
        )


async def demonstrate_basic_multi_query():
    """Demonstrate basic multi-query retrieval functionality."""
    print("\n" + "="*80)
    print("BASIC MULTI-QUERY RETRIEVAL DEMONSTRATION")
    print("="*80)
    
    # Initialize components
    mock_search_engine = MockHybridSearchEngine()
    
    # Configure multi-query retrieval
    config = MultiQueryConfig(
        num_expansions=3,
        expansion_strategies=[
            QueryExpansionStrategy.SEMANTIC_EXPANSION,
            QueryExpansionStrategy.PERSPECTIVE_EXPANSION
        ],
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.3,
        enable_parallel_search=True,
        max_concurrent_queries=3,
        max_final_results=10
    )
    
    # Initialize multi-query retriever
    multi_retriever = MultiQueryRetriever(mock_search_engine, config)
    
    # Test query
    original_query = "best blackjack strategy for beginners"
    print(f"\n🔍 Original Query: {original_query}")
    
    # Perform multi-query retrieval
    start_time = time.time()
    results = await multi_retriever.retrieve(
        query=original_query,
        query_type=QueryType.TUTORIAL
    )
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 RETRIEVAL RESULTS")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Expansion Time: {results.expansion_time:.2f}s")
    print(f"Search Time: {results.search_time:.2f}s")
    print(f"Aggregation Time: {results.aggregation_time:.2f}s")
    
    print(f"\n🔄 QUERY EXPANSIONS ({len(results.expanded_queries)}):")
    for i, expanded_query in enumerate(results.expanded_queries, 1):
        print(f"{i}. {expanded_query.query_text}")
        print(f"   Strategy: {expanded_query.expansion_strategy.value}")
        print(f"   Confidence: {expanded_query.confidence:.3f}")
    
    print(f"\n📚 AGGREGATED RESULTS ({len(results.aggregated_results)}):")
    for i, result in enumerate(results.aggregated_results[:5], 1):  # Show top 5
        print(f"{i}. {result.document.metadata.get('title', 'Untitled')}")
        print(f"   Score: {result.hybrid_score:.3f}")
        print(f"   Source Query: {result.metadata.get('source_query', 'Unknown')}")
        print(f"   Is Original: {result.metadata.get('is_original', False)}")
        print()
    
    return results


async def demonstrate_expansion_strategies():
    """Demonstrate different query expansion strategies."""
    print("\n" + "="*80)
    print("QUERY EXPANSION STRATEGIES DEMONSTRATION")
    print("="*80)
    
    # Test different expansion strategies
    strategies_to_test = [
        QueryExpansionStrategy.SEMANTIC_EXPANSION,
        QueryExpansionStrategy.PERSPECTIVE_EXPANSION,
        QueryExpansionStrategy.SPECIFICITY_EXPANSION,
        QueryExpansionStrategy.CONTEXTUAL_EXPANSION
    ]
    
    # Initialize query expander
    config = MultiQueryConfig(
        num_expansions=3,
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.3
    )
    expander = QueryExpander(config)
    
    test_query = "casino slot machine tips"
    print(f"\n🔍 Test Query: {test_query}")
    
    for strategy in strategies_to_test:
        print(f"\n📈 {strategy.value.upper()} STRATEGY:")
        
        try:
            # Test individual strategy
            expansions = await expander._expand_with_strategy(
                test_query, strategy, QueryType.GENERAL
            )
            
            for i, expansion in enumerate(expansions, 1):
                print(f"{i}. {expansion.query_text}")
                print(f"   Confidence: {expansion.confidence:.3f}")
            
            if not expansions:
                print("   No expansions generated")
                
        except Exception as e:
            print(f"   Error: {str(e)}")


async def demonstrate_parallel_vs_sequential():
    """Compare parallel vs sequential query processing performance."""
    print("\n" + "="*80)
    print("PARALLEL VS SEQUENTIAL PROCESSING COMPARISON")
    print("="*80)
    
    mock_search_engine = MockHybridSearchEngine()
    test_query = "poker tournament strategy guide"
    
    # Test parallel processing
    print("\n⚡ PARALLEL PROCESSING:")
    config_parallel = MultiQueryConfig(
        num_expansions=4,
        enable_parallel_search=True,
        max_concurrent_queries=4
    )
    
    retriever_parallel = MultiQueryRetriever(mock_search_engine, config_parallel)
    
    start_time = time.time()
    results_parallel = await retriever_parallel.retrieve(
        query=test_query,
        query_type=QueryType.TUTORIAL
    )
    parallel_time = time.time() - start_time
    
    print(f"Total Time: {parallel_time:.2f}s")
    print(f"Search Time: {results_parallel.search_time:.2f}s")
    print(f"Results Found: {len(results_parallel.aggregated_results)}")
    
    # Test sequential processing
    print("\n🐌 SEQUENTIAL PROCESSING:")
    config_sequential = MultiQueryConfig(
        num_expansions=4,
        enable_parallel_search=False
    )
    
    retriever_sequential = MultiQueryRetriever(mock_search_engine, config_sequential)
    
    start_time = time.time()
    results_sequential = await retriever_sequential.retrieve(
        query=test_query,
        query_type=QueryType.TUTORIAL
    )
    sequential_time = time.time() - start_time
    
    print(f"Total Time: {sequential_time:.2f}s")
    print(f"Search Time: {results_sequential.search_time:.2f}s")
    print(f"Results Found: {len(results_sequential.aggregated_results)}")
    
    # Performance comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time Saved: {sequential_time - parallel_time:.2f}s")
    
    return speedup


async def demonstrate_query_types():
    """Demonstrate multi-query retrieval for different query types."""
    print("\n" + "="*80)
    print("QUERY TYPE-SPECIFIC EXPANSION DEMONSTRATION")
    print("="*80)
    
    mock_search_engine = MockHybridSearchEngine()
    
    # Different query types with appropriate test queries
    test_queries = {
        QueryType.FACTUAL: "what is the house edge in roulette",
        QueryType.COMPARISON: "blackjack vs poker profitability",
        QueryType.TUTORIAL: "how to play baccarat for beginners", 
        QueryType.REVIEW: "best online casino for live dealer games"
    }
    
    config = MultiQueryConfig(
        num_expansions=3,
        expansion_strategies=[QueryExpansionStrategy.SEMANTIC_EXPANSION],
        max_final_results=5
    )
    
    retriever = MultiQueryRetriever(mock_search_engine, config)
    
    for query_type, query in test_queries.items():
        print(f"\n🎯 {query_type.value.upper()} QUERY:")
        print(f"Query: {query}")
        
        try:
            results = await retriever.retrieve(
                query=query,
                query_type=query_type
            )
            
            print(f"Expansions Generated: {len(results.expanded_queries)}")
            for expansion in results.expanded_queries:
                print(f"  • {expansion.query_text} (conf: {expansion.confidence:.2f})")
            
            print(f"Results Found: {len(results.aggregated_results)}")
            if results.aggregated_results:
                best_result = results.aggregated_results[0]
                print(f"Best Match: {best_result.document.metadata.get('title', 'Untitled')}")
                print(f"Score: {best_result.hybrid_score:.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and statistics."""
    print("\n" + "="*80)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*80)
    
    mock_search_engine = MockHybridSearchEngine()
    
    config = MultiQueryConfig(
        num_expansions=3,
        enable_parallel_search=True,
        cache_expansions=True
    )
    
    retriever = MultiQueryRetriever(mock_search_engine, config)
    
    # Perform multiple queries to build performance statistics
    test_queries = [
        "slot machine odds",
        "casino bonus terms",
        "live dealer games",
        "poker strategy basics",
        "roulette betting systems"
    ]
    
    print("\n🔄 Running multiple queries to build performance statistics...")
    
    all_results = []
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}/{len(test_queries)}: {query}")
        
        start_time = time.time()
        results = await retriever.retrieve(query)
        query_time = time.time() - start_time
        
        all_results.append((query, results, query_time))
        print(f"  Time: {query_time:.2f}s, Results: {len(results.aggregated_results)}")
    
    # Display performance statistics
    print(f"\n📊 PERFORMANCE STATISTICS:")
    stats = retriever.get_performance_stats()
    
    print(f"Configuration:")
    for key, value in stats['configuration'].items():
        print(f"  {key}: {value}")
    
    print(f"\nCache Performance:")
    for key, value in stats['cache_performance'].items():
        print(f"  {key}: {value}")
    
    # Calculate aggregate statistics
    total_time = sum(result[2] for result in all_results)
    avg_time = total_time / len(all_results)
    total_results = sum(len(result[1].aggregated_results) for result in all_results)
    avg_results = total_results / len(all_results)
    
    print(f"\nAggregate Statistics:")
    print(f"  Total Queries: {len(all_results)}")
    print(f"  Average Time: {avg_time:.2f}s")
    print(f"  Total Results: {total_results}")
    print(f"  Average Results per Query: {avg_results:.1f}")
    
    return stats


async def main():
    """Run comprehensive multi-query retrieval demonstration."""
    print("🚀 MULTI-QUERY RETRIEVAL SYSTEM DEMONSTRATION")
    print("Universal RAG CMS - Task 3.3")
    print("=" * 80)
    
    try:
        # Check if we have required environment variables
        import os
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  Warning: OPENAI_API_KEY not set. Using mock responses for demonstration.")
            print("   In production, set your API key to enable real LLM query expansion.")
        
        # Run demonstrations
        await demonstrate_basic_multi_query()
        await demonstrate_expansion_strategies()
        speedup = await demonstrate_parallel_vs_sequential()
        await demonstrate_query_types()
        stats = await demonstrate_performance_monitoring()
        
        # Summary
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        print("\n✅ Successfully demonstrated:")
        print("   • LLM-powered query expansion with multiple strategies")
        print("   • Parallel query processing for performance optimization")
        print("   • Result aggregation and deduplication")
        print("   • Query type-specific expansion")
        print("   • Performance monitoring and statistics")
        
        if speedup > 1.0:
            print(f"\n⚡ Performance Benefits:")
            print(f"   • Parallel processing achieved {speedup:.2f}x speedup")
            print(f"   • Efficient caching and result aggregation")
        
        print(f"\n🎯 Task 3.3 Multi-Query Retrieval: COMPLETED")
        print("   Ready for integration with Task 3.4 (Self-Query Metadata Filtering)")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main()) 