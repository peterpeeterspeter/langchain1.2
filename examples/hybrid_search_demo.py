#!/usr/bin/env python3
"""
Hybrid Search Infrastructure Demonstration
==========================================

Task 3.2: Comprehensive demonstration of hybrid search combining dense vector 
and sparse BM25 search with various fusion methods and configurations.

This script demonstrates:
- Dense vector search using embeddings
- Sparse BM25 keyword search 
- Hybrid search with multiple fusion algorithms
- Performance comparison between search methods
- Integration with contextual embedding system
- Configuration optimization and tuning
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

from src.retrieval import (
    HybridSearchEngine,
    ContextualHybridSearch, 
    HybridSearchConfig,
    SearchType,
    FusionMethod,
    ContextualEmbeddingSystem,
    RetrievalConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        metadata={"title": "Python Programming Overview", "category": "programming", "difficulty": "beginner"}
    ),
    Document(
        page_content="Machine learning algorithms can be supervised, unsupervised, or reinforcement learning. Supervised learning uses labeled data to train models for prediction tasks.",
        metadata={"title": "Machine Learning Fundamentals", "category": "ai", "difficulty": "intermediate"}
    ),
    Document(
        page_content="FastAPI is a modern web framework for building APIs with Python. It features automatic API documentation, type hints, and high performance comparable to NodeJS.",
        metadata={"title": "FastAPI Web Framework", "category": "web", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Vector databases store and query high-dimensional vector embeddings. They enable semantic search, recommendation systems, and similarity matching at scale.",
        metadata={"title": "Vector Database Technology", "category": "database", "difficulty": "advanced"}
    ),
    Document(
        page_content="Retrieval Augmented Generation (RAG) combines information retrieval with large language models. It enhances LLM responses with relevant context from knowledge bases.",
        metadata={"title": "RAG Architecture", "category": "ai", "difficulty": "advanced"}
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports complex queries, full-text search, and JSON data types with ACID compliance.",
        metadata={"title": "PostgreSQL Database", "category": "database", "difficulty": "intermediate"}
    ),
    Document(
        page_content="API rate limiting prevents abuse and ensures fair usage. Common strategies include token bucket, fixed window, and sliding window algorithms.",
        metadata={"title": "API Rate Limiting Strategies", "category": "api", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Microservices architecture breaks applications into small, independent services. Each service handles a specific business function and communicates via APIs.",
        metadata={"title": "Microservices Architecture", "category": "architecture", "difficulty": "advanced"}
    ),
    Document(
        page_content="Docker containers package applications with their dependencies. They provide consistent deployment environments across development, testing, and production.",
        metadata={"title": "Docker Containerization", "category": "devops", "difficulty": "intermediate"}
    ),
    Document(
        page_content="Elasticsearch is a distributed search engine built on Apache Lucene. It provides real-time search, analytics, and data visualization capabilities.",
        metadata={"title": "Elasticsearch Search Engine", "category": "search", "difficulty": "advanced"}
    )
]

# Test queries for different scenarios
TEST_QUERIES = [
    "Python programming language features",           # Should favor dense search
    "machine learning supervised algorithms",         # Hybrid should work well
    "API rate limiting token bucket",                # Should favor sparse/keyword search
    "vector database similarity search",             # Semantic query - dense preferred
    "PostgreSQL JSON ACID",                         # Mixed semantic + keyword
    "Docker microservices deployment",              # Conceptual relationship
    "FastAPI performance web framework",            # Specific technical terms
    "RAG retrieval augmented generation",           # Acronym + full term
    "database technology advanced",                  # General + specific
    "search engine real-time analytics"             # Multiple concepts
]


class HybridSearchDemo:
    """Demonstration class for hybrid search functionality."""
    
    def __init__(self):
        """Initialize the demo with embeddings and configurations."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.documents = SAMPLE_DOCUMENTS
        
        # Different configurations to test
        self.configs = {
            "balanced": HybridSearchConfig(
                dense_weight=0.7,
                sparse_weight=0.3,
                fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
                max_results=10
            ),
            "dense_heavy": HybridSearchConfig(
                dense_weight=0.8,
                sparse_weight=0.2,
                fusion_method=FusionMethod.WEIGHTED_SUM,
                max_results=10
            ),
            "sparse_heavy": HybridSearchConfig(
                dense_weight=0.4,
                sparse_weight=0.6,
                fusion_method=FusionMethod.WEIGHTED_SUM,
                max_results=10
            ),
            "rrf_optimized": HybridSearchConfig(
                dense_weight=0.6,
                sparse_weight=0.4,
                fusion_method=FusionMethod.RECIPROCAL_RANK_FUSION,
                rrf_k=50,
                max_results=10
            )
        }
        
        self.search_engines = {}
        
    async def setup_search_engines(self):
        """Initialize search engines with different configurations."""
        logger.info("Setting up hybrid search engines...")
        
        for config_name, config in self.configs.items():
            engine = HybridSearchEngine(
                documents=self.documents,
                embeddings_model=self.embeddings,
                config=config
            )
            self.search_engines[config_name] = engine
            
        logger.info(f"Initialized {len(self.search_engines)} search engine configurations")
    
    async def demonstrate_search_types(self):
        """Demonstrate different search types with the same query."""
        print("\n" + "="*80)
        print("SEARCH TYPE COMPARISON")
        print("="*80)
        
        query = "Python programming language features"
        print(f"Query: '{query}'\n")
        
        engine = self.search_engines["balanced"]
        
        # Test each search type
        search_types = [SearchType.DENSE_ONLY, SearchType.SPARSE_ONLY, SearchType.HYBRID, SearchType.ADAPTIVE]
        
        for search_type in search_types:
            print(f"\n--- {search_type.value.upper()} SEARCH ---")
            
            results = await engine.search(query, search_type, max_results=5)
            
            print(f"Results: {results.total_results}")
            print(f"Total time: {results.total_time*1000:.1f}ms")
            print(f"Dense time: {results.dense_search_time*1000:.1f}ms")
            print(f"Sparse time: {results.sparse_search_time*1000:.1f}ms")
            if results.fusion_time > 0:
                print(f"Fusion time: {results.fusion_time*1000:.1f}ms")
            
            print("\nTop 3 Results:")
            for i, result in enumerate(results.results[:3]):
                print(f"  {i+1}. {result.document.metadata.get('title', 'No Title')}")
                print(f"     Dense: {result.dense_score:.3f}, Sparse: {result.sparse_score:.3f}, Hybrid: {result.hybrid_score:.3f}")
                print(f"     Content: {result.document.page_content[:100]}...")
    
    async def demonstrate_fusion_methods(self):
        """Demonstrate different fusion methods."""
        print("\n" + "="*80)
        print("FUSION METHOD COMPARISON") 
        print("="*80)
        
        query = "vector database similarity search"
        print(f"Query: '{query}'\n")
        
        # Test RRF vs Weighted Sum
        rrf_engine = self.search_engines["balanced"]  # Uses RRF
        weighted_engine = self.search_engines["dense_heavy"]  # Uses weighted sum
        
        print("--- RECIPROCAL RANK FUSION ---")
        rrf_results = await rrf_engine.search(query, SearchType.HYBRID, max_results=5)
        self._print_fusion_results(rrf_results)
        
        print("\n--- WEIGHTED SUM FUSION ---")  
        weighted_results = await weighted_engine.search(query, SearchType.HYBRID, max_results=5)
        self._print_fusion_results(weighted_results)
        
        # Compare result ordering
        print("\n--- RESULT ORDERING COMPARISON ---")
        print("RRF Top 3:")
        for i, result in enumerate(rrf_results.results[:3]):
            print(f"  {i+1}. {result.document.metadata.get('title', 'No Title')} (Score: {result.hybrid_score:.3f})")
        
        print("\nWeighted Sum Top 3:")
        for i, result in enumerate(weighted_results.results[:3]):
            print(f"  {i+1}. {result.document.metadata.get('title', 'No Title')} (Score: {result.hybrid_score:.3f})")
    
    def _print_fusion_results(self, results):
        """Helper to print fusion results."""
        print(f"Total time: {results.total_time*1000:.1f}ms")
        print(f"Fusion method: {results.search_metadata.get('fusion_method', 'unknown')}")
        print(f"Dense results: {results.search_metadata.get('dense_results', 0)}")
        print(f"Sparse results: {results.search_metadata.get('sparse_results', 0)}")
        print(f"Final results: {results.total_results}")
    
    async def demonstrate_configuration_impact(self):
        """Demonstrate impact of different configurations."""
        print("\n" + "="*80)
        print("CONFIGURATION IMPACT ANALYSIS")
        print("="*80)
        
        query = "API rate limiting token bucket"
        print(f"Query: '{query}'\n")
        
        config_results = {}
        
        for config_name, engine in self.search_engines.items():
            results = await engine.search(query, SearchType.HYBRID, max_results=5)
            config_results[config_name] = results
            
            print(f"--- {config_name.upper()} CONFIG ---")
            print(f"Dense weight: {engine.config.dense_weight}")
            print(f"Sparse weight: {engine.config.sparse_weight}")
            print(f"Fusion method: {engine.config.fusion_method.value}")
            print(f"Total time: {results.total_time*1000:.1f}ms")
            print(f"Top result: {results.results[0].document.metadata.get('title', 'No Title') if results.results else 'None'}")
            if results.results:
                print(f"Top score: {results.results[0].hybrid_score:.3f}")
            print()
    
    async def demonstrate_performance_analysis(self):
        """Demonstrate performance analysis across multiple queries."""
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80)
        
        engine = self.search_engines["balanced"]
        
        # Run multiple queries to collect performance data
        total_times = []
        dense_times = []
        sparse_times = []
        fusion_times = []
        
        print("Running performance test across multiple queries...")
        
        for i, query in enumerate(TEST_QUERIES[:5]):  # Test first 5 queries
            results = await engine.search(query, SearchType.HYBRID, max_results=10)
            
            total_times.append(results.total_time)
            dense_times.append(results.dense_search_time)
            sparse_times.append(results.sparse_search_time)
            fusion_times.append(results.fusion_time)
            
            print(f"Query {i+1}: {results.total_time*1000:.1f}ms total")
        
        # Calculate and display statistics
        print(f"\nPerformance Statistics (5 queries):")
        print(f"Average total time: {sum(total_times)/len(total_times)*1000:.1f}ms")
        print(f"Average dense time: {sum(dense_times)/len(dense_times)*1000:.1f}ms")
        print(f"Average sparse time: {sum(sparse_times)/len(sparse_times)*1000:.1f}ms")
        print(f"Average fusion time: {sum(fusion_times)/len(fusion_times)*1000:.1f}ms")
        
        # Get engine statistics
        stats = engine.get_performance_stats()
        print(f"\nEngine Statistics:")
        print(f"Total searches: {stats['total_searches']}")
        print(f"Average times: {stats['average_times']}")
    
    async def demonstrate_contextual_integration(self):
        """Demonstrate integration with contextual embedding system."""
        print("\n" + "="*80)
        print("CONTEXTUAL HYBRID SEARCH INTEGRATION")
        print("="*80)
        
        # Setup contextual system
        contextual_config = RetrievalConfig(
            context_strategy="combined",
            context_window_size=2,
            max_context_length=500
        )
        
        contextual_system = ContextualEmbeddingSystem(contextual_config)
        
        # Initialize contextual hybrid search
        contextual_search = ContextualHybridSearch(
            contextual_system=contextual_system,
            embeddings_model=self.embeddings,
            hybrid_config=self.configs["balanced"]
        )
        
        print("Setting up contextual search index...")
        await contextual_search.setup_search_index(self.documents)
        
        # Test contextual search
        query = "machine learning supervised algorithms"
        print(f"\nQuery: '{query}'")
        
        # Compare regular vs contextual search
        regular_results = await self.search_engines["balanced"].search(query, SearchType.HYBRID, max_results=3)
        contextual_results = await contextual_search.contextual_search(query, SearchType.HYBRID, max_results=3)
        
        print("\n--- REGULAR HYBRID SEARCH ---")
        for i, result in enumerate(regular_results.results):
            print(f"{i+1}. {result.document.metadata.get('title', 'No Title')}")
            print(f"   Score: {result.hybrid_score:.3f}")
            print(f"   Content: {result.document.page_content[:100]}...")
        
        print("\n--- CONTEXTUAL HYBRID SEARCH ---")
        for i, result in enumerate(contextual_results.results):
            print(f"{i+1}. {result.document.metadata.get('title', 'No Title')}")
            print(f"   Score: {result.hybrid_score:.3f}")
            print(f"   Contextual: {result.metadata.get('contextual_search', False)}")
            print(f"   Quality: {result.metadata.get('contextual_quality', 0.0):.3f}")
            print(f"   Content: {result.document.page_content[:100]}...")
    
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        print("HYBRID SEARCH INFRASTRUCTURE DEMONSTRATION")
        print("Task 3.2: Comprehensive Hybrid Search System")
        print("="*80)
        
        # Setup
        await self.setup_search_engines()
        
        # Run demonstrations
        await self.demonstrate_search_types()
        await self.demonstrate_fusion_methods()
        await self.demonstrate_configuration_impact()
        await self.demonstrate_performance_analysis()
        await self.demonstrate_contextual_integration()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nKey Achievements:")
        print("✅ Dense vector search with embeddings")
        print("✅ Sparse BM25 keyword search")
        print("✅ Hybrid search with multiple fusion methods")
        print("✅ Configurable search weights and parameters")
        print("✅ Performance optimization and parallel execution")
        print("✅ Integration with contextual embedding system")
        print("✅ Comprehensive performance monitoring")
        
        print(f"\nDocuments indexed: {len(self.documents)}")
        print(f"Search configurations tested: {len(self.configs)}")
        print(f"Fusion methods demonstrated: {len([FusionMethod.RECIPROCAL_RANK_FUSION, FusionMethod.WEIGHTED_SUM])}")


async def main():
    """Main demonstration function."""
    demo = HybridSearchDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Set up environment
    import os
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key-here")
    
    # Run demonstration
    asyncio.run(main()) 