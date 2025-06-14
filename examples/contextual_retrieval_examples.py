"""
Contextual Retrieval System - Practical Usage Examples

This module provides comprehensive examples demonstrating the capabilities
of the Universal RAG CMS Contextual Retrieval System.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Core imports
from src.retrieval import (
    create_contextual_retrieval_system,
    RetrievalConfig,
    RetrievalStrategy,
    ContextualConfig,
    HybridConfig,
    MultiQueryConfig,
    SelfQueryConfig,
    MMRConfig
)
from src.chains.enhanced_confidence_scoring_system import create_universal_rag_enhancement_system
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document


class ContextualRetrievalExamples:
    """Comprehensive examples for contextual retrieval system."""
    
    def __init__(self):
        """Initialize the examples with default configuration."""
        self.retrieval_system = None
        self.enhancement_system = None
        self.setup_complete = False
    
    async def setup_systems(self):
        """Initialize the retrieval and enhancement systems."""
        print("🚀 Setting up Contextual Retrieval System...")
        
        # Create configuration
        config = RetrievalConfig(
            # Basic settings
            default_strategy=RetrievalStrategy.FULL_CONTEXTUAL,
            max_results=20,
            enable_caching=True,
            
            # Contextual embedding configuration
            contextual_config=ContextualConfig(
                context_window_size=2,
                max_context_length=1000,
                include_document_title=True,
                include_section_headers=True,
                include_breadcrumbs=True
            ),
            
            # Hybrid search configuration
            hybrid_config=HybridConfig(
                dense_weight=0.7,
                sparse_weight=0.3,
                enable_score_normalization=True
            ),
            
            # Multi-query configuration
            multi_query_config=MultiQueryConfig(
                num_expansions=3,
                parallel_processing=True,
                timeout_seconds=10
            ),
            
            # Self-query configuration
            self_query_config=SelfQueryConfig(
                enable_llm_analysis=True,
                analysis_confidence_threshold=0.7,
                enable_fuzzy_matching=True
            ),
            
            # MMR configuration
            mmr_config=MMRConfig(
                lambda_param=0.7,  # 70% relevance, 30% diversity
                enable_diversity_boost=True,
                diversity_threshold=0.8
            )
        )
        
        # Initialize systems
        self.retrieval_system = create_contextual_retrieval_system(
            config=config,
            enable_task2_integration=True
        )
        
        self.enhancement_system = create_universal_rag_enhancement_system()
        
        self.setup_complete = True
        print("✅ Systems initialized successfully!")
    
    async def example_1_basic_retrieval(self):
        """Example 1: Basic contextual retrieval."""
        print("\n" + "="*60)
        print("📚 EXAMPLE 1: Basic Contextual Retrieval")
        print("="*60)
        
        if not self.setup_complete:
            await self.setup_systems()
        
        # Sample query
        query = "What are the best casino bonuses for new players?"
        
        print(f"Query: {query}")
        print("\n🔍 Performing contextual retrieval...")
        
        # Perform retrieval
        start_time = datetime.now()
        results = await self.retrieval_system.retrieve(
            query=query,
            strategy=RetrievalStrategy.FULL_CONTEXTUAL,
            max_results=5
        )
        end_time = datetime.now()
        
        # Display results
        retrieval_time = (end_time - start_time).total_seconds() * 1000
        print(f"⚡ Retrieval completed in {retrieval_time:.1f}ms")
        print(f"📄 Retrieved {len(results)} documents")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Title: {result.metadata.get('title', 'Unknown')}")
            print(f"Score: {result.metadata.get('score', 0):.3f}")
            print(f"Source: {result.metadata.get('source', 'Unknown')}")
            print(f"Content: {result.page_content[:200]}...")
        
        return results
    
    async def example_2_strategy_comparison(self):
        """Example 2: Compare different retrieval strategies."""
        print("\n" + "="*60)
        print("⚖️ EXAMPLE 2: Strategy Comparison")
        print("="*60)
        
        if not self.setup_complete:
            await self.setup_systems()
        
        query = "Compare online casino welcome bonuses"
        strategies = [
            RetrievalStrategy.DENSE_ONLY,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.CONTEXTUAL,
            RetrievalStrategy.FULL_CONTEXTUAL
        ]
        
        print(f"Query: {query}")
        print("\n🔬 Testing different retrieval strategies...")
        
        results_comparison = {}
        
        for strategy in strategies:
            print(f"\n🧪 Testing {strategy.value}...")
            
            start_time = datetime.now()
            results = await self.retrieval_system.retrieve(
                query=query,
                strategy=strategy,
                max_results=3
            )
            end_time = datetime.now()
            
            retrieval_time = (end_time - start_time).total_seconds() * 1000
            avg_score = sum(r.metadata.get('score', 0) for r in results) / len(results) if results else 0
            
            results_comparison[strategy.value] = {
                'time_ms': retrieval_time,
                'num_results': len(results),
                'avg_score': avg_score,
                'results': results
            }
            
            print(f"  ⚡ Time: {retrieval_time:.1f}ms")
            print(f"  📄 Results: {len(results)}")
            print(f"  📊 Avg Score: {avg_score:.3f}")
        
        # Summary comparison
        print("\n📊 STRATEGY COMPARISON SUMMARY")
        print("-" * 50)
        for strategy, metrics in results_comparison.items():
            print(f"{strategy:20} | {metrics['time_ms']:6.1f}ms | {metrics['avg_score']:.3f} score")
        
        return results_comparison
    
    async def example_3_task2_integration(self):
        """Example 3: Full pipeline with Task 2 integration."""
        print("\n" + "="*60)
        print("🔗 EXAMPLE 3: Task 2 Integration Pipeline")
        print("="*60)
        
        if not self.setup_complete:
            await self.setup_systems()
        
        query = "What are the most reliable online casinos with high ratings?"
        query_type = "review"
        
        print(f"Query: {query}")
        print(f"Query Type: {query_type}")
        print("\n🔄 Running integrated pipeline...")
        
        # Step 1: Contextual retrieval
        print("1️⃣ Performing contextual retrieval...")
        start_time = datetime.now()
        documents = await self.retrieval_system.retrieve(
            query=query,
            strategy=RetrievalStrategy.FULL_CONTEXTUAL,
            max_results=10
        )
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
        print(f"   ✅ Retrieved {len(documents)} documents in {retrieval_time:.1f}ms")
        
        # Step 2: Simulate response generation (in real scenario, use your RAG chain)
        print("2️⃣ Generating response...")
        response_content = self._simulate_response_generation(query, documents)
        print(f"   ✅ Generated response ({len(response_content)} characters)")
        
        # Step 3: Task 2 enhancement
        print("3️⃣ Applying Task 2 enhancements...")
        start_time = datetime.now()
        enhanced_response = await self.enhancement_system.enhance_rag_response(
            response_content=response_content,
            query=query,
            query_type=query_type,
            sources=[doc.metadata for doc in documents],
            generation_metadata={
                "retrieval_method": "contextual_hybrid",
                "num_sources": len(documents),
                "retrieval_time_ms": retrieval_time
            }
        )
        enhancement_time = (datetime.now() - start_time).total_seconds() * 1000
        print(f"   ✅ Enhanced response in {enhancement_time:.1f}ms")
        
        # Display results
        print("\n📊 ENHANCED RESPONSE ANALYSIS")
        print("-" * 40)
        print(f"Confidence Score: {enhanced_response.confidence_score:.3f}")
        print(f"Quality Level: {enhanced_response.response_quality_level.value}")
        print(f"Cached: {enhanced_response.cached}")
        print(f"Source Quality: {enhanced_response.source_quality_score:.3f}")
        print(f"Response Validation: {enhanced_response.response_validation_score:.3f}")
        
        if enhanced_response.quality_factors:
            print(f"\n🔍 Quality Factors:")
            factors = enhanced_response.quality_factors
            print(f"  Content Quality: {factors.content_quality:.3f}")
            print(f"  Source Quality: {factors.source_quality:.3f}")
            print(f"  Query Matching: {factors.query_matching:.3f}")
            print(f"  Technical Factors: {factors.technical_factors:.3f}")
        
        print(f"\n📝 Enhanced Response:")
        print(f"{enhanced_response.content[:500]}...")
        
        return enhanced_response
    
    async def example_4_advanced_filtering(self):
        """Example 4: Advanced metadata filtering with self-query."""
        print("\n" + "="*60)
        print("🎯 EXAMPLE 4: Advanced Metadata Filtering")
        print("="*60)
        
        if not self.setup_complete:
            await self.setup_systems()
        
        # Queries with implicit filters
        test_queries = [
            "Find casino reviews from 2024 with ratings above 4 stars",
            "Show me slot games with high RTP and bonus features",
            "What are the latest poker tournaments with buy-ins under $100?",
            "Find articles about blackjack strategy published recently"
        ]
        
        print("🔍 Testing self-query metadata filtering...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i} ---")
            print(f"Query: {query}")
            
            # Perform retrieval with self-query filtering
            start_time = datetime.now()
            results = await self.retrieval_system.retrieve(
                query=query,
                strategy=RetrievalStrategy.SELF_QUERY,
                max_results=5
            )
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            
            print(f"⚡ Retrieved {len(results)} filtered results in {retrieval_time:.1f}ms")
            
            # Show extracted filters (if available in metadata)
            if results and 'extracted_filters' in results[0].metadata:
                filters = results[0].metadata['extracted_filters']
                print(f"🎯 Extracted Filters: {filters}")
            
            # Show top result
            if results:
                top_result = results[0]
                print(f"📄 Top Result:")
                print(f"   Title: {top_result.metadata.get('title', 'Unknown')}")
                print(f"   Score: {top_result.metadata.get('score', 0):.3f}")
                print(f"   Content: {top_result.page_content[:150]}...")
    
    async def example_5_performance_optimization(self):
        """Example 5: Performance optimization and monitoring."""
        print("\n" + "="*60)
        print("⚡ EXAMPLE 5: Performance Optimization")
        print("="*60)
        
        if not self.setup_complete:
            await self.setup_systems()
        
        # Test queries for performance analysis
        test_queries = [
            "Best casino bonuses",
            "Slot machine strategies",
            "Poker tournament tips",
            "Blackjack card counting",
            "Roulette betting systems"
        ]
        
        print("📊 Running performance analysis...")
        
        # Collect performance metrics
        performance_data = []
        
        for query in test_queries:
            print(f"\n🧪 Testing: {query}")
            
            # Multiple runs for average
            times = []
            for run in range(3):
                start_time = datetime.now()
                results = await self.retrieval_system.retrieve(
                    query=query,
                    strategy=RetrievalStrategy.FULL_CONTEXTUAL,
                    max_results=10
                )
                end_time = datetime.now()
                
                retrieval_time = (end_time - start_time).total_seconds() * 1000
                times.append(retrieval_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            performance_data.append({
                'query': query,
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'num_results': len(results)
            })
            
            print(f"   ⚡ Avg: {avg_time:.1f}ms | Min: {min_time:.1f}ms | Max: {max_time:.1f}ms")
        
        # Performance summary
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 60)
        overall_avg = sum(p['avg_time_ms'] for p in performance_data) / len(performance_data)
        print(f"Overall Average Latency: {overall_avg:.1f}ms")
        
        # Get system performance metrics
        try:
            metrics = self.retrieval_system.get_performance_metrics()
            print(f"\n🔍 System Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠️ Could not retrieve system metrics: {e}")
        
        return performance_data
    
    async def example_6_custom_configuration(self):
        """Example 6: Custom configuration for specific use cases."""
        print("\n" + "="*60)
        print("⚙️ EXAMPLE 6: Custom Configuration")
        print("="*60)
        
        # Different configurations for different use cases
        configurations = {
            "speed_optimized": RetrievalConfig(
                default_strategy=RetrievalStrategy.HYBRID,
                max_results=5,
                hybrid_config=HybridConfig(
                    dense_weight=0.8,
                    sparse_weight=0.2
                ),
                multi_query_config=MultiQueryConfig(
                    num_expansions=2,
                    parallel_processing=True
                ),
                mmr_config=MMRConfig(
                    lambda_param=0.9  # Favor relevance over diversity
                )
            ),
            
            "quality_optimized": RetrievalConfig(
                default_strategy=RetrievalStrategy.FULL_CONTEXTUAL,
                max_results=15,
                contextual_config=ContextualConfig(
                    context_window_size=3,
                    max_context_length=1500
                ),
                multi_query_config=MultiQueryConfig(
                    num_expansions=4,
                    parallel_processing=True
                ),
                mmr_config=MMRConfig(
                    lambda_param=0.6  # Balance relevance and diversity
                )
            ),
            
            "diversity_optimized": RetrievalConfig(
                default_strategy=RetrievalStrategy.FULL_CONTEXTUAL,
                max_results=20,
                mmr_config=MMRConfig(
                    lambda_param=0.4,  # Favor diversity
                    enable_diversity_boost=True,
                    diversity_threshold=0.7
                ),
                multi_query_config=MultiQueryConfig(
                    num_expansions=5,
                    parallel_processing=True
                )
            )
        }
        
        query = "Casino game strategies and tips"
        
        print(f"Query: {query}")
        print("\n🔧 Testing different configurations...")
        
        results_by_config = {}
        
        for config_name, config in configurations.items():
            print(f"\n🧪 Testing {config_name} configuration...")
            
            # Create system with custom config
            custom_system = create_contextual_retrieval_system(
                config=config,
                enable_task2_integration=True
            )
            
            start_time = datetime.now()
            results = await custom_system.retrieve(
                query=query,
                max_results=config.max_results
            )
            end_time = datetime.now()
            
            retrieval_time = (end_time - start_time).total_seconds() * 1000
            avg_score = sum(r.metadata.get('score', 0) for r in results) / len(results) if results else 0
            
            # Calculate diversity score (simplified)
            diversity_score = self._calculate_diversity_score(results)
            
            results_by_config[config_name] = {
                'time_ms': retrieval_time,
                'num_results': len(results),
                'avg_score': avg_score,
                'diversity_score': diversity_score,
                'results': results
            }
            
            print(f"   ⚡ Time: {retrieval_time:.1f}ms")
            print(f"   📄 Results: {len(results)}")
            print(f"   📊 Avg Score: {avg_score:.3f}")
            print(f"   🎯 Diversity: {diversity_score:.3f}")
        
        # Configuration comparison
        print("\n📊 CONFIGURATION COMPARISON")
        print("-" * 70)
        print(f"{'Configuration':<20} | {'Time (ms)':<10} | {'Avg Score':<10} | {'Diversity':<10}")
        print("-" * 70)
        for config_name, metrics in results_by_config.items():
            print(f"{config_name:<20} | {metrics['time_ms']:<10.1f} | {metrics['avg_score']:<10.3f} | {metrics['diversity_score']:<10.3f}")
        
        return results_by_config
    
    def _simulate_response_generation(self, query: str, documents: List[Document]) -> str:
        """Simulate response generation for demonstration purposes."""
        # In a real scenario, this would be your RAG chain
        response_parts = [
            f"Based on the retrieved information about '{query}', here are the key findings:",
            "",
            "Key Points:"
        ]
        
        for i, doc in enumerate(documents[:3], 1):
            title = doc.metadata.get('title', f'Source {i}')
            content_snippet = doc.page_content[:100].replace('\n', ' ')
            response_parts.append(f"{i}. {title}: {content_snippet}...")
        
        response_parts.extend([
            "",
            "Summary:",
            "The retrieved documents provide comprehensive information addressing your query. "
            "The contextual retrieval system has identified the most relevant sources based on "
            "semantic similarity, keyword matching, and contextual understanding."
        ])
        
        return "\n".join(response_parts)
    
    def _calculate_diversity_score(self, results: List[Document]) -> float:
        """Calculate a simple diversity score based on content similarity."""
        if len(results) < 2:
            return 1.0
        
        # Simplified diversity calculation
        # In practice, you'd use more sophisticated similarity measures
        unique_sources = set()
        unique_titles = set()
        
        for result in results:
            source = result.metadata.get('source', '')
            title = result.metadata.get('title', '')
            
            if source:
                unique_sources.add(source)
            if title:
                unique_titles.add(title)
        
        # Diversity based on source and title uniqueness
        source_diversity = len(unique_sources) / len(results) if results else 0
        title_diversity = len(unique_titles) / len(results) if results else 0
        
        return (source_diversity + title_diversity) / 2


async def run_all_examples():
    """Run all contextual retrieval examples."""
    print("🚀 CONTEXTUAL RETRIEVAL SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    
    examples = ContextualRetrievalExamples()
    
    try:
        # Run all examples
        await examples.example_1_basic_retrieval()
        await examples.example_2_strategy_comparison()
        await examples.example_3_task2_integration()
        await examples.example_4_advanced_filtering()
        await examples.example_5_performance_optimization()
        await examples.example_6_custom_configuration()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


async def run_specific_example(example_number: int):
    """Run a specific example by number."""
    examples = ContextualRetrievalExamples()
    
    example_methods = {
        1: examples.example_1_basic_retrieval,
        2: examples.example_2_strategy_comparison,
        3: examples.example_3_task2_integration,
        4: examples.example_4_advanced_filtering,
        5: examples.example_5_performance_optimization,
        6: examples.example_6_custom_configuration
    }
    
    if example_number in example_methods:
        await example_methods[example_number]()
    else:
        print(f"❌ Example {example_number} not found. Available examples: 1-6")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            asyncio.run(run_specific_example(example_num))
        except ValueError:
            print("❌ Please provide a valid example number (1-6)")
    else:
        asyncio.run(run_all_examples()) 