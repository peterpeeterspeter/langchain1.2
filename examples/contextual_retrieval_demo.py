#!/usr/bin/env python3
"""
Task 3.5: Contextual Retrieval System Demo
Complete integration of all Task 3 components with Task 2's enhanced confidence scoring system.

This demonstration shows the full contextual retrieval system in action:
- Task 3.1: Contextual Embedding System  
- Task 3.2: Hybrid Search Infrastructure
- Task 3.3: Multi-Query Retrieval
- Task 3.4: Self-Query Metadata Filtering
- Task 3.5: MMR & Task 2 Integration

Requirements:
- Supabase account and credentials
- OpenAI or Anthropic API key
- Required packages: supabase, langchain, numpy
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import required LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document

# Import Supabase
from supabase import create_client, Client

# Import our contextual retrieval system
from retrieval.contextual_retrieval import (
    ContextualRetrievalSystem,
    RetrievalConfig,
    MaximalMarginalRelevance,
    RetrievalOptimizer,
    create_contextual_retrieval_system
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ContextualRetrievalDemo:
    """
    Comprehensive demonstration of the contextual retrieval system.
    """
    
    def __init__(self):
        self.supabase_client = None
        self.embeddings = None
        self.llm = None
        self.retrieval_system = None
        
        # Demo queries for testing
        self.demo_queries = [
            "What are the best casino games for beginners?",
            "Find me recent promotions with high bonuses",
            "Compare poker strategies for online vs live games",
            "What are the most trusted casino sites in 2024?",
            "Explain the difference between slots and table games",
            "Show me user reviews for European casinos",
            "What are the latest gambling regulations in the US?",
            "Find VIP programs with the best loyalty rewards"
        ]
        
        # Configuration for different test scenarios
        self.test_scenarios = {
            "basic": RetrievalConfig(
                enable_caching=False,
                enable_source_quality_analysis=False,
                enable_metadata_filtering=False,
                max_query_variations=0
            ),
            "hybrid_only": RetrievalConfig(
                enable_caching=True,
                enable_source_quality_analysis=False,
                enable_metadata_filtering=False,
                max_query_variations=0,
                dense_weight=0.7,
                sparse_weight=0.3
            ),
            "multi_query": RetrievalConfig(
                enable_caching=True,
                enable_source_quality_analysis=False,
                enable_metadata_filtering=False,
                max_query_variations=3,
                dense_weight=0.7,
                sparse_weight=0.3
            ),
            "full_contextual": RetrievalConfig(
                enable_caching=True,
                enable_source_quality_analysis=True,
                enable_metadata_filtering=True,
                max_query_variations=3,
                dense_weight=0.7,
                sparse_weight=0.3,
                mmr_lambda=0.7,
                quality_threshold=0.6
            )
        }
    
    def setup_credentials(self):
        """Set up API credentials and clients."""
        
        # Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL", "https://ambjsovdhizjxwhhnbtd.supabase.co")
        supabase_key = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc2Mzc2NDYsImV4cCI6MjA2MzIxMzY0Nn0.3H8N2Fk22RAV1gHzDB5pCi9GokGwroG34v15I5Cq8_g")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
        
        self.supabase_client = create_client(supabase_url, supabase_key)
        logger.info("✅ Supabase client initialized")
        
        # Set up embeddings (OpenAI by default)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_api_key
            )
            logger.info("✅ OpenAI embeddings initialized")
        else:
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        # Set up LLM (try Anthropic first, fallback to OpenAI)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                api_key=anthropic_api_key,
                temperature=0.1
            )
            logger.info("✅ Anthropic LLM initialized")
        elif openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=openai_api_key,
                temperature=0.1
            )
            logger.info("✅ OpenAI LLM initialized")
        else:
            raise ValueError("Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    async def demo_mmr_algorithm(self):
        """Demonstrate the MMR algorithm in isolation."""
        
        logger.info("\n🎯 === MMR Algorithm Demonstration ===")
        
        # Create some sample documents with similar content
        sample_docs = [
            Document(
                page_content="Blackjack is a popular card game in casinos with strategic elements.",
                metadata={"game_type": "card", "complexity": "medium", "id": "doc1"}
            ),
            Document(
                page_content="Blackjack strategy involves basic card counting and betting patterns.",
                metadata={"game_type": "card", "complexity": "medium", "id": "doc2"}
            ),
            Document(
                page_content="Slot machines are the most popular casino games with simple gameplay.",
                metadata={"game_type": "slot", "complexity": "low", "id": "doc3"}
            ),
            Document(
                page_content="Poker requires skill and strategy, unlike pure chance games.",
                metadata={"game_type": "card", "complexity": "high", "id": "doc4"}
            ),
            Document(
                page_content="Roulette is a wheel-based game of chance with various betting options.",
                metadata={"game_type": "wheel", "complexity": "low", "id": "doc5"}
            )
        ]
        
        # Generate embeddings for documents
        doc_embeddings = []
        for doc in sample_docs:
            if hasattr(self.embeddings, 'aembed_query'):
                embedding = await self.embeddings.aembed_query(doc.page_content)
            else:
                embedding = self.embeddings.embed_query(doc.page_content)
            doc_embeddings.append((doc, embedding))
        
        # Generate query embedding
        query = "What casino games should beginners try?"
        if hasattr(self.embeddings, 'aembed_query'):
            query_embedding = await self.embeddings.aembed_query(query)
        else:
            query_embedding = self.embeddings.embed_query(query)
        
        # Test different MMR lambda values
        lambda_values = [0.0, 0.3, 0.7, 1.0]
        
        for lambda_val in lambda_values:
            config = RetrievalConfig(mmr_lambda=lambda_val)
            mmr = MaximalMarginalRelevance(config)
            
            selected_docs = await mmr.apply_mmr(
                query_embedding=query_embedding,
                documents_with_embeddings=doc_embeddings,
                k=3
            )
            
            logger.info(f"\n📊 MMR Results (λ={lambda_val}):")
            for i, doc in enumerate(selected_docs):
                logger.info(
                    f"  {i+1}. {doc.metadata.get('id', 'unknown')} - "
                    f"Relevance: {doc.metadata.get('relevance_score', 0):.3f}, "
                    f"Diversity: {doc.metadata.get('diversity_score', 0):.3f}, "
                    f"Game: {doc.metadata.get('game_type', 'unknown')}"
                )
    
    async def demo_retrieval_scenarios(self):
        """Demonstrate different retrieval scenarios."""
        
        logger.info("\n🔄 === Retrieval Scenarios Comparison ===")
        
        test_query = "What are the most trusted online casino sites?"
        
        for scenario_name, config in self.test_scenarios.items():
            logger.info(f"\n📋 Testing scenario: {scenario_name.upper()}")
            
            # Create retrieval system with specific config
            retrieval_system = await create_contextual_retrieval_system(
                supabase_client=self.supabase_client,
                embeddings=self.embeddings,
                llm=self.llm,
                config=config,
                enable_task2_integration=config.enable_source_quality_analysis
            )
            
            try:
                # Perform retrieval
                start_time = asyncio.get_event_loop().time()
                results = await retrieval_system._aget_relevant_documents(test_query)
                end_time = asyncio.get_event_loop().time()
                
                # Log results
                logger.info(f"✅ Retrieved {len(results)} documents in {(end_time - start_time)*1000:.1f}ms")
                
                if results:
                    # Show top 3 results
                    for i, doc in enumerate(results[:3]):
                        metadata = doc.metadata
                        logger.info(
                            f"  {i+1}. Score: {metadata.get('retrieval_score', 'N/A'):.3f} | "
                            f"Quality: {metadata.get('source_quality', 'N/A')} | "
                            f"MMR: {metadata.get('mmr_score', 'N/A')} | "
                            f"Method: {metadata.get('retrieval_method', 'unknown')}"
                        )
                
                # Get performance metrics
                metrics = retrieval_system.get_performance_metrics()
                logger.info(f"📊 Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"❌ Scenario {scenario_name} failed: {e}")
    
    async def demo_full_integration(self):
        """Demonstrate the full contextual retrieval system."""
        
        logger.info("\n🚀 === Full Contextual Retrieval System Demo ===")
        
        # Create the full system
        self.retrieval_system = await create_contextual_retrieval_system(
            supabase_client=self.supabase_client,
            embeddings=self.embeddings,
            llm=self.llm,
            config=self.test_scenarios["full_contextual"],
            enable_task2_integration=True
        )
        
        logger.info("🎯 System Components Initialized:")
        logger.info("  ✅ Contextual Embedding System")
        logger.info("  ✅ Hybrid Search Infrastructure") 
        logger.info("  ✅ Multi-Query Retrieval")
        logger.info("  ✅ Self-Query Metadata Filtering")
        logger.info("  ✅ MMR Diversity Selection")
        logger.info("  ✅ Task 2 Integration (Source Quality + Caching)")
        
        # Test all demo queries
        total_time = 0.0
        successful_queries = 0
        
        for i, query in enumerate(self.demo_queries):
            logger.info(f"\n🔍 Query {i+1}/{len(self.demo_queries)}: {query}")
            
            try:
                start_time = asyncio.get_event_loop().time()
                results = await self.retrieval_system._aget_relevant_documents(query)
                end_time = asyncio.get_event_loop().time()
                
                query_time = end_time - start_time
                total_time += query_time
                successful_queries += 1
                
                logger.info(f"✅ Retrieved {len(results)} documents in {query_time*1000:.1f}ms")
                
                if results:
                    # Show top result with full metadata
                    top_result = results[0]
                    metadata = top_result.metadata
                    
                    logger.info("🏆 Top Result:")
                    logger.info(f"  Content: {top_result.page_content[:100]}...")
                    logger.info(f"  Retrieval Score: {metadata.get('retrieval_score', 'N/A')}")
                    logger.info(f"  Source Quality: {metadata.get('source_quality', 'N/A')}")
                    logger.info(f"  MMR Position: {metadata.get('mmr_position', 'N/A')}")
                    logger.info(f"  Diversity Score: {metadata.get('diversity_score', 'N/A')}")
                    logger.info(f"  Enhanced by Task 2: {metadata.get('enhanced_by_task2', False)}")
                    logger.info(f"  Cached: {metadata.get('cached', False)}")
                
            except Exception as e:
                logger.error(f"❌ Query failed: {e}")
        
        # Final performance summary
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            logger.info(f"\n📊 Performance Summary:")
            logger.info(f"  Successful queries: {successful_queries}/{len(self.demo_queries)}")
            logger.info(f"  Average query time: {avg_time*1000:.1f}ms")
            
            # Get system metrics
            metrics = self.retrieval_system.get_performance_metrics()
            logger.info(f"  Cache hit rate: {metrics.get('cache_hit_rate', 0)*100:.1f}%")
            logger.info(f"  MMR selections: {metrics.get('mmr_selections', 0)}")
            logger.info(f"  Quality enhancements: {metrics.get('quality_enhancements', 0)}")
            logger.info(f"  Task 2 integration: {metrics.get('task2_integration', False)}")
    
    async def demo_performance_optimization(self):
        """Demonstrate performance optimization capabilities."""
        
        logger.info("\n⚡ === Performance Optimization Demo ===")
        
        if not self.retrieval_system:
            logger.warning("Retrieval system not initialized. Skipping optimization demo.")
            return
        
        # Create validation queries with expected results
        validation_queries = [
            ("blackjack strategy", ["blackjack_guide_1", "card_counting_2", "basic_strategy_3"]),
            ("slot machine tips", ["slots_guide_1", "rtp_explanation_2", "bonus_features_3"]),
            ("casino bonuses", ["bonus_guide_1", "wagering_requirements_2", "free_spins_3"])
        ]
        
        logger.info(f"🎯 Optimizing with {len(validation_queries)} validation queries...")
        
        try:
            # Run optimization
            optimization_results = await self.retrieval_system.optimize_performance(validation_queries)
            
            logger.info("✅ Optimization completed!")
            logger.info(f"📊 Results:")
            logger.info(f"  Best parameters: {optimization_results.get('best_parameters', {})}")
            logger.info(f"  Optimization score: {optimization_results.get('optimization_score', 0):.3f}")
            logger.info(f"  Validation queries: {optimization_results.get('validation_queries_count', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
    
    async def run_all_demos(self):
        """Run all demonstrations."""
        
        logger.info("🎬 Starting Contextual Retrieval System Demonstrations")
        logger.info("=" * 60)
        
        try:
            # Setup
            self.setup_credentials()
            
            # Run individual demos
            await self.demo_mmr_algorithm()
            await self.demo_retrieval_scenarios()
            await self.demo_full_integration()
            await self.demo_performance_optimization()
            
            logger.info("\n🎉 All demonstrations completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise


def main():
    """Main function to run the demonstrations."""
    
    print("🌟 Task 3.5: Contextual Retrieval System Demo")
    print("Complete integration of all Task 3 components with Task 2")
    print("-" * 60)
    
    # Check for required environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these variables and try again.")
        return
    
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Missing LLM API key. Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    # Run the demo
    demo = ContextualRetrievalDemo()
    
    try:
        asyncio.run(demo.run_all_demos())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 