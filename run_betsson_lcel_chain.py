#!/usr/bin/env python3
"""
Production LCEL Chain Test for Betsson Casino Review
Using the complete Universal RAG Chain with all advanced features enabled.

Features Enabled:
✅ Native LangChain Redis Semantic Caching 
✅ Enhanced Confidence Scoring (4-factor assessment)
✅ Template System v2.0 (Local Hub pattern)
✅ Comprehensive Web Research (95-field casino analysis)
✅ Screenshot Evidence Capture
✅ DataForSEO Image Integration
✅ WordPress Auto-Publishing
✅ FTI Content Processing
✅ Security & Monitoring
✅ All Native LCEL Patterns
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our production Universal RAG Chain
from src.chains.universal_rag_lcel import create_universal_rag_chain

async def run_betsson_production_chain():
    """Run the complete production LCEL chain for Betsson casino review."""
    
    logger.info("🚀 Starting Betsson Production LCEL Chain Test")
    logger.info("=" * 60)
    
    # Create the production chain with ALL features enabled
    logger.info("📊 Initializing Universal RAG Chain with ALL features...")
    
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",  # Use cost-effective model with good context handling
        temperature=0.1,
        enable_caching=True,                    # ✅ Redis Semantic Cache
        enable_contextual_retrieval=True,      # ✅ Advanced retrieval
        enable_prompt_optimization=True,       # ✅ Query analysis & optimization
        enable_enhanced_confidence=True,       # ✅ 4-factor confidence scoring
        enable_template_system_v2=True,        # ✅ Local Hub templates
        enable_dataforseo_images=True,         # ✅ Professional image search
        enable_wordpress_publishing=True,      # ✅ Auto-publishing
        enable_fti_processing=True,            # ✅ Content type detection
        enable_security=True,                  # ✅ Security validation
        enable_profiling=True,                 # ✅ Performance monitoring
        enable_web_search=True,               # ✅ Tavily web search
        enable_comprehensive_web_research=True, # ✅ 95-field casino analysis
        enable_screenshot_evidence=True,       # ✅ Evidence capture
        enable_hyperlink_generation=True,      # ✅ Authoritative links
        enable_response_storage=True          # ✅ Response vectorization
    )
    
    logger.info("✅ Chain initialized successfully!")
    logger.info(f"📈 Active features: {chain._count_active_features()}")
    
    # Test query for Betsson casino
    test_query = "Betsson casino review 2024 - bonuses, games, safety, and player experience"
    
    logger.info(f"🎯 Test Query: {test_query}")
    logger.info("🔄 Processing with LCEL chain...")
    
    start_time = time.time()
    
    try:
        # Run the production chain with WordPress publishing enabled
        result = await chain.ainvoke(
            {"query": test_query},
            publish_to_wordpress=True  # Enable auto-publishing for this test
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("✅ Chain execution completed successfully!")
        logger.info(f"⏱️ Total execution time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        
        # Display comprehensive results
        print("\n" + "=" * 80)
        print("🎰 BETSSON CASINO REVIEW - PRODUCTION LCEL RESULTS")
        print("=" * 80)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   • Response Time: {total_time:.2f}s")
        print(f"   • Confidence Score: {result.confidence_score:.1%}")
        print(f"   • Sources Found: {len(result.sources)}")
        print(f"   • Cached Response: {result.cached}")
        print(f"   • Token Usage: {result.token_usage}")
        
        if hasattr(result, 'query_analysis') and result.query_analysis:
            qa = result.query_analysis
            print(f"\n🔍 QUERY ANALYSIS:")
            print(f"   • Query Type: {qa.get('query_type', 'N/A')}")
            print(f"   • Expertise Level: {qa.get('expertise_level', 'N/A')}")
            print(f"   • Casino Detected: {qa.get('casino_name', 'N/A')}")
        
        print(f"\n📝 CONTENT PREVIEW (First 500 chars):")
        content_preview = result.answer[:500].replace('\n', ' ')
        print(f"   {content_preview}...")
        
        print(f"\n🔗 TOP SOURCES:")
        for i, source in enumerate(result.sources[:5], 1):
            title = source.get('title', 'No title')[:60]
            url = source.get('url', 'No URL')[:80]
            quality = source.get('quality_score', 0)
            print(f"   {i}. {title} (Quality: {quality:.1%})")
            print(f"      {url}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"betsson_production_lcel_{timestamp}.json"
        
        result_data = {
            "timestamp": timestamp,
            "query": test_query,
            "execution_time": total_time,
            "confidence_score": result.confidence_score,
            "cached": result.cached,
            "token_usage": result.token_usage,
            "query_analysis": result.query_analysis if hasattr(result, 'query_analysis') else None,
            "content": result.answer,
            "sources": result.sources,
            "metadata": result.metadata,
            "chain_features": {
                "total_active_features": chain._count_active_features(),
                "caching_enabled": True,
                "web_research_enabled": True,
                "wordpress_publishing": True,
                "template_system_v2": True,
                "confidence_scoring": True
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Check if WordPress publishing was successful
        if result.metadata.get('wordpress_published'):
            wp_data = result.metadata.get('wordpress_data', {})
            print(f"\n📰 WORDPRESS PUBLISHING:")
            print(f"   • Published: ✅ Yes")
            print(f"   • Post ID: {wp_data.get('id', 'N/A')}")
            print(f"   • URL: {wp_data.get('link', 'N/A')}")
            print(f"   • Status: {wp_data.get('status', 'N/A')}")
        else:
            print(f"\n📰 WORDPRESS PUBLISHING: ❌ Not published")
            if 'wordpress_error' in result.metadata:
                print(f"   • Error: {result.metadata['wordpress_error']}")
        
        # Display cache statistics
        cache_stats = chain.get_cache_stats()
        if cache_stats:
            print(f"\n🗄️ CACHE STATISTICS:")
            print(f"   • Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"   • Total Queries: {cache_stats.get('total_queries', 0)}")
            print(f"   • Cache Hits: {cache_stats.get('hits', 0)}")
        
        print("\n" + "=" * 80)
        print("🎉 BETSSON PRODUCTION LCEL CHAIN TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Chain execution failed: {str(e)}")
        logger.error(f"📍 Error type: {type(e).__name__}")
        import traceback
        logger.error(f"📋 Traceback:\n{traceback.format_exc()}")
        
        print(f"\n❌ ERROR: {str(e)}")
        print(f"See logs for detailed traceback.")
        
        return None

async def main():
    """Main execution function."""
    try:
        result = await run_betsson_production_chain()
        if result:
            print(f"\n✅ Test completed successfully!")
            print(f"📊 Final confidence score: {result.confidence_score:.1%}")
        else:
            print(f"\n❌ Test failed - check logs for details")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        logger.error(f"Main execution error: {str(e)}")

if __name__ == "__main__":
    print("🎰 Betsson Production LCEL Chain Test")
    print("Using Universal RAG Chain with ALL advanced features")
    print("-" * 60)
    
    asyncio.run(main()) 