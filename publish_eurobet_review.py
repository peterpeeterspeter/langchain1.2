#!/usr/bin/env python3
"""
🎰 EUROBET CASINO REVIEW - LIVE WORDPRESS PUBLISHING
Universal RAG CMS with All Enterprise Features ENABLED
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def create_eurobet_review():
    """Create and publish comprehensive Eurobet Casino review"""
    print("🎰 Creating Eurobet Casino Review with Full Universal RAG Chain...")
    
    try:
        # Import the Universal RAG Chain
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        print("✅ Imported Universal RAG Chain successfully")
        
        # Initialize the chain with ALL enterprise features enabled
        chain = UniversalRAGChain(
            model_name='gpt-4.1-mini',
            temperature=0.1,
            enable_caching=True,
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=True,
            enable_dataforseo_images=True,
            enable_wordpress_publishing=True,  # ✅ ENABLED for publishing
            enable_fti_processing=True,
            enable_security=True,
            enable_profiling=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_screenshot_evidence=True,
            enable_hyperlink_generation=True
        )
        
        print("✅ Universal RAG Chain initialized with ALL enterprise features")
        
        # Eurobet review query
        eurobet_query = """Create a comprehensive professional review of Eurobet Casino for Italian players. 
        
        Focus on:
        - AAMS/ADM licensing and regulatory compliance
        - Casino games portfolio (slots, live dealer, table games)
        - Sports betting integration and live betting features
        - Mobile app experience for iOS and Android
        - Welcome bonus and promotional offers
        - Payment methods for Italian players
        - Customer support in Italian language
        - Security measures and responsible gaming tools
        - Overall user experience and site navigation
        
        Provide detailed analysis with pros/cons, ratings, and final recommendation.
        This review should be ready for publication on crashcasino.io as mt_listing custom post type."""
        
        print(f"🔍 Running query: {eurobet_query[:100]}...")
        
        # Execute the chain with full research and publishing
        result = await chain.ainvoke({"query": eurobet_query})
        
        print("✅ Review generation completed!")
        print(f"📊 Response confidence: {result.confidence_score:.2f}")
        print(f"⏱️ Response time: {result.response_time:.2f}s")
        print(f"📝 Content length: {len(result.answer)} characters")
        
        # Check WordPress publishing results
        if hasattr(result, 'metadata') and result.metadata:
            wordpress_published = result.metadata.get('wordpress_published', False)
            wordpress_post_id = result.metadata.get('wordpress_post_id')
            wordpress_url = result.metadata.get('wordpress_url')
            wordpress_error = result.metadata.get('wordpress_error')
            
            if wordpress_published and wordpress_post_id:
                print("🎉 ✅ WORDPRESS PUBLISHING SUCCESSFUL!")
                print(f"📝 WordPress Post ID: {wordpress_post_id}")
                print(f"🌐 Published URL: {wordpress_url}")
                print(f"✏️ Edit URL: {result.metadata.get('wordpress_edit_url', 'N/A')}")
                print(f"🏷️ Category: {result.metadata.get('wordpress_category', 'N/A')}")
                print(f"📊 Custom Fields: {result.metadata.get('wordpress_custom_fields_count', 0)}")
                print(f"🏷️ Tags: {result.metadata.get('wordpress_tags_count', 0)}")
            elif wordpress_error:
                print(f"❌ WordPress Publishing Failed: {wordpress_error}")
            else:
                print("⚠️ WordPress publishing status unclear")
        
        # Show sources
        print(f"📚 Sources used: {len(result.sources)}")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"   {i}. {source.get('url', 'Unknown source')}")
        
        # Display the review content (first 500 chars)
        print("\n📄 Generated Review Preview:")
        print("=" * 50)
        print(result.answer[:500] + "..." if len(result.answer) > 500 else result.answer)
        print("=" * 50)
        
        return result
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you're in the correct directory and all dependencies are installed")
        return None
        
    except Exception as e:
        print(f"❌ Error creating Eurobet review: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting Eurobet Casino Review Publication...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}")
    
    # Run the async function
    result = asyncio.run(create_eurobet_review())
    
    if result:
        print("🎉 Eurobet Casino Review completed successfully!")
    else:
        print("❌ Eurobet Casino Review failed")
        sys.exit(1) 