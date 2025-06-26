#!/usr/bin/env python3
"""
Test the default Universal RAG Chain with fixed WordPress configuration
This should work without any special setup since fixes are now in the default chain
"""

import os
import asyncio
import sys
import time
from datetime import datetime

# ✅ CRITICAL: Set WordPress environment variables BEFORE importing the chain
# Using the WORKING configuration that published Post IDs 51371 and 51406
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "your-wordpress-password-here"

# Set other required environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
os.environ["SUPABASE_URL"] = os.getenv("SUPABASE_URL", "")
os.environ["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY", "")
os.environ["DATAFORSEO_LOGIN"] = os.getenv("DATAFORSEO_LOGIN", "")
os.environ["DATAFORSEO_PASSWORD"] = os.getenv("DATAFORSEO_PASSWORD", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# Now import the chain
sys.path.append('src')
from chains.universal_rag_lcel import create_universal_rag_chain

async def test_default_chain():
    """Test the default chain with Betway casino review"""
    
    print("🧪 Testing Default Universal RAG Chain with Fixed WordPress Configuration")
    print("=" * 80)
    
    # Create the default chain (should have all fixes applied)
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_wordpress_publishing=True,
        enable_dataforseo_images=True,
        enable_web_search=True,
        enable_comprehensive_web_research=True
    )
    
    print(f"✅ Chain created successfully")
    print(f"📝 WordPress enabled: {chain.enable_wordpress_publishing}")
    print(f"🖼️ Images enabled: {chain.enable_dataforseo_images}")
    print(f"🌐 Web search enabled: {chain.enable_web_search}")
    
    # Test query for Betway casino review
    betway_query = """Create a comprehensive professional Betway Casino review for MT Casino custom post type. 
    
    Include detailed analysis of:
    - Licensing and regulation
    - Game selection and providers  
    - Welcome bonuses and promotions
    - Payment methods and processing times
    - Mobile compatibility
    - Customer support
    - Security measures
    - User experience
    - Pros and cons
    - Overall rating and verdict
    
    Ensure high-quality content suitable for WordPress publishing with proper SEO optimization."""
    
    print(f"\n🎰 Testing query: Betway Casino Review")
    print(f"📝 Query length: {len(betway_query)} characters")
    
    start_time = time.time()
    
    try:
        # Test the chain with WordPress publishing enabled
        query_input = {
            "question": betway_query,
            "publish_to_wordpress": True  # Enable WordPress publishing
        }
        
        print(f"\n🚀 Starting chain execution...")
        result = await chain.ainvoke(query_input, publish_to_wordpress=True)
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Chain execution completed in {processing_time:.2f} seconds")
        print(f"📊 Response length: {len(result.answer)} characters")
        print(f"📈 Confidence score: {result.confidence_score:.3f}")
        print(f"🔍 Sources: {len(result.sources)} sources found")
        
        # Check for WordPress publishing results
        if hasattr(result, 'metadata'):
            wp_published = result.metadata.get('wordpress_published', False)
            wp_post_id = result.metadata.get('wordpress_post_id')
            wp_url = result.metadata.get('wordpress_url')
            
            print(f"\n📝 WordPress Publishing Results:")
            print(f"   Published: {wp_published}")
            if wp_post_id:
                print(f"   Post ID: {wp_post_id}")
            if wp_url:
                print(f"   URL: {wp_url}")
                
        # Save result to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_default_chain_{timestamp}.json"
        
        result_data = {
            "query": betway_query,
            "response": result.answer,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "metadata": result.metadata if hasattr(result, 'metadata') else {},
            "processing_time": processing_time,
            "timestamp": timestamp
        }
        
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Print content preview
        print(f"\n📄 Content Preview (first 500 chars):")
        print("=" * 50)
        print(result.answer[:500] + "..." if len(result.answer) > 500 else result.answer)
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_default_chain()) 