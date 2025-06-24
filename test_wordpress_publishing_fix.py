#!/usr/bin/env python3
"""
Test WordPress Publishing Fix - Verify async context manager fix works
"""

import os
import asyncio
import sys
import logging
from pathlib import Path

# Configure detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set WordPress environment variables (using working credentials)
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "q8ZU 4UHD 90vI Ej55 U0Jh yh8c"

async def test_wordpress_publishing_fix():
    """Test the fixed WordPress publishing with Universal RAG Chain"""
    
    print("🔧 TESTING WORDPRESS PUBLISHING FIX")
    print("=" * 60)
    
    try:
        # Import Universal RAG Chain
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        print("✅ Successfully imported Universal RAG Chain")
        
        # Create chain with WordPress publishing enabled
        chain = create_universal_rag_chain(
            model_name="gpt-4o-mini",
            temperature=0.1,
            enable_wordpress_publishing=True,
            enable_dataforseo_images=False,  # Disable to focus on WordPress
            enable_web_search=False,
            enable_comprehensive_web_research=False
        )
        
        print("✅ Universal RAG Chain created")
        print(f"   WordPress service: {chain.wordpress_service}")
        print(f"   WordPress config: {chain.wordpress_service.config if chain.wordpress_service else 'None'}")
        
        # Test query with WordPress publishing
        test_query = "Create a comprehensive review of BC.Game casino focusing on crash games and crypto features."
        
        inputs = {
            "question": test_query,
            "publish_to_wordpress": True  # This is the key flag that triggers publishing
        }
        
        print(f"📝 Test query: {test_query}")
        print("🚀 Running Universal RAG Chain with WordPress publishing...")
        
        # Run the chain
        response = await chain.ainvoke(inputs)
        
        print("🎉 ✅ UNIVERSAL RAG CHAIN COMPLETED!")
        print(f"📊 Response confidence: {response.confidence_score:.2f}")
        print(f"📝 Content length: {len(response.answer)} characters")
        
        # Check WordPress publishing result
        if response.metadata.get("wordpress_published"):
            wordpress_url = response.metadata.get("wordpress_url", "")
            post_id = response.metadata.get("wordpress_post_id", "")
            print(f"🌟 ✅ WORDPRESS PUBLISHING SUCCESS!")
            print(f"📝 Post ID: {post_id}")
            print(f"🔗 Post URL: {wordpress_url}")
            print(f"📊 Categories: {response.metadata.get('wordpress_category', 'N/A')}")
            print(f"🏷️ Custom Fields: {response.metadata.get('wordpress_custom_fields_count', 0)}")
            print(f"🔖 Tags: {response.metadata.get('wordpress_tags_count', 0)}")
        else:
            print(f"❌ WordPress publishing failed")
            if response.metadata.get("wordpress_error"):
                print(f"💡 Error: {response.metadata.get('wordpress_error')}")
        
        return response
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_wordpress_publishing_fix())
    
    if result and result.metadata.get("wordpress_published"):
        print("\n🎉 SUCCESS! WordPress publishing fix works correctly!")
    else:
        print("\n❌ FAILED! WordPress publishing still has issues.") 