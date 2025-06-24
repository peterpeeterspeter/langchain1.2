#!/usr/bin/env python3
"""
🔧 DIRECT WORDPRESS PUBLISHING FIX TEST
Test WordPress publishing by bypassing the integration layer and using WordPressRESTPublisher directly
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# Set WordPress environment variables
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "q8ZU 4UHD 90vI Ej55 U0Jh yh8c"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain
from integrations.wordpress_publisher import WordPressRESTPublisher, WordPressConfig

async def test_direct_wordpress_fix():
    """Test bypassing the WordPress integration and using WordPressRESTPublisher directly"""
    
    print("🔧 DIRECT WORDPRESS PUBLISHING FIX TEST")
    print("=" * 60)
    print("🎯 Goal: Bypass integration layer, use WordPressRESTPublisher directly")
    print("📝 WordPress Site: https://www.crashcasino.io")
    print()
    
    # First, generate content with the RAG chain (but skip WordPress publishing)
    print("🚀 Step 1: Generate content with RAG chain...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_comprehensive_web_research=True,
        enable_wordpress_publishing=False,  # Disable automatic WordPress publishing
        enable_dataforseo_images=True,
        enable_web_search=True,
        enable_cache_bypass=False
    )
    
    # Simple TrustDice query
    trustdice_query = "Write a short TrustDice Casino review covering licensing, games, and bonuses"
    
    start_time = time.time()
    
    # Generate content without WordPress publishing
    query_input = {
        "question": trustdice_query,
        "publish_to_wordpress": False  # Don't auto-publish
    }
    
    print("⚡ Generating content...")
    response = await rag_chain.ainvoke(query_input)
    
    processing_time = time.time() - start_time
    
    print(f"✅ Content generated in {processing_time:.2f} seconds")
    print(f"📊 Content Length: {len(response.answer)} characters")
    print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
    
    # Now, manually publish to WordPress using WordPressRESTPublisher directly
    print("\n🚀 Step 2: Direct WordPress publishing...")
    
    try:
        # Create WordPress config directly
        wp_config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="nmlwh",
            application_password="q8ZU 4UHD 90vI Ej55 U0Jh yh8c"
        )
        
        # Create post data
        post_data = {
            "title": "TrustDice Casino Review - Direct Publishing Test",
            "content": response.answer,
            "status": "publish",  # Publish immediately
            "categories": [1],  # General category
            "tags": ["trustdice", "casino", "review", "direct-test"],
            "custom_fields": {
                "review_type": "direct_test",
                "confidence_score": response.confidence_score,
                "processing_time": processing_time,
                "test_timestamp": datetime.now().isoformat()
            }
        }
        
        print(f"📝 Post Title: {post_data['title']}")
        print(f"📊 Content Length: {len(post_data['content'])} characters")
        print(f"🏷️ Categories: {post_data['categories']}")
        print(f"🔖 Tags: {post_data['tags']}")
        
        # Use WordPressRESTPublisher directly with async context manager
        async with WordPressRESTPublisher(wp_config) as publisher:
            print("⚡ Publishing to WordPress...")
            result = await publisher.publish_post(**post_data)
            
            if result:
                print(f"\n🎉 ✅ DIRECT WORDPRESS PUBLISHING SUCCESS!")
                print(f"📝 Post ID: {result.get('id')}")
                print(f"🔗 Post URL: {result.get('link')}")
                print(f"📊 Post Status: {result.get('status')}")
                print(f"📅 Publication Date: {result.get('date')}")
                
                # Save results
                results = {
                    "method": "direct_wordpress_publishing",
                    "success": True,
                    "post_id": result.get('id'),
                    "post_url": result.get('link'),
                    "post_status": result.get('status'),
                    "content_length": len(response.answer),
                    "confidence_score": response.confidence_score,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                filename = f"direct_wordpress_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Results saved to: {filename}")
                print(f"\n🌟 SUCCESS: Direct WordPress publishing works perfectly!")
                print(f"🔧 Issue: The problem is in the Universal RAG Chain's integration layer")
                
                return True
                
            else:
                print(f"\n❌ WordPress publishing returned None")
                return False
        
    except Exception as e:
        print(f"\n❌ Direct WordPress publishing failed: {str(e)}")
        print(f"💡 Error type: {type(e).__name__}")
        import traceback
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return False
    
    finally:
        print("\n" + "=" * 60)
        print("🏆 DIRECT WORDPRESS PUBLISHING TEST COMPLETE")

if __name__ == "__main__":
    success = asyncio.run(test_direct_wordpress_fix())
    
    if success:
        print("\n🎉 CONCLUSION: WordPress publishing works when used directly!")
        print("🔧 NEXT STEP: Fix the Universal RAG Chain integration layer")
    else:
        print("\n⚠️ CONCLUSION: Even direct publishing failed - deeper investigation needed") 