#!/usr/bin/env python3
"""
🎰 TRUSTDICE CASINO - FINAL WORDPRESS PUBLISHING FIX
Bypass flag propagation issues by running generation and WordPress publishing separately
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# ✅ Set WordPress environment variables
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"  
os.environ["WORDPRESS_PASSWORD"] = "your-wordpress-password-here"

print("🔧 WordPress environment variables set")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain
from integrations.wordpress_publisher import WordPressRESTPublisher, WordPressConfig

async def run_trustdice_final_fix():
    """Run TrustDice with guaranteed WordPress publishing by doing it separately"""
    
    print("🎰 TRUSTDICE CASINO - FINAL WORDPRESS FIX")
    print("=" * 60)
    print("🎯 Strategy: Generate content + Publish separately")
    print("📝 WordPress Site: https://www.crashcasino.io")
    print()
    
    # Step 1: Generate content WITHOUT WordPress publishing
    print("🚀 Step 1: Generate TrustDice content...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_comprehensive_web_research=True,
        enable_wordpress_publishing=False,  # Disable automatic publishing
        enable_dataforseo_images=True,
        enable_web_search=True
    )
    
    trustdice_query = "Create a comprehensive TrustDice Casino review covering licensing, cryptocurrency features, games, bonuses, mobile experience, support, security, and final rating."
    
    start_time = time.time()
    
    # Generate content
    query_input = {
        "question": trustdice_query,
        "publish_to_wordpress": False  # Don't auto-publish
    }
    
    print("⚡ Generating content...")
    response = await rag_chain.ainvoke(query_input)
    
    generation_time = time.time() - start_time
    
    print(f"✅ Content generated in {generation_time:.2f} seconds")
    print(f"📊 Content Length: {len(response.answer)} characters")
    print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
    
    # Step 2: Publish to WordPress directly
    print("\n🚀 Step 2: Publishing to WordPress directly...")
    
    try:
        # Create WordPress config
        wp_config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="nmlwh",
            application_password="your-wordpress-password-here"
        )
        
        # Create post data  
        post_data = {
            "title": "TrustDice Casino Review - Comprehensive Analysis",
            "content": response.answer,
            "status": "publish",  # Publish immediately
            "categories": [1],  # General category
            "tags": ["trustdice", "casino", "review", "final-fix"],
            "custom_fields": {
                "review_type": "casino_review",
                "confidence_score": response.confidence_score,
                "generation_time": generation_time,
                "method": "final_fix_direct_publishing",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        print(f"📝 Post Title: {post_data['title']}")
        print(f"📊 Content Length: {len(post_data['content'])} characters")
        print(f"🏷️ Categories: {post_data['categories']}")
        print(f"🔖 Tags: {post_data['tags']}")
        
        # Publish directly with async context manager
        async with WordPressRESTPublisher(wp_config) as publisher:
            print("⚡ Publishing to WordPress...")
            result = await publisher.publish_post(**post_data)
            
            if result:
                total_time = time.time() - start_time
                
                print(f"\n🎉 ✅ FINAL FIX SUCCESS!")
                print(f"📝 Post ID: {result.get('id')}")
                print(f"🔗 Post URL: {result.get('link')}")
                print(f"📊 Post Status: {result.get('status')}")
                print(f"📅 Publication Date: {result.get('date')}")
                print(f"⏱️ Total Time: {total_time:.2f} seconds")
                
                # Save results
                results = {
                    "method": "final_fix_separate_publishing",
                    "success": True,
                    "post_id": result.get('id'),
                    "post_url": result.get('link'),
                    "post_status": result.get('status'),
                    "content_length": len(response.answer),
                    "confidence_score": response.confidence_score,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "timestamp": datetime.now().isoformat(),
                    "full_content": response.answer
                }
                
                filename = f"trustdice_final_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\n💾 Results saved to: {filename}")
                print(f"\n🌟 CONCLUSION: WordPress publishing works perfectly when done separately!")
                print(f"🔧 SOLUTION: Bypass chain flag propagation by handling publishing directly")
                
                return True
                
            else:
                print(f"\n❌ WordPress publishing returned None")
                return False
        
    except Exception as e:
        print(f"\n❌ WordPress publishing failed: {str(e)}")
        print(f"💡 Error type: {type(e).__name__}")
        import traceback
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return False
    
    finally:
        print("\n" + "=" * 60)
        print("🏆 TRUSTDICE FINAL FIX TEST COMPLETE")

if __name__ == "__main__":
    success = asyncio.run(run_trustdice_final_fix())
    
    if success:
        print("\n🎉 SUCCESS! TrustDice casino review published to WordPress!")
        print("🔧 WordPress publishing works when done separately from the chain")
    else:
        print("\n⚠️ Publishing failed - needs further investigation") 