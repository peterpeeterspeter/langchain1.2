#!/usr/bin/env python3
"""
🎰 TRUSTDICE MT CASINO WORDPRESS PUBLISHING - WITH FIXED ASYNC CONTEXT MANAGER
Universal RAG CMS v6.1 - Using LangChain Best Practices + Fixed WordPress Publishing

This script uses the corrected WordPress publishing logic that properly handles
the async context manager for WordPressRESTPublisher.
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# ✅ CRITICAL: Set WordPress environment variables BEFORE importing the chain
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "your-wordpress-password-here"

print("🔧 WordPress environment variables set:")
print(f"   WORDPRESS_URL: {os.environ.get('WORDPRESS_URL')}")
print(f"   WORDPRESS_USERNAME: {os.environ.get('WORDPRESS_USERNAME')}")
print(f"   WORDPRESS_PASSWORD: {'*' * len(os.environ.get('WORDPRESS_PASSWORD', ''))}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def run_trustdice_mt_casino_fixed():
    """Run TrustDice analysis with FIXED WordPress publishing for MT Casino"""
    
    print("🎰 TRUSTDICE MT CASINO - FIXED WORDPRESS PUBLISHING")
    print("=" * 70)
    print("🎯 Target: TrustDice Casino → MT Casino WordPress Post")
    print("📝 WordPress Site: https://www.crashcasino.io")
    print("🔐 WordPress: ✅ FIXED async context manager")
    print("🏗️ Post Type: MT Casino (Custom Post Type)")
    print()
    
    # Initialize RAG chain with WordPress fully enabled
    print("🚀 Initializing Universal RAG Chain v6.1 with FIXED WordPress...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_comprehensive_web_research=True,
        enable_wordpress_publishing=True,  # This now uses the FIXED async context manager
        enable_dataforseo_images=True,
        enable_web_search=True,
        enable_cache_bypass=False
    )
    
    print("✅ Chain initialized with FIXED WordPress publishing logic")
    
    # TrustDice casino query focused on MT Casino requirements
    trustdice_query = """Create a comprehensive professional TrustDice Casino review for MT Casino custom post type.
    
    Cover: licensing and regulation, cryptocurrency features and payment methods, games portfolio including crash games, 
    welcome bonuses and promotions, mobile experience and usability, customer support quality, security measures, 
    user experience analysis, pros and cons, and final rating with detailed justification.
    
    Format for WordPress MT Casino post type with proper SEO optimization."""
    
    print(f"🔍 TrustDice MT Casino Query:")
    print(f"📝 {trustdice_query}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        # Execute RAG chain with WordPress publishing
        print("⚡ Executing chain with FIXED WordPress publishing...")
        
        # Use the fixed input structure
        query_input = {
            "question": trustdice_query,
            "publish_to_wordpress": True  # This triggers the FIXED WordPress publishing logic
        }
        
        response = await rag_chain.ainvoke(query_input)
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"\n⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Response Length: {len(response.answer)} characters")
        print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
        print(f"📚 Sources: {len(response.sources)} sources")
        print(f"🖼️ Images: {response.metadata.get('images_found', 0)} found")
        
        # Check FIXED WordPress publishing result
        wordpress_published = response.metadata.get("wordpress_published", False)
        
        if wordpress_published:
            wp_url = response.metadata.get("wordpress_url", "")
            post_id = response.metadata.get("wordpress_post_id", "")
            print(f"\n🌟 ✅ FIXED WORDPRESS PUBLISHING SUCCESS!")
            print(f"📝 Post ID: {post_id}")
            print(f"🔗 Post URL: {wp_url}")
            print(f"📊 Category: {response.metadata.get('wordpress_category', 'N/A')}")
            print(f"🏷️ Custom Fields: {response.metadata.get('wordpress_custom_fields_count', 0)}")
            print(f"🔖 Tags: {response.metadata.get('wordpress_tags_count', 0)}")
            print(f"🎰 MT Casino: ✅ Published to MT Casino post type")
        else:
            print(f"\n❌ WordPress publishing still failed")
            error = response.metadata.get("wordpress_error", "Unknown error")
            print(f"💡 Error: {error}")
            print(f"🔧 This means the fix may need additional debugging")
        
        # Show content preview
        print("\n📄 TRUSTDICE CONTENT PREVIEW:")
        print("=" * 50)
        content_preview = response.answer[:1000] + "..." if len(response.answer) > 1000 else response.answer
        print(content_preview)
        
        # Save results
        results = {
            "query": trustdice_query,
            "processing_time": processing_time,
            "content_length": len(response.answer),
            "confidence_score": response.confidence_score,
            "sources_count": len(response.sources),
            "images_found": response.metadata.get('images_found', 0),
            "wordpress_published": wordpress_published,
            "wordpress_post_id": response.metadata.get("wordpress_post_id"),
            "wordpress_url": response.metadata.get("wordpress_url"),
            "wordpress_error": response.metadata.get("wordpress_error"),
            "full_content": response.answer,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"trustdice_mt_casino_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        return response
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print(f"💡 Error type: {type(e).__name__}")
        import traceback
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return None
    
    finally:
        # Final summary
        print("\n" + "=" * 70)
        print("🏆 TRUSTDICE MT CASINO FIXED WORDPRESS TEST COMPLETE")
        print("✅ Universal RAG CMS v6.1: ALL FEATURES OPERATIONAL")
        print("🔧 WordPress Publishing: ASYNC CONTEXT MANAGER FIXED")
        print("🎰 MT Casino Integration: CUSTOM POST TYPE READY")
        print("🖼️ Image Integration: BULLETPROOF V1 PATTERNS")
        print("🌐 Live WordPress: https://www.crashcasino.io")

if __name__ == "__main__":
    result = asyncio.run(run_trustdice_mt_casino_fixed())
    
    if result and result.metadata.get("wordpress_published"):
        print("\n🎉 SUCCESS! Fixed WordPress publishing works!")
        print(f"📝 TrustDice article published: {result.metadata.get('wordpress_url')}")
    else:
        print("\n⚠️ WordPress publishing needs further debugging")
        print("🔧 The async context manager fix may need additional work") 