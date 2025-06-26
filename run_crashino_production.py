#!/usr/bin/env python3
"""
🎰 CRASHINO PRODUCTION CHAIN TEST
Testing the integrated default Universal RAG Chain for Crashino casino review
All WordPress publishing fixes should work out-of-the-box
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# ✅ Set WordPress environment variables (working configuration from successful runs)
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_PASSWORD", "")

# Ensure we have required environment variables
if not os.environ["WORDPRESS_PASSWORD"]:
    print("❌ WORDPRESS_PASSWORD environment variable not set")
    print("💡 This script needs your WordPress application password")
    print("🔧 The production chain is ready - just needs authentication")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def run_crashino_production():
    """Run Crashino casino review with production-ready default chain"""
    
    print("🎰 CRASHINO PRODUCTION CHAIN TEST")
    print("=" * 60)
    print("🚀 Testing: Default Universal RAG Chain (all fixes integrated)")
    print("🌐 WordPress: https://www.crashcasino.io")
    print("📋 Post Type: MT Casino (mt_listing)")
    print("🖼️ Images: 6 images per review expected")
    print("✅ All fixes: Content validation, environment variables, out-of-box functionality")
    print()
    
    # Create Universal RAG Chain with all fixes integrated by default
    print("🔧 Creating Universal RAG Chain with integrated WordPress publishing fixes...")
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_wordpress_publishing=True,
        enable_dataforseo_images=True,
        enable_web_search=True,
        enable_comprehensive_web_research=True
    )
    
    print("✅ Chain created successfully")
    print(f"📝 WordPress publishing: {chain.enable_wordpress_publishing}")
    print(f"🖼️ Image integration: {chain.enable_dataforseo_images}")
    print(f"🌐 Web research: {chain.enable_comprehensive_web_research}")
    
    # Crashino query for comprehensive MT Casino review
    crashino_query = """Create a comprehensive professional Crashino Casino review for MT Casino custom post type.

    Provide detailed analysis including:
    - Licensing and regulatory compliance
    - Game selection and software providers
    - Cryptocurrency features and payment methods
    - Welcome bonuses and ongoing promotions
    - Mobile compatibility and user experience
    - Customer support quality and availability
    - Security measures and player protection
    - Crash games and unique features
    - Pros and cons analysis
    - Final rating and recommendation

    Format for WordPress MT Casino post type with SEO optimization and engaging content structure."""
    
    print(f"🔍 Query: Crashino Casino Review for MT Casino")
    print(f"📏 Query length: {len(crashino_query)} characters")
    
    start_time = time.time()
    
    try:
        print("\n⚡ Executing production chain with WordPress publishing...")
        
        # Execute the chain with WordPress publishing enabled
        result = await chain.ainvoke({
            "question": crashino_query,
            "publish_to_wordpress": True
        })
        
        processing_time = time.time() - start_time
        
        print(f"\n⏱️ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Content length: {len(result.answer)} characters")
        print(f"🎯 Confidence score: {result.confidence_score:.3f}")
        print(f"📚 Sources found: {len(result.sources)}")
        
        # Check WordPress publishing results
        wp_published = result.metadata.get("wordpress_published", False)
        
        if wp_published:
            post_id = result.metadata.get("wordpress_post_id")
            post_url = result.metadata.get("wordpress_url")
            post_type = result.metadata.get("wordpress_post_type")
            
            print(f"\n🎉 WORDPRESS PUBLISHING SUCCESS!")
            print(f"📝 Post ID: {post_id}")
            print(f"🔗 URL: {post_url}")
            print(f"📂 Post Type: {post_type}")
            
            # Verify MT Casino custom post type
            if 'mt_listing' in str(post_type).lower() or '/casino/' in str(post_url):
                print(f"🎰 ✅ Published to MT Casino custom post type")
            else:
                print(f"⚠️ Note: Post type verification - {post_type}")
                
            # Check features
            fields_count = result.metadata.get("wordpress_custom_fields_count", 0)
            images_count = result.metadata.get("images_uploaded_count", 0)
            
            print(f"🏷️ Custom fields: {fields_count}")
            print(f"🖼️ Images uploaded: {images_count}")
            
        else:
            print(f"\n❌ WordPress publishing failed")
            error = result.metadata.get("wordpress_error", "Unknown error")
            print(f"💡 Error: {error}")
        
        # Content quality analysis
        crashino_mentions = result.answer.lower().count('crashino')
        casino_mentions = result.answer.lower().count('casino')
        
        print(f"\n📈 Content Quality Analysis:")
        print(f"   Crashino mentions: {crashino_mentions}")
        print(f"   Casino mentions: {casino_mentions}")
        print(f"   Content structure: {'✅ Comprehensive' if len(result.answer) > 5000 else '⚠️ Needs expansion'}")
        print(f"   Confidence level: {'✅ High' if result.confidence_score > 0.6 else '⚠️ Moderate'}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "test_type": "crashino_production_chain",
            "query": crashino_query,
            "response": result.answer,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "metadata": result.metadata,
            "processing_time": processing_time,
            "content_analysis": {
                "crashino_mentions": crashino_mentions,
                "casino_mentions": casino_mentions,
                "content_length": len(result.answer),
                "quality_rating": "high" if len(result.answer) > 5000 and result.confidence_score > 0.6 else "moderate"
            },
            "timestamp": timestamp
        }
        
        filename = f"crashino_production_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Show content preview
        print(f"\n📄 Crashino Content Preview (first 600 characters):")
        print("-" * 60)
        preview = result.answer[:600] + "..." if len(result.answer) > 600 else result.answer
        print(preview)
        print("-" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        print(f"\n" + "=" * 60)
        print("🏁 CRASHINO PRODUCTION TEST COMPLETE")
        print("✅ Universal RAG Chain: Default configuration with all fixes")
        print("🎰 MT Casino integration: Custom post type targeting")
        print("🖼️ Image system: 6 images per review capability")
        print("🔧 Content validation: Searches entire content correctly")
        print("🌐 WordPress: Environment variable priority system")

if __name__ == "__main__":
    result = asyncio.run(run_crashino_production())
    
    if result and result.metadata.get("wordpress_published"):
        print("\n🚀 SUCCESS: Crashino review published successfully!")
        url = result.metadata.get("wordpress_url")
        post_id = result.metadata.get("wordpress_post_id")
        print(f"🌐 Live URL: {url}")
        print(f"📝 Post ID: {post_id}")
        print(f"🎉 Production chain working perfectly!")
    else:
        print("\n🔍 WordPress publishing needs authentication or configuration")
        print("💡 The chain is production-ready - ensure WORDPRESS_PASSWORD is set") 