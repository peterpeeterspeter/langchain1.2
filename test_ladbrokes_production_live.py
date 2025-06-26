#!/usr/bin/env python3
"""
🎰 LIVE PRODUCTION TEST: Ladbrokes Casino Review 
Testing the integrated default chain fixes for WordPress publishing
All fixes should now work out-of-box since they're in the default chain
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# ✅ Set WordPress environment variables (working configuration)
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"  
os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_PASSWORD", "")

# Ensure we have the password
if not os.environ["WORDPRESS_PASSWORD"]:
    print("❌ WORDPRESS_PASSWORD environment variable not set")
    print("💡 Please set your WordPress application password in the environment")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def test_ladbrokes_production():
    """Test Ladbrokes review with fixed default chain"""
    
    print("🎰 LIVE PRODUCTION TEST: Ladbrokes Casino Review")
    print("=" * 60)
    print("🎯 Testing: Default Universal RAG Chain (all fixes integrated)")
    print("🌐 WordPress: https://www.crashcasino.io")
    print("📋 Post Type: MT Casino (mt_listing)")
    print("🖼️ Images: 6 images per review expected")
    print("🔧 Config: Uses working environment from successful Post IDs 51371, 51406")
    print()
    
    # Create default chain (should have all fixes)
    print("🚀 Creating Universal RAG Chain with integrated fixes...")
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
    
    # Ladbrokes query for MT Casino custom post type
    query = """Create a comprehensive professional Ladbrokes Casino review for MT Casino custom post type.

    Provide detailed analysis including:
    - Licensing and regulatory compliance
    - Game selection and software providers
    - Cryptocurrency and payment options
    - Welcome bonuses and ongoing promotions  
    - Mobile compatibility and user experience
    - Customer support quality and availability
    - Security measures and player protection
    - Pros and cons analysis
    - Final rating and recommendation

    Format for WordPress publishing with SEO optimization."""
    
    print(f"🔍 Query: Ladbrokes Casino Review")
    print(f"📏 Query length: {len(query)} characters")
    
    start_time = time.time()
    
    try:
        print("\n⚡ Executing production chain...")
        
        result = await chain.ainvoke({
            "question": query,
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
            
            # Check if it's MT Casino custom post type
            if 'mt_listing' in str(post_type).lower() or '/casino/' in str(post_url):
                print(f"🎰 ✅ Published to MT Casino custom post type")
            else:
                print(f"⚠️ Note: Post type verification needed")
                
            # Check custom fields
            fields_count = result.metadata.get("wordpress_custom_fields_count", 0)
            images_count = result.metadata.get("images_uploaded_count", 0)
            
            print(f"🏷️ Custom fields: {fields_count}")
            print(f"🖼️ Images uploaded: {images_count}")
            
        else:
            print(f"\n❌ WordPress publishing failed")
            error = result.metadata.get("wordpress_error", "Unknown error")
            print(f"💡 Error: {error}")
        
        # Content quality check
        ladbrokes_mentions = result.answer.lower().count('ladbrokes')
        print(f"\n📈 Content Quality:")
        print(f"   Ladbrokes mentions: {ladbrokes_mentions}")
        print(f"   Content structure: {'✅ Passed' if len(result.answer) > 5000 else '⚠️ Short'}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "test_type": "ladbrokes_production_live",
            "query": query,
            "response": result.answer,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "metadata": result.metadata,
            "processing_time": processing_time,
            "ladbrokes_mentions": ladbrokes_mentions,
            "timestamp": timestamp
        }
        
        filename = f"ladbrokes_live_test_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Show content preview
        print(f"\n📄 Content Preview (first 500 characters):")
        print("-" * 50)
        preview = result.answer[:500] + "..." if len(result.answer) > 500 else result.answer
        print(preview)
        print("-" * 50)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        print(f"\n" + "=" * 60)
        print("🏁 PRODUCTION TEST COMPLETE")
        print("🔧 All fixes integrated in default chain")
        print("🎰 MT Casino custom post type targeting")
        print("🖼️ Image upload and embedding")
        print("✅ Content validation fixes applied")

if __name__ == "__main__":
    result = asyncio.run(test_ladbrokes_production())
    
    if result and result.metadata.get("wordpress_published"):
        print("\n🚀 SUCCESS: Production chain working perfectly!")
        url = result.metadata.get("wordpress_url")
        print(f"🌐 Live URL: {url}")
    else:
        print("\n🔍 Check environment variables and API keys") 