#!/usr/bin/env python3
"""
🎰 BWIN CASINO PRODUCTION REVIEW
Generate and publish a comprehensive Bwin Casino review using the production chain
All WordPress publishing fixes are integrated
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def create_bwin_casino_review():
    """Generate comprehensive Bwin Casino review with WordPress publishing"""
    
    print("🎰 BWIN CASINO PRODUCTION REVIEW")
    print("=" * 60)
    print("🚀 Using: Default Universal RAG Chain (all fixes integrated)")
    print("🌐 Publishing to: https://www.crashcasino.io")
    print("📋 Post Type: MT Casino (mt_listing)")
    print("🖼️ Images: 6 images per review expected")
    print()
    
    # Create Universal RAG Chain with all production features
    print("🔧 Creating Universal RAG Chain with WordPress publishing...")
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
    
    # Comprehensive Bwin Casino review query
    bwin_query = """Create a comprehensive professional Bwin Casino review for MT Casino custom post type.

    Provide detailed analysis including:
    - Licensing and regulatory compliance (UK Gambling Commission, MGA Malta)
    - Game selection and software providers (NetEnt, Microgaming, Evolution Gaming)
    - Sports betting integration and unique features
    - Welcome bonuses and ongoing promotions
    - Mobile app compatibility and user experience
    - Customer support quality and availability
    - Security measures and player protection
    - Payment methods and withdrawal processes
    - VIP program and loyalty rewards
    - Pros and cons analysis
    - Final rating and recommendation

    Format for WordPress MT Casino post type with SEO optimization and engaging content structure. Focus on Bwin's established reputation as a major European operator since 1997."""
    
    print(f"🔍 Query: Bwin Casino Review for MT Casino")
    print(f"📏 Query length: {len(bwin_query)} characters")
    
    start_time = time.time()
    
    try:
        print("\n⚡ Executing production chain with WordPress publishing...")
        
        # Execute the chain with WordPress publishing enabled
        result = await chain.ainvoke({
            "question": bwin_query,
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
        bwin_mentions = result.answer.lower().count('bwin')
        casino_mentions = result.answer.lower().count('casino')
        sports_mentions = result.answer.lower().count('sports')
        
        print(f"\n📈 Content Quality Analysis:")
        print(f"   Bwin mentions: {bwin_mentions}")
        print(f"   Casino mentions: {casino_mentions}")
        print(f"   Sports mentions: {sports_mentions}")
        print(f"   Content structure: {'✅ Comprehensive' if len(result.answer) > 5000 else '⚠️ Needs expansion'}")
        print(f"   Confidence level: {'✅ High' if result.confidence_score > 0.6 else '⚠️ Moderate'}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "test_type": "bwin_casino_production_review",
            "query": bwin_query,
            "response": result.answer,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "metadata": result.metadata,
            "processing_time": processing_time,
            "content_analysis": {
                "bwin_mentions": bwin_mentions,
                "casino_mentions": casino_mentions,
                "sports_mentions": sports_mentions,
                "content_length": len(result.answer),
                "quality_rating": "high" if len(result.answer) > 5000 and result.confidence_score > 0.6 else "moderate"
            },
            "timestamp": timestamp
        }
        
        filename = f"bwin_casino_production_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Show content preview
        print(f"\n📄 Bwin Casino Content Preview (first 600 characters):")
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
        print("🏁 BWIN CASINO PRODUCTION REVIEW COMPLETE")
        print("✅ Universal RAG Chain: Default configuration with all fixes")
        print("🎰 MT Casino integration: Custom post type targeting")
        print("🖼️ Image system: 6 images per review capability")
        print("🔧 Content validation: Searches entire content correctly")
        print("🌐 WordPress: Environment variable priority system")

if __name__ == "__main__":
    print("🚀 Starting Bwin Casino production review...")
    result = asyncio.run(create_bwin_casino_review())
    
    if result and result.metadata.get("wordpress_published"):
        print("\n🚀 SUCCESS: Bwin Casino review published successfully!")
        url = result.metadata.get("wordpress_url")
        post_id = result.metadata.get("wordpress_post_id")
        print(f"🌐 Live URL: {url}")
        print(f"📝 Post ID: {post_id}")
        print(f"🎉 Production chain working perfectly!")
    else:
        print("\n🔍 Review generated but WordPress publishing may need verification")
        print("💡 Check the content quality and error details above") 