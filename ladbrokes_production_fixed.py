#!/usr/bin/env python3
"""
🎰 LADBROKES PRODUCTION TEST - FIXED CONFIGURATION 
Based on the successful TrustDice MT Casino script that produced Post ID 51371

This script uses the EXACT working configuration from run_trustdice_mt_casino_fixed.py
but adapted for Ladbrokes casino review production testing.
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# ✅ CRITICAL: Set WordPress environment variables BEFORE importing the chain
# Using EXACT same variable names and values as the WORKING script
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "your-wordpress-password-here"

print("🔧 WordPress environment variables set (WORKING CONFIGURATION):")
print(f"   WORDPRESS_URL: {os.environ.get('WORDPRESS_URL')}")
print(f"   WORDPRESS_USERNAME: {os.environ.get('WORDPRESS_USERNAME')}")
print(f"   WORDPRESS_PASSWORD: {'*' * len(os.environ.get('WORDPRESS_PASSWORD', ''))}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def run_ladbrokes_production_fixed():
    """Run Ladbrokes analysis using FIXED WordPress publishing configuration"""
    
    print("🎰 LADBROKES PRODUCTION TEST - FIXED CONFIGURATION")
    print("=" * 70)
    print("🎯 Target: Ladbrokes Casino → MT Casino WordPress Post")
    print("📝 WordPress Site: https://www.crashcasino.io")
    print("🔐 WordPress: ✅ EXACT working configuration from Post ID 51371")
    print("🏗️ Post Type: MT Casino (Custom Post Type)")
    print()
    
    # Initialize RAG chain with EXACT same configuration as working script
    print("🚀 Initializing Universal RAG Chain v6.1 with WORKING configuration...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_comprehensive_web_research=True,
        enable_wordpress_publishing=True,
        enable_dataforseo_images=True,
        enable_web_search=True,
        enable_cache_bypass=False
    )
    
    print("✅ Chain initialized with WORKING WordPress publishing configuration")
    
    # Ladbrokes casino query focused on MT Casino requirements
    ladbrokes_query = """Create a comprehensive professional Ladbrokes Casino review for MT Casino custom post type.
    
    Cover: licensing and regulation, cryptocurrency features and payment methods, games portfolio including crash games, 
    welcome bonuses and promotions, mobile experience and usability, customer support quality, security measures, 
    user experience analysis, pros and cons, and final rating with detailed justification.
    
    Format for WordPress MT Casino post type with proper SEO optimization."""
    
    print(f"🔍 Ladbrokes MT Casino Query:")
    print(f"📝 {ladbrokes_query}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        # Execute RAG chain with EXACT same input structure as working script
        print("⚡ Executing chain with WORKING WordPress publishing...")
        
        # Use the EXACT input structure from working script
        query_input = {
            "question": ladbrokes_query,
            "publish_to_wordpress": True
        }
        
        response = await rag_chain.ainvoke(query_input)
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"\n⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Response Length: {len(response.answer)} characters")
        print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
        print(f"📚 Sources: {len(response.sources)} sources")
        print(f"🖼️ Images: {response.metadata.get('images_found', 0)} found")
        
        # Check Ladbrokes content quality
        ladbrokes_count = response.answer.lower().count('ladbrokes')
        print(f"🏷️ Ladbrokes mentions: {ladbrokes_count}")
        
        # Check WordPress publishing result
        wordpress_published = response.metadata.get("wordpress_published", False)
        
        if wordpress_published:
            wp_url = response.metadata.get("wordpress_url", "")
            post_id = response.metadata.get("wordpress_post_id", "")
            post_type = response.metadata.get("wordpress_post_type", "unknown")
            print(f"\n🌟 ✅ WORDPRESS PUBLISHING SUCCESS!")
            print(f"📝 Post ID: {post_id}")
            print(f"🔗 Post URL: {wp_url}")
            print(f"📂 Post Type: {post_type}")
            print(f"📊 Category: {response.metadata.get('wordpress_category', 'N/A')}")
            print(f"🏷️ Custom Fields: {response.metadata.get('wordpress_custom_fields_count', 0)}")
            print(f"🔖 Tags: {response.metadata.get('wordpress_tags_count', 0)}")
            
            # Verify MT Casino post type
            if 'mt_listing' in str(post_type).lower() or '/casino/' in str(wp_url):
                print(f"🎰 MT Casino: ✅ Published to MT Casino custom post type")
            else:
                print(f"⚠️ Warning: May not be MT Casino post type")
                
        else:
            print(f"\n❌ WordPress publishing failed")
            error = response.metadata.get("wordpress_error", "Unknown error")
            print(f"💡 Error: {error}")
        
        # Show content preview
        print("\n📄 LADBROKES CONTENT PREVIEW:")
        print("=" * 50)
        content_preview = response.answer[:1000] + "..." if len(response.answer) > 1000 else response.answer
        print(content_preview)
        
        # Save results
        results = {
            "method": "ladbrokes_production_fixed",
            "query": ladbrokes_query,
            "processing_time": processing_time,
            "content_length": len(response.answer),
            "confidence_score": response.confidence_score,
            "sources_count": len(response.sources),
            "images_found": response.metadata.get('images_found', 0),
            "ladbrokes_mentions": ladbrokes_count,
            "wordpress_published": wordpress_published,
            "wordpress_post_id": response.metadata.get("wordpress_post_id"),
            "wordpress_url": response.metadata.get("wordpress_url"),
            "wordpress_post_type": response.metadata.get("wordpress_post_type"),
            "wordpress_error": response.metadata.get("wordpress_error"),
            "full_content": response.answer,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"ladbrokes_production_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
        print("🏆 LADBROKES PRODUCTION TEST COMPLETE")
        print("✅ Universal RAG CMS v6.1: WORKING CONFIGURATION APPLIED")
        print("🔧 WordPress Publishing: EXACT SAME ENV VARS AS POST 51371")
        print("🎰 MT Casino Integration: CUSTOM POST TYPE TARGETING")
        print("🖼️ Image Integration: BULLETPROOF V1 PATTERNS")
        print("🌐 Live WordPress: https://www.crashcasino.io")

if __name__ == "__main__":
    result = asyncio.run(run_ladbrokes_production_fixed())
    
    if result and result.metadata.get("wordpress_published"):
        print("\n🎉 SUCCESS! Ladbrokes published using working configuration!")
        print(f"📝 Ladbrokes article published: {result.metadata.get('wordpress_url')}")
    else:
        print("\n⚠️ WordPress publishing still needs debugging")
        print("🔧 The environment variables or authentication may need adjustment") 