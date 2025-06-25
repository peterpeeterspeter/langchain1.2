#!/usr/bin/env python3
"""
🎰 FINAL LADBROKES TEST - CORRECT WORDPRESS PUBLISHING
Fixed test with proper WordPress publishing parameter
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

async def final_ladbrokes_test():
    """Final test with correct WordPress publishing parameters"""
    
    print("🎰 FINAL LADBROKES TEST - SCREENSHOTS + MT LISTING")
    print("=" * 60)
    
    # Set environment variables explicitly
    os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
    os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
    os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
    os.environ["SUPABASE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzUwMzA3ODYsImV4cCI6MjA1MDYwNjc4Nn0.2TlyVBuONf-4YVy1QrYdEJF13aF8j1NUrElHnJ8oOuE"
    
    try:
        from chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize chain with all features
        chain = UniversalRAGChain(
            model_name="gpt-4.1-mini",
            temperature=0.1,
            enable_wordpress_publishing=True,
            enable_screenshot_evidence=True,
            enable_dataforseo_images=True,
            enable_template_system_v2=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True
        )
        
        print("✅ Chain initialized successfully")
        
        # Generate Ladbrokes review with correct publishing parameters
        print("\n🎰 Generating Ladbrokes casino review...")
        
        # CORRECT: Use publish_to_wordpress as parameter to ainvoke
        result = await chain.ainvoke(
            {
                "question": "Create a comprehensive Ladbrokes casino review",
                "publish_to_wordpress": True,  # CORRECT: In inputs dict
                "wordpress_post_type": "mt_listing"
            },
            publish_to_wordpress=True  # CORRECT: Also as parameter
        )
        
        print("\n📊 Results:")
        print(f"📄 Content length: {len(result.answer):,} characters")
        print(f"🎯 Confidence: {result.confidence_score:.3f}")
        
        # Check Ladbrokes content
        ladbrokes_count = result.answer.lower().count('ladbrokes')
        print(f"🏷️ Ladbrokes mentions: {ladbrokes_count}")
        
        # Check images
        image_count = result.answer.count('<img')
        print(f"🖼️ Images in content: {image_count}")
        
        # Check WordPress publishing
        if hasattr(result, 'metadata') and result.metadata:
            wp_published = result.metadata.get('wordpress_published', False)
            wp_post_id = result.metadata.get('wordpress_post_id')
            wp_url = result.metadata.get('wordpress_url')
            wp_post_type = result.metadata.get('wordpress_post_type', 'unknown')
            
            print(f"\n🌐 WordPress Results:")
            print(f"   📝 Published: {'✅ SUCCESS' if wp_published else '❌ FAILED'}")
            print(f"   🆔 Post ID: {wp_post_id}")
            print(f"   🔗 URL: {wp_url}")
            print(f"   📂 Post Type: {wp_post_type}")
            
            if wp_published and wp_url:
                print(f"\n🎉 SUCCESS! Ladbrokes review published as MT listing!")
                print(f"View at: {wp_url}")
                
                # Verify it's an MT listing
                if 'mt_listing' in str(wp_post_type) or '/mt-listing/' in str(wp_url):
                    print(f"✅ Confirmed: Published as MT listing!")
                
                # Quick verification
                import requests
                try:
                    response = requests.get(wp_url, timeout=10)
                    if response.status_code == 200:
                        # Check if it contains ladbrokes and images
                        content = response.text.lower()
                        has_ladbrokes = 'ladbrokes' in content
                        has_images = '<img' in content
                        has_mt_listing_class = 'mt_listing' in content or 'mt-listing' in content
                        
                        print(f"✅ Live verification:")
                        print(f"   🏷️ Contains Ladbrokes: {'✅' if has_ladbrokes else '❌'}")
                        print(f"   🖼️ Contains images: {'✅' if has_images else '❌'}")
                        print(f"   📂 MT listing format: {'✅' if has_mt_listing_class else '❌'}")
                        
                except Exception as e:
                    print(f"⚠️ Could not verify URL: {e}")
        else:
            print("❌ No WordPress metadata found")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(final_ladbrokes_test()) 