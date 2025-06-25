#!/usr/bin/env python3
"""
🎰 WORKING LADBROKES TEST - CORRECT CREDENTIALS + ALL FEATURES
Final working test with proper WordPress credentials, screenshots, and MT listing
"""

import os
import sys
from pathlib import Path

# SET CORRECT ENVIRONMENT VARIABLES FIRST
print("🔧 Setting correct WordPress environment variables...")
os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"  # CORRECT USERNAME!
os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
os.environ["WORDPRESS_APP_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
os.environ["SUPABASE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzUwMzA3ODYsImV4cCI6MjA1MDYwNjc4Nn0.2TlyVBuONf-4YVy1QrYdEJF13aF8j1NUrElHnJ8oOuE"

print(f"✅ WORDPRESS_SITE_URL = {os.environ.get('WORDPRESS_SITE_URL', 'NOT SET')}")
print(f"✅ WORDPRESS_USERNAME = {os.environ.get('WORDPRESS_USERNAME', 'NOT SET')}")  # Should be nmlwh
print(f"✅ WORDPRESS_PASSWORD = {'SET' if os.environ.get('WORDPRESS_PASSWORD') else 'NOT SET'}")
print(f"✅ WORDPRESS_APP_PASSWORD = {'SET' if os.environ.get('WORDPRESS_APP_PASSWORD') else 'NOT SET'}")

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def working_ladbrokes_test():
    """Working test with correct credentials and all features"""
    
    print("🎰 WORKING LADBROKES TEST - ALL FEATURES ENABLED")
    print("=" * 60)
    
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
        
        print("✅ Chain initialized successfully with all features")
        
        # Generate Ladbrokes review with correct publishing parameters
        print("\n🎰 Generating and publishing Ladbrokes casino review...")
        
        # Use correct publishing parameters based on chain code analysis
        result = await chain.ainvoke(
            {
                "question": "Create a comprehensive Ladbrokes casino review with screenshots and images",
                "publish_to_wordpress": True,  # In inputs dict
                "wordpress_post_type": "mt_listing",  # Force MT listing
                "enable_screenshot_capture": True,  # Enable screenshots
                "capture_screenshots": True
            },
            publish_to_wordpress=True  # Also as parameter
        )
        
        print("\n📊 FINAL RESULTS:")
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
                print(f"\n🎉 SUCCESS! Ladbrokes review published!")
                print(f"Live URL: {wp_url}")
                
                # Verify it's an MT listing and has all content
                import requests
                try:
                    response = requests.get(wp_url, timeout=10)
                    if response.status_code == 200:
                        content = response.text.lower()
                        has_ladbrokes = 'ladbrokes' in content
                        has_images = '<img' in content or 'wp-image' in content
                        has_mt_listing = 'mt_listing' in content or 'mt-listing' in content
                        
                        print(f"✅ Live verification:")
                        print(f"   🏷️ Contains Ladbrokes: {'✅' if has_ladbrokes else '❌'}")
                        print(f"   🖼️ Contains images: {'✅' if has_images else '❌'}")
                        print(f"   📂 MT listing format: {'✅' if has_mt_listing else '❌'}")
                        print(f"   📊 Content size: {len(content):,} characters")
                        
                        if has_ladbrokes and has_images:
                            print(f"\n🏆 COMPLETE SUCCESS! Ladbrokes review published with images and correct content!")
                        else:
                            print(f"\n⚠️ Content issues detected - review the published post")
                            
                    else:
                        print(f"⚠️ URL returned status: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ Could not verify URL: {e}")
            else:
                print("❌ WordPress publishing failed")
                
        else:
            print("❌ No WordPress metadata found in result")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(working_ladbrokes_test()) 