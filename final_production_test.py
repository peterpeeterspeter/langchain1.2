#!/usr/bin/env python3
"""
🎰 FINAL PRODUCTION TEST - LADBROKES REVIEW WITH WORDPRESS PUBLISHING
This script ensures ALL environment variables are properly set and forces successful publishing

CRITICAL FIXES:
1. ✅ Explicit WordPress environment variable setting
2. ✅ Force WordPress service initialization 
3. ✅ Enable MT listing custom post type
4. ✅ Screenshot functionality with created bucket
5. ✅ Complete debugging output
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def final_production_test():
    """Final production test with guaranteed WordPress publishing"""
    
    print("🎰 FINAL PRODUCTION TEST - LADBROKES REVIEW WITH WORDPRESS PUBLISHING")
    print("=" * 90)
    
    # CRITICAL: Set ALL WordPress environment variables explicitly
    print("\n📋 STEP 1: Explicit Environment Configuration")
    
    # WordPress credentials - explicit setting
    os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
    os.environ["WORDPRESS_USERNAME"] = os.getenv("WORDPRESS_USERNAME", "admin")
    os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
    
    # Supabase - confirmed working
    os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
    os.environ["SUPABASE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzUwMzA3ODYsImV4cCI6MjA1MDYwNjc4Nn0.2TlyVBuONf-4YVy1QrYdEJF13aF8j1NUrElHnJ8oOuE"
    
    # Debug: Print all WordPress-related environment variables
    print("📝 WordPress Environment Variables:")
    for key, value in os.environ.items():
        if "WORDPRESS" in key.upper():
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"   {key}: {masked_value}")
    
    # Verify critical variables
    required_vars = {
        "WORDPRESS_SITE_URL": os.getenv("WORDPRESS_SITE_URL"),
        "WORDPRESS_USERNAME": os.getenv("WORDPRESS_USERNAME"), 
        "WORDPRESS_PASSWORD": os.getenv("WORDPRESS_PASSWORD"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY")
    }
    
    print("\n✅ Required Variables Check:")
    all_set = True
    for var, value in required_vars.items():
        status = "✅" if value and len(value) > 5 else "❌"
        print(f"   {status} {var}: {len(value) if value else 0} chars")
        if not value or len(value) < 5:
            all_set = False
    
    if not all_set:
        print("❌ Critical environment variables missing! Cannot proceed.")
        return
    
    # STEP 2: Initialize chain with optimized settings
    print("\n⚙️ STEP 2: Initialize Universal RAG Chain (Optimized)")
    
    try:
        from chains.universal_rag_lcel import UniversalRAGChain
        
        # Create chain with focused settings for WordPress publishing
        chain = UniversalRAGChain(
            model_name="gpt-4.1-mini",
            temperature=0.1,
            enable_caching=True,
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=True,
            enable_dataforseo_images=True,
            enable_wordpress_publishing=True,  # CRITICAL
            enable_fti_processing=True,
            enable_security=True,
            enable_profiling=False,  # Disable for speed
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_screenshot_evidence=True,  # Screenshots bucket now exists
            enable_hyperlink_generation=True,
            enable_response_storage=True
        )
        
        print("✅ Universal RAG Chain initialized")
        print(f"   📊 Features enabled: {chain._count_active_features()}")
        
        # Manually verify WordPress service initialization
        if hasattr(chain, 'wordpress_service'):
            print("✅ WordPress service detected")
            if hasattr(chain.wordpress_service, 'config'):
                config = chain.wordpress_service.config
                print(f"   🌐 Site URL: {config.site_url}")
                print(f"   👤 Username: {config.username}")
                print(f"   🔑 Password: {'*' * len(config.password) if config.password else 'MISSING'}")
        else:
            print("❌ WordPress service NOT initialized")
            
    except Exception as e:
        print(f"❌ Chain initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # STEP 3: Generate Ladbrokes review with explicit publishing request
    print("\n🎰 STEP 3: Generate Ladbrokes Review + WordPress Publishing")
    
    query = "Create a comprehensive Ladbrokes casino review"
    print(f"📝 Query: {query}")
    
    try:
        # Explicit publishing configuration
        chain._publish_to_wordpress = True
        print("🔧 WordPress publishing explicitly enabled at chain level")
        
        start_time = datetime.now()
        
        # Generate with explicit WordPress publishing flag
        result = await chain.ainvoke(
            {"question": query},
            config={
                "publish_to_wordpress": True,
                "wordpress_post_type": "mt_listing",  # Force MT listing
                "force_wordpress_publish": True
            }
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Generation completed in {duration:.1f} seconds")
        
        # STEP 4: Comprehensive result analysis
        print("\n📊 STEP 4: Comprehensive Result Analysis")
        
        # Content quality
        print(f"📄 Content length: {len(result.answer):,} characters")
        print(f"🎯 Confidence score: {result.confidence_score:.3f}")
        print(f"📚 Sources used: {len(result.sources)}")
        
        # Ladbrokes content verification
        ladbrokes_mentions = result.answer.lower().count('ladbrokes')
        print(f"🏷️ Ladbrokes mentions: {ladbrokes_mentions}")
        print(f"📝 Content alignment: {'✅ Excellent' if ladbrokes_mentions > 20 else '⚠️ Poor'}")
        
        # Image/screenshot verification
        image_count = result.answer.count('<img')
        screenshot_mentions = result.answer.lower().count('screenshot')
        print(f"🖼️ Images in content: {image_count}")
        print(f"📸 Screenshot references: {screenshot_mentions}")
        
        # WordPress publishing verification
        if hasattr(result, 'metadata') and result.metadata:
            wp_published = result.metadata.get('wordpress_published', False)
            wp_post_id = result.metadata.get('wordpress_post_id')
            wp_url = result.metadata.get('wordpress_url')
            wp_post_type = result.metadata.get('wordpress_post_type', 'unknown')
            wp_screenshots = result.metadata.get('wordpress_screenshot_count', 0)
            
            print(f"\n🌐 WordPress Publishing Results:")
            print(f"   📝 Published: {'✅ SUCCESS' if wp_published else '❌ FAILED'}")
            if wp_post_id:
                print(f"   🆔 Post ID: {wp_post_id}")
            if wp_url:
                print(f"   🔗 URL: {wp_url}")
            print(f"   📂 Post Type: {wp_post_type}")
            print(f"   📸 Screenshots: {wp_screenshots}")
            
            if wp_published:
                print(f"\n🎉 🎉 🎉 COMPLETE SUCCESS! 🎉 🎉 🎉")
                print(f"Ladbrokes casino review successfully published!")
                print(f"✅ URL: {wp_url}")
                print(f"✅ Post ID: {wp_post_id}")
                print(f"✅ Content Quality: {ladbrokes_mentions} Ladbrokes mentions")
                print(f"✅ Images: {image_count} images embedded")
                
                # Test the published URL
                try:
                    import requests
                    response = requests.get(wp_url, timeout=10)
                    if response.status_code == 200:
                        print(f"✅ URL verification: LIVE and accessible")
                    else:
                        print(f"⚠️ URL status: {response.status_code}")
                except Exception as url_error:
                    print(f"⚠️ URL verification failed: {url_error}")
            else:
                print(f"\n❌ WordPress publishing failed")
                # Check for error details in metadata
                wp_error = result.metadata.get('wordpress_error', 'Unknown error')
                print(f"   ❌ Error: {wp_error}")
        else:
            print("\n❌ No WordPress metadata found in result")
        
        return result
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(final_production_test()) 