#!/usr/bin/env python3
"""
Debug WordPress Publishing - Direct Test
"""

import os
import asyncio
import sys
import logging
from pathlib import Path

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set WordPress environment variables
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "your-wordpress-password-here"

async def test_wordpress_direct():
    """Test WordPress publishing directly"""
    
    print("🔧 DEBUGGING WORDPRESS PUBLISHING DIRECTLY")
    print("=" * 60)
    
    try:
        # Import WordPress components
        from integrations.wordpress_publisher import (
            WordPressConfig,
            WordPressIntegration,
            WordPressRESTPublisher
        )
        
        print("✅ Successfully imported WordPress components")
        
        # Create WordPress config
        wp_config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="nmlwh",
            application_password="your-wordpress-password-here"
        )
        
        print(f"✅ WordPress config created: {wp_config.site_url}")
        
        # Create WordPress integration
        wp_integration = WordPressIntegration(wordpress_config=wp_config)
        print("✅ WordPress integration created")
        
        # Test simple post data
        test_post_data = {
            "title": "Test Post - Debug WordPress Publishing",
            "content": "<p>This is a test post to debug WordPress publishing. Contains <strong>HTML formatting</strong> and images.</p>",
            "status": "draft",
            "categories": [],
            "tags": ["test", "debug"],
            "meta_description": "Test post for debugging WordPress publishing",
            "custom_fields": {
                "test_field": "test_value",
                "debug_timestamp": "2025-06-21T12:22:00Z"
            }
        }
        
        print("📝 Test post data prepared")
        print(f"   Title: {test_post_data['title']}")
        print(f"   Content length: {len(test_post_data['content'])} chars")
        print(f"   Status: {test_post_data['status']}")
        
        # Try direct publishing
        print("🚀 Attempting direct WordPress publishing...")
        
        async with WordPressRESTPublisher(wp_config) as publisher:
            result = await publisher.publish_post(**test_post_data)
            
        print("🎉 ✅ WORDPRESS PUBLISHING SUCCESSFUL!")
        print(f"📝 Post ID: {result.get('id')}")
        print(f"🔗 Post URL: {result.get('link')}")
        print(f"📊 Post Status: {result.get('status')}")
        print(f"📅 Date: {result.get('date')}")
        
        return result
        
    except Exception as e:
        print(f"❌ WordPress publishing failed: {e}")
        print(f"💡 Error type: {type(e).__name__}")
        import traceback
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return None

async def test_wordpress_via_integration():
    """Test WordPress publishing via WordPressIntegration"""
    
    print("\\n🔧 TESTING VIA WORDPRESS INTEGRATION")
    print("=" * 60)
    
    try:
        from integrations.wordpress_publisher import WordPressConfig, WordPressIntegration
        
        # Create config and integration
        wp_config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="nmlwh",
            application_password="your-wordpress-password-here"
        )
        
        wp_integration = WordPressIntegration(wordpress_config=wp_config)
        
        print("✅ WordPress integration via WordPressIntegration created")
        
        # Test if publisher is available
        if wp_integration.publisher:
            print("✅ Publisher available via integration")
        else:
            print("❌ Publisher not available via integration")
            
        # Test the integration's config
        print(f"🌐 Site URL: {wp_integration.config.site_url}")
        print(f"👤 Username: {wp_integration.config.username}")
        print(f"🔐 Password: {'*' * len(wp_integration.config.application_password)}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_wordpress_direct())
    asyncio.run(test_wordpress_via_integration()) 