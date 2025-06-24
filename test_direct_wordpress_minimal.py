#!/usr/bin/env python3
"""
🔧 MINIMAL WORDPRESS PUBLISHING TEST
Bypass all chain complexity and test WordPress publishing directly
"""

import os
import asyncio
import sys

# Set WordPress environment variables
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "q8ZU 4UHD 90vI Ej55 U0Jh yh8c"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_minimal_wordpress():
    """Minimal WordPress publishing test"""
    
    print("🔧 MINIMAL WORDPRESS PUBLISHING TEST")
    print("=" * 50)
    
    try:
        from integrations.wordpress_publisher import WordPressRESTPublisher, WordPressConfig
        
        # Create WordPress config
        wp_config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="nmlwh",
            application_password="q8ZU 4UHD 90vI Ej55 U0Jh yh8c"
        )
        
        print(f"✅ WordPress config created")
        
        # Create simple post data
        post_data = {
            "title": "TrustDice Casino Test - Minimal Publishing",
            "content": "<h2>Test Content</h2><p>This is a minimal test of WordPress publishing functionality.</p>",
            "status": "publish",
            "categories": [],
            "tags": ["test", "trustdice", "casino"],
            "meta_description": "Test post for WordPress publishing",
            "custom_fields": {
                "test_field": "test_value",
                "review_type": "minimal_test"
            }
        }
        
        print(f"✅ Post data created")
        
        # Test WordPress publishing
        async with WordPressRESTPublisher(wp_config) as publisher:
            print("✅ WordPress publisher created")
            
            result = await publisher.publish_post(**post_data)
            
            if result:
                print(f"🎉 SUCCESS! Post published:")
                print(f"   📝 Post ID: {result.get('id')}")
                print(f"   🔗 URL: {result.get('link')}")
                print(f"   📅 Date: {result.get('date')}")
                return True
            else:
                print("❌ FAILED! Result is None")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Testing minimal WordPress publishing...")
    success = await test_minimal_wordpress()
    
    if success:
        print("\n🎉 MINIMAL WORDPRESS PUBLISHING WORKS!")
        print("✅ The issue is in the Universal RAG Chain integration")
    else:
        print("\n❌ WORDPRESS PUBLISHING FAILED!")
        print("🔧 Check WordPress credentials and connectivity")

if __name__ == "__main__":
    asyncio.run(main()) 