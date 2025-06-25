#!/usr/bin/env python3
"""
🎰 DIRECT WORDPRESS TEST
Direct test of WordPress publishing functionality without the chain
"""

import os
import sys
from pathlib import Path

# SET ENVIRONMENT VARIABLES FIRST
print("🔧 Setting WordPress environment variables...")
os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "ai_publisher"
os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
os.environ["WORDPRESS_APP_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")

print(f"✅ WORDPRESS_SITE_URL = {os.environ.get('WORDPRESS_SITE_URL', 'NOT SET')}")
print(f"✅ WORDPRESS_USERNAME = {os.environ.get('WORDPRESS_USERNAME', 'NOT SET')}")
print(f"✅ WORDPRESS_PASSWORD = {'SET' if os.environ.get('WORDPRESS_PASSWORD') else 'NOT SET'}")
print(f"✅ WORDPRESS_APP_PASSWORD = {'SET' if os.environ.get('WORDPRESS_APP_PASSWORD') else 'NOT SET'}")

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def direct_wordpress_test():
    """Direct test of WordPress publishing"""
    
    print("🎰 DIRECT WORDPRESS TEST")
    print("=" * 50)
    
    try:
        # Import WordPress components
        from integrations.wordpress_publisher import WordPressConfig, WordPressRESTPublisher
        print("✅ Successfully imported WordPress components")
        
        # Create config with explicit values
        config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="ai_publisher",
            application_password=os.environ.get("WORDPRESS_APP_PASSWORD", "")
        )
        print(f"✅ WordPressConfig created successfully!")
        print(f"   Site URL: {config.site_url}")
        print(f"   Username: {config.username}")
        print(f"   App Password: {'SET' if config.application_password else 'NOT SET'}")
        
        # Create publisher
        async with WordPressRESTPublisher(config) as publisher:
            print("✅ WordPress publisher created and connected")
            
            # Test basic authentication
            auth_result = await publisher._make_wp_request("GET", "/wp-json/wp/v2/users/me")
            if auth_result:
                print(f"✅ Authentication successful! User: {auth_result.get('name', 'Unknown')}")
            else:
                print("❌ Authentication failed!")
                return
            
            # Create a simple Ladbrokes review post
            print("\n🎰 Publishing Ladbrokes review...")
            
            ladbrokes_content = """
            <h1>Ladbrokes Casino Review</h1>
            <p>Ladbrokes is one of the UK's most established gambling operators, offering a comprehensive casino experience.</p>
            
            <h2>Key Features</h2>
            <ul>
                <li>Licensed by the UK Gambling Commission</li>
                <li>Wide selection of slots and table games</li>
                <li>Live casino section</li>
                <li>Mobile-optimized platform</li>
            </ul>
            
            <h2>Our Verdict</h2>
            <p>Ladbrokes provides a solid casino experience with good game variety and trusted licensing.</p>
            """
            
            # Publish as MT listing
            try:
                result = await publisher.publish_post(
                    title="Ladbrokes Casino Review - Comprehensive Analysis",
                    content=ladbrokes_content,
                    status="publish",
                    meta_description="Complete review of Ladbrokes Casino covering games, bonuses, licensing and more.",
                    custom_fields={
                        "post_type": "mt_listing",
                        "casino_name": "Ladbrokes",
                        "overall_rating": "8.5",
                        "_wp_post_type": "mt_listing"
                    }
                )
                
                print("\n📊 PUBLISHING RESULTS:")
                if result.get('success'):
                    print(f"✅ Published successfully!")
                    print(f"   🆔 Post ID: {result.get('post_id')}")
                    print(f"   🔗 URL: {result.get('post_url')}")
                    print(f"   📂 Post Type: {result.get('post_type', 'post')}")
                    
                    # Verify it's live
                    import requests
                    try:
                        response = requests.get(result['post_url'], timeout=10)
                        if response.status_code == 200:
                            content = response.text.lower()
                            has_ladbrokes = 'ladbrokes' in content
                            print(f"✅ Live verification:")
                            print(f"   🏷️ Contains Ladbrokes: {'✅' if has_ladbrokes else '❌'}")
                        else:
                            print(f"⚠️ URL returned status: {response.status_code}")
                    except Exception as e:
                        print(f"⚠️ Could not verify URL: {e}")
                        
                else:
                    print(f"❌ Publishing failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as publish_error:
                print(f"❌ Publishing error: {publish_error}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(direct_wordpress_test()) 