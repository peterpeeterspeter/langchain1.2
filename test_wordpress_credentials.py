#!/usr/bin/env python3
"""
Test WordPress Credentials Loading
"""

import os
import sys
sys.path.append('src')

def test_wordpress_credentials():
    """Test that WordPress credentials are loading correctly"""
    
    print("🔍 TESTING: WordPress Credentials Loading")
    print("=" * 50)
    
    # Test environment variables
    print("📋 Environment Variables:")
    site_url = os.getenv("WORDPRESS_SITE_URL", "")
    username = os.getenv("WORDPRESS_USERNAME", "")
    app_password = os.getenv("WORDPRESS_APP_PASSWORD", "")
    
    print(f"✅ WORDPRESS_SITE_URL: {site_url}")
    print(f"✅ WORDPRESS_USERNAME: {username}")
    print(f"✅ WORDPRESS_APP_PASSWORD: {'*' * len(app_password[:8])}...{app_password[-4:] if len(app_password) > 4 else '*'}")
    
    # Test WordPress integration
    print(f"\n🔧 Testing WordPress Integration Import:")
    try:
        from integrations.wordpress_publisher import WordPressConfig, WordPressIntegration
        print("✅ WordPress imports successful")
        
        # Test configuration
        wp_config = WordPressConfig(
            site_url=site_url,
            username=username,
            application_password=app_password
        )
        print(f"✅ WordPress config created")
        print(f"  📄 Site URL: {wp_config.site_url}")
        print(f"  👤 Username: {wp_config.username}")
        print(f"  🔑 Password: {'*' * 10}")
        
        # Test service
        wp_service = WordPressIntegration(wordpress_config=wp_config)
        print(f"✅ WordPress service created")
        
        # Test if all credentials are present
        missing = []
        if not site_url:
            missing.append("WORDPRESS_SITE_URL")
        if not username:
            missing.append("WORDPRESS_USERNAME")
        if not app_password:
            missing.append("WORDPRESS_APP_PASSWORD")
            
        if missing:
            print(f"❌ Missing credentials: {', '.join(missing)}")
            return False
        else:
            print(f"✅ All WordPress credentials present")
            return True
            
    except Exception as e:
        print(f"❌ WordPress integration import failed: {e}")
        return False

if __name__ == "__main__":
    test_wordpress_credentials() 