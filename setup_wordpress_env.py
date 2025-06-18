#!/usr/bin/env python3
"""
🔧 WordPress Environment Setup for Universal RAG CMS v6.0
Configures WordPress integration environment variables
"""

import os

def setup_wordpress_env():
    """Setup WordPress environment variables"""
    
    print("🔧 WordPress Integration Setup")
    print("=" * 50)
    print("This will configure WordPress publishing for Universal RAG CMS v6.0")
    print()
    
    # Check current environment
    current_url = os.getenv("WORDPRESS_SITE_URL", "")
    current_user = os.getenv("WORDPRESS_USERNAME", "")
    current_pass = os.getenv("WORDPRESS_APP_PASSWORD", "")
    
    print("📋 Current Configuration:")
    print(f"  WORDPRESS_SITE_URL: {'✅ Set' if current_url else '❌ Not set'}")
    print(f"  WORDPRESS_USERNAME: {'✅ Set' if current_user else '❌ Not set'}")
    print(f"  WORDPRESS_APP_PASSWORD: {'✅ Set' if current_pass else '❌ Not set'}")
    print()
    
    if current_url and current_user and current_pass:
        print("✅ WordPress already configured!")
        print("🚀 WordPress publishing should work in your next chain execution")
        return True
    
    print("🔧 WordPress Setup Required:")
    print()
    print("1. 📝 WordPress Application Password Setup:")
    print("   • Log into your WordPress Admin dashboard")
    print("   • Go to: Users → Your Profile")
    print("   • Scroll to: 'Application Passwords' section")
    print("   • Add New Application Password")
    print("   • Name it: 'Universal RAG CMS'")
    print("   • Copy the generated password (xxxx-xxxx-xxxx-xxxx format)")
    print()
    
    print("2. 🔐 Environment Variables Setup:")
    print("   Add these to your environment or .env file:")
    print()
    print("   export WORDPRESS_SITE_URL='https://your-site.com'")
    print("   export WORDPRESS_USERNAME='your_admin_username'") 
    print("   export WORDPRESS_APP_PASSWORD='xxxx-xxxx-xxxx-xxxx'")
    print()
    
    print("3. 🔄 Restart your application after setting variables")
    print()
    
    print("📋 Example Configuration:")
    print("=" * 30)
    print("WORDPRESS_SITE_URL='https://myblog.com'")
    print("WORDPRESS_USERNAME='admin'")
    print("WORDPRESS_APP_PASSWORD='AbCd-EfGh-IjKl-MnOp'")
    print()
    
    print("🚀 Once configured, WordPress publishing will:")
    print("  ✅ Automatically publish generated content as draft posts")
    print("  ✅ Include proper titles and formatting")
    print("  ✅ Handle images and media uploads")
    print("  ✅ Provide post URLs in response metadata")
    print()
    
    return False

def test_wordpress_connection():
    """Test WordPress connection with current environment"""
    
    print("🔍 Testing WordPress Connection...")
    print("=" * 40)
    
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from integrations.wordpress_publisher import WordPressConfig, WordPressIntegration
        
        # Try to create config
        config = WordPressConfig()
        print("✅ WordPress configuration created successfully")
        
        # Try to create service
        service = WordPressIntegration(config=config)
        print("✅ WordPress service initialized successfully")
        
        print("🚀 WordPress integration is ready!")
        return True
        
    except Exception as e:
        print(f"❌ WordPress connection failed: {e}")
        print()
        print("💡 This is expected if environment variables aren't set")
        return False

if __name__ == "__main__":
    print("🎰 Universal RAG CMS v6.0 - WordPress Setup")
    print("=" * 60)
    print()
    
    # Setup environment
    configured = setup_wordpress_env()
    
    if configured:
        # Test connection if already configured
        test_wordpress_connection()
    
    print("📚 For more info, see WordPress integration documentation")
    print("🔧 After setup, re-run your Betway chain to test publishing!") 