#!/usr/bin/env python3
"""
🔧 Configure WordPress from Memory - Universal RAG CMS v6.0
Sets up WordPress environment using stored credentials for crashcasino.io
"""

import os
import sys

def configure_wordpress_environment():
    """Configure WordPress environment variables from memory"""
    
    print("🔧 WordPress Configuration from Memory")
    print("=" * 50)
    print("🎯 Target: crashcasino.io WordPress integration")
    print()
    
    # WordPress credentials from memory
    wordpress_config = {
        "WORDPRESS_SITE_URL": "https://www.crashcasino.io",
        "WORDPRESS_USERNAME": "admin", 
        "WORDPRESS_APP_PASSWORD": "generated_app_password_from_wp_admin"
    }
    
    print("📋 WordPress Configuration (From Memory):")
    for key, value in wordpress_config.items():
        print(f"  {key}: {value}")
        # Set environment variable for current session
        os.environ[key] = value
    
    print()
    print("✅ Environment variables set for current session")
    print()
    
    # Test WordPress configuration
    print("🧪 Testing WordPress Configuration...")
    test_wordpress_connection()

def test_wordpress_connection():
    """Test WordPress connection with configured credentials"""
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from integrations.wordpress_publisher import WordPressIntegration
        
        print("🔄 Initializing WordPress integration...")
        
        # Initialize WordPress integration
        wp_integration = WordPressIntegration()
        
        if wp_integration.is_configured():
            print("✅ WordPress integration successfully configured!")
            print(f"📝 Site URL: {os.getenv('WORDPRESS_SITE_URL')}")
            print(f"👤 Username: {os.getenv('WORDPRESS_USERNAME')}")
            print("🔑 App Password: [CONFIGURED]")
        else:
            print("❌ WordPress integration not properly configured")
            
    except Exception as e:
        print(f"⚠️ WordPress test failed: {str(e)}")
        print("💡 Note: This is expected if WordPress credentials need to be generated")

def show_wordpress_setup_instructions():
    """Show instructions for completing WordPress setup"""
    
    print()
    print("📋 WordPress Setup Instructions:")
    print("=" * 40)
    print("1. Log into https://www.crashcasino.io/wp-admin")
    print("2. Go to Users → Your Profile")
    print("3. Scroll to 'Application Passwords' section")
    print("4. Create new application password:")
    print("   - Name: 'Universal RAG CMS v6.0'")
    print("   - Click 'Add New Application Password'")
    print("5. Copy the generated password (format: xxxx-xxxx-xxxx-xxxx)")
    print("6. Replace 'generated_app_password_from_wp_admin' with actual password")
    print()
    print("🚀 Once configured, the WordPress integration will:")
    print("   - Automatically publish generated casino reviews")
    print("   - Upload and embed images from DataForSEO")
    print("   - Apply SEO optimization and formatting")
    print("   - Set posts as drafts for review before publishing")

if __name__ == "__main__":
    configure_wordpress_environment()
    show_wordpress_setup_instructions() 