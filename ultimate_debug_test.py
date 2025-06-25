#!/usr/bin/env python3
"""
🎰 ULTIMATE DEBUG TEST - ENVIRONMENT VARIABLE TRACING
Debug script that sets environment variables BEFORE imports and traces configuration
"""

import os
import sys
from pathlib import Path

# SET ENVIRONMENT VARIABLES FIRST - BEFORE ANY IMPORTS
print("🔧 Setting WordPress environment variables...")
os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "ai_publisher"
os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
os.environ["WORDPRESS_APP_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")
os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
os.environ["SUPABASE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anhhbmJibiIsInJvbGUiOiJhbm9uIiwiaWF0IjoxNzM2ODU1ODM3LCJleHAiOjIwNTI0MzE4Mzd9.invalid_test_key"

print(f"✅ WORDPRESS_SITE_URL = {os.environ.get('WORDPRESS_SITE_URL', 'NOT SET')}")
print(f"✅ WORDPRESS_USERNAME = {os.environ.get('WORDPRESS_USERNAME', 'NOT SET')}")
print(f"✅ WORDPRESS_PASSWORD = {'SET' if os.environ.get('WORDPRESS_PASSWORD') else 'NOT SET'}")
print(f"✅ WORDPRESS_APP_PASSWORD = {'SET' if os.environ.get('WORDPRESS_APP_PASSWORD') else 'NOT SET'}")

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def ultimate_debug_test():
    """Ultimate test with proper environment setup and WordPress configuration debugging"""
    
    print("🎰 ULTIMATE DEBUG TEST - ENVIRONMENT VARIABLES + WORDPRESS CONFIG")
    print("=" * 80)
    
    # Test WordPress configuration creation
    print("\n🔧 Testing WordPress configuration...")
    try:
        from integrations.wordpress_publisher import WordPressConfig
        print("✅ Successfully imported WordPressConfig")
        
        # Try to create config
        try:
            config = WordPressConfig()
            print(f"✅ WordPressConfig created successfully!")
            print(f"   Site URL: {config.site_url}")
            print(f"   Username: {config.username}")
            print(f"   App Password: {'SET' if config.application_password else 'NOT SET'}")
        except Exception as config_error:
            print(f"❌ WordPressConfig creation failed: {config_error}")
            
            # Try manual config creation
            print("\n🔧 Trying manual config creation...")
            try:
                manual_config = WordPressConfig(
                    site_url="https://www.crashcasino.io",
                    username="ai_publisher", 
                    application_password=os.environ.get("WORDPRESS_APP_PASSWORD", "")
                )
                print(f"✅ Manual WordPressConfig created successfully!")
                config = manual_config
            except Exception as manual_error:
                print(f"❌ Manual config creation also failed: {manual_error}")
                return
            
    except Exception as import_error:
        print(f"❌ Failed to import WordPressConfig: {import_error}")
        return
    
    # Test chain initialization
    print("\n🚀 Testing Universal RAG Chain initialization...")
    try:
        from chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize chain
        chain = UniversalRAGChain(
            llm_model="gpt-4.1-mini",
            enable_wordpress_publishing=True,
            wordpress_config={
                "site_url": "https://www.crashcasino.io",
                "username": "ai_publisher",
                "application_password": os.environ.get("WORDPRESS_APP_PASSWORD", "")
            }
        )
        print("✅ Chain initialized successfully with WordPress config")
        
        # Test Ladbrokes review generation and publishing
        print("\n🎰 Generating and publishing Ladbrokes review...")
        
        # Create inputs with explicit publishing flag
        inputs = {
            "query": "Create a comprehensive Ladbrokes casino review",
            "publish_to_wordpress": True,
            "wordpress_post_type": "mt_listing",  # Force MT listing
            "enable_screenshot_capture": False    # Disable screenshots for now
        }
        
        print(f"📝 Running chain with inputs: {list(inputs.keys())}")
        
        result = await chain.ainvoke(inputs)
        
        print("\n📊 FINAL RESULTS:")
        print(f"📄 Content length: {len(result.get('content', ''))} characters")
        print(f"🎯 Confidence: {result.get('confidence_score', 'N/A')}")
        print(f"🏷️ Ladbrokes mentions: {result.get('content', '').count('Ladbrokes') + result.get('content', '').count('ladbrokes')}")
        
        wp_result = result.get('wordpress_result', {})
        print(f"\n🌐 WordPress Results:")
        print(f"   📝 Published: {'✅ SUCCESS' if wp_result.get('success') else '❌ FAILED'}")
        print(f"   🆔 Post ID: {wp_result.get('post_id', 'None')}")
        print(f"   🔗 URL: {wp_result.get('post_url', 'None')}")
        print(f"   📂 Post Type: {wp_result.get('post_type', 'unknown')}")
        if not wp_result.get('success'):
            print(f"   ❌ Error: {wp_result.get('error', 'Unknown error')}")
        
    except Exception as chain_error:
        print(f"❌ Chain initialization/execution failed: {chain_error}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(ultimate_debug_test()) 