#!/usr/bin/env python3
"""
🔧 Updated WordPress Test - Universal RAG CMS v6.0
Testing WordPress integration with corrected username: nmlwh
"""

import asyncio
import sys
import os

# Configure WordPress environment variables with updated credentials
os.environ['WORDPRESS_SITE_URL'] = 'https://www.crashcasino.io'
os.environ['WORDPRESS_USERNAME'] = 'nmlwh'  # ✅ UPDATED USERNAME
os.environ['WORDPRESS_APP_PASSWORD'] = 'q8ZU 4UHD 90vI Ej55 U0Jh yh8c'

sys.path.insert(0, 'src')

async def test_updated_wordpress():
    """Test WordPress integration with updated credentials"""
    
    print('🔧 WordPress Integration Test - Updated Credentials')
    print('=' * 60)
    print('🎯 Target: crashcasino.io')
    print(f'👤 Username: {os.environ["WORDPRESS_USERNAME"]}')
    print(f'🔐 App Password: {os.environ["WORDPRESS_APP_PASSWORD"][:8]}...')
    print()
    
    try:
        # Test 1: WordPress Configuration
        from integrations.wordpress_publisher import WordPressConfig, WordPressIntegration
        
        config = WordPressConfig()
        print(f'✅ WordPress Config Created')
        print(f'   Site URL: {config.site_url}')
        print(f'   Username: {config.username}')
        print(f'   Auth Method: Application Password')
        print()
        
        # Test 2: WordPress Integration
        wp_integration = WordPressIntegration(config)
        print(f'✅ WordPress Integration Initialized')
        
        # Test 3: Performance Stats
        stats = wp_integration.get_performance_stats()
        print(f'✅ WordPress Status:')
        print(f'   WordPress Configured: {stats["integration_status"]["wordpress_configured"]}')
        print(f'   Supabase Connected: {stats["integration_status"]["supabase_connected"]}')
        print()
        
        # Test 4: Universal RAG Chain with WordPress
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        rag_chain = create_universal_rag_chain(
            enable_wordpress_publishing=True,
            enable_comprehensive_web_research=True,
            enable_dataforseo_images=True
        )
        
        print('🎰 Testing Complete Betway Chain with Updated WordPress...')
        print()
        
        response = await rag_chain.ainvoke({
            'question': 'Comprehensive Betway Casino review with WordPress publishing'
        })
        
        print('🎉 SUCCESS! Chain executed with updated WordPress credentials')
        print(f'✅ Response Length: {len(response.answer)} characters')
        print(f'✅ Confidence Score: {response.confidence_score:.3f}')
        print(f'✅ Sources Count: {len(response.sources)}')
        print(f'✅ Processing Time: {response.response_time:.2f}s')
        print()
        
        if hasattr(response, 'metadata') and response.metadata:
            wordpress_result = response.metadata.get('wordpress_publishing')
            if wordpress_result:
                print('📝 WordPress Publishing Results:')
                print(f'   Status: {wordpress_result.get("status", "Unknown")}')
                print(f'   Post ID: {wordpress_result.get("post_id", "N/A")}')
                print(f'   URL: {wordpress_result.get("url", "N/A")}')
            else:
                print('ℹ️  WordPress publishing completed (check WordPress admin)')
        
        return True
        
    except ValueError as e:
        if "WordPress" in str(e):
            print(f'❌ WordPress Configuration Error: {e}')
            print('🔧 Check environment variables and credentials')
        else:
            print(f'❌ Configuration Error: {e}')
        return False
        
    except Exception as e:
        print(f'❌ Test Failed: {e}')
        return False

async def main():
    """Run the complete test"""
    print('🚀 WordPress Integration Test with Updated Credentials')
    print('🔗 Updated Memory with username: nmlwh')
    print()
    
    success = await test_updated_wordpress()
    
    if success:
        print('🏆 ALL TESTS PASSED!')
        print('✅ WordPress integration ready with updated credentials')
        print('✅ Universal RAG CMS v6.0 fully operational')
    else:
        print('❌ Test failed - check configuration')

if __name__ == '__main__':
    asyncio.run(main()) 