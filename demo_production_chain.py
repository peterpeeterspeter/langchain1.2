#!/usr/bin/env python3
"""
🚀 PRODUCTION UNIVERSAL RAG CMS DEMO WITH WORDPRESS PUBLISHING
Uses the actual production chain with proper WordPress integration
"""

import sys
import os
import asyncio
import logging
sys.path.insert(0, 'src')

# Suppress logging for cleaner output
logging.getLogger().setLevel(logging.ERROR)

from chains.universal_rag_lcel import create_universal_rag_chain
from integrations.wordpress_publisher import WordPressIntegration, WordPressConfig

async def main():
    print('🚀 PRODUCTION UNIVERSAL RAG CMS WITH WORDPRESS PUBLISHING')
    print('=' * 70)
    
    try:
        # WordPress configuration from our memory
        wordpress_config = WordPressConfig(
            site_url="https://www.crashcasino.io",
            username="nmlwh", 
            application_password="your-wordpress-password-here"
        )
        
        print('📝 Creating Production Universal RAG Chain...')
        rag_chain = create_universal_rag_chain()
        
        print('📰 Creating WordPress Integration...')
        wordpress = WordPressIntegration(wordpress_config)
        
        # Query for casino review
        query = 'Review Betway Casino focusing on games and user experience with detailed analysis'
        
        print(f'🎯 Query: {query}')
        print('⏳ Generating comprehensive casino review...')
        
        # Generate content with full RAG chain
        response = await rag_chain.ainvoke({'query': query})
        
        # Extract content
        if hasattr(response, 'content'):
            content = response.content
            confidence = getattr(response, 'confidence_score', 0.0)
        elif hasattr(response, 'answer'):
            content = response.answer
            confidence = 0.0
        else:
            content = str(response)
            confidence = 0.0
            
        print('✅ Content Generated!')
        print(f'📄 Length: {len(content):,} characters')
        print(f'🖼️  Images: {"YES" if "<img" in content else "NO"}')
        print(f'🔍 Confidence: {confidence:.3f}')
        
        # Publish to WordPress
        print('\n📰 Publishing to WordPress...')
        
        result = await wordpress.publish_rag_content(
            query=query,
            rag_response=content,
            title='Betway Casino Review: Complete 2025 Analysis with Images',
            featured_image_query='Betway casino games interface'
        )
        
        if result.get('id'):
            post_id = result.get('id')
            post_url = result.get('link', f"https://www.crashcasino.io/?p={post_id}")
            
            print('🎉 WORDPRESS PUBLISHING SUCCESS!')
            print(f'📝 Post ID: {post_id}')
            print(f'🔗 WordPress URL: {post_url}')
            print(f'⏱️  Processing time: {result.get("processing_time", "N/A")}s')
            
            # Show content preview
            print('\n📖 CONTENT PREVIEW:')
            print('=' * 50)
            preview = content[:1000] + '...' if len(content) > 1000 else content
            print(preview)
            
        else:
            print('❌ WordPress publishing failed:')
            print(f'Error: {result.get("error", "Unknown error")}')
            
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 