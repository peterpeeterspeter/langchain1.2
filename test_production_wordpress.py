#!/usr/bin/env python3
import sys
import asyncio
import os
sys.path.insert(0, 'src')

# Set WordPress credentials from memory
os.environ['WORDPRESS_URL'] = "https://www.crashcasino.io"
os.environ['WORDPRESS_USERNAME'] = "nmlwh" 
os.environ['WORDPRESS_PASSWORD'] = "q8ZU 4UHD 90vI Ej55 U0Jh yh8c"

from chains.universal_rag_lcel import create_universal_rag_chain

async def main():
    print('🚀 PRODUCTION UNIVERSAL RAG CHAIN - NAPOLEON CASINO WITH WORDPRESS PUBLISHING')
    print('=' * 80)
    
    # Create production chain with all 12 features
    chain = create_universal_rag_chain()
    
    # Run with WordPress publishing requested
    result = await chain.ainvoke({
        'query': 'Napoleon casino review 2025',
        'publish_to_wordpress': True
    })
    
    print(f'📊 Features Active: {chain._count_active_features()}/12')
    print(f'🔍 Confidence: {result.confidence_score:.3f}')
    print(f'📄 Content Length: {len(result.answer):,} chars')
    print(f'🖼️  Images: {"YES" if "<img" in result.answer else "NO"}')
    
    # Check for WordPress publishing in metadata
    wp_published = result.metadata.get('wordpress_published', False)
    wp_post_id = result.metadata.get('wordpress_post_id')
    wp_url = result.metadata.get('wordpress_url')
    
    print(f'📝 WordPress: {"PUBLISHED" if wp_published else "NOT PUBLISHED"}')
    if wp_post_id:
        print(f'🔗 WordPress URL: {wp_url or f"https://www.crashcasino.io/?p={wp_post_id}"}')
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main()) 