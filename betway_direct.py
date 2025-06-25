#!/usr/bin/env python3
import asyncio
import sys
import os

# Configure WordPress from memory
os.environ['WORDPRESS_SITE_URL'] = 'https://www.crashcasino.io'
os.environ['WORDPRESS_USERNAME'] = 'admin'
os.environ['WORDPRESS_APP_PASSWORD'] = 'your-wordpress-password-here'

sys.path.insert(0, 'src')
from chains.universal_rag_lcel import create_universal_rag_chain

async def run_direct():
    print('🎰 BETWAY DIRECT CHAIN EXECUTION')
    print('=' * 50)
    
    rag_chain = create_universal_rag_chain(
        enable_comprehensive_web_research=True,
        enable_wordpress_publishing=True,
        enable_performance_tracking=True
    )
    
    response = await rag_chain.ainvoke({
        'question': 'Comprehensive Betway Casino review with games, bonuses, licensing, and user experience'
    })
    
    print(f'✅ Response Generated!')
    print(f'📊 Answer Length: {len(response.answer)} characters')
    print(f'🎯 Confidence: {response.confidence_score:.3f}')
    print(f'📚 Sources: {len(response.sources)}')
    print(f'⏱️ Time: {response.response_time:.2f}s')
    
    # Show content preview
    print(f'\n📄 CONTENT PREVIEW:')
    print(response.answer[:800] + '...')
    
    # Check WordPress publishing
    if hasattr(response, 'metadata') and response.metadata.get('wordpress_published'):
        print(f'\n✅ WORDPRESS: Published successfully!')
        print(f'🔗 URL: {response.metadata.get("wordpress_url", "Check site")}')
    else:
        print(f'\n⚠️ WORDPRESS: Check logs for publishing status')
    
    return response

if __name__ == "__main__":
    asyncio.run(run_direct()) 