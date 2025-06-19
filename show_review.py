#!/usr/bin/env python3
import sys
import asyncio
sys.path.insert(0, 'src')

from chains.universal_rag_lcel import create_universal_rag_chain

async def main():
    print('🎰 GENERATING COMPLETE BETWAY CASINO REVIEW...')
    print('=' * 80)
    
    chain = create_universal_rag_chain()
    query = 'Review Betway Casino focusing on games and user experience'
    response = await chain.ainvoke({'query': query})
    
    # Handle different response types
    if hasattr(response, 'content'):
        content = response.content
    elif isinstance(response, str):
        content = response  
    else:
        content = str(response)
    
    print(content)
    print('=' * 80)
    print(f'📊 FINAL STATS:')
    print(f'   📄 Length: {len(content):,} characters')
    print(f'   🖼️  Images: {"YES - EMBEDDED!" if "<img" in content else "NO"}')
    print(f'   🎨 HTML: {"YES - FORMATTED!" if "<div" in content else "NO"}')
    print(f'   📱 Responsive: {"YES" if "max-width" in content else "NO"}')
    print(f'   ⚡ Lazy Load: {"YES" if "loading=" in content else "NO"}')

if __name__ == "__main__":
    asyncio.run(main()) 