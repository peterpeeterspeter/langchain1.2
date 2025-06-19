#!/usr/bin/env python3
import sys
import asyncio
import logging
sys.path.insert(0, 'src')

# Suppress logging
logging.getLogger().setLevel(logging.CRITICAL)

from chains.universal_rag_lcel import create_universal_rag_chain

async def main():
    chain = create_universal_rag_chain()
    response = await chain.ainvoke({'query': 'Review Betway Casino focusing on games and user experience'})
    
    # Extract content from response
    if hasattr(response, 'answer'):
        content = response.answer
    elif hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)
    
    print(content)

if __name__ == "__main__":
    asyncio.run(main()) 