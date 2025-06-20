#!/usr/bin/env python3

import sys
sys.path.append('.')
import asyncio
from src.chains.universal_rag_lcel import create_universal_rag_chain

async def test_rag_response_storage():
    print('🔍 Testing RAG Response Storage...')
    
    # Create chain with storage enabled, web search disabled to isolate the test
    chain = create_universal_rag_chain(
        enable_web_search=False, 
        enable_comprehensive_web_research=False,
        enable_response_storage=True
    )
    
    print(f'✅ Response Storage Enabled: {chain.enable_response_storage}')
    print(f'✅ Supabase Available: {chain.supabase_client is not None}')
    
    # Test with a simple query
    response = await chain.ainvoke({'query': 'What is a crash casino game?'})
    
    print(f'✅ Response confidence: {response.confidence_score}')
    print(f'✅ Storage threshold: 0.5 (stores if > 0.5)')
    print(f'✅ Should store response: {response.confidence_score > 0.5}')
    
    if response.confidence_score > 0.5:
        print('🎯 RAG Response should have been stored in Supabase vector store!')
    else:
        print('⚠️ Confidence too low, response was not stored')
    
    return response

if __name__ == '__main__':
    asyncio.run(test_rag_response_storage()) 