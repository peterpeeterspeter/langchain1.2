#!/usr/bin/env python3
"""
🧪 TEST: Auto-Initialization Demo for Universal RAG Chain
Tests the new auto-initialization features that fix the sources=[] issue
"""

import asyncio
import os
from datetime import datetime
from src.chains.universal_rag_lcel import UniversalRAGChain

async def test_auto_initialization():
    print("🧪 Testing Universal RAG Chain Auto-Initialization")
    print("=" * 60)
    
    # Check environment
    supabase_url = os.getenv("SUPABASE_URL") 
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    print(f"🔍 Environment Check:")
    print(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Missing'}")
    print(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Missing'}")
    print()
    
    # Test auto-initialization (NO vector_store or supabase_client passed)
    print("🚀 Creating UniversalRAGChain with AUTO-INITIALIZATION...")
    
    chain = UniversalRAGChain(
        model_name="gpt-4o",
        enable_caching=True,
        enable_prompt_optimization=True,
        enable_enhanced_confidence=True
        # ✅ NOTE: NO vector_store or supabase_client parameters!
    )
    
    print(f"   Chain created: ✅")
    print(f"   Supabase client: {'✅ Auto-initialized' if chain.supabase_client else '❌ Still None'}")
    print(f"   Vector store: {'✅ Auto-initialized' if chain.vector_store else '❌ Still None'}")
    print()
    
    # Test query
    test_query = "What are the best features of Betway Casino?"
    print(f"🎯 Testing Query: '{test_query}'")
    print("=" * 60)
    
    start_time = datetime.now()
    try:
        response = await chain.ainvoke({'query': test_query})
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        # Results
        print(f"📊 RESULTS:")
        print(f"   Sources Found: {len(response.sources)} ({'✅ SUCCESS' if response.sources else '❌ STILL EMPTY'})")
        print(f"   Response Time: {response_time:.1f}ms ({'✅ FAST' if response_time < 5000 else '❌ SLOW'})")
        print(f"   Confidence: {response.confidence_score:.1%}")
        print(f"   Cached: {response.cached}")
        print()
        
        if response.sources:
            print(f"🎯 Source Details:")
            for i, source in enumerate(response.sources[:3], 1):
                print(f"   {i}. {source.get('title', 'Untitled')}")
                print(f"      Authority: {source.get('authority_score', 'N/A')}")
                print(f"      URL: {source.get('url', 'N/A')}")
        
        print(f"\n📝 Answer Preview:")
        print(f"   {response.answer[:200]}{'...' if len(response.answer) > 200 else ''}")
        
        # Final verdict
        success = len(response.sources) > 0 and response_time < 10000
        print(f"\n🏆 OVERALL RESULT: {'✅ SUCCESS - Auto-initialization FIXED the issue!' if success else '❌ STILL BROKEN'}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"🏆 OVERALL RESULT: ❌ FAILED - Auto-initialization did not work")

if __name__ == "__main__":
    asyncio.run(test_auto_initialization()) 