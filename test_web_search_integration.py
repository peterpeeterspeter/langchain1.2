#!/usr/bin/env python3
"""
Test script to verify web search integration in Universal RAG Chain
Tests both Tavily web search and overall integration
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables only")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src to Python path
sys.path.append('src')

async def test_web_search_integration():
    """Test web search integration in Universal RAG Chain"""
    
    print("="*70)
    print("🌐 TESTING WEB SEARCH INTEGRATION")
    print("="*70)
    
    # Test 1: Check Tavily availability
    print("\n1. 🔍 Checking Tavily Search availability...")
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily_available = True
        print("   ✅ Tavily search module imported successfully")
    except ImportError as e:
        tavily_available = False
        print(f"   ❌ Tavily search not available: {e}")
        return
    
    # Test 2: Check API key
    print("\n2. 🔑 Checking Tavily API key...")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        print("   ✅ TAVILY_API_KEY found in environment")
        print(f"   🔑 Key starts with: {tavily_api_key[:8]}...")
    else:
        print("   ❌ TAVILY_API_KEY not found in environment")
        print("   💡 Set TAVILY_API_KEY for web search functionality")
    
    # Test 3: Direct Tavily test (if key available)
    if tavily_api_key:
        print("\n3. 🧪 Testing direct Tavily search...")
        try:
            search_tool = TavilySearchResults(
                max_results=3,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True
            )
            
            results = search_tool.invoke({"query": "Betway casino review 2024"})
            print(f"   ✅ Direct Tavily search successful: {len(results)} results")
            
            for i, result in enumerate(results[:2], 1):
                print(f"      {i}. {result.get('title', 'No title')[:60]}...")
                
        except Exception as e:
            print(f"   ❌ Direct Tavily search failed: {e}")
    
    # Test 4: Universal RAG Chain integration
    print("\n4. 🚀 Testing Universal RAG Chain with web search...")
    
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain with web search enabled
        chain = create_universal_rag_chain(
            model_name="gpt-4o-mini",
            enable_web_search=True,
            enable_contextual_retrieval=False,  # Disable to focus on web search
            enable_dataforseo_images=False,     # Disable to focus on web search
            enable_wordpress_publishing=False,  # Disable for testing
            enable_security=False,              # Disable for testing
            enable_profiling=False              # Disable for testing
        )
        
        print("   ✅ Universal RAG Chain created with web search enabled")
        
        # Check if web search was initialized
        if hasattr(chain, 'web_search_tool') and chain.web_search_tool:
            print("   ✅ Web search tool properly initialized")
        else:
            print("   ⚠️ Web search tool not initialized (check API key)")
        
        # Test the web search step directly
        if chain.web_search_tool:
            print("\n5. 🔍 Testing web search step in isolation...")
            test_inputs = {"question": "Betway casino bonuses 2024"}
            
            try:
                web_results = await chain._gather_web_search_results(test_inputs)
                print(f"   ✅ Web search step successful: {len(web_results)} results found")
                
                for i, result in enumerate(web_results[:3], 1):
                    print(f"      {i}. {result.get('title', 'No title')[:50]}...")
                    print(f"         URL: {result.get('url', 'No URL')[:60]}...")
                    print(f"         Source: {result.get('source', 'No source')}")
                    
            except Exception as e:
                print(f"   ❌ Web search step failed: {e}")
        
        # Test full chain (if everything works)
        if chain.web_search_tool and tavily_api_key:
            print("\n6. 🎯 Testing full Universal RAG Chain with web search...")
            try:
                start_time = datetime.now()
                
                response = await chain.ainvoke({
                    "question": "What are the latest Betway casino promotions?"
                })
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                print(f"   ✅ Full chain successful in {duration:.2f}s")
                print(f"   📝 Content length: {len(response.answer)} characters")
                print(f"   📊 Sources found: {len(response.sources)}")
                print(f"   🎯 Confidence: {response.confidence_score:.2f}")
                
                # Check for web search sources
                web_sources = [s for s in response.sources if s.get('source_type') == 'web_search']
                print(f"   🌐 Web search sources: {len(web_sources)}")
                
                if web_sources:
                    print("   ✅ WEB SEARCH INTEGRATION SUCCESSFUL!")
                    for i, source in enumerate(web_sources[:2], 1):
                        print(f"      Web {i}: {source.get('metadata', {}).get('title', 'No title')[:50]}...")
                else:
                    print("   ⚠️ No web search sources found in response")
                
            except Exception as e:
                print(f"   ❌ Full chain test failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"   ❌ Chain creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🏁 WEB SEARCH INTEGRATION TEST COMPLETE")
    print("="*70)
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   • Tavily module available: {'✅ Yes' if tavily_available else '❌ No'}")
    print(f"   • API key configured: {'✅ Yes' if tavily_api_key else '❌ No'}")
    print("   • Integration status: Run test to see results")
    
    if not tavily_api_key:
        print("\n💡 To enable web search:")
        print("   1. Get a Tavily API key from https://tavily.com")
        print("   2. Set environment variable: export TAVILY_API_KEY='your_key_here'")
        print("   3. Re-run this test")

if __name__ == "__main__":
    asyncio.run(test_web_search_integration()) 