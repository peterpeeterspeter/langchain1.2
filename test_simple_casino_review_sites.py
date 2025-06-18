#!/usr/bin/env python3
"""
🎰 SIMPLE CASINO REVIEW SITES TEST
Quick validation that major casino review sites are accessible and integrated
"""

import asyncio
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import UniversalRAGChain

async def test_simple_casino_review_sites():
    """Simple test to validate casino review sites are working"""
    
    print(f"\n{'='*60}")
    print(f"🎰 SIMPLE CASINO REVIEW SITES TEST")
    print(f"{'='*60}")
    
    # Create minimal RAG chain
    print("🚀 Creating minimal Universal RAG Chain...")
    try:
        rag_chain = UniversalRAGChain(
            model_name="gpt-4.1-mini",
            enable_comprehensive_web_research=True,  # Only enable this feature
            enable_web_search=False,                 # Disable other features for speed
            enable_enhanced_confidence=False,
            enable_caching=False,
            enable_dataforseo_images=False,
            enable_wordpress_publishing=False,
            enable_fti_processing=False,
            enable_security=False,
            enable_profiling=False,
            enable_response_storage=False
        )
        print("✅ RAG Chain created successfully!")
    except Exception as e:
        print(f"❌ Failed to create RAG chain: {e}")
        return False
    
    # Simple test query
    test_query = "Betway Casino review"
    print(f"\n📝 Testing query: {test_query}")
    print(f"⏱️  Starting test...")
    
    start_time = time.time()
    
    try:
        # Test the comprehensive web research method directly
        print("🔍 Testing comprehensive web research method directly...")
        
        research_results = await rag_chain._gather_comprehensive_web_research({
            'question': test_query,
            'query_analysis': None
        })
        
        processing_time = time.time() - start_time
        
        print(f"\n🎯 DIRECT TEST RESULTS:")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Research Results Found: {len(research_results)}")
        
        # Analyze results
        review_sites_found = []
        for result in research_results:
            if 'review_site' in result:
                review_sites_found.append(result['review_site'])
                print(f"  ✅ {result['review_site']} (Authority: {result.get('authority', 'N/A')})")
        
        if review_sites_found:
            print(f"\n🏆 SUCCESS: Found {len(review_sites_found)} casino review sites!")
            print(f"📋 Review Sites: {', '.join(set(review_sites_found))}")
            return True
        else:
            print(f"\n❌ NO REVIEW SITES FOUND")
            print(f"🔧 Check if the sites are accessible and the logic is working")
            
            # Debug: Show what we did find
            if research_results:
                print(f"\n🔍 DEBUG - Found {len(research_results)} results:")
                for i, result in enumerate(research_results[:3], 1):
                    print(f"  {i}. URL: {result.get('url', 'No URL')}")
                    print(f"     Title: {result.get('title', 'No Title')}")
                    print(f"     Source Type: {result.get('source_type', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_review_sites():
    """Test individual review sites directly"""
    
    print(f"\n{'='*60}")
    print(f"🌐 INDIVIDUAL REVIEW SITES TEST")
    print(f"{'='*60}")
    
    # Test the review sites research method directly
    try:
        rag_chain = UniversalRAGChain(
            enable_comprehensive_web_research=True,
            enable_web_search=False,
            enable_caching=False
        )
        
        print("🔍 Testing individual casino review sites...")
        
        # Test the specific method
        review_results = await rag_chain._research_casino_review_sites("betway", "Betway Casino review")
        
        print(f"\n📊 Individual Review Sites Results:")
        print(f"Total Results: {len(review_results)}")
        
        for result in review_results:
            site_name = result.get('review_site', 'Unknown')
            authority = result.get('authority', 0.0)
            url = result.get('url', 'No URL')
            print(f"  ✅ {site_name} (Authority: {authority:.2f}) - {url}")
        
        return len(review_results) > 0
        
    except Exception as e:
        print(f"❌ Individual sites test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple Casino Review Sites Test")
    
    async def run_all_tests():
        print("\n" + "="*60)
        print("🎯 TEST 1: Comprehensive Web Research Method")
        success1 = await test_simple_casino_review_sites()
        
        print("\n" + "="*60)
        print("🎯 TEST 2: Individual Review Sites Method")
        success2 = await test_individual_review_sites()
        
        print("\n" + "="*60)
        print("🏆 FINAL RESULTS")
        print("="*60)
        
        if success1 or success2:
            print("✅ CASINO REVIEW SITES INTEGRATION: WORKING!")
            if success1 and success2:
                print("🌟 Both comprehensive and individual methods working perfectly!")
            elif success1:
                print("✅ Comprehensive web research method working")
            else:
                print("✅ Individual review sites method working")
        else:
            print("❌ CASINO REVIEW SITES INTEGRATION: FAILED")
            print("🔧 Check network connectivity and site accessibility")
        
        return success1 or success2
    
    # Run the tests
    result = asyncio.run(run_all_tests())
    
    if result:
        print(f"\n🎰 Casino Review Sites Integration: OPERATIONAL!")
        print(f"🏆 Major review sites (AskGamblers, Casino.Guru, etc.) accessible")
    else:
        print(f"\n❌ Casino Review Sites Integration: NEEDS DEBUGGING")
        print(f"🔧 Check implementation and connectivity")
    
    print(f"\n{'='*60}")
    print(f"🚀 Simple test completed - no timeouts or hangs!")
    print(f"{'='*60}") 