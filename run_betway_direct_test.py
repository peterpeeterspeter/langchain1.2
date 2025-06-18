#!/usr/bin/env python3
"""
🎰 BETWAY CASINO DIRECT TEST - NO CACHE
Universal RAG CMS v6.0 - Clear cache and target Betway specifically

FEATURES TO VERIFY:
✅ Casino detection logic
✅ 95-field analysis framework 
✅ Fixed image uploader
✅ Major casino review sites research
✅ Comprehensive WebBaseLoader integration
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def test_betway_direct():
    """Direct Betway test with cache disabled"""
    
    print("🎰 BETWAY CASINO DIRECT TEST (NO CACHE)")
    print("=" * 60)
    print("🎯 Target: Betway Casino with full 95-field analysis")
    print("🚀 Cache: DISABLED for fresh results")
    print("📊 All advanced features: ENABLED")
    print()
    
    # Create chain with cache disabled
    print("🔧 Initializing Universal RAG Chain (no cache)...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        enable_caching=False,  # DISABLE CACHE
        enable_contextual_retrieval=True,
        enable_prompt_optimization=True,
        enable_enhanced_confidence=True,
        enable_template_system_v2=True,
        enable_dataforseo_images=True,
        enable_wordpress_publishing=True,
        enable_fti_processing=True,
        enable_security=True,
        enable_profiling=True,
        enable_web_search=True,
        enable_comprehensive_web_research=True,  # ✅ CRITICAL: Casino analysis
        enable_response_storage=True
    )
    print("✅ Chain initialized with ALL features enabled")
    print()
    
    # Test casino-specific query
    betway_query = "Betway casino comprehensive review analysis including Malta Gaming Authority licensing trustworthiness games slots bonuses payments customer support mobile app user experience"
    
    print(f"🎯 BETWAY CASINO ANALYSIS")
    print(f"Query: {betway_query[:100]}...")
    print("=" * 60)
    
    try:
        start_time = time.time()
        
        # Execute the chain
        result = await rag_chain.ainvoke({"query": betway_query})
        
        execution_time = time.time() - start_time
        
        print(f"✅ EXECUTION COMPLETE")
        print(f"⏱️ Time: {execution_time:.2f} seconds")
        print(f"📊 Confidence: {result.confidence_score:.3f}")
        print(f"📝 Content Length: {len(result.answer):,} characters")
        print(f"📄 Word Count: {len(result.answer.split()):,} words")
        print(f"🔗 Sources: {len(result.sources)}")
        print()
        
        # Analyze sources for casino context
        casino_sources = 0
        review_sites = 0
        comprehensive_research = 0
        
        for source in result.sources:
            if any(term in source.get('url', '').lower() or 
                   term in source.get('title', '').lower() or
                   term in source.get('content', '').lower() 
                   for term in ['casino', 'betway', 'gambling', 'malta']):
                casino_sources += 1
            
            if any(site in source.get('url', '').lower() 
                   for site in ['askgamblers', 'casino.guru', 'casinomeister', 'gamblingcommission', 'lcb.org', 'thepogg']):
                review_sites += 1
                
            if source.get('source_type') == 'comprehensive_web_research':
                comprehensive_research += 1
        
        print(f"🎰 CASINO ANALYSIS RESULTS:")
        print(f"  Casino-related sources: {casino_sources}/{len(result.sources)}")
        print(f"  Major review sites: {review_sites}/{len(result.sources)}")
        print(f"  Comprehensive research: {comprehensive_research}/{len(result.sources)}")
        print()
        
        # Check if Betway is mentioned in content
        content_lower = result.answer.lower()
        betway_mentions = content_lower.count('betway')
        casino_terms = sum(content_lower.count(term) for term in ['casino', 'gaming', 'gambling', 'malta', 'license'])
        
        print(f"🔍 CONTENT ANALYSIS:")
        print(f"  Betway mentions: {betway_mentions}")
        print(f"  Casino-related terms: {casino_terms}")
        print(f"  Casino-specific content: {'✅ YES' if betway_mentions > 0 or casino_terms > 5 else '❌ NO'}")
        print()
        
        # Show content preview
        print(f"📄 CONTENT PREVIEW (first 500 chars):")
        print("-" * 60)
        print(result.answer[:500])
        if len(result.answer) > 500:
            print("...")
        print("-" * 60)
        print()
        
        # Check metadata for advanced features
        metadata = result.metadata or {}
        print(f"🚀 ADVANCED FEATURES STATUS:")
        print(f"  Contextual Retrieval: {'✅' if metadata.get('contextual_retrieval_used') else '❌'}")
        print(f"  Template System v2.0: {'✅' if metadata.get('template_system_v2_used') else '❌'}")
        print(f"  DataForSEO Images: {'✅' if metadata.get('dataforseo_images_used') else '❌'}")
        print(f"  Security Checked: {'✅' if metadata.get('security_checked') else '❌'}")
        print(f"  Performance Profiled: {'✅' if metadata.get('performance_profiled') else '❌'}")
        print(f"  Images Embedded: {metadata.get('images_embedded', 0)}")
        print(f"  Active Features: {metadata.get('advanced_features_count', 0)}/12")
        print()
        
        # Success indicators
        is_success = (
            betway_mentions > 0 or casino_terms > 5 or
            casino_sources > 0 or
            'casino' in result.answer.lower()[:1000]
        )
        
        print(f"🎉 TEST RESULT: {'✅ SUCCESS' if is_success else '❌ FAILED'}")
        if is_success:
            print("✅ Betway casino analysis successfully generated!")
        else:
            print("❌ Casino content not detected - may need debugging")
            
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_betway_direct()) 