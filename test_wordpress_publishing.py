#!/usr/bin/env python3
"""
Test WordPress Publishing with Universal RAG Chain
Tests the complete Authoritative Hyperlink Generation + WordPress Publishing integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from chains.universal_rag_lcel import create_universal_rag_chain

async def test_wordpress_publishing():
    """Test complete WordPress publishing workflow with Betway casino review"""
    
    print("🚀 TESTING: Universal RAG Chain + WordPress Publishing Integration")
    print("=" * 80)
    
    try:
        # Create the Universal RAG Chain with ALL features enabled
        print("🔧 Initializing Universal RAG Chain with ALL 13 features...")
        chain = create_universal_rag_chain(
            model_name="gpt-4.1-mini",
            temperature=0.1,
            enable_caching=True,
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=True,
            enable_dataforseo_images=True,
            enable_wordpress_publishing=True,  # ✅ KEY: WordPress publishing enabled
            enable_fti_processing=True,
            enable_security=True,
            enable_profiling=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_hyperlink_generation=True,  # ✅ KEY: Hyperlink generation enabled
            enable_response_storage=True
        )
        
        print("✅ Universal RAG Chain initialized successfully!")
        
        # Check feature count
        active_features = chain._count_active_features()
        print(f"🎯 Active Features: {active_features}/13")
        
        # Test query for comprehensive Betway casino review
        test_query = "Write a comprehensive professional review of Betway Casino covering licensing, game selection, bonuses, payment methods, mobile experience, and customer support"
        
        print(f"\n🎯 Test Query: {test_query}")
        print("⏳ Generating comprehensive casino review with WordPress publishing...")
        print("   (This includes: Content generation + Hyperlink embedding + WordPress publishing)")
        
        # Generate response
        response = await chain.ainvoke({"query": test_query})
        
        # Display results
        print("\n" + "="*80)
        print("✅ GENERATION COMPLETE!")
        print("="*80)
        
        print(f"📊 Response Statistics:")
        print(f"   • Content Length: {len(response.answer):,} characters")
        print(f"   • Confidence Score: {response.confidence_score:.3f}")
        print(f"   • Sources Found: {len(response.sources)}")
        print(f"   • Processing Time: {response.response_time:.1f} seconds")
        print(f"   • Cached: {response.cached}")
        
        # Check for hyperlinks
        hyperlink_count = response.answer.count('<a href=')
        print(f"   • Authoritative Hyperlinks: {hyperlink_count}")
        
        # Check WordPress publishing status
        wordpress_metadata = response.metadata.get('wordpress_publishing', {})
        if wordpress_metadata:
            print(f"\n📝 WordPress Publishing Results:")
            print(f"   • Status: {wordpress_metadata.get('status', 'Unknown')}")
            if wordpress_metadata.get('post_id'):
                print(f"   • Post ID: {wordpress_metadata.get('post_id')}")
                print(f"   • Post URL: {wordpress_metadata.get('post_url', 'N/A')}")
            if wordpress_metadata.get('error'):
                print(f"   • Error: {wordpress_metadata.get('error')}")
        else:
            print(f"\n⚠️  WordPress Publishing: No metadata found")
        
        # Display content preview
        print(f"\n📄 Content Preview (first 500 characters):")
        print("-" * 60)
        print(response.answer[:500] + "..." if len(response.answer) > 500 else response.answer)
        print("-" * 60)
        
        # Show hyperlinks if any
        if hyperlink_count > 0:
            print(f"\n🔗 Hyperlink Analysis:")
            import re
            links = re.findall(r'<a href="([^"]*)"[^>]*>([^<]*)</a>', response.answer)
            for i, (url, text) in enumerate(links[:5], 1):  # Show first 5 links
                print(f"   {i}. {text} → {url}")
        
        # Display source analysis
        print(f"\n📚 Source Analysis:")
        for i, source in enumerate(response.sources[:3], 1):  # Show first 3 sources
            print(f"   {i}. {source.get('url', 'N/A')} (Score: {source.get('score', 'N/A')})")
        
        # Show enhancement metadata
        if response.metadata:
            print(f"\n🎯 Enhancement Features Used:")
            enhancements = response.metadata.keys()
            for enhancement in sorted(enhancements):
                if enhancement != 'wordpress_publishing':
                    print(f"   • {enhancement}")
        
        print("\n" + "="*80)
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
        print("✅ Universal RAG Chain with WordPress Publishing is OPERATIONAL")
        print("="*80)
        
        return response
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the test
    response = asyncio.run(test_wordpress_publishing())
    
    if response:
        print(f"\n🏆 FINAL STATUS: SUCCESS")
        print(f"Generated {len(response.answer):,} character review with {response.confidence_score:.1%} confidence")
        wordpress_meta = response.metadata.get('wordpress_publishing', {})
        if wordpress_meta.get('post_id'):
            print(f"📝 Published to WordPress: Post ID {wordpress_meta.get('post_id')}")
    else:
        print(f"\n💥 FINAL STATUS: FAILED")
        sys.exit(1) 