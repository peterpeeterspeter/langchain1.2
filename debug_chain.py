#!/usr/bin/env python3
"""
Debug Universal RAG Chain - Metadata Testing
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

async def test_metadata_capture():
    """Test WordPress and hyperlink metadata capture"""
    
    print("🔍 DEBUG: Universal RAG Chain Metadata Testing")
    print("=" * 60)
    
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        print("✅ Imports successful")
        
        # Test with all features enabled and minimal web research
        print("\n🔧 Testing metadata-enabled chain...")
        start_time = time.time()
        
        chain = create_universal_rag_chain(
            enable_wordpress_publishing=True,
            enable_hyperlink_generation=True,
            enable_comprehensive_web_research=False  # Disable to speed up
        )
        
        chain_time = time.time() - start_time
        print(f"✅ Chain created in {chain_time:.2f}s")
        
        # Simple query to test metadata
        query = "Write a brief Betway Casino review focusing on safety and licensing"
        print(f"\n🎯 Query: {query}")
        print("⏳ Processing (checking metadata capture)...")
        
        invoke_start = time.time()
        result = await chain.ainvoke({"query": query})
        invoke_time = time.time() - invoke_start
        
        print(f"\n✅ COMPLETED in {invoke_time:.2f}s")
        print(f"📄 Content: {len(result.answer):,} characters")
        print(f"🎯 Confidence: {result.confidence_score:.3f}")
        print(f"📚 Sources: {len(result.sources)}")
        
        # ✅ NEW: Test metadata capture
        print("\n🔍 METADATA ANALYSIS:")
        print(f"📋 Total metadata keys: {len(result.metadata)}")
        
        # Check WordPress metadata
        wp_metadata = result.metadata.get('wordpress_publishing', {})
        if wp_metadata:
            print(f"✅ WordPress metadata found: {wp_metadata}")
        else:
            print("❌ No WordPress metadata found")
        
        # Check hyperlink metadata  
        hyperlink_metadata = result.metadata.get('hyperlink_generation', {})
        if hyperlink_metadata:
            print(f"✅ Hyperlink metadata found: {hyperlink_metadata}")
            links_added = hyperlink_metadata.get('links_added', 0)
            print(f"🔗 Links added: {links_added}")
        else:
            print("❌ No hyperlink metadata found")
        
        # Check content for hyperlinks
        content_hyperlinks = result.answer.count('<a href')
        print(f"🔗 HTML links in content: {content_hyperlinks}")
        
        print("\n📋 All metadata keys:")
        for key, value in result.metadata.items():
            if isinstance(value, dict):
                print(f"  - {key}: {len(value)} subkeys")
            else:
                print(f"  - {key}: {type(value).__name__}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_metadata_capture()) 