#!/usr/bin/env python3
"""
Test WordPress publishing with proper environment variables
"""

import os
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set WordPress environment variables
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_PASSWORD"] = "q8ZU 4UHD 90vI Ej55 U0Jh yh8c"

async def test_wordpress_publishing():
    """Test the complete WordPress publishing pipeline"""
    
    print("🎰 TESTING WORDPRESS PUBLISHING WITH ENVIRONMENT VARIABLES")
    print("=" * 70)
    print(f"🌐 WordPress URL: {os.environ['WORDPRESS_URL']}")
    print(f"👤 WordPress User: {os.environ['WORDPRESS_USERNAME']}")
    print(f"🔐 WordPress Pass: {os.environ['WORDPRESS_PASSWORD'][:10]}...")
    print()
    
    try:
        # Import the Universal RAG Chain
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        print("✅ Successfully imported Universal RAG Chain")
        
        # Create chain with WordPress enabled
        rag_chain = create_universal_rag_chain(
            enable_comprehensive_web_research=True,
            enable_wordpress_publishing=True,
            enable_cache_bypass=False,
            enable_performance_tracking=True
        )
        
        print("✅ Universal RAG Chain initialized")
        
        # Test query with WordPress publishing flag
        test_query = "Write a concise Betway Casino review focusing on licensing and games"
        
        query_input = {
            "question": test_query,
            "publish_to_wordpress": True
        }
        
        print(f"🔍 Testing query: {test_query}")
        print("⚡ Executing with WordPress publishing enabled...")
        
        response = await rag_chain.ainvoke(query_input)
        
        print(f"✅ Response generated: {len(response.answer)} characters")
        print(f"🎯 Confidence: {response.confidence_score}")
        print(f"📚 Sources: {len(response.sources)}")
        
        # Check if WordPress publishing was successful
        wordpress_result = response.metadata.get('wordpress_published', False)
        if wordpress_result:
            print("🎉 ✅ WORDPRESS PUBLISHING SUCCESSFUL!")
            wp_post_id = response.metadata.get('wordpress_post_id')
            if wp_post_id:
                print(f"📝 WordPress Post ID: {wp_post_id}")
                print(f"🔗 URL: https://www.crashcasino.io/?p={wp_post_id}")
        else:
            print("❌ WordPress publishing failed")
            print("📋 Response metadata:", response.metadata)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_wordpress_publishing()) 