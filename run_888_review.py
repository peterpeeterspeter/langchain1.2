#!/usr/bin/env python3
"""
Run 888 Casino Review with Native LangChain Hub LCEL
==================================================

Test the Universal RAG LCEL chain with native hub integration for 888 Casino review.
"""

import asyncio
import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent / "src"))

from chains.universal_rag_lcel import create_universal_rag_chain

async def run_888_review():
    """Run 888 Casino review with native LangChain Hub integration"""
    
    print("🎰 Running 888 Casino Review with Native LangChain Hub LCEL")
    print("=" * 60)
    
    # Create the Universal RAG Chain with all features
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_template_system_v2=True,  # ✅ Native hub integration
        enable_caching=True,
        enable_web_search=True,
        enable_comprehensive_web_research=True,
        enable_hyperlink_generation=True
    )
    
    # 888 Casino review query
    query = "Review 888 Casino - detailed analysis of bonuses, games, licensing, and overall experience"
    
    print(f"🎯 Query: {query}")
    print("-" * 60)
    
    try:
        # Run the LCEL chain
        response = await rag_chain.ainvoke({
            "question": query
        })
        
        print("✅ 888 Casino Review Generated Successfully!")
        print("=" * 60)
        print(f"📊 Confidence Score: {response.confidence_score:.1%}")
        print(f"🔗 Sources: {len(response.sources)}")
        print(f"💰 Cached: {response.cached}")
        print(f"⏱️ Response Time: {response.response_time:.2f}s")
        print("=" * 60)
        
        # Display the full review
        print("📝 888 CASINO REVIEW:")
        print(response.answer)
        
        print("\n" + "=" * 60)
        print("🔗 SOURCES:")
        for i, source in enumerate(response.sources[:5], 1):
            print(f"{i}. {source.get('title', 'Unknown')} - {source.get('url', 'No URL')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_888_review()) 