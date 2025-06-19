#!/usr/bin/env python3
"""
🔥 REAL PRODUCTION UNIVERSAL RAG CMS 🔥
BULLETPROOF - USES ONLY THE REAL CHAIN WITH ALL 11 FEATURES

CHANGE KEYWORD BELOW AND RUN - THAT'S IT!
NO DEMO SCRIPTS, NO CONFUSION, JUST THE REAL SYSTEM!
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# IMPORT THE REAL UNIVERSAL RAG CHAIN - NOT DEMO PIPELINE!
from src.chains.universal_rag_lcel import create_universal_rag_chain

# =============================================
# 🎯 CHANGE THIS KEYWORD TO WHATEVER YOU WANT:
# =============================================
KEYWORD = "Napoleon casino review 2025"
# =============================================

async def run_real_production():
    """
    🚀 Run the REAL Universal RAG Chain with ALL 11 features
    This is NOT a demo - this is the REAL production system!
    """
    print("🔥 REAL PRODUCTION UNIVERSAL RAG CMS")
    print("=" * 80)
    print(f"🎯 Query: {KEYWORD}")
    print("🔄 Initializing REAL Universal RAG Chain...")
    print("-" * 80)
    
    # Create the REAL chain - NOT demo pipeline!
    chain = create_universal_rag_chain()
    
    # Verify this is the REAL chain with ALL features
    feature_count = chain._count_active_features()
    print(f"✅ REAL CHAIN LOADED: {feature_count} FEATURES ACTIVE")
    
    if feature_count != 12:
        print(f"❌ ERROR: Expected 12 features, got {feature_count}")
        print("❌ This means some features failed to initialize!")
        return
    
    print("🚀 Running REAL Universal RAG Chain...")
    start_time = datetime.now()
    
    # Run the REAL chain
    result = await chain.ainvoke({'query': KEYWORD})
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the complete article
    article_filename = f"REAL_RAG_article_{timestamp}.md"
    with open(article_filename, 'w', encoding='utf-8') as f:
        f.write(f"# REAL Universal RAG CMS - {KEYWORD}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds\n")
        f.write(f"Features Used: {feature_count}/12\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(result.answer)
    
    # Save technical details
    tech_filename = f"REAL_RAG_technical_{timestamp}.json"
    tech_data = {
        "query": KEYWORD,
        "confidence_score": result.confidence_score,
        "processing_time_seconds": processing_time,
        "features_active": feature_count,
        "sources_count": len(result.sources),
        "cached": result.cached,
        "token_usage": result.token_usage,
        "metadata": result.metadata,
        "timestamp": datetime.now().isoformat(),
        "system": "REAL_UNIVERSAL_RAG_CHAIN"
    }
    
    with open(tech_filename, 'w', encoding='utf-8') as f:
        json.dump(tech_data, f, indent=2, default=str)
    
    # Display results
    print("\n" + "=" * 80)
    print("🎉 REAL UNIVERSAL RAG CHAIN - EXECUTION COMPLETE!")
    print("=" * 80)
    print(f"📊 Confidence Score: {result.confidence_score:.3f}")
    print(f"📚 Sources Found: {len(result.sources)}")
    print(f"⚡ Processing Time: {processing_time:.2f} seconds")
    print(f"🔥 Features Active: {feature_count}/12")
    print(f"💾 Cached Result: {result.cached}")
    print(f"🎯 Token Usage: {result.token_usage}")
    print("=" * 80)
    print(f"📄 Article saved: {article_filename}")
    print(f"🔧 Technical data: {tech_filename}")
    print("=" * 80)
    
    # Show preview of content
    preview = result.answer[:500] + "..." if len(result.answer) > 500 else result.answer
    print("\n📖 CONTENT PREVIEW:")
    print("-" * 50)
    print(preview)
    print("-" * 50)
    
    # Show feature breakdown
    if 'advanced_features_count' in result.metadata:
        print(f"\n🔥 METADATA FEATURES: {result.metadata['advanced_features_count']}")
    
    if 'confidence_breakdown' in result.metadata:
        breakdown = result.metadata['confidence_breakdown']
        print("\n📊 CONFIDENCE BREAKDOWN:")
        for factor, score in breakdown.items():
            print(f"  {factor}: {score:.3f}")
    
    print("\n✅ REAL PRODUCTION SYSTEM COMPLETE!")

if __name__ == "__main__":
    print("🔥 BULLETPROOF REAL UNIVERSAL RAG CMS")
    print("🎯 Uses ONLY the REAL chain with ALL 12 features")
    print("🚫 NO demo scripts, NO confusion!")
    print()
    
    try:
        asyncio.run(run_real_production())
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("❌ Check that you're in the right directory and all imports work") 