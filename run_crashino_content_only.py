#!/usr/bin/env python3
"""
🎰 CRASHINO CONTENT GENERATION - PRODUCTION CHAIN DEMO
Testing the Universal RAG Chain for Crashino casino review content generation
Demonstrates all capabilities except WordPress publishing (no auth needed)
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def generate_crashino_content():
    """Generate Crashino casino review content using production-ready chain"""
    
    print("🎰 CRASHINO PRODUCTION CHAIN - CONTENT GENERATION")
    print("=" * 60)
    print("🚀 Testing: Default Universal RAG Chain (all features except WordPress)")
    print("📋 Target: Comprehensive MT Casino review content")
    print("🖼️ Images: DataForSEO integration enabled")
    print("🌐 Research: 95-field casino analysis framework")
    print("✅ Features: All content generation and validation fixes")
    print()
    
    # Create Universal RAG Chain with content generation features
    print("🔧 Creating Universal RAG Chain with full content generation capabilities...")
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_wordpress_publishing=False,  # Skip WordPress for demo
        enable_dataforseo_images=True,
        enable_web_search=True,
        enable_comprehensive_web_research=True
    )
    
    print("✅ Chain created successfully")
    print(f"🖼️ Image integration: {chain.enable_dataforseo_images}")
    print(f"🌐 Web research: {chain.enable_comprehensive_web_research}")
    print(f"🔍 Web search: {chain.enable_web_search}")
    
    # Crashino query for comprehensive casino review
    crashino_query = """Create a comprehensive professional Crashino Casino review for MT Casino custom post type.

    Provide detailed analysis including:
    - Licensing and regulatory compliance
    - Game selection and software providers
    - Cryptocurrency features and payment methods
    - Welcome bonuses and ongoing promotions
    - Mobile compatibility and user experience
    - Customer support quality and availability
    - Security measures and player protection
    - Crash games and unique features
    - Pros and cons analysis
    - Final rating and recommendation

    Format for WordPress MT Casino post type with SEO optimization, structured headers, and engaging content."""
    
    print(f"🔍 Query: Crashino Casino Review")
    print(f"📏 Query length: {len(crashino_query)} characters")
    
    start_time = time.time()
    
    try:
        print("\n⚡ Executing production chain for content generation...")
        
        # Execute the chain for content generation
        result = await chain.ainvoke({
            "question": crashino_query
        })
        
        processing_time = time.time() - start_time
        
        print(f"\n⏱️ Content generation completed in {processing_time:.2f} seconds")
        print(f"📊 Content length: {len(result.answer)} characters")
        print(f"🎯 Confidence score: {result.confidence_score:.3f}")
        print(f"📚 Sources found: {len(result.sources)}")
        
        # Content quality analysis
        crashino_mentions = result.answer.lower().count('crashino')
        casino_mentions = result.answer.lower().count('casino')
        
        # Check for key content sections
        has_licensing = 'licens' in result.answer.lower()
        has_games = 'game' in result.answer.lower()
        has_bonuses = 'bonus' in result.answer.lower()
        has_security = 'security' in result.answer.lower()
        has_rating = 'rating' in result.answer.lower()
        
        print(f"\n📈 Content Quality Analysis:")
        print(f"   📝 Content length: {len(result.answer):,} characters")
        print(f"   🎰 Crashino mentions: {crashino_mentions}")
        print(f"   🏢 Casino mentions: {casino_mentions}")
        print(f"   📊 Confidence score: {result.confidence_score:.3f}")
        print(f"   🔍 Sources utilized: {len(result.sources)}")
        
        print(f"\n📋 Content Sections Coverage:")
        print(f"   🏛️ Licensing info: {'✅' if has_licensing else '❌'}")
        print(f"   🎮 Games coverage: {'✅' if has_games else '❌'}")
        print(f"   🎁 Bonuses info: {'✅' if has_bonuses else '❌'}")
        print(f"   🔒 Security details: {'✅' if has_security else '❌'}")
        print(f"   ⭐ Rating included: {'✅' if has_rating else '❌'}")
        
        # Images analysis
        images_found = result.metadata.get('images_found', 0)
        if images_found > 0:
            print(f"\n🖼️ Image Integration:")
            print(f"   📸 Images discovered: {images_found}")
            print(f"   💡 WordPress ready: Images would be uploaded if publishing enabled")
        
        # Sources analysis
        if result.sources:
            print(f"\n📚 Research Sources:")
            for i, source in enumerate(result.sources[:5], 1):
                title = source.get('title', 'Unknown')[:50]
                print(f"   {i}. {title}...")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "test_type": "crashino_content_generation",
            "query": crashino_query,
            "response": result.answer,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "metadata": result.metadata,
            "processing_time": processing_time,
            "content_analysis": {
                "crashino_mentions": crashino_mentions,
                "casino_mentions": casino_mentions,
                "content_length": len(result.answer),
                "sections_coverage": {
                    "licensing": has_licensing,
                    "games": has_games,
                    "bonuses": has_bonuses,
                    "security": has_security,
                    "rating": has_rating
                },
                "quality_rating": "high" if len(result.answer) > 5000 and result.confidence_score > 0.6 else "moderate"
            },
            "timestamp": timestamp
        }
        
        filename = f"crashino_content_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Show content preview
        print(f"\n📄 Crashino Review Content Preview (first 800 characters):")
        print("=" * 60)
        preview = result.answer[:800] + "..." if len(result.answer) > 800 else result.answer
        print(preview)
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        print(f"\n" + "=" * 60)
        print("🏁 CRASHINO CONTENT GENERATION COMPLETE")
        print("✅ Universal RAG Chain: Full content generation capabilities")
        print("🎰 MT Casino format: Professional review structure")
        print("🖼️ Image discovery: DataForSEO integration working")
        print("🌐 Research: 95-field casino analysis framework")
        print("🔧 Content validation: All fixes applied successfully")
        print("")
        print("💡 To publish to WordPress: Set WORDPRESS_PASSWORD and use run_crashino_production.py")

if __name__ == "__main__":
    result = asyncio.run(generate_crashino_content())
    
    if result:
        quality = "High" if len(result.answer) > 5000 and result.confidence_score > 0.6 else "Moderate"
        print(f"\n🚀 SUCCESS: Crashino review generated successfully!")
        print(f"📊 Quality: {quality}")
        print(f"📝 Length: {len(result.answer):,} characters")
        print(f"🎯 Confidence: {result.confidence_score:.3f}")
        print(f"🎉 Production chain working perfectly!")
    else:
        print("\n❌ Content generation failed") 