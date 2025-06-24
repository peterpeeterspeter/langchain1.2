#!/usr/bin/env python3
"""
Enhanced Bitcasino review with MT Casino publishing
Uses your superior Universal RAG CMS + MT Casino integration
"""

import asyncio
import sys
import os
from datetime import datetime
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain
from integrations.wordpress_publisher import WordPressIntegration

async def main():
    """Enhanced Bitcasino publishing with MT Casino support"""
    
    print("🎰 ENHANCED BITCASINO → MT CASINO PUBLISHING")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Create your superior Universal RAG Chain (all 12+ features)
        print("🚀 Initializing Universal RAG CMS v6.1.0...")
        rag_chain = create_universal_rag_chain(
            enable_wordpress_publishing=False,  # We'll handle publishing separately
            enable_comprehensive_web_research=True,  # Your 95-field analysis
            enable_dataforseo_images=True,          # Your bulletproof image system
            enable_enhanced_confidence=True,        # Your enhanced confidence scoring
            enable_template_system_v2=True,         # Your template system v2.0
            enable_hyperlink_generation=True,       # Authoritative hyperlinks
            enable_web_search=True                   # Tavily web search
        )
        
        # Enhanced query for comprehensive analysis
        bitcasino_query = """Write a comprehensive professional Bitcasino review covering their Bitcoin casino platform, game selection, bonuses, security, licensing, payment methods, and user experience. Include rating and detailed analysis."""
        
        print("⚡ Executing Universal RAG Chain (all 12+ features)...")
        print("📊 Features active: Contextual Retrieval, Enhanced Confidence, Template System v2.0, DataForSEO Images")
        
        # Generate comprehensive content
        response = await rag_chain.ainvoke({'query': bitcasino_query})
        
        print(f"\n✅ CONTENT GENERATION SUCCESS!")
        print(f"📝 Content Generated: {len(response.answer):,} characters")
        print(f"🔍 Confidence Score: {response.confidence_score:.3f}")
        print(f"📊 Sources: {len(response.sources)}")
        print(f"⏱️ Processing Time: {response.response_time:.2f}s")
        
        # Check for image uploads
        if hasattr(response, 'images_uploaded') or any('image' in str(source) for source in response.sources):
            print("🖼️ Images successfully processed and uploaded")
        
        # Enhanced WordPress publishing with MT Casino support
        print("\n🎨 PUBLISHING AS MT CASINO...")
        
        wordpress = WordPressIntegration()
        
        # Extract structured casino data from your 95-field analysis
        structured_data = None
        if hasattr(response, 'casino_intelligence'):
            structured_data = response.casino_intelligence
        elif hasattr(response, 'metadata') and response.metadata.get('casino_intelligence'):
            structured_data = response.metadata.get('casino_intelligence')
        
        # Check if MT Casino publishing is available
        async with wordpress.publisher:
            if hasattr(wordpress.publisher, 'publish_mt_casino'):
                print("🎰 Using MT Casino publishing method...")
                wordpress_result = await wordpress.publisher.publish_mt_casino(
                    title="Bitcasino Review 2025: Bitcoin Casino Analysis",
                    content=response.answer,
                    structured_casino_data=structured_data,
                    status="publish"
                )
            else:
                print("🔄 Using casino intelligence publishing method...")
                wordpress_result = await wordpress.publish_casino_intelligence_content(
                    query=bitcasino_query,
                    rag_response=response.answer,
                    structured_casino_data=structured_data,
                    title="Bitcasino Review 2025: Bitcoin Casino Analysis"
                )
        
        # Results analysis
        if wordpress_result and wordpress_result.get('success'):
            print(f"\n🎉 SUCCESS! MT CASINO PUBLISHED!")
            print(f"📝 Post ID: {wordpress_result.get('post_id')}")
            print(f"🔗 Live URL: {wordpress_result.get('link')}")
            print(f"📊 Post Type: {wordpress_result.get('post_type', 'post')}")
            if wordpress_result.get('edit_link'):
                print(f"✏️ Edit: {wordpress_result.get('edit_link')}")
            
            # Check if fallback was used
            if wordpress_result.get('fallback'):
                print("⚠️ Note: Used fallback to regular post with MT Casino styling")
                if wordpress_result.get('original_error'):
                    print(f"   Original error: {wordpress_result.get('original_error')}")
        else:
            print(f"❌ WordPress publishing failed")
            if wordpress_result:
                print(f"   Error: {wordpress_result.get('error', 'Unknown error')}")
        
        # Performance summary
        total_time = time.time() - start_time
        print(f"\n⏱️ TOTAL PROCESSING TIME: {total_time:.2f}s")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "query": bitcasino_query,
            "response": response.answer,
            "confidence_score": response.confidence_score,
            "sources": response.sources,
            "processing_time": response.response_time,
            "total_time": total_time,
            "wordpress_result": wordpress_result,
            "casino_intelligence": structured_data,
            "timestamp": timestamp,
            "features_used": {
                "contextual_retrieval": True,
                "enhanced_confidence": True,
                "template_system_v2": True,
                "dataforseo_images": True,
                "comprehensive_web_research": True,
                "mt_casino_publishing": hasattr(wordpress.publisher, 'publish_mt_casino') if 'wordpress' in locals() else False
            }
        }
        
        filename = f"bitcasino_mt_casino_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n💾 COMPREHENSIVE RESULTS SAVED: {filename}")
        print("🌐 Live WordPress: https://www.crashcasino.io")
        
        # Performance insights
        print(f"\n📈 PERFORMANCE INSIGHTS:")
        print(f"   • Content generation: {response.response_time:.2f}s")
        print(f"   • WordPress publishing: {(total_time - response.response_time):.2f}s")
        print(f"   • Characters per second: {len(response.answer) / response.response_time:.0f}")
        
        return results
        
    except Exception as e:
        print(f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = asyncio.run(main()) 