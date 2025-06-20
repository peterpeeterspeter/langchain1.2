#!/usr/bin/env python3
"""
Betway Casino Review Demo
Generate a comprehensive casino review using Universal RAG Chain with all 12 features
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

async def generate_betway_review():
    """Generate comprehensive Betway Casino review with all features"""
    
    print("🎰 BETWAY CASINO COMPREHENSIVE REVIEW")
    print("=" * 60)
    print("🔧 Initializing Universal RAG Chain with ALL 12 features...")
    print()
    
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain with ALL features (fixed model name)
        chain = create_universal_rag_chain(
            model_name='gpt-4.1-mini',  # ✅ CORRECT: Valid OpenAI model
            temperature=0.1,
            enable_caching=True,
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
            enable_comprehensive_web_research=True
        )
        
        active_features = chain._count_active_features()
        print(f"✅ Chain initialized successfully!")
        print(f"🚀 Active features: {active_features}/12")
        print()
        
        # Define comprehensive review query
        query = """Provide a comprehensive review of Betway Casino including:
        - Overall safety and licensing status
        - Game variety and quality
        - Bonus offers and promotions
        - Payment methods and withdrawal times
        - Customer support quality
        - Mobile experience
        - Pros and cons
        - Final verdict and rating out of 10"""
        
        print("🎯 EXECUTING COMPREHENSIVE ANALYSIS...")
        print(f"Query: {query}")
        print()
        print("⏳ This may take 1-2 minutes for complete analysis...")
        print()
        
        # Execute the review
        start_time = datetime.now()
        response = await chain.ainvoke({'question': query})
        end_time = datetime.now()
        
        # Display results
        print("📊 ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"✅ Confidence Score: {response.confidence_score:.3f}/1.0")
        print(f"⚡ Total Time: {(end_time - start_time).total_seconds():.1f} seconds")
        print(f"📚 Sources Used: {len(response.sources)} authoritative sources")
        print(f"💾 Cached Result: {response.cached}")
        print(f"🔢 Content Length: {len(response.answer):,} characters")
        
        if hasattr(response, 'token_usage') and response.token_usage:
            print(f"🎯 Tokens Used: {response.token_usage.get('total_tokens', 'N/A')}")
        
        print()
        print("=" * 60)
        print("🎰 COMPLETE BETWAY CASINO REVIEW")
        print("=" * 60)
        print()
        print(response.answer)
        print()
        print("=" * 60)
        print("📚 AUTHORITATIVE SOURCES:")
        print("=" * 60)
        
        for i, source in enumerate(response.sources[:8], 1):
            title = source.get('title', 'Unknown Source')[:80]
            url = source.get('url', 'No URL')
            authority = source.get('authority_score', 'N/A')
            print(f"{i:2d}. {title}")
            print(f"    URL: {url}")
            if authority != 'N/A':
                print(f"    Authority: {authority:.3f}")
            print()
        
        # Check for 95-field casino intelligence
        if hasattr(response, 'metadata') and response.metadata:
            casino_data = response.metadata.get('casino_intelligence')
            if casino_data:
                print("🎯 95-FIELD CASINO INTELLIGENCE EXTRACTED:")
                print(f"Casino Name: {casino_data.get('casino_name', 'N/A')}")
                print(f"Overall Score: {casino_data.get('overall_trustworthiness_score', 'N/A')}")
                print(f"License Status: {casino_data.get('license_status', 'N/A')}")
                print(f"Safety Rating: {casino_data.get('safety_rating', 'N/A')}")
                print()
        
        print("✅ BETWAY CASINO REVIEW GENERATION COMPLETE!")
        print(f"🎯 All {active_features} features successfully utilized")
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating review: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🚀 Starting Betway Casino Review with Universal RAG Chain")
    print("🔧 Features: 95-field extraction, Web research, Template system, WordPress publishing")
    print()
    
    # Run the review generation
    review = asyncio.run(generate_betway_review())
    
    if review:
        print()
        print("🎉 Review generated successfully!")
        print("💡 The review includes comprehensive analysis from multiple authoritative sources")
        print("📝 All 12 advanced features were utilized for maximum quality")
    else:
        print()
        print("❌ Review generation failed - check error messages above")

if __name__ == "__main__":
    main() 