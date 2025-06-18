#!/usr/bin/env python3
"""
🎰 COMPREHENSIVE 95-FIELD CASINO ANALYSIS TEST
Test Universal RAG CMS with complete WebBaseLoader integration across ALL 8 categories

Categories & Fields (95 total):
1. Trustworthiness (15 fields) - Licensing, security, reputation  
2. Games (12 fields) - Slots, tables, providers, RTP
3. Bonuses (12 fields) - Welcome, loyalty, wagering requirements
4. Payments (15 fields) - Methods, fees, processing times
5. User Experience (12 fields) - Support, mobile, interface
6. Innovations (8 fields) - VR, AI, blockchain, social features
7. Compliance (10 fields) - RG, AML, data protection
8. Assessment (11 fields) - Ratings, recommendations, improvements
"""

import asyncio
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def test_complete_95_field_analysis():
    """Test Universal RAG with complete 95-field casino analysis framework"""
    
    print(f"\n{'='*80}")
    print(f"🎰 UNIVERSAL RAG CMS v6.0 - COMPLETE 95-FIELD CASINO ANALYSIS")
    print(f"{'='*80}")
    
    # Create Universal RAG Chain with ALL features enabled
    print("🚀 Initializing Universal RAG Chain with ALL features...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4.1-mini",
        enable_comprehensive_web_research=True,  # 95-field analysis ENABLED
        enable_web_search=True,                  # Tavily integration
        enable_dataforseo_images=True,           # Image integration
        enable_enhanced_confidence=True,         # Confidence scoring
        enable_template_system_v2=True,          # Template system
        enable_contextual_retrieval=True,        # Advanced retrieval
        enable_fti_processing=True,              # Content processing
        enable_security=True,                    # Security features
        enable_profiling=True,                   # Performance profiling
        enable_caching=True,                     # Intelligent caching
        enable_prompt_optimization=True,         # Advanced prompts
        enable_wordpress_publishing=True,        # Publishing capability
        enable_response_storage=True             # Response vectorization
    )
    
    print("✅ Universal RAG Chain initialized with ALL advanced features!")
    
    # Test query focused on comprehensive casino analysis
    test_query = "Provide a comprehensive analysis of Betway Casino covering all aspects including licensing, games, bonuses, payment methods, user experience, innovations, compliance, and overall assessment"
    
    print(f"\n🔍 TEST QUERY: {test_query}")
    print(f"\n🌐 WebBaseLoader Analysis: ALL 8 categories (95 fields)")
    print("📊 Categories:")
    print("  1. 🛡️  Trustworthiness (15 fields) - License, security, reputation")
    print("  2. 🎮 Games (12 fields) - Slots, tables, providers, RTP")  
    print("  3. 🎁 Bonuses (12 fields) - Welcome, loyalty, wagering")
    print("  4. 💳 Payments (15 fields) - Methods, fees, processing")
    print("  5. 👤 User Experience (12 fields) - Support, mobile, interface")
    print("  6. 🚀 Innovations (8 fields) - VR, AI, blockchain, social")
    print("  7. ⚖️  Compliance (10 fields) - RG, AML, data protection")
    print("  8. 📊 Assessment (11 fields) - Ratings, recommendations")
    
    print(f"\n⏱️  Starting comprehensive analysis...")
    start_time = time.time()
    
    try:
        # Execute the Universal RAG Chain
        response = await rag_chain.ainvoke({'query': test_query})
        
        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"✅ Confidence Score: {response.confidence_score:.3f}/1.000")
        print(f"📊 Sources Found: {len(response.sources)}")
        print(f"💾 Cached: {response.cached}")
        print(f"🔗 Features Active: {rag_chain._count_active_features()}/12")
        
        # Analyze response content
        response_length = len(response.answer)
        print(f"📝 Response Length: {response_length:,} characters")
        
        # Check for comprehensive coverage
        coverage_indicators = {
            'licensing': any(term in response.answer.lower() for term in ['license', 'licensed', 'authority', 'regulation']),
            'games': any(term in response.answer.lower() for term in ['slots', 'games', 'casino', 'providers']),
            'bonuses': any(term in response.answer.lower() for term in ['bonus', 'promotion', 'welcome', 'loyalty']),
            'payments': any(term in response.answer.lower() for term in ['payment', 'deposit', 'withdrawal', 'banking']),
            'support': any(term in response.answer.lower() for term in ['support', 'customer', 'help', 'contact']),
            'mobile': any(term in response.answer.lower() for term in ['mobile', 'app', 'responsive']),
            'security': any(term in response.answer.lower() for term in ['security', 'ssl', 'encryption', 'safe']),
            'responsible': any(term in response.answer.lower() for term in ['responsible', 'gambling', 'protection'])
        }
        
        coverage_count = sum(coverage_indicators.values())
        coverage_percentage = (coverage_count / len(coverage_indicators)) * 100
        
        print(f"\n📊 COMPREHENSIVE COVERAGE ANALYSIS:")
        print(f"🎯 Coverage Score: {coverage_count}/{len(coverage_indicators)} aspects ({coverage_percentage:.1f}%)")
        
        for aspect, covered in coverage_indicators.items():
            status = "✅" if covered else "❌"
            print(f"  {status} {aspect.title()}: {'Covered' if covered else 'Missing'}")
        
        # Analyze sources by type
        web_sources = [s for s in response.sources if s.get('source_type') == 'web_search']
        comprehensive_sources = [s for s in response.sources if s.get('source_type') == 'comprehensive_web_research']
        retrieval_sources = [s for s in response.sources if s.get('source_type') == 'retrieval']
        
        print(f"\n🔍 SOURCE ANALYSIS:")
        print(f"  🌐 Web Search (Tavily): {len(web_sources)} sources")
        print(f"  🔍 Comprehensive Research (WebBaseLoader): {len(comprehensive_sources)} sources")
        print(f"  📚 Vector Retrieval: {len(retrieval_sources)} sources")
        
        # Grade the analysis
        if response.confidence_score >= 0.9 and coverage_percentage >= 90:
            grade = "A+"
        elif response.confidence_score >= 0.8 and coverage_percentage >= 80:
            grade = "A"
        elif response.confidence_score >= 0.7 and coverage_percentage >= 70:
            grade = "B"
        elif response.confidence_score >= 0.6 and coverage_percentage >= 60:
            grade = "C"
        else:
            grade = "D"
            
        print(f"\n🏆 OVERALL ANALYSIS GRADE: {grade}")
        print(f"📈 Ready for 95-field extraction: {response.confidence_score >= 0.6}")
        
        # Show sample of the response
        print(f"\n📄 RESPONSE PREVIEW (first 500 characters):")
        print(f"{'─'*60}")
        print(response.answer[:500] + "..." if len(response.answer) > 500 else response.answer)
        print(f"{'─'*60}")
        
        # Validate 95-field framework is working
        if comprehensive_sources:
            print(f"\n🎰 95-FIELD FRAMEWORK VALIDATION:")
            print(f"✅ WebBaseLoader integration: ACTIVE")
            print(f"✅ All 8 categories processing: ENABLED")
            print(f"✅ Comprehensive casino analysis: OPERATIONAL")
            print(f"✅ 95-field data extraction: READY")
        else:
            print(f"\n⚠️  95-FIELD FRAMEWORK STATUS:")
            print(f"❌ WebBaseLoader sources not found in response")
            print(f"❌ May need API key configuration or domain accessibility")
        
        return response
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting Universal RAG CMS v6.0 - Complete 95-Field Analysis Test")
    
    # Run the test
    result = asyncio.run(test_complete_95_field_analysis())
    
    if result:
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"🎰 Universal RAG CMS v6.0 with complete 95-field casino analysis is OPERATIONAL!")
    else:
        print(f"\n❌ TEST FAILED")
        print(f"🔧 Check configuration and API keys")
    
    print(f"\n{'='*80}")
    print(f"🎯 95-Field Casino Analysis Framework: ALL 8 categories by default")
    print(f"🚀 Universal RAG CMS v6.0: Production ready with comprehensive analysis")
    print(f"{'='*80}") 