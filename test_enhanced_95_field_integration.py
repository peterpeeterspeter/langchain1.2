#!/usr/bin/env python3
"""
🔍 ENHANCED 95-FIELD INTEGRATION TEST
Validates that structured casino data is properly extracted and used in content generation

NEW FEATURES TESTED:
✅ Structured data extraction from comprehensive web research
✅ Casino-specific template selection
✅ Enhanced context integration with 95-field framework
✅ End-to-end content generation with structured data
"""

import asyncio
import sys
import os
import re
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def test_enhanced_95_field_integration():
    """Test enhanced 95-field integration with structured data extraction"""
    
    print(f"\n{'='*80}")
    print(f"🔍 ENHANCED 95-FIELD INTEGRATION TEST")
    print(f"{'='*80}")
    print("🎯 Goal: Validate structured data extraction and content generation")
    print()
    
    # Create Universal RAG Chain with comprehensive web research enabled
    print("🚀 Initializing Universal RAG Chain with enhanced integration...")
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4.1-mini",
        enable_comprehensive_web_research=True,  # Core feature
        enable_web_search=True,                  # Additional coverage
        enable_template_system_v2=True,          # Enhanced templates
        enable_enhanced_confidence=True,         # Quality scoring
        temperature=0.1
    )
    
    print("✅ Universal RAG Chain initialized with enhanced 95-field integration!")
    print()
    
    # Test query designed to trigger comprehensive casino analysis
    test_query = "Comprehensive analysis of Betway Casino including licensing trustworthiness games software bonuses promotions payment methods user experience mobile support"
    
    print(f"🔍 TEST QUERY: {test_query}")
    print("⏱️  Starting enhanced analysis...")
    print()
    
    try:
        start_time = datetime.now()
        
        # Execute the analysis
        response = await rag_chain.ainvoke({"question": test_query})
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"✅ ANALYSIS COMPLETED!")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
        print(f"📊 Total Sources: {len(response.sources)}")
        print()
        
        # Analyze structured data integration
        print("🔍 STRUCTURED DATA INTEGRATION ANALYSIS:")
        print("-" * 60)
        
        # Check if structured casino data appears in the response
        content = response.answer.lower()
        structured_indicators = {
            'licensing_data': any(term in content for term in ['malta gaming authority', 'uk gambling commission', 'licensed by']),
            'games_data': any(term in content for term in ['slot games', 'live casino', 'game providers']),
            'payment_data': any(term in content for term in ['visa', 'mastercard', 'deposit methods', 'withdrawal']),
            'support_data': any(term in content for term in ['24/7', 'customer support', 'mobile app']),
            'structured_sections': any(term in content for term in ['trustworthiness', 'games & software', 'payment methods']),
            'specific_numbers': bool(re.search(r'\d+[+\s]*(?:slots?|games)', content)),
            'authority_mentions': any(term in content for term in ['authority', 'regulation', 'compliance'])
        }
        
        integration_score = sum(structured_indicators.values())
        total_possible = len(structured_indicators)
        integration_percentage = (integration_score / total_possible) * 100
        
        print(f"📊 INTEGRATION SCORE: {integration_score}/{total_possible} ({integration_percentage:.1f}%)")
        
        for indicator, found in structured_indicators.items():
            status = "✅" if found else "❌"
            print(f"  {status} {indicator.replace('_', ' ').title()}: {'Found' if found else 'Missing'}")
        
        # Check for comprehensive web research sources
        comprehensive_sources = [s for s in response.sources if s.get('source_type') == 'comprehensive_web_research']
        
        print(f"\n🌐 SOURCE ANALYSIS:")
        print(f"  🔍 Comprehensive Research Sources: {len(comprehensive_sources)}")
        
        if comprehensive_sources:
            total_authority = sum(s.get('authority', 0.7) for s in comprehensive_sources)
            avg_authority = total_authority / len(comprehensive_sources)
            
            print(f"  🏆 Average Authority Score: {avg_authority:.3f}")
            
            # Show sample sources
            for i, source in enumerate(comprehensive_sources[:3], 1):
                url = source.get('url', 'Unknown')
                authority = source.get('authority', 0.0)
                review_site = source.get('review_site', 'Direct Site')
                print(f"    {i}. {review_site} - {url} (Authority: {authority:.2f})")
        
        # Evaluate template effectiveness
        print(f"\n📝 TEMPLATE EVALUATION:")
        
        # Check if casino-specific structure was used
        has_structured_sections = any(section in response.answer for section in [
            'Executive Summary', 'Licensing', 'Games & Software', 'Bonuses', 'Payment Methods', 'User Experience'
        ])
        
        section_count = len(re.findall(r'#{1,3}\s+[A-Z]', response.answer))
        word_count = len(response.answer.split())
        
        print(f"  📋 Structured Sections: {'Yes' if has_structured_sections else 'No'}")
        print(f"  📊 Section Count: {section_count}")
        print(f"  📝 Word Count: {word_count:,}")
        print(f"  🎯 Template Type: {'Casino-Specific' if has_structured_sections else 'Generic'}")
        
        # Overall integration grade
        if integration_percentage >= 90 and has_structured_sections and len(comprehensive_sources) >= 2:
            grade = "A+"
        elif integration_percentage >= 80 and has_structured_sections:
            grade = "A"
        elif integration_percentage >= 70:
            grade = "B"
        elif integration_percentage >= 60:
            grade = "C"
        else:
            grade = "D"
        
        print(f"\n🏆 OVERALL INTEGRATION GRADE: {grade}")
        
        # Show response preview with structure analysis
        print(f"\n📄 RESPONSE PREVIEW (first 800 characters):")
        print(f"{'─'*80}")
        preview = response.answer[:800] + "..." if len(response.answer) > 800 else response.answer
        print(preview)
        print(f"{'─'*80}")
        
        # Validate 95-field framework utilization
        print(f"\n🎰 95-FIELD FRAMEWORK UTILIZATION:")
        utilization_factors = {
            'structured_data_extracted': integration_score >= 5,
            'casino_template_used': has_structured_sections,
            'comprehensive_sources_found': len(comprehensive_sources) >= 1,
            'authority_sources_used': len([s for s in comprehensive_sources if s.get('authority', 0) > 0.85]) >= 1 if comprehensive_sources else False,
            'specific_data_points': bool(re.search(r'\d+', response.answer)),
            'multiple_categories_covered': integration_score >= 4
        }
        
        utilization_score = sum(utilization_factors.values())
        total_factors = len(utilization_factors)
        utilization_percentage = (utilization_score / total_factors) * 100
        
        print(f"✅ Framework Utilization: {utilization_score}/{total_factors} ({utilization_percentage:.1f}%)")
        
        for factor, achieved in utilization_factors.items():
            status = "✅" if achieved else "❌"
            print(f"  {status} {factor.replace('_', ' ').title()}")
        
        # Final assessment
        if utilization_percentage >= 90:
            assessment = "EXCELLENT - Full 95-field framework integration achieved"
        elif utilization_percentage >= 75:
            assessment = "GOOD - Strong integration with minor gaps"
        elif utilization_percentage >= 60:
            assessment = "FAIR - Basic integration working, needs improvement"
        else:
            assessment = "POOR - Integration not working effectively"
        
        print(f"\n🎯 FINAL ASSESSMENT: {assessment}")
        
        return {
            'success': True,
            'integration_score': integration_percentage,
            'utilization_score': utilization_percentage,
            'grade': grade,
            'comprehensive_sources': len(comprehensive_sources),
            'response_length': len(response.answer)
        }
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🚀 Starting Enhanced 95-Field Integration Test")
    
    # Run the test
    result = asyncio.run(test_enhanced_95_field_integration())
    
    if result.get('success'):
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"📊 Integration Score: {result['integration_score']:.1f}%")
        print(f"🎯 Utilization Score: {result['utilization_score']:.1f}%")
        print(f"🏆 Grade: {result['grade']}")
        print(f"🔍 Comprehensive Sources: {result['comprehensive_sources']}")
        
        if result['integration_score'] >= 75:
            print(f"\n🎉 95-Field Framework is being effectively leveraged!")
        else:
            print(f"\n⚠️  95-Field Framework needs optimization")
    else:
        print(f"\n❌ TEST FAILED: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}")
    print(f"🎯 Enhanced 95-Field Integration Test Complete")
    print(f"{'='*80}") 