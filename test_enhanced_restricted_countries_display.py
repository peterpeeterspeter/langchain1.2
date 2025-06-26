#!/usr/bin/env python3
"""
Enhanced Restricted Countries Display Test
Tests the comprehensive geographic restrictions enhancements for affiliate compliance.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chains.universal_rag_lcel import create_universal_rag_chain

async def test_enhanced_restricted_countries_display():
    """Test enhanced restricted countries display for affiliate compliance"""
    
    print("🧪 Testing Enhanced Restricted Countries Display for Affiliate Compliance")
    print("=" * 80)
    
    # Create test chain
    test_chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_comprehensive_web_research=True,
        enable_template_system_v2=True,
        enable_wordpress_publishing=False,  # Disable for testing
        enable_profiling=True
    )
    
    # Mock structured data with extensive restricted countries
    mock_structured_data = {
        'casino_name': 'Test Casino',
        'overall_rating': 8.5,
        'safety_score': 8.0,
        'confidence_score': 0.85,
        'extraction_timestamp': datetime.now().isoformat(),
        'data_sources': ['official_site', 'terms_conditions', 'license_info', 'review_sites'],
        'schema_version': '1.0',
        'trustworthiness': {
            'license_authorities': ['Malta Gaming Authority', 'UK Gambling Commission'],
            'years_in_operation': 10,
            'ssl_certification': True,
            'age_verification': True,
            'responsible_gambling_tools': ['deposit_limits', 'self_exclusion', 'reality_check']
        },
        'games': {
            'slot_count': 2500,
            'table_games_count': 150,
            'live_casino_available': True,
            'providers': ['NetEnt', 'Microgaming', 'Pragmatic Play']
        },
        'payments': {
            'crypto_support': True,
            'withdrawal_processing_time': '24-48 hours'
        },
        'innovations': {
            'vr_gaming': False,
            'ai_personalization': True
        },
        'bonuses': {
            'welcome_bonus_amount': '€1000 + 200 Free Spins'
        },
        # Extensive restricted countries list
        'geographic_restrictions': 'Restricted in United States, United Kingdom, France, Germany, Spain, Italy, Netherlands, Belgium, Australia, Denmark, Sweden, Norway, Finland, Turkey, Israel, Iran, North Korea, Syria',
        'restricted_countries': ['United States', 'United Kingdom', 'France', 'Germany', 'Spain', 'Italy', 'Netherlands', 'Belgium', 'Australia', 'Denmark', 'Sweden', 'Norway', 'Finland', 'Turkey', 'Israel'],
        'country_restrictions': 'Not available in US, UK, and other SEPA countries'
    }
    
    print("🔍 Testing Restricted Countries Extraction...")
    
    # Test restricted countries extraction
    restricted_countries = test_chain._extract_restricted_countries(mock_structured_data)
    print(f"✅ Extracted {len(restricted_countries)} restricted countries:")
    for i, country in enumerate(restricted_countries, 1):
        print(f"   {i}. {country}")
    
    print("\n" + "="*50)
    print("🎨 Testing Enhanced Display Components...")
    print("="*50)
    
    # Test intelligence insights with geographic restrictions
    print("\n📊 Testing Intelligence Insights with Geographic Restrictions:")
    insights = test_chain._generate_intelligence_insights(mock_structured_data, "casino_review")
    print(insights)
    
    print("\n" + "-"*50)
    
    # Test compliance notice with geographic restrictions
    print("\n⚠️ Testing Enhanced Compliance Notice:")
    compliance_notice = test_chain._generate_compliance_notice(mock_structured_data)
    print(compliance_notice)
    
    print("\n" + "-"*50)
    
    # Test comprehensive geographic restrictions disclosure
    print("\n🌍 Testing Comprehensive Geographic Restrictions Disclosure:")
    geo_disclosure = test_chain._generate_geographic_restrictions_disclosure(mock_structured_data)
    print(geo_disclosure)
    
    print("\n" + "-"*50)
    
    # Test content formatting with geographic restrictions
    print("\n📝 Testing Content Formatting with Geographic Restrictions:")
    
    # Mock structured content for formatting test
    class MockStructuredContent:
        def __init__(self):
            self.title = "Test Casino Review - Complete Analysis"
            self.executive_summary = "Test Casino offers extensive gaming with significant geographic restrictions."
            self.main_sections = [
                {"header": "Game Selection", "content": "Over 2500 games available."},
                {"header": "Banking Options", "content": "Multiple payment methods including crypto."}
            ]
            self.pros_list = ["Extensive game library", "Crypto support", "Licensed operation"]
            self.cons_list = ["Extensive geographic restrictions", "Complex bonus terms"]
            self.key_takeaways = ["Verify country eligibility before registering", "Strong game selection"]
            self.final_verdict = "Good casino with extensive restrictions."
            self.overall_rating = 8.5
    
    mock_content = MockStructuredContent()
    formatted_content = test_chain._format_structured_casino_content(mock_content, mock_structured_data)
    
    # Show key sections
    lines = formatted_content.split('\n')
    geo_section_start = None
    for i, line in enumerate(lines):
        if '🌍 Geographic Restrictions' in line:
            geo_section_start = i
            break
    
    if geo_section_start:
        print("Geographic Restrictions Section Found:")
        for i in range(geo_section_start, min(geo_section_start + 15, len(lines))):
            print(f"   {lines[i]}")
    else:
        print("❌ Geographic Restrictions section not found in content")
    
    print("\n" + "="*50)
    print("🧮 Testing Integration in Full Content Generation...")
    print("="*50)
    
    # Test post-processing with casino intelligence
    sample_content = """# Test Casino Review
    
    ## Executive Summary
    Test Casino offers a comprehensive gaming experience.
    
    ## Game Selection
    Over 2500 games available from top providers.
    """
    
    enhanced_content = await test_chain._post_process_with_casino_intelligence(
        sample_content, 
        mock_structured_data, 
        "casino_review",
        "Test Casino review"
    )
    
    # Check if enhanced content includes geographic restrictions
    if "🌍" in enhanced_content and "GEOGRAPHIC RESTRICTIONS" in enhanced_content:
        print("✅ Post-processing successfully added geographic restrictions disclosure")
        
        # Find and display the geographic restrictions section
        lines = enhanced_content.split('\n')
        geo_start = None
        for i, line in enumerate(lines):
            if "GEOGRAPHIC RESTRICTIONS" in line:
                geo_start = i
                break
        
        if geo_start:
            print("\nGeographic Restrictions in Enhanced Content:")
            for i in range(geo_start, min(geo_start + 20, len(lines))):
                if lines[i].strip():  # Only show non-empty lines
                    print(f"   {lines[i]}")
    else:
        print("❌ Geographic restrictions disclosure not found in enhanced content")
    
    print("\n" + "="*50)
    print("📋 Test Summary")
    print("="*50)
    
    test_results = {
        'restricted_countries_extracted': len(restricted_countries),
        'has_intelligence_insights': bool("🚫" in insights and "Geographic Restrictions" in insights),
        'has_compliance_notice': bool("🌍" in compliance_notice and "NOT AVAILABLE" in compliance_notice),
        'has_comprehensive_disclosure': bool("GEOGRAPHIC RESTRICTIONS - IMPORTANT NOTICE" in geo_disclosure),
        'has_content_formatting': bool(geo_section_start is not None),
        'has_post_processing': bool("🌍" in enhanced_content and "GEOGRAPHIC RESTRICTIONS" in enhanced_content),
        'affiliate_disclosure_present': bool("affiliate" in compliance_notice.lower()),
        'compliance_warnings_present': bool("verify" in geo_disclosure.lower() and "eligibility" in geo_disclosure.lower())
    }
    
    print("✅ Test Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Overall assessment
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n🏆 Overall Score: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Enhanced restricted countries display is working correctly.")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Most tests passed. Minor issues may need attention.")
    else:
        print("⚠️ Several tests failed. Review implementation.")
    
    # Save detailed results
    detailed_results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'restricted_countries_found': restricted_countries,
        'sample_insights': insights,
        'sample_compliance_notice': compliance_notice,
        'sample_geo_disclosure': geo_disclosure[:500] + "..." if len(geo_disclosure) > 500 else geo_disclosure
    }
    
    with open('enhanced_restricted_countries_test_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: enhanced_restricted_countries_test_results.json")
    print("🔍 Enhanced Restricted Countries Display Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_restricted_countries_display()) 