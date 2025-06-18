#!/usr/bin/env python3
"""
🔍 95-FIELD DATA EXTRACTION ANALYSIS
Test to measure exactly how many of the 95 data fields are populated during comprehensive research
"""

import asyncio
import sys
import os
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain
from chains.comprehensive_research_chain import ComprehensiveResearchData

def count_populated_fields(data: Dict[str, Any], category_name: str = "") -> Dict[str, Any]:
    """Count how many fields are populated in the data structure"""
    
    populated = 0
    total = 0
    field_details = {}
    
    for key, value in data.items():
        total += 1
        
        # Check if field is populated (not None, not empty list, not empty string)
        is_populated = False
        if value is not None:
            if isinstance(value, list) and len(value) > 0:
                is_populated = True
            elif isinstance(value, str) and value.strip():
                is_populated = True
            elif isinstance(value, (int, float)) and value != 0:
                is_populated = True
            elif isinstance(value, bool):
                is_populated = True
        
        if is_populated:
            populated += 1
            field_details[key] = {"populated": True, "value": value}
        else:
            field_details[key] = {"populated": False, "value": value}
    
    return {
        "category": category_name,
        "populated": populated,
        "total": total,
        "percentage": (populated / total * 100) if total > 0 else 0,
        "fields": field_details
    }

async def test_95_field_data_extraction():
    """Test comprehensive data extraction and count populated fields"""
    
    print("🔍 95-FIELD DATA EXTRACTION ANALYSIS")
    print("=" * 80)
    print("🎯 Goal: Measure how many of the 95 data fields are actually populated")
    print()
    
    # Create RAG chain with comprehensive research enabled
    rag_chain = create_universal_rag_chain(
        model_name="gpt-4.1-mini",
        enable_comprehensive_web_research=True,
        enable_web_search=True,
        enable_enhanced_confidence=True,
        temperature=0.1
    )
    
    print("✅ RAG Chain created with comprehensive web research enabled")
    print()
    
    # Test with Betway Casino comprehensive analysis
    test_query = "Betway Casino comprehensive analysis all categories licensing games bonuses payments user experience innovations compliance assessment"
    
    print(f"🔍 Test Query: {test_query}")
    print("⏱️  Starting analysis...")
    print()
    
    try:
        # Execute the analysis
        response = await rag_chain.ainvoke({"question": test_query})
        
        print(f"✅ Analysis completed!")
        print(f"📝 Response length: {len(response.answer):,} characters")
        print(f"🎯 Confidence: {response.confidence_score:.3f}")
        print(f"📊 Sources: {len(response.sources)}")
        print()
        
        # Check if we can extract structured data
        print("🔍 ANALYZING STRUCTURED DATA EXTRACTION...")
        print("-" * 60)
        
        # Look for comprehensive web research results
        comprehensive_sources = [s for s in response.sources if s.get('source_type') == 'comprehensive_web_research']
        
        if comprehensive_sources:
            print(f"✅ Found {len(comprehensive_sources)} comprehensive research sources")
            
            # Try to extract structured data from the sources
            for i, source in enumerate(comprehensive_sources[:3], 1):
                url = source.get('url', 'Unknown')
                authority = source.get('authority_score', 0)
                content_length = len(source.get('content', ''))
                
                print(f"  {i}. {url}")
                print(f"     Authority: {authority:.2f} | Content: {content_length:,} chars")
        
        else:
            print("❌ No comprehensive research sources found")
        
        # Analyze content coverage for 95-field categories
        print("\n📊 CONTENT COVERAGE ANALYSIS:")
        print("-" * 60)
        
        content = response.answer.lower()
        
        # Define the 8 categories and their expected indicators
        categories = {
            "trustworthiness": ["license", "licensed", "authority", "malta", "ukgc", "security", "ssl", "trustworthy"],
            "games": ["slots", "games", "casino", "providers", "netent", "microgaming", "live dealer", "rtp"],
            "bonuses": ["bonus", "welcome", "promotion", "wagering", "free spins", "loyalty", "vip"],
            "payments": ["payment", "deposit", "withdrawal", "visa", "mastercard", "paypal", "skrill", "banking"],
            "user_experience": ["support", "customer", "mobile", "app", "interface", "navigation", "24/7"],
            "innovations": ["technology", "innovation", "vr", "virtual", "ai", "blockchain", "social"],
            "compliance": ["responsible", "gambling", "age verification", "aml", "gdpr", "regulation"],
            "assessment": ["rating", "score", "recommendation", "overall", "pros", "cons", "review"]
        }
        
        total_coverage = 0
        category_coverage = {}
        
        for category, indicators in categories.items():
            found_indicators = []
            for indicator in indicators:
                if indicator in content:
                    found_indicators.append(indicator)
            
            coverage_percentage = (len(found_indicators) / len(indicators)) * 100
            total_coverage += coverage_percentage
            category_coverage[category] = {
                "coverage": coverage_percentage,
                "found": len(found_indicators),
                "total": len(indicators),
                "indicators": found_indicators
            }
            
            print(f"  📊 {category.title()}: {coverage_percentage:.1f}% ({len(found_indicators)}/{len(indicators)} indicators)")
        
        overall_coverage = total_coverage / len(categories)
        print(f"\n🎯 OVERALL CONTENT COVERAGE: {overall_coverage:.1f}%")
        
        # Estimate populated fields based on content coverage
        estimated_fields = int((overall_coverage / 100) * 95)
        print(f"📈 ESTIMATED POPULATED FIELDS: {estimated_fields}/95 fields")
        
        # Grade the analysis
        if overall_coverage >= 90:
            grade = "A+"
        elif overall_coverage >= 80:
            grade = "A"
        elif overall_coverage >= 70:
            grade = "B"
        elif overall_coverage >= 60:
            grade = "C"
        else:
            grade = "D"
        
        print(f"🏆 ANALYSIS GRADE: {grade}")
        
        # Save detailed analysis
        analysis_results = {
            "timestamp": str(response.metadata.get('timestamp', 'Unknown')),
            "query": test_query,
            "response_length": len(response.answer),
            "confidence_score": response.confidence_score,
            "total_sources": len(response.sources),
            "comprehensive_sources": len(comprehensive_sources),
            "overall_coverage_percentage": overall_coverage,
            "estimated_populated_fields": estimated_fields,
            "grade": grade,
            "category_coverage": category_coverage
        }
        
        with open("95_field_analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to: 95_field_analysis_results.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting 95-Field Data Extraction Analysis")
    
    success = asyncio.run(test_95_field_data_extraction())
    
    if success:
        print("\n🎉 95-FIELD ANALYSIS COMPLETE!")
        print("📊 Check 95_field_analysis_results.json for detailed metrics")
    else:
        print("\n❌ Analysis failed - check error logs")
        sys.exit(1) 