#!/usr/bin/env python3
"""
Quick test of enhanced restricted countries display with real Betway Casino query
"""

import asyncio
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chains.universal_rag_lcel import create_universal_rag_chain

async def test_betway_enhanced_display():
    """Test enhanced display with real Betway Casino query"""
    
    print("🎰 Testing Enhanced Restricted Countries Display - Betway Casino")
    print("=" * 70)
    
    # Create chain
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_comprehensive_web_research=True,
        enable_template_system_v2=True,
        enable_wordpress_publishing=False,
        enable_profiling=False
    )
    
    # Test query
    query = "Betway Casino review with geographic restrictions analysis"
    
    print(f"📝 Query: {query}")
    print("🔍 Processing with enhanced geographic restrictions display...")
    print("-" * 70)
    
    try:
        # Process the query
        result = await chain.ainvoke(
            {"question": query},
            publish_to_wordpress=False
        )
        
        print("✅ Processing completed!")
        print(f"📊 Confidence Score: {result.confidence_score:.2f}")
        print(f"⏱️ Processing Time: {result.response_time:.2f}s")
        print(f"🔍 Sources: {len(result.sources)}")
        
        # Look for geographic restrictions in the content
        content = result.answer
        if "🌍" in content and ("GEOGRAPHIC RESTRICTIONS" in content or "Geographic Restrictions" in content):
            print("\n✅ Geographic Restrictions Section Found!")
            
            # Extract and display the geographic restrictions section
            lines = content.split('\n')
            geo_start = None
            for i, line in enumerate(lines):
                if "GEOGRAPHIC RESTRICTIONS" in line or ("🌍" in line and "Geographic" in line):
                    geo_start = i
                    break
            
            if geo_start:
                print("\n📋 Geographic Restrictions Display:")
                print("=" * 50)
                # Show the geographic restrictions section
                for i in range(geo_start, min(geo_start + 25, len(lines))):
                    if lines[i].strip():
                        print(lines[i])
                    if i > geo_start and "##" in lines[i] and "Geographic" not in lines[i]:
                        break  # Stop at next section
                print("=" * 50)
        else:
            print("\n❌ Geographic restrictions section not found in content")
        
        # Check for compliance features
        compliance_features = {
            'affiliate_disclosure': 'affiliate' in content.lower(),
            'age_verification': '18+' in content or 'age verification' in content.lower(),
            'responsible_gambling': 'responsible gambling' in content.lower() or 'gamble responsibly' in content.lower(),
            'country_verification': 'verify' in content.lower() and ('country' in content.lower() or 'eligibility' in content.lower()),
            'licensing_info': 'licens' in content.lower(),
            'geographic_warnings': '🌍' in content or '🚫' in content
        }
        
        print(f"\n🏆 Affiliate Compliance Features:")
        for feature, present in compliance_features.items():
            status = "✅" if present else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}")
        
        compliance_score = sum(compliance_features.values()) / len(compliance_features)
        print(f"\n📊 Compliance Score: {compliance_score:.1%}")
        
        if compliance_score >= 0.8:
            print("🎉 Excellent affiliate compliance!")
        elif compliance_score >= 0.6:
            print("✅ Good affiliate compliance")
        else:
            print("⚠️ Review compliance features")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_betway_enhanced_display()) 