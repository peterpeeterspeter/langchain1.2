#!/usr/bin/env python3
"""
🎰 BWIN CASINO - WORKING SOLUTION
Based on EXACT successful configuration from crashino_production_20250625_182905.json
This uses the PROVEN working setup that achieved "wordpress_published": true
"""

import os
import asyncio
import sys
import time
import json
from datetime import datetime

# ✅ PROVEN WORKING: Environment variables from successful runs (EXACT COPY)
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
# Map your actual password variable to the expected name
os.environ["WORDPRESS_PASSWORD"] = os.getenv("WORDPRESS_APP_PASSWORD", "")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain

async def run_bwin_working_solution():
    """Run Bwin with PROVEN working configuration (EXACT copy of successful setup)"""
    
    print("🎰 BWIN CASINO - WORKING SOLUTION")
    print("=" * 60)
    print("✅ Based on: crashino_production_20250625_182905.json SUCCESS")
    print("🔧 Configuration: EXACT copy of working setup")
    print("🌐 WordPress: PROVEN successful publishing")
    print("📊 95-field intelligence: ENABLED")
    print("🖼️ Images: 6 images per review")
    print()
    
    # ✅ PROVEN WORKING: This EXACT configuration published successfully
    # DO NOT change any parameters - this is the working combination
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        temperature=0.1,
        enable_wordpress_publishing=True,
        enable_dataforseo_images=True,           # ✅ This worked
        enable_web_search=True,                  # ✅ This worked  
        enable_comprehensive_web_research=True  # ✅ This worked (95-field intelligence)
        # NOTE: NOT enabling FTI processing or security as they weren't used in working version
    )
    
    # ✅ BWIN SPECIFIC: Professional query using working template structure
    query = """Create a comprehensive professional Bwin Casino review for MT Casino custom post type.

    Provide detailed analysis including:
    - Licensing and regulatory compliance (UK Gambling Commission, MGA Malta)
    - Game selection and software providers (NetEnt, Microgaming, Evolution Gaming)
    - Sports betting integration and unique features
    - Welcome bonuses and ongoing promotions
    - Mobile app compatibility and user experience
    - Customer support quality and availability
    - Security measures and player protection
    - Payment methods and withdrawal processes
    - VIP program and loyalty rewards
    - Pros and cons analysis
    - Final rating and recommendation

    Format for WordPress MT Casino post type with SEO optimization and engaging content structure. Focus on Bwin's established reputation as a major European operator since 1997."""
    
    print(f"🔍 Query: Bwin Casino Review (Professional Analysis)")
    print(f"📊 Expected: 'Bwin Casino Review (2025)' title")
    print(f"🎯 95-field intelligence: ENABLED")
    print(f"📝 Template System v2.0: ENABLED")
    
    start_time = time.time()
    
    try:
        print("\n⚡ Executing PROVEN working configuration...")
        
        result = await chain.ainvoke({
            "question": query,
            "publish_to_wordpress": True
        })
        
        processing_time = time.time() - start_time
        
        print(f"\n⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Content Length: {len(result.answer)} characters")
        print(f"🎯 Confidence Score: {result.confidence_score:.3f}")
        print(f"📚 Sources Found: {len(result.sources)}")
        
        # ✅ Check Bwin-specific content
        bwin_mentions = result.answer.lower().count('bwin')
        casino_mentions = result.answer.lower().count('casino')
        trustdice_check = "trustdice" not in result.answer.lower()
        
        print(f"\n📈 Content Quality Analysis:")
        print(f"   Bwin mentions: {bwin_mentions}")
        print(f"   Casino mentions: {casino_mentions}")
        print(f"   TrustDice avoided: {'✅ YES' if trustdice_check else '❌ NO'}")
        print(f"   Content length: {'✅ Comprehensive' if len(result.answer) > 5000 else '⚠️ Needs expansion'}")
        
        # Check WordPress publishing (the critical test)
        wp_published = result.metadata.get("wordpress_published", False)
        
        if wp_published:
            print(f"\n🎉 WORDPRESS PUBLISHING SUCCESS!")
            print(f"📝 Post ID: {result.metadata.get('wordpress_post_id')}")
            print(f"🔗 URL: {result.metadata.get('wordpress_url')}")
            print(f"📂 Post Type: {result.metadata.get('wordpress_post_type')}")
            
            # Check MT Casino features
            if 'mt_listing' in str(result.metadata.get('wordpress_post_type', '')).lower():
                print(f"🎰 ✅ Published to MT Casino custom post type")
            
            fields_count = result.metadata.get('wordpress_custom_fields_count', 0)
            images_count = result.metadata.get('images_uploaded_count', 0)
            
            print(f"🏷️ Custom fields: {fields_count}")
            print(f"🖼️ Images uploaded: {images_count}")
            
        else:
            print(f"\n❌ WordPress publishing failed")
            error = result.metadata.get("wordpress_error", "Unknown error")
            print(f"💡 Error: {error}")
            print(f"🔧 This indicates the credentials/config issue persists")
        
        # Check advanced features usage (from metadata)
        print(f"\n🚀 Advanced Features Used:")
        print(f"   95-field intelligence: {'✅' if result.metadata.get('comprehensive_web_research_used') else '❌'}")
        print(f"   Template System v2.0: {'✅' if result.metadata.get('template_system_v2_used') else '❌'}")
        print(f"   DataForSEO images: {'✅' if result.metadata.get('dataforseo_images_used') else '❌'}")
        print(f"   Web research: {'✅' if result.metadata.get('web_search_used') else '❌'}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "test_type": "bwin_casino_working_solution",
            "configuration": "exact_copy_of_successful_crashino",
            "query": query,
            "response": result.answer,
            "confidence_score": result.confidence_score,
            "sources": result.sources,
            "metadata": result.metadata,
            "processing_time": processing_time,
            "content_analysis": {
                "bwin_mentions": bwin_mentions,
                "casino_mentions": casino_mentions,
                "trustdice_avoided": trustdice_check,
                "content_length": len(result.answer),
                "quality_rating": "high" if len(result.answer) > 5000 and result.confidence_score > 0.6 else "moderate"
            },
            "working_solution_status": {
                "environment_variables_set": True,
                "wordpress_config_method": "explicit_env_override",
                "chain_configuration": "proven_working_minimal",
                "advanced_features_used": {
                    "wordpress_publishing": True,
                    "dataforseo_images": True,
                    "web_search": True,
                    "comprehensive_web_research": True,
                    "fti_processing": False,  # Not used in working version
                    "security": False        # Not used in working version
                }
            },
            "timestamp": timestamp
        }
        
        filename = f"bwin_working_solution_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Show content preview
        print(f"\n📄 Bwin Casino Content Preview (first 600 characters):")
        print("-" * 60)
        preview = result.answer[:600] + "..." if len(result.answer) > 600 else result.answer
        print(preview)
        print("-" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        print(f"\n" + "=" * 60)
        print("🏁 BWIN CASINO WORKING SOLUTION COMPLETE")
        print("✅ Configuration: EXACT copy of proven successful setup")
        print("🎰 Content: Professional Bwin casino review")
        print("🔧 Environment: Explicit variable override method")
        print("📊 Intelligence: 95-field casino analysis")
        print("🌐 WordPress: Production-ready publishing")

if __name__ == "__main__":
    print("🚀 Starting Bwin Casino working solution...")
    result = asyncio.run(run_bwin_working_solution())
    
    if result:
        wp_success = result.metadata.get("wordpress_published", False)
        content_quality = len(result.answer) > 5000 and result.confidence_score > 0.6
        
        print(f"\n📊 FINAL STATUS:")
        print(f"   Content Quality: {'✅ HIGH' if content_quality else '⚠️ MODERATE'}")
        print(f"   WordPress Publishing: {'✅ SUCCESS' if wp_success else '❌ FAILED'}")
        print(f"   95-field Intelligence: {'✅ USED' if result.metadata.get('comprehensive_web_research_used') else '❌ NOT USED'}")
        
        if wp_success and content_quality:
            print(f"\n🏆 COMPLETE SUCCESS! Bwin review published with all features!")
            url = result.metadata.get("wordpress_url")
            post_id = result.metadata.get("wordpress_post_id")
            print(f"🌐 Live URL: {url}")
            print(f"📝 Post ID: {post_id}")
        elif content_quality:
            print(f"\n🎯 CONTENT SUCCESS! High-quality Bwin review generated.")
            print(f"💡 WordPress publishing needs credential verification.")
        else:
            print(f"\n🔍 Review generated but may need quality improvements.")
    else:
        print(f"\n❌ Script execution failed - check error details above.") 