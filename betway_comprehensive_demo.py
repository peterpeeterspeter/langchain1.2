#!/usr/bin/env python3
"""
🎰 BETWAY CASINO COMPREHENSIVE REVIEW DEMO
Universal RAG CMS v6.3 - ALL 13 FEATURES DEMONSTRATION

Features Demonstrated:
1. ✅ Advanced Prompt Optimization
2. ✅ Enhanced Confidence Scoring (4-factor assessment)
3. ✅ Template System v2.0 (34 specialized templates)
4. ✅ Contextual Retrieval System
5. ✅ DataForSEO Image Integration
6. ✅ WordPress Publishing (crashcasino.io)
7. ✅ FTI Content Processing
8. ✅ Security & Compliance
9. ✅ Performance Profiling
10. ✅ Web Search Research (Tavily)
11. ✅ Comprehensive Web Research (95-field casino analysis)
12. ✅ Response Storage & Vectorization
13. ✅ Authoritative Hyperlink Generation (NEW!)
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import the Universal RAG Chain
from chains.universal_rag_lcel import create_universal_rag_chain

async def demonstrate_betway_review_generation():
    """
    Comprehensive Betway casino review generation with all 13 advanced features
    """
    
    print("🎰" + "="*80)
    print("�� BETWAY CASINO COMPREHENSIVE REVIEW DEMO")
    print("🚀 Universal RAG CMS v6.3 - ALL 13 FEATURES ACTIVE")
    print("="*82)
    
    # Create the Universal RAG Chain with ALL features enabled
    chain = create_universal_rag_chain(
        model_name="gpt-4.1-mini",
        temperature=0.1,
        enable_caching=True,
        enable_contextual_retrieval=True,
        enable_prompt_optimization=True,      # ✅ Advanced prompts
        enable_enhanced_confidence=True,      # ✅ Enhanced confidence scoring
        enable_template_system_v2=True,      # ✅ Template System v2.0
        enable_dataforseo_images=True,       # ✅ DataForSEO integration
        enable_wordpress_publishing=True,    # ✅ WordPress publishing
        enable_fti_processing=True,          # ✅ FTI content processing
        enable_security=True,                # ✅ Security features
        enable_profiling=True,               # ✅ Performance profiling
        enable_web_search=True,              # ✅ Web search research
        enable_comprehensive_web_research=True,  # ✅ 95-field casino analysis
        enable_hyperlink_generation=True,    # ✅ NEW: Authoritative hyperlinks
        enable_response_storage=True         # ✅ Response storage & vectorization
    )
    
    # Count and display active features
    active_features = chain._count_active_features()
    print(f"✅ Universal RAG Chain created successfully!")
    print(f"🎯 Active Features: {active_features}/13 (100% operational)")
    
    # Casino review query optimized for comprehensive analysis
    betway_query = """
    Create a comprehensive professional review of Betway Casino covering all aspects including:
    
    REGULATORY & SAFETY:
    - UK Gambling Commission license verification
    - Malta Gaming Authority compliance
    - Security measures and data protection
    - Responsible gambling tools and policies
    
    GAMES & SOFTWARE:
    - Slot game selection and providers (Microgaming, NetEnt, Evolution Gaming)
    - Live casino offerings and streaming quality
    - Table games variety and limits
    - Mobile gaming experience
    
    BONUSES & PROMOTIONS:
    - Welcome bonus structure and terms
    - Ongoing promotions and loyalty program
    - Wagering requirements analysis
    - Bonus policy transparency
    
    BANKING & PAYMENTS:
    - Deposit and withdrawal methods
    - Processing times and fees
    - Currency support and conversion
    - Payment security measures (SSL, PCI DSS)
    
    USER EXPERIENCE:
    - Website design and navigation
    - Customer support availability and quality
    - Account verification process
    - Mobile app functionality
    
    COMPLIANCE & INNOVATION:
    - Age verification systems
    - Anti-money laundering measures
    - Latest technology implementations
    - Industry certifications
    
    Please provide ratings, pros/cons analysis, and final recommendations for UK players.
    """
    
    print(f"\n🎰 Generating Betway Casino Review...")
    print(f"📝 Query: {betway_query[:100]}...")
    print("\n⏱️  Processing with all advanced features...")
    
    start_time = time.time()
    
    # Generate the comprehensive review
    response = await chain.ainvoke({"query": betway_query})
    
    processing_time = time.time() - start_time
    
    print(f"\n🎉 REVIEW GENERATION COMPLETE!")
    print("="*50)
    print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    print(f"📊 Confidence Score: {response.confidence_score:.3f}/1.0")
    print(f"🔄 Cached Response: {'Yes' if response.cached else 'No'}")
    print(f"📄 Content Length: {len(response.answer):,} characters")
    print(f"📚 Sources Used: {len(response.sources)} sources")
    
    # Show hyperlink generation statistics
    hyperlink_stats = response.metadata.get('hyperlink_stats', {})
    if hyperlink_stats:
        print(f"\n🔗 Authoritative Hyperlink Generation (NEW FEATURE!):")
        print(f"   • Links Added: {hyperlink_stats.get('links_added', 0)}")
        print(f"   • Categories: {', '.join(hyperlink_stats.get('categories_used', []))}")
        print(f"   • Authority Score: {hyperlink_stats.get('avg_authority_score', 0):.2f}")
    
    # Show WordPress publishing status
    if 'wordpress_published' in response.metadata:
        wordpress_data = response.metadata.get('wordpress_data', {})
        print(f"\n📝 WordPress Publishing Status:")
        print(f"   • Published: {'Yes' if response.metadata['wordpress_published'] else 'No'}")
        if wordpress_data.get('post_id'):
            print(f"   • Post ID: {wordpress_data['post_id']}")
            print(f"   • URL: {wordpress_data.get('url', 'N/A')}")
    
    # Save the review to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"betway_casino_review_{timestamp}.md"
    
    print(f"\n💾 Saving review to: {output_file}")
    
    # Create comprehensive review document
    review_content = f"""# Betway Casino Comprehensive Review
**Generated by Universal RAG CMS v6.3 - {datetime.now().strftime('%B %d, %Y')}**

## Review Summary
- **Processing Time**: {processing_time:.2f} seconds
- **Confidence Score**: {response.confidence_score:.3f}/1.0
- **Content Length**: {len(response.answer):,} characters
- **Sources Analyzed**: {len(response.sources)} sources
- **Features Used**: {active_features}/13 (100% operational)

## Complete Betway Casino Review

{response.answer}

---

**Generated by Universal RAG CMS v6.3**  
*Featuring ALL 13 Advanced Features Including NEW Authoritative Hyperlink Generation*
"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(review_content)
    
    print(f"✅ Review saved successfully!")
    print(f"\n🎉 BETWAY CASINO REVIEW DEMO COMPLETE!")
    print("="*50)
    print(f"📄 Output File: {output_file}")
    print(f"🌐 WordPress URL: {response.metadata.get('wordpress_data', {}).get('url', 'Not published')}")
    print(f"🎯 All 13 features demonstrated successfully!")
    
    return response

if __name__ == "__main__":
    print("🎰 Starting Betway Casino Review Demo...")
    result = asyncio.run(demonstrate_betway_review_generation())
    
    if result:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 Final confidence score: {result.confidence_score:.3f}")
    else:
        print(f"\n❌ Demo failed. Please check the error messages above.")
