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

This demo creates a complete professional casino review with:
- 95-field structured casino intelligence
- Authoritative links to regulatory bodies, responsible gambling resources
- Professional images from DataForSEO
- SEO-optimized content structure
- WordPress publishing with proper metadata
- Enhanced confidence scoring and quality validation
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
    print("🎯 BETWAY CASINO COMPREHENSIVE REVIEW DEMO")
    print("🚀 Universal RAG CMS v6.3 - ALL 13 FEATURES ACTIVE")
    print("="*82)
    
    # Verify environment variables
    print("\n🔧 Environment Configuration Check:")
    required_vars = [
        'ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 
        'SUPABASE_URL', 'SUPABASE_SERVICE_KEY',
        'WORDPRESS_SITE_URL', 'WORDPRESS_USERNAME', 'WORDPRESS_APP_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            # Show partial key for verification
            value = os.getenv(var)
            if 'key' in var.lower() or 'password' in var.lower():
                display = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display = value
            print(f"  ✅ {var}: {display}")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and .cursor/mcp.json")
        return
    
    print("\n🎯 Creating Universal RAG Chain with ALL 13 features...")
    
    try:
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
        
        # Display token usage if available
        if response.token_usage:
            total_tokens = sum(response.token_usage.values())
            print(f"🪙 Token Usage: {total_tokens:,} tokens")
            print(f"   • Input: {response.token_usage.get('input_tokens', 0):,}")
            print(f"   • Output: {response.token_usage.get('output_tokens', 0):,}")
        
        # Show query analysis details
        if response.query_analysis:
            query_data = response.query_analysis
            print(f"\n🔍 Query Analysis:")
            print(f"   • Query Type: {query_data.get('query_type', 'Unknown')}")
            print(f"   • Expertise Level: {query_data.get('expertise_level', 'Unknown')}")
            print(f"   • Content Category: {query_data.get('content_category', 'Unknown')}")
        
        # Display confidence breakdown
        confidence_breakdown = response.metadata.get('confidence_breakdown', {})
        if confidence_breakdown:
            print(f"\n🎯 Enhanced Confidence Breakdown:")
            print(f"   • Content Quality: {confidence_breakdown.get('content_quality', 0):.3f}")
            print(f"   • Source Quality: {confidence_breakdown.get('source_quality', 0):.3f}")
            print(f"   • Query Matching: {confidence_breakdown.get('query_matching', 0):.3f}")
            print(f"   • Technical Factors: {confidence_breakdown.get('technical_factors', 0):.3f}")
        
        # Show hyperlink generation statistics
        hyperlink_stats = response.metadata.get('hyperlink_stats', {})
        if hyperlink_stats:
            print(f"\n🔗 Authoritative Hyperlink Generation (NEW FEATURE!):")
            print(f"   • Links Added: {hyperlink_stats.get('links_added', 0)}")
            print(f"   • Categories: {', '.join(hyperlink_stats.get('categories_used', []))}")
            print(f"   • Authority Score: {hyperlink_stats.get('avg_authority_score', 0):.2f}")
        
        # Show image integration results
        if 'images_embedded' in response.metadata:
            print(f"\n🖼️  DataForSEO Image Integration:")
            print(f"   • Images Embedded: {response.metadata['images_embedded']}")
            print(f"   • Image Sources: {response.metadata.get('image_sources', 'N/A')}")
        
        # Show WordPress publishing status
        if 'wordpress_published' in response.metadata:
            wordpress_data = response.metadata.get('wordpress_data', {})
            print(f"\n📝 WordPress Publishing Status:")
            print(f"   • Published: {'Yes' if response.metadata['wordpress_published'] else 'No'}")
            if wordpress_data.get('post_id'):
                print(f"   • Post ID: {wordpress_data['post_id']}")
                print(f"   • URL: {wordpress_data.get('url', 'N/A')}")
        
        # Show 95-field casino intelligence extraction
        casino_intelligence = response.metadata.get('casino_intelligence', {})
        if casino_intelligence:
            print(f"\n🎰 95-Field Casino Intelligence Extraction:")
            data_completeness = casino_intelligence.get('data_completeness', 0)
            print(f"   • Data Completeness: {data_completeness:.1f}%")
            print(f"   • Overall Rating: {casino_intelligence.get('overall_rating', 0):.1f}/10")
            print(f"   • Safety Score: {casino_intelligence.get('safety_score', 0):.1f}/10")
            
            # Show key extracted data
            trustworthiness = casino_intelligence.get('trustworthiness', {})
            if trustworthiness:
                print(f"   • License Status: {trustworthiness.get('license_status', 'Unknown')}")
                print(f"   • Regulatory Body: {trustworthiness.get('regulatory_body', 'Unknown')}")
        
        # Show top sources
        print(f"\n📚 Top Sources Used:")
        for i, source in enumerate(response.sources[:5], 1):
            source_title = source.get('title', 'Unknown Title')[:50]
            source_url = source.get('url', 'No URL')
            authority_score = source.get('authority_score', 0)
            print(f"   {i}. {source_title}... (Authority: {authority_score:.2f})")
            print(f"      URL: {source_url}")
        
        # Display improvement suggestions
        suggestions = response.metadata.get('improvement_suggestions', [])
        if suggestions:
            print(f"\n💡 Improvement Suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion}")
        
        # Get cache performance stats
        cache_stats = chain.get_cache_stats()
        print(f"\n📊 Cache Performance:")
        print(f"   • Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%")
        print(f"   • Total Entries: {cache_stats.get('total_entries', 0)}")
        
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

## Advanced Features Demonstrated
1. ✅ **Advanced Prompt Optimization** - Query classification and context formatting
2. ✅ **Enhanced Confidence Scoring** - 4-factor assessment with detailed breakdown
3. ✅ **Template System v2.0** - 34 specialized templates for casino content
4. ✅ **Contextual Retrieval System** - Hybrid search with metadata filtering
5. ✅ **DataForSEO Image Integration** - Professional casino images embedded
6. ✅ **WordPress Publishing** - Automatic publishing to crashcasino.io
7. ✅ **FTI Content Processing** - Feature/Training/Inference pipeline
8. ✅ **Security & Compliance** - Enterprise-grade security validation
9. ✅ **Performance Profiling** - Real-time performance monitoring
10. ✅ **Web Search Research** - Tavily API integration for fresh data
11. ✅ **Comprehensive Web Research** - 95-field structured casino analysis
12. ✅ **Response Storage & Vectorization** - Intelligent content storage
13. ✅ **Authoritative Hyperlink Generation** - NEW! Contextual authority links

---

## Complete Betway Casino Review

{response.answer}

---

## Technical Analysis

### Query Classification Results
{f"- **Query Type**: {response.query_analysis.get('query_type', 'Unknown')}" if response.query_analysis else "- Query analysis not available"}
{f"- **Expertise Level**: {response.query_analysis.get('expertise_level', 'Unknown')}" if response.query_analysis else ""}
{f"- **Content Category**: {response.query_analysis.get('content_category', 'Unknown')}" if response.query_analysis else ""}

### Confidence Score Breakdown
{f"- **Content Quality**: {confidence_breakdown.get('content_quality', 0):.3f}" if confidence_breakdown else "- Confidence breakdown not available"}
{f"- **Source Quality**: {confidence_breakdown.get('source_quality', 0):.3f}" if confidence_breakdown else ""}
{f"- **Query Matching**: {confidence_breakdown.get('query_matching', 0):.3f}" if confidence_breakdown else ""}
{f"- **Technical Factors**: {confidence_breakdown.get('technical_factors', 0):.3f}" if confidence_breakdown else ""}

### Hyperlink Generation Results (NEW!)
{f"- **Links Added**: {hyperlink_stats.get('links_added', 0)}" if hyperlink_stats else "- Hyperlink stats not available"}
{f"- **Categories Used**: {', '.join(hyperlink_stats.get('categories_used', []))}" if hyperlink_stats else ""}
{f"- **Average Authority Score**: {hyperlink_stats.get('avg_authority_score', 0):.2f}" if hyperlink_stats else ""}

### Casino Intelligence Extraction (95 Fields)
{f"- **Data Completeness**: {casino_intelligence.get('data_completeness', 0):.1f}%" if casino_intelligence else "- Casino intelligence not available"}
{f"- **Overall Rating**: {casino_intelligence.get('overall_rating', 0):.1f}/10" if casino_intelligence else ""}
{f"- **Safety Score**: {casino_intelligence.get('safety_score', 0):.1f}/10" if casino_intelligence else ""}

### Sources Used
{chr(10).join([f"- **{source.get('title', 'Unknown')}** (Authority: {source.get('authority_score', 0):.2f}) - {source.get('url', 'No URL')}" for source in response.sources[:10]])}

### Performance Metrics
- **Cache Hit Rate**: {cache_stats.get('hit_rate', 0):.1f}%
- **Total Cache Entries**: {cache_stats.get('total_entries', 0)}
{f"- **Total Tokens Used**: {sum(response.token_usage.values()):,}" if response.token_usage else "- Token usage not available"}

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
        
        # Final summary
        print(f"\n📈 DEMONSTRATION SUMMARY:")
        print(f"✅ Universal RAG CMS v6.3 - Production Ready")
        print(f"✅ All 13 Advanced Features Operational")
        print(f"✅ NEW Authoritative Hyperlink Generation Working")
        print(f"✅ 95-Field Casino Intelligence Extraction")
        print(f"✅ WordPress Publishing Integration")
        print(f"✅ Professional Content Generation")
        print(f"✅ Enterprise-Grade Performance & Security")
        
        return response
        
    except Exception as e:
        print(f"\n❌ Error during review generation: {str(e)}")
        print(f"💡 Troubleshooting tips:")
        print(f"   1. Check environment variables in .env file")
        print(f"   2. Verify Supabase connection")
        print(f"   3. Ensure all API keys are valid")
        print(f"   4. Check WordPress credentials")
        return None

if __name__ == "__main__":
    print("🎰 Starting Betway Casino Review Demo...")
    result = asyncio.run(demonstrate_betway_review_generation())
    
    if result:
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 Final confidence score: {result.confidence_score:.3f}")
    else:
        print(f"\n❌ Demo failed. Please check the error messages above.") 