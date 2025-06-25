#!/usr/bin/env python3
"""
🎰 BETWAY CASINO REVIEW - LIVE WORDPRESS PUBLISHING
Publishes comprehensive Betway casino review with all 13 Universal RAG CMS features to crashcasino.io
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import the Universal RAG Chain and WordPress integration
from chains.universal_rag_lcel import create_universal_rag_chain
from integrations.wordpress_publisher import WordPressConfig, WordPressIntegration

async def publish_betway_to_crashcasino():
    """
    Generate and publish a comprehensive Betway casino review to crashcasino.io
    """
    print("🎰 BETWAY CASINO REVIEW - LIVE WORDPRESS PUBLISHING")
    print("🌐 Target: crashcasino.io")
    print("=" * 60)
    
    # WordPress credentials for crashcasino.io (from memory)
    wordpress_config = WordPressConfig(
        site_url="https://www.crashcasino.io",
        username="nmlwh",
        application_password="your-wordpress-password-here",
        default_status="publish",  # Publish immediately
        default_author_id=1,
        default_category_ids=[1],  # Default category
        max_concurrent_uploads=3,
        request_timeout=60  # Longer timeout for large content
    )
    
    print(f"📝 WordPress Site: {wordpress_config.site_url}")
    print(f"👤 Publishing User: {wordpress_config.username}")
    print(f"🔐 Authentication: Application Password")
    print(f"📊 Status: {wordpress_config.default_status}")
    
    # Create WordPress integration
    try:
        wp_integration = WordPressIntegration(
            wordpress_config=wordpress_config,
            supabase_client=None  # Will auto-initialize if needed
        )
        print("✅ WordPress integration initialized")
    except Exception as e:
        print(f"❌ WordPress integration failed: {e}")
        return None
    
    # Create Universal RAG Chain with ALL features enabled
    print("\n🚀 Initializing Universal RAG Chain...")
    print("📋 Features: ALL 13 advanced features enabled")
    
    chain = create_universal_rag_chain(
        enable_hyperlink_generation=True,      # ✅ Authoritative hyperlinks
        enable_wordpress_publishing=False,     # ✅ We'll handle manually for better control
        enable_comprehensive_web_research=True, # ✅ 95-field casino analysis
        enable_dataforseo_images=True,         # ✅ Professional images
        enable_template_system_v2=True,       # ✅ Advanced templates
        enable_enhanced_confidence=True,      # ✅ Enhanced confidence scoring
        enable_prompt_optimization=True,      # ✅ Optimized prompts
        enable_contextual_retrieval=True,     # ✅ Smart retrieval
        enable_fti_processing=True,           # ✅ Content processing
        enable_security=True,                 # ✅ Security features
        enable_profiling=True,                # ✅ Performance monitoring
        enable_web_search=True,               # ✅ Tavily web search
        enable_response_storage=True,         # ✅ Response storage
        enable_caching=True,                  # ✅ Smart caching
        model_name="gpt-4o-mini",
        temperature=0.1
    )
    
    print("✅ Universal RAG Chain initialized with all 13 features")
    
    # Comprehensive casino review query
    betway_query = """Create a comprehensive, professional Betway casino review for 2025 that covers:

    LICENSING & REGULATION:
    - Malta Gaming Authority and UK Gambling Commission licenses
    - Regulatory compliance and player protection measures
    - Legal status in different jurisdictions
    
    GAME SELECTION & SOFTWARE:
    - Detailed game portfolio analysis (slots, table games, live casino)
    - Software provider partnerships (NetEnt, Evolution Gaming, Microgaming, etc.)
    - Game quality, RTP rates, and unique offerings
    - Mobile compatibility and performance
    
    BONUSES & PROMOTIONS:
    - Welcome bonus package with exact terms and conditions
    - Ongoing promotions and VIP program details
    - Wagering requirements and bonus abuse prevention
    - Cashback and loyalty rewards
    
    BANKING & PAYMENTS:
    - Complete payment method analysis (cards, e-wallets, crypto)
    - Deposit and withdrawal limits, processing times
    - Transaction fees and currency support
    - Security measures for financial transactions
    
    SECURITY & FAIRNESS:
    - SSL encryption and data protection protocols
    - Game fairness and RNG certification
    - Responsible gambling tools and self-exclusion options
    - Problem gambling support resources
    
    USER EXPERIENCE:
    - Website design, navigation, and functionality
    - Mobile app performance and features
    - Customer support quality and availability
    - Account verification process
    
    PROS & CONS:
    - Balanced assessment of strengths and weaknesses
    - Comparison with competitor casinos
    - Value proposition for different player types
    
    FINAL VERDICT:
    - Overall rating out of 10 with detailed justification
    - Target audience recommendations
    - Key takeaways and actionable insights
    
    Include FAQ section addressing common player concerns. Make it comprehensive, accurate, and engaging for both novice and experienced players."""
    
    print(f"\n📝 Generating comprehensive Betway casino review...")
    print(f"🔍 Query Length: {len(betway_query):,} characters")
    print("⚡ Processing with all advanced features...")
    
    start_time = datetime.now()
    
    try:
        # Generate the comprehensive review
        response = await chain.ainvoke({'query': betway_query})
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ CONTENT GENERATION COMPLETE!")
        print(f"📄 Content Length: {len(response.answer):,} characters")
        print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📚 Sources Used: {len(response.sources)}")
        
        # Check for hyperlinks
        hyperlink_count = response.answer.count('<a href="http')
        print(f"🔗 Authoritative Hyperlinks: {hyperlink_count}")
        
        # Show sample hyperlinks
        if hyperlink_count > 0:
            print("\n🔗 EMBEDDED HYPERLINKS:")
            import re
            links = re.findall(r'<a href="(https?://[^"]+)"[^>]*>([^<]+)</a>', response.answer)
            for i, (url, text) in enumerate(links[:7], 1):
                print(f"  {i}. [{text}]({url})")
        
        # Extract title from content
        title_match = re.search(r'#\s*([^#\n]+)', response.answer)
        if title_match:
            article_title = title_match.group(1).strip()
        else:
            article_title = "Betway Casino Review 2025: Complete Analysis & Rating"
        
        print(f"\n📰 Article Title: {article_title}")
        
        # Save locally first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"betway_wordpress_{timestamp}.md"
        
        with open(local_filename, 'w', encoding='utf-8') as f:
            f.write(response.answer)
        
        print(f"💾 Article saved locally: {local_filename}")
        
        # Extract meta description (first paragraph)
        meta_description = ""
        paragraphs = re.findall(r'<p>([^<]+)</p>', response.answer)
        if paragraphs:
            meta_description = paragraphs[0][:155] + "..." if len(paragraphs[0]) > 155 else paragraphs[0]
        else:
            # Fallback: get first sentence after title
            content_without_title = re.sub(r'^#[^#\n]+\n+', '', response.answer, flags=re.MULTILINE)
            first_sentence = content_without_title.split('.')[0]
            meta_description = (first_sentence[:155] + "...") if len(first_sentence) > 155 else first_sentence
        
        print(f"📝 Meta Description: {meta_description}")
        
        # Publish to WordPress
        print(f"\n🚀 PUBLISHING TO WORDPRESS...")
        print(f"🌐 Publishing to: {wordpress_config.site_url}")
        
        wp_start_time = datetime.now()
        
        try:
            # Use casino intelligence publishing for better categorization
            publish_result = await wp_integration.publish_casino_intelligence_content(
                query=betway_query,
                rag_response=response.answer,
                structured_casino_data=response.metadata.get('structured_casino_data'),
                title=article_title,
                featured_image_query="Betway casino review 2025 professional"
            )
            
            wp_end_time = datetime.now()
            wp_processing_time = (wp_end_time - wp_start_time).total_seconds()
            
            if publish_result.get('success'):
                wordpress_url = publish_result.get('post_url', '')
                post_id = publish_result.get('post_id', '')
                
                print(f"\n🎉 WORDPRESS PUBLISHING SUCCESS!")
                print(f"📝 Article URL: {wordpress_url}")
                print(f"🆔 Post ID: {post_id}")
                print(f"📊 Publishing Time: {wp_processing_time:.2f} seconds")
                print(f"🏷️ Categories: {publish_result.get('categories', [])}")
                print(f"🔖 Tags: {publish_result.get('tags', [])}")
                
                if publish_result.get('featured_image_id'):
                    print(f"🖼️ Featured Image ID: {publish_result.get('featured_image_id')}")
                
                # Final summary
                print(f"\n" + "="*70)
                print(f"🎯 BETWAY CASINO REVIEW - PUBLICATION COMPLETE")
                print(f"="*70)
                print(f"✅ Content Generated: {len(response.answer):,} characters")
                print(f"✅ Hyperlinks Embedded: {hyperlink_count}")
                print(f"✅ Generation Time: {processing_time:.2f}s")
                print(f"✅ Publishing Time: {wp_processing_time:.2f}s")
                print(f"✅ Total Time: {(processing_time + wp_processing_time):.2f}s")
                print(f"✅ Confidence Score: {response.confidence_score:.3f}")
                print(f"✅ WordPress Published: YES")
                print(f"✅ Live URL: {wordpress_url}")
                print(f"📁 Local Backup: {local_filename}")
                
                return {
                    'success': True,
                    'wordpress_url': wordpress_url,
                    'post_id': post_id,
                    'local_file': local_filename,
                    'content_length': len(response.answer),
                    'hyperlinks': hyperlink_count,
                    'confidence_score': response.confidence_score,
                    'processing_time': processing_time,
                    'publishing_time': wp_processing_time
                }
                
            else:
                error_msg = publish_result.get('error', 'Unknown publication error')
                print(f"\n❌ WORDPRESS PUBLISHING FAILED")
                print(f"💥 Error: {error_msg}")
                print(f"📄 Content generated successfully but not published")
                print(f"💾 Local file available: {local_filename}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'local_file': local_filename,
                    'content_length': len(response.answer),
                    'hyperlinks': hyperlink_count,
                    'confidence_score': response.confidence_score
                }
        
        except Exception as wp_error:
            print(f"\n❌ WORDPRESS PUBLISHING ERROR: {wp_error}")
            print(f"📄 Content generated successfully but publishing failed")
            print(f"💾 Local file available: {local_filename}")
            
            return {
                'success': False,
                'error': str(wp_error),
                'local_file': local_filename,
                'content_length': len(response.answer),
                'hyperlinks': hyperlink_count,
                'confidence_score': response.confidence_score
            }
    
    except Exception as e:
        print(f"\n❌ CONTENT GENERATION ERROR: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🚀 Starting Betway Casino Review Publication...")
    result = asyncio.run(publish_betway_to_crashcasino())
    
    if result and result.get('success'):
        print(f"\n🎉 SUCCESS! Article published at: {result.get('wordpress_url')}")
    else:
        print(f"\n💥 FAILED! Error: {result.get('error') if result else 'Unknown error'}") 