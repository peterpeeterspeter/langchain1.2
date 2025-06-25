#!/usr/bin/env python3
"""
🎰 TRUSTDICE CASINO REVIEW - LIVE WORDPRESS PUBLISHING
Publishes comprehensive TrustDice casino review with all 13 Universal RAG CMS features to crashcasino.io
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

async def publish_trustdice_to_crashcasino():
    """
    Generate and publish a comprehensive TrustDice casino review to crashcasino.io
    """
    print("🎰 TRUSTDICE CASINO REVIEW - LIVE WORDPRESS PUBLISHING")
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
    trustdice_query = """Create a comprehensive, professional TrustDice casino review for 2025 that covers:

    LICENSING & REGULATION:
    - Curacao eGaming license and regulatory compliance
    - Player protection measures and legal status
    - Cryptocurrency-specific regulations and compliance
    
    GAME SELECTION & SOFTWARE:
    - Detailed game portfolio analysis (slots, table games, live casino, crash games)
    - Software provider partnerships (Pragmatic Play, Evolution Gaming, NetEnt, etc.)
    - Provably fair games and blockchain verification
    - Mobile compatibility and performance
    
    CRYPTOCURRENCY FEATURES:
    - Supported cryptocurrencies (Bitcoin, Ethereum, Litecoin, etc.)
    - Cryptocurrency deposit and withdrawal processes
    - Blockchain transaction verification
    - Anonymous playing capabilities
    
    BONUSES & PROMOTIONS:
    - Welcome bonus package with exact terms and conditions
    - Cryptocurrency-specific bonuses and rewards
    - VIP program and loyalty rewards
    - Wagering requirements and bonus policies
    
    BANKING & PAYMENTS:
    - Complete cryptocurrency payment analysis
    - Traditional payment methods available
    - Transaction processing times and fees
    - Security measures for crypto transactions
    
    SECURITY & FAIRNESS:
    - SSL encryption and blockchain security
    - Provably fair game verification
    - Responsible gambling tools
    - Account security features
    
    USER EXPERIENCE:
    - Website design and cryptocurrency integration
    - Mobile app performance and features
    - Customer support quality and availability
    - Account registration and verification process
    
    PROS & CONS:
    - Balanced assessment of strengths and weaknesses
    - Comparison with other crypto casinos
    - Value proposition for crypto players
    
    FINAL VERDICT:
    - Overall rating out of 10 with detailed justification
    - Target audience recommendations for crypto enthusiasts
    - Key takeaways and actionable insights
    
    Include FAQ section addressing common crypto casino concerns. Make it comprehensive, accurate, and engaging for cryptocurrency players and traditional gamblers alike."""
    
    print(f"\n📝 Generating comprehensive TrustDice casino review...")
    print(f"🔍 Query Length: {len(trustdice_query):,} characters")
    print("⚡ Processing with all advanced features...")
    
    start_time = datetime.now()
    
    try:
        # Generate the comprehensive review
        response = await chain.ainvoke({'query': trustdice_query})
        
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
            article_title = "TrustDice Casino Review 2025: Complete Crypto Casino Analysis"
        
        print(f"\n📰 Article Title: {article_title}")
        
        # Save locally first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"trustdice_wordpress_{timestamp}.md"
        
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
                         # Use casino intelligence publishing for better categorization with MT Casino integration
             wp_result = await wp_integration.publish_casino_intelligence_content(
                 query=trustdice_query,
                 rag_response=response.answer,
                 structured_casino_data=response.metadata.get('structured_casino_data'),
                 title=article_title,
                 featured_image_query="TrustDice casino cryptocurrency bitcoin"
             )
            
            wp_end_time = datetime.now()
            wp_processing_time = (wp_end_time - wp_start_time).total_seconds()
            
                         if wp_result and wp_result.get('success'):
                 wordpress_url = wp_result.get('post_url', '')
                 post_id = wp_result.get('post_id', '')
                 
                 print(f"\n🎉 WORDPRESS PUBLISHING SUCCESS!")
                 print(f"📝 Article URL: {wordpress_url}")
                 print(f"🆔 Post ID: {post_id}")
                 print(f"📊 Publishing Time: {wp_processing_time:.2f} seconds")
                 print(f"🏷️ Categories: {wp_result.get('categories', [])}")
                 print(f"🔖 Tags: {wp_result.get('tags', [])}")
                 print(f"🎰 MT Casino Used: {wp_result.get('mt_casino_features_used', False)}")
                 
                 if wp_result.get('featured_image_id'):
                     print(f"🖼️ Featured Image ID: {wp_result.get('featured_image_id')}")
                
                # Show image upload results
                if wp_result.get('image_upload_results'):
                    successful_uploads = [r for r in wp_result['image_upload_results'] if r.get('success')]
                    print(f"📸 Images Uploaded: {len(successful_uploads)}/{len(wp_result['image_upload_results'])}")
                    
                    for i, result in enumerate(successful_uploads[:5], 1):
                        print(f"  {i}. ID {result.get('id')}: {result.get('title', 'Unknown')}")
                
                print(f"\n🌐 LIVE ARTICLE: {wp_result.get('url')}")
                print(f"📱 Mobile Version: {wp_result.get('url')}?mobile=1")
                
                # Calculate total performance metrics
                total_time = processing_time + wp_processing_time
                words_per_second = len(response.answer.split()) / total_time
                
                print(f"\n📊 PERFORMANCE METRICS:")
                print(f"  📝 Content Generation: {processing_time:.2f}s")
                print(f"  🌐 WordPress Publishing: {wp_processing_time:.2f}s")
                print(f"  ⚡ Total Time: {total_time:.2f}s")
                print(f"  📈 Words/Second: {words_per_second:.1f}")
                print(f"  🎯 Success Rate: 100%")
                
                                 return {
                     'success': True,
                     'wordpress_url': wordpress_url,
                     'post_id': post_id,
                     'local_file': local_filename,
                     'content_length': len(response.answer),
                     'hyperlinks': hyperlink_count,
                     'confidence_score': response.confidence_score,
                     'processing_time': processing_time,
                     'publishing_time': wp_processing_time,
                     'mt_casino_used': wp_result.get('mt_casino_features_used', False)
                 }
                
            else:
                error_msg = wp_result.get('error', 'Unknown error') if wp_result else 'No response from WordPress'
                print(f"\n❌ WORDPRESS PUBLISHING FAILED!")
                print(f"🔍 Error: {error_msg}")
                print(f"⏱️ Attempted Time: {wp_processing_time:.2f} seconds")
                return None
                
        except Exception as wp_error:
            print(f"\n❌ WORDPRESS PUBLISHING ERROR!")
            print(f"🔍 Exception: {str(wp_error)}")
            print(f"📝 Content was saved locally as: {local_filename}")
            return None
        
    except Exception as e:
        print(f"\n❌ CONTENT GENERATION FAILED!")
        print(f"🔍 Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Run the TrustDice casino review and WordPress publishing
    result = asyncio.run(publish_trustdice_to_crashcasino())
    
    if result and result.get('success'):
        print(f"\n🎉 SUCCESS! TrustDice article published at: {result.get('wordpress_url')}")
        print(f"🎰 MT Casino Integration: {'✅' if result.get('mt_casino_used') else '❌'}")
    else:
        print(f"\n💥 FAILED! Error: {result.get('error') if result else 'Unknown error'}") 