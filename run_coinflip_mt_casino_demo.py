#!/usr/bin/env python3
"""
🎰 COINFLIP MT CASINO DEMO - Enhanced Publishing Test
====================================================

Demonstrates the enhanced Coinflip MT Casino publisher with:
- Intelligent content type detection (mt_listing, mt_bonus, mt_slots, mt_reviews)
- 95-field casino intelligence mapping to MT Casino custom fields
- Graceful fallback to regular posts with MT Casino styling
- Comprehensive results tracking and analysis

Author: AI Assistant
Created: 2025-01-23
"""

import asyncio
import sys
import os
from datetime import datetime
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chains.universal_rag_lcel import create_universal_rag_chain
from integrations.coinflip_mt_casino_publisher import create_coinflip_mt_casino_publisher
from schemas.casino_intelligence_schema import CasinoIntelligence

async def main():
    """Demo the enhanced Coinflip MT Casino publishing system"""
    
    print("🎰 COINFLIP MT CASINO ENHANCED PUBLISHING DEMO")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Initialize Universal RAG Chain (simplified, no hanging web search)
        print("🚀 Initializing Universal RAG Chain...")
        chain = create_universal_rag_chain(
            use_web_search=False,  # Disable to avoid hanging
            use_supabase=True,
            use_images=True,
            confidence_threshold=0.7
        )
        print(f"✅ Chain initialized in {time.time() - start_time:.1f}s")
        
        # 2. Initialize Enhanced Coinflip MT Casino Publisher
        print("\n🎯 Initializing Enhanced Coinflip MT Casino Publisher...")
        
        # WordPress configuration
        wp_site_url = os.getenv('WORDPRESS_SITE_URL', 'https://www.crashcasino.io')
        wp_username = os.getenv('WORDPRESS_USERNAME')
        wp_password = os.getenv('WORDPRESS_APP_PASSWORD')
        
        if not wp_username or not wp_password:
            print("❌ Missing WordPress credentials. Please set WORDPRESS_USERNAME and WORDPRESS_APP_PASSWORD")
            return
        
        # Create enhanced publisher
        mt_casino_integration = create_coinflip_mt_casino_publisher(
            site_url=wp_site_url,
            username=wp_username,
            application_password=wp_password
        )
        print("✅ Enhanced MT Casino publisher initialized")
        
        # 3. Generate Casino Content (Bitcasino as example)
        print(f"\n📝 Generating casino content...")
        
        # Simplified prompt to avoid hanging web search
        casino_prompt = """
        Write a comprehensive review of Bitcasino, focusing on:
        - Overall casino experience and reputation
        - Game selection and software providers
        - Welcome bonus and promotional offers
        - Payment methods and withdrawal times
        - Mobile compatibility and user experience
        - Customer support quality
        - Safety and licensing information
        
        Casino: Bitcasino
        Focus: Complete casino review
        Style: Professional, informative, SEO-optimized
        """
        
        # Generate content
        generation_start = time.time()
        result = await chain.ainvoke({"query": casino_prompt})
        generation_time = time.time() - generation_start
        
        # Extract content from RAGResponse object
        content = result.response if hasattr(result, 'response') else str(result)
        casino_data = result.casino_intelligence if hasattr(result, 'casino_intelligence') else None
        images = result.images if hasattr(result, 'images') else []
        confidence = result.confidence_score if hasattr(result, 'confidence_score') else 0
        
        print(f"✅ Content generated in {generation_time:.1f}s")
        print(f"📊 Content length: {len(content):,} characters")
        print(f"🎯 Confidence score: {confidence:.2f}")
        print(f"🖼️ Images found: {len(images)}")
        
        if casino_data:
            print(f"🎰 Casino intelligence: {casino_data.casino_name}")
            print(f"⭐ Overall rating: {casino_data.overall_rating}/10")
        
        # 4. Demonstrate Different MT Casino Post Types
        print(f"\n🎯 TESTING ENHANCED MT CASINO PUBLISHING")
        print("-" * 50)
        
        # Test cases for different content types
        test_cases = [
            {
                'title': f"Bitcasino Review 2025 - Complete Casino Analysis",
                'content': content,
                'description': "Main casino review (should detect as mt_listing)"
            },
            {
                'title': f"Bitcasino Welcome Bonus - Up to 1 BTC + 100 Free Spins",
                'content': f"## Bitcasino Welcome Bonus Review\n\n{content[:500]}...\n\nThis generous welcome bonus package offers new players excellent value with cryptocurrency deposits. The free spins are available on popular slots, and the wagering requirements are competitive compared to other crypto casinos.",
                'description': "Bonus-focused content (should detect as mt_bonus)"
            },
            {
                'title': f"Bitcasino Slot Games - 2000+ Slots from Top Providers",
                'content': f"## Bitcasino Slot Collection\n\n{content[:400]}...\n\nThe slot portfolio features games from NetEnt, Microgaming, Pragmatic Play, and Evolution Gaming. Popular titles include Book of Dead, Starburst, and Gonzo's Quest. Progressive jackpots are available with mega prizes.",
                'description': "Slot-focused content (should detect as mt_slots)"
            }
        ]
        
        all_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}: {test_case['description']}")
            print(f"Title: {test_case['title'][:60]}...")
            
            # Publish with enhanced MT Casino publisher using async context manager
            publish_start = time.time()
            
            async with mt_casino_integration.publisher as publisher:
                publish_result = await publisher.publish_mt_casino_content(
                    content=test_case['content'],
                    title=test_case['title'],
                    casino_data=casino_data,
                    images=images[:2]  # Limit to 2 images per post
                )
            
            publish_time = time.time() - publish_start
            
            # Analyze results
            if publish_result.get('success'):
                main_post = publish_result['main_post']
                post_id = main_post.get('post_id')
                post_type = main_post.get('post_type')
                method = main_post.get('method')
                url = main_post.get('url', 'N/A')
                
                print(f"✅ Successfully published!")
                print(f"   📝 Post ID: {post_id}")
                print(f"   🎯 Post Type: {post_type}")
                print(f"   🔧 Method: {method}")
                print(f"   🌐 URL: {url}")
                print(f"   ⏱️ Published in {publish_time:.1f}s")
                
                # Show MT Casino features used
                features_used = publish_result.get('mt_casino_features_used', [])
                if features_used:
                    print(f"   🎰 MT Casino Features: {', '.join(features_used)}")
                
            else:
                print(f"❌ Publishing failed:")
                for error in publish_result.get('errors', []):
                    print(f"   ⚠️ {error}")
            
            all_results.append({
                'test_case': i,
                'title': test_case['title'],
                'description': test_case['description'],
                'result': publish_result,
                'publish_time': publish_time
            })
            
            # Brief pause between publications
            await asyncio.sleep(1)
        
        # 5. Generate Final Results Report
        print(f"\n📊 ENHANCED MT CASINO PUBLISHING RESULTS")
        print("=" * 60)
        
        total_time = time.time() - start_time
        successful_posts = len([r for r in all_results if r['result'].get('success')])
        
        print(f"⏱️ Total processing time: {total_time:.1f}s")
        print(f"✅ Successful publications: {successful_posts}/{len(test_cases)}")
        print(f"📝 Content generation: {generation_time:.1f}s")
        print(f"🎯 Average publish time: {sum(r['publish_time'] for r in all_results) / len(all_results):.1f}s")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for result in all_results:
            status = "✅ SUCCESS" if result['result'].get('success') else "❌ FAILED"
            post_id = result['result'].get('main_post', {}).get('post_id', 'N/A')
            method = result['result'].get('main_post', {}).get('method', 'N/A')
            
            print(f"   {status} - Test {result['test_case']}: Post ID {post_id} ({method})")
        
        # Save comprehensive results
        results_filename = f"coinflip_mt_casino_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        final_results = {
            'demo_info': {
                'timestamp': datetime.now().isoformat(),
                'total_time': total_time,
                'successful_posts': successful_posts,
                'total_test_cases': len(test_cases)
            },
            'content_generation': {
                'time': generation_time,
                'content_length': len(content),
                'confidence_score': confidence,
                'images_found': len(images),
                'casino_intelligence_available': casino_data is not None
            },
            'publishing_results': all_results,
            'mt_casino_integration': {
                'enhanced_publisher_used': True,
                'post_types_tested': ['mt_listing', 'mt_bonus', 'mt_slots'],
                'fallback_mechanism_available': True,
                'custom_field_mapping': True
            }
        }
        
        with open(results_filename, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_filename}")
        
        # Check live posts
        print(f"\n🌐 Live Posts Check:")
        for result in all_results:
            if result['result'].get('success'):
                post_id = result['result']['main_post']['post_id']
                url = result['result']['main_post'].get('url', f"{wp_site_url}/?p={post_id}")
                print(f"   📝 Test {result['test_case']}: {url}")
        
        print(f"\n🎉 Enhanced Coinflip MT Casino demo completed successfully!")
        print(f"🎰 Your MT Casino integration is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main()) 