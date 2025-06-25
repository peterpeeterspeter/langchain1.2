#!/usr/bin/env python3
"""
🎰 BETWAY CASINO REVIEW - LIVE WORDPRESS PUBLISHING
Universal RAG CMS v6.3 - PUBLISH TO CRASHCASINO.IO
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import the Universal RAG Chain
from chains.universal_rag_lcel import create_universal_rag_chain

async def publish_betway_review_live():
    """
    Generate and publish a comprehensive Betway casino review to WordPress
    """
    print("🎰 BETWAY CASINO REVIEW - LIVE WORDPRESS PUBLISHING")
    print("=" * 60)
    
    # WordPress credentials from memory
    wordpress_config = {
        'site_url': 'https://www.crashcasino.io',
        'username': 'nmlwh',
        'app_password': 'your-wordpress-password-here'
    }
    
    print(f"📝 Target WordPress Site: {wordpress_config['site_url']}")
    print(f"👤 Publishing as: {wordpress_config['username']}")
    
    # Create production chain with WordPress publishing enabled
    print("🚀 Initializing Universal RAG Chain with WordPress publishing...")
    chain = create_universal_rag_chain(
        enable_hyperlink_generation=True,  # ✅ Hyperlinks enabled
        enable_wordpress_publishing=True,  # ✅ WordPress enabled  
        enable_comprehensive_web_research=True,  # ✅ 95-field analysis
        enable_dataforseo_images=True,     # ✅ Images enabled
        model_name="gpt-4.1-mini",
        temperature=0.1
    )
    
    # Initialize WordPress properly
    if hasattr(chain, 'wordpress_publisher') and chain.wordpress_publisher:
        try:
            # Configure WordPress manually if auto-init failed
            chain.wordpress_publisher.site_url = wordpress_config['site_url']
            chain.wordpress_publisher.username = wordpress_config['username'] 
            chain.wordpress_publisher.app_password = wordpress_config['app_password']
            print("✅ WordPress credentials configured manually")
        except Exception as e:
            print(f"⚠️ WordPress configuration warning: {e}")
    
    # Generate comprehensive Betway casino review
    query = """Create a comprehensive Betway casino review for 2025 covering all essential aspects:
    - Licensing and regulatory compliance 
    - Game selection and software providers
    - Bonus offers and promotional terms
    - Payment methods and withdrawal speeds
    - Security features and player protection
    - Mobile compatibility and user experience
    - Responsible gambling tools
    - Overall rating and recommendation
    
    Include actionable insights, pros/cons, and FAQ section."""
    
    print(f"📝 Query: {query[:100]}...")
    print("⚡ Processing with ALL 13 features including hyperlink generation...")
    
    start_time = datetime.now()
    
    try:
        # Generate the comprehensive review
        response = await chain.ainvoke({'query': query})
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ GENERATION COMPLETE!")
        print(f"📄 Content Length: {len(response.answer):,} characters")
        print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📚 Sources Used: {len(response.sources)}")
        
        # Check for hyperlinks
        hyperlink_count = response.answer.count('<a href="http')
        print(f"🔗 Hyperlinks Embedded: {hyperlink_count}")
        
        # Show sample hyperlinks
        if hyperlink_count > 0:
            print("\n🔗 SAMPLE HYPERLINKS:")
            import re
            links = re.findall(r'<a href="(https?://[^"]+)"[^>]*>([^<]+)</a>', response.answer)
            for i, (url, text) in enumerate(links[:5], 1):
                print(f"  {i}. [{text}]({url})")
        
        # Save the article locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"betway_published_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.answer)
        
        print(f"\n💾 Article saved locally: {filename}")
        
        # Check if WordPress publishing succeeded
        wordpress_published = False
        if response.metadata.get('wordpress_published'):
            wordpress_url = response.metadata.get('wordpress_url', '')
            print(f"\n🚀 WORDPRESS PUBLISHING SUCCESS!")
            print(f"📝 Article URL: {wordpress_url}")
            print(f"🏷️ Categories: {response.metadata.get('wordpress_categories', [])}")
            print(f"🔖 Tags: {response.metadata.get('wordpress_tags', [])}")
            wordpress_published = True
        else:
            print(f"\n⚠️ WordPress publishing was not completed")
            print(f"📄 Article generated successfully but not published to WordPress")
            print(f"💡 Reason: WordPress configuration may need adjustment")
        
        # Summary
        print(f"\n" + "="*60)
        print(f"🎯 BETWAY CASINO REVIEW SUMMARY")
        print(f"="*60)
        print(f"✅ Content Generated: {len(response.answer):,} characters")
        print(f"✅ Hyperlinks Added: {hyperlink_count}")
        print(f"✅ Processing Time: {processing_time:.2f}s")
        print(f"✅ Confidence Score: {response.confidence_score:.3f}")
        print(f"{'✅' if wordpress_published else '⚠️'} WordPress Published: {'YES' if wordpress_published else 'NO'}")
        print(f"📁 Local File: {filename}")
        
        return response
        
    except Exception as e:
        print(f"\n❌ ERROR during generation: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(publish_betway_review_live()) 