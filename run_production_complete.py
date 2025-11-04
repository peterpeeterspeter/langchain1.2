#!/usr/bin/env python3
"""
Complete Production Test - FIXED WordPress Publishing
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Note: All credentials should be set in environment variables or .env file
# This script will use whatever is available in the environment

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def run_production():
    """Run complete production workflow"""
    
    print("=" * 80)
    print("🎰 CYBET PRODUCTION - COMPLETE WORKFLOW")
    print("=" * 80)
    print()
    
    try:
        from src.agents.factory import create_agent_based_cms
        
        print("🔧 Initializing Production CMS...")
        cms = create_agent_based_cms(
            llm_model="gpt-4o-mini",
            temperature=0.2,
            enable_research=True,
            enable_writing=True,
            enable_affiliate=True,
            enable_images=True,
            enable_publishing=True,
            max_affiliate_links=3,
            max_images=5,
            enable_checkpoints=False
        )
        print("✅ Production CMS initialized")
        print()
        
        test_query = "CyBet Casino Review 2025"
        target_sites = ["crashcasino"]  # ✅ FIXED: Use site_id, not domain
        
        print(f"📝 Query: {test_query}")
        print(f"🌐 Publishing to: {target_sites[0]}")
        print()
        print("🚀 Starting Complete Workflow...")
        print()
        
        start_time = datetime.now()
        
        # Run workflow
        result = await cms.run(
            query=test_query,
            target_sites=target_sites
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print()
        print("=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        print()
        
        print(f"⏱️  Duration: {duration:.2f}s ({duration/60:.1f} min)")
        print()
        
        # Research
        research_data = result.get('research_data', {})
        print("🔍 Research:")
        print(f"   ✅ Quality: {research_data.get('research_quality', 0):.2f}")
        print(f"   ✅ URLs: {len(result.get('research_urls', []))}")
        print(f"   ✅ Screenshots: {len(result.get('screenshots', []))}")
        print()
        
        # Writing
        content = result.get('final_content', '') or result.get('draft_content', '')
        print("✍️  Writing:")
        print(f"   ✅ Content: {len(content)} chars")
        print(f"   ✅ HTML: {'Yes' if '<h1' in content or '<p>' in content else 'No'}")
        print()
        
        # Images
        images = result.get('images', [])
        print("🖼️  Images:")
        print(f"   ✅ Total: {len(images)}")
        wp_media_ids = result.get('wordpress_media_ids', [])
        if wp_media_ids:
            print(f"   ✅ WordPress Media IDs: {wp_media_ids}")
        print()
        
        # Affiliate
        affiliate_links = result.get('affiliate_links', [])
        print("🔗 Affiliate:")
        print(f"   ✅ Links: {len(affiliate_links)}")
        print()
        
        # Publishing
        published_posts = result.get('published_posts', [])
        print("📮 Publishing:")
        if published_posts:
            for post in published_posts:
                print(f"   ✅ Post ID: {post.get('post_id')}")
                print(f"   ✅ URL: {post.get('post_url', 'N/A')}")
                print(f"   ✅ Site: {post.get('site_name', 'N/A')}")
        else:
            print("   ⚠️  No posts published")
        print()
        
        # Errors
        errors = result.get('errors', [])
        if errors:
            print("❌ Errors:")
            for error in errors[:5]:
                print(f"   • {error}")
            print()
        
        # Success check
        success = (
            bool(research_data) and
            len(content) > 1000 and
            len(published_posts) > 0  # ✅ Now checking for published posts
        )
        
        print("=" * 80)
        if success:
            print("🎉 SUCCESS! Article published!")
        else:
            print("⚠️  PARTIAL SUCCESS")
        print("=" * 80)
        
        return success
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(run_production())
    sys.exit(0 if result else 1)

