#!/usr/bin/env python3
"""
Complete Production Test - Fixed with Proper LCEL + LangGraph Integration
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Set all credentials
os.environ["WORDPRESS_SITE_URL"] = "https://crashcasino.io"
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_APP_PASSWORD"] = "NTve VyqU PF1J BSCF 4F41 pnrW"
os.environ["WORDPRESS_PASSWORD"] = "NTve VyqU PF1J BSCF 4F41 pnrW"
os.environ["WORDPRESS_VERIFY_SSL"] = "false"  # Disable SSL verification
os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzYzNzY0NiwiZXhwIjoyMDYzMjEzNjQ2fQ.ZSgK7qEdhCUkbAcAgeeDz23t-TrkX_m7H9O-WH5z5xs"
os.environ["SUPABASE_ANON_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc2Mzc2NDYsImV4cCI6MjA2MzIxMzY0Nn0.3H8N2Fk22RAV1gHzDB5pCi9GokGwroG34v15I5Cq8_g"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def run_production():
    """Run complete production workflow"""
    
    print("=" * 80)
    print("🎰 COINCASINO PRODUCTION - LangChain LCEL + LangGraph")
    print("=" * 80)
    print()
    
    try:
        from src.agents.factory import create_agent_based_cms
        
        print("🔧 Initializing Production CMS with LCEL + LangGraph...")
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
        
        test_query = "Coincasino Review 2025"
        target_sites = ["crashcasino"]  # Use site_id from registry, not domain
        
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
            len(images) > 0
        )
        
        print("=" * 80)
        if success:
            print("🎉 SUCCESS!")
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

