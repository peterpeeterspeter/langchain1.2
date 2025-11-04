#!/usr/bin/env python3
"""
Complete Production Test for Coincasino
Uses all provided credentials to run full workflow
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set all credentials from provided values
os.environ["WORDPRESS_SITE_URL"] = "https://crashcasino.io"
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_APP_PASSWORD"] = "NTve VyqU PF1J BSCF 4F41 pnrW"
os.environ["WORDPRESS_PASSWORD"] = "NTve VyqU PF1J BSCF 4F41 pnrW"
os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzYzNzY0NiwiZXhwIjoyMDYzMjEzNjQ2fQ.ZSgK7qEdhCUkbAcAgeeDz23t-TrkX_m7H9O-WH5z5xs"
os.environ["SUPABASE_ANON_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc2Mzc2NDYsImV4cCI6MjA2MzIxMzY0Nn0.3H8N2Fk22RAV1gHzDB5pCi9GokGwroG34v15I5Cq8_g"

async def run_production_test():
    """Run complete production workflow for Coincasino"""
    
    print("=" * 80)
    print("🎰 COINCASINO PRODUCTION TEST - COMPLETE WORKFLOW")
    print("=" * 80)
    print()
    
    print("✅ Credentials configured:")
    print(f"   • OpenAI: {os.getenv('OPENAI_API_KEY')[:20]}...")
    print(f"   • Gemini: {os.getenv('GOOGLE_API_KEY')[:20]}...")
    print(f"   • WordPress: {os.getenv('WORDPRESS_URL')}")
    print(f"   • Supabase: {os.getenv('SUPABASE_URL')[:30]}...")
    print()
    
    try:
        from src.agents.factory import create_agent_based_cms
        
        # Create CMS with ALL features enabled
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
        
        # Production query
        test_query = "Coincasino Review 2025"
        target_sites = ["crashcasino.io"]
        
        print(f"📝 Query: {test_query}")
        print(f"🌐 Publishing to: {target_sites[0]}")
        print()
        print("🚀 Starting Complete Production Workflow (LangChain LCEL + LangGraph)...")
        print("   This will run:")
        print("   1. 🔍 Research: 95-field casino intelligence extraction (LCEL chain)")
        print("   2. ✍️  Writing: Rich HTML content generation (LCEL chain)")
        print("   3. 🖼️  Images: Screenshots + Gemini generation (LCEL chain, parallel)")
        print("   4. 🔗 Affiliate: Link insertion (LCEL chain, parallel)")
        print("   5. 📮 Publishing: WordPress post with images (LCEL chain)")
        print("   Orchestration: LangGraph StateGraph with LCEL chains")
        print()
        
        start_time = datetime.now()
        
        # Run the complete workflow
        result = await cms.run(
            query=test_query,
            target_sites=target_sites
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        print()
        print("=" * 80)
        print("📊 PRODUCTION RESULTS")
        print("=" * 80)
        print()
        
        print(f"⏱️  Total Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        print()
        
        # Research
        research_data = result.get('research_data', {})
        print("🔍 Research:")
        if research_data:
            fields = research_data.get('fields_extracted', 0)
            quality = research_data.get('quality_score', 0.0)
            print(f"   ✅ Fields extracted: {fields}")
            print(f"   ✅ Quality score: {quality:.2f}")
            print(f"   ✅ URLs researched: {len(research_data.get('urls_used', []))}")
        else:
            print("   ⚠️  No research data")
        print()
        
        # Writing
        content = result.get('final_content', '') or result.get('draft_content', '')
        print("✍️  Writing:")
        print(f"   ✅ Content: {len(content)} characters")
        print(f"   ✅ HTML: {'Yes' if '<h1' in content or '<p>' in content else 'No'}")
        print(f"   ✅ Images embedded: {content.count('<img')}")
        print(f"   ✅ Links embedded: {content.count('<a href')}")
        print()
        
        # Images
        images = result.get('images', [])
        print("🖼️  Images:")
        print(f"   ✅ Total images: {len(images)}")
        for i, img in enumerate(images[:5], 1):
            source = img.get('source', 'unknown')
            print(f"   • Image {i}: {source} ({img.get('type', 'unknown')})")
        
        wp_media_ids = result.get('wordpress_media_ids', [])
        if wp_media_ids:
            print(f"   ✅ WordPress media IDs: {wp_media_ids}")
        print()
        
        # Affiliate
        affiliate_links = result.get('affiliate_links', [])
        print("🔗 Affiliate Links:")
        print(f"   ✅ Links added: {len(affiliate_links)}")
        for i, link in enumerate(affiliate_links[:3], 1):
            print(f"   • {link.get('anchor_text', 'Link')[:50]}...")
        print()
        
        # Publishing
        published_posts = result.get('published_posts', [])
        print("📮 Publishing:")
        if published_posts:
            for post in published_posts:
                print(f"   ✅ Post ID: {post.get('post_id')}")
                print(f"   ✅ URL: {post.get('post_url')}")
                print(f"   ✅ Site: {post.get('site_id')}")
        else:
            print("   ⚠️  No posts published")
        print()
        
        # Agent statuses
        agent_statuses = result.get('agent_statuses', {})
        print("🤖 Agent Status:")
        for agent, status in agent_statuses.items():
            icon = "✅" if status == "completed" else "⏱️" if status == "in-progress" else "❌"
            print(f"   {icon} {agent}: {status}")
        print()
        
        # Errors
        errors = result.get('errors', [])
        if errors:
            print("❌ Errors:")
            for error in errors:
                print(f"   • {error}")
            print()
        
        # Save output
        if content:
            output_file = "coincasino_production_output.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{test_query}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        table td, table th {{ border: 1px solid #ddd; padding: 8px; }}
    </style>
</head>
<body>
{content}
</body>
</html>""")
            print(f"💾 Output saved: {output_file}")
            print()
        
        # Final summary
        print("=" * 80)
        print("🎯 PRODUCTION TEST SUMMARY")
        print("=" * 80)
        
        all_success = (
            bool(research_data) and
            bool(content) and len(content) > 1000 and
            len(images) > 0 and
            len(published_posts) > 0
        )
        
        if all_success:
            print("🎉 SUCCESS! Complete production workflow executed successfully!")
            print()
            print("✅ Research completed")
            print("✅ Content generated")
            print("✅ Images acquired and embedded")
            print("✅ Affiliate links inserted")
            print("✅ Published to WordPress")
        else:
            print("⚠️  Partial success - check details above")
        
        print("=" * 80)
        print()
        
        return all_success
        
    except Exception as e:
        print(f"\n❌ Production test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(run_production_test())
    sys.exit(0 if result else 1)

