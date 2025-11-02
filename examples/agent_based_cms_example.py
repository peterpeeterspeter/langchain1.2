#!/usr/bin/env python3
"""
Agent-Based CMS Example
Demonstrates complete workflow: Research → Writing → Affiliate Links → Images → Publishing
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.factory import create_agent_based_cms
from agents.state import ArticleCMSState


async def main():
    """Run complete Agent-Based CMS workflow"""
    
    print("🚀 Agent-Based CMS - Complete Workflow Example")
    print("=" * 60)
    
    # Configuration
    query = "Betway Casino Review 2025"
    target_sites = ["crashcasino"]  # Site IDs from registry
    
    print(f"📝 Query: {query}")
    print(f"🌐 Target Sites: {', '.join(target_sites)}")
    print()
    
    # Create CMS orchestrator
    print("⚙️  Initializing Agent-Based CMS...")
    cms = create_agent_based_cms(
        llm_model="gpt-4o-mini",
        temperature=0.2,
        enable_research=True,
        enable_writing=True,
        enable_affiliate=True,
        enable_images=True,
        enable_publishing=True,
        max_affiliate_links=5,
        max_images=5
    )
    
    print("✅ CMS initialized with all agents")
    print()
    
    # Run workflow
    print("🔄 Starting CMS Workflow...")
    print("-" * 60)
    
    try:
        final_state = await cms.run(
            query=query,
            target_sites=target_sites
        )
        
        # Display results
        print()
        print("=" * 60)
        print("📊 Workflow Results")
        print("=" * 60)
        
        # Research results
        research_data = final_state.get("research_data", {})
        if research_data:
            print(f"\n🔍 Research:")
            print(f"  - Web results: {len(research_data.get('web_search_results', []))}")
            print(f"  - URLs researched: {len(final_state.get('research_urls', []))}")
            print(f"  - Screenshots: {len(final_state.get('screenshots', []))}")
            print(f"  - Quality score: {research_data.get('research_quality', 0.0):.2f}")
        
        # Writing results
        final_content = final_state.get("final_content", "")
        if final_content:
            print(f"\n✍️  Writing:")
            print(f"  - Content length: {len(final_content)} characters")
            print(f"  - SEO title: {final_state.get('seo_metadata', {}).get('title', 'N/A')}")
            print(f"  - SEO description: {final_state.get('seo_metadata', {}).get('description', 'N/A')[:100]}...")
        
        # Affiliate results
        affiliate_links = final_state.get("affiliate_links", [])
        if affiliate_links:
            print(f"\n🔗 Affiliate Links:")
            print(f"  - Links inserted: {len(affiliate_links)}")
            for link in affiliate_links[:3]:
                print(f"    • {link.get('anchor_text', 'N/A')} → {link.get('url', 'N/A')[:50]}...")
        
        # Image results
        images = final_state.get("images", [])
        if images:
            print(f"\n🖼️  Images:")
            print(f"  - Images selected: {len(images)}")
            print(f"  - WordPress media IDs: {len(final_state.get('wordpress_media_ids', []))}")
        
        # Publishing results
        published_posts = final_state.get("published_posts", [])
        if published_posts:
            print(f"\n📤 Publishing:")
            print(f"  - Sites published: {len(published_posts)}")
            for post in published_posts:
                print(f"    • {post.get('site_name', 'N/A')}: Post ID {post.get('post_id', 'N/A')}")
                print(f"      URL: {post.get('post_url', 'N/A')}")
        else:
            print(f"\n📤 Publishing:")
            print(f"  - No posts published (check site registry and configuration)")
        
        # Errors
        errors = final_state.get("errors", [])
        if errors:
            print(f"\n⚠️  Errors:")
            for error in errors:
                print(f"  - {error}")
        
        # Agent statuses
        agent_statuses = final_state.get("agent_statuses", {})
        print(f"\n🤖 Agent Statuses:")
        for agent_name, status in agent_statuses.items():
            status_icon = "✅" if status == "completed" else "⏱️" if status == "in_progress" else "❌"
            print(f"  {status_icon} {agent_name}: {status}")
        
        print()
        print("=" * 60)
        print("✅ Workflow Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

