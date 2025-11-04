#!/usr/bin/env python3
"""
Fix WordPress Publishing - Debug and Test
"""

import asyncio
import os
import sys
from pathlib import Path

# Set credentials
os.environ["WORDPRESS_SITE_URL"] = "https://crashcasino.io"
os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
os.environ["WORDPRESS_USERNAME"] = "nmlwh"
os.environ["WORDPRESS_APP_PASSWORD"] = "NTve VyqU PF1J BSCF 4F41 pnrW"
os.environ["WORDPRESS_VERIFY_SSL"] = "false"  # Disable SSL verification for testing
os.environ["SUPABASE_URL"] = "https://ambjsovdhizjxwhhnbtd.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzYzNzY0NiwiZXhwIjoyMDYzMjEzNjQ2fQ.ZSgK7qEdhCUkbAcAgeeDz23t-TrkX_m7H9O-WH5z5xs"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_publishing():
    """Test publishing directly"""
    print("=" * 80)
    print("TESTING WORDPRESS PUBLISHING")
    print("=" * 80)
    print()
    
    # Test 1: Check site registry
    print("1. Checking Site Registry...")
    try:
        from src.integrations.wordpress_site_registry import WordPressSiteRegistry
        registry = WordPressSiteRegistry()
        sites = await registry.get_sites(active_only=True)
        print(f"   ✅ Found {len(sites)} sites:")
        for site in sites:
            print(f"      - {site.site_id}: {site.site_name}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Test publishing tool directly
    print("2. Testing Publishing Tool...")
    try:
        from src.agents.tools.publishing_tools import wordpress_publish_tool
        
        # Use correct site_id from registry
        site_id = "crashcasino"  # Use the actual site_id from registry
        
        result = await wordpress_publish_tool.ainvoke({
            "site_id": site_id,
            "title": "Test Post - Coincasino Review 2025",
            "content": "<h1>Test Content</h1><p>This is a test post to verify WordPress publishing is working.</p>",
            "status": "publish",
            "meta_description": "Test post for debugging WordPress publishing"
        })
        
        print(f"   ✅ Publishing result: {result}")
        if result.get("success"):
            print(f"   ✅ Post ID: {result.get('post_id')}")
            print(f"   ✅ Post URL: {result.get('post_url')}")
        else:
            print(f"   ❌ Error: {result.get('error')}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Test with full agent workflow
    print("3. Testing Publishing Agent...")
    try:
        from langchain_openai import ChatOpenAI
        from src.agents.publishing_agent import PublishingAgent
        from src.agents.state import create_initial_state
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        agent = PublishingAgent(llm=llm)
        
        # Create test state with correct site_id
        state = create_initial_state(
            query="Coincasino Review 2025",
            target_sites=["crashcasino"]  # Use correct site_id
        )
        state["final_content"] = "<h1>Coincasino Review 2025</h1><p>This is a comprehensive review...</p>"
        state["seo_metadata"] = {
            "title": "Coincasino Review 2025",
            "description": "Complete review of Coincasino"
        }
        
        result = await agent.execute(state)
        print(f"   ✅ Agent result: success={result.success}")
        print(f"   ✅ Published posts: {len(result.state_updates.get('published_posts', []))}")
        if result.state_updates.get('published_posts'):
            for post in result.state_updates['published_posts']:
                print(f"      - Post ID: {post.get('post_id')}, URL: {post.get('post_url')}")
        if result.error:
            print(f"   ❌ Error: {result.error}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_publishing())

