#!/usr/bin/env python3
"""
Check WordPress Publishing Results
Verify if the Betway casino review was successfully published
"""

import sys
import os
import asyncio
import requests
from datetime import datetime

sys.path.append('src')

def check_wordpress_site():
    """Check the WordPress site for recent posts"""
    
    print("🔍 CHECKING: WordPress Publishing Results")
    print("=" * 60)
    
    # WordPress site details
    site_url = os.getenv("WORDPRESS_SITE_URL", "https://www.crashcasino.io")
    username = os.getenv("WORDPRESS_USERNAME", "nmlwh")
    app_password = os.getenv("WORDPRESS_APP_PASSWORD", "")
    
    print(f"🌐 Site: {site_url}")
    print(f"👤 User: {username}")
    print(f"🔑 Password: {'*' * 8}...{app_password[-4:] if app_password else 'NOT_SET'}")
    print()
    
    if not app_password:
        print("❌ No WordPress app password found")
        return
    
    try:
        # Check WordPress REST API for recent posts
        api_url = f"{site_url}/wp-json/wp/v2/posts"
        
        # Use basic auth with application password
        auth = (username, app_password)
        
        print("📡 Fetching recent WordPress posts...")
        
        # Get posts from today
        response = requests.get(
            api_url,
            auth=auth,
            params={
                'per_page': 10,
                'status': 'publish,draft',
                '_fields': 'id,title,status,date,link,excerpt'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            posts = response.json()
            print(f"✅ Found {len(posts)} recent posts")
            print()
            
            # Look for Betway-related posts
            betway_posts = [p for p in posts if 'betway' in p.get('title', {}).get('rendered', '').lower()]
            
            if betway_posts:
                print("🎯 FOUND BETWAY POSTS:")
                for post in betway_posts:
                    print(f"📄 ID: {post['id']}")
                    print(f"🏷️  Title: {post['title']['rendered']}")
                    print(f"📅 Date: {post['date']}")
                    print(f"📊 Status: {post['status']}")
                    print(f"🔗 URL: {post['link']}")
                    print()
                    
                    # Check if it was published today
                    post_date = datetime.fromisoformat(post['date'].replace('Z', '+00:00'))
                    today = datetime.now().date()
                    if post_date.date() == today:
                        print("🎉 ✅ PUBLISHED TODAY! WordPress publishing SUCCESS!")
                        return True
            else:
                print("📋 No Betway posts found in recent posts")
                print("📝 Recent posts:")
                for post in posts[:5]:
                    print(f"   - {post['title']['rendered']}")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Error checking WordPress: {e}")
    
    return False

async def check_universal_rag_metadata():
    """Check if we have metadata from the last Universal RAG Chain run"""
    
    print("\n🔍 CHECKING: Last Chain Execution Metadata")
    print("=" * 60)
    
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create a chain to check if it has recent metadata
        chain = create_universal_rag_chain()
        
        # Check if the chain has stored WordPress metadata
        if hasattr(chain, '_current_wordpress_metadata'):
            wp_meta = chain._current_wordpress_metadata
            print(f"📝 WordPress Metadata: {wp_meta}")
            return wp_meta
        else:
            print("📝 No stored WordPress metadata found")
            
    except Exception as e:
        print(f"❌ Error checking chain metadata: {e}")
    
    return None

if __name__ == "__main__":
    print("🚀 WordPress Publishing Verification")
    print("=" * 80)
    
    # Check WordPress site
    published = check_wordpress_site()
    
    # Check chain metadata
    asyncio.run(check_universal_rag_metadata())
    
    if published:
        print("\n🎉 SUCCESS: Betway casino review published to WordPress!")
    else:
        print("\n❓ Publication status unclear - check WordPress dashboard manually")
        print(f"🔗 Dashboard: https://www.crashcasino.io/wp-admin/") 