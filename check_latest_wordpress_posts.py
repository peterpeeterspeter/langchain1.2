#!/usr/bin/env python3
"""
Check Latest WordPress Posts for Images
Finds the most recent posts to see if new ones were created with images
"""

import os
import requests
import json
import re
from requests.auth import HTTPBasicAuth
from datetime import datetime

def check_latest_posts():
    """Check the latest WordPress posts for images"""
    
    # WordPress credentials
    site_url = "https://www.crashcasino.io"
    username = "nmlwh"
    app_password = "your-wordpress-password-here"
    
    print("🔍 CHECKING LATEST WORDPRESS POSTS FOR IMAGES")
    print("=" * 60)
    
    # Get the latest posts
    api_url = f"{site_url}/wp-json/wp/v2/posts"
    params = {
        'per_page': 5,  # Get last 5 posts
        'orderby': 'date',
        'order': 'desc'
    }
    
    try:
        response = requests.get(
            api_url,
            auth=HTTPBasicAuth(username, app_password),
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            posts = response.json()
            
            print(f"📊 Found {len(posts)} recent posts")
            print()
            
            for i, post in enumerate(posts, 1):
                title = post.get('title', {}).get('rendered', 'N/A')
                date = post.get('date', 'N/A')
                post_id = post.get('id', 'N/A')
                status = post.get('status', 'N/A')
                
                # Parse date for better display
                try:
                    parsed_date = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    display_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    display_date = date
                
                print(f"📄 POST {i}: {title}")
                print(f"   ID: {post_id}")
                print(f"   Date: {display_date}")
                print(f"   Status: {status}")
                
                # Check content for images
                content = post.get('content', {}).get('rendered', '')
                img_tags = re.findall(r'<img[^>]*>', content, re.IGNORECASE)
                
                print(f"   🖼️ Images: {len(img_tags)} found")
                
                if img_tags:
                    print(f"   ✅ HAS IMAGES!")
                    # Show first image tag
                    print(f"   Sample: {img_tags[0][:100]}...")
                else:
                    print(f"   ❌ No images")
                
                # Check for Betway content
                if 'betway' in content.lower():
                    print(f"   🎰 Contains 'Betway' content")
                
                print()
                
            # Look specifically for today's Betway posts
            today = datetime.now().strftime('%Y-%m-%d')
            betway_posts_today = [
                post for post in posts 
                if 'betway' in post.get('title', {}).get('rendered', '').lower()
                and today in post.get('date', '')
            ]
            
            if betway_posts_today:
                print("🎰 TODAY'S BETWAY POSTS:")
                for post in betway_posts_today:
                    content = post.get('content', {}).get('rendered', '')
                    img_count = len(re.findall(r'<img[^>]*>', content, re.IGNORECASE))
                    print(f"   📄 {post.get('title', {}).get('rendered', 'N/A')}")
                    print(f"      ID: {post.get('id')}")
                    print(f"      Images: {img_count}")
                    print(f"      URL: {post.get('link', 'N/A')}")
            else:
                print("🔍 No Betway posts found today")
                
        else:
            print(f"❌ Failed to fetch posts: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    check_latest_posts() 