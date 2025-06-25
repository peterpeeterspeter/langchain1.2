#!/usr/bin/env python3
"""
Check WordPress Post Content for Images
Analyzes the published Betway post to see if images are present
"""

import os
import requests
import json
import re
from requests.auth import HTTPBasicAuth

def check_post_content():
    """Check the WordPress post content for images"""
    
    # WordPress credentials
    site_url = "https://www.crashcasino.io"
    username = "nmlwh"
    app_password = "your-wordpress-password-here"
    
    print("🔍 ANALYZING WORDPRESS POST CONTENT FOR IMAGES")
    print("=" * 60)
    
    # Get the specific Betway post (ID: 51132)
    post_id = 51132
    api_url = f"{site_url}/wp-json/wp/v2/posts/{post_id}"
    
    try:
        response = requests.get(
            api_url,
            auth=HTTPBasicAuth(username, app_password),
            timeout=30
        )
        
        if response.status_code == 200:
            post_data = response.json()
            
            print(f"📄 Post Title: {post_data.get('title', {}).get('rendered', 'N/A')}")
            print(f"📅 Date: {post_data.get('date', 'N/A')}")
            print(f"📊 Status: {post_data.get('status', 'N/A')}")
            print(f"🔗 URL: {post_data.get('link', 'N/A')}")
            print()
            
            # Get the content
            content = post_data.get('content', {}).get('rendered', '')
            
            print(f"📝 Content Length: {len(content)} characters")
            print()
            
            # Check for images
            img_tags = re.findall(r'<img[^>]*>', content, re.IGNORECASE)
            print(f"🖼️ Image Tags Found: {len(img_tags)}")
            
            if img_tags:
                print("✅ IMAGES PRESENT:")
                for i, img in enumerate(img_tags[:3], 1):  # Show first 3
                    print(f"   {i}. {img[:100]}...")
            else:
                print("❌ NO IMAGES FOUND IN CONTENT")
            
            # Check for image-related text
            image_keywords = ['casino', 'betway', 'game', 'slot', 'bonus']
            for keyword in image_keywords:
                if keyword.lower() in content.lower():
                    print(f"   ✅ Contains '{keyword}' (potential image context)")
            
            # Show content preview
            print("\\n📄 CONTENT PREVIEW (first 500 chars):")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            
            # Check for specific image-related patterns
            print("\\n🔍 IMAGE ANALYSIS:")
            print("-" * 30)
            
            # Check for WordPress media URLs
            wp_media_urls = re.findall(r'https?://[^\\s]*crashcasino[^\\s]*\\.(jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
            print(f"📸 WordPress Media URLs: {len(wp_media_urls)}")
            
            # Check for external image URLs
            external_imgs = re.findall(r'https?://[^\\s]*\\.(jpg|jpeg|png|gif|webp)', content, re.IGNORECASE)
            print(f"🌐 External Image URLs: {len(external_imgs)}")
            
            # Check for alt text
            alt_texts = re.findall(r'alt=["\']([^"\']*)["\']', content, re.IGNORECASE)
            print(f"🏷️ Alt Texts: {len(alt_texts)}")
            
            if alt_texts:
                for i, alt in enumerate(alt_texts[:3], 1):
                    print(f"   {i}. {alt}")
                    
        else:
            print(f"❌ Failed to fetch post: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    check_post_content() 