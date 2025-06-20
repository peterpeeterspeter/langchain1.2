#!/usr/bin/env python3
"""
🚀 BETWAY CASINO REVIEW - WORDPRESS PUBLISHER
==============================================

Publishes the generated Betway Casino review from Supabase to WordPress
using the Enhanced Casino WordPress Publisher from Task 18.

Author: AI Assistant
Date: 2025-01-20
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core imports
from supabase import create_client
import requests
import json
from typing import Dict, Any, List, Optional

class WordPressPublisher:
    """Enhanced WordPress publisher for casino reviews"""
    
    def __init__(self):
        self.wp_url = os.getenv('WORDPRESS_URL', 'https://your-wordpress-site.com')
        self.wp_username = os.getenv('WORDPRESS_USERNAME')
        self.wp_password = os.getenv('WORDPRESS_APP_PASSWORD')
        
    async def publish_casino_review(self, casino_data: Dict[str, Any]) -> Dict[str, Any]:
        """Publish casino review to WordPress"""
        try:
            # Prepare WordPress post data
            post_data = {
                'title': f"{casino_data['name']} Review: Comprehensive Analysis 2025",
                'content': self._format_review_for_wordpress(casino_data),
                'status': 'publish',
                'categories': [1],  # Casino Reviews category
                'tags': ['casino review', 'online casino', casino_data['name'].lower()],
                'meta': {
                    'casino_rating': casino_data.get('rating', 0),
                    'review_date': datetime.now().isoformat(),
                    'casino_name': casino_data['name']
                }
            }
            
            # WordPress REST API endpoint
            api_endpoint = f"{self.wp_url}/wp-json/wp/v2/posts"
            
            # Prepare authentication
            auth = (self.wp_username, self.wp_password)
            
            # Make the request
            response = requests.post(
                api_endpoint,
                json=post_data,
                auth=auth,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 201:
                result = response.json()
                return {
                    'success': True,
                    'post_id': result['id'],
                    'url': result['link'],
                    'title': result['title']['rendered']
                }
            else:
                return {
                    'success': False,
                    'error': f"WordPress API error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Publishing error: {str(e)}"
            }
    
    def _format_review_for_wordpress(self, casino_data: Dict[str, Any]) -> str:
        """Format casino review content for WordPress"""
        content = casino_data['review_content']
        
        # Add rating box at the top
        rating_box = f"""
<div class="casino-rating-box" style="background: #f8f9fa; border: 2px solid #007cba; padding: 20px; margin: 20px 0; border-radius: 8px;">
    <h3 style="margin-top: 0; color: #007cba;">⭐ Overall Rating: {casino_data.get('rating', 'N/A')}/10</h3>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <h4 style="color: #28a745; margin-bottom: 10px;">✅ Pros:</h4>
            <ul>
                {''.join([f'<li>{pro}</li>' for pro in casino_data.get('pros', [])])}
            </ul>
        </div>
        <div>
            <h4 style="color: #dc3545; margin-bottom: 10px;">❌ Cons:</h4>
            <ul>
                {''.join([f'<li>{con}</li>' for con in casino_data.get('cons', [])])}
            </ul>
        </div>
    </div>
</div>
"""
        
        # Add bonus information
        bonuses = casino_data.get('bonuses', {})
        if bonuses:
            bonus_section = f"""
<div class="bonus-highlight" style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; border-radius: 5px;">
    <h3 style="margin-top: 0;">🎁 Welcome Bonus</h3>
    <p><strong>{bonuses.get('welcome_bonus', 'Contact casino for details')}</strong></p>
    <p><small>Wagering Requirements: {bonuses.get('wagering_requirements', 'Check T&C')}</small></p>
</div>
"""
            content = rating_box + bonus_section + content
        else:
            content = rating_box + content
        
        # Add licensing information
        licensing = casino_data.get('licensing', [])
        if licensing:
            license_section = f"""
<div class="licensing-info" style="background: #e8f5e8; border: 1px solid #c3e6c3; padding: 15px; margin: 20px 0; border-radius: 5px;">
    <h3 style="margin-top: 0;">🏛️ Licensed & Regulated By:</h3>
    <ul>
        {''.join([f'<li>{license}</li>' for license in licensing])}
    </ul>
</div>
"""
            content = content + license_section
        
        # Add disclaimer
        disclaimer = """
<div class="disclaimer" style="background: #f8f9fa; border-left: 4px solid #007cba; padding: 15px; margin: 30px 0; font-size: 14px;">
    <p><strong>Disclaimer:</strong> This review is for informational purposes only. Gambling can be addictive. Please play responsibly and only gamble what you can afford to lose. This content may contain affiliate links.</p>
</div>
"""
        
        return content + disclaimer

async def main():
    """Main publishing function"""
    print("🚀 Starting Betway Casino Review WordPress Publishing...")
    
    try:
        # Initialize Supabase client
        supabase_url = 'https://ambjsovdhizjxwhhnbtd.supabase.co'
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_key:
            print("❌ SUPABASE_SERVICE_ROLE_KEY environment variable not set")
            return
        
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")
        
        # Retrieve the complete Betway review from Supabase
        print("📖 Retrieving Betway review from database...")
        response = supabase.table('documents').select('content').eq('metadata->>article_id', '117').execute()
        
        if not response.data:
            print('❌ No Betway review found in database')
            return
        
        # Combine all content sections
        review_sections = [item['content'] for item in response.data]
        full_review = '\n\n'.join(review_sections)
        
        print(f'✅ Retrieved review: {len(full_review)} characters')
        print(f"📝 Content preview: {full_review[:200]}...")
        
        # Initialize WordPress publisher
        publisher = WordPressPublisher()
        
        # Prepare casino data for WordPress
        casino_data = {
            'name': 'Betway Casino',
            'review_content': full_review,
            'rating': 8.5,  # Based on review analysis
            'pros': [
                'Strong licensing from UKGC and other authorities',
                'Over 450 games including slots, table games, live casino',
                'Mobile-optimized platform and dedicated app',
                'Quick payouts, especially with e-wallets',
                'Responsible gambling tools and player safety'
            ],
            'cons': [
                'High wagering requirements (up to 50x)',
                'Limited ongoing promotions beyond welcome bonus',
                'Smaller table game selection compared to competitors'
            ],
            'bonuses': {
                'welcome_bonus': '100% deposit match up to $1,000',
                'wagering_requirements': '50x deposit + bonus'
            },
            'licensing': [
                'UK Gambling Commission (UKGC)',
                'Alderney Gambling Control Commission', 
                'New Jersey Division of Gaming Enforcement',
                'Pennsylvania Gaming Control Board'
            ]
        }
        
        print("🚀 Publishing to WordPress...")
        
        # Check if WordPress credentials are available
        if not all([os.getenv('WORDPRESS_URL'), os.getenv('WORDPRESS_USERNAME'), os.getenv('WORDPRESS_APP_PASSWORD')]):
            print("⚠️  WordPress credentials not configured. Simulating publication...")
            print("📝 Review ready for WordPress publishing:")
            print(f"   Title: {casino_data['name']} Review: Comprehensive Analysis 2025")
            print(f"   Content Length: {len(casino_data['review_content'])} characters")
            print(f"   Rating: {casino_data['rating']}/10")
            print(f"   Pros: {len(casino_data['pros'])} items")
            print(f"   Cons: {len(casino_data['cons'])} items")
            print("✅ Review formatted and ready for WordPress!")
            
            # Update Supabase to mark as ready for publishing
            update_result = supabase.table('casino_reviews').update({
                'title': f"{casino_data['name']} Review: Comprehensive Analysis 2025",
                'full_review': full_review,
                'rating': casino_data['rating'],
                'published': False,  # Set to True when actually published
                'review_text': full_review[:1000] + '...' if len(full_review) > 1000 else full_review
            }).eq('casino_name', 'Betway Casino Review Casino').execute()
            
            print("✅ Updated Supabase with review metadata")
            return
        
        # Publish to WordPress
        result = await publisher.publish_casino_review(casino_data)
        
        if result.get('success'):
            print(f'🎉 SUCCESS: Published to WordPress!')
            print(f'📝 Post ID: {result.get("post_id")}')
            print(f'🔗 URL: {result.get("url")}')
            print(f'📄 Title: {result.get("title")}')
            
            # Update Supabase with WordPress info
            update_result = supabase.table('casino_reviews').update({
                'wordpress_post_id': result.get('post_id'),
                'wordpress_url': result.get('url'),
                'published': True,
                'title': result.get('title'),
                'full_review': full_review,
                'rating': casino_data['rating']
            }).eq('casino_name', 'Betway Casino Review Casino').execute()
            
            print('✅ Updated Supabase with WordPress publishing info')
        else:
            print(f'❌ Publishing failed: {result.get("error")}')
            
    except Exception as e:
        print(f"❌ Error during publishing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 