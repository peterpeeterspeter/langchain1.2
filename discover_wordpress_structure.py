#!/usr/bin/env python3
"""
WordPress Site Structure Discovery Tool
Helps you discover your WordPress categories, custom fields, tags, and content structure
for casino review content publishing.
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import aiohttp
from datetime import datetime

class WordPressSiteDiscovery:
    """Discover WordPress site structure for casino review content"""
    
    def __init__(self, site_url: str, username: str, app_password: str):
        self.site_url = site_url.rstrip('/')
        self.username = username
        self.app_password = app_password
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Setup authentication
        import base64
        credentials = f"{username}:{app_password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def discover_site_structure(self) -> Dict[str, Any]:
        """Discover complete WordPress site structure"""
        
        print("🔍 Discovering your WordPress site structure...")
        
        structure = {
            "site_info": await self._get_site_info(),
            "categories": await self._get_categories(),
            "tags": await self._get_tags(),
            "custom_fields": await self._discover_custom_fields(),
            "post_types": await self._get_post_types(),
            "media_info": await self._get_media_info(),
            "users": await self._get_users(),
            "discovery_timestamp": datetime.now().isoformat()
        }
        
        return structure
    
    async def _get_site_info(self) -> Dict[str, Any]:
        """Get basic site information"""
        try:
            url = f"{self.site_url}/wp-json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "url": data.get("url", ""),
                        "home": data.get("home", ""),
                        "gmt_offset": data.get("gmt_offset", 0),
                        "timezone_string": data.get("timezone_string", ""),
                        "namespaces": data.get("namespaces", [])
                    }
        except Exception as e:
            print(f"⚠️ Could not get site info: {e}")
        
        return {}
    
    async def _get_categories(self) -> List[Dict[str, Any]]:
        """Get all WordPress categories"""
        categories = []
        try:
            url = f"{self.site_url}/wp-json/wp/v2/categories?per_page=100"
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    for cat in data:
                        categories.append({
                            "id": cat["id"],
                            "name": cat["name"],
                            "slug": cat["slug"],
                            "description": cat["description"],
                            "count": cat["count"],
                            "parent": cat["parent"],
                            "link": cat["link"]
                        })
                    
                    print(f"✅ Found {len(categories)} categories")
                    
                    # Show casino-related categories
                    casino_cats = [c for c in categories if any(term in c["name"].lower() 
                                  for term in ["casino", "review", "game", "bonus", "gambling"])]
                    if casino_cats:
                        print(f"🎰 Casino-related categories:")
                        for cat in casino_cats:
                            print(f"   - {cat['name']} (ID: {cat['id']}, Posts: {cat['count']})")
                    
        except Exception as e:
            print(f"⚠️ Could not get categories: {e}")
        
        return categories
    
    async def _get_tags(self) -> List[Dict[str, Any]]:
        """Get existing WordPress tags"""
        tags = []
        try:
            url = f"{self.site_url}/wp-json/wp/v2/tags?per_page=100"
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    for tag in data:
                        tags.append({
                            "id": tag["id"],
                            "name": tag["name"],
                            "slug": tag["slug"],
                            "description": tag["description"],
                            "count": tag["count"],
                            "link": tag["link"]
                        })
                    
                    print(f"✅ Found {len(tags)} existing tags")
                    
                    # Show popular casino tags
                    casino_tags = [t for t in tags if any(term in t["name"].lower() 
                                  for term in ["casino", "review", "game", "bonus", "gambling", "slot"])]
                    if casino_tags:
                        print(f"🎰 Casino-related tags:")
                        for tag in casino_tags[:10]:  # Show top 10
                            print(f"   - {tag['name']} (ID: {tag['id']}, Uses: {tag['count']})")
                    
        except Exception as e:
            print(f"⚠️ Could not get tags: {e}")
        
        return tags
    
    async def _discover_custom_fields(self) -> Dict[str, Any]:
        """Discover custom fields from existing posts"""
        custom_fields = {
            "discovered_fields": [],
            "casino_specific_fields": [],
            "meta_fields": []
        }
        
        try:
            # Get recent posts with meta data
            url = f"{self.site_url}/wp-json/wp/v2/posts?per_page=20&_embed=true"
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    posts = await response.json()
                    
                    all_meta_keys = set()
                    casino_fields = set()
                    
                    for post in posts:
                        # Check for meta fields in post meta
                        if "meta" in post and post["meta"]:
                            for key, value in post["meta"].items():
                                if value:  # Only non-empty values
                                    all_meta_keys.add(key)
                                    
                                    # Identify casino-specific fields
                                    if any(term in key.lower() for term in 
                                          ["casino", "rating", "bonus", "license", "provider", 
                                           "deposit", "withdrawal", "wagering", "game"]):
                                        casino_fields.add(key)
                    
                    custom_fields["discovered_fields"] = list(all_meta_keys)
                    custom_fields["casino_specific_fields"] = list(casino_fields)
                    
                    print(f"✅ Discovered {len(all_meta_keys)} custom field keys")
                    if casino_fields:
                        print(f"🎰 Casino-specific custom fields found:")
                        for field in sorted(casino_fields):
                            print(f"   - {field}")
                    
        except Exception as e:
            print(f"⚠️ Could not discover custom fields: {e}")
        
        return custom_fields
    
    async def _get_post_types(self) -> List[Dict[str, Any]]:
        """Get available post types"""
        post_types = []
        try:
            url = f"{self.site_url}/wp-json/wp/v2/types"
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    for type_key, type_data in data.items():
                        post_types.append({
                            "name": type_key,
                            "label": type_data.get("name", ""),
                            "description": type_data.get("description", ""),
                            "public": type_data.get("public", False),
                            "hierarchical": type_data.get("hierarchical", False),
                            "rest_base": type_data.get("rest_base", "")
                        })
                    
                    print(f"✅ Found {len(post_types)} post types")
                    
        except Exception as e:
            print(f"⚠️ Could not get post types: {e}")
        
        return post_types
    
    async def _get_media_info(self) -> Dict[str, Any]:
        """Get media library information"""
        media_info = {"total_media": 0, "recent_media": []}
        
        try:
            url = f"{self.site_url}/wp-json/wp/v2/media?per_page=10"
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    media_items = await response.json()
                    media_info["total_media"] = len(media_items)
                    
                    for item in media_items[:5]:  # Show 5 recent items
                        media_info["recent_media"].append({
                            "id": item["id"],
                            "title": item["title"]["rendered"],
                            "source_url": item["source_url"],
                            "media_type": item["media_type"],
                            "mime_type": item["mime_type"]
                        })
                    
                    print(f"✅ Media library has {len(media_items)} recent items")
                    
        except Exception as e:
            print(f"⚠️ Could not get media info: {e}")
        
        return media_info
    
    async def _get_users(self) -> List[Dict[str, Any]]:
        """Get WordPress users (authors)"""
        users = []
        try:
            url = f"{self.site_url}/wp-json/wp/v2/users"
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    for user in data:
                        users.append({
                            "id": user["id"],
                            "name": user["name"],
                            "slug": user["slug"],
                            "description": user["description"],
                            "link": user["link"]
                        })
                    
                    print(f"✅ Found {len(users)} users/authors")
                    
        except Exception as e:
            print(f"⚠️ Could not get users: {e}")
        
        return users
    
    def generate_configuration_code(self, structure: Dict[str, Any]) -> str:
        """Generate Python code for WordPress configuration"""
        
        categories = structure.get("categories", [])
        custom_fields = structure.get("custom_fields", {}).get("casino_specific_fields", [])
        
        config_code = f'''
# 🎰 WordPress Configuration for Your Casino Review Site
# Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Site: {structure.get("site_info", {}).get("name", "Your Site")}

# ✅ UPDATE THESE CATEGORY IDS TO MATCH YOUR SITE:
WORDPRESS_CATEGORY_MAPPING = {{
'''
        
        # Generate category mapping
        casino_categories = [c for c in categories if any(term in c["name"].lower() 
                           for term in ["casino", "review", "game", "bonus"])]
        
        if casino_categories:
            for cat in casino_categories:
                category_type = "casino_review"
                if "bonus" in cat["name"].lower():
                    category_type = "bonus_analysis"
                elif "game" in cat["name"].lower():
                    category_type = "game_guide"
                elif "news" in cat["name"].lower():
                    category_type = "news"
                
                config_code += f'    "{category_type}": [{cat["id"]}],  # "{cat["name"]}"\n'
        else:
            config_code += '''    "casino_review": [5, 12],      # UPDATE: Your "Casino Reviews" category ID
    "game_guide": [8],             # UPDATE: Your "Game Guides" category ID  
    "bonus_analysis": [15],        # UPDATE: Your "Bonuses" category ID
    "comparison": [18, 5],         # UPDATE: Your "Comparisons" category ID
    "news": [22],                  # UPDATE: Your "News" category ID
'''
        
        config_code += '}\n\n'
        
        # Generate custom fields mapping
        config_code += '# ✅ UPDATE THESE CUSTOM FIELD NAMES TO MATCH YOUR SITE:\n'
        config_code += 'WORDPRESS_CUSTOM_FIELDS_MAPPING = {\n'
        
        if custom_fields:
            config_code += '    # Discovered fields from your site:\n'
            for field in custom_fields:
                config_code += f'    "{field}": "{field}",  # Discovered field\n'
        else:
            config_code += '''    # Standard casino review fields (update names to match your fields):
    "casino_rating": "casino_rating",
    "bonus_amount": "bonus_amount", 
    "license_info": "license_info",
    "min_deposit": "min_deposit",
    "withdrawal_time": "withdrawal_time",
    "wagering_requirements": "wagering_requirements",
    "game_providers": "game_providers",
    "payment_methods": "payment_methods",
    "mobile_compatible": "mobile_compatible",
    "live_chat_support": "live_chat_support",
'''
        
        config_code += '}\n\n'
        
        # Generate environment variables
        config_code += '''# ✅ ENVIRONMENT VARIABLES TO SET:
# Add these to your .env file:
WORDPRESS_SITE_URL="{site_url}"
WORDPRESS_USERNAME="{username}"
WORDPRESS_APP_PASSWORD="your_application_password"

# ✅ HOW TO APPLY THIS CONFIGURATION:
# 1. Update the category IDs above to match your actual WordPress categories
# 2. Update the custom field names to match your site's custom fields
# 3. Set the environment variables in your .env file
# 4. The system will automatically use these mappings when publishing casino reviews
'''.format(
            site_url=structure.get("site_info", {}).get("url", "https://yoursite.com"),
            username=structure.get("site_info", {}).get("name", "admin")
        )
        
        return config_code

async def main():
    """Main discovery function"""
    
    print("🎰 WordPress Casino Review Site Structure Discovery")
    print("=" * 60)
    
    # Get WordPress credentials
    site_url = input("Enter your WordPress site URL (e.g., https://yoursite.com): ").strip()
    username = input("Enter your WordPress username: ").strip()
    app_password = input("Enter your WordPress application password: ").strip()
    
    if not all([site_url, username, app_password]):
        print("❌ All fields are required!")
        return
    
    print(f"\n🔍 Discovering structure for: {site_url}")
    print("-" * 60)
    
    async with WordPressSiteDiscovery(site_url, username, app_password) as discovery:
        try:
            # Discover site structure
            structure = await discovery.discover_site_structure()
            
            # Save structure to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"wordpress_structure_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(structure, f, indent=2)
            
            print(f"\n✅ Site structure saved to: {json_filename}")
            
            # Generate configuration code
            config_code = discovery.generate_configuration_code(structure)
            config_filename = f"wordpress_config_{timestamp}.py"
            with open(config_filename, 'w') as f:
                f.write(config_code)
            
            print(f"✅ Configuration code generated: {config_filename}")
            
            # Display summary
            print(f"\n📊 DISCOVERY SUMMARY:")
            print(f"   - Categories: {len(structure.get('categories', []))}")
            print(f"   - Tags: {len(structure.get('tags', []))}")
            print(f"   - Custom Fields: {len(structure.get('custom_fields', {}).get('discovered_fields', []))}")
            print(f"   - Casino Fields: {len(structure.get('custom_fields', {}).get('casino_specific_fields', []))}")
            print(f"   - Post Types: {len(structure.get('post_types', []))}")
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Review the generated configuration file: {config_filename}")
            print(f"2. Update the category IDs and custom field names to match your site")
            print(f"3. Set the environment variables in your .env file")
            print(f"4. The enhanced WordPress publishing will automatically use your configuration")
            
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            print("\nPlease check:")
            print("- Your WordPress site URL is correct")
            print("- Your username and application password are valid")
            print("- Your WordPress site has REST API enabled")

if __name__ == "__main__":
    asyncio.run(main()) 