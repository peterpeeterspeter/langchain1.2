#!/usr/bin/env python3
"""
Update the publish_post method to process embedded images
"""

# Read the current file
with open('src/integrations/wordpress_publisher.py', 'r') as f:
    content = f.read()

# Find and replace the publish_post method to include embedded image processing
old_code = '''        try:
            # Format content with rich HTML
            formatted_content = self.html_formatter.format_content(content, title, meta_description)'''

new_code = '''        try:
            # 🔧 NEW: Process embedded images in content FIRST
            processed_content, embedded_media_ids = await self.process_embedded_images_in_content(content)
            
            # Format content with rich HTML (now with WordPress-hosted images)
            formatted_content = self.html_formatter.format_content(processed_content, title, meta_description)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Updated publish_post method to process embedded images")
else:
    print("❌ Could not find the target code to replace")
    exit(1)

# Also update the result to include embedded media count
old_result = '''                    logger.info(f"Post published successfully: {result['id']}")
                    return result'''

new_result = '''                    # Add embedded media info to result
                    result['embedded_media_count'] = len(embedded_media_ids)
                    result['embedded_media_ids'] = embedded_media_ids
                    
                    logger.info(f"Post published successfully: {result['id']} with {len(embedded_media_ids)} embedded images")
                    return result'''

if old_result in content:
    content = content.replace(old_result, new_result)
    print("✅ Updated result to include embedded media information")
else:
    print("⚠️ Could not find result code to update (non-critical)")

# Write back the modified content
with open('src/integrations/wordpress_publisher.py', 'w') as f:
    f.write(content)

print("✅ WordPress publisher updated with embedded image processing")
