#!/usr/bin/env python3
"""
Update the Universal RAG Chain to pass embedded images data to WordPress
"""

# Read the current file
with open('src/chains/universal_rag_lcel.py', 'r') as f:
    content = f.read()

# Find the WordPress publishing section and update it
old_code = '''            # ✅ NEW: Find featured image from our DataForSEO results
            featured_image_url = await self._select_featured_image()
            
            # Create enhanced WordPress post data
            post_data = {
                "title": title,
                "content": final_content,
                "status": "publish",  # Publish directly to live site
                "categories": category_ids,
                "tags": tags,
                "meta_description": meta_description,
                "custom_fields": custom_fields,
                "featured_image_url": featured_image_url
            }'''

new_code = '''            # ✅ NEW: Find featured image from our DataForSEO results
            featured_image_url = await self._select_featured_image()
            
            # 🔧 NEW: Pass embedded images data to WordPress for processing
            embedded_images_data = []
            if hasattr(self, '_last_images') and self._last_images:
                embedded_images_data = self._last_images
                logging.info(f"🖼️ Passing {len(embedded_images_data)} embedded images to WordPress")
            
            # Create enhanced WordPress post data
            post_data = {
                "title": title,
                "content": final_content,  # This already has embedded images with external URLs
                "status": "publish",  # Publish directly to live site
                "categories": category_ids,
                "tags": tags,
                "meta_description": meta_description,
                "custom_fields": custom_fields,
                "featured_image_url": featured_image_url,
                # 🔧 NEW: Pass embedded images metadata (for reference/debugging)
                "embedded_images": embedded_images_data
            }'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Updated Universal RAG Chain to pass embedded images data")
else:
    print("❌ Could not find the target WordPress publishing code")
    exit(1)

# Also update the WordPress metadata capture to include embedded image info
old_metadata = '''            # Capture WordPress metadata
            wordpress_metadata = {
                "published": True,
                "post_id": result.get("id"),
                "post_url": result.get("link"),
                "edit_url": f"{self.wordpress_service.config.site_url}/wp-admin/post.php?post={result.get('id')}&action=edit",
                "title": title,
                "category": content_type,
                "custom_fields_count": len(custom_fields),
                "tags_count": len(tags),
                "featured_image_set": bool(featured_image_url),
                "status": "publish"
            }'''

new_metadata = '''            # Capture WordPress metadata with embedded image info
            wordpress_metadata = {
                "published": True,
                "post_id": result.get("id"),
                "post_url": result.get("link"),
                "edit_url": f"{self.wordpress_service.config.site_url}/wp-admin/post.php?post={result.get('id')}&action=edit",
                "title": title,
                "category": content_type,
                "custom_fields_count": len(custom_fields),
                "tags_count": len(tags),
                "featured_image_set": bool(featured_image_url),
                "embedded_images_processed": result.get("embedded_media_count", 0),
                "embedded_media_ids": result.get("embedded_media_ids", []),
                "status": "publish"
            }'''

if old_metadata in content:
    content = content.replace(old_metadata, new_metadata)
    print("✅ Updated WordPress metadata to include embedded image info")
else:
    print("⚠️ Could not find WordPress metadata code (non-critical)")

# Update the logging to show embedded image processing
old_logging = '''            logging.info(f"✅ Published to WordPress: {title} (ID: {result.get('id')}) with {len(custom_fields)} custom fields")'''

new_logging = '''            logging.info(f"✅ Published to WordPress: {title} (ID: {result.get('id')}) with {len(custom_fields)} custom fields and {result.get('embedded_media_count', 0)} embedded images")'''

if old_logging in content:
    content = content.replace(old_logging, new_logging)
    print("✅ Updated logging to show embedded image processing")
else:
    print("⚠️ Could not find logging code (non-critical)")

# Write back the modified content
with open('src/chains/universal_rag_lcel.py', 'w') as f:
    f.write(content)

print("✅ Universal RAG Chain updated with embedded image support")
