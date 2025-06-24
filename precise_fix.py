#!/usr/bin/env python3
"""
Precisely add the embedded image processing method
"""

import re

# Read the current file
with open('src/integrations/wordpress_publisher.py', 'r') as f:
    lines = f.readlines()

# Find the line with "async def publish_post(self," and insert before it
insert_line = None
for i, line in enumerate(lines):
    if line.strip().startswith('async def publish_post(self,'):
        insert_line = i
        break

if insert_line is None:
    print("❌ Could not find publish_post method")
    exit(1)

# The new method lines to insert
new_method_lines = [
    "    async def process_embedded_images_in_content(self, content: str) -> Tuple[str, List[int]]:\n",
    "        \"\"\"\n",
    "        🔧 NEW: Process embedded images in HTML content\n",
    "        \n",
    "        This method:\n",
    "        1. Scans content for <img> tags with external URLs\n",
    "        2. Downloads and uploads each image to WordPress Media Library\n",
    "        3. Replaces external URLs with WordPress-hosted URLs\n",
    "        4. Returns processed content and list of uploaded media IDs\n",
    "        \"\"\"\n",
    "        if not content or '<img' not in content:\n",
    "            return content, []\n",
    "        \n",
    "        try:\n",
    "            from bs4 import BeautifulSoup\n",
    "            soup = BeautifulSoup(content, 'html.parser')\n",
    "            uploaded_media_ids = []\n",
    "            \n",
    "            # Find all img tags\n",
    "            img_tags = soup.find_all('img')\n",
    "            logger.info(f\"🔍 Found {len(img_tags)} images to process\")\n",
    "            \n",
    "            for img_tag in img_tags:\n",
    "                src = img_tag.get('src')\n",
    "                if not src or src.startswith('data:') or src.startswith('/wp-content/'):\n",
    "                    continue  # Skip data URLs and already WordPress-hosted images\n",
    "                \n",
    "                try:\n",
    "                    # Download the external image\n",
    "                    logger.info(f\"📥 Downloading image: {src}\")\n",
    "                    image_data = await self.image_processor.download_image(self.session, src)\n",
    "                    \n",
    "                    if not image_data:\n",
    "                        logger.warning(f\"⚠️ Failed to download image: {src}\")\n",
    "                        continue\n",
    "                    \n",
    "                    data, filename = image_data\n",
    "                    \n",
    "                    # Process and optimize the image\n",
    "                    processed_data, processed_filename, metadata = await self.image_processor.process_image(data, filename)\n",
    "                    \n",
    "                    # Upload to WordPress Media Library\n",
    "                    media_url = urljoin(self.config.site_url, \"/wp-json/wp/v2/media\")\n",
    "                    \n",
    "                    # Prepare multipart data\n",
    "                    form_data = aiohttp.FormData()\n",
    "                    form_data.add_field('file', processed_data, filename=processed_filename, content_type='image/jpeg')\n",
    "                    \n",
    "                    # Use existing alt text or generate one\n",
    "                    alt_text = img_tag.get('alt', img_tag.get('title', 'Uploaded image'))\n",
    "                    form_data.add_field('alt_text', alt_text)\n",
    "                    form_data.add_field('caption', img_tag.get('title', ''))\n",
    "                    \n",
    "                    # Upload to WordPress\n",
    "                    headers = self.auth_manager.headers.copy()\n",
    "                    headers.pop('Content-Type', None)  # Let aiohttp set this for multipart\n",
    "                    \n",
    "                    async with self.session.post(media_url, headers=headers, data=form_data) as response:\n",
    "                        if response.status in [200, 201]:\n",
    "                            result = await response.json()\n",
    "                            media_id = result['id']\n",
    "                            new_url = result['source_url']\n",
    "                            \n",
    "                            # Replace the src with WordPress URL\n",
    "                            img_tag['src'] = new_url\n",
    "                            \n",
    "                            # Add WordPress-specific classes for theme compatibility\n",
    "                            existing_classes = img_tag.get('class', [])\n",
    "                            if isinstance(existing_classes, str):\n",
    "                                existing_classes = existing_classes.split()\n",
    "                            \n",
    "                            wp_classes = ['wp-image-' + str(media_id), 'aligncenter', 'size-full']\n",
    "                            img_tag['class'] = ' '.join(set(existing_classes + wp_classes))\n",
    "                            \n",
    "                            # Add WordPress-specific attributes\n",
    "                            img_tag['data-id'] = media_id\n",
    "                            \n",
    "                            uploaded_media_ids.append(media_id)\n",
    "                            self.stats['images_processed'] += 1\n",
    "                            \n",
    "                            logger.info(f\"✅ Uploaded image {media_id}: {new_url}\")\n",
    "                        else:\n",
    "                            error_text = await response.text()\n",
    "                            logger.error(f\"❌ Image upload failed: {error_text}\")\n",
    "                            \n",
    "                except Exception as e:\n",
    "                    logger.error(f\"❌ Error processing image {src}: {e}\")\n",
    "                    continue\n",
    "            \n",
    "            # Return the modified content\n",
    "            processed_content = str(soup)\n",
    "            logger.info(f\"✅ Processed {len(uploaded_media_ids)} images successfully\")\n",
    "            \n",
    "            return processed_content, uploaded_media_ids\n",
    "            \n",
    "        except ImportError:\n",
    "            logger.error(\"❌ BeautifulSoup not available. Install with: pip install beautifulsoup4\")\n",
    "            return content, []\n",
    "        except Exception as e:\n",
    "            logger.error(f\"❌ Error processing embedded images: {e}\")\n",
    "            return content, []\n",
    "\n",
]

# Insert the new method lines
new_lines = lines[:insert_line] + new_method_lines + lines[insert_line:]

# Write back the modified content
with open('src/integrations/wordpress_publisher.py', 'w') as f:
    f.writelines(new_lines)

print(f"✅ Added process_embedded_images_in_content method before line {insert_line + 1}")

# Now update the publish_post method
with open('src/integrations/wordpress_publisher.py', 'r') as f:
    content = f.read()

# Update the format content line
old_line = "            # Format content with rich HTML"
new_line = "            # 🔧 NEW: Process embedded images in content FIRST\n            processed_content, embedded_media_ids = await self.process_embedded_images_in_content(content)\n            \n            # Format content with rich HTML (now with WordPress-hosted images)"

content = content.replace(old_line, new_line)

# Update the formatted_content line
content = content.replace(
    "formatted_content = self.html_formatter.format_content(content, title, meta_description)",
    "formatted_content = self.html_formatter.format_content(processed_content, title, meta_description)"
)

# Update the return statement
old_return = """                    logger.info(f"Post published successfully: {result['id']}")
                    return result"""

new_return = """                    # Add embedded media info to result
                    result['embedded_media_count'] = len(embedded_media_ids)
                    result['embedded_media_ids'] = embedded_media_ids
                    
                    logger.info(f"Post published successfully: {result['id']} with {len(embedded_media_ids)} embedded images")
                    return result"""

content = content.replace(old_return, new_return)

# Write back
with open('src/integrations/wordpress_publisher.py', 'w') as f:
    f.write(content)

print("✅ Updated publish_post method to use embedded image processing")
