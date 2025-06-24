#!/usr/bin/env python3
"""
Fix to add embedded image processing to WordPress publisher
"""

# Read the current file
with open('src/integrations/wordpress_publisher.py', 'r') as f:
    content = f.read()

# Find the position to insert the new method (before publish_post method)
insert_position = content.find('    async def publish_post(self,')

if insert_position == -1:
    print("❌ Could not find publish_post method")
    exit(1)

# The new method to insert
new_method = '''    async def process_embedded_images_in_content(self, content: str) -> Tuple[str, List[int]]:
        """
        🔧 NEW: Process embedded images in HTML content
        
        This method:
        1. Scans content for <img> tags with external URLs
        2. Downloads and uploads each image to WordPress Media Library
        3. Replaces external URLs with WordPress-hosted URLs
        4. Returns processed content and list of uploaded media IDs
        """
        if not content or '<img' not in content:
            return content, []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            uploaded_media_ids = []
            
            # Find all img tags
            img_tags = soup.find_all('img')
            logger.info(f"🔍 Found {len(img_tags)} images to process")
            
            for img_tag in img_tags:
                src = img_tag.get('src')
                if not src or src.startswith('data:') or src.startswith('/wp-content/'):
                    continue  # Skip data URLs and already WordPress-hosted images
                
                try:
                    # Download the external image
                    logger.info(f"📥 Downloading image: {src}")
                    image_data = await self.image_processor.download_image(self.session, src)
                    
                    if not image_data:
                        logger.warning(f"⚠️ Failed to download image: {src}")
                        continue
                    
                    data, filename = image_data
                    
                    # Process and optimize the image
                    processed_data, processed_filename, metadata = await self.image_processor.process_image(data, filename)
                    
                    # Upload to WordPress Media Library
                    media_url = urljoin(self.config.site_url, "/wp-json/wp/v2/media")
                    
                    # Prepare multipart data
                    form_data = aiohttp.FormData()
                    form_data.add_field('file', processed_data, filename=processed_filename, content_type='image/jpeg')
                    
                    # Use existing alt text or generate one
                    alt_text = img_tag.get('alt', img_tag.get('title', 'Uploaded image'))
                    form_data.add_field('alt_text', alt_text)
                    form_data.add_field('caption', img_tag.get('title', ''))
                    
                    # Upload to WordPress
                    headers = self.auth_manager.headers.copy()
                    headers.pop('Content-Type', None)  # Let aiohttp set this for multipart
                    
                    async with self.session.post(media_url, headers=headers, data=form_data) as response:
                        if response.status in [200, 201]:
                            result = await response.json()
                            media_id = result['id']
                            new_url = result['source_url']
                            
                            # Replace the src with WordPress URL
                            img_tag['src'] = new_url
                            
                            # Add WordPress-specific classes for theme compatibility
                            existing_classes = img_tag.get('class', [])
                            if isinstance(existing_classes, str):
                                existing_classes = existing_classes.split()
                            
                            wp_classes = ['wp-image-' + str(media_id), 'aligncenter', 'size-full']
                            img_tag['class'] = ' '.join(set(existing_classes + wp_classes))
                            
                            # Add WordPress-specific attributes
                            img_tag['data-id'] = media_id
                            
                            uploaded_media_ids.append(media_id)
                            self.stats['images_processed'] += 1
                            
                            logger.info(f"✅ Uploaded image {media_id}: {new_url}")
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Image upload failed: {error_text}")
                            
                except Exception as e:
                    logger.error(f"❌ Error processing image {src}: {e}")
                    continue
            
            # Return the modified content
            processed_content = str(soup)
            logger.info(f"✅ Processed {len(uploaded_media_ids)} images successfully")
            
            return processed_content, uploaded_media_ids
            
        except ImportError:
            logger.error("❌ BeautifulSoup not available. Install with: pip install beautifulsoup4")
            return content, []
        except Exception as e:
            logger.error(f"❌ Error processing embedded images: {e}")
            return content, []

'''

# Insert the new method
new_content = content[:insert_position] + new_method + '\n    ' + content[insert_position:]

# Write back the modified content
with open('src/integrations/wordpress_publisher.py', 'w') as f:
    f.write(new_content)

print("✅ Added process_embedded_images_in_content method to WordPressRESTPublisher")
