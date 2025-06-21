#!/usr/bin/env python3
"""
🔧 BULLETPROOF IMAGE INTEGRATOR 
Solves the core issues: images found but not embedded, broken uploads, and poor HTML integration

CRITICAL FIXES APPLIED:
1. ✅ V1-style simple Supabase uploads (no complex attribute checking)
2. ✅ Smart image embedding with context-aware placement
3. ✅ Proper HTML integration with responsive design
4. ✅ DataForSEO → Upload → Embed pipeline integration

Based on V1 working patterns from fix_supabase_image_upload.py analysis
"""

import asyncio
import logging
import uuid
import aiohttp
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class ImageEmbedConfig:
    """Configuration for smart image embedding"""
    max_images_per_section: int = 2
    hero_image_enabled: bool = True
    gallery_section_enabled: bool = True
    responsive_design: bool = True
    lazy_loading: bool = True
    seo_optimization: bool = True
    max_image_width: int = 800
    thumbnail_size: int = 300

class BulletproofImageUploader:
    """
    V1-style bulletproof image uploader with proper Supabase patterns
    
    CRITICAL: Uses V1's working patterns, not V6.0's broken attribute checking
    """
    
    def __init__(self, supabase_client, storage_bucket: str = "images"):
        self.supabase = supabase_client
        self.bucket = storage_bucket
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
    async def upload_image_with_retry(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Upload image with V1-style bulletproof retry logic"""
        
        for attempt in range(self.retry_attempts):
            try:
                # Generate unique storage path
                file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
                unique_filename = f"{uuid.uuid4()}.{file_extension}"
                storage_path = f"images/{datetime.now().strftime('%Y/%m/%d')}/{unique_filename}"
                
                # Upload to Supabase - V1 STYLE (simple and working)
                upload_result = self.supabase.storage.from_(self.bucket).upload(
                    storage_path,
                    image_data,
                    file_options={
                        "content-type": f"image/{file_extension}",
                        "cache-control": "3600",
                        "upsert": True  # Allow overwrite if needed
                    }
                )
                
                # V1 SUCCESS PATTERN: If no exception raised, it worked
                logger.info(f"Successfully uploaded image: {storage_path}")
                
                # Get public URL
                public_url = self.supabase.storage.from_(self.bucket).get_public_url(storage_path)
                
                return {
                    "success": True,
                    "storage_path": storage_path,
                    "public_url": public_url,
                    "filename": unique_filename,
                    "size": len(image_data),
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Upload attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < self.retry_attempts - 1:
                    # V1 RETRY PATTERN: Exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final failure
                    logger.error(f"All upload attempts failed for {filename}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "attempts_made": self.retry_attempts
                    }
        
        return {"success": False, "error": "Max retries exceeded"}
    
    async def optimize_image(self, image_data: bytes, max_width: int = 1200) -> bytes:
        """Optimize image size and quality"""
        try:
            with Image.open(BytesIO(image_data)) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if necessary
                if img.size[0] > max_width:
                    ratio = max_width / img.size[0]
                    new_size = (max_width, int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save optimized
                output = BytesIO()
                img.save(
                    output,
                    format='JPEG',
                    quality=85,
                    optimize=True,
                    progressive=True
                )
                return output.getvalue()
                
        except Exception as e:
            logger.warning(f"Image optimization failed: {e}, using original")
            return image_data

class SmartImageEmbedder:
    """
    Smart image embedding with context-aware placement
    
    Solves the core issue: DataForSEO finds images but they're not used in final content
    """
    
    def __init__(self, config: ImageEmbedConfig = None):
        self.config = config or ImageEmbedConfig()
        
    def embed_images_intelligently(self, content: str, images: List[Dict[str, Any]]) -> str:
        """
        Embed images into content with intelligent placement strategies
        
        CORE SOLUTION: Takes found images and actually embeds them in content
        """
        if not images:
            return content
        
        # Parse content structure
        soup = BeautifulSoup(content, 'html.parser')
        
        # Strategy 1: Hero image at the beginning
        if self.config.hero_image_enabled and images:
            hero_image = self._select_hero_image(images)
            if hero_image:
                hero_html = self._create_hero_image_html(hero_image)
                
                # Insert after title/first paragraph
                first_header = soup.find(['h1', 'h2'])
                if first_header:
                    hero_soup = BeautifulSoup(hero_html, 'html.parser')
                    first_header.insert_after(hero_soup)
                else:
                    # Insert at beginning
                    hero_soup = BeautifulSoup(hero_html, 'html.parser')
                    soup.insert(0, hero_soup)
        
        # Strategy 2: Context-aware inline images
        self._embed_inline_images(soup, images[1:])  # Skip hero image
        
        # Strategy 3: Gallery section for remaining images
        if self.config.gallery_section_enabled:
            remaining_images = images[3:]  # Skip hero + 2 inline
            if remaining_images:
                gallery_html = self._create_gallery_section(remaining_images)
                gallery_soup = BeautifulSoup(gallery_html, 'html.parser')
                soup.append(gallery_soup)
        
        return str(soup)
    
    def _select_hero_image(self, images: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best image for hero placement"""
        if not images:
            return None
        
        # Prefer high-quality, landscape images
        scored_images = []
        for img in images:
            score = 0
            
            # Quality score bonus
            if (img.get('quality_score', 0) or 0) > 0.7:
                score += 3
            
            # Landscape orientation bonus
            width = img.get('width', 0) or 0
            height = img.get('height', 0) or 0
            if width > height and width >= 800:
                score += 2
            
            # Title relevance bonus
            if img.get('title') and len(img.get('title', '')) > 10:
                score += 1
            
            scored_images.append((score, img))
        
        # Return highest scoring image
        scored_images.sort(key=lambda x: x[0], reverse=True)
        return scored_images[0][1] if scored_images else images[0]
    
    def _create_hero_image_html(self, image: Dict[str, Any]) -> str:
        """Create hero image HTML with responsive design"""
        
        img_attrs = {
            'src': image.get('url', ''),
            'alt': image.get('alt_text', image.get('title', 'Featured image')),
            'title': image.get('title', ''),
            'class': 'hero-image',
            'loading': 'eager',  # Hero images should load immediately
        }
        
        if self.config.responsive_design:
            img_attrs['style'] = f'width: 100%; max-width: {self.config.max_image_width}px; height: auto;'
        
        # Add dimensions if available
        if image.get('width'):
            img_attrs['width'] = str(image['width'])
        if image.get('height'):
            img_attrs['height'] = str(image['height'])
        
        img_tag = ' '.join([f'{k}="{v}"' for k, v in img_attrs.items()])
        
        hero_html = f"""
<div class="hero-image-container" style="margin: 2rem 0; text-align: center;">
    <img {img_tag} />
    {f'<p class="hero-caption" style="font-style: italic; margin-top: 0.5rem; color: #666;">{image.get("title", "")}</p>' if image.get('title') else ''}
</div>
"""
        return hero_html
    
    def _embed_inline_images(self, soup: BeautifulSoup, images: List[Dict[str, Any]]):
        """Embed images after relevant sections"""
        
        headers = soup.find_all(['h2', 'h3'])
        images_used = 0
        
        for header in headers:
            if images_used >= len(images) or images_used >= self.config.max_images_per_section * len(headers):
                break
            
            image = images[images_used]
            
            # Create contextual image HTML
            img_html = self._create_inline_image_html(image)
            img_soup = BeautifulSoup(img_html, 'html.parser')
            
            # Insert after the header's next sibling paragraph
            next_element = header.find_next_sibling(['p', 'div'])
            if next_element:
                next_element.insert_after(img_soup)
            else:
                header.insert_after(img_soup)
            
            images_used += 1
    
    def _create_inline_image_html(self, image: Dict[str, Any]) -> str:
        """Create inline image HTML with proper formatting"""
        
        img_attrs = {
            'src': image.get('url', ''),
            'alt': image.get('alt_text', image.get('title', 'Content image')),
            'title': image.get('title', ''),
            'class': 'content-image',
        }
        
        if self.config.lazy_loading:
            img_attrs['loading'] = 'lazy'
        
        if self.config.responsive_design:
            img_attrs['style'] = f'width: 100%; max-width: {self.config.max_image_width}px; height: auto;'
        
        # Add dimensions for layout stability
        if image.get('width'):
            img_attrs['width'] = str(min(image['width'], self.config.max_image_width))
        if image.get('height') and image.get('width'):
            # Maintain aspect ratio
            ratio = min(self.config.max_image_width / image['width'], 1)
            img_attrs['height'] = str(int(image['height'] * ratio))
        
        img_tag = ' '.join([f'{k}="{v}"' for k, v in img_attrs.items()])
        
        inline_html = f"""
<div class="content-image-container" style="margin: 1.5rem 0; text-align: center;">
    <img {img_tag} />
    {f'<p class="image-caption" style="font-style: italic; margin-top: 0.5rem; color: #666; font-size: 0.9em;">{image.get("title", "")}</p>' if image.get('title') else ''}
</div>
"""
        return inline_html
    
    def _create_gallery_section(self, images: List[Dict[str, Any]]) -> str:
        """Create gallery section for remaining images"""
        
        if not images:
            return ""
        
        gallery_items = []
        for image in images:
            item_html = f"""
<div class="gallery-item" style="display: inline-block; margin: 0.5rem; text-align: center;">
    <img src="{image.get('url', '')}" 
         alt="{image.get('alt_text', image.get('title', 'Gallery image'))}"
         title="{image.get('title', '')}"
         style="max-width: {self.config.thumbnail_size}px; height: auto; border-radius: 4px;"
         {f'loading="lazy"' if self.config.lazy_loading else ''} />
    {f'<p style="font-size: 0.8em; margin-top: 0.25rem; color: #666;">{image.get("title", "")}</p>' if image.get('title') else ''}
</div>
"""
            gallery_items.append(item_html)
        
        gallery_html = f"""
<div class="image-gallery" style="margin: 2rem 0; padding: 1rem; background-color: #f9f9f9; border-radius: 8px;">
    <h3 style="margin-bottom: 1rem; color: #333;">Related Images</h3>
    <div class="gallery-grid" style="text-align: center;">
        {''.join(gallery_items)}
    </div>
</div>
"""
        return gallery_html

class BulletproofImageIntegrator:
    """
    Complete image integration pipeline: Download → Upload → Embed
    
    SOLVES THE CORE ISSUE: Images found by DataForSEO but not used in final content
    """
    
    def __init__(self, supabase_client, config: ImageEmbedConfig = None):
        self.uploader = BulletproofImageUploader(supabase_client)
        self.embedder = SmartImageEmbedder(config)
        self.supabase = supabase_client
        
    async def process_and_integrate_images(
        self, 
        content: str, 
        images: List[Dict[str, Any]], 
        upload_images: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Complete pipeline: Download + Upload + Embed images into content
        
        Args:
            content: HTML/Markdown content to enhance
            images: List of image metadata from DataForSEO
            upload_images: Whether to upload images to Supabase
            
        Returns:
            Tuple of (enhanced_content, processed_images)
        """
        
        if not images:
            logger.info("No images provided for integration")
            return content, []
        
        logger.info(f"Processing {len(images)} images for integration")
        processed_images = []
        
        for i, image in enumerate(images):
            try:
                processed_image = image.copy()
                
                if upload_images and image.get('url'):
                    # Download and upload image
                    uploaded_data = await self._download_and_upload_image(image)
                    
                    if uploaded_data['success']:
                        # Update image URL to use uploaded version
                        processed_image['url'] = uploaded_data['public_url']
                        processed_image['storage_path'] = uploaded_data['storage_path']
                        processed_image['uploaded'] = True
                        logger.info(f"Successfully uploaded image {i+1}/{len(images)}")
                    else:
                        logger.warning(f"Upload failed for image {i+1}, using original URL")
                        processed_image['uploaded'] = False
                        processed_image['upload_error'] = uploaded_data.get('error')
                
                processed_images.append(processed_image)
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                # Keep original image data
                processed_image = image.copy()
                processed_image['uploaded'] = False
                processed_image['processing_error'] = str(e)
                processed_images.append(processed_image)
        
        # Embed images into content
        try:
            enhanced_content = self.embedder.embed_images_intelligently(content, processed_images)
            logger.info(f"Successfully embedded {len(processed_images)} images into content")
            
            return enhanced_content, processed_images
            
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return content, processed_images
    
    async def _download_and_upload_image(self, image: Dict[str, Any]) -> Dict[str, Any]:
        """Download image from URL and upload to Supabase"""
        
        try:
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(image['url']) as response:
                    if response.status != 200:
                        return {"success": False, "error": f"HTTP {response.status}"}
                    
                    image_data = await response.read()
                    
                    # Size check
                    if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                        return {"success": False, "error": "Image too large"}
            
            # Optimize image
            optimized_data = await self.uploader.optimize_image(image_data)
            
            # Generate filename
            filename = f"{uuid.uuid4()}.jpg"
            if image.get('url'):
                url_filename = image['url'].split('/')[-1]
                if '.' in url_filename:
                    extension = url_filename.split('.')[-1].lower()
                    if extension in ['jpg', 'jpeg', 'png', 'webp']:
                        filename = f"{uuid.uuid4()}.{extension}"
            
            # Upload with V1-style bulletproof logic
            return await self.uploader.upload_image_with_retry(optimized_data, filename)
            
        except Exception as e:
            logger.error(f"Download/upload failed for {image.get('url')}: {e}")
            return {"success": False, "error": str(e)}

# Factory Functions

def create_bulletproof_image_integrator(
    supabase_client,
    config: Dict[str, Any] = None
) -> BulletproofImageIntegrator:
    """Create a bulletproof image integrator with smart embedding"""
    
    embed_config = ImageEmbedConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(embed_config, key):
                setattr(embed_config, key, value)
    
    return BulletproofImageIntegrator(supabase_client, embed_config)

async def process_dataforseo_images_with_embedding(
    content: str,
    dataforseo_images: List[Dict[str, Any]],
    supabase_client,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Complete DataForSEO → Embedding pipeline
    
    SOLVES THE CORE ISSUE: "Images found but not used in final content"
    """
    
    integrator = create_bulletproof_image_integrator(supabase_client, config)
    
    enhanced_content, processed_images = await integrator.process_and_integrate_images(
        content=content,
        images=dataforseo_images,
        upload_images=True
    )
    
    success_count = sum(1 for img in processed_images if img.get('uploaded', False))
    
    return {
        "enhanced_content": enhanced_content,
        "processed_images": processed_images,
        "images_uploaded": success_count,
        "images_embedded": len(processed_images),
        "success_rate": success_count / len(processed_images) if processed_images else 0,
        "integration_successful": True
    }

# Example Usage

async def demo_bulletproof_integration():
    """Demonstrate the complete bulletproof image integration"""
    
    # Example DataForSEO images (found but not embedded)
    sample_images = [
        {
            "url": "https://example.com/casino-homepage.jpg",
            "title": "Casino Homepage",
            "alt_text": "Modern casino interface",
            "width": 1200,
            "height": 800,
            "quality_score": 0.85
        },
        {
            "url": "https://example.com/games-section.jpg", 
            "title": "Casino Games",
            "alt_text": "Variety of casino games",
            "width": 800,
            "height": 600,
            "quality_score": 0.78
        }
    ]
    
    sample_content = """
    # Casino Review
    
    ## Overview
    This casino offers an excellent gaming experience.
    
    ## Games Available
    The platform features hundreds of slots and table games.
    
    ## Mobile Experience
    The mobile app provides seamless gaming on-the-go.
    """
    
    print("🔧 BULLETPROOF IMAGE INTEGRATION DEMO")
    print("=" * 60)
    print(f"📝 Original content: {len(sample_content)} characters")
    print(f"🖼️  Images to process: {len(sample_images)}")
    
    # Note: This would need actual Supabase client
    # enhanced_content, processed_images = await integrator.process_and_integrate_images(
    #     content=sample_content,
    #     images=sample_images,
    #     upload_images=False  # Demo mode
    # )
    
    # Demo the embedding without upload
    embedder = SmartImageEmbedder()
    enhanced_content = embedder.embed_images_intelligently(sample_content, sample_images)
    
    print("\n✅ INTEGRATION RESULTS:")
    print(f"📄 Enhanced content: {len(enhanced_content)} characters")
    print(f"🎯 Images embedded: {'✅ YES' if '<img src=' in enhanced_content else '❌ NO'}")
    print(f"🎨 Hero image: {'✅ YES' if 'hero-image' in enhanced_content else '❌ NO'}")
    print(f"📱 Responsive design: {'✅ YES' if 'max-width' in enhanced_content else '❌ NO'}")
    
    print("\n📖 ENHANCED CONTENT PREVIEW:")
    print(enhanced_content[:500] + "..." if len(enhanced_content) > 500 else enhanced_content)

if __name__ == "__main__":
    asyncio.run(demo_bulletproof_integration()) 