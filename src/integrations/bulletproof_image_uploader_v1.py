#!/usr/bin/env python3
"""
Bulletproof Image Uploader V1
Robust image downloading and uploading with retry logic
"""

import asyncio
import aiohttp
import aiofiles
import os
import time
from typing import Dict, Optional, List
from PIL import Image
import io
import requests


class WordPressCredentials:
    """WordPress credentials for API access"""
    def __init__(self, base_url: str, username: str, application_password: str):
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/wp-json/wp/v2'):
            self.base_url += '/wp-json/wp/v2'
        self.username = username
        self.application_password = application_password
        
        # Create auth header
        import base64
        auth_string = f"{username}:{application_password}"
        auth_bytes = auth_string.encode('ascii')
        auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
        self.auth_header = f"Basic {auth_b64}"


class BulletproofImageUploader:
    """Bulletproof image uploader with retry logic"""
    
    def __init__(self, wp_credentials: WordPressCredentials, max_retries: int = 5, timeout: int = 30):
        self.wp_credentials = wp_credentials
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = None
        
        # Success tracking
        self.download_attempts = 0
        self.download_successes = 0
        self.upload_attempts = 0
        self.upload_successes = 0
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; BulletproofUploader/1.0)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def optimize_image(self, image_data: bytes, max_size: tuple = (1920, 1080), quality: int = 85) -> bytes:
        """Optimize image for web and WordPress compatibility"""
        try:
            # Open image
            img = Image.open(io.BytesIO(image_data))
            
            # ALWAYS convert to RGB for WordPress JPEG compatibility
            if img.mode != 'RGB':
                # Handle transparency by adding white background
                if img.mode in ('RGBA', 'LA'):
                    white_bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        white_bg.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    else:
                        white_bg.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = white_bg
                else:
                    img = img.convert('RGB')
            
            # Resize if needed
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # ALWAYS save as JPEG for maximum WordPress compatibility
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            optimized_data = output.getvalue()
            
            print(f"   🎨 Image optimized: {len(image_data):,} → {len(optimized_data):,} bytes ({img.size[0]}x{img.size[1]}) as JPEG")
            return optimized_data
            
        except Exception as e:
            print(f"   ⚠️ Image optimization failed: {e}")
            # Even if optimization fails, try to convert to basic JPEG
            try:
                img = Image.open(io.BytesIO(image_data))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85)
                return output.getvalue()
            except:
                return image_data  # Last resort: return original
    
    def download_image_with_retry(self, url: str, optimize: bool = True) -> Optional[bytes]:
        """Download image with retry logic - SYNCHRONOUS VERSION"""
        if not url:
            return None
        
        self.download_attempts += 1
        
        for attempt in range(self.max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; BulletproofUploader/1.0)',
                    'Accept': 'image/*,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
                
                print(f"   🔄 Download attempt {attempt + 1}/{self.max_retries}: {url[:60]}...")
                
                response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    print(f"   ⚠️ Invalid content type: {content_type}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                # Download content
                image_data = response.content
                
                if len(image_data) < 1024:  # Less than 1KB
                    print(f"   ⚠️ Image too small: {len(image_data)} bytes")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None
                
                # Optimize if requested
                if optimize:
                    image_data = self.optimize_image(image_data)
                
                print(f"   ✅ Downloaded: {len(image_data):,} bytes")
                self.download_successes += 1
                return image_data
                
            except Exception as e:
                print(f"   ❌ Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"   💥 Download failed after {self.max_retries} attempts")
        
        return None
    
    def upload_image_to_wordpress(self, image_data: bytes, filename: str, 
                                 title: str = "", alt_text: str = "") -> Dict:
        """Upload image to WordPress - SYNCHRONOUS VERSION"""
        if not image_data:
            return {'success': False, 'error': 'No image data'}
        
        self.upload_attempts += 1
        
        for attempt in range(self.max_retries):
            try:
                print(f"   📤 Upload attempt {attempt + 1}/{self.max_retries}: {filename}")
                
                # Prepare upload
                url = f"{self.wp_credentials.base_url}/media"
                
                # Force JPG extension for WordPress compatibility
                if not filename.lower().endswith(('.jpg', '.jpeg')):
                    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
                    filename = f"{base_name}.jpg"
                
                files = {
                    'file': (filename, image_data, 'image/jpeg')
                }
                
                data = {}
                if title:
                    data['title'] = title
                if alt_text:
                    data['alt_text'] = alt_text
                
                headers = {
                    'Authorization': self.wp_credentials.auth_header
                }
                
                # Upload
                response = requests.post(url, files=files, data=data, headers=headers, timeout=self.timeout)
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    
                    media_id = result.get('id')
                    source_url = result.get('source_url')
                    
                    if media_id and source_url:
                        print(f"   ✅ Uploaded: ID {media_id}")
                        self.upload_successes += 1
                        return {
                            'success': True,
                            'id': media_id,
                            'source_url': source_url,
                            'url': source_url,  # Alias for compatibility
                            'title': title,
                            'alt_text': alt_text
                        }
                    else:
                        print(f"   ⚠️ Upload succeeded but missing data: {result}")
                
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get('message', error_msg)
                except:
                    pass
                
                print(f"   ❌ Upload attempt {attempt + 1} failed: {error_msg}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {'success': False, 'error': f'Upload failed after {self.max_retries} attempts: {error_msg}'}
                    
            except Exception as e:
                print(f"   ❌ Upload attempt {attempt + 1} exception: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {'success': False, 'error': f'Upload failed after {self.max_retries} attempts: {str(e)}'}
        
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def process_images_batch(self, image_urls: List[str], base_filename: str = "image") -> List[Dict]:
        """Process a batch of images with bulletproof downloading and uploading"""
        print(f"🔫 Bulletproof processing {len(image_urls)} images...")
        
        results = []
        
        for i, url in enumerate(image_urls):
            if not url:
                continue
            
            filename = f"{base_filename}_{i+1}.jpg"
            print(f"\n🔫 Processing image {i+1}/{len(image_urls)}")
            
            # Download
            image_data = self.download_image_with_retry(url)
            if not image_data:
                results.append({'success': False, 'url': url, 'error': 'Download failed'})
                continue
            
            # Upload
            upload_result = self.upload_image_to_wordpress(
                image_data, filename, 
                title=f"Article Image {i+1}",
                alt_text=f"Image {i+1}"
            )
            
            if upload_result.get('success'):
                results.append(upload_result)
            else:
                results.append({'success': False, 'url': url, 'error': upload_result.get('error')})
        
        successful = len([r for r in results if r.get('success')])
        print(f"\n🔫 Bulletproof batch complete: {successful}/{len(image_urls)} successful")
        
        return results
    
    def get_stats(self) -> Dict:
        """Get upload statistics"""
        download_rate = (self.download_successes / self.download_attempts * 100) if self.download_attempts > 0 else 0
        upload_rate = (self.upload_successes / self.upload_attempts * 100) if self.upload_attempts > 0 else 0
        
        return {
            'download_attempts': self.download_attempts,
            'download_successes': self.download_successes,
            'download_success_rate': f"{download_rate:.1f}%",
            'upload_attempts': self.upload_attempts,
            'upload_successes': self.upload_successes,
            'upload_success_rate': f"{upload_rate:.1f}%"
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.download_attempts = 0
        self.download_successes = 0
        self.upload_attempts = 0
        self.upload_successes = 0


def create_wordpress_credentials_from_env() -> Optional[WordPressCredentials]:
    """Create WordPress credentials from environment variables"""
    wordpress_url = os.environ.get('WORDPRESS_URL')
    wordpress_username = os.environ.get('WORDPRESS_USERNAME') 
    wordpress_password = os.environ.get('WORDPRESS_PASSWORD')
    
    if wordpress_url and wordpress_username and wordpress_password:
        return WordPressCredentials(wordpress_url, wordpress_username, wordpress_password)
    return None


def create_bulletproof_uploader() -> Optional[BulletproofImageUploader]:
    """Create bulletproof uploader from environment variables"""
    credentials = create_wordpress_credentials_from_env()
    if credentials:
        return BulletproofImageUploader(credentials)
    return None


# Utility functions
async def test_bulletproof_uploader():
    """Test the bulletproof uploader"""
    print("🧪 Testing Bulletproof Image Uploader V1")
    
    try:
        uploader = create_bulletproof_uploader()
        if not uploader:
            print("❌ No WordPress credentials found in environment")
            return
        
        # Test with a sample image URL
        test_url = "https://via.placeholder.com/800x600/0066cc/ffffff?text=Test+Image"
        
        print(f"🔍 Testing download: {test_url}")
        image_data = uploader.download_image_with_retry(test_url)
        
        if image_data:
            print("✅ Download successful")
            
            result = uploader.upload_image_to_wordpress(
                image_data, "test_image.jpg", 
                title="Test Image V1", 
                alt_text="Bulletproof test image V1"
            )
            
            if result.get('success'):
                print(f"✅ Upload successful: ID {result['id']}")
            else:
                print(f"❌ Upload failed: {result.get('error')}")
        else:
            print("❌ Download failed")
        
        print(f"\n📊 Stats: {uploader.get_stats()}")
        
    except Exception as e:
        print(f"❌ Test error: {e}")


if __name__ == "__main__":
    print("🔫 Bulletproof Image Uploader V1")
    asyncio.run(test_bulletproof_uploader()) 