#!/usr/bin/env python3
"""
WordPress REST API Publisher for Universal RAG CMS
Enterprise-grade WordPress integration with advanced features

This module provides comprehensive WordPress publishing capabilities including:
- Multi-authentication support (Application Passwords, JWT, OAuth2)
- Bulletproof image processing with retry mechanisms
- Rich HTML formatting and responsive design
- Smart contextual image embedding
- Comprehensive error recovery
- Integration with Tasks 1 & 5 (Supabase + DataForSEO)
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import aiohttp
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from supabase import create_client, Client

# Import casino intelligence schema
try:
    from ..schemas.casino_intelligence_schema import CasinoIntelligence
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Casino intelligence schema not found. Casino-specific features disabled.")
    CasinoIntelligence = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WordPressConfig:
    """WordPress configuration with environment-driven defaults"""
    site_url: str = field(default_factory=lambda: os.getenv("WORDPRESS_SITE_URL", ""))
    username: str = field(default_factory=lambda: os.getenv("WORDPRESS_USERNAME", ""))
    application_password: str = field(default_factory=lambda: os.getenv("WORDPRESS_APP_PASSWORD", ""))
    jwt_token: Optional[str] = field(default_factory=lambda: os.getenv("WORDPRESS_JWT_TOKEN"))
    oauth_token: Optional[str] = field(default_factory=lambda: os.getenv("WORDPRESS_OAUTH_TOKEN"))
    
    # Publishing defaults
    default_status: str = "publish"  # draft, publish, private
    default_author_id: int = 1
    default_category_ids: List[int] = field(default_factory=list)
    default_tags: List[str] = field(default_factory=list)
    
    # Performance settings
    max_concurrent_uploads: int = 3
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Image processing
    max_image_size: Tuple[int, int] = (1920, 1080)
    image_quality: int = 85
    image_formats: Set[str] = field(default_factory=lambda: {"JPEG", "PNG", "WebP"})
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.site_url:
            raise ValueError("WordPress site URL is required")
        if not self.username:
            raise ValueError("WordPress username is required")
        if not (self.application_password or self.jwt_token or self.oauth_token):
            raise ValueError("At least one authentication method is required")

class WordPressAuthManager:
    """Multi-authentication manager for WordPress REST API"""
    
    def __init__(self, config: WordPressConfig):
        self.config = config
        self._auth_headers = {}
        self._auth_method = None
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup authentication headers based on available credentials"""
        if self.config.application_password:
            # Application Password authentication (recommended)
            credentials = f"{self.config.username}:{self.config.application_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            self._auth_headers = {
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/json",
                "User-Agent": "UniversalRAGCMS/1.0"
            }
            self._auth_method = "application_password"
            logger.info("Using Application Password authentication")
            
        elif self.config.jwt_token:
            # JWT authentication
            self._auth_headers = {
                "Authorization": f"Bearer {self.config.jwt_token}",
                "Content-Type": "application/json",
                "User-Agent": "UniversalRAGCMS/1.0"
            }
            self._auth_method = "jwt"
            logger.info("Using JWT authentication")
            
        elif self.config.oauth_token:
            # OAuth2 authentication
            self._auth_headers = {
                "Authorization": f"Bearer {self.config.oauth_token}",
                "Content-Type": "application/json",
                "User-Agent": "UniversalRAGCMS/1.0"
            }
            self._auth_method = "oauth2"
            logger.info("Using OAuth2 authentication")
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return self._auth_headers.copy()
    
    @property
    def auth_method(self) -> str:
        """Get current authentication method"""
        return self._auth_method
    
    async def verify_authentication(self, session: aiohttp.ClientSession) -> bool:
        """Verify authentication with WordPress"""
        try:
            url = urljoin(self.config.site_url, "/wp-json/wp/v2/users/me")
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    user_data = await response.json()
                    logger.info(f"Authentication verified for user: {user_data.get('name', 'Unknown')}")
                    return True
                else:
                    logger.error(f"Authentication failed with status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Authentication verification failed: {e}")
            return False

class BulletproofImageProcessor:
    """Advanced image processing with retry mechanisms and optimization"""
    
    def __init__(self, config: WordPressConfig):
        self.config = config
        self.supported_formats = {
            'image/jpeg': 'JPEG',
            'image/png': 'PNG',
            'image/webp': 'WebP',
            'image/gif': 'GIF'
        }
    
    async def process_image(self, image_data: bytes, filename: str) -> Tuple[bytes, str, Dict[str, Any]]:
        """Process image with optimization and format conversion"""
        try:
            # Open image
            with Image.open(BytesIO(image_data)) as img:
                # Get original metadata
                original_size = img.size
                original_format = img.format
                original_mode = img.mode
                
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
                if img.size[0] > self.config.max_image_size[0] or img.size[1] > self.config.max_image_size[1]:
                    img.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)
                
                # Apply auto-orientation
                img = ImageOps.exif_transpose(img)
                
                # Save optimized image
                output = BytesIO()
                img.save(
                    output,
                    format='JPEG',
                    quality=self.config.image_quality,
                    optimize=True,
                    progressive=True
                )
                
                processed_data = output.getvalue()
                
                # Generate metadata
                metadata = {
                    'original_size': original_size,
                    'processed_size': img.size,
                    'original_format': original_format,
                    'processed_format': 'JPEG',
                    'original_mode': original_mode,
                    'compression_ratio': len(processed_data) / len(image_data),
                    'file_size_reduction': len(image_data) - len(processed_data)
                }
                
                # Generate optimized filename
                name_parts = os.path.splitext(filename)
                optimized_filename = f"{name_parts[0]}_optimized.jpg"
                
                logger.info(f"Image processed: {filename} -> {optimized_filename} "
                          f"({len(image_data)} -> {len(processed_data)} bytes)")
                
                return processed_data, optimized_filename, metadata
                
        except Exception as e:
            logger.error(f"Image processing failed for {filename}: {e}")
            # Return original data if processing fails
            return image_data, filename, {'error': str(e)}
    
    async def download_image(self, session: aiohttp.ClientSession, url: str) -> Optional[Tuple[bytes, str]]:
        """Download image with retry mechanism"""
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)) as response:
                    if response.status == 200:
                        data = await response.read()
                        # Extract filename from URL or Content-Disposition
                        filename = self._extract_filename(url, response.headers)
                        return data, filename
                    else:
                        logger.warning(f"Failed to download image {url}: HTTP {response.status}")
                        
            except Exception as e:
                logger.warning(f"Image download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        return None
    
    def _extract_filename(self, url: str, headers: Dict[str, str]) -> str:
        """Extract filename from URL or headers"""
        # Try Content-Disposition header first
        if 'content-disposition' in headers:
            cd = headers['content-disposition']
            filename_match = re.search(r'filename="([^"]+)"', cd)
            if filename_match:
                return filename_match.group(1)
        
        # Extract from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if filename and '.' in filename:
            return filename
        
        # Generate fallback filename
        return f"image_{uuid4().hex[:8]}.jpg"

class RichHTMLFormatter:
    """Advanced HTML formatting with responsive design and SEO optimization"""
    
    def __init__(self):
        self.soup_parser = 'html.parser'
    
    def format_content(self, content: str, title: str = "", meta_description: str = "") -> str:
        """Format content with rich HTML structure"""
        soup = BeautifulSoup(content, self.soup_parser)
        
        # Add structured data and meta information
        if title:
            soup = self._add_title_structure(soup, title)
        
        # Enhance paragraphs and text formatting
        soup = self._enhance_typography(soup)
        
        # Add responsive image classes
        soup = self._make_images_responsive(soup)
        
        # Add table styling
        soup = self._enhance_tables(soup)
        
        # Add call-to-action styling
        soup = self._enhance_cta_elements(soup)
        
        # Add schema markup for gaming content
        soup = self._add_schema_markup(soup, title, meta_description)
        
        return str(soup)
    
    def _add_title_structure(self, soup: BeautifulSoup, title: str) -> BeautifulSoup:
        """Add proper heading structure"""
        # Ensure proper H1 if not present
        if not soup.find('h1'):
            h1 = soup.new_tag('h1', **{'class': 'entry-title'})
            h1.string = title
            if soup.body:
                soup.body.insert(0, h1)
            else:
                soup.insert(0, h1)
        
        return soup
    
    def _enhance_typography(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Enhance typography with proper classes"""
        # Add classes to paragraphs
        for p in soup.find_all('p'):
            if not p.get('class'):
                p['class'] = ['content-paragraph']
        
        # Enhance lists
        for ul in soup.find_all('ul'):
            if not ul.get('class'):
                ul['class'] = ['content-list']
        
        for ol in soup.find_all('ol'):
            if not ol.get('class'):
                ol['class'] = ['content-ordered-list']
        
        return soup
    
    def _make_images_responsive(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Add responsive classes to images"""
        for img in soup.find_all('img'):
            current_classes = img.get('class', [])
            if isinstance(current_classes, str):
                current_classes = current_classes.split()
            
            current_classes.extend(['responsive-image', 'wp-image'])
            img['class'] = current_classes
            
            # Add loading lazy if not present
            if not img.get('loading'):
                img['loading'] = 'lazy'
        
        return soup
    
    def _enhance_tables(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Add responsive table styling"""
        for table in soup.find_all('table'):
            current_classes = table.get('class', [])
            if isinstance(current_classes, str):
                current_classes = current_classes.split()
            
            current_classes.extend(['wp-table', 'responsive-table'])
            table['class'] = current_classes
        
        return soup
    
    def _enhance_cta_elements(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Enhance call-to-action elements"""
        # Look for common CTA patterns
        cta_patterns = ['play now', 'sign up', 'join', 'register', 'get bonus']
        
        for a in soup.find_all('a'):
            link_text = a.get_text().lower()
            if any(pattern in link_text for pattern in cta_patterns):
                current_classes = a.get('class', [])
                if isinstance(current_classes, str):
                    current_classes = current_classes.split()
                
                current_classes.extend(['cta-button', 'wp-element-button'])
                a['class'] = current_classes
        
        return soup
    
    def _add_schema_markup(self, soup: BeautifulSoup, title: str, description: str) -> BeautifulSoup:
        """Add JSON-LD schema markup for gaming content"""
        if title and description:
            schema = {
                "@context": "https://schema.org",
                "@type": "Article",
                "headline": title,
                "description": description,
                "datePublished": datetime.now().isoformat(),
                "author": {
                    "@type": "Organization",
                    "name": "Crash Casino"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Crash Casino",
                    "url": "https://crashcasino.io"
                }
            }
            
            script = soup.new_tag('script', type='application/ld+json')
            script.string = json.dumps(schema, indent=2)
            
            if soup.head:
                soup.head.append(script)
            else:
                soup.insert(0, script)
        
        return soup

class ErrorRecoveryManager:
    """Enterprise-grade error handling and recovery"""
    
    def __init__(self, config: WordPressConfig):
        self.config = config
        self.error_counts = {}
        self.circuit_breaker_states = {}
    
    async def execute_with_retry(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Operation failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        logger.error(f"Operation failed after {self.config.retry_attempts} attempts")
        raise last_exception
    
    def circuit_breaker(self, operation_name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Circuit breaker decorator for operations"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                now = time.time()
                
                # Check circuit breaker state
                if operation_name in self.circuit_breaker_states:
                    state = self.circuit_breaker_states[operation_name]
                    if state['state'] == 'open':
                        if now - state['last_failure'] < recovery_timeout:
                            raise Exception(f"Circuit breaker open for {operation_name}")
                        else:
                            # Try to close circuit breaker
                            state['state'] = 'half-open'
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Reset on success
                    if operation_name in self.circuit_breaker_states:
                        del self.circuit_breaker_states[operation_name]
                    
                    return result
                    
                except Exception as e:
                    # Track failures
                    if operation_name not in self.error_counts:
                        self.error_counts[operation_name] = 0
                    
                    self.error_counts[operation_name] += 1
                    
                    # Open circuit breaker if threshold reached
                    if self.error_counts[operation_name] >= failure_threshold:
                        self.circuit_breaker_states[operation_name] = {
                            'state': 'open',
                            'last_failure': now
                        }
                        logger.error(f"Circuit breaker opened for {operation_name}")
                    
                    raise e
            
            return wrapper
        return decorator

class WordPressRESTPublisher:
    """Main WordPress publisher with enterprise features"""
    
    def __init__(self, config: WordPressConfig, supabase_client: Optional[Client] = None, llm=None):
        self.config = config
        self.auth_manager = WordPressAuthManager(config)
        self.image_processor = BulletproofImageProcessor(config)
        self.html_formatter = RichHTMLFormatter()
        self.error_manager = ErrorRecoveryManager(config)
        self.supabase = supabase_client
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 🏗️ NATIVE LANGCHAIN: LLM integration for metadata generation
        self.llm = llm
        if not self.llm:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                logger.info("Initialized ChatOpenAI for WordPress metadata generation")
            except ImportError:
                logger.warning("LangChain OpenAI not available. Casino metadata will use fallback extraction.")
                self.llm = None
        
        # Performance tracking
        self.stats = {
            'posts_published': 0,
            'images_processed': 0,
            'errors_recovered': 0,
            'total_processing_time': 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_uploads)
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        # Verify authentication
        if not await self.auth_manager.verify_authentication(self.session):
            raise Exception("WordPress authentication failed")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_wp_request(self, method: str, endpoint: str, json: Dict[str, Any] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        ✅ FIXED: Generic WordPress REST API request method for MT Casino custom post types
        This method was missing and causing the MT Casino publishing to fail
        """
        try:
            # Ensure endpoint starts with proper base URL
            if endpoint.startswith('/wp-json/'):
                url = f"{self.config.site_url.rstrip('/')}{endpoint}"
            else:
                url = f"{self.config.site_url.rstrip('/')}/wp-json/wp/v2/{endpoint.lstrip('/')}"
            
            # Prepare request parameters
            request_kwargs = {
                'headers': self.auth_manager.headers,
                'timeout': aiohttp.ClientTimeout(total=self.config.request_timeout)
            }
            
            # Add JSON data if provided
            if json:
                request_kwargs['json'] = json
            
            # Add any additional kwargs
            request_kwargs.update(kwargs)
            
            logger.info(f"🔧 Making {method} request to: {url}")
            
            # Make the request based on method
            if method.upper() == 'POST':
                async with self.session.post(url, **request_kwargs) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        logger.info(f"✅ {method} request successful: {response.status}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ {method} request failed: {response.status} - {error_text}")
                        return None
                        
            elif method.upper() == 'GET':
                async with self.session.get(url, **request_kwargs) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ {method} request successful: {response.status}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ {method} request failed: {response.status} - {error_text}")
                        return None
                        
            elif method.upper() in ['PUT', 'PATCH']:
                async with self.session.request(method, url, **request_kwargs) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        logger.info(f"✅ {method} request successful: {response.status}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ {method} request failed: {response.status} - {error_text}")
                        return None
            else:
                logger.error(f"❌ Unsupported HTTP method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"❌ WordPress API request failed: {e}")
            return None
    
    async def publish_post(self, 
                          title: str, 
                          content: str, 
                          status: str = None,
                          featured_image_url: str = None,
                          categories: List[int] = None,
                          tags: List[str] = None,
                          meta_description: str = "",
                          custom_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """Publish a post to WordPress with all enhancements"""
        start_time = time.time()
        
        try:
            # 🔧 NEW: Process embedded images in content FIRST (before formatting)
            processed_content, embedded_media_count = await self._process_embedded_images(content)
            
            # Format content with rich HTML (now with WordPress-hosted images)
            formatted_content = self.html_formatter.format_content(processed_content, title, meta_description)
            
            # Process featured image if provided
            featured_media_id = None
            if featured_image_url:
                featured_media_id = await self._upload_featured_image(featured_image_url, title)
            
            # Prepare post data
            post_data = {
                'title': title,
                'content': formatted_content,
                'status': status or self.config.default_status,
                'author': self.config.default_author_id,
                'excerpt': meta_description[:150] if meta_description else "",
                'meta': custom_fields or {}
            }
            
            # Add categories and tags (ensure categories are integers)
            if categories:
                post_data['categories'] = [int(cat) if isinstance(cat, (str, float)) else cat for cat in categories]
            elif self.config.default_category_ids:
                post_data['categories'] = [int(cat) if isinstance(cat, (str, float)) else cat for cat in self.config.default_category_ids]
            
            if tags:
                post_data['tags'] = await self._get_or_create_tags(tags)
            elif self.config.default_tags:
                post_data['tags'] = await self._get_or_create_tags(self.config.default_tags)
            
            # Add featured media
            if featured_media_id:
                post_data['featured_media'] = featured_media_id
            
            # Publish post
            url = urljoin(self.config.site_url, "/wp-json/wp/v2/posts")
            async with self.session.post(url, 
                                       headers=self.auth_manager.headers, 
                                       json=post_data) as response:
                
                if response.status in [200, 201]:
                    result = await response.json()
                    
                    # 🔧 NEW: Add embedded media info to result
                    result['embedded_media_count'] = embedded_media_count
                    
                    # Log to Supabase if available
                    if self.supabase:
                        await self._log_publication(result, start_time)
                    
                    # Update stats
                    self.stats['posts_published'] += 1
                    self.stats['total_processing_time'] += time.time() - start_time
                    
                    logger.info(f"Post published successfully: {result['id']} with {embedded_media_count} embedded images")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"WordPress API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Failed to publish post '{title}': {e}")
            raise
    
    async def _process_embedded_images(self, content: str) -> Tuple[str, int]:
        """🔧 NEW: Simple embedded image processor using existing V1 BulletproofImageProcessor"""
        if not content or '<img' not in content:
            return content, 0
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            embedded_count = 0
            
            # Find all img tags with external URLs
            for img_tag in soup.find_all('img'):
                src = img_tag.get('src')
                if not src or src.startswith('data:') or src.startswith('/wp-content/'):
                    continue  # Skip data URLs and WordPress-hosted images
                
                try:
                    # Use existing V1 image processor to download and upload
                    logger.info(f"📥 Processing embedded image: {src}")
                    
                    # Download using existing V1 method
                    image_data = await self.image_processor.download_image(self.session, src)
                    if not image_data:
                        continue
                    
                    data, filename = image_data
                    
                    # Process using existing V1 method
                    processed_data, processed_filename, metadata = await self.image_processor.process_image(data, filename)
                    
                    # Upload using existing V1 approach
                    media_url = urljoin(self.config.site_url, "/wp-json/wp/v2/media")
                    form_data = aiohttp.FormData()
                    form_data.add_field('file', processed_data, filename=processed_filename, content_type='image/jpeg')
                    form_data.add_field('alt_text', img_tag.get('alt', 'Embedded image'))
                    
                    headers = self.auth_manager.headers.copy()
                    headers.pop('Content-Type', None)
                    
                    async with self.session.post(media_url, headers=headers, data=form_data) as response:
                        if response.status in [200, 201]:
                            result = await response.json()
                            # Replace external URL with WordPress URL
                            img_tag['src'] = result['source_url']
                            # Add WordPress classes
                            img_tag['class'] = f"wp-image-{result['id']} aligncenter size-full"
                            embedded_count += 1
                            self.stats['images_processed'] += 1
                            logger.info(f"✅ Uploaded embedded image: {result['id']}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process embedded image {src}: {e}")
                    continue
            
            return str(soup), embedded_count
            
        except ImportError:
            logger.warning("BeautifulSoup not available for embedded image processing")
            return content, 0
        except Exception as e:
            logger.error(f"Error processing embedded images: {e}")
            return content, 0
    
    async def _upload_featured_image(self, image_url: str, alt_text: str = "") -> Optional[int]:
        """Upload and set featured image"""
        try:
            # Download image
            image_data = await self.image_processor.download_image(self.session, image_url)
            if not image_data:
                return None
            
            data, filename = image_data
            
            # Process image
            processed_data, processed_filename, metadata = await self.image_processor.process_image(data, filename)
            
            # Upload to WordPress media library
            media_url = urljoin(self.config.site_url, "/wp-json/wp/v2/media")
            
            # Prepare multipart data
            form_data = aiohttp.FormData()
            form_data.add_field('file', processed_data, filename=processed_filename, content_type='image/jpeg')
            form_data.add_field('alt_text', alt_text)
            form_data.add_field('caption', f"Optimized image - {metadata.get('compression_ratio', 1):.2f}x compression")
            
            # Upload
            headers = self.auth_manager.headers.copy()
            headers.pop('Content-Type', None)  # Let aiohttp set this for multipart
            
            async with self.session.post(media_url, headers=headers, data=form_data) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    self.stats['images_processed'] += 1
                    logger.info(f"Image uploaded successfully: {result['id']}")
                    return result['id']
                else:
                    error_text = await response.text()
                    logger.error(f"Image upload failed: {error_text}")
                    return None
        
        except Exception as e:
            logger.error(f"Failed to upload featured image: {e}")
            return None
    
    async def _get_or_create_tags(self, tag_names: List[str]) -> List[int]:
        """Get existing tags or create new ones"""
        tag_ids = []
        
        for tag_name in tag_names:
            try:
                # Search for existing tag
                search_url = urljoin(self.config.site_url, f"/wp-json/wp/v2/tags?search={tag_name}")
                async with self.session.get(search_url, headers=self.auth_manager.headers) as response:
                    if response.status == 200:
                        tags = await response.json()
                        if tags:
                            tag_ids.append(tags[0]['id'])
                            continue
                
                # Create new tag if not found
                create_url = urljoin(self.config.site_url, "/wp-json/wp/v2/tags")
                tag_data = {'name': tag_name, 'slug': tag_name.lower().replace(' ', '-')}
                
                async with self.session.post(create_url, 
                                           headers=self.auth_manager.headers, 
                                           json=tag_data) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        tag_ids.append(result['id'])
                    else:
                        logger.warning(f"Failed to create tag '{tag_name}'")
            
            except Exception as e:
                logger.warning(f"Error processing tag '{tag_name}': {e}")
        
        return tag_ids
    
    async def _log_publication(self, post_result: Dict[str, Any], start_time: float):
        """Log publication to Supabase audit trail"""
        try:
            log_data = {
                'wordpress_post_id': post_result['id'],
                'title': post_result['title']['rendered'],
                'status': post_result['status'],
                'url': post_result['link'],
                'processing_time': time.time() - start_time,
                'auth_method': self.auth_manager.auth_method,
                'published_at': datetime.now().isoformat(),
                'metadata': {
                    'featured_media': post_result.get('featured_media'),
                    'categories': post_result.get('categories', []),
                    'tags': post_result.get('tags', [])
                }
            }
            
            self.supabase.table('wordpress_publications').insert(log_data).execute()
            logger.info(f"Publication logged to Supabase: {post_result['id']}")
            
        except Exception as e:
            logger.warning(f"Failed to log publication to Supabase: {e}")
    
    async def publish_casino_content(self, 
                                    title: str, 
                                    content: str, 
                                    structured_casino_data: Optional[Dict[str, Any]] = None,
                                    status: str = None,
                                    featured_image_url: str = None) -> Dict[str, Any]:
        """🎰 NATIVE LANGCHAIN: Publish casino content with 95-field intelligence using LangChain components"""
        
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
        from pydantic import BaseModel, Field
        
        try:
            # Define WordPress metadata schema using Pydantic
            class CasinoWordPressMetadata(BaseModel):
                """🏗️ NATIVE LANGCHAIN: WordPress metadata schema for casino content"""
                yoast_title: str = Field(description="SEO-optimized title for Yoast")
                yoast_description: str = Field(description="Meta description for Yoast")
                casino_name: str = Field(description="Primary casino name")
                overall_rating: Optional[float] = Field(default=7.5, description="Overall rating 0-10", ge=0, le=10)
                license_status: str = Field(description="License verification status")
                game_provider_count: Optional[int] = Field(default=0, description="Number of game providers", ge=0)
                payment_method_count: Optional[int] = Field(default=0, description="Number of payment methods", ge=0)
                welcome_bonus: str = Field(description="Primary welcome bonus")
                content_category: str = Field(description="WordPress content category")
                tags: List[str] = Field(description="WordPress tags for categorization")
                custom_fields: Dict[str, Any] = Field(description="WordPress custom fields")
            
            # Create PydanticOutputParser for metadata extraction
            metadata_parser = PydanticOutputParser(pydantic_object=CasinoWordPressMetadata)
            
            # Build LangChain chain for metadata generation
            metadata_prompt = ChatPromptTemplate.from_template("""
            You are a WordPress SEO specialist. Extract and generate optimized metadata for this casino content.
            
            Content Title: {title}
            Content Preview: {content_preview}
            Structured Casino Data: {casino_data}
            
            Generate WordPress-optimized metadata including SEO title, description, categorization, and custom fields.
            
            {format_instructions}
            """)
            
            # Create LCEL chain for metadata generation
            metadata_chain = (
                RunnableParallel({
                    "title": RunnablePassthrough(),
                    "content_preview": RunnableLambda(lambda x: x["content"][:500] + "..."),
                    "casino_data": RunnableLambda(lambda x: str(x.get("structured_casino_data", {}))),
                    "format_instructions": RunnableLambda(lambda x: metadata_parser.get_format_instructions())
                })
                | metadata_prompt
                | self.llm if hasattr(self, 'llm') else None
                | metadata_parser
            )
            
            # Generate metadata using LangChain if LLM is available
            wordpress_metadata = None
            if hasattr(self, 'llm') and self.llm:
                try:
                    wordpress_metadata = await metadata_chain.ainvoke({
                        "title": title,
                        "content": content,
                        "structured_casino_data": structured_casino_data
                    })
                except Exception as e:
                    logger.warning(f"LangChain metadata generation failed, using fallback: {e}")
            
            # Fallback metadata extraction if LangChain fails
            if not wordpress_metadata:
                wordpress_metadata = self._extract_casino_metadata_fallback(title, content, structured_casino_data)
            
            # Convert to WordPress custom fields
            custom_fields = self._convert_to_wordpress_custom_fields(structured_casino_data, wordpress_metadata)
            
            # Determine categories and tags
            categories = await self._determine_casino_categories(wordpress_metadata)
            tags = wordpress_metadata.tags if hasattr(wordpress_metadata, 'tags') else []
            
            # Publish using existing publish_post method with enhanced metadata
            return await self.publish_post(
                title=wordpress_metadata.yoast_title if hasattr(wordpress_metadata, 'yoast_title') else title,
                content=content,
                status=status,
                featured_image_url=featured_image_url,
                categories=categories,
                tags=tags,
                meta_description=wordpress_metadata.yoast_description if hasattr(wordpress_metadata, 'yoast_description') else "",
                custom_fields=custom_fields
            )
            
        except Exception as e:
            logger.error(f"Casino content publishing failed: {e}")
            # Fallback to standard publishing
            return await self.publish_post(title, content, status, featured_image_url)
    
    def _extract_casino_metadata_fallback(self, title: str, content: str, structured_data: Optional[Dict[str, Any]]) -> Any:
        """Fallback metadata extraction when LangChain is unavailable"""
        from dataclasses import dataclass
        from typing import List, Dict, Any
        
        @dataclass
        class FallbackMetadata:
            yoast_title: str
            yoast_description: str
            casino_name: str
            overall_rating: float
            license_status: str
            game_provider_count: int
            payment_method_count: int
            welcome_bonus: str
            content_category: str
            tags: List[str]
            custom_fields: Dict[str, Any]
        
        # Extract basic info from structured data or content
        casino_name = "Unknown Casino"
        overall_rating = 7.5
        
        if structured_data:
            casino_name = structured_data.get('casino_name', casino_name)
            overall_rating = structured_data.get('overall_rating', overall_rating)
        
        return FallbackMetadata(
            yoast_title=f"{title} - Complete Review & Analysis",
            yoast_description=f"Comprehensive review of {casino_name}. Rating: {overall_rating}/10. Licensed casino with detailed analysis.",
            casino_name=casino_name,
            overall_rating=overall_rating,
            license_status="Verified" if structured_data else "Unknown",
            game_provider_count=len(structured_data.get('gaming', {}).get('game_providers', [])) if structured_data else 0,
            payment_method_count=len(structured_data.get('banking', {}).get('deposit_methods', [])) if structured_data else 0,
            welcome_bonus=structured_data.get('promotions', {}).get('welcome_bonus', {}).get('title', "Available") if structured_data else "Available",
            content_category="casino-reviews",
            tags=[casino_name.lower().replace(' ', '-'), 'casino-review', 'online-casino'],
            custom_fields={}
        )
    
    def _convert_to_wordpress_custom_fields(self, structured_data: Optional[Dict[str, Any]], metadata: Any) -> Dict[str, Any]:
        """🏗️ NATIVE LANGCHAIN: Convert 95-field casino data to WordPress custom fields"""
        
        custom_fields = {}
        
        # Basic metadata
        if hasattr(metadata, 'casino_name'):
            custom_fields['casino_name'] = metadata.casino_name
            custom_fields['overall_rating'] = str(metadata.overall_rating)
            custom_fields['license_status'] = metadata.license_status
        
        # Convert 95-field structured data to WordPress custom fields
        if structured_data:
            # Trustworthiness fields
            if 'trustworthiness' in structured_data:
                trust = structured_data['trustworthiness']
                custom_fields.update({
                    'license_info': json.dumps(trust.get('license_info', {})),
                    'security_measures': json.dumps(trust.get('security_measures', {})),
                    'fair_play_certification': json.dumps(trust.get('fair_play_certification', {}))
                })
            
            # Gaming fields
            if 'gaming' in structured_data:
                gaming = structured_data['gaming']
                custom_fields.update({
                    'game_providers': json.dumps(gaming.get('game_providers', [])),
                    'game_categories': json.dumps(gaming.get('game_categories', {})),
                    'live_casino': json.dumps(gaming.get('live_casino', {}))
                })
            
            # Banking fields
            if 'banking' in structured_data:
                banking = structured_data['banking']
                custom_fields.update({
                    'deposit_methods': json.dumps(banking.get('deposit_methods', [])),
                    'withdrawal_methods': json.dumps(banking.get('withdrawal_methods', [])),
                    'processing_times': json.dumps(banking.get('processing_times', {})),
                    'transaction_limits': json.dumps(banking.get('transaction_limits', {}))
                })
            
            # Promotions fields
            if 'promotions' in structured_data:
                promotions = structured_data['promotions']
                custom_fields.update({
                    'welcome_bonus': json.dumps(promotions.get('welcome_bonus', {})),
                    'ongoing_promotions': json.dumps(promotions.get('ongoing_promotions', [])),
                    'loyalty_program': json.dumps(promotions.get('loyalty_program', {}))
                })
            
            # User Experience fields
            if 'user_experience' in structured_data:
                ux = structured_data['user_experience']
                custom_fields.update({
                    'website_design': json.dumps(ux.get('website_design', {})),
                    'mobile_experience': json.dumps(ux.get('mobile_experience', {})),
                    'customer_support': json.dumps(ux.get('customer_support', {}))
                })
            
            # Analytics fields
            if 'analytics' in structured_data:
                analytics = structured_data['analytics']
                custom_fields.update({
                    'data_completeness': str(analytics.get('data_completeness', 0)),
                    'last_updated': analytics.get('last_updated', ''),
                    'data_sources': json.dumps(analytics.get('data_sources', []))
                })
        
        # Add Yoast SEO fields
        if hasattr(metadata, 'yoast_title'):
            custom_fields['_yoast_wpseo_title'] = metadata.yoast_title
            custom_fields['_yoast_wpseo_metadesc'] = metadata.yoast_description
        
        return custom_fields
    
    async def _determine_casino_categories(self, metadata: Any) -> List[int]:
        """Determine WordPress categories for casino content"""
        # This would typically map to actual WordPress category IDs
        # For now, return default casino review category
        return [1]  # Default category ID - should be configured per WordPress site
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_processing_time = (self.stats['total_processing_time'] / 
                             max(self.stats['posts_published'], 1))
        
        return {
            **self.stats,
            'average_processing_time': avg_processing_time,
            'auth_method': self.auth_manager.auth_method,
            'success_rate': (self.stats['posts_published'] / 
                           max(self.stats['posts_published'] + self.stats['errors_recovered'], 1))
        }

    async def upload_screenshot_to_wordpress(
        self, 
        screenshot_data: bytes, 
        screenshot_metadata: Dict[str, Any],
        alt_text: str = None,
        caption: str = None,
        post_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Upload screenshot to WordPress using V1 bulletproof patterns
        
        Args:
            screenshot_data: Raw screenshot binary data
            screenshot_metadata: Metadata from screenshot capture (URL, timestamp, etc.)
            alt_text: Custom alt text (auto-generated if None)
            caption: Custom caption (auto-generated if None)  
            post_id: WordPress post ID to associate with (optional)
            
        Returns:
            WordPress media ID if successful, None if failed
        """
        try:
            # Generate filename from metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            url = screenshot_metadata.get('url', 'unknown')
            domain = url.split('/')[2] if '//' in url else 'screenshot'
            domain = domain.replace('.', '_')
            capture_type = screenshot_metadata.get('capture_type', 'fullpage')
            filename = f"screenshot_{domain}_{capture_type}_{timestamp}.jpg"
            
            # Process screenshot using existing V1 bulletproof image processor
            processed_data, processed_filename, processing_metadata = await self.image_processor.process_image(
                screenshot_data, 
                filename
            )
            
            # Generate alt text if not provided
            if not alt_text:
                alt_text = self._generate_screenshot_alt_text(screenshot_metadata)
            
            # Generate caption if not provided
            if not caption:
                caption = self._generate_screenshot_caption(screenshot_metadata)
            
            # Upload to WordPress media library using existing patterns
            media_url = urljoin(self.config.site_url, "/wp-json/wp/v2/media")
            
            # Prepare multipart data
            form_data = aiohttp.FormData()
            form_data.add_field('file', processed_data, filename=processed_filename, content_type='image/jpeg')
            form_data.add_field('alt_text', alt_text)
            form_data.add_field('caption', caption)
            
            # Add post association if provided
            if post_id:
                form_data.add_field('post', str(post_id))
            
            # Upload with existing retry and error handling
            headers = self.auth_manager.headers.copy()
            headers.pop('Content-Type', None)  # Let aiohttp set this for multipart
            
            async with self.session.post(media_url, headers=headers, data=form_data) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    
                    # Update stats
                    self.stats['images_processed'] += 1
                    
                    # Log to Supabase if available
                    if self.supabase:
                        await self._log_screenshot_upload(result, screenshot_metadata)
                    
                    logger.info(f"✅ Screenshot uploaded successfully: Media ID {result['id']}")
                    return result['id']
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Screenshot upload failed: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to upload screenshot: {e}")
            return None

    def _generate_screenshot_alt_text(self, screenshot_metadata: Dict[str, Any]) -> str:
        """Generate accessibility-compliant alt text from screenshot metadata"""
        url = screenshot_metadata.get('url', 'website')
        capture_type = screenshot_metadata.get('capture_type', 'screenshot')
        timestamp = screenshot_metadata.get('timestamp')
        
        # Extract domain name for readability
        domain = 'website'
        if url and '//' in url:
            try:
                domain = url.split('/')[2].replace('www.', '')
            except:
                domain = 'website'
        
        # Generate descriptive alt text
        if capture_type == 'full_page':
            alt_text = f"Full page screenshot of {domain} website"
        elif capture_type == 'viewport':
            alt_text = f"Viewport screenshot of {domain} website"
        elif capture_type == 'element':
            element_info = screenshot_metadata.get('element_info', {})
            selector = element_info.get('selector', 'page element')
            alt_text = f"Screenshot of {selector} on {domain} website"
        else:
            alt_text = f"Screenshot of {domain} website"
        
        # Add timestamp context if available
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%B %Y")
                alt_text += f" captured in {date_str}"
            except:
                pass
        
        return alt_text

    def _generate_screenshot_caption(self, screenshot_metadata: Dict[str, Any]) -> str:
        """Generate descriptive caption from screenshot metadata"""
        url = screenshot_metadata.get('url', '')
        capture_type = screenshot_metadata.get('capture_type', 'screenshot')
        viewport_size = screenshot_metadata.get('viewport_size', {})
        file_size = screenshot_metadata.get('file_size', 0)
        
        # Start with basic caption
        caption_parts = []
        
        # Add capture type
        if capture_type == 'full_page':
            caption_parts.append("Full page screenshot")
        elif capture_type == 'viewport':
            caption_parts.append("Viewport screenshot")
        elif capture_type == 'element':
            caption_parts.append("Element screenshot")
        else:
            caption_parts.append("Screenshot")
        
        # Add viewport info if available
        if viewport_size and viewport_size.get('width') and viewport_size.get('height'):
            width = viewport_size['width']
            height = viewport_size['height']
            caption_parts.append(f"captured at {width}x{height} resolution")
        
        # Add source URL (truncated for readability)
        if url:
            display_url = url
            if len(url) > 50:
                display_url = url[:47] + "..."
            caption_parts.append(f"from {display_url}")
        
        # Add file size if significant
        if file_size > 0:
            size_mb = file_size / (1024 * 1024)
            if size_mb >= 0.1:
                caption_parts.append(f"({size_mb:.1f}MB)")
        
        return " ".join(caption_parts)

    async def _log_screenshot_upload(self, upload_result: Dict[str, Any], screenshot_metadata: Dict[str, Any]):
        """Log screenshot upload to Supabase for tracking and analytics"""
        try:
            if not self.supabase:
                return
            
            log_data = {
                'content_type': 'screenshot_upload',
                'wordpress_media_id': upload_result['id'],
                'media_url': upload_result.get('source_url', ''),
                'original_url': screenshot_metadata.get('url', ''),
                'capture_type': screenshot_metadata.get('capture_type', 'unknown'),
                'file_size': screenshot_metadata.get('file_size', 0),
                'viewport_size': screenshot_metadata.get('viewport_size', {}),
                'processing_metadata': {
                    'alt_text': upload_result.get('alt_text', ''),
                    'caption': upload_result.get('caption', ''),
                    'mime_type': upload_result.get('mime_type', ''),
                    'upload_timestamp': datetime.now().isoformat()
                }
            }
            
            # Insert into Supabase (assuming cms_audit_log table exists)
            await self.supabase.table('cms_audit_log').insert(log_data).execute()
            logger.info(f"📊 Screenshot upload logged to Supabase")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log screenshot upload: {e}")

    async def embed_screenshots_in_content(
        self, 
        content: str, 
        screenshot_media_ids: List[int],
        embed_style: str = "inline"
    ) -> str:
        """
        Embed uploaded screenshots in content using existing HTML formatter patterns
        
        Args:
            content: Original content HTML
            screenshot_media_ids: List of WordPress media IDs for screenshots
            embed_style: Embedding style ("inline", "gallery", "figure")
            
        Returns:
            Content with embedded screenshots
        """
        if not screenshot_media_ids:
            return content
        
        try:
            # Get media details from WordPress
            media_details = []
            for media_id in screenshot_media_ids:
                try:
                    media_url = urljoin(self.config.site_url, f"/wp-json/wp/v2/media/{media_id}")
                    async with self.session.get(media_url, headers=self.auth_manager.headers) as response:
                        if response.status == 200:
                            media_data = await response.json()
                            media_details.append(media_data)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch media details for ID {media_id}: {e}")
                    continue
            
            if not media_details:
                logger.warning("⚠️ No valid media details found for screenshot embedding")
                return content
            
            # Parse content and add screenshots
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            if embed_style == "inline":
                # Add screenshots inline after first paragraph
                first_p = soup.find('p')
                if first_p:
                    for media in media_details:
                        screenshot_html = self._create_screenshot_embed_html(media, "inline")
                        screenshot_soup = BeautifulSoup(screenshot_html, 'html.parser')
                        first_p.insert_after(screenshot_soup)
                        first_p = screenshot_soup  # Insert next one after this one
            
            elif embed_style == "gallery":
                # Create a screenshot gallery
                gallery_html = self._create_screenshot_gallery_html(media_details)
                gallery_soup = BeautifulSoup(gallery_html, 'html.parser')
                
                # Insert gallery after the first heading or at the beginning
                first_heading = soup.find(['h1', 'h2', 'h3'])
                if first_heading:
                    first_heading.insert_after(gallery_soup)
                else:
                    soup.insert(0, gallery_soup)
            
            elif embed_style == "figure":
                # Add each screenshot as a figure with caption
                for media in media_details:
                    figure_html = self._create_screenshot_embed_html(media, "figure")
                    figure_soup = BeautifulSoup(figure_html, 'html.parser')
                    soup.append(figure_soup)
            
            return str(soup)
            
        except ImportError:
            logger.warning("⚠️ BeautifulSoup not available for screenshot embedding")
            return content
        except Exception as e:
            logger.error(f"❌ Failed to embed screenshots: {e}")
            return content

    def _create_screenshot_embed_html(self, media_data: Dict[str, Any], style: str = "inline") -> str:
        """Create HTML for embedding a single screenshot"""
        media_id = media_data['id']
        source_url = media_data['source_url']
        alt_text = media_data.get('alt_text', 'Screenshot')
        caption = media_data.get('caption', {}).get('rendered', '')
        
        if style == "inline":
            return f'''
            <div class="screenshot-embed wp-block-image aligncenter">
                <img src="{source_url}" alt="{alt_text}" class="wp-image-{media_id} screenshot-evidence" loading="lazy" />
                {f'<p class="screenshot-caption">{caption}</p>' if caption else ''}
            </div>
            '''
        
        elif style == "figure":
            return f'''
            <figure class="wp-block-image aligncenter size-large screenshot-figure">
                <img src="{source_url}" alt="{alt_text}" class="wp-image-{media_id}" loading="lazy" />
                <figcaption class="wp-element-caption">{caption or alt_text}</figcaption>
            </figure>
            '''
        
        else:  # default inline
            return f'<img src="{source_url}" alt="{alt_text}" class="wp-image-{media_id} screenshot-inline" loading="lazy" />'

    def _create_screenshot_gallery_html(self, media_list: List[Dict[str, Any]]) -> str:
        """Create HTML for a screenshot gallery"""
        gallery_items = []
        
        for media in media_list:
            media_id = media['id']
            source_url = media['source_url']
            alt_text = media.get('alt_text', 'Screenshot')
            
            gallery_items.append(f'''
                <div class="screenshot-gallery-item">
                    <img src="{source_url}" alt="{alt_text}" class="wp-image-{media_id}" loading="lazy" />
                </div>
            ''')
        
        gallery_html = f'''
        <div class="screenshot-gallery wp-block-gallery">
            <h3 class="screenshot-gallery-title">Visual Evidence</h3>
            <div class="screenshot-gallery-grid">
                {"".join(gallery_items)}
            </div>
        </div>
        '''
        
        return gallery_html

class WordPressIntegration:
    """🏗️ NATIVE LANGCHAIN: High-level WordPress integration facade with 95-field casino intelligence"""
    
    def __init__(self, wordpress_config: WordPressConfig = None, supabase_client: Client = None, llm=None):
        self.config = wordpress_config or WordPressConfig()
        self.supabase = supabase_client or self._create_supabase_client()
        self.llm = llm
        self.publisher: Optional[WordPressRESTPublisher] = None
    
    def _create_supabase_client(self) -> Optional[Client]:
        """Create Supabase client if credentials available"""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            if url and key:
                return create_client(url, key)
        except Exception as e:
            logger.warning(f"Failed to create Supabase client: {e}")
        return None
    
    async def publish_rag_content(self, 
                                 query: str, 
                                 rag_response: str, 
                                 title: str = None,
                                 featured_image_query: str = None) -> Dict[str, Any]:
        """Publish RAG-generated content with smart enhancements"""
        
        # Generate title if not provided
        if not title:
            title = self._generate_title_from_query(query)
        
        # Extract meta description from content
        meta_description = self._extract_meta_description(rag_response)
        
        # Find contextual image if DataForSEO integration available
        featured_image_url = None
        if featured_image_query:
            featured_image_url = await self._find_contextual_image(featured_image_query)
        
        # Publish with full integration including LLM
        async with WordPressRESTPublisher(self.config, self.supabase, self.llm) as publisher:
            self.publisher = publisher
            
            result = await publisher.publish_post(
                title=title,
                content=rag_response,
                status=self.config.default_status,
                featured_image_url=featured_image_url,
                meta_description=meta_description,
                custom_fields={
                    'rag_query': query,
                    'generated_at': datetime.now().isoformat(),
                    'content_type': 'rag_generated'
                }
            )
            
            return result
    
    async def publish_casino_intelligence_content(self, 
                                                 query: str, 
                                                 rag_response: str, 
                                                 structured_casino_data: Optional[Dict[str, Any]] = None,
                                                 title: str = None,
                                                 featured_image_query: str = None) -> Dict[str, Any]:
        """🎰 NATIVE LANGCHAIN: Publish casino content with 95-field intelligence using enhanced LangChain integration"""
        
        # Generate casino-specific title if not provided
        if not title:
            title = self._generate_title_from_query(query)
            if structured_casino_data and structured_casino_data.get('casino_name'):
                casino_name = structured_casino_data['casino_name']
                overall_rating = structured_casino_data.get('overall_rating', 0)
                title = f"{casino_name} Review {overall_rating}/10 - Complete Analysis & Guide"
        
        # Find contextual casino image
        featured_image_url = None
        if featured_image_query:
            featured_image_url = await self._find_contextual_image(featured_image_query)
        elif structured_casino_data and structured_casino_data.get('casino_name'):
            # Try to find casino-specific image
            casino_name = structured_casino_data['casino_name']
            featured_image_url = await self._find_contextual_image(f"{casino_name} casino logo review")
        
        # Publish using enhanced casino publisher with LangChain integration
        async with WordPressRESTPublisher(self.config, self.supabase, self.llm) as publisher:
            self.publisher = publisher
            
            result = await publisher.publish_casino_content(
                title=title,
                content=rag_response,
                structured_casino_data=structured_casino_data,
                featured_image_url=featured_image_url
            )
            
            return result
    
    def _generate_title_from_query(self, query: str) -> str:
        """Generate SEO-friendly title from query"""
        # Simple title generation - can be enhanced with LLM
        title = query.strip()
        if not title.endswith('?'):
            title = f"Complete Guide: {title}"
        else:
            title = title.replace('?', ' - Everything You Need to Know')
        
        return title[:60]  # SEO-friendly length
    
    def _extract_meta_description(self, content: str) -> str:
        """Extract meta description from content"""
        # Remove HTML tags and get first paragraph
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        
        # Get first sentence or paragraph
        sentences = text.split('.')
        if sentences:
            description = sentences[0].strip()
            return description[:155] + "..." if len(description) > 155 else description
        
        return ""
    
    async def _find_contextual_image(self, query: str) -> Optional[str]:
        """Find contextual image using DataForSEO integration"""
        try:
            # This would integrate with DataForSEO image search
            # For now, return None - implementation depends on Task 5 integration
            logger.info(f"Image search for query: {query}")
            return None
        except Exception as e:
            logger.warning(f"Contextual image search failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        base_stats = self.publisher.get_stats() if self.publisher else {}
        
        return {
            **base_stats,
            'config': {
                'site_url': self.config.site_url,
                'auth_method': self.config.default_status,
                'max_concurrent_uploads': self.config.max_concurrent_uploads
            },
            'integration_status': {
                'supabase_connected': self.supabase is not None,
                'wordpress_configured': bool(self.config.site_url and self.config.username)
            }
        }

    async def publish_content_with_screenshots(
        self,
        title: str,
        content: str,
        screenshot_urls: List[str] = None,
        screenshot_types: List[str] = None,
        embed_style: str = "inline",
        **kwargs
    ) -> Dict[str, Any]:
        """
        🖼️ Publish content with integrated screenshot capture and embedding
        
        Args:
            title: Post title
            content: Post content
            screenshot_urls: List of URLs to capture screenshots from
            screenshot_types: List of screenshot types ("full_page", "viewport", "element")
            embed_style: How to embed screenshots ("inline", "gallery", "figure")
            **kwargs: Additional arguments passed to publish_post
            
        Returns:
            Publishing result with screenshot information
        """
        screenshot_media_ids = []
        
        try:
            # Capture screenshots if URLs provided
            if screenshot_urls:
                from .playwright_screenshot_engine import ScreenshotService, ScreenshotConfig
                from .browser_pool_manager import BrowserPoolManager
                
                # Initialize screenshot components
                browser_pool = BrowserPoolManager()
                screenshot_config = ScreenshotConfig(
                    format='png',
                    quality=85,
                    full_page=True,
                    timeout_ms=30000
                )
                screenshot_service = ScreenshotService(browser_pool, screenshot_config)
                
                logger.info(f"📷 Capturing {len(screenshot_urls)} screenshots for WordPress publishing")
                
                # Capture each screenshot
                for i, url in enumerate(screenshot_urls):
                    try:
                        # Determine screenshot type
                        capture_type = screenshot_types[i] if screenshot_types and i < len(screenshot_types) else "full_page"
                        
                        # Capture screenshot
                        if capture_type == "full_page":
                            result = await screenshot_service.capture_full_page_screenshot(url)
                        elif capture_type == "viewport":
                            result = await screenshot_service.capture_viewport_screenshot(url)
                        else:  # Default to full page
                            result = await screenshot_service.capture_full_page_screenshot(url)
                        
                        if result.success and result.screenshot_data:
                            # Prepare metadata for WordPress upload
                            screenshot_metadata = {
                                'url': url,
                                'capture_type': capture_type,
                                'timestamp': result.timestamp,
                                'file_size': result.file_size,
                                'viewport_size': result.viewport_size,
                                'element_info': result.element_info
                            }
                            
                            # Upload to WordPress using our new method
                            async with WordPressRESTPublisher(self.config, self.supabase, self.llm) as publisher:
                                media_id = await publisher.upload_screenshot_to_wordpress(
                                    result.screenshot_data,
                                    screenshot_metadata
                                )
                                
                                if media_id:
                                    screenshot_media_ids.append(media_id)
                                    logger.info(f"✅ Screenshot {i+1}/{len(screenshot_urls)} uploaded: Media ID {media_id}")
                                else:
                                    logger.warning(f"⚠️ Failed to upload screenshot {i+1} from {url}")
                        else:
                            logger.warning(f"⚠️ Failed to capture screenshot from {url}: {result.error_message}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing screenshot {i+1} from {url}: {e}")
                        continue
                
                # Cleanup browser pool
                await browser_pool.cleanup()
                
            # Embed screenshots in content if any were uploaded
            if screenshot_media_ids:
                async with WordPressRESTPublisher(self.config, self.supabase, self.llm) as publisher:
                    content = await publisher.embed_screenshots_in_content(
                        content, 
                        screenshot_media_ids, 
                        embed_style
                    )
                    logger.info(f"📝 Embedded {len(screenshot_media_ids)} screenshots in content")
            
            # Publish content with embedded screenshots
            result = await self.publish_rag_content(
                query=kwargs.get('query', title),
                rag_response=content,
                title=title,
                featured_image_query=kwargs.get('featured_image_query')
            )
            
            # Add screenshot information to result
            result['screenshot_info'] = {
                'screenshots_captured': len(screenshot_media_ids),
                'screenshot_media_ids': screenshot_media_ids,
                'embed_style': embed_style,
                'source_urls': screenshot_urls or []
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to publish content with screenshots: {e}")
            raise

# Factory function for easy initialization
def create_wordpress_integration(
    site_url: str = None,
    username: str = None, 
    application_password: str = None,
    **config_kwargs
) -> WordPressIntegration:
    """Create WordPress integration with environment defaults"""
    
    config = WordPressConfig(
        site_url=site_url or os.getenv("WORDPRESS_SITE_URL", ""),
        username=username or os.getenv("WORDPRESS_USERNAME", ""),
        application_password=application_password or os.getenv("WORDPRESS_APP_PASSWORD", ""),
        **config_kwargs
    )
    
    return WordPressIntegration(config)

# Example usage and testing
async def example_usage():
    """Example usage of WordPress integration with screenshot capture"""
    
    # Create integration
    wp = create_wordpress_integration()
    
    # Sample content for testing
    sample_content = """
    <h2>Best Online Casino Bonuses for New Players</h2>
    
    <p>New players have access to some of the most generous casino bonuses available online. 
    Here's your complete guide to finding and claiming the best welcome offers.</p>
    
    <h3>Types of Welcome Bonuses</h3>
    <ul>
        <li><strong>Match Bonuses:</strong> Casino matches your deposit up to a certain amount</li>
        <li><strong>Free Spins:</strong> Complimentary spins on popular slot games</li>
        <li><strong>No Deposit Bonuses:</strong> Free money just for signing up</li>
    </ul>
    
    <h3>Top Recommended Casinos</h3>
    <p>Based on our comprehensive analysis, here are the top casinos offering 
    exceptional welcome bonuses for new players in 2024.</p>
    """
    
    try:
        # Example 1: Regular content publishing (without screenshots)
        print("📝 Example 1: Regular content publishing...")
        result1 = await wp.publish_rag_content(
            query="What are the best online casino bonuses for new players?",
            rag_response=sample_content,
            title="Best Online Casino Bonuses for New Players 2024",
            featured_image_query="casino bonus welcome offer"
        )
        print(f"✅ Regular post published: ID {result1['id']}")
        
        # Example 2: Content publishing with screenshot capture
        print("\n📷 Example 2: Content with screenshot capture...")
        result2 = await wp.publish_content_with_screenshots(
            title="Casino Screenshot Review - Visual Evidence",
            content=sample_content,
            screenshot_urls=[
                "https://www.betmgm.com/",
                "https://www.draftkings.com/casino"
            ],
            screenshot_types=["full_page", "viewport"],
            embed_style="gallery",
            query="Casino review with visual evidence",
            featured_image_query="casino interface screenshot"
        )
        
        print(f"✅ Post with screenshots published!")
        print(f"📝 Post ID: {result2['id']}")
        print(f"🔗 URL: {result2['link']}")
        print(f"📷 Screenshots captured: {result2['screenshot_info']['screenshots_captured']}")
        print(f"🖼️ Media IDs: {result2['screenshot_info']['screenshot_media_ids']}")
        
        # Example 3: Casino intelligence content with screenshots
        print("\n🎰 Example 3: Casino intelligence with screenshots...")
        casino_data = {
            'casino_name': 'BetMGM Casino',
            'overall_rating': 8.5,
            'license_status': 'Licensed and Regulated',
            'game_provider_count': 15,
            'payment_method_count': 8,
            'welcome_bonus': '100% up to $1000 + 200 Free Spins'
        }
        
        result3 = await wp.publish_casino_intelligence_content(
            query="BetMGM Casino review with visual evidence",
            rag_response=sample_content,
            structured_casino_data=casino_data,
            title="BetMGM Casino Review 8.5/10 - Complete Visual Analysis",
            featured_image_query="BetMGM casino logo"
        )
        
        print(f"✅ Casino intelligence post published: ID {result3['id']}")
        
        # Get comprehensive performance stats
        stats = wp.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Posts published: {stats.get('posts_published', 0)}")
        print(f"   Images processed: {stats.get('images_processed', 0)}")
        print(f"   WordPress configured: {stats['integration_status']['wordpress_configured']}")
        print(f"   Supabase connected: {stats['integration_status']['supabase_connected']}")
        
    except Exception as e:
        print(f"❌ Publishing failed: {e}")
        import traceback
        traceback.print_exc()

# Example of standalone screenshot upload
async def example_screenshot_upload():
    """Example of uploading a screenshot independently"""
    
    try:
        from .playwright_screenshot_engine import ScreenshotService, ScreenshotConfig
        from .browser_pool_manager import BrowserPoolManager
        
        # Initialize components
        browser_pool = BrowserPoolManager()
        screenshot_config = ScreenshotConfig(format='png', quality=85)
        screenshot_service = ScreenshotService(browser_pool, screenshot_config)
        
        # Capture screenshot
        result = await screenshot_service.capture_full_page_screenshot("https://www.example.com")
        
        if result.success:
            # Upload to WordPress
            wp_config = WordPressConfig()
            async with WordPressRESTPublisher(wp_config) as publisher:
                media_id = await publisher.upload_screenshot_to_wordpress(
                    result.screenshot_data,
                    {
                        'url': 'https://www.example.com',
                        'capture_type': 'full_page',
                        'timestamp': result.timestamp,
                        'file_size': result.file_size,
                        'viewport_size': result.viewport_size
                    }
                )
                
                if media_id:
                    print(f"✅ Screenshot uploaded independently: Media ID {media_id}")
                else:
                    print("❌ Failed to upload screenshot")
        
        await browser_pool.cleanup()
        
    except Exception as e:
        print(f"❌ Standalone screenshot upload failed: {e}")

if __name__ == "__main__":
    # Run examples
    import asyncio
    print("🚀 Testing WordPress Screenshot Publishing Integration")
    print("=" * 60)
    asyncio.run(example_usage())
    print("\n" + "=" * 60)
    print("🔧 Testing standalone screenshot upload")
    asyncio.run(example_screenshot_upload()) 