#!/usr/bin/env python3
"""
Enhanced DataForSEO Image Search Integration with LangChain
Production-ready implementation with advanced features

Features:
- Native DataForSEO API integration with rate limiting
- Enhanced image processing with metadata extraction
- Supabase integration for media asset storage
- Batch processing capabilities (up to 100 tasks)
- Intelligent caching system
- Cost optimization through request batching
- Production-ready error handling and retry mechanisms
"""

import asyncio
import aiohttp
import base64
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from urllib.parse import urlparse
import logging

# Pydantic for structured data
from pydantic import BaseModel, Field, ConfigDict, validator
from supabase import create_client, Client

# Image processing
from PIL import Image
import requests
from io import BytesIO

# Rate limiting
import asyncio
from asyncio import Semaphore
from collections import deque

# LangChain imports
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===

@dataclass
class DataForSEOConfig:
    """Enhanced DataForSEO configuration with rate limiting"""
    
    # API Configuration
    login: str = field(default_factory=lambda: os.getenv("DATAFORSEO_LOGIN", ""))
    password: str = field(default_factory=lambda: os.getenv("DATAFORSEO_PASSWORD", ""))
    api_endpoint: str = "https://api.dataforseo.com/v3"
    
    # Rate Limiting (DataForSEO limits: 2,000 requests/minute, max 30 simultaneous)
    max_requests_per_minute: int = 1800  # Conservative limit
    max_concurrent_requests: int = 25     # Conservative concurrent limit
    
    # Batch Processing
    max_batch_size: int = 100
    batch_timeout_seconds: int = 30
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Caching Configuration
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    cache_max_size: int = 10000
    
    # Image Processing
    max_image_size_mb: int = 10
    supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp", "gif"])
    generate_thumbnails: bool = True
    thumbnail_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(150, 150), (300, 300)])
    
    # Supabase Configuration
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", ""))
    storage_bucket: str = "images"

# === MODELS ===

class ImageSearchType(Enum):
    """Image search types supported by DataForSEO"""
    GOOGLE_IMAGES = "google"
    BING_IMAGES = "bing"
    YANDEX_IMAGES = "yandex"

class ImageSize(Enum):
    """Image size filters"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"

class ImageType(Enum):
    """Image type filters"""
    PHOTO = "photo"
    CLIPART = "clipart"
    LINE_DRAWING = "line_drawing"
    ANIMATED = "animated"

class ImageColor(Enum):
    """Image color filters"""
    COLOR = "color"
    BLACK_AND_WHITE = "black_and_white"
    TRANSPARENT = "transparent"

class ImageSearchRequest(BaseModel):
    """Enhanced image search request model"""
    keyword: str = Field(description="Search keyword")
    search_engine: ImageSearchType = Field(default=ImageSearchType.GOOGLE_IMAGES)
    location_code: int = Field(default=2840, description="Location code (2840 = United States)")
    language_code: str = Field(default="en", description="Language code")
    
    # Search filters
    image_size: Optional[ImageSize] = None
    image_type: Optional[ImageType] = None
    image_color: Optional[ImageColor] = None
    safe_search: bool = True
    
    # Result configuration
    max_results: int = Field(default=20, ge=1, le=100)
    include_metadata: bool = True
    download_images: bool = False
    
    # Processing options
    extract_text: bool = False
    generate_alt_text: bool = True
    quality_filter: bool = True

class ImageMetadata(BaseModel):
    """Enhanced image metadata model"""
    url: str
    title: Optional[str] = None
    alt_text: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    format: Optional[str] = None
    source_url: Optional[str] = None
    source_domain: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    # Enhanced metadata
    extracted_text: Optional[str] = None
    generated_alt_text: Optional[str] = None
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # Processing metadata
    downloaded: bool = False
    storage_path: Optional[str] = None
    supabase_id: Optional[str] = None
    processing_errors: List[str] = field(default_factory=list)

class ImageSearchResult(BaseModel):
    """Enhanced image search result model"""
    request_id: str
    keyword: str
    search_engine: str
    total_results: int
    images: List[ImageMetadata]
    
    # Performance metrics
    search_duration_ms: float
    processing_duration_ms: float
    api_cost_estimate: float
    
    # Quality metrics
    average_quality_score: Optional[float] = None
    high_quality_count: int = 0
    
    # Caching metadata
    cached: bool = False
    cache_key: Optional[str] = None

# === RATE LIMITER ===

class RateLimiter:
    """Advanced rate limiter for DataForSEO API compliance"""
    
    def __init__(self, max_requests_per_minute: int, max_concurrent: int):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.request_times = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire rate limit permission"""
        await self.semaphore.acquire()
        
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            # Check if we're at the rate limit
            if len(self.request_times) >= self.max_requests_per_minute:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(now)
    
    def release(self):
        """Release rate limit permission"""
        self.semaphore.release()

# === CACHE MANAGER ===

class ImageSearchCache:
    """Intelligent caching system for image search results"""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache: Dict[str, Tuple[ImageSearchResult, float]] = {}
        self.access_times: Dict[str, float] = {}
    
    def _generate_cache_key(self, request: ImageSearchRequest) -> str:
        """Generate cache key from request"""
        key_data = {
            "keyword": request.keyword,
            "search_engine": request.search_engine.value,
            "location_code": request.location_code,
            "language_code": request.language_code,
            "max_results": request.max_results,
            "filters": {
                "size": request.image_size.value if request.image_size else None,
                "type": request.image_type.value if request.image_type else None,
                "color": request.image_color.value if request.image_color else None,
                "safe_search": request.safe_search
            }
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, request: ImageSearchRequest) -> Optional[ImageSearchResult]:
        """Get cached result if available and not expired"""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
                return None
            
            # Update access time
            self.access_times[cache_key] = time.time()
            
            # Mark as cached
            result.cached = True
            result.cache_key = cache_key
            
            return result
        
        return None
    
    def set(self, request: ImageSearchRequest, result: ImageSearchResult):
        """Cache search result with LRU eviction"""
        cache_key = self._generate_cache_key(request)
        
        # Evict if at max size
        if len(self.cache) >= self.max_size:
            # Find least recently used
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        # Store result
        self.cache[cache_key] = (result, time.time())
        self.access_times[cache_key] = time.time()
        
        logger.info(f"Cached search result for key: {cache_key}")

# === MAIN INTEGRATION CLASS ===

class EnhancedDataForSEOImageSearch:
    """
    Enhanced DataForSEO Image Search Integration
    
    Features:
    - Rate-limited API requests with batch processing
    - Enhanced image metadata extraction
    - Supabase integration for asset storage
    - Intelligent caching system
    - Cost optimization and error handling
    """
    
    def __init__(self, config: Optional[DataForSEOConfig] = None):
        self.config = config or DataForSEOConfig()
        self.rate_limiter = RateLimiter(
            self.config.max_requests_per_minute,
            self.config.max_concurrent_requests
        )
        self.cache = ImageSearchCache(
            self.config.cache_max_size,
            self.config.cache_ttl_hours
        ) if self.config.enable_caching else None
        
        # Initialize Supabase client
        if self.config.supabase_url and self.config.supabase_key:
            self.supabase = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
        else:
            self.supabase = None
            logger.warning("Supabase not configured - media storage disabled")
        
        # API credentials
        self.auth_header = self._create_auth_header()
    
    def _create_auth_header(self) -> str:
        """Create basic auth header for DataForSEO API"""
        credentials = f"{self.config.login}:{self.config.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"
    
    async def search_images(self, request: ImageSearchRequest) -> ImageSearchResult:
        """
        Search for images using DataForSEO API with enhanced processing
        """
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(request)
            if cached_result:
                logger.info(f"Cache hit for keyword: {request.keyword}")
                return cached_result
        
        # Perform API search
        try:
            result = await self._perform_search(request)
            
            # Process images if requested
            if request.download_images or request.generate_alt_text:
                await self._process_images(result, request)
            
            # Cache result
            if self.cache:
                self.cache.set(request, result)
            
            # Calculate performance metrics
            result.search_duration_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Image search completed: {len(result.images)} images found")
            return result
            
        except Exception as e:
            logger.error(f"Image search failed: {str(e)}")
            raise
    
    async def _perform_search(self, request: ImageSearchRequest) -> ImageSearchResult:
        """Perform the actual DataForSEO API search"""
        
        # Acquire rate limit
        await self.rate_limiter.acquire()
        
        try:
            # Prepare API request - Fixed URL format for DataForSEO
            api_url = f"{self.config.api_endpoint}/serp/{request.search_engine.value}/images/live/advanced"
            
            payload = {
                "keyword": request.keyword,
                "location_code": request.location_code,
                "language_code": request.language_code,
                "device": "desktop",
                "os": "windows",
                "depth": request.max_results
            }
            
            # Add filters in correct format for DataForSEO
            if request.image_size:
                payload["image_size"] = request.image_size.value
            if request.image_type:
                payload["image_type"] = request.image_type.value
            if request.image_color:
                payload["image_color"] = request.image_color.value
            if request.safe_search:
                payload["safe_search"] = True
            
            headers = {
                "Authorization": self.auth_header,
                "Content-Type": "application/json"
            }
            
            # Make API request with retry logic
            response_data = await self._make_request_with_retry(api_url, payload, headers)
            
            # Parse response
            return await self._parse_search_response(response_data, request)
            
        finally:
            self.rate_limiter.release()
    
    async def _make_request_with_retry(self, url: str, payload: Dict, headers: Dict) -> Dict:
        """Make API request with exponential backoff retry"""
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=[payload],  # DataForSEO expects array
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.config.batch_timeout_seconds)
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            if data.get("status_code") == 20000:
                                return data
                            else:
                                raise Exception(f"API error: {data.get('status_message', 'Unknown error')}")
                        else:
                            raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise e
                
                # Calculate delay with exponential backoff
                delay = self.config.retry_delay_seconds
                if self.config.exponential_backoff:
                    delay *= (2 ** attempt)
                
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                await asyncio.sleep(delay)
    
    async def _parse_search_response(self, response_data: Dict, request: ImageSearchRequest) -> ImageSearchResult:
        """Parse DataForSEO API response into structured result"""
        
        request_id = str(uuid.uuid4())
        images = []
        
        try:
            tasks = response_data.get("tasks", [])
            if not tasks:
                raise Exception("No tasks in response")
            
            task = tasks[0]
            if task.get("status_code") != 20000:
                raise Exception(f"Task failed: {task.get('status_message', 'Unknown error')}")
            
            results = task.get("result", [])
            if not results:
                logger.warning("No results in task")
                return ImageSearchResult(
                    request_id=request_id,
                    keyword=request.keyword,
                    search_engine=request.search_engine.value,
                    total_results=0,
                    images=[],
                    search_duration_ms=0,
                    processing_duration_ms=0,
                    api_cost_estimate=0.0
                )
            
            result = results[0]
            items = result.get("items", [])
            
            # Process each image item
            for item in items:
                try:
                    image_meta = ImageMetadata(
                        url=item.get("source_url", ""),
                        title=item.get("title", ""),
                        alt_text=item.get("alt", ""),
                        width=item.get("width"),
                        height=item.get("height"),
                        format=item.get("format", "").lower(),
                        source_url=item.get("source_url", ""),
                        source_domain=self._extract_domain(item.get("source_url", "")),
                        thumbnail_url=item.get("thumbnail", "")
                    )
                    
                    # Calculate quality score
                    if request.quality_filter:
                        image_meta.quality_score = self._calculate_quality_score(image_meta)
                    
                    images.append(image_meta)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse image item: {str(e)}")
                    continue
            
            # Calculate metrics
            total_results = result.get("total_count", len(images))
            high_quality_count = sum(1 for img in images if img.quality_score and img.quality_score > 0.7)
            average_quality = sum(img.quality_score for img in images if img.quality_score) / len(images) if images else 0
            
            return ImageSearchResult(
                request_id=request_id,
                keyword=request.keyword,
                search_engine=request.search_engine.value,
                total_results=total_results,
                images=images,
                search_duration_ms=0,  # Will be set by caller
                processing_duration_ms=0,
                api_cost_estimate=self._estimate_api_cost(len(images)),
                average_quality_score=average_quality,
                high_quality_count=high_quality_count
            )
            
        except Exception as e:
            logger.error(f"Failed to parse search response: {str(e)}")
            raise
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""
    
    def _calculate_quality_score(self, image: ImageMetadata) -> float:
        """Calculate image quality score based on various factors"""
        score = 0.0
        
        # Size score (larger is generally better, up to a point)
        if image.width and image.height:
            pixel_count = image.width * image.height
            if pixel_count > 300 * 300:  # Minimum decent size
                score += 0.3
            if pixel_count > 800 * 600:  # Good size
                score += 0.2
            if pixel_count > 1920 * 1080:  # High resolution
                score += 0.1
        else:
            # If no size info, give a base score
            score += 0.1
        
        # Format score - be more generous with unknown formats
        if image.format:
            if image.format.lower() in ["jpg", "jpeg", "png"]:
                score += 0.2
            elif image.format.lower() in ["webp"]:
                score += 0.15
            else:
                score += 0.1  # Unknown format gets some points
        else:
            score += 0.1  # No format info gets base points
        
        # Alt text score
        if image.alt_text and len(image.alt_text) > 10:
            score += 0.15
        elif image.alt_text and len(image.alt_text) > 3:
            score += 0.1
        
        # Title score
        if image.title and len(image.title) > 10:
            score += 0.15
        elif image.title and len(image.title) > 3:
            score += 0.1
        
        # Source domain score (trusted domains get bonus)
        if image.source_domain:
            trusted_domains = ['shutterstock.com', 'istockphoto.com', 'freepik.com', 'unsplash.com']
            if any(domain in image.source_domain.lower() for domain in trusted_domains):
                score += 0.1
        
        return min(score, 1.0)
    
    def _estimate_api_cost(self, result_count: int) -> float:
        """Estimate API cost based on DataForSEO pricing"""
        # Rough estimate: $0.0001 per result
        return result_count * 0.0001
    
    async def _process_images(self, result: ImageSearchResult, request: ImageSearchRequest):
        """Process images with enhanced metadata extraction"""
        
        processing_start = time.time()
        
        for image in result.images:
            try:
                # Generate alt text if requested
                if request.generate_alt_text and not image.alt_text:
                    image.generated_alt_text = await self._generate_alt_text(image)
                
                # Download and store image if requested
                if request.download_images:
                    await self._download_and_store_image(image)
                
            except Exception as e:
                image.processing_errors.append(str(e))
                logger.warning(f"Failed to process image {image.url}: {str(e)}")
        
        result.processing_duration_ms = (time.time() - processing_start) * 1000
    
    async def _generate_alt_text(self, image: ImageMetadata) -> str:
        """Generate alt text for image (placeholder implementation)"""
        # This would integrate with an image captioning model
        # For now, use basic heuristics
        
        alt_parts = []
        
        if image.title:
            alt_parts.append(image.title)
        
        if image.width and image.height:
            alt_parts.append(f"{image.width}x{image.height}")
        
        if image.format:
            alt_parts.append(f"{image.format.upper()} image")
        
        return " - ".join(alt_parts) if alt_parts else "Image"
    
    async def _download_and_store_image(self, image: ImageMetadata):
        """Download image and store in Supabase"""
        
        if not self.supabase:
            image.processing_errors.append("Supabase not configured")
            return
        
        try:
            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(image.url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download: HTTP {response.status}")
                    
                    image_data = await response.read()
                    
                    # Check size limit
                    if len(image_data) > self.config.max_image_size_mb * 1024 * 1024:
                        raise Exception(f"Image too large: {len(image_data)} bytes")
                    
                    # Generate storage path
                    file_extension = image.format or "jpg"
                    filename = f"{uuid.uuid4()}.{file_extension}"
                    storage_path = f"images/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
                    
                    # Upload to Supabase storage - V1 STYLE (simple and working)
                    try:
                        upload_result = self.supabase.storage.from_(self.config.storage_bucket).upload(
                            storage_path,
                            image_data,
                            file_options={
                                "content-type": f"image/{file_extension}",
                                "cache-control": "3600"
                            }
                        )
                        # V1 SUCCESS PATTERN: If no exception raised, it worked
                        logger.info(f"Successfully uploaded image: {storage_path}")
                        
                    except Exception as e:
                        raise Exception(f"Supabase upload failed: {str(e)}")
                    
                    # Store metadata in database
                    media_record = {
                        "id": str(uuid.uuid4()),
                        "storage_path": storage_path,
                        "file_name": filename,
                        "mime_type": f"image/{file_extension}",
                        "alt_text": image.alt_text or image.generated_alt_text,
                        "metadata": {
                            "original_url": image.url,
                            "source_domain": image.source_domain,
                            "width": image.width,
                            "height": image.height,
                            "quality_score": image.quality_score,
                            "dataforseo_metadata": {
                                "title": image.title,
                                "thumbnail_url": image.thumbnail_url
                            }
                        }
                    }
                    
                    db_result = self.supabase.table("media_assets").insert(media_record).execute()
                    
                    if db_result.data:
                        image.downloaded = True
                        image.storage_path = storage_path
                        image.supabase_id = media_record["id"]
                        logger.info(f"Successfully stored image: {storage_path}")
                    else:
                        raise Exception("Database insert failed")
        
        except Exception as e:
            image.processing_errors.append(str(e))
            logger.error(f"Failed to download and store image {image.url}: {str(e)}")
    
    async def batch_search(self, requests: List[ImageSearchRequest]) -> List[ImageSearchResult]:
        """Perform batch image searches with optimal batching"""
        
        if len(requests) > self.config.max_batch_size:
            # Split into smaller batches
            results = []
            for i in range(0, len(requests), self.config.max_batch_size):
                batch = requests[i:i + self.config.max_batch_size]
                batch_results = await self._process_batch(batch)
                results.extend(batch_results)
            return results
        else:
            return await self._process_batch(requests)
    
    async def _process_batch(self, requests: List[ImageSearchRequest]) -> List[ImageSearchResult]:
        """Process a batch of search requests concurrently"""
        
        # Create tasks for concurrent execution
        tasks = [self.search_images(request) for request in requests]
        
        # Execute with proper error handling
        results = []
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {str(result)}")
                # Create error result
                error_result = ImageSearchResult(
                    request_id=str(uuid.uuid4()),
                    keyword=requests[i].keyword,
                    search_engine=requests[i].search_engine.value,
                    total_results=0,
                    images=[],
                    search_duration_ms=0,
                    processing_duration_ms=0,
                    api_cost_estimate=0.0
                )
                results.append(error_result)
            else:
                results.append(result)
        
        return results
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search performance and usage"""
        
        analytics = {
            "cache_stats": {
                "size": len(self.cache.cache) if self.cache else 0,
                "hit_rate": 0.0,  # Would need to track hits/misses
                "max_size": self.config.cache_max_size
            },
            "rate_limiter_stats": {
                "current_requests": len(self.rate_limiter.request_times),
                "max_per_minute": self.config.max_requests_per_minute,
                "max_concurrent": self.config.max_concurrent_requests
            },
            "configuration": {
                "batch_size": self.config.max_batch_size,
                "cache_enabled": self.config.enable_caching,
                "supabase_enabled": self.supabase is not None
            }
        }
        
        return analytics

# === LANGCHAIN TOOL WRAPPER ===

class DataForSEOImageSearchTool(BaseTool):
    """LangChain Tool wrapper for DataForSEO Image Search"""
    
    name: str = "dataforseo_image_search"
    description: str = """
    Search for images using DataForSEO API. 
    Input should be a search keyword or JSON with search parameters.
    Returns structured image search results with metadata.
    """
    
    # Use model_config to allow arbitrary types
    model_config = {"arbitrary_types_allowed": True}
    
    # Define search_client as a field
    search_client: Optional[EnhancedDataForSEOImageSearch] = None
    
    def __init__(self, config: Optional[DataForSEOConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.search_client = EnhancedDataForSEOImageSearch(config)
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the image search synchronously"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(query, run_manager))
                    return future.result()
            else:
                return asyncio.run(self._arun(query, run_manager))
        except RuntimeError:
            # No event loop, safe to use asyncio.run
            return asyncio.run(self._arun(query, run_manager))
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the image search asynchronously"""
        try:
            # Parse input
            if query.startswith('{'):
                # JSON input
                params = json.loads(query)
                request = ImageSearchRequest(**params)
            else:
                # Simple keyword search
                request = ImageSearchRequest(keyword=query)
            
            # Perform search
            result = await self.search_client.search_images(request)
            
            # Format response for LangChain
            response = {
                "keyword": result.keyword,
                "total_results": result.total_results,
                "images_found": len(result.images),
                "average_quality": result.average_quality_score,
                "high_quality_count": result.high_quality_count,
                "search_duration_ms": result.search_duration_ms,
                "cached": result.cached,
                "images": [
                    {
                        "url": img.url,
                        "title": img.title,
                        "alt_text": img.alt_text,
                        "dimensions": f"{img.width}x{img.height}" if img.width and img.height else None,
                        "format": img.format,
                        "quality_score": img.quality_score,
                        "source_domain": img.source_domain
                    }
                    for img in result.images[:10]  # Limit to top 10 for readability
                ]
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"DataForSEO image search failed: {str(e)}")
            return f"Error: {str(e)}"

def create_dataforseo_tool(
    login: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> DataForSEOImageSearchTool:
    """Create a LangChain-compatible DataForSEO image search tool"""
    config = DataForSEOConfig()
    
    if login:
        config.login = login
    if password:
        config.password = password
    
    # Update config with any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DataForSEOImageSearchTool(config)

# === FACTORY FUNCTION ===

def create_dataforseo_image_search(
    login: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> EnhancedDataForSEOImageSearch:
    """
    Factory function to create DataForSEO image search instance
    
    Args:
        login: DataForSEO API login
        password: DataForSEO API password
        **kwargs: Additional configuration options
    
    Returns:
        Configured EnhancedDataForSEOImageSearch instance
    """
    
    config = DataForSEOConfig()
    
    if login:
        config.login = login
    if password:
        config.password = password
    
    # Apply any additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return EnhancedDataForSEOImageSearch(config)

# === EXAMPLE USAGE ===

async def example_usage():
    """Example usage of the enhanced DataForSEO image search"""
    
    # Create search instance
    search_engine = create_dataforseo_image_search()
    
    # Create search request
    request = ImageSearchRequest(
        keyword="casino games",
        search_engine=ImageSearchType.GOOGLE_IMAGES,
        max_results=20,
        image_size=ImageSize.LARGE,
        image_type=ImageType.PHOTO,
        safe_search=True,
        download_images=True,
        generate_alt_text=True
    )
    
    try:
        # Perform search
        result = await search_engine.search_images(request)
        
        print(f"Search Results for '{request.keyword}':")
        print(f"Total results: {result.total_results}")
        print(f"Images found: {len(result.images)}")
        print(f"High quality images: {result.high_quality_count}")
        print(f"Average quality score: {result.average_quality_score:.2f}")
        print(f"Search duration: {result.search_duration_ms:.2f}ms")
        print(f"Processing duration: {result.processing_duration_ms:.2f}ms")
        print(f"Estimated cost: ${result.api_cost_estimate:.4f}")
        print(f"Cached: {result.cached}")
        
        # Display first few images
        for i, image in enumerate(result.images[:3]):
            print(f"\nImage {i+1}:")
            print(f"  URL: {image.url}")
            print(f"  Title: {image.title}")
            print(f"  Size: {image.width}x{image.height}")
            print(f"  Quality: {image.quality_score:.2f}")
            print(f"  Downloaded: {image.downloaded}")
            if image.storage_path:
                print(f"  Storage: {image.storage_path}")
        
        # Get analytics
        analytics = await search_engine.get_search_analytics()
        print(f"\nAnalytics: {json.dumps(analytics, indent=2)}")
        
    except Exception as e:
        print(f"Search failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(example_usage())
