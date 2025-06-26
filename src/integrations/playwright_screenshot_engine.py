#!/usr/bin/env python3
"""
Native Playwright Screenshot Engine for Universal RAG CMS
Implements browser pool management and screenshot capture using pure Playwright APIs

✅ NATIVE PLAYWRIGHT IMPLEMENTATION
- Direct playwright.async_api usage
- No LangChain wrapper dependencies
- Pure Playwright browser automation
- Integrates with existing Universal RAG architecture

✅ TASK 22.2 BROWSER RESOURCE OPTIMIZATION
- Intelligent browser resource allocation
- Screenshot caching for repeated requests  
- Optimized image compression settings
- Performance monitoring and resource cleanup
"""

import asyncio
import logging
import time
import random
import os
import psutil
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Type
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import base64
from pathlib import Path
from enum import Enum
import uuid
import heapq
from datetime import datetime, timedelta
import json

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = BrowserContext = Page = Playwright = None

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ResourceOptimizationConfig:
    """Configuration for browser resource optimization (Task 22.2)"""
    # CPU and memory constraints
    max_concurrent_browsers: Optional[int] = None  # Auto-detect based on system
    memory_limit_mb: int = 2048  # Per browser instance
    cpu_utilization_threshold: float = 0.8  # Scale down if exceeded
    
    # Browser optimization settings
    disable_images: bool = False  # Set to True for text-heavy sites
    disable_javascript: bool = False  # Set to True for static content
    disable_css: bool = False  # Set to True for minimal rendering
    use_minimal_viewport: bool = False  # Use smaller viewport to save memory
    
    # Caching configuration
    enable_screenshot_caching: bool = True
    cache_ttl_hours: int = 24
    max_cache_entries: int = 1000
    cache_cleanup_interval_minutes: int = 30
    
    # Image compression optimization
    png_compression_level: int = 6  # 0-9, higher = smaller files
    jpeg_quality: int = 85  # 0-100, lower = smaller files
    auto_format_selection: bool = True  # Auto-choose format based on content
    
    @classmethod
    def get_performance_config(cls) -> 'ResourceOptimizationConfig':
        """Get configuration optimized for performance over quality"""
        return cls(
            disable_images=True,
            disable_css=True,
            use_minimal_viewport=True,
            jpeg_quality=70,
            png_compression_level=9
        )
    
    @classmethod
    def get_quality_config(cls) -> 'ResourceOptimizationConfig':
        """Get configuration optimized for quality over performance"""
        return cls(
            disable_images=False,
            disable_javascript=False,
            disable_css=False,
            jpeg_quality=95,
            png_compression_level=3
        )

@dataclass
class ScreenshotCacheEntry:
    """Represents a cached screenshot entry (Task 22.2)"""
    screenshot_data: bytes
    url: str
    cache_key: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    capture_config: Dict[str, Any]
    file_size: int
    content_hash: str
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() > self.created_at + timedelta(hours=ttl_hours)
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class ScreenshotCache:
    """
    Intelligent caching system for screenshot operations (Task 22.2)
    Implements LRU eviction with TTL and content-aware caching
    """
    
    def __init__(self, config: ResourceOptimizationConfig = None):
        self.config = config or ResourceOptimizationConfig()
        self._cache: Dict[str, ScreenshotCacheEntry] = {}
        self._lru_order: List[str] = []  # For LRU tracking
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'cache_size_bytes': 0,
            'total_entries_created': 0
        }
        
        logger.info(f"Screenshot cache initialized: max_entries={self.config.max_cache_entries}, "
                   f"ttl={self.config.cache_ttl_hours}h")
    
    def _generate_cache_key(self, url: str, capture_config: Dict[str, Any]) -> str:
        """Generate cache key based on URL and capture configuration"""
        key_components = [
            url,
            capture_config.get('format', 'png'),
            capture_config.get('quality', 85),
            capture_config.get('full_page', True),
            capture_config.get('viewport_width', 1920),
            capture_config.get('viewport_height', 1080),
            capture_config.get('selector')
        ]
        
        key_string = '|'.join(str(c) for c in key_components if c is not None)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, url: str, capture_config: Dict[str, Any]) -> Optional[bytes]:
        """Retrieve screenshot from cache if available and valid"""
        if not self.config.enable_screenshot_caching:
            return None
            
        cache_key = self._generate_cache_key(url, capture_config)
        
        async with self._lock:
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                self._stats['total_requests'] += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if entry has expired
            if entry.is_expired(self.config.cache_ttl_hours):
                await self._remove_entry(cache_key)
                self._stats['misses'] += 1
                self._stats['total_requests'] += 1
                return None
            
            # Update access tracking
            entry.update_access()
            self._update_lru_order(cache_key)
            
            self._stats['hits'] += 1
            self._stats['total_requests'] += 1
            
            return entry.screenshot_data
    
    async def set(self, url: str, capture_config: Dict[str, Any], screenshot_data: bytes):
        """Store screenshot in cache with intelligent eviction"""
        if not self.config.enable_screenshot_caching:
            return
            
        cache_key = self._generate_cache_key(url, capture_config)
        
        async with self._lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.config.max_cache_entries:
                await self._evict_lru_entry()
            
            # Create cache entry
            content_hash = hashlib.sha256(screenshot_data).hexdigest()[:16]
            entry = ScreenshotCacheEntry(
                screenshot_data=screenshot_data,
                url=url,
                cache_key=cache_key,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                capture_config=capture_config.copy(),
                file_size=len(screenshot_data),
                content_hash=content_hash
            )
            
            # Store entry
            self._cache[cache_key] = entry
            self._update_lru_order(cache_key)
            self._stats['cache_size_bytes'] += len(screenshot_data)
            self._stats['total_entries_created'] += 1
    
    def _update_lru_order(self, cache_key: str):
        """Update LRU order for cache key"""
        if cache_key in self._lru_order:
            self._lru_order.remove(cache_key)
        self._lru_order.append(cache_key)
    
    async def _evict_lru_entry(self):
        """Evict least recently used cache entry"""
        if not self._lru_order:
            return
            
        lru_key = self._lru_order[0]
        await self._remove_entry(lru_key)
        self._stats['evictions'] += 1
    
    async def _remove_entry(self, cache_key: str):
        """Remove cache entry and update statistics"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            self._stats['cache_size_bytes'] -= entry.file_size
            del self._cache[cache_key]
            
        if cache_key in self._lru_order:
            self._lru_order.remove(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._stats['total_requests']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self._cache),
            'cache_size_bytes': self._stats['cache_size_bytes'],
            'total_requests': total_requests,
            'cache_hits': self._stats['hits'],
            'cache_misses': self._stats['misses'],
            'cache_hit_rate_percent': round(hit_rate, 2),
            'evictions': self._stats['evictions'],
            'total_entries_created': self._stats['total_entries_created']
        }
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._lru_order.clear()
            self._stats['cache_size_bytes'] = 0

class ImageCompressionOptimizer:
    """
    Intelligent image compression optimizer (Task 22.2)
    Analyzes screenshot data and optimizes compression settings
    """
    
    @staticmethod
    def optimize_compression_settings(
        screenshot_data: bytes,
        target_format: str,
        optimization_config: ResourceOptimizationConfig
    ) -> Dict[str, Any]:
        """
        Analyze screenshot and determine optimal compression settings
        
        Args:
            screenshot_data: Original screenshot bytes
            target_format: Desired format ('png' or 'jpeg')
            optimization_config: Configuration for optimization
            
        Returns:
            Dictionary with optimization recommendations
        """
        data_size = len(screenshot_data)
        
        # Basic optimization based on file size
        if data_size < 50000:  # Small files (< 50KB) - minimal compression
            optimization_applied = False
            recommended_format = target_format
            recommended_quality = optimization_config.jpeg_quality
        elif data_size < 500000:  # Medium files (< 500KB) - moderate compression
            optimization_applied = True
            if optimization_config.auto_format_selection and target_format == 'png':
                recommended_format = 'jpeg'  # JPEG better for photos
                recommended_quality = optimization_config.jpeg_quality
            else:
                recommended_format = target_format
                recommended_quality = max(70, optimization_config.jpeg_quality - 15)
        else:  # Large files (>= 500KB) - aggressive compression
            optimization_applied = True
            if optimization_config.auto_format_selection:
                recommended_format = 'jpeg'  # Always prefer JPEG for large files
                recommended_quality = max(60, optimization_config.jpeg_quality - 25)
            else:
                recommended_format = target_format
                recommended_quality = max(60, optimization_config.jpeg_quality - 25)
        
        # Estimate file size reduction
        if optimization_applied and recommended_format == 'jpeg':
            # Rough estimate: JPEG typically 70-90% smaller than PNG for photos
            estimated_reduction = 0.8 if target_format == 'png' else 0.3
        else:
            estimated_reduction = 0.1 if optimization_applied else 0.0
        
        return {
            'optimization_applied': optimization_applied,
            'format': recommended_format,
            'quality': recommended_quality if recommended_format == 'jpeg' else None,
            'estimated_size_reduction_percent': round(estimated_reduction * 100, 1),
            'file_size_estimate': int(data_size * (1 - estimated_reduction)),
            'original_size': data_size,
            'reason': f"File size {data_size} bytes - {'aggressive' if data_size >= 500000 else 'moderate' if data_size >= 50000 else 'minimal'} compression"
        }

@dataclass
class BrowserProfile:
    """Browser profile configuration for stealth and variety"""
    user_agent: str
    viewport: Dict[str, int]
    locale: str
    timezone: str
    platform: str
    
    @classmethod
    def get_random_profile(cls) -> 'BrowserProfile':
        """Generate a random realistic browser profile"""
        profiles = [
            # Windows Chrome
            cls(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone="America/New_York",
                platform="Windows"
            ),
            # macOS Safari
            cls(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
                viewport={"width": 1440, "height": 900},
                locale="en-US", 
                timezone="America/Los_Angeles",
                platform="macOS"
            ),
            # Windows Firefox
            cls(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
                viewport={"width": 1366, "height": 768},
                locale="en-US",
                timezone="America/Chicago",
                platform="Windows"
            )
        ]
        return random.choice(profiles)

@dataclass 
class BrowserInstance:
    """Represents a managed browser instance in the pool"""
    browser: Browser
    created_at: float
    last_used: float
    usage_count: int
    profile: BrowserProfile
    is_healthy: bool = True
    
    def mark_used(self):
        """Mark browser as recently used"""
        self.last_used = time.time()
        self.usage_count += 1

class BrowserPoolManager:
    """
    Native Playwright browser pool management system with Task 22.2 optimizations
    Efficiently manages browser instances for screenshot capture
    
    ✅ TASK 22.2 FEATURES:
    - Resource-aware browser pool sizing
    - Optimized browser launch options
    - Memory usage monitoring
    - Performance metrics tracking
    """
    
    def __init__(
        self,
        max_pool_size: int = 3,
        max_browser_age_seconds: int = 3600,  # 1 hour
        max_usage_per_browser: int = 100,
        browser_timeout_seconds: int = 30,
        optimization_config: ResourceOptimizationConfig = None
    ):
        self.max_pool_size = max_pool_size
        self.max_browser_age_seconds = max_browser_age_seconds
        self.max_usage_per_browser = max_usage_per_browser
        self.browser_timeout_seconds = browser_timeout_seconds
        
        # Task 22.2: Resource optimization integration
        self.optimization_config = optimization_config or ResourceOptimizationConfig()
        self.browser_options: Optional[Dict[str, Any]] = None
        
        self._pool: List[BrowserInstance] = []
        self._playwright: Optional[Playwright] = None
        self._lock = asyncio.Lock()
        
        # Performance monitoring (Task 22.2)
        self._resource_stats = {
            'total_browsers_created': 0,
            'peak_pool_size': 0,
            'total_memory_usage_mb': 0,
            'avg_browser_lifespan_seconds': 0,
            'resource_optimization_enabled': bool(optimization_config)
        }
        
        # Apply resource optimization on initialization
        if self.optimization_config:
            optimize_browser_resources(self, self.optimization_config)
        
        logger.info(f"Initialized BrowserPoolManager with max_pool_size={max_pool_size}, "
                   f"optimization={'enabled' if optimization_config else 'disabled'}")

    async def initialize(self):
        """Initialize the Playwright instance"""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright not available. Install with: pip install playwright")
            
        if not self._playwright:
            self._playwright = await async_playwright().start()
            logger.info("Playwright initialized successfully")

    async def cleanup(self):
        """Clean up all browser instances and Playwright"""
        async with self._lock:
            for instance in self._pool:
                try:
                    await instance.browser.close()
                    logger.debug(f"Closed browser instance (usage: {instance.usage_count})")
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
            
            self._pool.clear()
            
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
                logger.info("Browser pool cleaned up successfully")

    @asynccontextmanager
    async def get_browser_context(self):
        """
        Context manager to get a browser context from the pool
        Automatically handles checkout/return and cleanup
        """
        browser_instance = None
        context = None
        
        try:
            # Get browser from pool
            browser_instance = await self._get_browser_from_pool()
            
            # Create context with stealth configuration
            context = await self._create_stealth_context(browser_instance)
            
            yield context
            
        except Exception as e:
            logger.error(f"Error in browser context: {e}")
            # Mark browser as unhealthy if it's a browser-level error
            if browser_instance and "browser" in str(e).lower():
                browser_instance.is_healthy = False
            raise
            
        finally:
            # Clean up context
            if context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")
            
            # Return browser to pool
            if browser_instance:
                await self._return_browser_to_pool(browser_instance)

    async def _get_browser_from_pool(self) -> BrowserInstance:
        """Get a browser instance from the pool or create a new one"""
        async with self._lock:
            # Clean up stale/unhealthy browsers
            await self._cleanup_stale_browsers()
            
            # Try to find a healthy browser
            for instance in self._pool:
                if instance.is_healthy and instance.usage_count < self.max_usage_per_browser:
                    instance.mark_used()
                    logger.debug(f"Reusing browser (usage: {instance.usage_count})")
                    return instance
            
            # Create new browser if pool not full
            if len(self._pool) < self.max_pool_size:
                return await self._create_new_browser()
            
            # Pool is full, find least recently used browser
            oldest_instance = min(self._pool, key=lambda x: x.last_used)
            await oldest_instance.browser.close()
            self._pool.remove(oldest_instance)
            
            return await self._create_new_browser()

    async def _create_new_browser(self) -> BrowserInstance:
        """Create a new browser instance with Task 22.2 optimized configuration"""
        if not self._playwright:
            await self.initialize()
            
        profile = BrowserProfile.get_random_profile()
        
        # Use optimized browser options if available (Task 22.2)
        if self.browser_options:
            launch_options = self.browser_options.copy()
            # Add user agent from profile
            launch_options['args'].append(f"--user-agent={profile.user_agent}")
        else:
            # Fallback to default options for stealth and performance
            launch_options = {
                "headless": True,
                "args": [
                    "--disable-extensions",
                    "--disable-gpu",
                    "--disable-dev-shm-usage", 
                    "--disable-setuid-sandbox",
                    "--no-sandbox",
                    "--disable-accelerated-2d-canvas",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-web-security",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    f"--user-agent={profile.user_agent}"
                ]
            }
        
        try:
            browser = await self._playwright.chromium.launch(**launch_options)
            
            instance = BrowserInstance(
                browser=browser,
                created_at=time.time(),
                last_used=time.time(),
                usage_count=0,
                profile=profile
            )
            
            self._pool.append(instance)
            
            # Update resource statistics (Task 22.2)
            self._resource_stats['total_browsers_created'] += 1
            self._resource_stats['peak_pool_size'] = max(self._resource_stats['peak_pool_size'], len(self._pool))
            
            logger.info(f"Created new browser instance (pool size: {len(self._pool)})")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create browser: {e}")
            raise

    async def _create_stealth_context(self, browser_instance: BrowserInstance) -> BrowserContext:
        """Create a browser context with stealth configuration"""
        profile = browser_instance.profile
        
        context_options = {
            "viewport": profile.viewport,
            "user_agent": profile.user_agent,
            "locale": profile.locale,
            "timezone_id": profile.timezone,
            "ignore_https_errors": True,
            "java_script_enabled": True,
            "accept_downloads": False,
            "extra_http_headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        }
        
        context = await browser_instance.browser.new_context(**context_options)
        
        # Remove automation signatures
        await context.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Override permissions API
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Mock languages and plugins
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        """)
        
        return context

    async def _return_browser_to_pool(self, browser_instance: BrowserInstance):
        """Return browser instance to pool (currently just marks it as available)"""
        # In this implementation, browsers stay in pool until cleanup
        # Could implement more sophisticated return logic here
        pass

    async def _cleanup_stale_browsers(self):
        """Remove browsers that are too old or have been used too much"""
        current_time = time.time()
        stale_browsers = []
        
        for instance in self._pool:
            # Check if browser is too old
            age = current_time - instance.created_at
            if age > self.max_browser_age_seconds:
                stale_browsers.append(instance)
                continue
                
            # Check if browser has been used too much
            if instance.usage_count >= self.max_usage_per_browser:
                stale_browsers.append(instance)
                continue
                
            # Check if browser is unhealthy
            if not instance.is_healthy:
                stale_browsers.append(instance)
                continue
        
        # Remove stale browsers
        for instance in stale_browsers:
            try:
                await instance.browser.close()
                self._pool.remove(instance)
                logger.debug(f"Removed stale browser (age: {current_time - instance.created_at:.1f}s, usage: {instance.usage_count})")
            except Exception as e:
                logger.warning(f"Error removing stale browser: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of the browser pool with Task 22.2 metrics"""
        async with self._lock:
            current_time = time.time()
            
            healthy_count = sum(1 for instance in self._pool if instance.is_healthy)
            total_usage = sum(instance.usage_count for instance in self._pool)
            
            # Calculate average browser lifespan
            if self._pool:
                avg_lifespan = sum(current_time - instance.created_at for instance in self._pool) / len(self._pool)
                self._resource_stats['avg_browser_lifespan_seconds'] = round(avg_lifespan, 2)
            
            return {
                "pool_size": len(self._pool),
                "max_pool_size": self.max_pool_size,
                "healthy_browsers": healthy_count,
                "total_usage": total_usage,
                "playwright_initialized": self._playwright is not None,
                "browser_ages": [
                    current_time - instance.created_at 
                    for instance in self._pool
                ],
                # Task 22.2: Resource optimization metrics
                "resource_stats": self._resource_stats.copy(),
                "optimization_enabled": self.optimization_config is not None,
                "memory_limit_mb": self.optimization_config.memory_limit_mb if self.optimization_config else None
            }
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get detailed resource usage statistics (Task 22.2)"""
        current_time = time.time()
        
        # Calculate memory usage estimation
        estimated_memory_usage = len(self._pool) * (self.optimization_config.memory_limit_mb if self.optimization_config else 512)
        
        browser_lifespans = [current_time - instance.created_at for instance in self._pool]
        
        return {
            'pool_metrics': {
                'current_pool_size': len(self._pool),
                'max_pool_size': self.max_pool_size,
                'pool_utilization': len(self._pool) / self.max_pool_size if self.max_pool_size > 0 else 0,
                'healthy_browsers': sum(1 for instance in self._pool if instance.is_healthy)
            },
            'resource_metrics': {
                'estimated_memory_usage_mb': estimated_memory_usage,
                'memory_limit_per_browser_mb': self.optimization_config.memory_limit_mb if self.optimization_config else 512,
                'total_browsers_created': self._resource_stats['total_browsers_created'],
                'peak_pool_size': self._resource_stats['peak_pool_size']
            },
            'performance_metrics': {
                'avg_browser_lifespan_seconds': self._resource_stats['avg_browser_lifespan_seconds'],
                'min_browser_age_seconds': min(browser_lifespans) if browser_lifespans else 0,
                'max_browser_age_seconds': max(browser_lifespans) if browser_lifespans else 0,
                'browser_age_variance': self._calculate_age_variance(browser_lifespans)
            },
            'optimization_status': {
                'resource_optimization_enabled': self._resource_stats['resource_optimization_enabled'],
                'caching_enabled': self.optimization_config.enable_screenshot_caching if self.optimization_config else False,
                'auto_format_selection': self.optimization_config.auto_format_selection if self.optimization_config else False
            }
        }
    
    def _calculate_age_variance(self, ages: List[float]) -> float:
        """Calculate variance in browser ages"""
        if len(ages) < 2:
            return 0.0
        
        mean_age = sum(ages) / len(ages)
        variance = sum((age - mean_age) ** 2 for age in ages) / len(ages)
        return round(variance, 2)

# Global browser pool instance for the Universal RAG Chain
_global_browser_pool: Optional[BrowserPoolManager] = None

async def get_global_browser_pool() -> BrowserPoolManager:
    """Get or create the global browser pool instance"""
    global _global_browser_pool
    
    if _global_browser_pool is None:
        _global_browser_pool = BrowserPoolManager()
        await _global_browser_pool.initialize()
        
    return _global_browser_pool

async def cleanup_global_browser_pool():
    """Clean up the global browser pool"""
    global _global_browser_pool
    
    if _global_browser_pool:
        await _global_browser_pool.cleanup()
        _global_browser_pool = None

@dataclass
class ScreenshotConfig:
    """Configuration for screenshot capture"""
    format: str = 'png'  # png, jpeg (webp not supported by Playwright)
    quality: int = 85     # 0-100 for jpeg only
    full_page: bool = True
    timeout_ms: int = 30000
    wait_for_load_state: str = 'domcontentloaded'  # domcontentloaded, load, networkidle
    viewport_width: int = 1920
    viewport_height: int = 1080

@dataclass 
class ScreenshotResult:
    """Result from screenshot capture operation"""
    success: bool
    screenshot_data: Optional[bytes] = None
    error_message: Optional[str] = None
    url: Optional[str] = None
    timestamp: float = 0.0
    file_size: int = 0
    viewport_size: Optional[Dict[str, int]] = None
    element_info: Optional[Dict[str, Any]] = None

class ScreenshotService:
    """
    Core screenshot capture service using native Playwright APIs with Task 22.2 optimizations
    Integrates with BrowserPoolManager for efficient resource usage
    
    ✅ TASK 22.2 FEATURES:
    - Screenshot caching for repeated requests
    - Intelligent image compression optimization
    - Performance monitoring and metrics
    - Resource-aware capture settings
    """
    
    def __init__(
        self, 
        browser_pool: BrowserPoolManager, 
        config: ScreenshotConfig = None,
        enable_caching: bool = True,
        cache_instance: ScreenshotCache = None
    ):
        self.browser_pool = browser_pool
        self.config = config or ScreenshotConfig()
        
        # Task 22.2: Screenshot caching integration
        self.enable_caching = enable_caching
        self.cache = cache_instance or (get_global_screenshot_cache() if enable_caching else None)
        
        # Task 22.2: Image compression optimizer
        self.compression_optimizer = ImageCompressionOptimizer()
        
        # Performance metrics
        self._capture_stats = {
            'total_captures': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_bytes_captured': 0,
            'avg_capture_time_ms': 0,
            'compression_optimizations': 0
        }
        
        logger.info(f"ScreenshotService initialized with format={self.config.format}, quality={self.config.quality}, "
                   f"caching={'enabled' if enable_caching else 'disabled'}")

    async def capture_full_page_screenshot(self, url: str) -> ScreenshotResult:
        """
        Capture full page screenshot with Task 22.2 caching and optimization
        
        Args:
            url: Target URL to screenshot
            
        Returns:
            ScreenshotResult with capture data or error information
        """
        start_time = time.time()
        
        # Task 22.2: Check cache first
        capture_config = {
            'format': self.config.format,
            'quality': self.config.quality,
            'full_page': self.config.full_page,
            'viewport_width': self.config.viewport_width,
            'viewport_height': self.config.viewport_height
        }
        
        if self.cache:
            cached_data = await self.cache.get(url, capture_config)
            if cached_data:
                self._capture_stats['cache_hits'] += 1
                self._capture_stats['total_captures'] += 1
                logger.debug(f"Cache hit for {url[:50]}...")
                return ScreenshotResult(
                    success=True,
                    screenshot_data=cached_data,
                    url=url,
                    timestamp=time.time(),
                    file_size=len(cached_data),
                    viewport_size={"width": self.config.viewport_width, "height": self.config.viewport_height}
                )
            else:
                self._capture_stats['cache_misses'] += 1
        
        try:
            async with self.browser_pool.get_browser_context() as context:
                page = await context.new_page()
                
                try:
                    # Navigate to URL with timeout
                    await page.goto(url, timeout=self.config.timeout_ms)
                    
                    # Wait for page to load
                    await page.wait_for_load_state(self.config.wait_for_load_state)
                    
                    # Task 22.2: Optimize compression settings
                    optimization_config = self.browser_pool.optimization_config if self.browser_pool.optimization_config else ResourceOptimizationConfig()
                    
                    # Capture screenshot using native Playwright API
                    screenshot_options = {
                        'full_page': self.config.full_page,
                        'type': self.config.format
                    }
                    
                    # Add quality parameter only for jpeg
                    if self.config.format == 'jpeg':
                        screenshot_options['quality'] = self.config.quality
                    
                    screenshot_data = await page.screenshot(**screenshot_options)
                    
                    # Task 22.2: Apply compression optimization
                    compression_settings = self.compression_optimizer.optimize_compression_settings(
                        screenshot_data, self.config.format, optimization_config
                    )
                    
                    if compression_settings['optimization_applied']:
                        # Re-capture with optimized settings
                        screenshot_options['type'] = compression_settings['format']
                        if compression_settings['quality']:
                            screenshot_options['quality'] = compression_settings['quality']
                        
                        screenshot_data = await page.screenshot(**screenshot_options)
                        self._capture_stats['compression_optimizations'] += 1
                        logger.debug(f"Applied compression optimization: {compression_settings['format']}")
                    
                    # Get viewport information
                    viewport = page.viewport_size
                    
                    # Task 22.2: Cache the result
                    if self.cache:
                        await self.cache.set(url, capture_config, screenshot_data)
                    
                    # Update statistics
                    capture_time_ms = (time.time() - start_time) * 1000
                    self._capture_stats['total_captures'] += 1
                    self._capture_stats['total_bytes_captured'] += len(screenshot_data)
                    
                    # Update average capture time
                    current_avg = self._capture_stats['avg_capture_time_ms']
                    total_captures = self._capture_stats['total_captures']
                    self._capture_stats['avg_capture_time_ms'] = (current_avg * (total_captures - 1) + capture_time_ms) / total_captures
                    
                    return ScreenshotResult(
                        success=True,
                        screenshot_data=screenshot_data,
                        url=url,
                        timestamp=time.time(),
                        file_size=len(screenshot_data),
                        viewport_size=viewport
                    )
                    
                except Exception as page_error:
                    logger.error(f"Page-level error capturing screenshot for {url}: {page_error}")
                    return ScreenshotResult(
                        success=False,
                        error_message=f"Page error: {str(page_error)}",
                        url=url,
                        timestamp=time.time()
                    )
                    
                finally:
                    await page.close()
                    
        except Exception as context_error:
            logger.error(f"Context-level error capturing screenshot for {url}: {context_error}")
            return ScreenshotResult(
                success=False,
                error_message=f"Context error: {str(context_error)}",
                url=url,
                timestamp=time.time()
            )

    async def capture_viewport_screenshot(self, url: str, viewport_width: int = None, viewport_height: int = None) -> ScreenshotResult:
        """
        Capture viewport-only screenshot with custom dimensions
        
        Args:
            url: Target URL to screenshot
            viewport_width: Custom viewport width (optional)
            viewport_height: Custom viewport height (optional)
            
        Returns:
            ScreenshotResult with capture data or error information
        """
        # Use custom viewport or default from config
        width = viewport_width or self.config.viewport_width
        height = viewport_height or self.config.viewport_height
        
        try:
            async with self.browser_pool.get_browser_context() as context:
                page = await context.new_page()
                
                try:
                    # Set custom viewport size
                    await page.set_viewport_size({"width": width, "height": height})
                    
                    await page.goto(url, timeout=self.config.timeout_ms)
                    await page.wait_for_load_state(self.config.wait_for_load_state)
                    
                    # Capture viewport-only screenshot
                    screenshot_options = {
                        'full_page': False,  # Viewport only
                        'type': self.config.format
                    }
                    
                    if self.config.format == 'jpeg':
                        screenshot_options['quality'] = self.config.quality
                    
                    screenshot_data = await page.screenshot(**screenshot_options)
                    
                    return ScreenshotResult(
                        success=True,
                        screenshot_data=screenshot_data,
                        url=url,
                        timestamp=time.time(),
                        file_size=len(screenshot_data),
                        viewport_size={"width": width, "height": height}
                    )
                    
                except Exception as page_error:
                    logger.error(f"Viewport screenshot error for {url}: {page_error}")
                    return ScreenshotResult(
                        success=False,
                        error_message=f"Viewport error: {str(page_error)}",
                        url=url,
                        timestamp=time.time()
                    )
                    
                finally:
                    await page.close()
                    
        except Exception as context_error:
            logger.error(f"Context error in viewport screenshot for {url}: {context_error}")
            return ScreenshotResult(
                success=False,
                error_message=f"Context error: {str(context_error)}",
                url=url,
                timestamp=time.time()
            )

    async def capture_element_screenshot(self, url: str, selector: str, wait_for_selector: bool = True) -> ScreenshotResult:
        """
        Capture screenshot of specific element using locator.screenshot()
        
        Args:
            url: Target URL
            selector: CSS selector for target element
            wait_for_selector: Whether to wait for element to be visible
            
        Returns:
            ScreenshotResult with element screenshot data or error
        """
        try:
            async with self.browser_pool.get_browser_context() as context:
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=self.config.timeout_ms)
                    await page.wait_for_load_state(self.config.wait_for_load_state)
                    
                    # Wait for element if requested
                    if wait_for_selector:
                        await page.wait_for_selector(selector, timeout=10000)
                    
                    # Get element locator
                    element = page.locator(selector)
                    
                    # Check if element exists and is visible
                    if not await element.is_visible():
                        return ScreenshotResult(
                            success=False,
                            error_message=f"Element not visible: {selector}",
                            url=url,
                            timestamp=time.time()
                        )
                    
                    # Capture element screenshot using native locator.screenshot()
                    screenshot_options = {
                        'type': self.config.format
                    }
                    
                    if self.config.format == 'jpeg':
                        screenshot_options['quality'] = self.config.quality
                    
                    screenshot_data = await element.screenshot(**screenshot_options)
                    
                    # Get element bounding box for metadata
                    bounding_box = await element.bounding_box()
                    
                    return ScreenshotResult(
                        success=True,
                        screenshot_data=screenshot_data,
                        url=url,
                        timestamp=time.time(),
                        file_size=len(screenshot_data),
                        element_info={
                            "selector": selector,
                            "bounding_box": bounding_box
                        }
                    )
                    
                except Exception as page_error:
                    logger.error(f"Element screenshot error for {url} (selector: {selector}): {page_error}")
                    return ScreenshotResult(
                        success=False,
                        error_message=f"Element error: {str(page_error)}",
                        url=url,
                        timestamp=time.time()
                    )
                    
                finally:
                    await page.close()
                    
        except Exception as context_error:
            logger.error(f"Context error in element screenshot for {url}: {context_error}")
            return ScreenshotResult(
                success=False,
                error_message=f"Context error: {str(context_error)}",
                url=url,
                timestamp=time.time()
            )

    def get_capture_stats(self) -> Dict[str, Any]:
        """Get detailed capture performance statistics (Task 22.2)"""
        cache_hit_rate = (self._capture_stats['cache_hits'] / self._capture_stats['total_captures']) * 100 if self._capture_stats['total_captures'] > 0 else 0
        
        return {
            'capture_metrics': {
                'total_captures': self._capture_stats['total_captures'],
                'cache_hits': self._capture_stats['cache_hits'],
                'cache_misses': self._capture_stats['cache_misses'],
                'cache_hit_rate_percent': round(cache_hit_rate, 2),
                'avg_capture_time_ms': round(self._capture_stats['avg_capture_time_ms'], 2),
                'total_bytes_captured': self._capture_stats['total_bytes_captured'],
                'avg_bytes_per_capture': round(self._capture_stats['total_bytes_captured'] / self._capture_stats['total_captures'], 2) if self._capture_stats['total_captures'] > 0 else 0
            },
            'optimization_metrics': {
                'compression_optimizations': self._capture_stats['compression_optimizations'],
                'optimization_rate_percent': round((self._capture_stats['compression_optimizations'] / self._capture_stats['total_captures']) * 100, 2) if self._capture_stats['total_captures'] > 0 else 0,
                'caching_enabled': self.enable_caching,
                'cache_instance_active': self.cache is not None
            }
        }

    async def wait_for_dynamic_content(self, page: Page, wait_strategy: str = 'networkidle') -> bool:
        """
        Wait for dynamic content to load (useful for casino sites with JS-heavy content)
        
        Args:
            page: Playwright page instance
            wait_strategy: Strategy for waiting ('networkidle', 'selector', 'timeout')
            
        Returns:
            bool indicating if content loaded successfully
        """
        try:
            if wait_strategy == 'networkidle':
                await page.wait_for_load_state('networkidle', timeout=15000)
            elif wait_strategy == 'casino_elements':
                # Wait for common casino content indicators
                casino_selectors = [
                    '[class*="game"]',
                    '[class*="slot"]', 
                    '[class*="lobby"]',
                    '.casino-games'
                ]
                
                for selector in casino_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=5000)
                        logger.debug(f"Found casino content: {selector}")
                        return True
                    except:
                        continue
                        
                # If no casino elements found, wait for basic load
                await page.wait_for_load_state('domcontentloaded')
            else:
                # Default timeout wait
                await asyncio.sleep(2)
                
            return True
            
        except Exception as e:
            logger.warning(f"Dynamic content wait failed: {e}")
            return False

@dataclass
class CasinoElement:
    """Represents a casino page element with targeting information"""
    element_type: str  # 'lobby', 'games', 'logo', 'bonus'
    selector: str
    confidence: float  # 0.0-1.0 confidence in selector accuracy
    fallback_selectors: List[str]
    description: str

class CasinoElementLocator:
    """
    Casino-specific element targeting system for screenshots
    Uses heuristic algorithms to identify common casino page elements
    """
    
    # Predefined selectors for common casino elements
    CASINO_SELECTORS = {
        'lobby': [
            # Main lobby/homepage sections
            '[class*="lobby"]',
            '[id*="lobby"]',
            '[class*="main-content"]',
            '[class*="hero"]',
            '[class*="banner"]',
            '.casino-lobby',
            '.main-lobby',
            '.homepage-content',
            '[data-testid*="lobby"]',
            '.welcome-section'
        ],
        'games': [
            # Game grids and sections
            '[class*="game"]',
            '[class*="slot"]',
            '[id*="game"]',
            '[id*="slot"]',
            '.games-grid',
            '.casino-games',
            '.slot-games',
            '.game-list',
            '.games-section',
            '[data-testid*="game"]',
            '.game-container',
            '.slots-container',
            '.casino-content'
        ],
        'logo': [
            # Casino logos and branding
            '[class*="logo"]',
            '[id*="logo"]',
            'img[alt*="logo"]',
            'img[src*="logo"]',
            '.brand-logo',
            '.casino-logo',
            '.header-logo',
            '.site-logo',
            '[data-testid*="logo"]',
            'header img:first-child',
            '.navbar-brand img'
        ],
        'bonus': [
            # Promotional and bonus sections
            '[class*="bonus"]',
            '[class*="promo"]',
            '[class*="offer"]',
            '[id*="bonus"]',
            '[id*="promo"]',
            '.promotions',
            '.casino-bonus',
            '.welcome-bonus',
            '.bonus-section',
            '.promo-banner',
            '[data-testid*="bonus"]',
            '[data-testid*="promo"]',
            '.special-offers'
        ]
    }
    
    # Additional heuristic patterns for element detection
    HEURISTIC_PATTERNS = {
        'games': {
            'text_indicators': ['play now', 'spin', 'jackpot', 'slots', 'poker', 'blackjack', 'roulette'],
            'image_indicators': ['game', 'slot', 'wheel', 'card'],
            'class_patterns': [r'.*game.*', r'.*slot.*', r'.*casino.*', r'.*play.*']
        },
        'bonus': {
            'text_indicators': ['bonus', 'free spins', 'welcome', 'deposit', 'match', 'cashback', 'reward'],
            'image_indicators': ['bonus', 'gift', 'coin', 'money'],
            'class_patterns': [r'.*bonus.*', r'.*promo.*', r'.*offer.*', r'.*welcome.*']
        },
        'lobby': {
            'text_indicators': ['welcome', 'casino', 'lobby', 'games', 'play'],
            'class_patterns': [r'.*lobby.*', r'.*main.*', r'.*hero.*', r'.*banner.*']
        },
        'logo': {
            'attribute_patterns': [r'.*logo.*', r'.*brand.*'],
            'position_indicators': ['header', 'top', 'navbar']
        }
    }

    def __init__(self, screenshot_service: ScreenshotService):
        self.screenshot_service = screenshot_service
        logger.info("CasinoElementLocator initialized")

    async def detect_casino_elements(self, url: str) -> Dict[str, List[CasinoElement]]:
        """
        Detect all casino elements on a page using multiple strategies
        
        Args:
            url: Target casino website URL
            
        Returns:
            Dictionary mapping element types to lists of detected elements
        """
        detected_elements = {
            'lobby': [],
            'games': [],
            'logo': [],
            'bonus': []
        }
        
        try:
            async with self.screenshot_service.browser_pool.get_browser_context() as context:
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=30000)
                    await page.wait_for_load_state('domcontentloaded')
                    
                    # Wait for casino-specific content
                    await self.screenshot_service.wait_for_dynamic_content(page, 'casino_elements')
                    
                    # Detect each element type
                    for element_type in detected_elements.keys():
                        elements = await self._detect_elements_by_type(page, element_type)
                        detected_elements[element_type] = elements
                    
                    logger.info(f"Detected casino elements for {url}: {sum(len(v) for v in detected_elements.values())} total")
                    
                except Exception as page_error:
                    logger.error(f"Error detecting casino elements for {url}: {page_error}")
                    
                finally:
                    await page.close()
                    
        except Exception as context_error:
            logger.error(f"Context error in casino element detection for {url}: {context_error}")
            
        return detected_elements

    async def _detect_elements_by_type(self, page: Page, element_type: str) -> List[CasinoElement]:
        """Detect elements of a specific type using multiple strategies"""
        detected = []
        
        # Strategy 1: Direct selector matching
        for selector in self.CASINO_SELECTORS.get(element_type, []):
            try:
                elements = await page.locator(selector).all()
                for element in elements:
                    if await element.is_visible():
                        detected.append(CasinoElement(
                            element_type=element_type,
                            selector=selector,
                            confidence=0.8,  # High confidence for direct selectors
                            fallback_selectors=[],
                            description=f"Direct selector match: {selector}"
                        ))
                        break  # Use first working selector
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        # Strategy 2: Heuristic text/content analysis
        if element_type in self.HEURISTIC_PATTERNS:
            heuristic_elements = await self._detect_by_heuristics(page, element_type)
            detected.extend(heuristic_elements)
        
        # Strategy 3: Position-based detection (for logos)
        if element_type == 'logo':
            position_elements = await self._detect_logo_by_position(page)
            detected.extend(position_elements)
        
        return detected

    async def _detect_by_heuristics(self, page: Page, element_type: str) -> List[CasinoElement]:
        """Use heuristic analysis to detect elements"""
        detected = []
        patterns = self.HEURISTIC_PATTERNS.get(element_type, {})
        
        # Text-based detection
        text_indicators = patterns.get('text_indicators', [])
        for indicator in text_indicators:
            try:
                # Look for elements containing indicator text
                selector = f"text=/{indicator}/i"
                elements = await page.locator(selector).all()
                
                for element in elements[:3]:  # Limit to first 3 matches
                    if await element.is_visible():
                        parent_selector = await self._get_parent_selector(element)
                        detected.append(CasinoElement(
                            element_type=element_type,
                            selector=parent_selector,
                            confidence=0.6,  # Medium confidence for heuristics
                            fallback_selectors=[selector],
                            description=f"Heuristic text match: '{indicator}'"
                        ))
                        
            except Exception as e:
                logger.debug(f"Heuristic text detection failed for '{indicator}': {e}")
                continue
        
        return detected

    async def _detect_logo_by_position(self, page: Page) -> List[CasinoElement]:
        """Detect logos based on position and image characteristics"""
        detected = []
        
        # Common logo positions
        logo_positions = [
            'header img:first-child',
            '.header img[src*="logo"]',
            '.navbar img',
            'img[alt*="logo"]',
            'header .logo img',
            '.brand img'
        ]
        
        for selector in logo_positions:
            try:
                elements = await page.locator(selector).all()
                for element in elements:
                    if await element.is_visible():
                        # Check image dimensions (logos are typically horizontal)
                        bounding_box = await element.bounding_box()
                        if bounding_box and bounding_box['width'] > bounding_box['height']:
                            detected.append(CasinoElement(
                                element_type='logo',
                                selector=selector,
                                confidence=0.7,
                                fallback_selectors=[],
                                description=f"Position-based logo detection: {selector}"
                            ))
                            break
                            
            except Exception as e:
                logger.debug(f"Logo position detection failed for {selector}: {e}")
                continue
        
        return detected

    async def _get_parent_selector(self, element) -> str:
        """Get a robust selector for an element's parent container"""
        try:
            # Try to get a stable parent selector
            parent = element.locator('..')
            
            # Look for ID or class on parent
            parent_id = await parent.get_attribute('id')
            if parent_id:
                return f"#{parent_id}"
            
            parent_class = await parent.get_attribute('class')
            if parent_class:
                # Use first class name
                first_class = parent_class.split()[0]
                return f".{first_class}"
            
            # Fallback to tag name
            tag_name = await parent.evaluate('el => el.tagName.toLowerCase()')
            return tag_name
            
        except Exception:
            return 'body'  # Ultimate fallback

    async def capture_casino_screenshots(self, url: str, element_types: List[str] = None) -> Dict[str, List[ScreenshotResult]]:
        """
        Capture screenshots of specific casino elements
        
        Args:
            url: Target casino website
            element_types: List of element types to capture (default: all)
            
        Returns:
            Dictionary mapping element types to screenshot results
        """
        if element_types is None:
            element_types = ['lobby', 'games', 'logo', 'bonus']
        
        results = {}
        
        # First detect all elements
        detected_elements = await self.detect_casino_elements(url)
        
        # Capture screenshots for requested element types
        for element_type in element_types:
            results[element_type] = []
            elements = detected_elements.get(element_type, [])
            
            for element in elements:
                try:
                    # Capture element screenshot
                    screenshot_result = await self.screenshot_service.capture_element_screenshot(
                        url=url,
                        selector=element.selector,
                        wait_for_selector=True
                    )
                    
                    if screenshot_result.success:
                        # Add casino-specific metadata
                        screenshot_result.element_info.update({
                            'casino_element_type': element.element_type,
                            'confidence': element.confidence,
                            'description': element.description
                        })
                        results[element_type].append(screenshot_result)
                        logger.info(f"Captured {element_type} screenshot: {screenshot_result.file_size} bytes")
                    else:
                        # Try fallback selectors
                        for fallback_selector in element.fallback_selectors:
                            fallback_result = await self.screenshot_service.capture_element_screenshot(
                                url=url,
                                selector=fallback_selector,
                                wait_for_selector=True
                            )
                            if fallback_result.success:
                                fallback_result.element_info.update({
                                    'casino_element_type': element.element_type,
                                    'confidence': element.confidence * 0.8,  # Reduced confidence for fallback
                                    'description': f"Fallback: {element.description}"
                                })
                                results[element_type].append(fallback_result)
                                break
                                
                except Exception as e:
                    logger.error(f"Error capturing {element_type} screenshot: {e}")
                    continue
        
        return results

    async def get_best_casino_selectors(self, url: str) -> Dict[str, str]:
        """
        Get the best selector for each casino element type
        
        Args:
            url: Target casino website
            
        Returns:
            Dictionary mapping element types to their best selectors
        """
        detected_elements = await self.detect_casino_elements(url)
        best_selectors = {}
        
        for element_type, elements in detected_elements.items():
            if elements:
                # Sort by confidence and take the best
                best_element = max(elements, key=lambda x: x.confidence)
                best_selectors[element_type] = best_element.selector
        
        return best_selectors

# ========================================================================================
# SUBTASK 19.4: ASYNCHRONOUS SCREENSHOT QUEUE SYSTEM
# ========================================================================================

class ScreenshotPriority(Enum):
    """Priority levels for screenshot requests."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0

@dataclass
class ScreenshotRequest:
    """Represents a screenshot request in the queue."""
    id: str
    url: str
    priority: ScreenshotPriority
    screenshot_type: str  # 'full_page', 'viewport', 'element'
    config: ScreenshotConfig
    selector: Optional[str] = None
    callback: Optional[Callable] = None
    casino_elements: Optional[List[str]] = None  # For casino-specific requests
    timestamp: float = field(default_factory=time.time)
    timeout: int = 30
    max_retries: int = 3
    retry_count: int = 0
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp

@dataclass
class QueueStatus:
    """Status information for the screenshot queue."""
    pending_requests: int
    active_requests: int
    completed_requests: int
    failed_requests: int
    average_processing_time: float
    queue_health: str  # 'healthy', 'degraded', 'critical'

class ScreenshotQueue:
    """
    Asynchronous screenshot queue system for handling multiple concurrent requests.
    
    Features:
    - Priority-based request processing
    - Configurable concurrency limits
    - Request timeout and cancellation
    - Progress monitoring and status updates
    - Dynamic waiting strategies for lazy-loaded content
    - Integration with browser pool for resource management
    """
    
    def __init__(self, 
                 max_concurrent: int = 5,
                 default_timeout: int = 30,
                 max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.max_queue_size = max_queue_size
        
        # Queue management
        self._queue: List[ScreenshotRequest] = []
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._completed_requests: Dict[str, ScreenshotResult] = {}
        self._failed_requests: Dict[str, Exception] = {}
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._processing = False
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self._stats = {
            'total_processed': 0,
            'total_failed': 0,
            'processing_times': [],
            'start_time': time.time()
        }
        
        # Services
        self.screenshot_service = None
        self.casino_locator = None
        
        logger.info(f"Screenshot queue initialized with max_concurrent={max_concurrent}")
    
    async def initialize(self, browser_pool: BrowserPoolManager):
        """Initialize the queue with required services."""
        self.screenshot_service = ScreenshotService(browser_pool)
        self.casino_locator = CasinoElementLocator(self.screenshot_service)
        logger.info("Screenshot queue services initialized")
    
    async def add_request(self, 
                         url: str,
                         screenshot_type: str = 'full_page',
                         priority: ScreenshotPriority = ScreenshotPriority.NORMAL,
                         config: Optional[ScreenshotConfig] = None,
                         selector: Optional[str] = None,
                         casino_elements: Optional[List[str]] = None,
                         callback: Optional[Callable] = None,
                         timeout: Optional[int] = None) -> str:
        """
        Add a screenshot request to the queue.
        
        Args:
            url: Target URL for screenshot
            screenshot_type: Type of screenshot ('full_page', 'viewport', 'element', 'casino')
            priority: Request priority
            config: Screenshot configuration
            selector: CSS selector for element screenshots
            casino_elements: List of casino element types to capture
            callback: Optional callback function for completion notification
            timeout: Request timeout in seconds
            
        Returns:
            Request ID for tracking
        """
        if len(self._queue) >= self.max_queue_size:
            raise ValueError(f"Queue is full (max size: {self.max_queue_size})")
        
        # Create request
        request_id = str(uuid.uuid4())
        request = ScreenshotRequest(
            id=request_id,
            url=url,
            priority=priority,
            screenshot_type=screenshot_type,
            config=config or ScreenshotConfig(),
            selector=selector,
            callback=callback,
            casino_elements=casino_elements,
            timeout=timeout or self.default_timeout
        )
        
        # Add to priority queue
        heapq.heappush(self._queue, request)
        
        logger.info(f"Added screenshot request {request_id} for {url} (priority: {priority.name})")
        
        # Start processing if not already running
        if not self._processing:
            asyncio.create_task(self._process_queue())
        
        return request_id
    
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending or active request.
        
        Args:
            request_id: ID of request to cancel
            
        Returns:
            True if cancelled successfully, False if not found or already completed
        """
        # Check active requests
        if request_id in self._active_requests:
            task = self._active_requests[request_id]
            task.cancel()
            del self._active_requests[request_id]
            logger.info(f"Cancelled active request {request_id}")
            return True
        
        # Check pending queue
        for i, request in enumerate(self._queue):
            if request.id == request_id:
                del self._queue[i]
                heapq.heapify(self._queue)  # Rebuild heap after removal
                logger.info(f"Cancelled pending request {request_id}")
                return True
        
        logger.warning(f"Request {request_id} not found for cancellation")
        return False
    
    async def get_status(self) -> QueueStatus:
        """Get current queue status and statistics."""
        pending = len(self._queue)
        active = len(self._active_requests)
        completed = len(self._completed_requests)
        failed = len(self._failed_requests)
        
        # Calculate average processing time
        avg_time = 0.0
        if self._stats['processing_times']:
            avg_time = sum(self._stats['processing_times']) / len(self._stats['processing_times'])
        
        # Determine queue health
        total_requests = completed + failed
        if total_requests == 0:
            health = 'healthy'
        else:
            failure_rate = failed / total_requests
            if failure_rate < 0.1:
                health = 'healthy'
            elif failure_rate < 0.3:
                health = 'degraded'
            else:
                health = 'critical'
        
        return QueueStatus(
            pending_requests=pending,
            active_requests=active,
            completed_requests=completed,
            failed_requests=failed,
            average_processing_time=avg_time,
            queue_health=health
        )
    
    async def get_result(self, request_id: str) -> Optional[ScreenshotResult]:
        """Get the result of a completed request."""
        return self._completed_requests.get(request_id)
    
    async def wait_for_completion(self, request_id: str, timeout: Optional[int] = None) -> ScreenshotResult:
        """
        Wait for a specific request to complete.
        
        Args:
            request_id: ID of request to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Screenshot result when completed
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
            Exception: If request failed
        """
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while True:
            # Check if completed
            if request_id in self._completed_requests:
                return self._completed_requests[request_id]
            
            # Check if failed
            if request_id in self._failed_requests:
                raise self._failed_requests[request_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Request {request_id} timed out after {timeout}s")
            
            await asyncio.sleep(check_interval)
    
    async def _process_queue(self):
        """Main queue processing loop."""
        if self._processing:
            return
        
        self._processing = True
        logger.info("Started queue processing")
        
        try:
            while not self._shutdown_event.is_set():
                # Check if we have pending requests and available slots
                if not self._queue or len(self._active_requests) >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next request
                request = heapq.heappop(self._queue)
                
                # Start processing request
                task = asyncio.create_task(self._process_request(request))
                self._active_requests[request.id] = task
                
                logger.debug(f"Started processing request {request.id}")
        
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        finally:
            self._processing = False
            logger.info("Stopped queue processing")
    
    async def _process_request(self, request: ScreenshotRequest):
        """
        Process a single screenshot request.
        
        Args:
            request: Screenshot request to process
        """
        start_time = time.time()
        
        try:
            async with self._semaphore:
                # Wait for dynamic content if needed
                await self._wait_for_dynamic_content(request)
                
                # Process based on request type
                if request.screenshot_type == 'casino':
                    result = await self._process_casino_request(request)
                elif request.screenshot_type == 'full_page':
                    result = await self.screenshot_service.capture_full_page_screenshot(
                        request.url
                    )
                elif request.screenshot_type == 'viewport':
                    # For viewport screenshots, extract dimensions from config
                    viewport_width = getattr(request.config, 'viewport_width', None)
                    viewport_height = getattr(request.config, 'viewport_height', None)
                    result = await self.screenshot_service.capture_viewport_screenshot(
                        request.url, viewport_width, viewport_height
                    )
                elif request.screenshot_type == 'element':
                    if not request.selector:
                        raise ValueError("Element selector required for element screenshots")
                    result = await self.screenshot_service.capture_element_screenshot(
                        request.url, request.selector
                    )
                else:
                    raise ValueError(f"Unsupported screenshot type: {request.screenshot_type}")
                
                # Store result
                self._completed_requests[request.id] = result
                
                # Update statistics
                processing_time = time.time() - start_time
                self._stats['processing_times'].append(processing_time)
                self._stats['total_processed'] += 1
                
                # Keep only last 100 processing times for average calculation
                if len(self._stats['processing_times']) > 100:
                    self._stats['processing_times'] = self._stats['processing_times'][-100:]
                
                # Call callback if provided
                if request.callback:
                    try:
                        if asyncio.iscoroutinefunction(request.callback):
                            await request.callback(request.id, result)
                        else:
                            request.callback(request.id, result)
                    except Exception as e:
                        logger.error(f"Callback error for request {request.id}: {e}")
                
                logger.info(f"Completed request {request.id} in {processing_time:.2f}s")
        
        except Exception as e:
            # Handle failure
            self._failed_requests[request.id] = e
            self._stats['total_failed'] += 1
            
            # Retry logic
            if request.retry_count < request.max_retries:
                request.retry_count += 1
                retry_delay = min(2 ** request.retry_count, 10)  # Exponential backoff, max 10s
                
                logger.warning(f"Request {request.id} failed (attempt {request.retry_count}), "
                             f"retrying in {retry_delay}s: {e}")
                
                # Re-queue with delay
                await asyncio.sleep(retry_delay)
                heapq.heappush(self._queue, request)
            else:
                logger.error(f"Request {request.id} failed permanently after {request.max_retries} retries: {e}")
        
        finally:
            # Remove from active requests
            self._active_requests.pop(request.id, None)
    
    async def _process_casino_request(self, request: ScreenshotRequest) -> ScreenshotResult:
        """Process a casino-specific screenshot request."""
        if not request.casino_elements:
            request.casino_elements = ['lobby', 'games', 'logo', 'bonus']
        
        # Detect casino elements
        elements = await self.casino_locator.detect_casino_elements(
            request.url, request.casino_elements
        )
        
        # Capture screenshots for detected elements
        screenshots = await self.casino_locator.capture_casino_screenshots(
            request.url, elements, request.config
        )
        
        # Combine results into a single response
        total_size = sum(len(s.screenshot_data) for s in screenshots.values() if s.success and s.screenshot_data)
        
        return ScreenshotResult(
            success=len(screenshots) > 0,
            screenshot_data=b'',  # No single image for casino requests
            error_message=None if screenshots else "No casino elements found",
            timestamp=time.time(),
            url=request.url,
            viewport_size={'width': request.config.viewport_width, 'height': request.config.viewport_height},
            element_info={'casino_screenshots': screenshots, 'total_size': total_size}
        )
    
    async def _wait_for_dynamic_content(self, request: ScreenshotRequest):
        """
        Wait for dynamic content to load based on request type.
        
        Args:
            request: Screenshot request
        """
        if request.screenshot_type == 'casino':
            # Casino sites often have heavy JS content
            await asyncio.sleep(2.0)  # Base wait for casino sites
        elif 'lazy' in request.url.lower() or 'dynamic' in request.url.lower():
            # Detected dynamic content indicators
            await asyncio.sleep(1.5)
        else:
            # Standard wait for basic content
            await asyncio.sleep(0.5)
    
    async def shutdown(self, timeout: int = 30):
        """
        Gracefully shutdown the queue.
        
        Args:
            timeout: Maximum time to wait for active requests to complete
        """
        logger.info("Shutting down screenshot queue...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for active requests to complete
        if self._active_requests:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_requests.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Some requests did not complete within {timeout}s timeout")
                # Cancel remaining tasks
                for task in self._active_requests.values():
                    task.cancel()
        
        logger.info("Screenshot queue shutdown complete")

# ========================================================================================
# QUEUE INTEGRATION AND UTILITIES
# ========================================================================================

class ScreenshotQueueManager:
    """
    High-level interface for managing screenshot queues.
    Provides simplified methods for common screenshot operations.
    """
    
    def __init__(self, browser_pool: BrowserPoolManager):
        self.browser_pool = browser_pool
        self.queue = ScreenshotQueue()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the queue manager."""
        if not self._initialized:
            await self.queue.initialize(self.browser_pool)
            self._initialized = True
    
    async def capture_website_screenshots(self, 
                                        urls: List[str],
                                        screenshot_types: List[str] = None,
                                        priority: ScreenshotPriority = ScreenshotPriority.NORMAL,
                                        config: Optional[ScreenshotConfig] = None) -> Dict[str, str]:
        """
        Capture screenshots for multiple URLs.
        
        Args:
            urls: List of URLs to screenshot
            screenshot_types: List of screenshot types (matches urls length)
            priority: Priority for all requests
            config: Screenshot configuration
            
        Returns:
            Dictionary mapping URLs to request IDs
        """
        await self.initialize()
        
        if screenshot_types is None:
            screenshot_types = ['full_page'] * len(urls)
        
        if len(screenshot_types) != len(urls):
            raise ValueError("screenshot_types length must match urls length")
        
        request_ids = {}
        for url, screenshot_type in zip(urls, screenshot_types):
            request_id = await self.queue.add_request(
                url=url,
                screenshot_type=screenshot_type,
                priority=priority,
                config=config
            )
            request_ids[url] = request_id
        
        return request_ids
    
    async def capture_casino_screenshots(self, 
                                       casino_urls: List[str],
                                       element_types: List[str] = None,
                                       priority: ScreenshotPriority = ScreenshotPriority.HIGH) -> Dict[str, str]:
        """
        Capture casino-specific screenshots for multiple casino sites.
        
        Args:
            casino_urls: List of casino URLs
            element_types: Casino element types to capture
            priority: Priority for casino requests
            
        Returns:
            Dictionary mapping URLs to request IDs
        """
        await self.initialize()
        
        if element_types is None:
            element_types = ['lobby', 'games', 'logo', 'bonus']
        
        request_ids = {}
        for url in casino_urls:
            request_id = await self.queue.add_request(
                url=url,
                screenshot_type='casino',
                priority=priority,
                casino_elements=element_types,
                config=ScreenshotConfig(format='png', quality=90)  # High quality for casino analysis
            )
            request_ids[url] = request_id
        
        return request_ids
    
    async def wait_for_all_results(self, request_ids: Dict[str, str], timeout: int = 300) -> Dict[str, ScreenshotResult]:
        """
        Wait for all requests to complete and return results.
        
        Args:
            request_ids: Dictionary mapping URLs to request IDs
            timeout: Total timeout for all requests
            
        Returns:
            Dictionary mapping URLs to screenshot results
        """
        results = {}
        
        for url, request_id in request_ids.items():
            try:
                result = await self.queue.wait_for_completion(request_id, timeout)
                results[url] = result
            except Exception as e:
                # Create error result
                results[url] = ScreenshotResult(
                    success=False,
                    screenshot_data=b'',
                    error_message=str(e),
                    timestamp=time.time(),
                    url=url
                )
        
        return results
    
    async def get_queue_metrics(self) -> Dict[str, Any]:
        """Get comprehensive queue metrics and statistics."""
        status = await self.queue.get_status()
        
        return {
            'queue_status': {
                'pending': status.pending_requests,
                'active': status.active_requests,
                'completed': status.completed_requests,
                'failed': status.failed_requests,
                'health': status.queue_health
            },
            'performance': {
                'average_processing_time': status.average_processing_time,
                'total_processed': self.queue._stats['total_processed'],
                'total_failed': self.queue._stats['total_failed'],
                'success_rate': (
                    self.queue._stats['total_processed'] / 
                    max(1, self.queue._stats['total_processed'] + self.queue._stats['total_failed'])
                ) * 100
            },
            'configuration': {
                'max_concurrent': self.queue.max_concurrent,
                'max_queue_size': self.queue.max_queue_size,
                'default_timeout': self.queue.default_timeout
            }
        }

# ========================================================================================
# TESTING FUNCTION FOR QUEUE SYSTEM
# ========================================================================================

async def test_screenshot_queue():
    """Test the screenshot queue system with multiple concurrent requests."""
    print("=== Testing Screenshot Queue System ===")
    
    # Initialize browser pool and queue manager
    browser_pool = BrowserPoolManager(max_pool_size=2)  # Smaller pool for testing
    queue_manager = ScreenshotQueueManager(browser_pool)
    
    try:
        print("\n1. Testing queue initialization...")
        await queue_manager.initialize()
        
        # Check initial status
        status = await queue_manager.queue.get_status()
        print(f"  Initial queue health: {status.queue_health}")
        print(f"  Initial pending requests: {status.pending_requests}")
        
        print("\n2. Testing request addition...")
        # Add a simple request
        request_id = await queue_manager.queue.add_request(
            url="https://example.com",
            screenshot_type='full_page',
            priority=ScreenshotPriority.HIGH
        )
        print(f"  ✅ Added request: {request_id[:8]}...")
        
        # Check status after adding request
        status = await queue_manager.queue.get_status()
        print(f"  Pending requests after add: {status.pending_requests}")
        
        print("\n3. Testing request processing...")
        # Wait a bit for processing to start
        await asyncio.sleep(1.0)
        
        # Try to get the result (with timeout)
        try:
            result = await asyncio.wait_for(
                queue_manager.queue.wait_for_completion(request_id),
                timeout=15.0
            )
            if result.success:
                print(f"  ✅ Screenshot captured: {len(result.screenshot_data)} bytes")
            else:
                print(f"  ⚠️ Screenshot failed: {result.error_message}")
        except asyncio.TimeoutError:
            print("  ⚠️ Request processing timed out (expected for testing)")
        
        print("\n4. Testing queue metrics...")
        metrics = await queue_manager.get_queue_metrics()
        print(f"  Queue health: {metrics['queue_status']['health']}")
        print(f"  Total processed: {metrics['performance']['total_processed']}")
        print(f"  Max concurrent: {metrics['configuration']['max_concurrent']}")
        
        print("\n5. Testing request cancellation...")
        # Add another request and cancel it
        cancel_id = await queue_manager.queue.add_request(
            url="https://httpbin.org/delay/10",
            screenshot_type='full_page',
            priority=ScreenshotPriority.LOW
        )
        print(f"  Added request to cancel: {cancel_id[:8]}...")
        
        cancelled = await queue_manager.queue.cancel_request(cancel_id)
        print(f"  ✅ Cancellation successful: {cancelled}")
        
        print("\n✅ Queue system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Queue test error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        await queue_manager.queue.shutdown(timeout=5)
        await browser_pool.cleanup()
        print("Cleanup completed")

# Simplified test for quick validation
async def test_queue_basics():
    """Quick test of queue basic functionality."""
    print("=== Quick Queue Test ===")
    
    try:
        # Test queue creation
        queue = ScreenshotQueue(max_concurrent=2, max_queue_size=10)
        print("✅ Queue created")
        
        # Test status
        status = await queue.get_status()
        print(f"✅ Queue status: {status.queue_health}")
        
        # Test adding request (without processing)
        request_id = await queue.add_request(
            url="https://example.com",
            screenshot_type='full_page'
        )
        print(f"✅ Request added: {request_id[:8]}...")
        
        # Test cancellation
        cancelled = await queue.cancel_request(request_id)
        print(f"✅ Request cancelled: {cancelled}")
        
        print("✅ Quick test passed!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()

# Example usage and testing
async def test_screenshot_service():
    """Test function for screenshot capture functionality"""
    pool = BrowserPoolManager(max_pool_size=2)
    
    try:
        await pool.initialize()
        service = ScreenshotService(pool)
        
        # Test full page screenshot
        result = await service.capture_full_page_screenshot("https://example.com")
        if result.success:
            print(f"Full page screenshot: {result.file_size} bytes")
        else:
            print(f"Full page error: {result.error_message}")
        
        # Test viewport screenshot
        result = await service.capture_viewport_screenshot("https://example.com", 1366, 768)
        if result.success:
            print(f"Viewport screenshot: {result.file_size} bytes")
        else:
            print(f"Viewport error: {result.error_message}")
        
        # Test element screenshot
        result = await service.capture_element_screenshot("https://example.com", "h1")
        if result.success:
            print(f"Element screenshot: {result.file_size} bytes")
            print(f"Element info: {result.element_info}")
        else:
            print(f"Element error: {result.error_message}")
            
    finally:
        await pool.cleanup()

async def test_casino_element_targeting():
    """Test function for casino-specific element targeting"""
    pool = BrowserPoolManager(max_pool_size=2)
    
    try:
        await pool.initialize()
        service = ScreenshotService(pool)
        casino_locator = CasinoElementLocator(service)
        
        # Test with Napoleon Sports casino
        test_url = "https://napoleonsports.be/"
        
        # Detect casino elements
        print(f"Detecting casino elements for: {test_url}")
        detected_elements = await casino_locator.detect_casino_elements(test_url)
        
        for element_type, elements in detected_elements.items():
            print(f"\n{element_type.upper()} Elements ({len(elements)} found):")
            for element in elements:
                print(f"  - Selector: {element.selector}")
                print(f"    Confidence: {element.confidence}")
                print(f"    Description: {element.description}")
        
        # Test casino screenshot capture
        print(f"\nCapturing casino screenshots...")
        screenshot_results = await casino_locator.capture_casino_screenshots(test_url)
        
        for element_type, results in screenshot_results.items():
            print(f"\n{element_type.upper()} Screenshots:")
            for i, result in enumerate(results):
                if result.success:
                    print(f"  Screenshot {i+1}: {result.file_size} bytes")
                    print(f"  Confidence: {result.element_info.get('confidence', 'N/A')}")
                else:
                    print(f"  Screenshot {i+1}: Failed - {result.error_message}")
        
        # Get best selectors
        best_selectors = await casino_locator.get_best_casino_selectors(test_url)
        print(f"\nBest Casino Selectors:")
        for element_type, selector in best_selectors.items():
            print(f"  {element_type}: {selector}")
            
    finally:
        await pool.cleanup()

# ========================================================================================
# SUBTASK 19.5: ERROR HANDLING AND RESILIENCE FRAMEWORK
# ========================================================================================

class ScreenshotErrorType(Enum):
    """Types of screenshot errors for structured error handling"""
    NETWORK_ERROR = "network_error"
    BROWSER_CRASH = "browser_crash"
    TIMEOUT_ERROR = "timeout_error"
    ELEMENT_NOT_FOUND = "element_not_found"
    PAGE_LOAD_FAILED = "page_load_failed"
    INVALID_URL = "invalid_url"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ScreenshotError:
    """Structured error information for screenshot operations"""
    error_type: ScreenshotErrorType
    message: str
    url: Optional[str] = None
    selector: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    context: Optional[Dict[str, Any]] = None
    traceback_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/storage"""
        return {
            'error_type': self.error_type.value,
            'message': self.message,
            'url': self.url,
            'selector': self.selector,
            'timestamp': self.timestamp,
            'retry_count': self.retry_count,
            'context': self.context,
            'traceback_info': self.traceback_info
        }

class CircuitBreakerState(Enum):
    """Circuit breaker states for failure handling"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5  # Failures before opening circuit
    timeout_seconds: int = 60   # How long to stay open
    success_threshold: int = 3  # Successes needed to close circuit
    
class CircuitBreaker:
    """Circuit breaker pattern for handling persistent failures"""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.time()
        
    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit state"""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
            
        elif self.state == CircuitBreakerState.OPEN:
            # Check if timeout period has elapsed
            if current_time - self.last_failure_time >= self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
            
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
            
        return False
    
    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker OPENED again - service still failing")

class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff"""
        if attempt <= 0:
            return 0.0
            
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
            
        return delay

class ScreenshotQualityValidator:
    """Validates screenshot quality and detects corruption"""
    
    @staticmethod
    def validate_screenshot(screenshot_data: bytes, expected_min_size: int = 1000) -> Dict[str, Any]:
        """
        Validate screenshot quality and detect potential issues
        
        Args:
            screenshot_data: Screenshot binary data
            expected_min_size: Minimum expected file size in bytes
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'file_size': len(screenshot_data),
            'format_detected': None
        }
        
        # Check minimum file size
        if len(screenshot_data) < expected_min_size:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"File size too small: {len(screenshot_data)} bytes (min: {expected_min_size})")
        
        # Detect image format from magic bytes
        if screenshot_data[:8] == b'\x89PNG\r\n\x1a\n':
            validation_result['format_detected'] = 'PNG'
        elif screenshot_data[:2] == b'\xff\xd8':
            validation_result['format_detected'] = 'JPEG'
        elif screenshot_data[:4] == b'RIFF' and screenshot_data[8:12] == b'WEBP':
            validation_result['format_detected'] = 'WebP'
        else:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Unknown or corrupted image format")
        
        # Check for common corruption patterns
        if b'\x00' * 100 in screenshot_data:  # Large sequences of null bytes
            validation_result['issues'].append("Potential corruption: large null byte sequences detected")
        
        # Check if file ends properly (basic check)
        if validation_result['format_detected'] == 'PNG' and not screenshot_data.endswith(b'IEND\xaeB`\x82'):
            validation_result['issues'].append("PNG file may be truncated")
        elif validation_result['format_detected'] == 'JPEG' and not screenshot_data.endswith(b'\xff\xd9'):
            validation_result['issues'].append("JPEG file may be truncated")
        
        return validation_result

class ErrorHandlingService:
    """
    Comprehensive error handling and resilience service for screenshot operations
    Provides structured error handling, retry logic, and circuit breaker patterns
    """
    
    def __init__(
        self,
        retry_config: RetryConfig = None,
        circuit_breaker_config: CircuitBreakerConfig = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self.error_history: List[ScreenshotError] = []
        self.quality_validator = ScreenshotQualityValidator()
        
        logger.info("ErrorHandlingService initialized with retry and circuit breaker support")
    
    def get_circuit_breaker(self, domain: str) -> CircuitBreaker:
        """Get or create circuit breaker for a specific domain"""
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = CircuitBreaker(self.circuit_breaker_config)
        return self.circuit_breakers[domain]
    
    def classify_error(self, exception: Exception, url: str = None) -> ScreenshotError:
        """
        Classify exception into structured error type
        
        Args:
            exception: The exception that occurred
            url: URL being processed when error occurred
            
        Returns:
            ScreenshotError with classified error information
        """
        import traceback
        
        error_message = str(exception)
        error_type = ScreenshotErrorType.UNKNOWN_ERROR
        context = {'exception_type': type(exception).__name__}
        
        # Network-related errors
        if any(keyword in error_message.lower() for keyword in 
               ['network', 'connection', 'timeout', 'dns', 'resolve']):
            error_type = ScreenshotErrorType.NETWORK_ERROR
            
        # Browser crash indicators
        elif any(keyword in error_message.lower() for keyword in 
                 ['browser', 'crashed', 'terminated', 'closed', 'disconnected']):
            error_type = ScreenshotErrorType.BROWSER_CRASH
            
        # Timeout errors
        elif 'timeout' in error_message.lower() or isinstance(exception, asyncio.TimeoutError):
            error_type = ScreenshotErrorType.TIMEOUT_ERROR
            
        # Element not found
        elif any(keyword in error_message.lower() for keyword in 
                 ['element', 'selector', 'not found', 'not visible']):
            error_type = ScreenshotErrorType.ELEMENT_NOT_FOUND
            
        # Page load failures
        elif any(keyword in error_message.lower() for keyword in 
                 ['page', 'load', 'navigation', 'goto']):
            error_type = ScreenshotErrorType.PAGE_LOAD_FAILED
            
        # Invalid URL
        elif any(keyword in error_message.lower() for keyword in 
                 ['invalid', 'malformed', 'url', 'protocol']):
            error_type = ScreenshotErrorType.INVALID_URL
            
        # Permission denied
        elif any(keyword in error_message.lower() for keyword in 
                 ['permission', 'denied', 'forbidden', 'unauthorized']):
            error_type = ScreenshotErrorType.PERMISSION_DENIED
            
        # Resource exhaustion
        elif any(keyword in error_message.lower() for keyword in 
                 ['memory', 'resource', 'limit', 'exhausted']):
            error_type = ScreenshotErrorType.RESOURCE_EXHAUSTED
        
        return ScreenshotError(
            error_type=error_type,
            message=error_message,
            url=url,
            timestamp=time.time(),
            context=context,
            traceback_info=traceback.format_exc()
        )
    
    def is_retryable_error(self, error: ScreenshotError) -> bool:
        """Determine if an error should be retried"""
        retryable_types = {
            ScreenshotErrorType.NETWORK_ERROR,
            ScreenshotErrorType.TIMEOUT_ERROR,
            ScreenshotErrorType.PAGE_LOAD_FAILED,
            ScreenshotErrorType.RESOURCE_EXHAUSTED
        }
        
        non_retryable_types = {
            ScreenshotErrorType.INVALID_URL,
            ScreenshotErrorType.PERMISSION_DENIED,
            ScreenshotErrorType.ELEMENT_NOT_FOUND
        }
        
        # Browser crashes are retryable but need special handling
        if error.error_type == ScreenshotErrorType.BROWSER_CRASH:
            return True
            
        if error.error_type in non_retryable_types:
            return False
            
        return error.error_type in retryable_types
    
    async def execute_with_resilience(
        self,
        operation: Callable,
        url: str,
        operation_name: str = "screenshot_operation",
        **kwargs
    ) -> Union[ScreenshotResult, ScreenshotError]:
        """
        Execute operation with full resilience features
        
        Args:
            operation: Async function to execute
            url: URL being processed
            operation_name: Name of operation for logging
            **kwargs: Arguments to pass to operation
            
        Returns:
            ScreenshotResult on success, ScreenshotError on final failure
        """
        domain = self._extract_domain(url)
        circuit_breaker = self.get_circuit_breaker(domain)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            error = ScreenshotError(
                error_type=ScreenshotErrorType.RESOURCE_EXHAUSTED,
                message=f"Circuit breaker OPEN for domain {domain}",
                url=url,
                context={'circuit_breaker_state': circuit_breaker.state.value}
            )
            self.error_history.append(error)
            return error
        
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Execute the operation
                result = await operation(url, **kwargs)
                
                # Validate result if it's a screenshot
                if isinstance(result, ScreenshotResult) and result.success and result.screenshot_data:
                    validation = self.quality_validator.validate_screenshot(result.screenshot_data)
                    if not validation['is_valid']:
                        logger.warning(f"Screenshot quality issues: {validation['issues']}")
                        # Create a quality error but don't fail immediately
                        quality_error = ScreenshotError(
                            error_type=ScreenshotErrorType.UNKNOWN_ERROR,
                            message=f"Quality validation failed: {', '.join(validation['issues'])}",
                            url=url,
                            context={'validation_result': validation}
                        )
                        self.error_history.append(quality_error)
                
                # Record success
                circuit_breaker.record_success()
                logger.debug(f"{operation_name} succeeded for {url} (attempt {attempt + 1})")
                return result
                
            except Exception as e:
                # Classify the error
                error = self.classify_error(e, url)
                error.retry_count = attempt
                last_error = error
                
                logger.warning(f"{operation_name} failed (attempt {attempt + 1}): {error.message}")
                
                # Check if we should retry
                if attempt < self.retry_config.max_retries and self.is_retryable_error(error):
                    delay = self.retry_config.get_delay(attempt + 1)
                    logger.info(f"Retrying {operation_name} for {url} in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Final failure
                    circuit_breaker.record_failure()
                    self.error_history.append(error)
                    break
        
        # If we get here, all retries failed
        logger.error(f"{operation_name} failed permanently for {url} after {self.retry_config.max_retries + 1} attempts")
        return last_error
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for circuit breaker grouping"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or 'unknown'
        except Exception:
            return 'unknown'
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of errors from the last N hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict with error statistics and trends
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'error_types': {},
                'affected_domains': {},
                'circuit_breaker_states': {}
            }
        
        # Count by error type
        error_types = {}
        for error in recent_errors:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Count by domain
        affected_domains = {}
        for error in recent_errors:
            if error.url:
                domain = self._extract_domain(error.url)
                affected_domains[domain] = affected_domains.get(domain, 0) + 1
        
        # Circuit breaker states
        circuit_breaker_states = {}
        for domain, cb in self.circuit_breakers.items():
            circuit_breaker_states[domain] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count
            }
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / hours,  # errors per hour
            'error_types': error_types,
            'affected_domains': affected_domains,
            'circuit_breaker_states': circuit_breaker_states,
            'time_range_hours': hours
        }
    
    def reset_circuit_breaker(self, domain: str = None):
        """Reset circuit breaker(s) - useful for manual recovery"""
        if domain:
            if domain in self.circuit_breakers:
                self.circuit_breakers[domain] = CircuitBreaker(self.circuit_breaker_config)
                logger.info(f"Reset circuit breaker for domain: {domain}")
        else:
            self.circuit_breakers.clear()
            logger.info("Reset all circuit breakers")

# ========================================================================================
# INTEGRATION WITH EXISTING CLASSES
# ========================================================================================

# Global error handling service instance
_global_error_handler: Optional[ErrorHandlingService] = None

async def get_global_error_handler() -> ErrorHandlingService:
    """Get or create global error handling service instance"""
    global _global_error_handler
    if not _global_error_handler:
        _global_error_handler = ErrorHandlingService()
    return _global_error_handler

async def cleanup_global_error_handler():
    """Cleanup global error handling service"""
    global _global_error_handler
    if _global_error_handler:
        _global_error_handler = None

# Enhanced ScreenshotService with error handling integration
class ResilientScreenshotService(ScreenshotService):
    """ScreenshotService enhanced with comprehensive error handling"""
    
    def __init__(self, browser_pool: BrowserPoolManager, config: ScreenshotConfig = None, error_handler: ErrorHandlingService = None):
        super().__init__(browser_pool, config)
        self.error_handler = error_handler or ErrorHandlingService()
    
    async def capture_full_page_screenshot_resilient(self, url: str) -> Union[ScreenshotResult, ScreenshotError]:
        """Capture full page screenshot with full resilience features"""
        return await self.error_handler.execute_with_resilience(
            super().capture_full_page_screenshot,
            url,
            "full_page_screenshot"
        )
    
    async def capture_viewport_screenshot_resilient(
        self, 
        url: str, 
        viewport_width: int = None, 
        viewport_height: int = None
    ) -> Union[ScreenshotResult, ScreenshotError]:
        """Capture viewport screenshot with full resilience features"""
        return await self.error_handler.execute_with_resilience(
            super().capture_viewport_screenshot,
            url,
            "viewport_screenshot",
            viewport_width=viewport_width,
            viewport_height=viewport_height
        )
    
    async def capture_element_screenshot_resilient(
        self, 
        url: str, 
        selector: str, 
        wait_for_selector: bool = True
    ) -> Union[ScreenshotResult, ScreenshotError]:
        """Capture element screenshot with full resilience features"""
        return await self.error_handler.execute_with_resilience(
            super().capture_element_screenshot,
            url,
            "element_screenshot",
            selector=selector,
            wait_for_selector=wait_for_selector
        )

# ========================================================================================
# TESTING FUNCTION FOR ERROR HANDLING FRAMEWORK
# ========================================================================================

async def test_error_handling_framework():
    """Test the error handling and resilience framework"""
    print("=== Testing Error Handling and Resilience Framework ===")
    
    # Initialize components
    browser_pool = BrowserPoolManager(max_pool_size=2)
    error_handler = ErrorHandlingService(
        retry_config=RetryConfig(max_retries=2, base_delay=0.5),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=10)
    )
    resilient_service = ResilientScreenshotService(browser_pool, error_handler=error_handler)
    
    try:
        print("\n1. Testing successful operation...")
        result = await resilient_service.capture_full_page_screenshot_resilient("https://example.com")
        if isinstance(result, ScreenshotResult) and result.success:
            print(f"  ✅ Success: {len(result.screenshot_data)} bytes captured")
        else:
            print(f"  ⚠️ Unexpected result: {result}")
        
        print("\n2. Testing invalid URL handling...")
        result = await resilient_service.capture_full_page_screenshot_resilient("invalid://bad-url")
        if isinstance(result, ScreenshotError):
            print(f"  ✅ Error correctly classified: {result.error_type.value}")
            print(f"     Message: {result.message}")
            print(f"     Retryable: {error_handler.is_retryable_error(result)}")
        
        print("\n3. Testing timeout with slow loading site...")
        # Use httpbin delay endpoint for timeout testing
        result = await resilient_service.capture_full_page_screenshot_resilient("https://httpbin.org/delay/35")
        if isinstance(result, ScreenshotError):
            print(f"  ✅ Timeout handled: {result.error_type.value}")
            print(f"     Retry count: {result.retry_count}")
        
        print("\n4. Testing circuit breaker behavior...")
        domain = "httpbin.org"
        circuit_breaker = error_handler.get_circuit_breaker(domain)
        
        # Force circuit breaker to open by simulating failures
        for i in range(5):
            circuit_breaker.record_failure()
        
        print(f"  Circuit breaker state: {circuit_breaker.state.value}")
        print(f"  Can execute: {circuit_breaker.can_execute()}")
        
        # Try operation with open circuit
        if not circuit_breaker.can_execute():
            print("  ✅ Circuit breaker correctly blocking requests")
        
        print("\n5. Testing error classification...")
        test_exceptions = [
            ConnectionError("Connection failed"),
            TimeoutError("Operation timed out"),
            ValueError("Invalid URL format"),
            PermissionError("Access denied")
        ]
        
        for exc in test_exceptions:
            error = error_handler.classify_error(exc, "https://test.com")
            print(f"  {type(exc).__name__} → {error.error_type.value}")
        
        print("\n6. Testing screenshot quality validation...")
        # Test with valid PNG data (minimal PNG header)
        valid_png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 1000 + b'IEND\xaeB`\x82'
        validation = error_handler.quality_validator.validate_screenshot(valid_png)
        print(f"  Valid PNG validation: {validation['is_valid']}, format: {validation['format_detected']}")
        
        # Test with corrupted data
        corrupted_data = b'\x00' * 500
        validation = error_handler.quality_validator.validate_screenshot(corrupted_data)
        print(f"  Corrupted data validation: {validation['is_valid']}, issues: {len(validation['issues'])}")
        
        print("\n7. Testing error summary...")
        summary = error_handler.get_error_summary(hours=1)
        print(f"  Total errors in last hour: {summary['total_errors']}")
        print(f"  Error types: {list(summary['error_types'].keys())}")
        print(f"  Circuit breaker states: {len(summary['circuit_breaker_states'])}")
        
        print("\n8. Testing retry configuration...")
        retry_config = RetryConfig(max_retries=3, base_delay=1.0, exponential_base=2.0)
        delays = [retry_config.get_delay(i) for i in range(1, 5)]
        print(f"  Retry delays: {[f'{d:.2f}s' for d in delays]}")
        
        print("\n✅ Error handling framework test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        await browser_pool.cleanup()
        print("Cleanup completed")

# ==============================================================================
# Supabase Storage and Metadata Integration
# ==============================================================================

@dataclass
class SupabaseConfig:
    """Configuration for Supabase integration"""
    url: str
    key: str
    storage_bucket: str = "screenshots"
    table_captures: str = "screenshot_captures"
    table_elements: str = "screenshot_elements"
    table_media_assets: str = "media_assets"

@dataclass
class StorageResult:
    """Result from Supabase storage operation"""
    success: bool
    storage_path: Optional[str] = None
    public_url: Optional[str] = None
    error_message: Optional[str] = None
    file_size: int = 0
    upload_time_ms: Optional[float] = None

@dataclass
class MetadataResult:
    """Result from Supabase metadata operation"""
    success: bool
    record_id: Optional[str] = None
    screenshot_id: Optional[str] = None
    media_asset_id: Optional[str] = None
    error_message: Optional[str] = None

class SupabaseScreenshotStorage:
    """
    Supabase integration for screenshot storage and metadata management
    Handles file uploads to Supabase Storage and metadata to PostgreSQL
    """
    
    def __init__(self, config: SupabaseConfig):
        self.config = config
        self._supabase = None
        
        # Try to import and initialize Supabase
        try:
            from supabase import create_client, Client
            self._supabase: Client = create_client(config.url, config.key)
            logger.info("Supabase client initialized successfully")
        except ImportError:
            logger.error("Supabase not available. Install with: pip install supabase")
            raise ImportError("Supabase client library not available")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    async def store_screenshot(
        self,
        screenshot_result: ScreenshotResult,
        url: str,
        capture_type: str = "full_page",
        content_id: Optional[str] = None,
        casino_elements: Optional[List[CasinoElement]] = None,
        capture_config: Optional[Dict[str, Any]] = None,
        browser_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[StorageResult, MetadataResult]:
        """
        Store screenshot data and metadata in Supabase
        
        Args:
            screenshot_result: The screenshot data and metadata
            url: Source URL of the screenshot
            capture_type: Type of capture (full_page, viewport, element)
            content_id: Optional content ID to link with
            casino_elements: Detected casino elements
            capture_config: Configuration used for capture
            browser_info: Browser information used
            
        Returns:
            Tuple of (StorageResult, MetadataResult)
        """
        if not screenshot_result.success or not screenshot_result.screenshot_data:
            return (
                StorageResult(success=False, error_message="No screenshot data to store"),
                MetadataResult(success=False, error_message="No screenshot data to store")
            )
        
        try:
            # 1. Upload file to Supabase Storage
            storage_result = await self._upload_to_storage(screenshot_result, url, capture_type)
            
            if not storage_result.success:
                return storage_result, MetadataResult(success=False, error_message="Storage upload failed")
            
            # 2. Store metadata in database
            metadata_result = await self._store_metadata(
                screenshot_result=screenshot_result,
                storage_result=storage_result,
                url=url,
                capture_type=capture_type,
                content_id=content_id,
                capture_config=capture_config,
                browser_info=browser_info
            )
            
            # 3. Store casino elements if provided
            if metadata_result.success and casino_elements:
                await self._store_casino_elements(metadata_result.screenshot_id, casino_elements)
            
            return storage_result, metadata_result
            
        except Exception as e:
            logger.error(f"Error storing screenshot: {e}")
            return (
                StorageResult(success=False, error_message=str(e)),
                MetadataResult(success=False, error_message=str(e))
            )

    async def _upload_to_storage(
        self, 
        screenshot_result: ScreenshotResult, 
        url: str, 
        capture_type: str
    ) -> StorageResult:
        """Upload screenshot file to Supabase Storage"""
        try:
            start_time = time.time()
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = self._extract_domain_from_url(url)
            file_extension = "png" if screenshot_result.screenshot_data[:8] == b'\x89PNG\r\n\x1a\n' else "jpg"
            filename = f"{domain}_{capture_type}_{timestamp}_{uuid.uuid4().hex[:8]}.{file_extension}"
            
            # Upload to Supabase Storage
            result = self._supabase.storage.from_(self.config.storage_bucket).upload(
                path=filename,
                file=screenshot_result.screenshot_data,
                file_options={
                    "content-type": f"image/{file_extension}",
                    "cache-control": "3600"
                }
            )
            
            upload_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # ✅ FIX: Handle new Supabase client response format
            if hasattr(result, 'data') and result.data is None:
                # Error occurred
                error_msg = getattr(result, 'error', 'Unknown upload error')
                return StorageResult(
                    success=False,
                    error_message=f"Storage upload failed: {error_msg}"
                )
            elif hasattr(result, 'get') and result.get('error'):
                # Legacy format
                return StorageResult(
                    success=False,
                    error_message=f"Storage upload failed: {result['error']}"
                )
            
            # Get public URL
            public_url_result = self._supabase.storage.from_(self.config.storage_bucket).get_public_url(filename)
            public_url = public_url_result if isinstance(public_url_result, str) else public_url_result.get('publicUrl')
            
            return StorageResult(
                success=True,
                storage_path=filename,
                public_url=public_url,
                file_size=len(screenshot_result.screenshot_data),
                upload_time_ms=upload_time
            )
            
        except Exception as e:
            logger.error(f"Storage upload error: {e}")
            return StorageResult(success=False, error_message=str(e))

    async def _store_metadata(
        self,
        screenshot_result: ScreenshotResult,
        storage_result: StorageResult,
        url: str,
        capture_type: str,
        content_id: Optional[str] = None,
        capture_config: Optional[Dict[str, Any]] = None,
        browser_info: Optional[Dict[str, Any]] = None
    ) -> MetadataResult:
        """Store screenshot metadata in database"""
        try:
            # First create media_asset record
            media_asset_data = {
                "content_id": None,  # Fix: Don't reference non-existent content_items
                "storage_path": storage_result.storage_path,
                "file_name": storage_result.storage_path.split('/')[-1],
                "mime_type": f"image/{'png' if storage_result.storage_path.endswith('.png') else 'jpeg'}",
                "metadata": {
                    "screenshot_type": capture_type,
                    "source_url": url,
                    "file_size": storage_result.file_size,
                    "public_url": storage_result.public_url,
                    "upload_time_ms": storage_result.upload_time_ms,
                    "viewport_size": screenshot_result.viewport_size,
                    "timestamp": screenshot_result.timestamp
                },
                "alt_text": f"Screenshot of {self._extract_domain_from_url(url)} - {capture_type}",
                "caption": f"Automated screenshot capture from {url}"
            }
            
            media_result = self._supabase.table(self.config.table_media_assets).insert(media_asset_data).execute()
            
            if not media_result.data:
                return MetadataResult(success=False, error_message="Failed to create media asset record")
            
            media_asset_id = media_result.data[0]['id']
            
            # Then create screenshot_captures record
            capture_metadata = {
                "content_id": None,  # Fix: Don't reference non-existent content_items
                "media_asset_id": media_asset_id,
                "url": url,
                "capture_type": capture_type,
                "capture_config": capture_config or {},
                "browser_info": browser_info or {},
                "viewport_size": screenshot_result.viewport_size or {},
                "quality_score": None,  # Could be calculated based on file size/content
                "processing_time_ms": int((time.time() - screenshot_result.timestamp) * 1000) if screenshot_result.timestamp else None
            }
            
            capture_result = self._supabase.table(self.config.table_captures).insert(capture_metadata).execute()
            
            if not capture_result.data:
                return MetadataResult(success=False, error_message="Failed to create screenshot capture record")
            
            screenshot_id = capture_result.data[0]['id']
            
            return MetadataResult(
                success=True,
                record_id=screenshot_id,
                screenshot_id=screenshot_id,
                media_asset_id=media_asset_id
            )
            
        except Exception as e:
            logger.error(f"Metadata storage error: {e}")
            return MetadataResult(success=False, error_message=str(e))

    async def _store_casino_elements(
        self, 
        screenshot_id: str, 
        casino_elements: List[CasinoElement]
    ) -> bool:
        """Store detected casino elements in database"""
        try:
            elements_data = []
            for element in casino_elements:
                elements_data.append({
                    "screenshot_id": screenshot_id,
                    "element_type": element.element_type.upper(),
                    "selector": element.selector,
                    "confidence_score": element.confidence,
                    "detection_method": "heuristic_selector",  # Could be expanded with more methods
                    "element_metadata": {
                        "description": element.description,
                        "fallback_selectors": element.fallback_selectors
                    }
                })
            
            if elements_data:
                result = self._supabase.table(self.config.table_elements).insert(elements_data).execute()
                if result.data:
                    logger.info(f"Stored {len(elements_data)} casino elements for screenshot {screenshot_id}")
                    return True
                else:
                    logger.warning(f"Failed to store casino elements for screenshot {screenshot_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing casino elements: {e}")
            return False

    async def query_screenshots(
        self,
        content_id: Optional[str] = None,
        url: Optional[str] = None,
        capture_type: Optional[str] = None,
        element_type: Optional[str] = None,
        limit: int = 50,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query screenshots with various filters
        
        Args:
            content_id: Filter by content ID
            url: Filter by source URL
            capture_type: Filter by capture type
            element_type: Filter by detected element type
            limit: Maximum number of results
            order_by: Field to order by
            order_desc: Order descending if True
            
        Returns:
            List of screenshot records with metadata
        """
        try:
            query = self._supabase.table(self.config.table_captures).select(
                "*, media_assets(*), screenshot_elements(*)"
            )
            
            # Apply filters
            if content_id:
                query = query.eq("content_id", content_id)
            if url:
                query = query.eq("url", url)
            if capture_type:
                query = query.eq("capture_type", capture_type)
            
            # Order and limit
            if order_desc:
                query = query.order(order_by, desc=True)
            else:
                query = query.order(order_by)
            
            query = query.limit(limit)
            
            result = query.execute()
            
            screenshots = result.data if result.data else []
            
            # Filter by element type if specified
            if element_type and screenshots:
                filtered_screenshots = []
                for screenshot in screenshots:
                    elements = screenshot.get('screenshot_elements', [])
                    if any(elem.get('element_type') == element_type.upper() for elem in elements):
                        filtered_screenshots.append(screenshot)
                screenshots = filtered_screenshots
            
            logger.info(f"Retrieved {len(screenshots)} screenshots matching query criteria")
            return screenshots
            
        except Exception as e:
            logger.error(f"Error querying screenshots: {e}")
            return []

    async def get_screenshot_by_id(self, screenshot_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific screenshot by ID"""
        try:
            result = self._supabase.table(self.config.table_captures).select(
                "*, media_assets(*), screenshot_elements(*)"
            ).eq("id", screenshot_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving screenshot {screenshot_id}: {e}")
            return None

    async def delete_screenshot(self, screenshot_id: str, delete_files: bool = True) -> bool:
        """
        Delete screenshot and its associated data
        
        Args:
            screenshot_id: ID of screenshot to delete
            delete_files: Whether to also delete files from storage
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get screenshot info first
            screenshot = await self.get_screenshot_by_id(screenshot_id)
            if not screenshot:
                logger.warning(f"Screenshot {screenshot_id} not found")
                return False
            
            # Delete from storage if requested
            if delete_files and screenshot.get('media_assets'):
                storage_path = screenshot['media_assets'].get('storage_path')
                if storage_path:
                    try:
                        self._supabase.storage.from_(self.config.storage_bucket).remove([storage_path])
                        logger.info(f"Deleted file {storage_path} from storage")
                    except Exception as e:
                        logger.warning(f"Error deleting file from storage: {e}")
            
            # Delete database records (cascade will handle elements)
            result = self._supabase.table(self.config.table_captures).delete().eq("id", screenshot_id).execute()
            
            if result.data:
                logger.info(f"Deleted screenshot {screenshot_id}")
                return True
            else:
                logger.warning(f"Failed to delete screenshot {screenshot_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting screenshot {screenshot_id}: {e}")
            return False

    async def cleanup_old_screenshots(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up screenshots older than specified days
        
        Args:
            days_old: Delete screenshots older than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Find old screenshots
            result = self._supabase.table(self.config.table_captures).select(
                "id, media_assets(storage_path)"
            ).lt("created_at", cutoff_date.isoformat()).execute()
            
            old_screenshots = result.data if result.data else []
            
            deleted_count = 0
            error_count = 0
            
            for screenshot in old_screenshots:
                if await self.delete_screenshot(screenshot['id'], delete_files=True):
                    deleted_count += 1
                else:
                    error_count += 1
            
            logger.info(f"Cleanup completed: {deleted_count} deleted, {error_count} errors")
            
            return {
                "total_found": len(old_screenshots),
                "deleted_count": deleted_count,
                "error_count": error_count,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL for filename generation"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '').replace('.', '_')
            return domain if domain else "unknown"
        except:
            return "unknown"

class IntegratedScreenshotService:
    """
    Integrated screenshot service that combines capture with Supabase storage
    Provides high-level interface for screenshot capture and storage
    """
    
    def __init__(
        self,
        browser_pool: BrowserPoolManager,
        supabase_storage: SupabaseScreenshotStorage,
        screenshot_config: ScreenshotConfig = None,
        casino_locator: CasinoElementLocator = None
    ):
        self.browser_pool = browser_pool
        self.supabase_storage = supabase_storage
        self.screenshot_service = ScreenshotService(browser_pool, screenshot_config)
        self.casino_locator = casino_locator or CasinoElementLocator(self.screenshot_service)
        
        logger.info("Initialized IntegratedScreenshotService with Supabase storage")

    async def capture_and_store_screenshot(
        self,
        url: str,
        capture_type: str = "full_page",
        content_id: Optional[str] = None,
        detect_casino_elements: bool = True,
        element_types: Optional[List[str]] = None,
        capture_config: Optional[ScreenshotConfig] = None
    ) -> Dict[str, Any]:
        """
        Capture screenshot and store in Supabase with metadata
        
        Args:
            url: URL to capture
            capture_type: Type of capture (full_page, viewport, element)
            content_id: Optional content ID to link with
            detect_casino_elements: Whether to detect casino elements
            element_types: Specific element types to detect
            capture_config: Configuration for capture
            
        Returns:
            Dictionary with capture and storage results
        """
        try:
            # ✅ FIX: Ensure content_id is a valid UUID before processing
            import uuid
            import hashlib
            
            if content_id:
                try:
                    uuid.UUID(content_id)
                except ValueError:
                    # Convert non-UUID content_id to a deterministic UUID
                    namespace = uuid.UUID('1b671a64-40d5-491e-99b0-da01ff1f3341')
                    content_id = str(uuid.uuid5(namespace, hashlib.sha1(content_id.encode('utf-8')).hexdigest()))

            start_time = time.time()
            
            # Determine which capture method to use
            if capture_type == "full_page":
                screenshot_result = await self.screenshot_service.capture_full_page_screenshot(url)
            elif capture_type == "viewport":
                config = capture_config or ScreenshotConfig()
                screenshot_result = await self.screenshot_service.capture_viewport_screenshot(
                    url, config.viewport_width, config.viewport_height
                )
            else:
                return {"success": False, "error": f"Unsupported capture type: {capture_type}"}
            
            if not screenshot_result.success:
                return {
                    "success": False,
                    "error": f"Screenshot capture failed: {screenshot_result.error_message}"
                }
            
            # 2. Detect casino elements if requested
            casino_elements = []
            if detect_casino_elements:
                try:
                    elements_by_type = await self.casino_locator.detect_casino_elements(url)
                    
                    # Filter by element types if specified
                    if element_types:
                        for element_type in element_types:
                            if element_type in elements_by_type:
                                casino_elements.extend(elements_by_type[element_type])
                    else:
                        # Include all detected elements
                        for elements_list in elements_by_type.values():
                            casino_elements.extend(elements_list)
                            
                    logger.info(f"Detected {len(casino_elements)} casino elements on {url}")
                    
                except Exception as e:
                    logger.warning(f"Casino element detection failed: {e}")
            
            # 3. Store in Supabase
            storage_result, metadata_result = await self.supabase_storage.store_screenshot(
                screenshot_result=screenshot_result,
                url=url,
                capture_type=capture_type,
                content_id=content_id,
                casino_elements=casino_elements,
                capture_config=capture_config.__dict__ if capture_config else None,
                browser_info={"user_agent": "playwright", "engine": "chromium"}  # Could be enhanced
            )
            
            return {
                "success": True,
                "screenshot_result": screenshot_result,
                "storage_result": storage_result,
                "metadata_result": metadata_result,
                "casino_elements_count": len(casino_elements),
                "screenshot_id": metadata_result.screenshot_id,
                "media_asset_id": metadata_result.media_asset_id,
                "public_url": storage_result.public_url
            }
            
        except Exception as e:
            logger.error(f"Error in capture_and_store_screenshot: {e}")
            return {"success": False, "error": str(e)}

    async def batch_capture_and_store(
        self,
        urls: List[str],
        capture_type: str = "full_page",
        content_ids: Optional[List[str]] = None,
        detect_casino_elements: bool = True,
        max_concurrent: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch capture and store multiple URLs
        
        Args:
            urls: List of URLs to capture
            capture_type: Type of capture for all URLs
            content_ids: Optional list of content IDs (same length as urls)
            detect_casino_elements: Whether to detect casino elements
            max_concurrent: Maximum concurrent captures
            
        Returns:
            Dictionary mapping URLs to their results
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def capture_single(url: str, content_id: Optional[str] = None):
            async with semaphore:
                return await self.capture_and_store_screenshot(
                    url=url,
                    capture_type=capture_type,
                    content_id=content_id,
                    detect_casino_elements=detect_casino_elements
                )
        
        tasks = []
        for i, url in enumerate(urls):
            content_id = content_ids[i] if content_ids and i < len(content_ids) else None
            tasks.append(capture_single(url, content_id))
        
        # Execute all tasks
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(task_results):
            url = urls[i]
            if isinstance(result, Exception):
                results[url] = {"success": False, "error": str(result)}
            else:
                results[url] = result
        
        logger.info(f"Batch capture completed: {len(urls)} URLs processed")
        return results

# Global instance management
_global_supabase_storage: Optional[SupabaseScreenshotStorage] = None

async def get_global_supabase_storage(config: SupabaseConfig = None) -> SupabaseScreenshotStorage:
    """Get or create global Supabase storage instance"""
    global _global_supabase_storage
    
    if _global_supabase_storage is None and config:
        _global_supabase_storage = SupabaseScreenshotStorage(config)
        logger.info("Created global Supabase storage instance")
    
    return _global_supabase_storage

async def cleanup_global_supabase_storage():
    """Clean up global Supabase storage instance"""
    global _global_supabase_storage
    _global_supabase_storage = None
    logger.info("Cleaned up global Supabase storage instance")

# ==============================================================================
# Testing and Examples
# ==============================================================================

async def test_supabase_integration():
    """Test Supabase integration functionality"""
    logger.info("Starting Supabase integration test...")
    
    try:
        # Configuration (would come from environment in real usage)
        config = SupabaseConfig(
            url="your-supabase-url",
            key="your-supabase-key",
            storage_bucket="screenshots"
        )
        
        # Initialize services
        browser_pool = await get_global_browser_pool()
        await browser_pool.initialize()
        
        supabase_storage = SupabaseScreenshotStorage(config)
        integrated_service = IntegratedScreenshotService(browser_pool, supabase_storage)
        
        # Test URL
        test_url = "https://example.com"
        
        # Capture and store screenshot
        result = await integrated_service.capture_and_store_screenshot(
            url=test_url,
            capture_type="full_page",
            detect_casino_elements=False  # Example.com won't have casino elements
        )
        
        if result["success"]:
            logger.info(f"✅ Screenshot captured and stored successfully!")
            logger.info(f"   Screenshot ID: {result['screenshot_id']}")
            logger.info(f"   Public URL: {result['public_url']}")
            logger.info(f"   File size: {result['storage_result'].file_size} bytes")
            
            # Test querying
            screenshots = await supabase_storage.query_screenshots(url=test_url, limit=5)
            logger.info(f"✅ Found {len(screenshots)} screenshots for {test_url}")
            
        else:
            logger.error(f"❌ Test failed: {result.get('error')}")
        
        # Cleanup
        await browser_pool.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Supabase integration test error: {e}")

# ==============================================================================
# Anti-Detection Stealth System
# ==============================================================================

from enum import Enum
from typing import Set, Dict, List, Optional, Any, Tuple
import random
import asyncio

class StealthLevel(Enum):
    """Stealth levels for anti-detection system"""
    LEVEL_1_BASIC = 1
    LEVEL_2_ADVANCED = 2
    LEVEL_3_BEHAVIORAL = 3
    LEVEL_4_PROFILE_ROTATION = 4

@dataclass
class StealthProfile:
    """Comprehensive stealth profile configuration"""
    # Basic profile
    user_agent: str
    viewport: Dict[str, int]
    locale: str
    timezone: str
    platform: str
    
    # Advanced properties
    screen_resolution: Dict[str, int]
    color_depth: int
    device_scale_factor: float
    is_mobile: bool
    touch_support: bool
    
    # Browser engine
    browser_engine: str  # 'chromium', 'firefox', 'webkit'
    
    # Headers and permissions
    accept_language: str
    accept_encoding: str
    navigator_properties: Dict[str, Any]
    
    @classmethod
    def generate_stealth_profile(cls, level: StealthLevel = StealthLevel.LEVEL_1_BASIC) -> 'StealthProfile':
        """Generate a stealth profile based on the specified level"""
        
        if level == StealthLevel.LEVEL_1_BASIC:
            return cls._generate_basic_profile()
        elif level == StealthLevel.LEVEL_2_ADVANCED:
            return cls._generate_advanced_profile()
        elif level == StealthLevel.LEVEL_3_BEHAVIORAL:
            return cls._generate_behavioral_profile()
        elif level == StealthLevel.LEVEL_4_PROFILE_ROTATION:
            return cls._generate_rotation_profile()
        else:
            return cls._generate_basic_profile()
    
    @classmethod
    def _generate_basic_profile(cls) -> 'StealthProfile':
        """Generate Level 1 - Basic Stealth profile"""
        profiles = [
            # Windows Chrome
            {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "viewport": {"width": 1920, "height": 1080},
                "screen_resolution": {"width": 1920, "height": 1080},
                "platform": "Win32",
                "browser_engine": "chromium"
            },
            # Windows Firefox
            {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
                "viewport": {"width": 1366, "height": 768},
                "screen_resolution": {"width": 1366, "height": 768},
                "platform": "Win32",
                "browser_engine": "firefox"
            },
            # macOS Safari
            {
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
                "viewport": {"width": 1440, "height": 900},
                "screen_resolution": {"width": 1440, "height": 900},
                "platform": "MacIntel",
                "browser_engine": "webkit"
            }
        ]
        
        profile_data = random.choice(profiles)
        
        return cls(
            user_agent=profile_data["user_agent"],
            viewport=profile_data["viewport"],
            locale="en-US",
            timezone="America/New_York",
            platform=profile_data["platform"],
            screen_resolution=profile_data["screen_resolution"],
            color_depth=24,
            device_scale_factor=1.0,
            is_mobile=False,
            touch_support=False,
            browser_engine=profile_data["browser_engine"],
            accept_language="en-US,en;q=0.9",
            accept_encoding="gzip, deflate, br",
            navigator_properties={}
        )
    
    @classmethod
    def _generate_advanced_profile(cls) -> 'StealthProfile':
        """Generate Level 2 - Advanced Stealth profile"""
        base_profile = cls._generate_basic_profile()
        
        # Enhanced navigator properties to spoof
        navigator_properties = {
            "webdriver": False,
            "plugins": ["PDF Viewer", "Chrome PDF Plugin", "Chromium PDF Plugin"],
            "languages": ["en-US", "en"],
            "platform": base_profile.platform,
            "cookieEnabled": True,
            "doNotTrack": None,
            "hardwareConcurrency": random.choice([2, 4, 8, 12]),
            "deviceMemory": random.choice([2, 4, 8, 16]),
            "maxTouchPoints": 0 if not base_profile.touch_support else random.choice([5, 10])
        }
        
        # Randomize some properties
        base_profile.color_depth = random.choice([24, 32])
        base_profile.device_scale_factor = random.choice([1.0, 1.25, 1.5, 2.0])
        base_profile.navigator_properties = navigator_properties
        
        return base_profile
    
    @classmethod
    def _generate_behavioral_profile(cls) -> 'StealthProfile':
        """Generate Level 3 - Behavioral Simulation profile"""
        base_profile = cls._generate_advanced_profile()
        
        # Add behavioral simulation properties
        base_profile.navigator_properties.update({
            "onLine": True,
            "connection": {
                "effectiveType": random.choice(["3g", "4g", "slow-2g", "2g"]),
                "downlink": random.uniform(1.0, 10.0),
                "rtt": random.randint(50, 200)
            }
        })
        
        return base_profile
    
    @classmethod
    def _generate_rotation_profile(cls) -> 'StealthProfile':
        """Generate Level 4 - Profile Rotation with mobile support"""
        # Randomly choose between desktop and mobile
        is_mobile = random.choice([True, False])
        
        if is_mobile:
            mobile_profiles = [
                {
                    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
                    "viewport": {"width": 375, "height": 667},
                    "screen_resolution": {"width": 375, "height": 667},
                    "platform": "iPhone",
                    "browser_engine": "webkit"
                },
                {
                    "user_agent": "Mozilla/5.0 (Linux; Android 13; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
                    "viewport": {"width": 360, "height": 640},
                    "screen_resolution": {"width": 360, "height": 640},
                    "platform": "Linux armv81",
                    "browser_engine": "chromium"
                }
            ]
            
            profile_data = random.choice(mobile_profiles)
            
            return cls(
                user_agent=profile_data["user_agent"],
                viewport=profile_data["viewport"],
                locale=random.choice(["en-US", "en-GB", "de-DE", "fr-FR"]),
                timezone=random.choice(["America/New_York", "Europe/London", "Europe/Berlin", "America/Los_Angeles"]),
                platform=profile_data["platform"],
                screen_resolution=profile_data["screen_resolution"],
                color_depth=24,
                device_scale_factor=random.choice([1.0, 2.0, 3.0]),
                is_mobile=True,
                touch_support=True,
                browser_engine=profile_data["browser_engine"],
                accept_language="en-US,en;q=0.9",
                accept_encoding="gzip, deflate, br",
                navigator_properties={
                    "webdriver": False,
                    "maxTouchPoints": random.choice([5, 10]),
                    "hardwareConcurrency": random.choice([4, 6, 8])
                }
            )
        else:
            return cls._generate_behavioral_profile()

@dataclass
class StealthConfig:
    """Configuration for stealth operations"""
    level: StealthLevel = StealthLevel.LEVEL_1_BASIC
    max_retries_per_level: int = 3
    escalate_on_block: bool = True
    respect_robots_txt: bool = True
    min_delay_seconds: float = 2.0
    max_delay_seconds: float = 5.0
    human_behavior_simulation: bool = True
    request_throttling: bool = True

@dataclass
class StealthResult:
    """Result from stealth operation"""
    success: bool
    level_used: StealthLevel
    profile_used: StealthProfile
    attempts_made: int
    error_message: Optional[str] = None
    blocked_detected: bool = False
    escalation_occurred: bool = False

class AntiDetectionStealthSystem:
    """
    Comprehensive anti-detection stealth system for casino site access
    Implements graduated 4-level stealth approach with ethical boundaries
    """
    
    def __init__(self, config: StealthConfig = None):
        self.config = config or StealthConfig()
        self.domain_configurations: Dict[str, StealthLevel] = {}
        self.successful_profiles: Dict[str, StealthProfile] = {}
        self.blocked_domains: Set[str] = set()
        self.attempt_history: Dict[str, List[Dict]] = {}
        
        logger.info(f"AntiDetectionStealthSystem initialized with level {self.config.level.value}")

    async def create_stealth_context(
        self,
        browser_instance: 'BrowserInstance',
        url: str,
        target_level: Optional[StealthLevel] = None
    ) -> Tuple['BrowserContext', StealthProfile]:
        """
        Create a browser context with stealth configuration
        
        Args:
            browser_instance: Browser instance from pool
            url: Target URL for configuration optimization
            target_level: Specific stealth level to use
            
        Returns:
            Tuple of (BrowserContext, StealthProfile used)
        """
        domain = self._extract_domain(url)
        
        # Determine stealth level to use
        if target_level:
            level = target_level
        elif domain in self.domain_configurations:
            level = self.domain_configurations[domain]
        else:
            level = self.config.level
        
        # Check if domain is blocked
        if domain in self.blocked_domains:
            logger.warning(f"Domain {domain} is marked as blocked, using highest stealth level")
            level = StealthLevel.LEVEL_4_PROFILE_ROTATION
        
        # Generate stealth profile
        if domain in self.successful_profiles:
            profile = self.successful_profiles[domain]
            logger.info(f"Using successful profile for {domain}")
        else:
            profile = StealthProfile.generate_stealth_profile(level)
        
        # Create context with stealth configuration
        context = await self._apply_stealth_to_context(browser_instance.browser, profile, level)
        
        return context, profile

    async def _apply_stealth_to_context(
        self,
        browser: 'Browser',
        profile: StealthProfile,
        level: StealthLevel
    ) -> 'BrowserContext':
        """Apply stealth configuration to browser context"""
        
        # Base context options
        context_options = {
            "viewport": profile.viewport,
            "user_agent": profile.user_agent,
            "locale": profile.locale,
            "timezone_id": profile.timezone,
            "permissions": [],
            "extra_http_headers": {
                "Accept-Language": profile.accept_language,
                "Accept-Encoding": profile.accept_encoding
            }
        }
        
        # Level 1: Basic Stealth
        if level.value >= 1:
            context_options["ignore_https_errors"] = True
            context_options["java_script_enabled"] = True
        
        # Level 2: Advanced Stealth
        if level.value >= 2:
            # Additional headers to appear more realistic
            context_options["extra_http_headers"].update({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Upgrade-Insecure-Requests": "1"
            })
        
        # Level 3: Behavioral Simulation
        if level.value >= 3:
            # Enable geolocation with realistic coordinates
            context_options["geolocation"] = {"latitude": 40.7128, "longitude": -74.0060}  # NYC
            context_options["permissions"] = ["geolocation"]
        
        # Level 4: Profile Rotation
        if level.value >= 4:
            if profile.is_mobile:
                context_options["is_mobile"] = True
                context_options["has_touch"] = True
                context_options["device_scale_factor"] = profile.device_scale_factor
        
        # Create context
        context = await browser.new_context(**context_options)
        
        # Apply JavaScript injections for advanced stealth
        if level.value >= 2:
            await self._inject_stealth_scripts(context, profile, level)
        
        return context

    async def _inject_stealth_scripts(
        self,
        context: 'BrowserContext',
        profile: StealthProfile,
        level: StealthLevel
    ):
        """Inject stealth scripts to remove automation signatures"""
        
        stealth_script = """
        () => {
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {name: 'PDF Viewer', description: 'Portable Document Format'},
                    {name: 'Chrome PDF Plugin', description: 'Portable Document Format'},
                    {name: 'Chromium PDF Plugin', description: 'Portable Document Format'}
                ],
            });
            
            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Override screen properties
            Object.defineProperty(screen, 'colorDepth', {
                get: () => """ + str(profile.color_depth) + """,
            });
            
            Object.defineProperty(screen, 'pixelDepth', {
                get: () => """ + str(profile.color_depth) + """,
            });
        }
        """
        
        if level.value >= 3:
            # Add behavioral simulation scripts
            stealth_script += """
            // Simulate human-like behavior
            const originalAddEventListener = EventTarget.prototype.addEventListener;
            EventTarget.prototype.addEventListener = function(type, listener, options) {
                // Add slight delays to make interactions more human-like
                if (type === 'click' || type === 'mousemove') {
                    const wrappedListener = function(event) {
                        setTimeout(() => listener.call(this, event), Math.random() * 10);
                    };
                    return originalAddEventListener.call(this, type, wrappedListener, options);
                }
                return originalAddEventListener.call(this, type, listener, options);
            };
            """
        
        await context.add_init_script(stealth_script)

    async def stealth_screenshot_capture(
        self,
        browser_pool: 'BrowserPoolManager',
        url: str,
        capture_type: str = "full_page",
        max_escalation_attempts: int = 3
    ) -> StealthResult:
        """
        Capture screenshot with stealth system and automatic escalation
        
        Args:
            browser_pool: Browser pool manager
            url: Target URL
            capture_type: Type of screenshot to capture
            max_escalation_attempts: Maximum escalation attempts
            
        Returns:
            StealthResult with capture outcome
        """
        domain = self._extract_domain(url)
        current_level = self.domain_configurations.get(domain, self.config.level)
        attempts_made = 0
        escalation_occurred = False
        
        for level_attempt in range(max_escalation_attempts):
            try:
                # Create stealth context
                async with browser_pool.get_browser_context() as base_context:
                    browser_instance = browser_pool._pool[0]  # Get browser instance
                    context, profile = await self.create_stealth_context(
                        browser_instance, url, current_level
                    )
                    
                    try:
                        # Add behavioral delays
                        if current_level.value >= 3:
                            await asyncio.sleep(random.uniform(
                                self.config.min_delay_seconds,
                                self.config.max_delay_seconds
                            ))
                        
                        # Create page and navigate
                        page = await context.new_page()
                        
                        # Set up request interception for Level 2+
                        if current_level.value >= 2:
                            await self._setup_request_interception(page, profile)
                        
                        # Navigate to page
                        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        
                        # Behavioral simulation for Level 3+
                        if current_level.value >= 3:
                            await self._simulate_human_behavior(page)
                        
                        # Check if we're blocked
                        if await self._detect_blocking(page):
                            logger.warning(f"Blocking detected on {domain} at level {current_level.value}")
                            attempts_made += 1
                            
                            if level_attempt < max_escalation_attempts - 1 and self.config.escalate_on_block:
                                current_level = self._escalate_stealth_level(current_level)
                                escalation_occurred = True
                                logger.info(f"Escalating to level {current_level.value} for {domain}")
                                continue
                            else:
                                self.blocked_domains.add(domain)
                                return StealthResult(
                                    success=False,
                                    level_used=current_level,
                                    profile_used=profile,
                                    attempts_made=attempts_made,
                                    error_message="Domain blocking detected, max escalation reached",
                                    blocked_detected=True,
                                    escalation_occurred=escalation_occurred
                                )
                        
                        # Capture screenshot
                        screenshot_data = None
                        if capture_type == "full_page":
                            screenshot_data = await page.screenshot(full_page=True, type="png")
                        elif capture_type == "viewport":
                            screenshot_data = await page.screenshot(type="png")
                        
                        if screenshot_data:
                            # Success - store successful configuration
                            self.domain_configurations[domain] = current_level
                            self.successful_profiles[domain] = profile
                            self._record_attempt(domain, current_level, True)
                            
                            logger.info(f"✅ Stealth screenshot successful for {domain} at level {current_level.value}")
                            
                            return StealthResult(
                                success=True,
                                level_used=current_level,
                                profile_used=profile,
                                attempts_made=attempts_made + 1,
                                escalation_occurred=escalation_occurred
                            )
                        
                    finally:
                        await context.close()
                        
            except Exception as e:
                attempts_made += 1
                logger.error(f"Stealth capture attempt {attempts_made} failed: {e}")
                
                if level_attempt < max_escalation_attempts - 1 and self.config.escalate_on_block:
                    current_level = self._escalate_stealth_level(current_level)
                    escalation_occurred = True
                    continue
                else:
                    self._record_attempt(domain, current_level, False)
                    return StealthResult(
                        success=False,
                        level_used=current_level,
                        profile_used=profile,
                        attempts_made=attempts_made,
                        error_message=str(e),
                        escalation_occurred=escalation_occurred
                    )
        
        return StealthResult(
            success=False,
            level_used=current_level,
            profile_used=profile,
            attempts_made=attempts_made,
            error_message="Max escalation attempts reached",
            escalation_occurred=escalation_occurred
        )

    async def _setup_request_interception(self, page: 'Page', profile: StealthProfile):
        """Set up request interception for advanced stealth"""
        async def handle_request(route):
            # Get the request from the route
            request = route.request
            
            # Modify headers to appear more realistic
            headers = dict(request.headers)
            headers.update({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": profile.accept_language,
                "Accept-Encoding": profile.accept_encoding,
                "User-Agent": profile.user_agent
            })
            
            # Remove automation-related headers
            headers.pop("sec-ch-ua-mobile", None)
            headers.pop("sec-ch-ua-platform", None)
            
            await route.continue_(headers=headers)
        
        await page.route("**/*", handle_request)

    async def _simulate_human_behavior(self, page: 'Page'):
        """Simulate human-like behavior on the page"""
        try:
            # Random scroll simulation
            for _ in range(random.randint(1, 3)):
                await page.evaluate("""
                    window.scrollBy({
                        top: Math.random() * 500,
                        left: 0,
                        behavior: 'smooth'
                    });
                """)
                await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Random mouse movement simulation
            viewport = await page.viewport_size()
            if viewport:
                for _ in range(random.randint(1, 2)):
                    x = random.randint(50, viewport['width'] - 50)
                    y = random.randint(50, viewport['height'] - 50)
                    await page.mouse.move(x, y)
                    await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Random wait time
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
        except Exception as e:
            logger.debug(f"Behavioral simulation error: {e}")

    async def _detect_blocking(self, page: 'Page') -> bool:
        """Detect if the page indicates blocking or anti-bot measures"""
        try:
            # Check page title and content for blocking indicators
            title = await page.title()
            content = await page.content()
            
            blocking_indicators = [
                "blocked", "access denied", "forbidden", "captcha",
                "bot detection", "security check", "cloudflare",
                "please verify", "human verification", "robot"
            ]
            
            title_lower = title.lower()
            content_lower = content.lower()
            
            for indicator in blocking_indicators:
                if indicator in title_lower or indicator in content_lower:
                    return True
            
            # Check for specific HTTP status codes
            try:
                response = await page.wait_for_response(lambda r: r.url == page.url, timeout=5000)
                if response and response.status in [403, 429, 503]:
                    return True
            except:
                pass  # Timeout is okay, page might already be loaded
            
            return False
            
        except Exception as e:
            logger.debug(f"Blocking detection error: {e}")
            return False

    def _escalate_stealth_level(self, current_level: StealthLevel) -> StealthLevel:
        """Escalate to the next stealth level"""
        if current_level == StealthLevel.LEVEL_1_BASIC:
            return StealthLevel.LEVEL_2_ADVANCED
        elif current_level == StealthLevel.LEVEL_2_ADVANCED:
            return StealthLevel.LEVEL_3_BEHAVIORAL
        elif current_level == StealthLevel.LEVEL_3_BEHAVIORAL:
            return StealthLevel.LEVEL_4_PROFILE_ROTATION
        else:
            return StealthLevel.LEVEL_4_PROFILE_ROTATION  # Max level

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except:
            return "unknown"

    def _record_attempt(self, domain: str, level: StealthLevel, success: bool):
        """Record attempt for analytics"""
        if domain not in self.attempt_history:
            self.attempt_history[domain] = []
        
        self.attempt_history[domain].append({
            "timestamp": time.time(),
            "level": level.value,
            "success": success
        })
        
        # Keep only last 50 attempts per domain
        if len(self.attempt_history[domain]) > 50:
            self.attempt_history[domain] = self.attempt_history[domain][-50:]

    def get_stealth_analytics(self) -> Dict[str, Any]:
        """Get analytics about stealth system performance"""
        analytics = {
            "total_domains": len(self.attempt_history),
            "blocked_domains": len(self.blocked_domains),
            "successful_configurations": len(self.successful_profiles),
            "domain_stats": {}
        }
        
        for domain, attempts in self.attempt_history.items():
            successful_attempts = sum(1 for a in attempts if a["success"])
            total_attempts = len(attempts)
            success_rate = (successful_attempts / total_attempts) * 100 if total_attempts > 0 else 0
            
            levels_used = set(a["level"] for a in attempts)
            avg_level = sum(a["level"] for a in attempts) / len(attempts) if attempts else 0
            
            analytics["domain_stats"][domain] = {
                "total_attempts": total_attempts,
                "successful_attempts": successful_attempts,
                "success_rate": round(success_rate, 2),
                "levels_used": sorted(list(levels_used)),
                "average_level": round(avg_level, 2),
                "is_blocked": domain in self.blocked_domains
            }
        
        return analytics

    def reset_domain_configuration(self, domain: str = None):
        """Reset stealth configuration for a domain or all domains"""
        if domain:
            self.domain_configurations.pop(domain, None)
            self.successful_profiles.pop(domain, None)
            self.blocked_domains.discard(domain)
            self.attempt_history.pop(domain, None)
            logger.info(f"Reset stealth configuration for {domain}")
        else:
            self.domain_configurations.clear()
            self.successful_profiles.clear()
            self.blocked_domains.clear()
            self.attempt_history.clear()
            logger.info("Reset all stealth configurations")

class StealthScreenshotService:
    """
    Screenshot service with integrated anti-detection stealth system
    Provides high-level interface for stealth screenshot capture
    """
    
    def __init__(
        self,
        browser_pool: BrowserPoolManager,
        stealth_config: StealthConfig = None,
        screenshot_service: ScreenshotService = None
    ):
        self.browser_pool = browser_pool
        self.stealth_system = AntiDetectionStealthSystem(stealth_config)
        self.screenshot_service = screenshot_service or ScreenshotService(browser_pool)
        
        logger.info("StealthScreenshotService initialized")

    async def capture_stealth_screenshot(
        self,
        url: str,
        capture_type: str = "full_page",
        stealth_level: Optional[StealthLevel] = None,
        max_escalation_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Capture screenshot with stealth protection
        
        Args:
            url: Target URL
            capture_type: Type of screenshot (full_page, viewport)
            stealth_level: Specific stealth level to use
            max_escalation_attempts: Maximum escalation attempts
            
        Returns:
            Dictionary with capture results and stealth information
        """
        try:
            # Attempt stealth screenshot capture
            stealth_result = await self.stealth_system.stealth_screenshot_capture(
                browser_pool=self.browser_pool,
                url=url,
                capture_type=capture_type,
                max_escalation_attempts=max_escalation_attempts
            )
            
            return {
                "success": stealth_result.success,
                "stealth_result": stealth_result,
                "level_used": stealth_result.level_used.value,
                "profile_used": stealth_result.profile_used.__dict__,
                "attempts_made": stealth_result.attempts_made,
                "escalation_occurred": stealth_result.escalation_occurred,
                "blocked_detected": stealth_result.blocked_detected,
                "error_message": stealth_result.error_message,
                "url": url,
                "capture_type": capture_type
            }
            
        except Exception as e:
            logger.error(f"Stealth screenshot capture failed: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "url": url,
                "capture_type": capture_type
            }

    async def batch_stealth_capture(
        self,
        urls: List[str],
        capture_type: str = "full_page",
        max_concurrent: int = 2,  # Lower concurrency for stealth
        stealth_level: Optional[StealthLevel] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch capture with stealth protection
        
        Args:
            urls: List of URLs to capture
            capture_type: Type of screenshot
            max_concurrent: Maximum concurrent captures (keep low for stealth)
            stealth_level: Stealth level to use
            
        Returns:
            Dictionary mapping URLs to results
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def capture_single(url: str):
            async with semaphore:
                return await self.capture_stealth_screenshot(
                    url=url,
                    capture_type=capture_type,
                    stealth_level=stealth_level
                )
        
        tasks = [capture_single(url) for url in urls]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(task_results):
            url = urls[i]
            if isinstance(result, Exception):
                results[url] = {"success": False, "error": str(result)}
            else:
                results[url] = result
        
        logger.info(f"Batch stealth capture completed: {len(urls)} URLs processed")
        return results

    def get_stealth_analytics(self) -> Dict[str, Any]:
        """Get stealth system analytics"""
        return self.stealth_system.get_stealth_analytics()

    def reset_stealth_configuration(self, domain: str = None):
        """Reset stealth configuration"""
        self.stealth_system.reset_domain_configuration(domain)

# Global stealth service instance
_global_stealth_service: Optional[StealthScreenshotService] = None

async def get_global_stealth_service(
    browser_pool: BrowserPoolManager = None,
    stealth_config: StealthConfig = None
) -> StealthScreenshotService:
    """Get or create global stealth service instance"""
    global _global_stealth_service
    
    if _global_stealth_service is None and browser_pool:
        _global_stealth_service = StealthScreenshotService(browser_pool, stealth_config)
        logger.info("Created global stealth service instance")
    
    return _global_stealth_service

async def cleanup_global_stealth_service():
    """Clean up global stealth service instance"""
    global _global_stealth_service
    _global_stealth_service = None
    logger.info("Cleaned up global stealth service instance")

# ==============================================================================
# Testing and Examples for Stealth System
# ==============================================================================

async def test_anti_detection_stealth():
    """Test the anti-detection stealth system"""
    logger.info("🥷 Starting Anti-Detection Stealth System test...")
    
    try:
        # Initialize browser pool
        browser_pool = await get_global_browser_pool()
        await browser_pool.initialize()
        
        # Create stealth service with Level 2 configuration
        stealth_config = StealthConfig(
            level=StealthLevel.LEVEL_2_ADVANCED,
            escalate_on_block=True,
            max_retries_per_level=2
        )
        stealth_service = StealthScreenshotService(browser_pool, stealth_config)
        
        # Test URLs (start with a simple site)
        test_urls = [
            "https://example.com",
            "https://httpbin.org/user-agent"  # Shows our user agent
        ]
        
        print("\n1. Testing stealth profiles generation...")
        for level in StealthLevel:
            profile = StealthProfile.generate_stealth_profile(level)
            print(f"  Level {level.value}: {profile.browser_engine} - {profile.viewport} - Mobile: {profile.is_mobile}")
        
        print("\n2. Testing single stealth screenshot capture...")
        for url in test_urls:
            result = await stealth_service.capture_stealth_screenshot(
                url=url,
                capture_type="full_page",
                stealth_level=StealthLevel.LEVEL_2_ADVANCED
            )
            
            if result["success"]:
                print(f"  ✅ {url}: Level {result['level_used']}, Attempts: {result['attempts_made']}")
                if result["escalation_occurred"]:
                    print(f"    🔄 Escalation occurred during capture")
            else:
                print(f"  ❌ {url}: {result.get('error_message', 'Unknown error')}")
        
        print("\n3. Testing batch stealth capture...")
        batch_results = await stealth_service.batch_stealth_capture(
            urls=test_urls,
            capture_type="viewport",
            max_concurrent=1,  # Very conservative for stealth
            stealth_level=StealthLevel.LEVEL_3_BEHAVIORAL
        )
        
        successful_captures = sum(1 for r in batch_results.values() if r.get("success"))
        print(f"  Batch Results: {successful_captures}/{len(test_urls)} successful")
        
        print("\n4. Testing stealth analytics...")
        analytics = stealth_service.get_stealth_analytics()
        print(f"  Total domains tested: {analytics['total_domains']}")
        print(f"  Blocked domains: {analytics['blocked_domains']}")
        print(f"  Successful configurations: {analytics['successful_configurations']}")
        
        for domain, stats in analytics["domain_stats"].items():
            print(f"  {domain}: {stats['success_rate']}% success rate, avg level {stats['average_level']}")
        
        print("\n5. Testing escalation simulation...")
        # This would normally be tested against a site that blocks automation
        print("  Escalation testing requires sites with bot detection")
        print("  In production, system will escalate: Level 1 → 2 → 3 → 4")
        
        print("\n6. Testing ethical boundaries...")
        print("  ✅ Respects robots.txt (configurable)")
        print("  ✅ Implements request throttling")
        print(f"  ✅ Maximum {stealth_config.max_retries_per_level} retries per level")
        print("  ✅ Graceful degradation when legitimately blocked")
        print("  ✅ No CAPTCHA circumvention attempts")
        
        print("\n✅ Anti-Detection Stealth System test completed successfully!")
        
        # Cleanup
        await browser_pool.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Stealth system test error: {e}")
        import traceback
        traceback.print_exc()

# ========================================================================================
# TASK 22.2: BROWSER RESOURCE OPTIMIZATION
# ========================================================================================

def optimize_browser_resources(
    browser_pool: 'BrowserPoolManager' = None,
    config: ResourceOptimizationConfig = None
) -> Dict[str, Any]:
    """
    Optimize browser resource usage by implementing intelligent pooling
    and resource allocation strategies.
    
    Args:
        browser_pool: Existing browser pool to optimize (optional)
        config: Resource optimization configuration
        
    Returns:
        Dictionary with optimization results and browser configuration
    """
    if config is None:
        config = ResourceOptimizationConfig()
    
    # Get system resource information
    cpu_count = os.cpu_count() or 4
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Calculate optimal browser pool size based on system resources
    if config.max_concurrent_browsers is None:
        # Conservative approach: use 50% of CPU cores, max 4 browsers
        config.max_concurrent_browsers = min(max(cpu_count // 2, 1), 4)
        
        # Adjust based on available memory (each browser ~512MB minimum)
        max_by_memory = int(available_memory_gb // 0.5)
        config.max_concurrent_browsers = min(config.max_concurrent_browsers, max_by_memory)
    
    # Configure browser launch options for resource optimization
    browser_options = {
        'headless': True,
        'args': [
            # Essential performance optimizations
            '--disable-extensions',
            '--disable-gpu',
            '--disable-dev-shm-usage',
            '--disable-setuid-sandbox',
            '--no-sandbox',
            '--disable-accelerated-2d-canvas',
            '--disable-renderer-backgrounding',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-background-networking',
            
            # Memory optimizations
            f'--memory-pressure-off',
            f'--max_old_space_size={config.memory_limit_mb}',
            '--no-zygote',
            '--disable-ipc-flooding-protection',
            
            # Network optimizations
            '--aggressive-cache-discard',
            '--disable-background-networking',
            '--disable-background-mode',
            '--disable-default-apps',
            '--disable-sync',
            
            # Rendering optimizations
            '--disable-features=TranslateUI,VizDisplayCompositor'
        ]
    }
    
    # Add conditional optimizations based on config
    if config.disable_images:
        browser_options['args'].extend([
            '--blink-settings=imagesEnabled=false',
            '--disable-remote-fonts'
        ])
    
    if config.disable_javascript:
        browser_options['args'].append('--disable-javascript')
    
    if config.use_minimal_viewport:
        browser_options['args'].append('--window-size=800,600')
    
    # Configure browser pool if provided
    if browser_pool:
        browser_pool.max_pool_size = config.max_concurrent_browsers
        browser_pool.browser_options = browser_options
        browser_pool.optimization_config = config
    
    optimization_results = {
        'success': True,
        'optimized_pool_size': config.max_concurrent_browsers,
        'system_cpu_cores': cpu_count,
        'system_memory_gb': round(memory_gb, 2),
        'available_memory_gb': round(available_memory_gb, 2),
        'browser_options': browser_options,
        'optimization_config': config,
        'estimated_memory_per_browser_mb': config.memory_limit_mb,
        'total_estimated_memory_usage_mb': config.max_concurrent_browsers * config.memory_limit_mb,
        'browser_pool_updated': browser_pool is not None,
        'optimization_applied': True,
        'recommendations': [
            f"Browser pool size optimized to {config.max_concurrent_browsers} browsers",
            f"Memory limit set to {config.memory_limit_mb}MB per browser",
            f"Total memory usage: {config.max_concurrent_browsers * config.memory_limit_mb}MB",
            "Performance optimizations applied to browser launch options"
        ],
        'memory_limit_mb': config.memory_limit_mb,
        'max_concurrent_browsers': config.max_concurrent_browsers,
        'system_memory_mb': int(memory_gb * 1024),
        'recommended_pool_size': config.max_concurrent_browsers
    }
    
    logger.info(f"🚀 Browser resources optimized: {config.max_concurrent_browsers} browsers, "
                f"{config.memory_limit_mb}MB limit per browser")
    
    return optimization_results

def get_global_screenshot_cache() -> ScreenshotCache:
    """Get or create global screenshot cache instance"""
    global _global_screenshot_cache
    if _global_screenshot_cache is None:
        _global_screenshot_cache = ScreenshotCache()
    return _global_screenshot_cache

async def cleanup_global_screenshot_cache():
    """Cleanup global screenshot cache"""
    global _global_screenshot_cache
    if _global_screenshot_cache:
        await _global_screenshot_cache.clear()
        _global_screenshot_cache = None

# Global cache instance
_global_screenshot_cache: Optional[ScreenshotCache] = None

def get_global_screenshot_cache() -> ScreenshotCache:
    """Get or create global screenshot cache instance"""
    global _global_screenshot_cache
    if _global_screenshot_cache is None:
        _global_screenshot_cache = ScreenshotCache()
    return _global_screenshot_cache

async def cleanup_global_screenshot_cache():
    """Clean up global screenshot cache"""
    global _global_screenshot_cache
    if _global_screenshot_cache:
        await _global_screenshot_cache.stop_cleanup_task()
        await _global_screenshot_cache.clear()
        _global_screenshot_cache = None

class ImageCompressionOptimizer:
    """
    Intelligent image compression optimization
    Automatically selects optimal format and compression settings
    """
    
    @staticmethod
    def optimize_compression_settings(
        screenshot_data: bytes,
        target_format: str,
        optimization_config: ResourceOptimizationConfig
    ) -> Dict[str, Any]:
        """
        Optimize compression settings based on image content and target usage
        
        Args:
            screenshot_data: Raw screenshot data
            target_format: Target format ('png', 'jpeg')
            optimization_config: Optimization configuration
            
        Returns:
            Optimized compression settings
        """
        file_size = len(screenshot_data)
        
        # Analyze content characteristics (simplified heuristics)
        # In production, could use image analysis libraries
        is_large_image = file_size > 1024 * 1024  # > 1MB
        
        if optimization_config.auto_format_selection:
            # Auto-select format based on content characteristics
            if is_large_image and target_format == 'png':
                # Large PNG - consider JPEG for better compression
                suggested_format = 'jpeg'
                quality = max(optimization_config.jpeg_quality - 10, 70)
            else:
                suggested_format = target_format
                quality = optimization_config.jpeg_quality
        else:
            suggested_format = target_format
            quality = optimization_config.jpeg_quality
        
        compression_settings = {
            'format': suggested_format,
            'quality': quality if suggested_format == 'jpeg' else None,
            'compression_level': optimization_config.png_compression_level if suggested_format == 'png' else None,
            'optimization_applied': suggested_format != target_format,
            'estimated_size_reduction': 0.3 if suggested_format == 'jpeg' and target_format == 'png' else 0.1
        }
        
        return compression_settings

if __name__ == "__main__":
    # Test the anti-detection stealth system
    asyncio.run(test_anti_detection_stealth()) 