"""
Cache Exceptions Module.

This module defines custom exception classes for the cache orchestration system.
It provides specific exception types for different cache-related error conditions
to enable proper error handling and debugging throughout the caching infrastructure.

Exception categories include:
- Connection-related exceptions for Redis client issues
- Configuration exceptions for invalid cache settings
- Operation exceptions for cache operation failures
- Orchestration exceptions for multi-cache coordination issues
- Analytics exceptions for monitoring and reporting failures

All exceptions extend from a base CacheException class to provide consistent
error handling patterns across the cache system.
"""

from typing import Optional, Any


class CacheException(Exception):
    """Base exception class for all cache-related errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CacheConnectionException(CacheException):
    """Exception raised for Redis connection-related issues."""
    
    def __init__(self, message: str, redis_url: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.redis_url = redis_url


class CacheConfigurationException(CacheException):
    """Exception raised for invalid cache configuration settings."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.config_key = config_key


class CacheOperationException(CacheException):
    """Exception raised for cache operation failures."""
    
    def __init__(self, message: str, operation: Optional[str] = None, cache_type: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.operation = operation
        self.cache_type = cache_type


class CacheOrchestrationException(CacheException):
    """Exception raised for multi-cache coordination issues."""
    
    def __init__(self, message: str, content_type: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.content_type = content_type


class CacheAnalyticsException(CacheException):
    """Exception raised for monitoring and reporting failures."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.metric_name = metric_name


class CacheKeyGenerationException(CacheException):
    """Exception raised for semantic key generation failures."""
    
    def __init__(self, message: str, query: Optional[str] = None, content_type: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.query = query
        self.content_type = content_type


class CacheTTLException(CacheException):
    """Exception raised for TTL strategy failures."""
    
    def __init__(self, message: str, ttl_value: Optional[int] = None, content_type: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.ttl_value = ttl_value
        self.content_type = content_type 