"""
Rate Limiter for Universal RAG CMS Security System

This module provides comprehensive rate limiting functionality including
token bucket, sliding window, and fixed window algorithms with support
for user-based, IP-based, and API key-based rate limiting.

Features:
- Multiple rate limiting algorithms (token bucket, sliding window, fixed window)
- User-based, IP-based, and API key-based limits
- Hierarchical rate limiting with role-based limits
- Distributed rate limiting support
- Rate limit bypass for emergency scenarios
- Comprehensive metrics and monitoring
- Integration with security violation tracking
- Configurable rate limit policies
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import hashlib

from ..models import UserRole, AuditAction, SecurityViolation
from .audit_logger import AuditLogger

# Configure logging
logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Rate limit scope types"""
    USER = "user"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


class RateLimitStatus(Enum):
    """Rate limit check results"""
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    BLOCKED = "blocked"
    BYPASS = "bypass"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    scope: RateLimitScope
    algorithm: RateLimitAlgorithm
    requests_per_window: int
    window_size_seconds: int
    burst_capacity: Optional[int] = None  # For token bucket
    refill_rate: Optional[float] = None   # For token bucket
    user_role: Optional[UserRole] = None
    endpoint_pattern: Optional[str] = None
    enabled: bool = True
    bypass_on_error: bool = False
    
    # Violation thresholds
    warning_threshold: float = 0.8  # Warn at 80% of limit
    block_threshold: float = 1.2    # Block at 120% of limit
    block_duration_seconds: int = 300  # 5 minutes


@dataclass
class RateLimitBucket:
    """Rate limit bucket state"""
    identifier: str
    config: RateLimitConfig
    current_count: int = 0
    last_reset: datetime = field(default_factory=datetime.utcnow)
    last_request: datetime = field(default_factory=datetime.utcnow)
    tokens: float = 0.0  # For token bucket
    request_timestamps: deque = field(default_factory=deque)  # For sliding window
    is_blocked: bool = False
    block_until: Optional[datetime] = None
    violation_count: int = 0


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    status: RateLimitStatus
    identifier: str
    scope: RateLimitScope
    requests_remaining: int
    reset_time: datetime
    retry_after_seconds: Optional[int] = None
    violation_logged: bool = False
    
    # Detailed information
    current_usage: int = 0
    limit: int = 0
    window_size: int = 0
    algorithm: Optional[RateLimitAlgorithm] = None
    
    # Headers for HTTP responses
    headers: Dict[str, str] = field(default_factory=dict)


class RateLimiter:
    """
    Comprehensive rate limiting system with multiple algorithms and scopes.
    
    Provides flexible rate limiting for API endpoints, user actions, and 
    system resources with support for role-based limits and security integration.
    """
    
    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Rate Limiter"""
        
        self.audit_logger = audit_logger
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Rate limit buckets storage
        self.buckets: Dict[str, RateLimitBucket] = {}
        
        # Default rate limit configurations
        self.default_configs = self._initialize_default_configs()
        
        # Custom configurations
        self.custom_configs: Dict[str, RateLimitConfig] = {}
        
        # Global settings
        self.global_rate_limiting_enabled = self.config.get('enabled', True)
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutes
        self.max_buckets = self.config.get('max_buckets', 10000)
        
        # Emergency bypass settings
        self.emergency_bypass_enabled = self.config.get('emergency_bypass', False)
        self.bypass_tokens = set(self.config.get('bypass_tokens', []))
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'allowed_requests': 0,
            'rate_limited_requests': 0,
            'blocked_requests': 0,
            'bypass_requests': 0,
            'violations_logged': 0,
            'active_buckets': 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        if self.cleanup_interval > 0:
            self._start_cleanup_task()
        
        self.logger.info("Rate Limiter initialized")
    
    def _initialize_default_configs(self) -> Dict[str, RateLimitConfig]:
        """Initialize default rate limit configurations"""
        
        return {
            # User-based limits by role
            f"user_{UserRole.SUPER_ADMIN.value}": RateLimitConfig(
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=10000,
                window_size_seconds=3600,  # 1 hour
                burst_capacity=1000,
                refill_rate=2.77,  # ~10000/hour
                user_role=UserRole.SUPER_ADMIN
            ),
            
            f"user_{UserRole.ADMIN.value}": RateLimitConfig(
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=5000,
                window_size_seconds=3600,
                burst_capacity=500,
                refill_rate=1.39,  # ~5000/hour
                user_role=UserRole.ADMIN
            ),
            
            f"user_{UserRole.CONTENT_MANAGER.value}": RateLimitConfig(
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=2000,
                window_size_seconds=3600,
                user_role=UserRole.CONTENT_MANAGER
            ),
            
            f"user_{UserRole.API_USER.value}": RateLimitConfig(
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=1000,
                window_size_seconds=3600,
                user_role=UserRole.API_USER
            ),
            
            f"user_{UserRole.READ_ONLY.value}": RateLimitConfig(
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=500,
                window_size_seconds=3600,
                user_role=UserRole.READ_ONLY
            ),
            
            f"user_{UserRole.ANONYMOUS.value}": RateLimitConfig(
                scope=RateLimitScope.USER,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=100,
                window_size_seconds=3600,
                user_role=UserRole.ANONYMOUS
            ),
            
            # IP-based limits
            "ip_default": RateLimitConfig(
                scope=RateLimitScope.IP_ADDRESS,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=1000,
                window_size_seconds=3600,
                warning_threshold=0.8,
                block_threshold=1.5,
                block_duration_seconds=600  # 10 minutes
            ),
            
            # API key-based limits
            "api_key_default": RateLimitConfig(
                scope=RateLimitScope.API_KEY,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=5000,
                window_size_seconds=3600,
                burst_capacity=100,
                refill_rate=1.39
            ),
            
            # Endpoint-specific limits
            "endpoint_search": RateLimitConfig(
                scope=RateLimitScope.ENDPOINT,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                requests_per_window=500,
                window_size_seconds=3600,
                endpoint_pattern="/api/search"
            ),
            
            "endpoint_upload": RateLimitConfig(
                scope=RateLimitScope.ENDPOINT,
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                requests_per_window=50,
                window_size_seconds=3600,
                endpoint_pattern="/api/upload"
            ),
            
            # Global limits
            "global_default": RateLimitConfig(
                scope=RateLimitScope.GLOBAL,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                requests_per_window=100000,
                window_size_seconds=3600,
                burst_capacity=5000,
                refill_rate=27.77
            )
        }
    
    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope,
        user_role: Optional[UserRole] = None,
        endpoint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier (user_id, IP, API key, etc.)
            scope: Rate limit scope type
            user_role: User role for role-based limits
            endpoint: Endpoint for endpoint-specific limits
            context: Additional context for rate limiting
            
        Returns:
            RateLimitResult with detailed information
        """
        
        try:
            # Check if rate limiting is globally disabled
            if not self.global_rate_limiting_enabled:
                return self._create_bypass_result(identifier, scope)
            
            # Check for emergency bypass
            if self._check_emergency_bypass(identifier, context):
                self.metrics['bypass_requests'] += 1
                return self._create_bypass_result(identifier, scope)
            
            # Get appropriate configuration
            config = self._get_rate_limit_config(scope, user_role, endpoint)
            if not config or not config.enabled:
                return self._create_bypass_result(identifier, scope)
            
            # Get or create bucket
            bucket_key = self._generate_bucket_key(identifier, scope, config)
            bucket = await self._get_or_create_bucket(bucket_key, identifier, config)
            
            # Check if currently blocked
            if bucket.is_blocked and bucket.block_until:
                if datetime.utcnow() < bucket.block_until:
                    self.metrics['blocked_requests'] += 1
                    return self._create_blocked_result(bucket, config)
                else:
                    # Unblock
                    bucket.is_blocked = False
                    bucket.block_until = None
            
            # Perform rate limit check based on algorithm
            result = await self._check_algorithm_specific_limit(bucket, config)
            
            # Update metrics
            self.metrics['total_requests'] += 1
            if result.status == RateLimitStatus.ALLOWED:
                self.metrics['allowed_requests'] += 1
            elif result.status == RateLimitStatus.RATE_LIMITED:
                self.metrics['rate_limited_requests'] += 1
            elif result.status == RateLimitStatus.BLOCKED:
                self.metrics['blocked_requests'] += 1
            
            # Log violations if necessary
            if result.status in [RateLimitStatus.RATE_LIMITED, RateLimitStatus.BLOCKED]:
                await self._log_rate_limit_violation(bucket, config, result, context)
                result.violation_logged = True
                self.metrics['violations_logged'] += 1
            
            # Generate HTTP headers
            result.headers = self._generate_rate_limit_headers(result, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            
            # Return bypass on error if configured
            if self.config.get('bypass_on_error', True):
                return self._create_bypass_result(identifier, scope)
            else:
                raise SecurityViolation(f"Rate limiting failed: {str(e)}")
    
    async def _check_algorithm_specific_limit(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Check rate limit using specific algorithm"""
        
        current_time = datetime.utcnow()
        
        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._check_token_bucket(bucket, config, current_time)
        elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._check_sliding_window(bucket, config, current_time)
        elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._check_fixed_window(bucket, config, current_time)
        elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return await self._check_leaky_bucket(bucket, config, current_time)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
    
    async def _check_token_bucket(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig,
        current_time: datetime
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm"""
        
        # Calculate tokens to add since last request
        time_diff = (current_time - bucket.last_request).total_seconds()
        tokens_to_add = time_diff * (config.refill_rate or 1.0)
        
        # Add tokens up to burst capacity
        bucket.tokens = min(
            config.burst_capacity or config.requests_per_window,
            bucket.tokens + tokens_to_add
        )
        bucket.last_request = current_time
        
        # Check if we have tokens available
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            bucket.current_count += 1
            
            return RateLimitResult(
                status=RateLimitStatus.ALLOWED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=int(bucket.tokens),
                reset_time=current_time + timedelta(seconds=config.window_size_seconds),
                current_usage=bucket.current_count,
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
        else:
            # Rate limited
            bucket.violation_count += 1
            
            return RateLimitResult(
                status=RateLimitStatus.RATE_LIMITED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=0,
                reset_time=current_time + timedelta(
                    seconds=int((1.0 - bucket.tokens) / (config.refill_rate or 1.0))
                ),
                retry_after_seconds=int((1.0 - bucket.tokens) / (config.refill_rate or 1.0)),
                current_usage=bucket.current_count,
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
    
    async def _check_sliding_window(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig,
        current_time: datetime
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm"""
        
        # Remove old requests outside the window
        window_start = current_time - timedelta(seconds=config.window_size_seconds)
        
        # Clean old timestamps
        while bucket.request_timestamps and bucket.request_timestamps[0] < window_start:
            bucket.request_timestamps.popleft()
        
        current_count = len(bucket.request_timestamps)
        
        if current_count < config.requests_per_window:
            # Allow request
            bucket.request_timestamps.append(current_time)
            bucket.current_count = current_count + 1
            
            return RateLimitResult(
                status=RateLimitStatus.ALLOWED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=config.requests_per_window - current_count - 1,
                reset_time=bucket.request_timestamps[0] + timedelta(seconds=config.window_size_seconds),
                current_usage=current_count + 1,
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
        else:
            # Rate limited
            bucket.violation_count += 1
            
            # Calculate when the oldest request will expire
            oldest_request = bucket.request_timestamps[0]
            reset_time = oldest_request + timedelta(seconds=config.window_size_seconds)
            retry_after = int((reset_time - current_time).total_seconds())
            
            return RateLimitResult(
                status=RateLimitStatus.RATE_LIMITED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=0,
                reset_time=reset_time,
                retry_after_seconds=max(1, retry_after),
                current_usage=current_count,
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
    
    async def _check_fixed_window(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig,
        current_time: datetime
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm"""
        
        # Calculate current window
        window_start_timestamp = int(current_time.timestamp()) // config.window_size_seconds
        bucket_window_start = int(bucket.last_reset.timestamp()) // config.window_size_seconds
        
        # Reset if we're in a new window
        if window_start_timestamp > bucket_window_start:
            bucket.current_count = 0
            bucket.last_reset = current_time
            bucket.violation_count = 0  # Reset violations on new window
        
        if bucket.current_count < config.requests_per_window:
            # Allow request
            bucket.current_count += 1
            
            # Calculate reset time (start of next window)
            next_window_start = (window_start_timestamp + 1) * config.window_size_seconds
            reset_time = datetime.fromtimestamp(next_window_start)
            
            return RateLimitResult(
                status=RateLimitStatus.ALLOWED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=config.requests_per_window - bucket.current_count,
                reset_time=reset_time,
                current_usage=bucket.current_count,
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
        else:
            # Rate limited
            bucket.violation_count += 1
            
            # Calculate reset time
            next_window_start = (window_start_timestamp + 1) * config.window_size_seconds
            reset_time = datetime.fromtimestamp(next_window_start)
            retry_after = int((reset_time - current_time).total_seconds())
            
            return RateLimitResult(
                status=RateLimitStatus.RATE_LIMITED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=0,
                reset_time=reset_time,
                retry_after_seconds=max(1, retry_after),
                current_usage=bucket.current_count,
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
    
    async def _check_leaky_bucket(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig,
        current_time: datetime
    ) -> RateLimitResult:
        """Check rate limit using leaky bucket algorithm"""
        
        # Calculate leaked requests since last check
        time_diff = (current_time - bucket.last_request).total_seconds()
        leak_rate = config.requests_per_window / config.window_size_seconds
        leaked_requests = time_diff * leak_rate
        
        # Apply leakage
        bucket.current_count = max(0, bucket.current_count - leaked_requests)
        bucket.last_request = current_time
        
        # Check if bucket is full
        if bucket.current_count < config.requests_per_window:
            # Allow request
            bucket.current_count += 1
            
            return RateLimitResult(
                status=RateLimitStatus.ALLOWED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=int(config.requests_per_window - bucket.current_count),
                reset_time=current_time + timedelta(
                    seconds=int((bucket.current_count - 1) / leak_rate)
                ),
                current_usage=int(bucket.current_count),
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
        else:
            # Rate limited
            bucket.violation_count += 1
            
            # Calculate time to wait
            overflow = bucket.current_count - config.requests_per_window + 1
            wait_time = overflow / leak_rate
            
            return RateLimitResult(
                status=RateLimitStatus.RATE_LIMITED,
                identifier=bucket.identifier,
                scope=config.scope,
                requests_remaining=0,
                reset_time=current_time + timedelta(seconds=int(wait_time)),
                retry_after_seconds=max(1, int(wait_time)),
                current_usage=int(bucket.current_count),
                limit=config.requests_per_window,
                window_size=config.window_size_seconds,
                algorithm=config.algorithm
            )
    
    def _get_rate_limit_config(
        self,
        scope: RateLimitScope,
        user_role: Optional[UserRole] = None,
        endpoint: Optional[str] = None
    ) -> Optional[RateLimitConfig]:
        """Get appropriate rate limit configuration"""
        
        # Check custom configurations first
        if endpoint and f"endpoint_{endpoint}" in self.custom_configs:
            return self.custom_configs[f"endpoint_{endpoint}"]
        
        if user_role and f"user_{user_role.value}" in self.custom_configs:
            return self.custom_configs[f"user_{user_role.value}"]
        
        # Check default configurations
        if scope == RateLimitScope.USER and user_role:
            config_key = f"user_{user_role.value}"
            if config_key in self.default_configs:
                return self.default_configs[config_key]
        
        if scope == RateLimitScope.ENDPOINT and endpoint:
            # Check for endpoint-specific config
            for key, config in self.default_configs.items():
                if (key.startswith("endpoint_") and 
                    config.endpoint_pattern and 
                    endpoint.startswith(config.endpoint_pattern)):
                    return config
        
        # Default scope configurations
        default_keys = {
            RateLimitScope.IP_ADDRESS: "ip_default",
            RateLimitScope.API_KEY: "api_key_default",
            RateLimitScope.GLOBAL: "global_default"
        }
        
        if scope in default_keys:
            return self.default_configs.get(default_keys[scope])
        
        return None
    
    async def _get_or_create_bucket(
        self,
        bucket_key: str,
        identifier: str,
        config: RateLimitConfig
    ) -> RateLimitBucket:
        """Get existing bucket or create new one"""
        
        if bucket_key not in self.buckets:
            # Check bucket limit
            if len(self.buckets) >= self.max_buckets:
                await self._cleanup_old_buckets()
            
            # Create new bucket
            bucket = RateLimitBucket(
                identifier=identifier,
                config=config,
                tokens=config.burst_capacity or config.requests_per_window
            )
            self.buckets[bucket_key] = bucket
        
        return self.buckets[bucket_key]
    
    def _generate_bucket_key(
        self,
        identifier: str,
        scope: RateLimitScope,
        config: RateLimitConfig
    ) -> str:
        """Generate unique bucket key"""
        
        key_parts = [scope.value, identifier]
        
        if config.user_role:
            key_parts.append(config.user_role.value)
        
        if config.endpoint_pattern:
            key_parts.append(config.endpoint_pattern)
        
        return ":".join(key_parts)
    
    def _check_emergency_bypass(
        self,
        identifier: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if request should bypass rate limiting"""
        
        if not self.emergency_bypass_enabled:
            return False
        
        # Check bypass tokens
        if context and context.get('bypass_token') in self.bypass_tokens:
            return True
        
        # Check emergency scenarios
        if context and context.get('emergency'):
            return True
        
        return False
    
    def _create_bypass_result(
        self,
        identifier: str,
        scope: RateLimitScope
    ) -> RateLimitResult:
        """Create bypass result"""
        
        return RateLimitResult(
            status=RateLimitStatus.BYPASS,
            identifier=identifier,
            scope=scope,
            requests_remaining=999999,
            reset_time=datetime.utcnow() + timedelta(hours=1),
            current_usage=0,
            limit=999999,
            window_size=3600
        )
    
    def _create_blocked_result(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Create blocked result"""
        
        retry_after = int((bucket.block_until - datetime.utcnow()).total_seconds()) if bucket.block_until else 0
        
        return RateLimitResult(
            status=RateLimitStatus.BLOCKED,
            identifier=bucket.identifier,
            scope=config.scope,
            requests_remaining=0,
            reset_time=bucket.block_until or datetime.utcnow(),
            retry_after_seconds=max(1, retry_after),
            current_usage=bucket.current_count,
            limit=config.requests_per_window,
            window_size=config.window_size_seconds,
            algorithm=config.algorithm
        )
    
    async def _log_rate_limit_violation(
        self,
        bucket: RateLimitBucket,
        config: RateLimitConfig,
        result: RateLimitResult,
        context: Optional[Dict[str, Any]]
    ):
        """Log rate limit violation"""
        
        if not self.audit_logger:
            return
        
        try:
            await self.audit_logger.log_event(
                action=AuditAction.RATE_LIMIT_EXCEEDED,
                user_id=context.get('user_id') if context else None,
                resource_type="rate_limit",
                resource_id=bucket.identifier,
                details={
                    'scope': config.scope.value,
                    'algorithm': config.algorithm.value,
                    'limit': config.requests_per_window,
                    'window_size': config.window_size_seconds,
                    'current_usage': result.current_usage,
                    'violation_count': bucket.violation_count,
                    'status': result.status.value,
                    'retry_after': result.retry_after_seconds,
                    'is_blocked': bucket.is_blocked
                },
                ip_address=context.get('ip_address') if context else None,
                user_agent=context.get('user_agent') if context else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log rate limit violation: {e}")
    
    def _generate_rate_limit_headers(
        self,
        result: RateLimitResult,
        config: RateLimitConfig
    ) -> Dict[str, str]:
        """Generate HTTP headers for rate limit response"""
        
        headers = {
            'X-RateLimit-Limit': str(config.requests_per_window),
            'X-RateLimit-Window': str(config.window_size_seconds),
            'X-RateLimit-Remaining': str(result.requests_remaining),
            'X-RateLimit-Reset': str(int(result.reset_time.timestamp())),
            'X-RateLimit-Algorithm': config.algorithm.value,
            'X-RateLimit-Scope': config.scope.value
        }
        
        if result.retry_after_seconds:
            headers['Retry-After'] = str(result.retry_after_seconds)
        
        if result.status != RateLimitStatus.ALLOWED:
            headers['X-RateLimit-Status'] = result.status.value
        
        return headers
    
    async def _cleanup_old_buckets(self):
        """Clean up old unused buckets"""
        
        current_time = datetime.utcnow()
        cleanup_threshold = current_time - timedelta(hours=1)
        
        keys_to_remove = []
        for key, bucket in self.buckets.items():
            if bucket.last_request < cleanup_threshold and not bucket.is_blocked:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.buckets[key]
        
        self.metrics['active_buckets'] = len(self.buckets)
        
        if keys_to_remove:
            self.logger.debug(f"Cleaned up {len(keys_to_remove)} old rate limit buckets")
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task"""
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_old_buckets()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def add_custom_config(self, name: str, config: RateLimitConfig):
        """Add custom rate limit configuration"""
        
        self.custom_configs[name] = config
        self.logger.info(f"Added custom rate limit config: {name}")
    
    def remove_custom_config(self, name: str) -> bool:
        """Remove custom rate limit configuration"""
        
        if name in self.custom_configs:
            del self.custom_configs[name]
            self.logger.info(f"Removed custom rate limit config: {name}")
            return True
        return False
    
    async def reset_user_limits(self, user_id: str) -> int:
        """Reset all rate limits for a user"""
        
        reset_count = 0
        keys_to_remove = []
        
        for key, bucket in self.buckets.items():
            if bucket.identifier == user_id:
                keys_to_remove.append(key)
                reset_count += 1
        
        for key in keys_to_remove:
            del self.buckets[key]
        
        if self.audit_logger:
            await self.audit_logger.log_event(
                action=AuditAction.RATE_LIMIT_RESET,
                user_id=user_id,
                resource_type="rate_limit",
                resource_id=user_id,
                details={'reset_buckets': reset_count}
            )
        
        return reset_count
    
    async def block_identifier(
        self,
        identifier: str,
        scope: RateLimitScope,
        duration_seconds: int,
        reason: str = "Manual block"
    ):
        """Manually block an identifier"""
        
        # Find relevant buckets
        for key, bucket in self.buckets.items():
            if bucket.identifier == identifier:
                bucket.is_blocked = True
                bucket.block_until = datetime.utcnow() + timedelta(seconds=duration_seconds)
                bucket.violation_count += 1
        
        if self.audit_logger:
            await self.audit_logger.log_event(
                action=AuditAction.MANUAL_BLOCK,
                user_id="system",
                resource_type="rate_limit",
                resource_id=identifier,
                details={
                    'scope': scope.value,
                    'duration_seconds': duration_seconds,
                    'reason': reason
                }
            )
    
    async def unblock_identifier(
        self,
        identifier: str,
        scope: RateLimitScope,
        reason: str = "Manual unblock"
    ):
        """Manually unblock an identifier"""
        
        unblocked_count = 0
        
        # Find and unblock relevant buckets
        for key, bucket in self.buckets.items():
            if bucket.identifier == identifier and bucket.is_blocked:
                bucket.is_blocked = False
                bucket.block_until = None
                bucket.violation_count = 0
                unblocked_count += 1
        
        if self.audit_logger:
            await self.audit_logger.log_event(
                action=AuditAction.MANUAL_UNBLOCK,
                user_id="system",
                resource_type="rate_limit",
                resource_id=identifier,
                details={
                    'scope': scope.value,
                    'unblocked_buckets': unblocked_count,
                    'reason': reason
                }
            )
    
    def get_rate_limit_status(self, identifier: str) -> List[Dict[str, Any]]:
        """Get current rate limit status for an identifier"""
        
        status_list = []
        
        for key, bucket in self.buckets.items():
            if bucket.identifier == identifier:
                status_list.append({
                    'scope': bucket.config.scope.value,
                    'algorithm': bucket.config.algorithm.value,
                    'current_count': bucket.current_count,
                    'limit': bucket.config.requests_per_window,
                    'window_size': bucket.config.window_size_seconds,
                    'is_blocked': bucket.is_blocked,
                    'block_until': bucket.block_until.isoformat() if bucket.block_until else None,
                    'violation_count': bucket.violation_count,
                    'last_request': bucket.last_request.isoformat(),
                    'tokens': bucket.tokens if bucket.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET else None
                })
        
        return status_list
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics"""
        
        self.metrics['active_buckets'] = len(self.buckets)
        
        return {
            **self.metrics,
            'success_rate': (
                self.metrics['allowed_requests'] / self.metrics['total_requests'] * 100
                if self.metrics['total_requests'] > 0 else 100
            ),
            'blocked_identifiers': sum(1 for bucket in self.buckets.values() if bucket.is_blocked),
            'custom_configs_count': len(self.custom_configs),
            'emergency_bypass_enabled': self.emergency_bypass_enabled
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on rate limiting system"""
        
        return {
            'status': 'healthy',
            'enabled': self.global_rate_limiting_enabled,
            'buckets_active': len(self.buckets),
            'buckets_limit': self.max_buckets,
            'cleanup_task_running': self._cleanup_task and not self._cleanup_task.done(),
            'emergency_bypass_enabled': self.emergency_bypass_enabled,
            'metrics': self.get_metrics()
        }
    
    async def shutdown(self):
        """Shutdown rate limiter and cleanup resources"""
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.buckets.clear()
        self.logger.info("Rate Limiter shutdown complete")


# Factory function
def create_rate_limiter(
    audit_logger: Optional[AuditLogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> RateLimiter:
    """Factory function to create a configured Rate Limiter"""
    
    return RateLimiter(
        audit_logger=audit_logger,
        config=config or {}
    )


# Export all necessary components
__all__ = [
    'RateLimiter',
    'RateLimitConfig',
    'RateLimitBucket',
    'RateLimitResult',
    'RateLimitAlgorithm',
    'RateLimitScope',
    'RateLimitStatus',
    'create_rate_limiter'
] 