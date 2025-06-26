"""
TTL Strategy Module.

This module implements Time-To-Live (TTL) strategies for cache management.
TTL values are handled in seconds (not timedelta objects) for compatibility
with Redis and LangChain's caching infrastructure.

The module provides content-type specific TTL configurations:
- News content: Shorter TTL for frequently changing information
- Review content: Moderate TTL for semi-static content
- Regulatory content: Longer TTL for stable regulatory information

All TTL values are configurable and can be adjusted based on content freshness
requirements and system performance characteristics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any
from enum import Enum
import time
from dataclasses import dataclass

from .content_type_config import ContentType, ContentTypeConfig
from .exceptions import CacheTTLException


class TTLCalculationStrategy(str, Enum):
    """Enumeration of TTL calculation strategies."""
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    CONTENT_BASED = "content_based"
    TIME_BASED = "time_based"


@dataclass
class TTLContext:
    """Context information for TTL calculation."""
    content_type: ContentType
    query_length: int = 0
    query_complexity: float = 1.0
    current_time: Optional[int] = None
    user_priority: str = "normal"
    cache_hit_rate: float = 0.0
    content_freshness_score: float = 1.0
    
    def __post_init__(self):
        if self.current_time is None:
            self.current_time = int(time.time())


class BaseTTLStrategy(ABC):
    """Abstract base class for TTL calculation strategies."""
    
    def __init__(self, config: ContentTypeConfig):
        self.config = config
        self.strategy_type = TTLCalculationStrategy.FIXED
    
    @abstractmethod
    def calculate_ttl(self, context: TTLContext) -> int:
        """Calculate TTL in seconds based on the provided context."""
        pass
    
    def validate_ttl(self, ttl_seconds: int) -> int:
        """Validate TTL value and ensure it's within acceptable bounds."""
        if ttl_seconds <= 0:
            raise CacheTTLException(
                f"TTL must be positive, got: {ttl_seconds}",
                ttl_value=ttl_seconds,
                content_type=self.config.content_type.value
            )
        
        # Ensure TTL is not too small (minimum 60 seconds)
        min_ttl = 60
        if ttl_seconds < min_ttl:
            ttl_seconds = min_ttl
        
        # Ensure TTL is not unreasonably large (maximum 7 days)
        max_ttl = 7 * 24 * 3600  # 7 days in seconds
        if ttl_seconds > max_ttl:
            ttl_seconds = max_ttl
        
        return ttl_seconds


class FixedTTLStrategy(BaseTTLStrategy):
    """Simple fixed TTL strategy using configuration defaults."""
    
    def __init__(self, config: ContentTypeConfig):
        super().__init__(config)
        self.strategy_type = TTLCalculationStrategy.FIXED
    
    def calculate_ttl(self, context: TTLContext) -> int:
        """Return the fixed TTL from configuration."""
        ttl = self.config.ttl_seconds
        return self.validate_ttl(ttl)


class DynamicTTLStrategy(BaseTTLStrategy):
    """Dynamic TTL strategy that adjusts based on cache performance."""
    
    def __init__(self, config: ContentTypeConfig, base_multiplier: float = 1.0):
        super().__init__(config)
        self.strategy_type = TTLCalculationStrategy.DYNAMIC
        self.base_multiplier = base_multiplier
    
    def calculate_ttl(self, context: TTLContext) -> int:
        """Calculate TTL based on cache hit rate and performance metrics."""
        base_ttl = self.config.ttl_seconds
        
        # Adjust based on cache hit rate
        hit_rate_multiplier = 1.0
        if context.cache_hit_rate > 0.8:
            # High hit rate - increase TTL
            hit_rate_multiplier = 1.5
        elif context.cache_hit_rate < 0.3:
            # Low hit rate - decrease TTL
            hit_rate_multiplier = 0.7
        
        # Adjust based on query complexity
        complexity_multiplier = min(context.query_complexity, 2.0)
        
        # Adjust based on user priority
        priority_multiplier = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.3
        }.get(context.user_priority, 1.0)
        
        calculated_ttl = int(
            base_ttl * 
            self.base_multiplier * 
            hit_rate_multiplier * 
            complexity_multiplier * 
            priority_multiplier
        )
        
        return self.validate_ttl(calculated_ttl)


class ContentBasedTTLStrategy(BaseTTLStrategy):
    """TTL strategy that adjusts based on content characteristics."""
    
    def __init__(self, config: ContentTypeConfig):
        super().__init__(config)
        self.strategy_type = TTLCalculationStrategy.CONTENT_BASED
    
    def calculate_ttl(self, context: TTLContext) -> int:
        """Calculate TTL based on content type and freshness requirements."""
        base_ttl = self.config.ttl_seconds
        
        # Adjust based on content freshness score
        freshness_multiplier = context.content_freshness_score
        
        # Adjust based on query length (longer queries might be more specific)
        length_multiplier = 1.0
        if context.query_length > 1000:
            length_multiplier = 1.2  # Longer queries get longer TTL
        elif context.query_length < 100:
            length_multiplier = 0.8  # Shorter queries get shorter TTL
        
        # Content type specific adjustments
        type_multiplier = {
            ContentType.NEWS: 0.5,      # News should expire quickly
            ContentType.REVIEWS: 1.0,    # Reviews are moderately stable
            ContentType.REGULATORY: 2.0  # Regulatory info is very stable
        }.get(context.content_type, 1.0)
        
        calculated_ttl = int(
            base_ttl * 
            freshness_multiplier * 
            length_multiplier * 
            type_multiplier
        )
        
        return self.validate_ttl(calculated_ttl)


class TimeBasedTTLStrategy(BaseTTLStrategy):
    """TTL strategy that adjusts based on time of day and usage patterns."""
    
    def __init__(self, config: ContentTypeConfig):
        super().__init__(config)
        self.strategy_type = TTLCalculationStrategy.TIME_BASED
    
    def calculate_ttl(self, context: TTLContext) -> int:
        """Calculate TTL based on time patterns and usage."""
        base_ttl = self.config.ttl_seconds
        current_time = context.current_time or int(time.time())
        
        # Get hour of day (0-23)
        hour = time.gmtime(current_time).tm_hour
        
        # Adjust based on time of day
        if 6 <= hour <= 18:  # Business hours
            time_multiplier = 1.2  # Higher activity, longer TTL
        elif 22 <= hour or hour <= 2:  # Night hours
            time_multiplier = 0.8  # Lower activity, shorter TTL
        else:
            time_multiplier = 1.0  # Normal hours
        
        # Weekend adjustment (if needed)
        weekday = time.gmtime(current_time).tm_wday
        if weekday >= 5:  # Weekend (Saturday=5, Sunday=6)
            time_multiplier *= 0.9
        
        calculated_ttl = int(base_ttl * time_multiplier)
        return self.validate_ttl(calculated_ttl)


class TTLStrategyFactory:
    """Factory for creating TTL strategies."""
    
    _strategies = {
        TTLCalculationStrategy.FIXED: FixedTTLStrategy,
        TTLCalculationStrategy.DYNAMIC: DynamicTTLStrategy,
        TTLCalculationStrategy.CONTENT_BASED: ContentBasedTTLStrategy,
        TTLCalculationStrategy.TIME_BASED: TimeBasedTTLStrategy
    }
    
    @classmethod
    def create_strategy(
        cls, 
        strategy_type: TTLCalculationStrategy, 
        config: ContentTypeConfig,
        **kwargs
    ) -> BaseTTLStrategy:
        """Create a TTL strategy of the specified type."""
        if strategy_type not in cls._strategies:
            raise CacheTTLException(
                f"Unknown TTL strategy type: {strategy_type}",
                content_type=config.content_type.value,
                details={"available_strategies": list(cls._strategies.keys())}
            )
        
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(config, **kwargs)


class TTLManager:
    """Manager class for handling TTL calculations across different strategies."""
    
    def __init__(self, default_strategy: TTLCalculationStrategy = TTLCalculationStrategy.FIXED):
        self.default_strategy = default_strategy
        self._strategies: Dict[ContentType, BaseTTLStrategy] = {}
    
    def register_strategy(self, content_type: ContentType, strategy: BaseTTLStrategy) -> None:
        """Register a strategy for a specific content type."""
        self._strategies[content_type] = strategy
    
    def get_ttl(self, context: TTLContext, config: ContentTypeConfig) -> int:
        """Get TTL for the given context and configuration."""
        try:
            strategy = self._strategies.get(context.content_type)
            
            if strategy is None:
                # Create default strategy for this content type
                strategy = TTLStrategyFactory.create_strategy(
                    self.default_strategy, 
                    config
                )
                self._strategies[context.content_type] = strategy
            
            return strategy.calculate_ttl(context)
            
        except Exception as e:
            raise CacheTTLException(
                f"Failed to calculate TTL for content type {context.content_type}: {str(e)}",
                content_type=context.content_type.value,
                details={"context": context.__dict__}
            )
    
    def update_strategy(
        self, 
        content_type: ContentType, 
        strategy_type: TTLCalculationStrategy,
        config: ContentTypeConfig,
        **kwargs
    ) -> None:
        """Update the strategy for a specific content type."""
        strategy = TTLStrategyFactory.create_strategy(strategy_type, config, **kwargs)
        self.register_strategy(content_type, strategy)
    
    def get_registered_strategies(self) -> Dict[ContentType, BaseTTLStrategy]:
        """Get all registered strategies."""
        return self._strategies.copy() 