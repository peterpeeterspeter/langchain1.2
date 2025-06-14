"""
Task 3.7: Performance Optimization System for Contextual Retrieval

This module provides comprehensive performance optimization for the contextual retrieval system,
including query optimization, caching strategies, and performance monitoring.

Key Features:
- Real-time performance monitoring and metrics collection
- Adaptive query optimization based on performance patterns
- Intelligent caching with TTL optimization
- Batch processing for high-throughput scenarios
- Connection pooling and resource management
- Performance profiling and bottleneck detection
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics
import json
from enum import Enum

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio_throttle import Throttler

# Import existing components
from .contextual_retrieval import ContextualRetrievalSystem, RetrievalConfig
from ..scoring.enhanced_confidence import IntelligentCache, SourceQualityAnalyzer

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    QUALITY_FOCUSED = "quality_focused"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUALITY_SCORE = "quality_score"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for retrieval operations."""
    
    # Timing metrics
    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    ranking_time_ms: float = 0.0
    cache_time_ms: float = 0.0
    
    # Quality metrics
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    source_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    token_count: int = 0
    api_calls: int = 0
    
    # Result metrics
    results_count: int = 0
    cache_hit: bool = False
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    # Context
    query_type: str = "unknown"
    strategy_used: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # Strategy settings
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    enable_adaptive_optimization: bool = True
    
    # Performance targets
    target_latency_ms: float = 500.0
    target_throughput_qps: float = 10.0
    target_cache_hit_rate: float = 0.7
    target_quality_score: float = 0.8
    
    # Resource limits
    max_concurrent_queries: int = 20
    max_memory_usage_mb: float = 1024.0
    max_api_calls_per_minute: int = 100
    
    # Optimization intervals
    metrics_collection_interval_s: int = 60
    optimization_interval_s: int = 300
    cache_cleanup_interval_s: int = 3600
    
    # Thresholds for optimization triggers
    latency_degradation_threshold: float = 1.5  # 50% increase
    quality_degradation_threshold: float = 0.1  # 10% decrease
    error_rate_threshold: float = 0.05  # 5% error rate


class PerformanceMonitor:
    """Real-time performance monitoring for contextual retrieval."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics_history = deque(maxlen=10000)
        self.current_metrics = defaultdict(list)
        self.performance_baselines = {}
        self.alerts = []
        
        # Performance tracking
        self.query_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        
        # Resource monitoring
        self.active_queries = 0
        self.peak_memory_usage = 0.0
        self.api_call_count = 0
        self.last_api_reset = datetime.now()
    
    async def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for analysis."""
        
        # Update counters
        self.query_count += 1
        self.total_latency += metrics.total_time_ms
        
        if metrics.cache_hit:
            self.cache_hits += 1
        
        if metrics.error_occurred:
            self.error_count += 1
        
        # Track resource usage
        self.peak_memory_usage = max(self.peak_memory_usage, metrics.memory_usage_mb)
        self.api_call_count += metrics.api_calls
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Update current metrics for real-time analysis
        self._update_current_metrics(metrics)
        
        # Check for performance alerts
        await self._check_performance_alerts(metrics)
    
    def _update_current_metrics(self, metrics: PerformanceMetrics):
        """Update current metrics for real-time monitoring."""
        
        # Keep last 100 measurements for each metric type
        for metric_type in PerformanceMetricType:
            if len(self.current_metrics[metric_type]) >= 100:
                self.current_metrics[metric_type].pop(0)
        
        self.current_metrics[PerformanceMetricType.LATENCY].append(metrics.total_time_ms)
        self.current_metrics[PerformanceMetricType.QUALITY_SCORE].append(metrics.confidence_score)
        self.current_metrics[PerformanceMetricType.CACHE_HIT_RATE].append(1.0 if metrics.cache_hit else 0.0)
        
        # Calculate throughput (queries per second)
        if len(self.metrics_history) >= 2:
            time_window = (self.metrics_history[-1].timestamp - self.metrics_history[-2].timestamp).total_seconds()
            if time_window > 0:
                throughput = 1.0 / time_window
                self.current_metrics[PerformanceMetricType.THROUGHPUT].append(throughput)
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance degradation and generate alerts."""
        
        alerts = []
        
        # Latency alert
        if metrics.total_time_ms > self.config.target_latency_ms * self.config.latency_degradation_threshold:
            alerts.append({
                "type": "latency_degradation",
                "message": f"Query latency {metrics.total_time_ms:.1f}ms exceeds threshold",
                "severity": "high",
                "timestamp": datetime.now()
            })
        
        # Quality alert
        if metrics.confidence_score < self.config.target_quality_score - self.config.quality_degradation_threshold:
            alerts.append({
                "type": "quality_degradation",
                "message": f"Quality score {metrics.confidence_score:.2f} below threshold",
                "severity": "medium",
                "timestamp": datetime.now()
            })
        
        # Error rate alert
        error_rate = self.error_count / max(self.query_count, 1)
        if error_rate > self.config.error_rate_threshold:
            alerts.append({
                "type": "high_error_rate",
                "message": f"Error rate {error_rate:.2%} exceeds threshold",
                "severity": "critical",
                "timestamp": datetime.now()
            })
        
        # Resource usage alerts
        if metrics.memory_usage_mb > self.config.max_memory_usage_mb * 0.9:
            alerts.append({
                "type": "high_memory_usage",
                "message": f"Memory usage {metrics.memory_usage_mb:.1f}MB approaching limit",
                "severity": "medium",
                "timestamp": datetime.now()
            })
        
        self.alerts.extend(alerts)
        
        # Log critical alerts
        for alert in alerts:
            if alert["severity"] == "critical":
                logger.error(f"Performance Alert: {alert['message']}")
            elif alert["severity"] == "high":
                logger.warning(f"Performance Alert: {alert['message']}")
    
    def get_current_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        
        if not self.metrics_history:
            return {"status": "no_data"}
        
        # Calculate averages from recent metrics
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 queries
        
        avg_latency = statistics.mean([m.total_time_ms for m in recent_metrics])
        avg_quality = statistics.mean([m.confidence_score for m in recent_metrics])
        cache_hit_rate = self.cache_hits / max(self.query_count, 1)
        error_rate = self.error_count / max(self.query_count, 1)
        
        # Calculate throughput
        if len(recent_metrics) >= 2:
            time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
            throughput = len(recent_metrics) / max(time_span, 1)
        else:
            throughput = 0.0
        
        return {
            "status": "active",
            "query_count": self.query_count,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_quality_score": round(avg_quality, 3),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "error_rate": round(error_rate, 3),
            "throughput_qps": round(throughput, 2),
            "peak_memory_mb": round(self.peak_memory_usage, 1),
            "active_queries": self.active_queries,
            "recent_alerts": len([a for a in self.alerts if a["timestamp"] > datetime.now() - timedelta(hours=1)])
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over specified time period."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"status": "insufficient_data"}
        
        # Group metrics by hour
        hourly_metrics = defaultdict(list)
        for metric in recent_metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_metrics[hour_key].append(metric)
        
        # Calculate trends
        trends = {}
        for hour, metrics in hourly_metrics.items():
            trends[hour.isoformat()] = {
                "avg_latency_ms": statistics.mean([m.total_time_ms for m in metrics]),
                "avg_quality": statistics.mean([m.confidence_score for m in metrics]),
                "query_count": len(metrics),
                "error_count": sum(1 for m in metrics if m.error_occurred),
                "cache_hits": sum(1 for m in metrics if m.cache_hit)
            }
        
        return {
            "status": "success",
            "time_period_hours": hours,
            "data_points": len(trends),
            "trends": trends
        }


class QueryOptimizer:
    """Intelligent query optimization based on performance patterns."""
    
    def __init__(self, config: OptimizationConfig, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        
        # Optimization parameters
        self.optimal_parameters = {
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "mmr_lambda": 0.7,
            "context_window": 2,
            "max_results": 10
        }
        
        # Performance history for parameter tuning
        self.parameter_performance = defaultdict(list)
        self.optimization_history = []
        
        # Learning rate for adaptive optimization
        self.learning_rate = 0.1
        self.exploration_rate = 0.1  # For exploration vs exploitation
    
    async def optimize_query_parameters(self, query: str, query_type: str) -> Dict[str, Any]:
        """Optimize query parameters based on historical performance."""
        
        # Get baseline parameters
        optimized_params = self.optimal_parameters.copy()
        
        # Apply strategy-specific optimizations
        if self.config.optimization_strategy == OptimizationStrategy.LATENCY_FOCUSED:
            optimized_params = await self._optimize_for_latency(optimized_params, query_type)
        elif self.config.optimization_strategy == OptimizationStrategy.QUALITY_FOCUSED:
            optimized_params = await self._optimize_for_quality(optimized_params, query_type)
        elif self.config.optimization_strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            optimized_params = await self._optimize_for_throughput(optimized_params, query_type)
        elif self.config.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            optimized_params = await self._adaptive_optimization(optimized_params, query, query_type)
        
        return optimized_params
    
    async def _optimize_for_latency(self, params: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Optimize parameters for minimum latency."""
        
        # Reduce context window for faster processing
        params["context_window"] = max(1, params["context_window"] - 1)
        
        # Reduce max results to speed up ranking
        params["max_results"] = min(5, params["max_results"])
        
        # Favor dense search (typically faster than sparse)
        params["dense_weight"] = min(0.9, params["dense_weight"] + 0.1)
        params["sparse_weight"] = 1.0 - params["dense_weight"]
        
        return params
    
    async def _optimize_for_quality(self, params: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Optimize parameters for maximum quality."""
        
        # Increase context window for better understanding
        params["context_window"] = min(4, params["context_window"] + 1)
        
        # Increase max results for better selection
        params["max_results"] = min(20, params["max_results"] + 5)
        
        # Balanced search weights for comprehensive results
        params["dense_weight"] = 0.6
        params["sparse_weight"] = 0.4
        
        # Increase MMR lambda for more diverse results
        params["mmr_lambda"] = min(0.9, params["mmr_lambda"] + 0.1)
        
        return params
    
    async def _optimize_for_throughput(self, params: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Optimize parameters for maximum throughput."""
        
        # Minimize processing overhead
        params["context_window"] = 1
        params["max_results"] = 3
        
        # Favor caching-friendly parameters
        params["dense_weight"] = 0.8  # Standardize for better caching
        params["sparse_weight"] = 0.2
        
        return params
    
    async def _adaptive_optimization(self, params: Dict[str, Any], query: str, query_type: str) -> Dict[str, Any]:
        """Adaptive optimization based on recent performance patterns."""
        
        # Get recent performance for similar queries
        recent_performance = await self._get_recent_performance_for_query_type(query_type)
        
        if not recent_performance:
            return params  # No data for optimization
        
        # Analyze performance patterns
        avg_latency = statistics.mean([p["latency"] for p in recent_performance])
        avg_quality = statistics.mean([p["quality"] for p in recent_performance])
        
        # Adaptive adjustments based on current performance vs targets
        if avg_latency > self.config.target_latency_ms:
            # Latency is too high, optimize for speed
            params = await self._optimize_for_latency(params, query_type)
        elif avg_quality < self.config.target_quality_score:
            # Quality is too low, optimize for quality
            params = await self._optimize_for_quality(params, query_type)
        
        # Exploration: occasionally try random variations
        if np.random.random() < self.exploration_rate:
            params = self._add_exploration_noise(params)
        
        return params
    
    async def _get_recent_performance_for_query_type(self, query_type: str) -> List[Dict[str, Any]]:
        """Get recent performance data for specific query type."""
        
        recent_metrics = [
            m for m in self.monitor.metrics_history
            if m.query_type == query_type and 
               m.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return [
            {
                "latency": m.total_time_ms,
                "quality": m.confidence_score,
                "cache_hit": m.cache_hit
            }
            for m in recent_metrics
        ]
    
    def _add_exploration_noise(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add small random variations for exploration."""
        
        noise_params = params.copy()
        
        # Add small random adjustments
        noise_params["dense_weight"] = np.clip(
            params["dense_weight"] + np.random.normal(0, 0.05), 0.1, 0.9
        )
        noise_params["sparse_weight"] = 1.0 - noise_params["dense_weight"]
        
        noise_params["mmr_lambda"] = np.clip(
            params["mmr_lambda"] + np.random.normal(0, 0.05), 0.1, 0.9
        )
        
        return noise_params
    
    async def update_parameter_performance(self, params: Dict[str, Any], metrics: PerformanceMetrics):
        """Update parameter performance tracking for learning."""
        
        # Create parameter signature
        param_signature = json.dumps(params, sort_keys=True)
        
        # Record performance for these parameters
        self.parameter_performance[param_signature].append({
            "latency": metrics.total_time_ms,
            "quality": metrics.confidence_score,
            "timestamp": metrics.timestamp
        })
        
        # Update optimal parameters if this combination performed better
        await self._update_optimal_parameters(params, metrics)
    
    async def _update_optimal_parameters(self, params: Dict[str, Any], metrics: PerformanceMetrics):
        """Update optimal parameters based on performance feedback."""
        
        # Calculate performance score (weighted combination of latency and quality)
        latency_score = max(0, 1.0 - (metrics.total_time_ms / self.config.target_latency_ms))
        quality_score = metrics.confidence_score
        
        # Weighted performance score based on strategy
        if self.config.optimization_strategy == OptimizationStrategy.LATENCY_FOCUSED:
            performance_score = latency_score * 0.8 + quality_score * 0.2
        elif self.config.optimization_strategy == OptimizationStrategy.QUALITY_FOCUSED:
            performance_score = latency_score * 0.2 + quality_score * 0.8
        else:
            performance_score = latency_score * 0.5 + quality_score * 0.5
        
        # Update optimal parameters using exponential moving average
        for key, value in params.items():
            if key in self.optimal_parameters:
                current_optimal = self.optimal_parameters[key]
                self.optimal_parameters[key] = (
                    current_optimal * (1 - self.learning_rate) +
                    value * self.learning_rate * performance_score
                )


class CacheOptimizer:
    """Intelligent cache optimization for contextual retrieval."""
    
    def __init__(self, cache: IntelligentCache, monitor: PerformanceMonitor):
        self.cache = cache
        self.monitor = monitor
        
        # Cache performance tracking
        self.cache_performance_history = deque(maxlen=1000)
        self.optimal_ttl_by_query_type = defaultdict(lambda: 24)  # Default 24 hours
        
        # TTL optimization parameters
        self.ttl_learning_rate = 0.1
        self.min_ttl_hours = 1
        self.max_ttl_hours = 168  # 1 week
    
    async def optimize_cache_strategy(self):
        """Optimize cache strategy based on performance patterns."""
        
        # Analyze cache performance
        cache_metrics = self.cache.get_performance_metrics()
        
        # Adjust TTL based on hit rates and staleness
        await self._optimize_ttl_by_query_type()
        
        # Optimize cache size and eviction strategy
        await self._optimize_cache_size_and_eviction()
        
        # Clean up expired and low-value entries
        await self._intelligent_cache_cleanup()
    
    async def _optimize_ttl_by_query_type(self):
        """Optimize TTL settings for different query types."""
        
        # Group recent metrics by query type
        recent_metrics = [
            m for m in self.monitor.metrics_history
            if m.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        query_type_performance = defaultdict(list)
        for metric in recent_metrics:
            query_type_performance[metric.query_type].append(metric)
        
        # Optimize TTL for each query type
        for query_type, metrics in query_type_performance.items():
            if len(metrics) < 10:  # Need sufficient data
                continue
            
            # Calculate cache effectiveness
            cache_hits = sum(1 for m in metrics if m.cache_hit)
            hit_rate = cache_hits / len(metrics)
            
            # Calculate average quality degradation over time
            quality_degradation = self._calculate_quality_degradation(metrics)
            
            # Adjust TTL based on hit rate and quality degradation
            current_ttl = self.optimal_ttl_by_query_type[query_type]
            
            if hit_rate < 0.5:  # Low hit rate, increase TTL
                new_ttl = min(self.max_ttl_hours, current_ttl * 1.2)
            elif quality_degradation > 0.1:  # High quality degradation, decrease TTL
                new_ttl = max(self.min_ttl_hours, current_ttl * 0.8)
            else:
                new_ttl = current_ttl  # Keep current TTL
            
            # Update with learning rate
            self.optimal_ttl_by_query_type[query_type] = (
                current_ttl * (1 - self.ttl_learning_rate) +
                new_ttl * self.ttl_learning_rate
            )
    
    def _calculate_quality_degradation(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate quality degradation over time for cached results."""
        
        cached_metrics = [m for m in metrics if m.cache_hit]
        if len(cached_metrics) < 5:
            return 0.0
        
        # Sort by timestamp
        cached_metrics.sort(key=lambda x: x.timestamp)
        
        # Calculate quality trend
        qualities = [m.confidence_score for m in cached_metrics]
        if len(qualities) < 2:
            return 0.0
        
        # Simple linear regression to detect quality degradation
        x = np.arange(len(qualities))
        slope = np.polyfit(x, qualities, 1)[0]
        
        # Return degradation rate (negative slope indicates degradation)
        return max(0.0, -slope)
    
    async def _optimize_cache_size_and_eviction(self):
        """Optimize cache size and eviction strategy."""
        
        cache_metrics = self.cache.get_performance_metrics()
        
        # Check if cache is frequently full (high eviction rate)
        if cache_metrics.get("cache_size_utilization", 0) > 0.9:
            # Consider increasing cache size or improving eviction strategy
            logger.info("Cache utilization high, consider increasing cache size")
        
        # Analyze eviction patterns
        eviction_rate = cache_metrics.get("evictions", 0) / max(cache_metrics.get("total_requests", 1), 1)
        if eviction_rate > 0.1:  # High eviction rate
            logger.warning(f"High cache eviction rate: {eviction_rate:.2%}")
    
    async def _intelligent_cache_cleanup(self):
        """Perform intelligent cache cleanup based on value scoring."""
        
        # Get cache entries with low value scores
        cache_info = self.cache.get_cache_info()
        
        low_value_entries = []
        for entry_info in cache_info.get("entries", []):
            if entry_info.get("cache_value_score", 0) < 0.3:
                low_value_entries.append(entry_info["key"])
        
        # Remove low-value entries
        for key in low_value_entries[:10]:  # Limit cleanup to avoid performance impact
            if key in self.cache.cache:
                del self.cache.cache[key]
        
        if low_value_entries:
            logger.info(f"Cleaned up {len(low_value_entries)} low-value cache entries")
    
    def get_optimal_ttl(self, query_type: str) -> int:
        """Get optimal TTL for a specific query type."""
        return int(self.optimal_ttl_by_query_type[query_type])


class BatchProcessor:
    """Batch processing for high-throughput retrieval scenarios."""
    
    def __init__(self, retrieval_system: ContextualRetrievalSystem, config: OptimizationConfig):
        self.retrieval_system = retrieval_system
        self.config = config
        
        # Batch processing settings
        self.batch_size = 10
        self.max_concurrent_batches = 3
        self.throttler = Throttler(rate_limit=config.max_api_calls_per_minute, period=60)
    
    async def batch_query(self, queries: List[str], query_types: List[str] = None) -> List[Dict[str, Any]]:
        """Process multiple queries in optimized batches."""
        
        if query_types is None:
            query_types = ["general"] * len(queries)
        
        # Group queries into batches
        batches = []
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            batch_types = query_types[i:i + self.batch_size]
            batches.append(list(zip(batch_queries, batch_types)))
        
        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_batch(batch):
            async with semaphore:
                return await self._process_single_batch(batch)
        
        # Execute all batches
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def _process_single_batch(self, batch: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Process a single batch of queries."""
        
        batch_results = []
        
        for query, query_type in batch:
            # Apply rate limiting
            async with self.throttler:
                try:
                    result = await self.retrieval_system.retrieve(query)
                    batch_results.append({
                        "query": query,
                        "query_type": query_type,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    batch_results.append({
                        "query": query,
                        "query_type": query_type,
                        "error": str(e),
                        "success": False
                    })
        
        return batch_results


class ConnectionPool:
    """Connection pooling for database and API connections."""
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self.connections = asyncio.Queue(maxsize=pool_size)
        self.active_connections = 0
        self.total_connections_created = 0
    
    async def initialize_pool(self, connection_factory):
        """Initialize the connection pool."""
        for _ in range(self.pool_size):
            connection = await connection_factory()
            await self.connections.put(connection)
            self.total_connections_created += 1
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        connection = await self.connections.get()
        self.active_connections += 1
        
        try:
            yield connection
        finally:
            await self.connections.put(connection)
            self.active_connections -= 1
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": self.pool_size,
            "active_connections": self.active_connections,
            "available_connections": self.connections.qsize(),
            "total_connections_created": self.total_connections_created
        }


class PerformanceOptimizer:
    """Main performance optimization orchestrator for contextual retrieval."""
    
    def __init__(
        self,
        retrieval_system: ContextualRetrievalSystem,
        cache: IntelligentCache,
        config: OptimizationConfig = None
    ):
        self.retrieval_system = retrieval_system
        self.cache = cache
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.monitor = PerformanceMonitor(self.config)
        self.query_optimizer = QueryOptimizer(self.config, self.monitor)
        self.cache_optimizer = CacheOptimizer(self.cache, self.monitor)
        self.batch_processor = BatchProcessor(self.retrieval_system, self.config)
        
        # Background tasks
        self._optimization_task = None
        self._monitoring_task = None
        
        logger.info("Performance Optimizer initialized")
    
    async def start_optimization(self):
        """Start background optimization tasks."""
        
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance optimization started")
    
    async def stop_optimization(self):
        """Stop background optimization tasks."""
        
        if self._optimization_task:
            self._optimization_task.cancel()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        logger.info("Performance optimization stopped")
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        
        while True:
            try:
                await asyncio.sleep(self.config.optimization_interval_s)
                
                # Run optimization tasks
                await self.cache_optimizer.optimize_cache_strategy()
                
                # Log optimization status
                logger.info("Optimization cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        
        while True:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval_s)
                
                # Collect system metrics
                performance_summary = self.monitor.get_current_performance_summary()
                
                # Log performance summary
                logger.info(f"Performance Summary: {performance_summary}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def optimized_retrieve(
        self,
        query: str,
        query_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """Perform optimized retrieval with performance monitoring."""
        
        start_time = time.time()
        metrics = PerformanceMetrics(query_type=query_type)
        
        try:
            # Optimize query parameters
            optimized_params = await self.query_optimizer.optimize_query_parameters(query, query_type)
            
            # Update retrieval config with optimized parameters
            config_updates = {
                "dense_weight": optimized_params["dense_weight"],
                "sparse_weight": optimized_params["sparse_weight"],
                "mmr_lambda": optimized_params["mmr_lambda"],
                "context_window_size": optimized_params["context_window"],
                "default_k": optimized_params["max_results"]
            }
            
            # Perform retrieval with optimized parameters
            embedding_start = time.time()
            result = await self.retrieval_system.retrieve(query, **config_updates)
            metrics.embedding_time_ms = (time.time() - embedding_start) * 1000
            
            # Calculate performance metrics
            metrics.total_time_ms = (time.time() - start_time) * 1000
            metrics.results_count = len(result) if isinstance(result, list) else 1
            metrics.cache_hit = getattr(result, 'cached', False) if hasattr(result, 'cached') else False
            
            # Calculate quality scores
            if hasattr(result, 'confidence_score'):
                metrics.confidence_score = result.confidence_score
            
            # Update parameter performance
            await self.query_optimizer.update_parameter_performance(optimized_params, metrics)
            
            # Record metrics
            await self.monitor.record_metrics(metrics)
            
            return result
            
        except Exception as e:
            metrics.error_occurred = True
            metrics.error_message = str(e)
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            await self.monitor.record_metrics(metrics)
            raise
    
    async def batch_optimized_retrieve(
        self,
        queries: List[str],
        query_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform batch retrieval with optimization."""
        
        return await self.batch_processor.batch_query(queries, query_types)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and metrics."""
        
        return {
            "performance_summary": self.monitor.get_current_performance_summary(),
            "optimization_config": {
                "strategy": self.config.optimization_strategy.value,
                "target_latency_ms": self.config.target_latency_ms,
                "target_quality_score": self.config.target_quality_score,
                "adaptive_enabled": self.config.enable_adaptive_optimization
            },
            "optimal_parameters": self.query_optimizer.optimal_parameters,
            "cache_metrics": self.cache.get_performance_metrics(),
            "recent_alerts": self.monitor.alerts[-10:],  # Last 10 alerts
            "optimization_active": self._optimization_task is not None and not self._optimization_task.done()
        }
    
    async def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        performance_trends = self.monitor.get_performance_trends(hours)
        cache_metrics = self.cache.get_performance_metrics()
        
        return {
            "report_period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "performance_trends": performance_trends,
            "cache_performance": cache_metrics,
            "optimization_effectiveness": await self._calculate_optimization_effectiveness(),
            "recommendations": await self._generate_optimization_recommendations()
        }
    
    async def _calculate_optimization_effectiveness(self) -> Dict[str, Any]:
        """Calculate the effectiveness of optimization efforts."""
        
        # Compare recent performance to baseline
        recent_metrics = [
            m for m in self.monitor.metrics_history
            if m.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if len(recent_metrics) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate improvements
        recent_avg_latency = statistics.mean([m.total_time_ms for m in recent_metrics])
        recent_avg_quality = statistics.mean([m.confidence_score for m in recent_metrics])
        recent_cache_hit_rate = sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics)
        
        return {
            "avg_latency_ms": round(recent_avg_latency, 2),
            "avg_quality_score": round(recent_avg_quality, 3),
            "cache_hit_rate": round(recent_cache_hit_rate, 3),
            "optimization_impact": "positive" if recent_avg_latency < self.config.target_latency_ms else "needs_improvement"
        }
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current performance."""
        
        recommendations = []
        performance_summary = self.monitor.get_current_performance_summary()
        
        # Latency recommendations
        if performance_summary.get("avg_latency_ms", 0) > self.config.target_latency_ms:
            recommendations.append("Consider reducing context window size or max results for faster retrieval")
        
        # Quality recommendations
        if performance_summary.get("avg_quality_score", 0) < self.config.target_quality_score:
            recommendations.append("Consider increasing context window or adjusting search weights for better quality")
        
        # Cache recommendations
        if performance_summary.get("cache_hit_rate", 0) < self.config.target_cache_hit_rate:
            recommendations.append("Consider increasing cache TTL or improving cache key generation")
        
        # Error rate recommendations
        if performance_summary.get("error_rate", 0) > self.config.error_rate_threshold:
            recommendations.append("Investigate error patterns and implement additional retry logic")
        
        return recommendations


# Factory function for easy initialization
def create_performance_optimizer(
    retrieval_system: ContextualRetrievalSystem,
    cache: IntelligentCache,
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
) -> PerformanceOptimizer:
    """Create a configured performance optimizer."""
    
    config = OptimizationConfig(optimization_strategy=optimization_strategy)
    return PerformanceOptimizer(retrieval_system, cache, config)


# Export main classes
__all__ = [
    'PerformanceOptimizer',
    'PerformanceMonitor',
    'QueryOptimizer',
    'CacheOptimizer',
    'BatchProcessor',
    'ConnectionPool',
    'OptimizationConfig',
    'PerformanceMetrics',
    'OptimizationStrategy',
    'create_performance_optimizer'
] 