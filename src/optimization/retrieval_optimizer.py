"""
Task 3.7: Performance Optimization Implementation
Universal RAG CMS - Contextual Retrieval Performance Optimization

This module provides comprehensive performance optimization for the contextual retrieval system,
including parameter tuning, caching strategies, connection pooling, and performance monitoring.
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json

# Import contextual retrieval components
try:
    from src.retrieval.contextual_retrieval import (
        ContextualRetrievalSystem, RetrievalConfig, RetrievalStrategy
    )
    from src.chains.enhanced_confidence_scoring_system import IntelligentCache
    CONTEXTUAL_RETRIEVAL_AVAILABLE = True
except ImportError:
    logging.warning("Contextual retrieval system not available. Running in basic mode.")
    CONTEXTUAL_RETRIEVAL_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for different scenarios."""
    SPEED_OPTIMIZED = "speed"           # Prioritize response time
    QUALITY_OPTIMIZED = "quality"       # Prioritize result quality
    BALANCED = "balanced"               # Balance speed and quality
    COST_OPTIMIZED = "cost"             # Minimize API costs
    MEMORY_OPTIMIZED = "memory"         # Minimize memory usage


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for retrieval operations."""
    
    # Timing metrics (in milliseconds)
    total_time_ms: float = 0.0
    contextual_embedding_time_ms: float = 0.0
    hybrid_search_time_ms: float = 0.0
    multi_query_time_ms: float = 0.0
    self_query_time_ms: float = 0.0
    mmr_computation_time_ms: float = 0.0
    cache_time_ms: float = 0.0
    
    # Quality metrics
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    coverage_score: float = 0.0
    
    # Resource utilization
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    api_calls_count: int = 0
    tokens_used: int = 0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0
    
    # Error tracking
    errors_count: int = 0
    warnings_count: int = 0
    
    # Configuration used
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    query_type: str = "unknown"
    result_count: int = 0


@dataclass
class OptimizationResult:
    """Results from parameter optimization experiments."""
    
    # Best configuration found
    optimal_config: RetrievalConfig
    
    # Performance improvements
    performance_improvement: float = 0.0  # Percentage improvement
    quality_improvement: float = 0.0
    speed_improvement: float = 0.0
    
    # Validation metrics
    validation_queries_tested: int = 0
    avg_precision_at_5: float = 0.0
    avg_precision_at_10: float = 0.0
    avg_response_time_ms: float = 0.0
    
    # Statistical significance
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Optimization metadata
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    iterations_completed: int = 0
    total_optimization_time_minutes: float = 0.0
    
    # Detailed results
    parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[PerformanceMetrics] = field(default_factory=list)


class ConnectionPoolManager:
    """
    Advanced connection pool management for database operations.
    Optimizes database connections for concurrent retrieval operations.
    """
    
    def __init__(self, 
                 pool_size: int = 20,
                 max_overflow: int = 10,
                 pool_timeout: int = 30,
                 recycle_time: int = 3600):
        """Initialize connection pool with optimized settings."""
        
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.recycle_time = recycle_time
        
        # Pool performance tracking
        self.pool_metrics = {
            'connections_created': 0,
            'connections_closed': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'avg_checkout_time_ms': 0.0,
            'active_connections': 0,
            'pool_utilization': 0.0
        }
        
        # Connection health monitoring
        self.unhealthy_connections = set()
        self.connection_last_used = {}
        
        logger.info(f"ConnectionPoolManager initialized: pool_size={pool_size}, max_overflow={max_overflow}")
    
    async def get_optimized_connection(self, operation_type: str = "read"):
        """Get an optimized database connection based on operation type."""
        
        start_time = time.time()
        
        try:
            # Select connection based on operation type
            if operation_type == "write":
                # Use primary connection for writes
                connection = await self._get_write_connection()
            elif operation_type == "analytics":
                # Use read replica for analytics
                connection = await self._get_analytics_connection()
            else:
                # Use load-balanced connection for reads
                connection = await self._get_read_connection()
            
            # Update metrics
            checkout_time = (time.time() - start_time) * 1000
            self._update_pool_metrics(checkout_time, hit=connection is not None)
            
            return connection
            
        except Exception as e:
            logger.error(f"Error getting optimized connection: {e}")
            self._update_pool_metrics(0, hit=False)
            raise
    
    async def _get_write_connection(self):
        """Get connection optimized for write operations."""
        # Implementation would depend on your specific database setup
        # This is a placeholder for the connection logic
        pass
    
    async def _get_read_connection(self):
        """Get connection optimized for read operations."""
        # Implementation would use read replicas if available
        pass
    
    async def _get_analytics_connection(self):
        """Get connection optimized for analytics queries."""
        # Implementation would use dedicated analytics connection
        pass
    
    def _update_pool_metrics(self, checkout_time_ms: float, hit: bool):
        """Update connection pool performance metrics."""
        
        if hit:
            self.pool_metrics['pool_hits'] += 1
        else:
            self.pool_metrics['pool_misses'] += 1
        
        # Update average checkout time
        current_avg = self.pool_metrics['avg_checkout_time_ms']
        total_operations = self.pool_metrics['pool_hits'] + self.pool_metrics['pool_misses']
        
        if total_operations > 1:
            self.pool_metrics['avg_checkout_time_ms'] = (
                (current_avg * (total_operations - 1) + checkout_time_ms) / total_operations
            )
        else:
            self.pool_metrics['avg_checkout_time_ms'] = checkout_time_ms
    
    def get_pool_health_status(self) -> Dict[str, Any]:
        """Get comprehensive pool health and performance status."""
        
        total_operations = self.pool_metrics['pool_hits'] + self.pool_metrics['pool_misses']
        hit_rate = self.pool_metrics['pool_hits'] / total_operations if total_operations > 0 else 0.0
        
        return {
            'pool_size': self.pool_size,
            'active_connections': self.pool_metrics['active_connections'],
            'pool_utilization': self.pool_metrics['pool_utilization'],
            'hit_rate': hit_rate,
            'avg_checkout_time_ms': self.pool_metrics['avg_checkout_time_ms'],
            'unhealthy_connections': len(self.unhealthy_connections),
            'total_operations': total_operations,
            'health_status': 'healthy' if hit_rate > 0.95 and self.pool_metrics['avg_checkout_time_ms'] < 100 else 'degraded'
        }


class CacheOptimizer:
    """
    Advanced caching optimization for contextual retrieval operations.
    Implements intelligent cache warming, eviction strategies, and performance tuning.
    """
    
    def __init__(self, cache_system: Optional[IntelligentCache] = None):
        """Initialize cache optimizer with intelligent cache system."""
        
        self.cache_system = cache_system
        self.cache_metrics = {
            'warm_cache_hits': 0,
            'cold_cache_misses': 0,
            'evictions_prevented': 0,
            'preload_success_rate': 0.0,
            'avg_cache_response_time_ms': 0.0
        }
        
        # Cache warming strategies
        self.warming_strategies = {
            'popular_queries': self._warm_popular_queries,
            'predictive_loading': self._warm_predictive_queries,
            'scheduled_refresh': self._warm_scheduled_content
        }
        
        logger.info("CacheOptimizer initialized with intelligent caching strategies")
    
    async def optimize_cache_performance(self, 
                                       query_patterns: List[str],
                                       optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> Dict[str, Any]:
        """Optimize cache performance based on query patterns and strategy."""
        
        optimization_results = {
            'cache_hit_rate_before': 0.0,
            'cache_hit_rate_after': 0.0,
            'avg_response_time_before_ms': 0.0,
            'avg_response_time_after_ms': 0.0,
            'optimizations_applied': []
        }
        
        try:
            # Measure baseline performance
            baseline_metrics = await self._measure_cache_performance(query_patterns)
            optimization_results['cache_hit_rate_before'] = baseline_metrics['hit_rate']
            optimization_results['avg_response_time_before_ms'] = baseline_metrics['avg_response_time_ms']
            
            # Apply optimizations based on strategy
            if optimization_strategy == OptimizationStrategy.SPEED_OPTIMIZED:
                await self._apply_speed_optimizations()
                optimization_results['optimizations_applied'].append('speed_optimizations')
                
            elif optimization_strategy == OptimizationStrategy.QUALITY_OPTIMIZED:
                await self._apply_quality_optimizations()
                optimization_results['optimizations_applied'].append('quality_optimizations')
                
            elif optimization_strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
                await self._apply_memory_optimizations()
                optimization_results['optimizations_applied'].append('memory_optimizations')
                
            else:  # BALANCED
                await self._apply_balanced_optimizations()
                optimization_results['optimizations_applied'].append('balanced_optimizations')
            
            # Warm cache with optimized strategy
            await self._warm_cache_strategically(query_patterns, optimization_strategy)
            optimization_results['optimizations_applied'].append('strategic_cache_warming')
            
            # Measure improved performance
            improved_metrics = await self._measure_cache_performance(query_patterns)
            optimization_results['cache_hit_rate_after'] = improved_metrics['hit_rate']
            optimization_results['avg_response_time_after_ms'] = improved_metrics['avg_response_time_ms']
            
            # Calculate improvement
            hit_rate_improvement = (
                (improved_metrics['hit_rate'] - baseline_metrics['hit_rate']) / 
                baseline_metrics['hit_rate'] * 100
            ) if baseline_metrics['hit_rate'] > 0 else 0
            
            speed_improvement = (
                (baseline_metrics['avg_response_time_ms'] - improved_metrics['avg_response_time_ms']) / 
                baseline_metrics['avg_response_time_ms'] * 100
            ) if baseline_metrics['avg_response_time_ms'] > 0 else 0
            
            optimization_results['hit_rate_improvement_percent'] = hit_rate_improvement
            optimization_results['speed_improvement_percent'] = speed_improvement
            
            logger.info(f"Cache optimization completed: {hit_rate_improvement:.1f}% hit rate improvement, {speed_improvement:.1f}% speed improvement")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return optimization_results
    
    async def _measure_cache_performance(self, query_patterns: List[str]) -> Dict[str, float]:
        """Measure baseline cache performance metrics."""
        
        if not self.cache_system:
            return {'hit_rate': 0.0, 'avg_response_time_ms': 1000.0}
        
        # Simulate queries to measure performance
        hits = 0
        total_time = 0.0
        
        for query in query_patterns[:20]:  # Sample first 20 queries
            start_time = time.time()
            result = await self.cache_system.get(query)
            end_time = time.time()
            
            if result is not None:
                hits += 1
            
            total_time += (end_time - start_time) * 1000
        
        hit_rate = hits / len(query_patterns[:20]) if query_patterns else 0.0
        avg_response_time = total_time / len(query_patterns[:20]) if query_patterns else 0.0
        
        return {
            'hit_rate': hit_rate,
            'avg_response_time_ms': avg_response_time
        }
    
    async def _apply_speed_optimizations(self):
        """Apply optimizations focused on speed."""
        
        if self.cache_system and hasattr(self.cache_system, 'config'):
            # Increase cache size for better hit rates
            self.cache_system.max_size = min(50000, self.cache_system.max_size * 1.5)
            
            # Use aggressive caching strategy
            from src.chains.enhanced_confidence_scoring_system import CacheStrategy
            self.cache_system.strategy = CacheStrategy.AGGRESSIVE
            
            # Reduce quality threshold for faster caching
            if hasattr(self.cache_system.strategy_config, CacheStrategy.AGGRESSIVE):
                self.cache_system.strategy_config[CacheStrategy.AGGRESSIVE]['quality_threshold'] = 0.3
    
    async def _apply_quality_optimizations(self):
        """Apply optimizations focused on quality."""
        
        if self.cache_system and hasattr(self.cache_system, 'config'):
            # Use conservative caching for higher quality
            from src.chains.enhanced_confidence_scoring_system import CacheStrategy
            self.cache_system.strategy = CacheStrategy.CONSERVATIVE
            
            # Increase quality threshold
            if hasattr(self.cache_system.strategy_config, CacheStrategy.CONSERVATIVE):
                self.cache_system.strategy_config[CacheStrategy.CONSERVATIVE]['quality_threshold'] = 0.8
    
    async def _apply_memory_optimizations(self):
        """Apply optimizations focused on memory usage."""
        
        if self.cache_system:
            # Reduce cache size
            self.cache_system.max_size = max(1000, int(self.cache_system.max_size * 0.7))
            
            # Implement aggressive eviction
            await self.cache_system.clear_expired()
    
    async def _apply_balanced_optimizations(self):
        """Apply balanced optimizations."""
        
        if self.cache_system and hasattr(self.cache_system, 'config'):
            # Use adaptive caching strategy
            from src.chains.enhanced_confidence_scoring_system import CacheStrategy
            self.cache_system.strategy = CacheStrategy.ADAPTIVE
    
    async def _warm_cache_strategically(self, query_patterns: List[str], strategy: OptimizationStrategy):
        """Warm cache based on optimization strategy."""
        
        if not self.cache_system:
            return
        
        # Select warming approach based on strategy
        if strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            # Warm most frequent queries
            await self.warming_strategies['popular_queries'](query_patterns[:50])
            
        elif strategy == OptimizationStrategy.QUALITY_OPTIMIZED:
            # Warm high-quality content
            await self.warming_strategies['scheduled_refresh'](query_patterns[:30])
            
        else:
            # Balanced approach
            await self.warming_strategies['predictive_loading'](query_patterns[:40])
    
    async def _warm_popular_queries(self, queries: List[str]):
        """Warm cache with popular queries."""
        # Implementation would pre-execute popular queries
        logger.info(f"Warming cache with {len(queries)} popular queries")
    
    async def _warm_predictive_queries(self, queries: List[str]):
        """Warm cache with predictively relevant queries."""
        # Implementation would use ML to predict likely queries
        logger.info(f"Warming cache with {len(queries)} predictive queries")
    
    async def _warm_scheduled_content(self, queries: List[str]):
        """Warm cache with scheduled content refresh."""
        # Implementation would refresh time-sensitive content
        logger.info(f"Warming cache with {len(queries)} scheduled content")


class RetrievalOptimizer:
    """
    Main class for comprehensive retrieval performance optimization.
    Integrates all optimization strategies and provides unified interface.
    """
    
    def __init__(self, 
                 retrieval_system: Optional['ContextualRetrievalSystem'] = None,
                 connection_pool: Optional[ConnectionPoolManager] = None,
                 cache_optimizer: Optional[CacheOptimizer] = None):
        """Initialize the retrieval optimizer with all subsystems."""
        
        self.retrieval_system = retrieval_system
        self.connection_pool = connection_pool or ConnectionPoolManager()
        self.cache_optimizer = cache_optimizer or CacheOptimizer()
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # Optimization parameters to test
        self.parameter_search_space = {
            'dense_weight': [0.5, 0.6, 0.7, 0.8, 0.9],
            'sparse_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
            'mmr_lambda': [0.5, 0.6, 0.7, 0.8, 0.9],
            'context_window_size': [1, 2, 3, 4],
            'max_query_variations': [1, 2, 3, 4, 5],
            'cache_ttl_hours': [6, 12, 24, 48, 72]
        }
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        logger.info("RetrievalOptimizer initialized with comprehensive optimization capabilities")
    
    async def optimize_retrieval_parameters(self,
                                          validation_queries: List[Tuple[str, List[str]]],
                                          optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                                          max_iterations: int = 50,
                                          target_improvement: float = 0.10) -> OptimizationResult:
        """
        Optimize retrieval parameters using advanced grid search with early stopping.
        
        Args:
            validation_queries: List of (query, expected_doc_ids) for validation
            optimization_strategy: Strategy to optimize for
            max_iterations: Maximum optimization iterations
            target_improvement: Stop early if this improvement is achieved
            
        Returns:
            OptimizationResult with best configuration and performance metrics
        """
        
        if not self.retrieval_system:
            raise ValueError("Retrieval system not initialized")
        
        logger.info(f"Starting parameter optimization with {len(validation_queries)} validation queries")
        start_time = time.time()
        
        # Measure baseline performance
        baseline_metrics = await self._evaluate_current_configuration(validation_queries)
        best_config = self.retrieval_system.config
        best_score = self._calculate_objective_score(baseline_metrics, optimization_strategy)
        
        optimization_result = OptimizationResult(
            optimal_config=best_config,
            optimization_strategy=optimization_strategy,
            validation_queries_tested=len(validation_queries)
        )
        
        # Grid search with intelligent sampling
        param_combinations = self._generate_parameter_combinations(optimization_strategy, max_iterations)
        
        iteration = 0
        for params in param_combinations:
            if iteration >= max_iterations:
                break
            
            try:
                # Update retrieval system configuration
                updated_config = self._update_config(self.retrieval_system.config, params)
                self.retrieval_system.config = updated_config
                
                # Evaluate configuration
                metrics = await self._evaluate_current_configuration(validation_queries)
                score = self._calculate_objective_score(metrics, optimization_strategy)
                
                # Track performance
                optimization_result.parameter_history.append(params.copy())
                optimization_result.performance_history.append(metrics)
                
                # Check if this is the best configuration
                if score > best_score:
                    improvement = (score - best_score) / best_score
                    best_score = score
                    best_config = updated_config
                    optimization_result.optimal_config = best_config
                    optimization_result.performance_improvement = improvement
                    
                    logger.info(f"Iteration {iteration}: New best score {score:.4f} (improvement: {improvement:.2%})")
                    
                    # Early stopping if target improvement achieved
                    if improvement >= target_improvement:
                        logger.info(f"Target improvement {target_improvement:.2%} achieved, stopping early")
                        break
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in optimization iteration {iteration}: {e}")
                continue
        
        # Apply best configuration
        self.retrieval_system.config = best_config
        
        # Calculate final metrics
        final_metrics = await self._evaluate_current_configuration(validation_queries)
        optimization_result.avg_precision_at_5 = final_metrics.relevance_score
        optimization_result.avg_response_time_ms = final_metrics.total_time_ms
        optimization_result.iterations_completed = iteration
        optimization_result.total_optimization_time_minutes = (time.time() - start_time) / 60
        
        # Calculate improvements
        if baseline_metrics.total_time_ms > 0:
            optimization_result.speed_improvement = (
                (baseline_metrics.total_time_ms - final_metrics.total_time_ms) / 
                baseline_metrics.total_time_ms
            )
        
        if baseline_metrics.relevance_score > 0:
            optimization_result.quality_improvement = (
                (final_metrics.relevance_score - baseline_metrics.relevance_score) / 
                baseline_metrics.relevance_score
            )
        
        # Store optimization result
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Parameter optimization completed: {optimization_result.performance_improvement:.2%} improvement in {iteration} iterations")
        
        return optimization_result
    
    def _generate_parameter_combinations(self, 
                                       strategy: OptimizationStrategy, 
                                       max_combinations: int) -> List[Dict[str, Any]]:
        """Generate intelligent parameter combinations based on optimization strategy."""
        
        combinations = []
        
        if strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            # Focus on parameters that improve speed
            for dense_w in [0.8, 0.9]:  # Higher dense weight for speed
                for mmr_l in [0.8, 0.9]:  # Higher lambda for less diversity computation
                    for cache_ttl in [24, 48]:  # Longer cache for speed
                        for max_vars in [1, 2]:  # Fewer query variations for speed
                            combinations.append({
                                'dense_weight': dense_w,
                                'sparse_weight': 1.0 - dense_w,
                                'mmr_lambda': mmr_l,
                                'cache_ttl_hours': cache_ttl,
                                'max_query_variations': max_vars
                            })
        
        elif strategy == OptimizationStrategy.QUALITY_OPTIMIZED:
            # Focus on parameters that improve quality
            for dense_w in [0.6, 0.7]:  # Balanced for quality
                for mmr_l in [0.5, 0.6, 0.7]:  # More diversity for quality
                    for context_w in [2, 3, 4]:  # More context for quality
                        for max_vars in [3, 4, 5]:  # More query variations for coverage
                            combinations.append({
                                'dense_weight': dense_w,
                                'sparse_weight': 1.0 - dense_w,
                                'mmr_lambda': mmr_l,
                                'context_window_size': context_w,
                                'max_query_variations': max_vars
                            })
        
        else:  # BALANCED or other strategies
            # Comprehensive grid search
            import itertools
            
            param_names = ['dense_weight', 'mmr_lambda', 'context_window_size', 'max_query_variations']
            param_values = [
                self.parameter_search_space['dense_weight'][:3],  # Top 3 values
                self.parameter_search_space['mmr_lambda'][:3],
                self.parameter_search_space['context_window_size'][:3],
                self.parameter_search_space['max_query_variations'][:3]
            ]
            
            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                param_dict['sparse_weight'] = 1.0 - param_dict['dense_weight']
                combinations.append(param_dict)
        
        # Limit combinations and add some randomization
        if len(combinations) > max_combinations:
            import random
            combinations = random.sample(combinations, max_combinations)
        
        return combinations
    
    def _update_config(self, base_config: RetrievalConfig, params: Dict[str, Any]) -> RetrievalConfig:
        """Update retrieval configuration with new parameters."""
        
        # Create a copy of the configuration
        updated_config = RetrievalConfig(
            dense_weight=params.get('dense_weight', base_config.dense_weight),
            sparse_weight=params.get('sparse_weight', base_config.sparse_weight),
            context_window_size=params.get('context_window_size', base_config.context_window_size),
            mmr_lambda=params.get('mmr_lambda', base_config.mmr_lambda),
            max_query_variations=params.get('max_query_variations', base_config.max_query_variations),
            cache_ttl_hours=params.get('cache_ttl_hours', base_config.cache_ttl_hours),
            enable_caching=base_config.enable_caching,
            parallel_retrieval=base_config.parallel_retrieval,
            max_workers=base_config.max_workers
        )
        
        return updated_config
    
    async def _evaluate_current_configuration(self, validation_queries: List[Tuple[str, List[str]]]) -> PerformanceMetrics:
        """Evaluate current configuration on validation queries."""
        
        if not self.retrieval_system:
            return PerformanceMetrics()
        
        total_precision = 0.0
        total_time = 0.0
        total_memory = 0.0
        successful_queries = 0
        
        for query, expected_docs in validation_queries:
            try:
                # Measure resource usage before
                memory_before = self.resource_monitor.get_memory_usage()
                
                # Execute query with timing
                start_time = time.time()
                results = await self.retrieval_system.aget_relevant_documents(query)
                end_time = time.time()
                
                # Measure resource usage after
                memory_after = self.resource_monitor.get_memory_usage()
                
                # Calculate metrics
                query_time = (end_time - start_time) * 1000  # Convert to ms
                memory_used = memory_after - memory_before
                
                # Calculate precision@5
                retrieved_ids = [doc.metadata.get('id', doc.page_content[:50]) for doc in results[:5]]
                relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in expected_docs)
                precision = relevant_retrieved / min(5, len(expected_docs)) if expected_docs else 0
                
                total_precision += precision
                total_time += query_time
                total_memory += memory_used
                successful_queries += 1
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                continue
        
        # Calculate average metrics
        if successful_queries > 0:
            avg_precision = total_precision / successful_queries
            avg_time = total_time / successful_queries
            avg_memory = total_memory / successful_queries
        else:
            avg_precision = 0.0
            avg_time = 1000.0  # High penalty for failed queries
            avg_memory = 0.0
        
        return PerformanceMetrics(
            total_time_ms=avg_time,
            relevance_score=avg_precision,
            memory_usage_mb=avg_memory,
            result_count=successful_queries,
            timestamp=datetime.now()
        )
    
    def _calculate_objective_score(self, metrics: PerformanceMetrics, strategy: OptimizationStrategy) -> float:
        """Calculate objective score based on optimization strategy."""
        
        if strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            # Optimize for speed (lower time is better)
            time_score = max(0, 1.0 - (metrics.total_time_ms / 2000))  # Normalize to 2 seconds
            return time_score * 0.8 + metrics.relevance_score * 0.2
        
        elif strategy == OptimizationStrategy.QUALITY_OPTIMIZED:
            # Optimize for quality
            return metrics.relevance_score * 0.8 + metrics.diversity_score * 0.2
        
        elif strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            # Optimize for memory usage
            memory_score = max(0, 1.0 - (metrics.memory_usage_mb / 1000))  # Normalize to 1GB
            return memory_score * 0.6 + metrics.relevance_score * 0.4
        
        else:  # BALANCED
            # Balance all factors
            time_score = max(0, 1.0 - (metrics.total_time_ms / 1500))
            memory_score = max(0, 1.0 - (metrics.memory_usage_mb / 500))
            
            return (
                metrics.relevance_score * 0.4 +
                time_score * 0.3 +
                memory_score * 0.2 +
                metrics.diversity_score * 0.1
            )
    
    async def benchmark_system_performance(self, 
                                         test_queries: List[str],
                                         concurrency_levels: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """Comprehensive system performance benchmarking."""
        
        benchmark_results = {
            'single_query_performance': {},
            'concurrency_performance': {},
            'resource_utilization': {},
            'system_limits': {}
        }
        
        # Single query performance
        if test_queries:
            single_query_metrics = await self._benchmark_single_queries(test_queries[:10])
            benchmark_results['single_query_performance'] = single_query_metrics
        
        # Concurrency performance
        for concurrency in concurrency_levels:
            if test_queries:
                concurrency_metrics = await self._benchmark_concurrent_queries(
                    test_queries[:concurrency], concurrency
                )
                benchmark_results['concurrency_performance'][f'level_{concurrency}'] = concurrency_metrics
        
        # Resource utilization analysis
        resource_metrics = self.resource_monitor.get_system_metrics()
        benchmark_results['resource_utilization'] = resource_metrics
        
        # System limits detection
        system_limits = await self._detect_system_limits(test_queries)
        benchmark_results['system_limits'] = system_limits
        
        return benchmark_results
    
    async def _benchmark_single_queries(self, queries: List[str]) -> Dict[str, float]:
        """Benchmark single query performance."""
        
        times = []
        memory_usage = []
        
        for query in queries:
            try:
                memory_before = self.resource_monitor.get_memory_usage()
                
                start_time = time.time()
                if self.retrieval_system:
                    await self.retrieval_system.aget_relevant_documents(query)
                end_time = time.time()
                
                memory_after = self.resource_monitor.get_memory_usage()
                
                times.append((end_time - start_time) * 1000)
                memory_usage.append(memory_after - memory_before)
                
            except Exception as e:
                logger.error(f"Error benchmarking query '{query}': {e}")
                continue
        
        return {
            'avg_response_time_ms': statistics.mean(times) if times else 0,
            'p95_response_time_ms': np.percentile(times, 95) if times else 0,
            'p99_response_time_ms': np.percentile(times, 99) if times else 0,
            'avg_memory_usage_mb': statistics.mean(memory_usage) if memory_usage else 0,
            'queries_tested': len(times)
        }
    
    async def _benchmark_concurrent_queries(self, queries: List[str], concurrency: int) -> Dict[str, float]:
        """Benchmark concurrent query performance."""
        
        if not self.retrieval_system:
            return {'error': 'Retrieval system not available'}
        
        start_time = time.time()
        memory_before = self.resource_monitor.get_memory_usage()
        
        # Execute queries concurrently
        tasks = [
            self.retrieval_system.aget_relevant_documents(query) 
            for query in queries[:concurrency]
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            memory_after = self.resource_monitor.get_memory_usage()
            
            # Count successful queries
            successful_queries = sum(1 for result in results if not isinstance(result, Exception))
            
            total_time = (end_time - start_time) * 1000
            memory_used = memory_after - memory_before
            
            return {
                'total_time_ms': total_time,
                'avg_time_per_query_ms': total_time / concurrency if concurrency > 0 else 0,
                'successful_queries': successful_queries,
                'success_rate': successful_queries / concurrency if concurrency > 0 else 0,
                'memory_usage_mb': memory_used,
                'throughput_qps': concurrency / (total_time / 1000) if total_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in concurrent benchmark: {e}")
            return {'error': str(e)}
    
    async def _detect_system_limits(self, test_queries: List[str]) -> Dict[str, Any]:
        """Detect system performance limits."""
        
        limits = {
            'max_concurrent_queries': 0,
            'memory_limit_mb': 0,
            'max_throughput_qps': 0,
            'breaking_point_detected': False
        }
        
        # Test increasing concurrency until failure
        concurrency = 1
        max_successful_concurrency = 0
        
        while concurrency <= 100 and not limits['breaking_point_detected']:
            try:
                test_batch = (test_queries * ((concurrency // len(test_queries)) + 1))[:concurrency]
                
                start_time = time.time()
                memory_before = self.resource_monitor.get_memory_usage()
                
                if self.retrieval_system:
                    tasks = [
                        self.retrieval_system.aget_relevant_documents(query) 
                        for query in test_batch
                    ]
                    
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=30  # 30 second timeout
                    )
                    
                    end_time = time.time()
                    memory_after = self.resource_monitor.get_memory_usage()
                    
                    # Check if most queries succeeded
                    successful = sum(1 for r in results if not isinstance(r, Exception))
                    success_rate = successful / concurrency
                    
                    if success_rate >= 0.8:  # 80% success rate threshold
                        max_successful_concurrency = concurrency
                        
                        # Calculate throughput
                        total_time = end_time - start_time
                        throughput = concurrency / total_time if total_time > 0 else 0
                        limits['max_throughput_qps'] = max(limits['max_throughput_qps'], throughput)
                        
                        # Track memory usage
                        memory_used = memory_after - memory_before
                        limits['memory_limit_mb'] = max(limits['memory_limit_mb'], memory_used)
                        
                    else:
                        limits['breaking_point_detected'] = True
                
                concurrency *= 2  # Double concurrency each iteration
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout at concurrency level {concurrency}")
                limits['breaking_point_detected'] = True
                break
            except Exception as e:
                logger.error(f"Error testing concurrency {concurrency}: {e}")
                limits['breaking_point_detected'] = True
                break
        
        limits['max_concurrent_queries'] = max_successful_concurrency
        
        return limits
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary and recommendations."""
        
        if not self.optimization_history:
            return {'message': 'No optimization runs completed yet'}
        
        latest_optimization = self.optimization_history[-1]
        
        summary = {
            'latest_optimization': {
                'performance_improvement': latest_optimization.performance_improvement,
                'quality_improvement': latest_optimization.quality_improvement,
                'speed_improvement': latest_optimization.speed_improvement,
                'iterations_completed': latest_optimization.iterations_completed,
                'optimization_time_minutes': latest_optimization.total_optimization_time_minutes
            },
            'optimal_configuration': {
                'dense_weight': latest_optimization.optimal_config.dense_weight,
                'sparse_weight': latest_optimization.optimal_config.sparse_weight,
                'mmr_lambda': latest_optimization.optimal_config.mmr_lambda,
                'context_window_size': latest_optimization.optimal_config.context_window_size,
                'max_query_variations': latest_optimization.optimal_config.max_query_variations,
                'cache_ttl_hours': latest_optimization.optimal_config.cache_ttl_hours
            },
            'system_health': {
                'connection_pool': self.connection_pool.get_pool_health_status(),
                'resource_utilization': self.resource_monitor.get_system_metrics()
            },
            'recommendations': self._generate_optimization_recommendations()
        }
        
        return summary
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate actionable optimization recommendations."""
        
        recommendations = []
        
        # Analyze performance history
        if len(self.performance_history) > 10:
            recent_metrics = self.performance_history[-10:]
            avg_response_time = statistics.mean([m.total_time_ms for m in recent_metrics])
            avg_memory_usage = statistics.mean([m.memory_usage_mb for m in recent_metrics])
            
            if avg_response_time > 1000:
                recommendations.append("Consider speed optimization: average response time is above 1 second")
            
            if avg_memory_usage > 500:
                recommendations.append("Consider memory optimization: high memory usage detected")
        
        # Check cache performance
        if hasattr(self.cache_optimizer, 'cache_system') and self.cache_optimizer.cache_system:
            cache_metrics = self.cache_optimizer.cache_system.get_performance_metrics()
            hit_rate = cache_metrics.get('hit_rate', 0)
            
            if hit_rate < 0.7:
                recommendations.append("Improve cache hit rate: consider cache warming or longer TTL")
        
        # Check connection pool health
        pool_health = self.connection_pool.get_pool_health_status()
        if pool_health['hit_rate'] < 0.9:
            recommendations.append("Connection pool optimization needed: low hit rate detected")
        
        if not recommendations:
            recommendations.append("System is performing well - no immediate optimizations needed")
        
        return recommendations


class ResourceMonitor:
    """Monitor system resource utilization during retrieval operations."""
    
    def __init__(self):
        """Initialize resource monitoring."""
        self.baseline_memory = psutil.virtual_memory().used
        self.baseline_cpu = psutil.cpu_percent()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.virtual_memory().used / (1024 * 1024)
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent()
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get comprehensive system metrics."""
        memory = psutil.virtual_memory()
        cpu_times = psutil.cpu_times()
        
        return {
            'memory_usage_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_total_mb': memory.total / (1024 * 1024),
            'cpu_usage_percent': psutil.cpu_percent(),
            'cpu_count': psutil.cpu_count(),
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_io_mb_sent': psutil.net_io_counters().bytes_sent / (1024 * 1024),
            'network_io_mb_recv': psutil.net_io_counters().bytes_recv / (1024 * 1024)
        }


# Factory function for easy initialization
def create_retrieval_optimizer(retrieval_system: Optional['ContextualRetrievalSystem'] = None) -> RetrievalOptimizer:
    """Create a fully configured retrieval optimizer."""
    
    connection_pool = ConnectionPoolManager(
        pool_size=20,
        max_overflow=10,
        pool_timeout=30
    )
    
    cache_optimizer = CacheOptimizer()
    
    optimizer = RetrievalOptimizer(
        retrieval_system=retrieval_system,
        connection_pool=connection_pool,
        cache_optimizer=cache_optimizer
    )
    
    logger.info("RetrievalOptimizer created with all optimization components")
    return optimizer


# Usage example
async def example_optimization_workflow():
    """Example workflow showing how to use the optimization system."""
    
    # This would be replaced with your actual retrieval system
    # retrieval_system = ContextualRetrievalSystem(...)
    
    optimizer = create_retrieval_optimizer()
    
    # Example validation queries
    validation_queries = [
        ("best casino games", ["doc1", "doc2", "doc3"]),
        ("how to play poker", ["doc4", "doc5"]),
        ("casino bonuses review", ["doc6", "doc7", "doc8"])
    ]
    
    # Optimize for balanced performance
    optimization_result = await optimizer.optimize_retrieval_parameters(
        validation_queries=validation_queries,
        optimization_strategy=OptimizationStrategy.BALANCED,
        max_iterations=30,
        target_improvement=0.15
    )
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    
    print(f"Optimization completed with {optimization_result.performance_improvement:.2%} improvement")
    print(f"Recommendations: {summary['recommendations']}")
    
    return optimization_result 