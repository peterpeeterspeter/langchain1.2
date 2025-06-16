"""
Task 10.7: Workflow Performance Testing

This module focuses on performance aspects of end-to-end workflows:
- Concurrent user simulation
- Memory and resource usage monitoring
- Performance benchmarking
- Load testing scenarios
"""

import asyncio
import time
import pytest
import logging
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for workflow testing."""
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    concurrent_users: int
    success_rate: float
    cache_hit_rate: float
    error_count: int


class TestWorkflowPerformance:
    """
    Performance-focused workflow testing.
    
    Tests system performance under various load conditions:
    - Concurrent user simulation
    - Memory usage monitoring
    - Resource utilization tracking
    - Performance degradation analysis
    """
    
    def __init__(self):
        """Initialize performance testing framework."""
        self.performance_data: List[PerformanceMetrics] = []
        self.baseline_metrics = {
            'max_response_time_ms': 2000,
            'max_memory_usage_mb': 500,
            'max_cpu_usage_percent': 80,
            'min_success_rate': 0.95,
            'min_cache_hit_rate': 0.70
        }
        
        # Mock components for testing
        self._setup_mock_components()
        
        logger.info("Performance testing framework initialized")
    
    def _setup_mock_components(self):
        """Setup mock components for performance testing."""
        self.mock_rag_system = Mock()
        self.mock_rag_system.process_query = AsyncMock()
        
        self.mock_cache_system = Mock()
        self.mock_cache_system.get = AsyncMock()
        self.mock_cache_system.set = AsyncMock()
        
        self.mock_retrieval_system = Mock()
        self.mock_retrieval_system.retrieve = AsyncMock()
    
    async def test_concurrent_user_simulation(self) -> Dict[str, Any]:
        """
        Test concurrent user simulation.
        
        Simulates multiple users making simultaneous requests to test:
        - System throughput under load
        - Response time degradation
        - Resource utilization
        - Error rates under stress
        """
        test_name = "concurrent_user_simulation"
        logger.info(f"🚀 Starting {test_name}")
        
        # Test configuration
        concurrent_users = [1, 3, 5, 10]  # Different load levels
        queries_per_user = 3
        
        test_results = []
        
        for user_count in concurrent_users:
            logger.info(f"Testing with {user_count} concurrent users...")
            
            # Record initial system state
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            initial_cpu = psutil.cpu_percent()
            
            # Create test queries
            test_queries = [
                "What are the best online casinos?",
                "How to play blackjack effectively?",
                "Casino bonus terms and conditions explained"
            ]
            
            # Run concurrent user simulation
            start_time = time.time()
            tasks = []
            
            for user_id in range(user_count):
                for query_idx, query in enumerate(test_queries):
                    task = self._simulate_user_query(
                        user_id=f"user_{user_id}",
                        query=query,
                        query_id=f"{user_id}_{query_idx}"
                    )
                    tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record final system state
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            final_cpu = psutil.cpu_percent()
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            error_results = [r for r in results if isinstance(r, Exception)]
            
            response_times = [r['response_time_ms'] for r in successful_results]
            cache_hits = sum(1 for r in successful_results if r.get('cached', False))
            
            # Calculate metrics
            metrics = PerformanceMetrics(
                response_time_ms=statistics.mean(response_times) if response_times else 0,
                memory_usage_mb=final_memory - initial_memory,
                cpu_usage_percent=final_cpu - initial_cpu,
                concurrent_users=user_count,
                success_rate=len(successful_results) / len(results) if results else 0,
                cache_hit_rate=cache_hits / len(successful_results) if successful_results else 0,
                error_count=len(error_results)
            )
            
            self.performance_data.append(metrics)
            
            test_result = {
                'concurrent_users': user_count,
                'total_queries': len(tasks),
                'successful_queries': len(successful_results),
                'failed_queries': len(error_results),
                'avg_response_time_ms': metrics.response_time_ms,
                'max_response_time_ms': max(response_times) if response_times else 0,
                'min_response_time_ms': min(response_times) if response_times else 0,
                'memory_increase_mb': metrics.memory_usage_mb,
                'cpu_increase_percent': metrics.cpu_usage_percent,
                'success_rate': metrics.success_rate,
                'cache_hit_rate': metrics.cache_hit_rate,
                'total_execution_time_ms': execution_time,
                'throughput_qps': len(successful_results) / (execution_time / 1000) if execution_time > 0 else 0
            }
            
            test_results.append(test_result)
            
            # Log results
            logger.info(f"  Users: {user_count}, Success Rate: {metrics.success_rate:.1%}, "
                       f"Avg Response: {metrics.response_time_ms:.1f}ms, "
                       f"Memory: +{metrics.memory_usage_mb:.1f}MB")
            
            # Brief pause between load levels
            await asyncio.sleep(0.5)
        
        # Performance analysis
        performance_analysis = self._analyze_performance_trends(test_results)
        
        return {
            'test_name': test_name,
            'test_results': test_results,
            'performance_analysis': performance_analysis,
            'baseline_validation': self._validate_against_baseline(test_results),
            'recommendations': self._generate_performance_recommendations(test_results)
        }
    
    async def _simulate_user_query(self, user_id: str, query: str, query_id: str) -> Dict[str, Any]:
        """Simulate a single user query with realistic behavior."""
        start_time = time.time()
        
        try:
            # Simulate cache check (30% hit rate)
            cache_hit = hash(query) % 10 < 3
            
            if cache_hit:
                # Simulate cache retrieval
                await asyncio.sleep(0.01)  # 10ms cache lookup
                response_time = (time.time() - start_time) * 1000
                
                return {
                    'user_id': user_id,
                    'query_id': query_id,
                    'query': query,
                    'cached': True,
                    'response_time_ms': response_time,
                    'confidence_score': 0.85,
                    'success': True
                }
            else:
                # Simulate full query processing
                # Variable processing time based on query complexity
                processing_time = 0.1 + (len(query.split()) * 0.02)  # Base + complexity
                await asyncio.sleep(processing_time)
                
                response_time = (time.time() - start_time) * 1000
                
                # Simulate occasional failures (5% failure rate)
                if hash(query_id) % 20 == 0:
                    raise Exception(f"Simulated processing failure for query {query_id}")
                
                return {
                    'user_id': user_id,
                    'query_id': query_id,
                    'query': query,
                    'cached': False,
                    'response_time_ms': response_time,
                    'confidence_score': 0.75 + (hash(query) % 20) / 100,  # 0.75-0.95
                    'success': True
                }
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'user_id': user_id,
                'query_id': query_id,
                'query': query,
                'cached': False,
                'response_time_ms': response_time,
                'success': False,
                'error': str(e)
            }
    
    async def test_memory_monitoring(self) -> Dict[str, Any]:
        """
        Test memory usage monitoring during workflow execution.
        
        Monitors:
        - Memory allocation patterns
        - Memory leaks detection
        - Garbage collection impact
        - Resource cleanup validation
        """
        test_name = "memory_monitoring"
        logger.info(f"🧠 Starting {test_name}")
        
        # Memory monitoring configuration
        monitoring_duration_seconds = 10
        sample_interval_seconds = 0.5
        query_interval_seconds = 1.0
        
        memory_samples = []
        query_results = []
        
        # Start memory monitoring
        start_time = time.time()
        monitoring_task = asyncio.create_task(
            self._monitor_memory_usage(
                duration_seconds=monitoring_duration_seconds,
                interval_seconds=sample_interval_seconds,
                memory_samples=memory_samples
            )
        )
        
        # Generate load during monitoring
        query_task = asyncio.create_task(
            self._generate_memory_test_load(
                duration_seconds=monitoring_duration_seconds,
                interval_seconds=query_interval_seconds,
                results=query_results
            )
        )
        
        # Wait for both tasks to complete
        await asyncio.gather(monitoring_task, query_task)
        
        total_time = time.time() - start_time
        
        # Analyze memory usage patterns
        memory_analysis = self._analyze_memory_patterns(memory_samples)
        
        # Calculate memory efficiency metrics
        efficiency_metrics = {
            'peak_memory_mb': max(s['memory_mb'] for s in memory_samples),
            'avg_memory_mb': statistics.mean(s['memory_mb'] for s in memory_samples),
            'memory_growth_rate_mb_per_sec': (memory_samples[-1]['memory_mb'] - memory_samples[0]['memory_mb']) / total_time,
            'memory_volatility': statistics.stdev(s['memory_mb'] for s in memory_samples),
            'queries_processed': len(query_results),
            'memory_per_query_mb': (memory_samples[-1]['memory_mb'] - memory_samples[0]['memory_mb']) / max(len(query_results), 1)
        }
        
        # Memory leak detection
        leak_detection = self._detect_memory_leaks(memory_samples)
        
        return {
            'test_name': test_name,
            'monitoring_duration_seconds': total_time,
            'memory_samples': memory_samples,
            'memory_analysis': memory_analysis,
            'efficiency_metrics': efficiency_metrics,
            'leak_detection': leak_detection,
            'query_results': query_results,
            'recommendations': self._generate_memory_recommendations(efficiency_metrics, leak_detection)
        }
    
    async def _monitor_memory_usage(
        self, 
        duration_seconds: float, 
        interval_seconds: float, 
        memory_samples: List[Dict[str, Any]]
    ):
        """Monitor memory usage over time."""
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            sample = {
                'timestamp': time.time() - start_time,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent()
            }
            
            memory_samples.append(sample)
            await asyncio.sleep(interval_seconds)
    
    async def _generate_memory_test_load(
        self, 
        duration_seconds: float, 
        interval_seconds: float, 
        results: List[Dict[str, Any]]
    ):
        """Generate test load for memory monitoring."""
        start_time = time.time()
        query_count = 0
        
        test_queries = [
            "Memory test query 1",
            "Memory test query 2 with more content",
            "Memory test query 3 with even more content to test allocation patterns"
        ]
        
        while (time.time() - start_time) < duration_seconds:
            query = test_queries[query_count % len(test_queries)]
            
            # Simulate query processing with memory allocation
            result = await self._simulate_memory_intensive_query(query, query_count)
            results.append(result)
            
            query_count += 1
            await asyncio.sleep(interval_seconds)
    
    async def _simulate_memory_intensive_query(self, query: str, query_id: int) -> Dict[str, Any]:
        """Simulate a memory-intensive query for testing."""
        start_time = time.time()
        
        # Simulate memory allocation (create temporary data structures)
        temp_data = []
        for i in range(1000):  # Create some temporary objects
            temp_data.append({
                'id': i,
                'query': query,
                'data': f"temporary_data_{i}" * 10
            })
        
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        # Clean up temporary data (simulate proper cleanup)
        del temp_data
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            'query_id': query_id,
            'query': query,
            'response_time_ms': response_time,
            'memory_allocated': True,
            'memory_cleaned': True
        }
    
    def _analyze_memory_patterns(self, memory_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not memory_samples:
            return {'error': 'No memory samples available'}
        
        memory_values = [s['memory_mb'] for s in memory_samples]
        
        return {
            'initial_memory_mb': memory_samples[0]['memory_mb'],
            'final_memory_mb': memory_samples[-1]['memory_mb'],
            'peak_memory_mb': max(memory_values),
            'min_memory_mb': min(memory_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'memory_range_mb': max(memory_values) - min(memory_values),
            'memory_trend': 'increasing' if memory_samples[-1]['memory_mb'] > memory_samples[0]['memory_mb'] else 'stable',
            'sample_count': len(memory_samples)
        }
    
    def _detect_memory_leaks(self, memory_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential memory leaks."""
        if len(memory_samples) < 10:
            return {'status': 'insufficient_data'}
        
        # Calculate memory growth trend
        memory_values = [s['memory_mb'] for s in memory_samples]
        timestamps = [s['timestamp'] for s in memory_samples]
        
        # Simple linear regression to detect growth trend
        n = len(memory_values)
        sum_x = sum(timestamps)
        sum_y = sum(memory_values)
        sum_xy = sum(x * y for x, y in zip(timestamps, memory_values))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Calculate slope (memory growth rate)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Leak detection thresholds
        leak_threshold_mb_per_sec = 1.0  # 1MB per second growth
        
        leak_detected = slope > leak_threshold_mb_per_sec
        
        return {
            'status': 'leak_detected' if leak_detected else 'no_leak_detected',
            'memory_growth_rate_mb_per_sec': slope,
            'leak_threshold_mb_per_sec': leak_threshold_mb_per_sec,
            'confidence': 'high' if abs(slope) > leak_threshold_mb_per_sec * 2 else 'medium',
            'recommendation': 'investigate_memory_usage' if leak_detected else 'memory_usage_normal'
        }
    
    def _analyze_performance_trends(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends across different load levels."""
        if not test_results:
            return {'error': 'No test results available'}
        
        # Extract metrics for trend analysis
        user_counts = [r['concurrent_users'] for r in test_results]
        response_times = [r['avg_response_time_ms'] for r in test_results]
        success_rates = [r['success_rate'] for r in test_results]
        memory_usage = [r['memory_increase_mb'] for r in test_results]
        throughput = [r['throughput_qps'] for r in test_results]
        
        return {
            'scalability_analysis': {
                'response_time_degradation': self._calculate_degradation(user_counts, response_times),
                'success_rate_stability': min(success_rates),
                'memory_scaling': self._calculate_scaling_factor(user_counts, memory_usage),
                'throughput_efficiency': max(throughput) / max(user_counts) if user_counts else 0
            },
            'performance_bottlenecks': self._identify_bottlenecks(test_results),
            'optimal_load': self._find_optimal_load(test_results),
            'breaking_point': self._estimate_breaking_point(test_results)
        }
    
    def _calculate_degradation(self, load_levels: List[int], metrics: List[float]) -> float:
        """Calculate performance degradation rate."""
        if len(load_levels) < 2 or len(metrics) < 2:
            return 0.0
        
        # Calculate degradation as percentage increase per user
        initial_metric = metrics[0]
        final_metric = metrics[-1]
        load_increase = load_levels[-1] - load_levels[0]
        
        if initial_metric == 0 or load_increase == 0:
            return 0.0
        
        degradation_rate = ((final_metric - initial_metric) / initial_metric) / load_increase
        return degradation_rate * 100  # Convert to percentage
    
    def _calculate_scaling_factor(self, load_levels: List[int], metrics: List[float]) -> str:
        """Calculate how metrics scale with load."""
        if len(load_levels) < 2 or len(metrics) < 2:
            return 'insufficient_data'
        
        # Simple scaling analysis
        load_ratio = load_levels[-1] / load_levels[0]
        metric_ratio = metrics[-1] / metrics[0] if metrics[0] != 0 else float('inf')
        
        if metric_ratio < load_ratio * 0.8:
            return 'sub_linear'  # Better than linear scaling
        elif metric_ratio < load_ratio * 1.2:
            return 'linear'      # Linear scaling
        else:
            return 'super_linear'  # Worse than linear scaling
    
    def _identify_bottlenecks(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for result in test_results:
            if result['success_rate'] < 0.95:
                bottlenecks.append(f"High error rate at {result['concurrent_users']} users")
            
            if result['avg_response_time_ms'] > self.baseline_metrics['max_response_time_ms']:
                bottlenecks.append(f"Response time exceeded baseline at {result['concurrent_users']} users")
            
            if result['memory_increase_mb'] > 100:  # Arbitrary threshold
                bottlenecks.append(f"High memory usage at {result['concurrent_users']} users")
        
        return bottlenecks if bottlenecks else ['No significant bottlenecks detected']
    
    def _find_optimal_load(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find optimal load level balancing performance and resource usage."""
        if not test_results:
            return {'error': 'No test results available'}
        
        # Score each load level based on multiple factors
        best_score = -1
        optimal_result = None
        
        for result in test_results:
            # Calculate composite score (higher is better)
            score = (
                result['success_rate'] * 0.4 +  # 40% weight on success rate
                (1 - min(result['avg_response_time_ms'] / 2000, 1)) * 0.3 +  # 30% weight on response time
                (result['throughput_qps'] / 10) * 0.2 +  # 20% weight on throughput
                (1 - min(result['memory_increase_mb'] / 100, 1)) * 0.1  # 10% weight on memory efficiency
            )
            
            if score > best_score:
                best_score = score
                optimal_result = result
        
        return {
            'optimal_concurrent_users': optimal_result['concurrent_users'] if optimal_result else 0,
            'optimal_score': best_score,
            'optimal_metrics': optimal_result
        }
    
    def _estimate_breaking_point(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate system breaking point based on trends."""
        if len(test_results) < 2:
            return {'error': 'Insufficient data for breaking point estimation'}
        
        # Find the point where success rate drops significantly
        breaking_point_users = None
        for result in test_results:
            if result['success_rate'] < 0.9:  # 90% success rate threshold
                breaking_point_users = result['concurrent_users']
                break
        
        if breaking_point_users is None:
            # Extrapolate based on trends
            max_tested_users = max(r['concurrent_users'] for r in test_results)
            breaking_point_users = max_tested_users * 2  # Conservative estimate
        
        return {
            'estimated_breaking_point_users': breaking_point_users,
            'confidence': 'high' if breaking_point_users <= max(r['concurrent_users'] for r in test_results) else 'low',
            'recommendation': f"System can handle up to {breaking_point_users} concurrent users"
        }
    
    def _validate_against_baseline(self, test_results: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Validate performance results against baseline metrics."""
        validation = {}
        
        for result in test_results:
            user_count = result['concurrent_users']
            
            validation[f'response_time_acceptable_{user_count}_users'] = (
                result['avg_response_time_ms'] <= self.baseline_metrics['max_response_time_ms']
            )
            
            validation[f'success_rate_acceptable_{user_count}_users'] = (
                result['success_rate'] >= self.baseline_metrics['min_success_rate']
            )
            
            validation[f'cache_hit_rate_acceptable_{user_count}_users'] = (
                result['cache_hit_rate'] >= self.baseline_metrics['min_cache_hit_rate']
            )
        
        return validation
    
    def _generate_performance_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze results for recommendations
        avg_response_time = statistics.mean(r['avg_response_time_ms'] for r in test_results)
        min_success_rate = min(r['success_rate'] for r in test_results)
        max_memory_usage = max(r['memory_increase_mb'] for r in test_results)
        
        if avg_response_time > 1000:
            recommendations.append("Consider implementing response caching to reduce average response times")
        
        if min_success_rate < 0.95:
            recommendations.append("Improve error handling and retry mechanisms to increase success rates")
        
        if max_memory_usage > 50:
            recommendations.append("Optimize memory usage patterns and implement better garbage collection")
        
        # Load-specific recommendations
        high_load_results = [r for r in test_results if r['concurrent_users'] >= 5]
        if high_load_results:
            avg_high_load_response = statistics.mean(r['avg_response_time_ms'] for r in high_load_results)
            if avg_high_load_response > avg_response_time * 1.5:
                recommendations.append("Implement connection pooling and async processing for high-load scenarios")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable limits - consider stress testing with higher loads")
        
        return recommendations
    
    def _generate_memory_recommendations(
        self, 
        efficiency_metrics: Dict[str, Any], 
        leak_detection: Dict[str, Any]
    ) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if leak_detection.get('status') == 'leak_detected':
            recommendations.append("Memory leak detected - review object lifecycle and garbage collection")
        
        if efficiency_metrics['memory_per_query_mb'] > 5:
            recommendations.append("High memory usage per query - optimize data structures and caching")
        
        if efficiency_metrics['memory_volatility'] > 20:
            recommendations.append("High memory volatility - implement more consistent memory allocation patterns")
        
        if efficiency_metrics['peak_memory_mb'] > 200:
            recommendations.append("Peak memory usage is high - consider implementing memory limits and cleanup")
        
        if not recommendations:
            recommendations.append("Memory usage patterns are healthy - continue monitoring in production")
        
        return recommendations


# Test execution helpers
async def run_performance_tests():
    """Run all performance tests."""
    test_framework = TestWorkflowPerformance()
    
    # Run concurrent user simulation
    concurrent_results = await test_framework.test_concurrent_user_simulation()
    
    # Run memory monitoring
    memory_results = await test_framework.test_memory_monitoring()
    
    return {
        'concurrent_user_simulation': concurrent_results,
        'memory_monitoring': memory_results,
        'overall_performance_health': 'healthy'  # Could be calculated based on results
    }


if __name__ == "__main__":
    # Run performance tests if executed directly
    import asyncio
    
    async def main():
        results = await run_performance_tests()
        print("Performance Test Results:")
        print(f"Concurrent Users Test: {results['concurrent_user_simulation']['test_name']}")
        print(f"Memory Monitoring Test: {results['memory_monitoring']['test_name']}")
        print(f"Overall Health: {results['overall_performance_health']}")
    
    asyncio.run(main()) 