"""
Performance benchmarks for configuration and monitoring systems.
"""

import pytest
import asyncio
import time
import sys
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
import statistics

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.config.prompt_config import (
    PromptOptimizationConfig,
    ConfigurationManager,
    QueryType
)
from tests.fixtures.test_configs import (
    TestConfigFixtures,
    MockSupabaseClient,
    PerformanceTestData
)


class PerformanceBenchmarkSuite:
    """Performance benchmark suite for configuration and monitoring."""
    
    def __init__(self):
        self.results = {}
        self.baseline_times = {}
        
    def record_benchmark(self, test_name: str, duration: float, iterations: int = 1):
        """Record benchmark results."""
        if test_name not in self.results:
            self.results[test_name] = []
        self.results[test_name].append({
            'duration': duration,
            'iterations': iterations,
            'avg_per_iteration': duration / iterations if iterations > 0 else duration
        })
    
    def get_statistics(self, test_name: str) -> Dict[str, float]:
        """Get statistical summary of benchmark results."""
        if test_name not in self.results:
            return {}
        
        durations = [r['avg_per_iteration'] for r in self.results[test_name]]
        return {
            'min': min(durations),
            'max': max(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
            'count': len(durations)
        }
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n=== Performance Benchmark Results ===")
        for test_name in sorted(self.results.keys()):
            stats = self.get_statistics(test_name)
            print(f"\n{test_name}:")
            print(f"  Mean: {stats['mean']:.4f}s")
            print(f"  Median: {stats['median']:.4f}s")
            print(f"  Min: {stats['min']:.4f}s")
            print(f"  Max: {stats['max']:.4f}s")
            print(f"  StdDev: {stats['stdev']:.4f}s")
            print(f"  Samples: {stats['count']}")


@pytest.fixture(scope="module")
def benchmark_suite():
    """Create a benchmark suite for the module."""
    return PerformanceBenchmarkSuite()


class TestConfigurationPerformance:
    """Performance tests for configuration management."""
    
    @pytest.mark.asyncio
    async def test_config_loading_performance(self, benchmark_suite):
        """Benchmark configuration loading performance."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Add test configuration
        test_config = TestConfigFixtures.get_custom_config()
        mock_client.configs.append({
            "id": "perf_test",
            "config_data": test_config.to_dict(),
            "is_active": True
        })
        
        # Benchmark cold loading (first time, no cache)
        iterations = 50
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            # Clear cache to simulate cold load
            manager._cached_config = None
            manager._cache_timestamp = None
            config = await manager.get_active_config()
            assert config is not None
        
        cold_load_time = time.perf_counter() - start_time
        benchmark_suite.record_benchmark("config_cold_load", cold_load_time, iterations)
        
        # Benchmark warm loading (cached)
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            config = await manager.get_active_config()
            assert config is not None
        
        warm_load_time = time.perf_counter() - start_time
        benchmark_suite.record_benchmark("config_warm_load", warm_load_time, iterations)
        
        # Cached loading should be significantly faster
        avg_cold = cold_load_time / iterations
        avg_warm = warm_load_time / iterations
        
        assert avg_warm < avg_cold * 0.1, "Cached loading should be at least 10x faster"
        assert avg_warm < 0.001, "Cached loading should be under 1ms"
    
    @pytest.mark.asyncio
    async def test_config_validation_performance(self, benchmark_suite):
        """Benchmark configuration validation performance."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Test different configuration sizes
        configs = [
            ("small", TestConfigFixtures.get_default_config()),
            ("large", TestConfigFixtures.get_custom_config()),
        ]
        
        for config_type, config in configs:
            config_dict = config.to_dict()
            iterations = 100
            
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                result = await manager.validate_config(config_dict)
                assert result["is_valid"] is True
            
            validation_time = time.perf_counter() - start_time
            benchmark_suite.record_benchmark(f"config_validation_{config_type}", validation_time, iterations)
            
            # Validation should be fast
            avg_time = validation_time / iterations
            assert avg_time < 0.01, f"Config validation should be under 10ms, got {avg_time:.4f}s"
    
    @pytest.mark.asyncio
    async def test_config_serialization_performance(self, benchmark_suite):
        """Benchmark configuration serialization performance."""
        configs = [
            ("default", TestConfigFixtures.get_default_config()),
            ("custom", TestConfigFixtures.get_custom_config()),
        ]
        
        for config_type, config in configs:
            iterations = 1000
            
            # Benchmark to_dict performance
            start_time = time.perf_counter()
            for _ in range(iterations):
                config_dict = config.to_dict()
                assert isinstance(config_dict, dict)
            dict_time = time.perf_counter() - start_time
            
            # Benchmark from_dict performance
            config_dict = config.to_dict()
            start_time = time.perf_counter()
            for _ in range(iterations):
                restored_config = PromptOptimizationConfig.from_dict(config_dict)
                assert isinstance(restored_config, PromptOptimizationConfig)
            from_dict_time = time.perf_counter() - start_time
            
            # Benchmark hash generation
            start_time = time.perf_counter()
            for _ in range(iterations):
                config_hash = config.get_hash()
                assert len(config_hash) == 64
            hash_time = time.perf_counter() - start_time
            
            benchmark_suite.record_benchmark(f"config_to_dict_{config_type}", dict_time, iterations)
            benchmark_suite.record_benchmark(f"config_from_dict_{config_type}", from_dict_time, iterations)
            benchmark_suite.record_benchmark(f"config_hash_{config_type}", hash_time, iterations)
            
            # All serialization operations should be fast
            assert dict_time / iterations < 0.001, "to_dict should be under 1ms"
            assert from_dict_time / iterations < 0.001, "from_dict should be under 1ms"
            assert hash_time / iterations < 0.001, "get_hash should be under 1ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_config_access(self, benchmark_suite):
        """Benchmark concurrent configuration access."""
        mock_client = MockSupabaseClient()
        manager = ConfigurationManager(mock_client)
        
        # Add test configuration
        test_config = TestConfigFixtures.get_custom_config()
        mock_client.configs.append({
            "id": "concurrent_test",
            "config_data": test_config.to_dict(),
            "is_active": True
        })
        
        async def load_config():
            """Load configuration once."""
            return await manager.get_active_config()
        
        # Benchmark concurrent access
        concurrent_loads = 20
        start_time = time.perf_counter()
        
        # Run concurrent config loads
        tasks = [load_config() for _ in range(concurrent_loads)]
        results = await asyncio.gather(*tasks)
        
        concurrent_time = time.perf_counter() - start_time
        benchmark_suite.record_benchmark("config_concurrent_access", concurrent_time, concurrent_loads)
        
        # All results should be the same (cached)
        assert all(r.get_hash() == results[0].get_hash() for r in results)
        
        # Concurrent access should not be much slower than single access
        avg_time = concurrent_time / concurrent_loads
        assert avg_time < 0.01, f"Concurrent access should be fast, got {avg_time:.4f}s per access"


class TestMonitoringPerformance:
    """Performance tests for monitoring systems."""
    
    @pytest.mark.asyncio
    async def test_metrics_processing_performance(self, benchmark_suite):
        """Benchmark metrics processing performance."""
        sample_metrics = PerformanceTestData.get_query_metrics_sample()
        
        # Benchmark metrics aggregation
        iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            # Simulate metrics processing
            total_queries = len(sample_metrics)
            avg_latency = sum(m["total_latency_ms"] for m in sample_metrics) / total_queries
            cache_hit_rate = sum(1 for m in sample_metrics if m["cache_hit"]) / total_queries
            error_rate = sum(1 for m in sample_metrics if m["error_occurred"]) / total_queries
            
            assert total_queries > 0
            assert avg_latency > 0
            assert 0 <= cache_hit_rate <= 1
            assert 0 <= error_rate <= 1
        
        metrics_time = time.perf_counter() - start_time
        benchmark_suite.record_benchmark("metrics_processing", metrics_time, iterations)
        
        # Metrics processing should be fast
        avg_time = metrics_time / iterations
        assert avg_time < 0.01, f"Metrics processing should be under 10ms, got {avg_time:.4f}s"
    
    @pytest.mark.asyncio
    async def test_performance_profile_analysis(self, benchmark_suite):
        """Benchmark performance profile analysis."""
        sample_profiles = PerformanceTestData.get_performance_profiles_sample()
        
        iterations = 50
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            # Simulate profile analysis
            total_profiles = len(sample_profiles)
            
            # Calculate percentiles
            latencies = [p["llm_inference_ms"] for p in sample_profiles]
            latencies.sort()
            
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            
            # Calculate costs
            total_cost = sum(p["total_cost"] for p in sample_profiles)
            avg_cost = total_cost / total_profiles
            
            assert p50 > 0
            assert p95 >= p50
            assert p99 >= p95
            assert total_cost >= 0
            assert avg_cost >= 0
        
        analysis_time = time.perf_counter() - start_time
        benchmark_suite.record_benchmark("profile_analysis", analysis_time, iterations)
        
        # Profile analysis should be reasonably fast
        avg_time = analysis_time / iterations
        assert avg_time < 0.1, f"Profile analysis should be under 100ms, got {avg_time:.4f}s"
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, benchmark_suite):
        """Benchmark performance with large datasets."""
        # Generate larger datasets
        large_metrics = []
        for i in range(10000):  # 10k metrics
            base_time = datetime.utcnow() - timedelta(hours=i // 100)
            large_metrics.append({
                "id": f"large_metric_{i}",
                "query_text": f"Large test query {i}",
                "query_type": QueryType.CASINO_REVIEW.value,
                "total_latency_ms": 1000 + (i % 2000),
                "relevance_score": max(0.1, 0.9 - (i % 100) * 0.01),
                "confidence_score": max(0.1, 0.85 - (i % 100) * 0.005),
                "cache_hit": i % 3 == 0,
                "error_occurred": i % 50 == 49,
                "created_at": base_time
            })
        
        # Benchmark large dataset processing
        start_time = time.perf_counter()
        
        # Simulate real processing tasks
        total_queries = len(large_metrics)
        error_count = sum(1 for m in large_metrics if m["error_occurred"])
        cache_hits = sum(1 for m in large_metrics if m["cache_hit"])
        
        # Group by time periods (simulate time-series analysis)
        time_groups = {}
        for metric in large_metrics:
            hour_key = metric["created_at"].replace(minute=0, second=0, microsecond=0)
            if hour_key not in time_groups:
                time_groups[hour_key] = []
            time_groups[hour_key].append(metric)
        
        # Calculate hourly averages
        hourly_stats = {}
        for hour, metrics in time_groups.items():
            hourly_stats[hour] = {
                'avg_latency': sum(m["total_latency_ms"] for m in metrics) / len(metrics),
                'error_rate': sum(1 for m in metrics if m["error_occurred"]) / len(metrics),
                'cache_rate': sum(1 for m in metrics if m["cache_hit"]) / len(metrics)
            }
        
        large_dataset_time = time.perf_counter() - start_time
        benchmark_suite.record_benchmark("large_dataset_processing", large_dataset_time, 1)
        
        # Verify results
        assert total_queries == 10000
        assert error_count > 0
        assert cache_hits > 0
        assert len(time_groups) > 0
        assert len(hourly_stats) > 0
        
        # Large dataset processing should complete in reasonable time
        assert large_dataset_time < 10.0, f"Large dataset processing took {large_dataset_time:.2f}s, should be under 10s"


class TestMemoryPerformance:
    """Memory usage and performance tests."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_config(self, benchmark_suite):
        """Test memory usage of configuration objects."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many configuration objects
        configs = []
        for i in range(1000):
            if i % 2 == 0:
                config = TestConfigFixtures.get_default_config()
            else:
                config = TestConfigFixtures.get_custom_config()
            configs.append(config)
        
        # Measure memory after creating configs
        config_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_config = (config_memory - baseline_memory) / len(configs)
        
        # Each config should use reasonable memory
        assert memory_per_config < 0.1, f"Each config uses {memory_per_config:.3f}MB, should be under 0.1MB"
        
        # Test serialization memory efficiency
        serialized_configs = []
        for config in configs[:100]:  # Test subset
            serialized_configs.append(config.to_dict())
        
        serialized_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del configs
        del serialized_configs
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage - Baseline: {baseline_memory:.1f}MB, "
              f"With configs: {config_memory:.1f}MB, "
              f"Final: {final_memory:.1f}MB")


@pytest.mark.asyncio
async def test_benchmark_summary(benchmark_suite):
    """Print final benchmark summary."""
    # This test runs last and prints the summary
    benchmark_suite.print_summary()
    
    # Verify we have some benchmark results
    assert len(benchmark_suite.results) > 0, "No benchmark results recorded"
    
    # Verify all benchmarks meet performance criteria
    for test_name, results in benchmark_suite.results.items():
        stats = benchmark_suite.get_statistics(test_name)
        
        # Set performance thresholds based on test type
        if "cold_load" in test_name:
            threshold = 0.1  # 100ms for cold load
        elif "warm_load" in test_name:
            threshold = 0.001  # 1ms for warm load
        elif "validation" in test_name:
            threshold = 0.01  # 10ms for validation
        elif "serialization" in test_name or "hash" in test_name:
            threshold = 0.001  # 1ms for serialization
        elif "large_dataset" in test_name:
            threshold = 10.0  # 10s for large dataset
        else:
            threshold = 1.0  # 1s default threshold
        
        assert stats['mean'] < threshold, \
            f"Performance regression: {test_name} took {stats['mean']:.4f}s, " \
            f"should be under {threshold}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 