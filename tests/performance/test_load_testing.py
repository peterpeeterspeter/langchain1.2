"""
Load Testing Module for Performance Benchmark Suite

This module provides specialized load testing capabilities including:
- Concurrent user simulation
- Stress testing with various load patterns
- System breaking point identification
- Resource exhaustion testing
- Throughput and latency analysis under load
"""

import asyncio
import time
import statistics
import psutil
import pytest
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import random

logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 0
    requests_per_user: Optional[int] = None
    think_time_ms: int = 100

@dataclass
class LoadTestResult:
    """Results from a load test execution."""
    config_name: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    throughput_qps: float
    error_rate: float
    memory_peak_mb: float
    cpu_peak_percent: float

class LoadTestingFramework:
    """Advanced load testing framework for performance validation."""
    
    def __init__(self):
        self.test_queries = [
            "What are the best online casinos for high rollers?",
            "How to calculate blackjack basic strategy?",
            "Casino welcome bonus comparison guide",
            "Progressive slot machine jackpot analysis",
            "Live dealer baccarat rules and strategies",
            "Online poker tournament selection tips",
            "Roulette wheel bias detection methods",
            "Mobile casino app security features",
            "Cryptocurrency casino deposit methods",
            "Responsible gambling limit setting tools"
        ]
        
        self.load_test_scenarios = [
            LoadTestConfig("baseline", 1, 30, 0),
            LoadTestConfig("light_load", 5, 60, 10),
            LoadTestConfig("moderate_load", 15, 120, 30),
            LoadTestConfig("heavy_load", 30, 180, 60),
            LoadTestConfig("stress_test", 50, 300, 120)
        ]
    
    async def run_load_test_suite(self) -> Dict[str, Any]:
        """Execute comprehensive load testing suite."""
        logger.info("🚀 Starting Load Testing Suite")
        
        suite_results = {
            'suite_start_time': datetime.now().isoformat(),
            'test_results': [],
            'system_limits': {},
            'recommendations': []
        }
        
        # Execute each load test scenario
        for config in self.load_test_scenarios:
            logger.info(f"🔄 Running load test: {config.name}")
            
            try:
                result = await self._execute_load_test(config)
                suite_results['test_results'].append(result)
                
                logger.info(f"   ✅ {config.name}: {result.throughput_qps:.1f} QPS, "
                          f"{result.avg_response_time_ms:.0f}ms avg, "
                          f"{result.error_rate:.1%} errors")
                
            except Exception as e:
                logger.error(f"   ❌ {config.name} failed: {str(e)}")
        
        # Analyze results
        suite_results['system_limits'] = self._analyze_system_limits(suite_results['test_results'])
        suite_results['recommendations'] = self._generate_recommendations(suite_results)
        
        logger.info("✅ Load Testing Suite Completed")
        return suite_results
    
    async def _execute_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Execute a single load test scenario."""
        
        start_time = datetime.now()
        requests_completed = []
        errors = []
        response_times = []
        
        # Resource monitoring
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        peak_cpu = 0.0
        
        # Create user simulation tasks
        user_tasks = []
        
        for i in range(config.concurrent_users):
            delay = (i * config.ramp_up_seconds / config.concurrent_users) if config.ramp_up_seconds > 0 else 0
            task = asyncio.create_task(
                self._simulate_user_session(config, delay, requests_completed, errors, response_times)
            )
            user_tasks.append(task)
        
        # Monitor resources during test
        async def monitor_resources():
            nonlocal peak_memory, peak_cpu
            end_time = time.time() + config.duration_seconds + config.ramp_up_seconds
            
            while time.time() < end_time:
                try:
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    current_cpu = process.cpu_percent()
                    
                    peak_memory = max(peak_memory, current_memory)
                    peak_cpu = max(peak_cpu, current_cpu)
                    
                    await asyncio.sleep(1)
                except Exception:
                    await asyncio.sleep(1)
        
        monitoring_task = asyncio.create_task(monitor_resources())
        
        # Wait for test completion
        await asyncio.sleep(config.duration_seconds + config.ramp_up_seconds)
        
        # Cancel tasks
        for task in user_tasks:
            if not task.done():
                task.cancel()
        monitoring_task.cancel()
        
        await asyncio.gather(*user_tasks, monitoring_task, return_exceptions=True)
        
        # Calculate results
        total_requests = len(requests_completed) + len(errors)
        successful_requests = len(requests_completed)
        failed_requests = len(errors)
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        duration_seconds = (datetime.now() - start_time).total_seconds()
        throughput_qps = successful_requests / duration_seconds if duration_seconds > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        return LoadTestResult(
            config_name=config.name,
            concurrent_users=config.concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            max_response_time_ms=max_response_time,
            min_response_time_ms=min_response_time,
            throughput_qps=throughput_qps,
            error_rate=error_rate,
            memory_peak_mb=peak_memory,
            cpu_peak_percent=peak_cpu
        )
    
    async def _simulate_user_session(self, config: LoadTestConfig, start_delay: float, 
                                   requests_completed: List, errors: List, response_times: List):
        """Simulate a single user session."""
        
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        
        session_start = time.time()
        session_end = session_start + config.duration_seconds
        request_count = 0
        
        while time.time() < session_end:
            if config.requests_per_user and request_count >= config.requests_per_user:
                break
            
            try:
                query = random.choice(self.test_queries)
                
                request_start = time.time()
                result = await self._simulate_request(query)
                request_end = time.time()
                
                response_time_ms = (request_end - request_start) * 1000
                response_times.append(response_time_ms)
                requests_completed.append({
                    'timestamp': request_end,
                    'response_time_ms': response_time_ms,
                    'query': query
                })
                
                request_count += 1
                
            except Exception as e:
                errors.append({
                    'timestamp': time.time(),
                    'error': str(e)
                })
            
            if config.think_time_ms > 0:
                await asyncio.sleep(config.think_time_ms / 1000)
    
    async def _simulate_request(self, query: str) -> Dict[str, Any]:
        """Simulate a single request with realistic processing."""
        
        base_time = 0.2  # 200ms base
        complexity_factor = len(query) / 100
        processing_time = base_time + (complexity_factor * 0.1)
        
        processing_time += random.uniform(-0.05, 0.15)
        processing_time = max(0.05, processing_time)
        
        if random.random() < 0.05:  # 5% slow queries
            processing_time *= 3
        
        await asyncio.sleep(processing_time)
        
        success = random.random() > 0.02  # 2% failure rate
        
        return {
            'success': success,
            'processing_time_ms': processing_time * 1000,
            'query': query
        }
    
    def _analyze_system_limits(self, test_results: List[LoadTestResult]) -> Dict[str, Any]:
        """Analyze system performance limits."""
        
        if not test_results:
            return {'status': 'no_results'}
        
        # Find maximum stable throughput
        stable_results = [r for r in test_results if r.error_rate < 0.05]
        max_stable_throughput = max([r.throughput_qps for r in stable_results]) if stable_results else 0
        
        # Find breaking point
        breaking_point = None
        for result in sorted(test_results, key=lambda x: x.concurrent_users):
            if result.error_rate > 0.1 or result.avg_response_time_ms > 5000:
                breaking_point = {
                    'concurrent_users': result.concurrent_users,
                    'error_rate': result.error_rate,
                    'avg_response_time_ms': result.avg_response_time_ms
                }
                break
        
        max_memory = max([r.memory_peak_mb for r in test_results])
        max_cpu = max([r.cpu_peak_percent for r in test_results])
        
        return {
            'max_stable_throughput_qps': max_stable_throughput,
            'breaking_point': breaking_point,
            'resource_limits': {
                'peak_memory_mb': max_memory,
                'peak_cpu_percent': max_cpu
            },
            'recommended_max_users': max([r.concurrent_users for r in stable_results]) if stable_results else 1
        }
    
    def _generate_recommendations(self, suite_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        
        system_limits = suite_results.get('system_limits', {})
        
        max_throughput = system_limits.get('max_stable_throughput_qps', 0)
        if max_throughput < 10:
            recommendations.append(f"Low maximum throughput ({max_throughput:.1f} QPS). Consider optimizing query processing.")
        
        breaking_point = system_limits.get('breaking_point')
        if breaking_point:
            recommendations.append(f"System breaks at {breaking_point['concurrent_users']} users. Implement better error handling.")
        
        peak_memory = system_limits.get('resource_limits', {}).get('peak_memory_mb', 0)
        if peak_memory > 500:
            recommendations.append(f"High memory usage ({peak_memory:.0f}MB). Implement memory optimization.")
        
        if not recommendations:
            recommendations.append("Load testing shows good performance characteristics.")
        
        return recommendations

@pytest.mark.performance
@pytest.mark.asyncio
class TestLoadTesting:
    """Pytest integration for load testing."""
    
    @pytest.fixture(scope="class")
    def load_testing_framework(self):
        return LoadTestingFramework()
    
    async def test_baseline_load(self, load_testing_framework):
        """Test baseline single-user performance."""
        config = LoadTestConfig("baseline_test", 1, 30, 0)
        result = await load_testing_framework._execute_load_test(config)
        
        assert result.total_requests > 0
        assert result.error_rate < 0.05
        assert result.avg_response_time_ms < 2000
        
        logger.info(f"✅ Baseline: {result.throughput_qps:.1f} QPS, {result.avg_response_time_ms:.0f}ms")
    
    async def test_comprehensive_load_suite(self, load_testing_framework):
        """Test the complete load testing suite."""
        suite_results = await load_testing_framework.run_load_test_suite()
        
        assert 'test_results' in suite_results
        assert 'system_limits' in suite_results
        assert len(suite_results['recommendations']) > 0
        
        logger.info("✅ Load testing suite completed")

if __name__ == "__main__":
    async def main():
        framework = LoadTestingFramework()
        results = await framework.run_load_test_suite()
        
        print("\n" + "="*60)
        print("LOAD TESTING RESULTS")
        print("="*60)
        
        system_limits = results['system_limits']
        print(f"Max Stable Throughput: {system_limits.get('max_stable_throughput_qps', 0):.1f} QPS")
        print(f"Recommended Max Users: {system_limits.get('recommended_max_users', 'Unknown')}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("="*60)
    
    asyncio.run(main()) 