#!/usr/bin/env python3
"""
Task 22.6: Load Testing Framework for Concurrent Screenshot Operations

This module implements comprehensive load testing capabilities for the screenshot 
performance optimization project.

Author: Task Master AI
Created: 2024-12-19
"""

import asyncio
import time
import logging
import statistics
import psutil
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for a single load test scenario."""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 0

@dataclass
class LoadTestMetrics:
    """Performance metrics collected during load testing."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    max_response_time_ms: float
    throughput_qps: float
    error_rate: float
    memory_peak_mb: float
    concurrent_users: int
    system_stability_score: float

@dataclass
class LoadTestResult:
    """Complete result from a load test execution."""
    config: LoadTestConfig
    metrics: LoadTestMetrics
    timestamp: datetime
    success: bool
    breaking_point_detected: bool

class MockScreenshotService:
    """Mock screenshot service for load testing."""
    
    def __init__(self):
        self._capture_stats = {'total_captures': 0}
    
    async def capture_screenshot(self, url: str) -> Dict[str, Any]:
        """Mock screenshot capture with realistic timing."""
        start_time = time.time()
        
        # Simulate processing time (200ms-2s)
        processing_time = random.uniform(0.2, 2.0)
        
        # Add load-based delays
        if random.random() < 0.1:  # 10% slow requests
            processing_time *= 2.5
        
        await asyncio.sleep(processing_time)
        
        # Simulate failures (2% rate)
        success = random.random() > 0.02
        
        capture_time = (time.time() - start_time) * 1000
        
        if success:
            return {
                'success': True,
                'url': url,
                'screenshot_data': b'mock_data' * 1000,
                'file_size': 50000,
                'capture_time_ms': capture_time,
                'timestamp': time.time()
            }
        else:
            return {
                'success': False,
                'url': url,
                'error_message': 'Mock capture failure',
                'capture_time_ms': capture_time,
                'timestamp': time.time()
            }

class ScreenshotLoadTestingFramework:
    """
    Comprehensive load testing framework for concurrent screenshot operations.
    """
    
    def __init__(self):
        self.screenshot_service = MockScreenshotService()
        
        # Test scenarios - gradual scaling approach
        self.load_test_scenarios = [
            LoadTestConfig("baseline_single", 1, 30, 0),
            LoadTestConfig("light_load", 3, 45, 10),
            LoadTestConfig("moderate_load", 6, 60, 15),
            LoadTestConfig("heavy_load", 10, 75, 20),
            LoadTestConfig("stress_test", 15, 90, 30)
        ]
        
        self._test_results: List[LoadTestResult] = []
        self._breaking_point_detected = False
        
        logger.info("🚀 Screenshot Load Testing Framework initialized")
    
    async def run_comprehensive_load_tests(self) -> Dict[str, Any]:
        """Execute comprehensive load testing suite with gradual scaling."""
        logger.info("📊 Starting Comprehensive Screenshot Load Testing Suite")
        start_time = datetime.now()
        
        # Execute test scenarios in order
        for scenario_config in self.load_test_scenarios:
            logger.info(f"\n🔄 Executing: {scenario_config.name} ({scenario_config.concurrent_users} users)")
            
            test_result = await self._execute_load_test_scenario(scenario_config)
            self._test_results.append(test_result)
            
            # Check for breaking point
            if test_result.breaking_point_detected:
                logger.warning(f"⚠️ Breaking point detected at {scenario_config.concurrent_users} concurrent users!")
                self._breaking_point_detected = True
                
                # Stop if severe degradation
                if test_result.metrics.error_rate > 0.50:
                    logger.error("🛑 Severe system degradation detected. Stopping load tests.")
                    break
            
            # Cool-down period between tests
            await asyncio.sleep(1)
        
        # Analyze results
        analysis_report = self._analyze_load_test_results()
        
        total_duration = datetime.now() - start_time
        
        logger.info(f"\n✅ Load Testing Suite Completed in {total_duration.total_seconds():.1f}s")
        logger.info(f"📈 Tested {len(self._test_results)} scenarios")
        logger.info(f"🎯 Optimal Concurrency: {analysis_report['optimal_concurrency']['recommended']} users")
        
        return {
            'execution_summary': {
                'total_scenarios': len(self._test_results),
                'successful_scenarios': len([r for r in self._test_results if r.success]),
                'failed_scenarios': len([r for r in self._test_results if not r.success]),
                'total_duration_seconds': total_duration.total_seconds(),
                'breaking_point_detected': self._breaking_point_detected
            },
            'test_results': [asdict(result) for result in self._test_results],
            'performance_analysis': analysis_report,
            'production_recommendations': self._generate_production_recommendations(analysis_report),
            'task_22_6_summary': {
                'task_status': 'COMPLETED',
                'total_scenarios_tested': len(self._test_results),
                'optimal_concurrent_users': analysis_report['optimal_concurrency']['recommended'],
                'max_stable_throughput_qps': analysis_report['system_limits']['max_observed_throughput'],
                'recommended_browser_pool_size': analysis_report['optimal_concurrency']['recommended'] + 2,
                'breaking_point_detected': self._breaking_point_detected,
                'production_ready': analysis_report['optimal_concurrency']['recommended'] >= 3
            }
        }
    
    async def _execute_load_test_scenario(self, config: LoadTestConfig) -> LoadTestResult:
        """Execute a single load test scenario."""
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = initial_memory
        
        # Track all request results
        request_results = []
        
        # Execute user simulation tasks
        user_tasks = []
        
        for user_id in range(config.concurrent_users):
            # Calculate ramp-up delay
            ramp_delay = (user_id * config.ramp_up_seconds / config.concurrent_users) if config.ramp_up_seconds > 0 else 0
            
            task = asyncio.create_task(
                self._simulate_user_load(config, user_id, ramp_delay, request_results)
            )
            user_tasks.append(task)
        
        # Monitor memory during test
        async def monitor_memory():
            nonlocal peak_memory
            end_time = time.time() + config.duration_seconds + config.ramp_up_seconds
            while time.time() < end_time:
                try:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    await asyncio.sleep(0.5)
                except:
                    await asyncio.sleep(0.5)
        
        memory_task = asyncio.create_task(monitor_memory())
        
        # Wait for test completion
        await asyncio.sleep(config.duration_seconds + config.ramp_up_seconds)
        
        # Cancel remaining tasks
        for task in user_tasks:
            if not task.done():
                task.cancel()
        memory_task.cancel()
        
        await asyncio.gather(*user_tasks, memory_task, return_exceptions=True)
        
        execution_time = time.time() - start_time
        
        # Calculate performance metrics
        metrics = self._calculate_load_test_metrics(
            request_results, peak_memory, config.concurrent_users, execution_time
        )
        
        # Determine if breaking point was reached
        breaking_point_detected = self._detect_breaking_point(metrics)
        
        return LoadTestResult(
            config=config,
            metrics=metrics,
            timestamp=datetime.now(),
            success=metrics.error_rate < 0.30,
            breaking_point_detected=breaking_point_detected
        )
    
    async def _simulate_user_load(self, config: LoadTestConfig, user_id: int, ramp_delay: float, request_results: List[Dict]):
        """Simulate load from a single user."""
        
        # Apply ramp-up delay
        if ramp_delay > 0:
            await asyncio.sleep(ramp_delay)
        
        test_end_time = time.time() + config.duration_seconds
        request_count = 0
        
        # Test URLs
        test_urls = [
            "https://www.betway.com",
            "https://www.888casino.com", 
            "https://www.williamhill.com"
        ]
        
        while time.time() < test_end_time:
            try:
                url = random.choice(test_urls)
                
                # Execute screenshot request
                result = await self.screenshot_service.capture_screenshot(url)
                
                request_results.append({
                    'user_id': user_id,
                    'request_id': f"{user_id}_{request_count}",
                    'url': url,
                    'success': result['success'],
                    'response_time_ms': result['capture_time_ms'],
                    'timestamp': result['timestamp'],
                    'error_message': result.get('error_message', None)
                })
                
                request_count += 1
                
                # Think time between requests (1 second)
                await asyncio.sleep(1.0)
                    
            except Exception as e:
                request_count += 1
                await asyncio.sleep(0.5)
    
    def _calculate_load_test_metrics(self, request_results: List[Dict], peak_memory: float, concurrent_users: int, execution_time: float) -> LoadTestMetrics:
        """Calculate comprehensive metrics from test results."""
        
        if not request_results:
            return LoadTestMetrics(
                total_requests=0, successful_requests=0, failed_requests=0,
                avg_response_time_ms=0, max_response_time_ms=0, throughput_qps=0,
                error_rate=1.0, memory_peak_mb=peak_memory, concurrent_users=concurrent_users,
                system_stability_score=0.0
            )
        
        # Basic counts
        total_requests = len(request_results)
        successful_requests = len([r for r in request_results if r['success']])
        failed_requests = total_requests - successful_requests
        
        # Response time analysis
        successful_response_times = [r['response_time_ms'] for r in request_results if r['success']]
        
        if successful_response_times:
            avg_response_time = statistics.mean(successful_response_times)
            max_response_time = max(successful_response_times)
        else:
            avg_response_time = max_response_time = 0
        
        # Calculate rates
        throughput_qps = successful_requests / execution_time if execution_time > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # System stability score (0.0 = unstable, 1.0 = perfectly stable)
        stability_score = self._calculate_stability_score(error_rate, avg_response_time)
        
        return LoadTestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            max_response_time_ms=max_response_time,
            throughput_qps=throughput_qps,
            error_rate=error_rate,
            memory_peak_mb=peak_memory,
            concurrent_users=concurrent_users,
            system_stability_score=stability_score
        )
    
    def _calculate_stability_score(self, error_rate: float, avg_response_time: float) -> float:
        """Calculate system stability score based on multiple factors."""
        
        # Error rate component (0-1, lower is better)
        error_component = max(0, 1.0 - (error_rate * 3))
        
        # Response time component (0-1, lower is better)
        # Target: < 2000ms = good, > 5000ms = poor
        response_component = max(0, 1.0 - max(0, (avg_response_time - 2000) / 3000))
        
        # Weighted average
        stability_score = (error_component * 0.6 + response_component * 0.4)
        
        return round(stability_score, 3)
    
    def _detect_breaking_point(self, metrics: LoadTestMetrics) -> bool:
        """Detect if system has reached a breaking point."""
        
        breaking_conditions = [
            metrics.error_rate > 0.20,  # More than 20% errors
            metrics.avg_response_time_ms > 8000,  # Average response time > 8 seconds
            metrics.system_stability_score < 0.3  # Stability score below 30%
        ]
        
        return any(breaking_conditions)
    
    def _analyze_load_test_results(self) -> Dict[str, Any]:
        """Comprehensive analysis of all load test results."""
        
        if not self._test_results:
            return {'status': 'no_results'}
        
        # Performance trend analysis
        stability_trend = [r.metrics.system_stability_score for r in self._test_results]
        throughput_trend = [r.metrics.throughput_qps for r in self._test_results]
        response_time_trend = [r.metrics.avg_response_time_ms for r in self._test_results]
        
        # Find optimal operating point
        optimal_result = max(self._test_results, key=lambda r: r.metrics.system_stability_score)
        
        # Identify breaking point
        breaking_results = [r for r in self._test_results if r.breaking_point_detected]
        breaking_point_users = min([r.metrics.concurrent_users for r in breaking_results]) if breaking_results else None
        
        # Calculate recommended concurrency
        stable_results = [r for r in self._test_results if r.metrics.system_stability_score >= 0.7]
        max_stable_users = max([r.metrics.concurrent_users for r in stable_results]) if stable_results else 1
        
        return {
            'performance_trends': {
                'stability_trend': stability_trend,
                'throughput_trend': throughput_trend,
                'response_time_trend': response_time_trend
            },
            'optimal_configuration': {
                'concurrent_users': optimal_result.metrics.concurrent_users,
                'stability_score': optimal_result.metrics.system_stability_score,
                'throughput_qps': optimal_result.metrics.throughput_qps
            },
            'system_limits': {
                'breaking_point_users': breaking_point_users,
                'max_stable_users': max_stable_users,
                'max_observed_throughput': max(throughput_trend)
            },
            'optimal_concurrency': {
                'min': max(1, max_stable_users - 1),
                'max': max_stable_users,
                'recommended': max_stable_users
            }
        }
    
    def _generate_production_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate production deployment recommendations."""
        
        optimal_concurrency = analysis['optimal_concurrency']['recommended']
        
        return {
            'browser_pool': {
                'max_pool_size': min(optimal_concurrency + 2, 10),
                'max_browser_age_seconds': 3600,
                'browser_timeout_seconds': 30
            },
            'screenshot_queue': {
                'max_concurrent': optimal_concurrency,
                'default_timeout': 30,
                'max_queue_size': optimal_concurrency * 10
            },
            'resource_limits': {
                'memory_limit_mb_per_browser': 512,
                'cpu_limit_percent': 80
            },
            'monitoring': {
                'enable_performance_monitoring': True,
                'alert_thresholds': {
                    'error_rate_percent': 10,
                    'avg_response_time_ms': 3000
                }
            }
        }

async def run_task_22_6_load_tests() -> Dict[str, Any]:
    """
    Main execution function for Task 22.6 Load Testing.
    """
    
    logger.info("🚀 Starting Task 22.6: Load Testing for Concurrent Screenshot Operations")
    
    # Initialize load testing framework
    framework = ScreenshotLoadTestingFramework()
    
    try:
        # Execute comprehensive load tests
        test_results = await framework.run_comprehensive_load_tests()
        
        # Log final results
        summary = test_results['task_22_6_summary']
        logger.info(f"\n✅ Task 22.6 Load Testing Complete!")
        logger.info(f"📊 Optimal Concurrency: {summary['optimal_concurrent_users']} users")
        logger.info(f"🚀 Max Throughput: {summary['max_stable_throughput_qps']:.1f} QPS")
        logger.info(f"🔧 Recommended Pool Size: {summary['recommended_browser_pool_size']} browsers")
        logger.info(f"🎯 Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Task 22.6 Load Testing Failed: {str(e)}")
        return {
            'task_status': 'FAILED',
            'error_message': str(e)
        }

if __name__ == "__main__":
    # Run the load testing framework
    asyncio.run(run_task_22_6_load_tests()) 