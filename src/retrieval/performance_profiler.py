"""
Task 3.7: Performance Profiler for Contextual Retrieval System

This module provides performance profiling and bottleneck detection.
"""

import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProfilingResult:
    """Performance profiling result."""
    
    operation_name: str
    total_time_ms: float
    stage_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceProfiler:
    """Performance profiler for contextual retrieval."""
    
    def __init__(self):
        self.profiling_results = deque(maxlen=1000)
        self.process = psutil.Process()
        logger.info("Performance Profiler initialized")
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile an operation."""
        
        result = ProfilingResult(operation_name=operation_name)
        start_time = time.time()
        memory_before = self._get_memory_usage_mb()
        
        try:
            yield result
        finally:
            result.total_time_ms = (time.time() - start_time) * 1000
            result.memory_usage_mb = self._get_memory_usage_mb() - memory_before
            result.cpu_usage_percent = self.process.cpu_percent()
            
            self._analyze_performance(result)
            self.profiling_results.append(result)
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def _analyze_performance(self, result: ProfilingResult):
        """Analyze performance and identify bottlenecks."""
        
        bottlenecks = []
        
        if result.total_time_ms > 1000:
            bottlenecks.append("high_latency")
        
        if result.memory_usage_mb > 100:
            bottlenecks.append("high_memory_usage")
        
        if result.cpu_usage_percent > 80:
            bottlenecks.append("high_cpu_usage")
        
        result.bottlenecks = bottlenecks
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        results = [r for r in self.profiling_results if r.timestamp > cutoff_time]
        
        if not results:
            return {"status": "no_data"}
        
        times = [r.total_time_ms for r in results]
        
        return {
            "status": "success",
            "total_operations": len(results),
            "avg_time_ms": np.mean(times),
            "p95_time_ms": np.percentile(times, 95),
            "max_time_ms": np.max(times)
        }


# Global profiler instance
global_profiler = PerformanceProfiler()


__all__ = ['PerformanceProfiler', 'ProfilingResult', 'global_profiler'] 