"""
Performance Profiler for RAG CMS
Provides detailed timing analysis, bottleneck identification, and optimization suggestions.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable, TypeVar, Coroutine
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import json
import threading
from functools import wraps
import traceback
from supabase import Client

T = TypeVar('T')

@dataclass
class TimingRecord:
    """Records timing information for a single operation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['TimingRecord'] = field(default_factory=list)
    error: Optional[str] = None
    
    def complete(self):
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def add_child(self, child: 'TimingRecord'):
        """Add a child timing record."""
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation_name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
            "children": [child.to_dict() for child in self.children]
        }

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    active_operations: int
    memory_usage_mb: float
    cpu_percent: float
    pending_tasks: int
    cache_size: int
    avg_response_time_ms: float

class PerformanceProfiler:
    """Advanced performance profiling and analysis."""
    
    def __init__(self, supabase_client: Client, enable_profiling: bool = True):
        self.client = supabase_client
        self.enabled = enable_profiling
        self.profiles_table = "performance_profiles"
        self.bottlenecks_table = "performance_bottlenecks"
        
        # Thread-local storage for nested profiling
        self._local = threading.local()
        
        # Performance history for trend analysis
        self.operation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.snapshot_history: deque = deque(maxlen=100)
        
        # Bottleneck detection thresholds
        self.bottleneck_thresholds = {
            "retrieval": 500,  # ms
            "embedding_generation": 200,
            "llm_generation": 2000,
            "cache_lookup": 50,
            "database_query": 100,
            "api_call": 1000
        }
        
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure profiling tables exist in Supabase."""
        # These would be in migration files
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS performance_profiles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            query_id TEXT NOT NULL,
            profile_data JSONB NOT NULL,
            total_duration_ms FLOAT NOT NULL,
            bottleneck_operations JSONB DEFAULT '[]',
            optimization_suggestions JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_profiles_query ON performance_profiles(query_id);
        CREATE INDEX idx_profiles_duration ON performance_profiles(total_duration_ms DESC);
        CREATE INDEX idx_profiles_time ON performance_profiles(created_at DESC);
        
        CREATE TABLE IF NOT EXISTS performance_bottlenecks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            operation_type TEXT NOT NULL,
            avg_duration_ms FLOAT NOT NULL,
            p95_duration_ms FLOAT NOT NULL,
            p99_duration_ms FLOAT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            impact_score FLOAT NOT NULL, -- 0-100
            suggested_optimizations JSONB DEFAULT '[]',
            detected_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_bottlenecks_impact ON performance_bottlenecks(impact_score DESC);
        CREATE INDEX idx_bottlenecks_type ON performance_bottlenecks(operation_type);
        """
    
    @property
    def current_profile(self) -> Optional[TimingRecord]:
        """Get current profiling context."""
        return getattr(self._local, 'profile_stack', [])[-1] if hasattr(self._local, 'profile_stack') and self._local.profile_stack else None
    
    @asynccontextmanager
    async def profile(self, operation_name: str, **metadata):
        """Context manager for profiling an operation."""
        if not self.enabled:
            yield
            return
        
        # Initialize thread-local stack if needed
        if not hasattr(self._local, 'profile_stack'):
            self._local.profile_stack = []
        
        # Create timing record
        record = TimingRecord(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata
        )
        
        # Add to parent if exists
        if self.current_profile:
            self.current_profile.add_child(record)
        
        # Push to stack
        self._local.profile_stack.append(record)
        
        try:
            yield record
        except Exception as e:
            record.error = str(e)
            raise
        finally:
            # Complete timing
            record.complete()
            
            # Pop from stack
            self._local.profile_stack.pop()
            
            # Store in history
            self.operation_history[operation_name].append(record.duration_ms)
            
            # If this was root operation, analyze and store
            if not self._local.profile_stack:
                await self._analyze_and_store_profile(record)
    
    def profile_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for profiling synchronous functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            operation_name = f"{func.__module__}.{func.__name__}"
            record = TimingRecord(
                operation_name=operation_name,
                start_time=time.time(),
                metadata={"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
            )
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                record.error = str(e)
                raise
            finally:
                record.complete()
                self.operation_history[operation_name].append(record.duration_ms)
        
        return wrapper
    
    def profile_async(self, func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        """Decorator for profiling async functions."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.enabled:
                return await func(*args, **kwargs)
            
            operation_name = f"{func.__module__}.{func.__name__}"
            async with self.profile(operation_name, args=str(args)[:100], kwargs=str(kwargs)[:100]):
                return await func(*args, **kwargs)
        
        return wrapper
    
    async def _analyze_and_store_profile(self, profile: TimingRecord):
        """Analyze completed profile and store results."""
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(profile)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(profile, bottlenecks)
        
        # Store profile
        profile_data = {
            "query_id": profile.metadata.get("query_id", "unknown"),
            "profile_data": profile.to_dict(),
            "total_duration_ms": profile.duration_ms,
            "bottleneck_operations": bottlenecks,
            "optimization_suggestions": suggestions
        }
        
        self.client.table(self.profiles_table).insert(profile_data).execute()
        
        # Update bottleneck statistics
        await self._update_bottleneck_stats(bottlenecks)
    
    def _identify_bottlenecks(self, profile: TimingRecord, parent_duration: Optional[float] = None) -> List[Dict[str, Any]]:
        """Recursively identify bottleneck operations."""
        bottlenecks = []
        
        # Use parent duration or total duration
        total_duration = parent_duration or profile.duration_ms
        
        # Check if this operation is a bottleneck
        threshold = self.bottleneck_thresholds.get(
            profile.operation_name.lower(), 
            0.3 * total_duration  # Default: 30% of total time
        )
        
        if profile.duration_ms > threshold:
            bottlenecks.append({
                "operation": profile.operation_name,
                "duration_ms": profile.duration_ms,
                "percentage_of_total": (profile.duration_ms / total_duration) * 100,
                "threshold_exceeded_by": profile.duration_ms - threshold
            })
        
        # Check children
        for child in profile.children:
            child_bottlenecks = self._identify_bottlenecks(child, profile.duration_ms)
            bottlenecks.extend(child_bottlenecks)
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, profile: TimingRecord, bottlenecks: List[Dict]) -> List[Dict[str, str]]:
        """Generate specific optimization suggestions based on bottlenecks."""
        suggestions = []
        
        for bottleneck in bottlenecks:
            operation = bottleneck["operation"].lower()
            duration = bottleneck["duration_ms"]
            
            if "retrieval" in operation:
                if duration > 1000:
                    suggestions.append({
                        "operation": bottleneck["operation"],
                        "issue": f"Retrieval taking {duration:.0f}ms",
                        "suggestion": "Consider reducing chunk size or implementing parallel retrieval",
                        "priority": "high"
                    })
                elif duration > 500:
                    suggestions.append({
                        "operation": bottleneck["operation"],
                        "issue": f"Slow retrieval ({duration:.0f}ms)",
                        "suggestion": "Enable hybrid search or optimize vector index parameters",
                        "priority": "medium"
                    })
            
            elif "embedding" in operation:
                suggestions.append({
                    "operation": bottleneck["operation"],
                    "issue": f"Embedding generation taking {duration:.0f}ms",
                    "suggestion": "Batch embed multiple chunks or use a faster embedding model",
                    "priority": "medium"
                })
            
            elif "llm" in operation or "generation" in operation:
                if duration > 3000:
                    suggestions.append({
                        "operation": bottleneck["operation"],
                        "issue": f"LLM generation taking {duration:.0f}ms",
                        "suggestion": "Consider using a smaller model for simple queries or implementing streaming",
                        "priority": "high"
                    })
            
            elif "cache" in operation:
                if duration > 100:
                    suggestions.append({
                        "operation": bottleneck["operation"],
                        "issue": f"Cache lookup taking {duration:.0f}ms",
                        "suggestion": "Optimize cache key generation or consider in-memory caching",
                        "priority": "low"
                    })
            
            elif "database" in operation or "supabase" in operation:
                if duration > 200:
                    suggestions.append({
                        "operation": bottleneck["operation"],
                        "issue": f"Database query taking {duration:.0f}ms",
                        "suggestion": "Add appropriate indexes or consider query optimization",
                        "priority": "medium"
                    })
        
        return suggestions
    
    async def _update_bottleneck_stats(self, bottlenecks: List[Dict]):
        """Update bottleneck statistics in database."""
        for bottleneck in bottlenecks:
            operation = bottleneck["operation"]
            
            # Get historical data for this operation
            history = list(self.operation_history[operation])
            if len(history) < 10:
                continue  # Not enough data
            
            # Calculate statistics
            avg_duration = np.mean(history)
            p95_duration = np.percentile(history, 95)
            p99_duration = np.percentile(history, 99)
            
            # Calculate impact score (0-100)
            # Based on frequency, duration, and variance
            frequency_score = min(len(history) / 100, 1.0) * 30
            duration_score = min(avg_duration / 1000, 1.0) * 50
            variance_score = (np.std(history) / avg_duration) * 20
            impact_score = frequency_score + duration_score + variance_score
            
            bottleneck_data = {
                "operation_type": operation,
                "avg_duration_ms": avg_duration,
                "p95_duration_ms": p95_duration,
                "p99_duration_ms": p99_duration,
                "occurrence_count": len(history),
                "impact_score": impact_score,
                "suggested_optimizations": self._get_operation_optimizations(operation, avg_duration)
            }
            
            self.client.table(self.bottlenecks_table).insert(bottleneck_data).execute()
    
    def _get_operation_optimizations(self, operation: str, avg_duration: float) -> List[str]:
        """Get specific optimizations for an operation type."""
        optimizations = []
        
        if "retrieval" in operation.lower():
            optimizations.extend([
                "Implement query result caching with semantic similarity",
                "Use hybrid search combining dense and sparse retrievers",
                "Optimize chunk size and overlap parameters",
                "Enable parallel chunk retrieval"
            ])
        elif "embedding" in operation.lower():
            optimizations.extend([
                "Batch process multiple texts in single API call",
                "Cache frequently used embeddings",
                "Consider using a local embedding model for lower latency"
            ])
        elif "llm" in operation.lower():
            optimizations.extend([
                "Implement response streaming for better UX",
                "Use smaller models for simple queries",
                "Enable prompt caching for repeated patterns",
                "Optimize token usage in prompts"
            ])
        
        return optimizations
    
    async def capture_performance_snapshot(self) -> PerformanceSnapshot:
        """Capture current system performance snapshot."""
        import psutil
        
        # Get system metrics
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Get operation metrics
        recent_operations = []
        for ops in self.operation_history.values():
            recent_operations.extend(list(ops)[-10:])  # Last 10 of each type
        
        avg_response_time = np.mean(recent_operations) if recent_operations else 0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            active_operations=len(getattr(self._local, 'profile_stack', [])),
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            pending_tasks=len(asyncio.all_tasks()),
            cache_size=0,  # Would get from cache manager
            avg_response_time_ms=avg_response_time
        )
        
        self.snapshot_history.append(snapshot)
        return snapshot
    
    async def get_optimization_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get recent profiles
        profiles_result = self.client.table(self.profiles_table).select(
            "total_duration_ms, bottleneck_operations, optimization_suggestions"
        ).gte("created_at", cutoff_time.isoformat()).execute()
        
        # Get bottleneck statistics
        bottlenecks_result = self.client.table(self.bottlenecks_table).select(
            "*"
        ).order("impact_score", desc=True).limit(10).execute()
        
        # Analyze patterns
        total_queries = len(profiles_result.data)
        avg_duration = np.mean([p["total_duration_ms"] for p in profiles_result.data]) if profiles_result.data else 0
        
        # Count bottleneck occurrences
        bottleneck_counts = defaultdict(int)
        for profile in profiles_result.data:
            for bottleneck in profile["bottleneck_operations"]:
                bottleneck_counts[bottleneck["operation"]] += 1
        
        # Compile report
        report = {
            "period": f"Last {hours} hours",
            "summary": {
                "total_profiled_queries": total_queries,
                "avg_query_duration_ms": avg_duration,
                "profiles_with_bottlenecks": sum(1 for p in profiles_result.data if p["bottleneck_operations"])
            },
            "top_bottlenecks": [
                {
                    "operation": b["operation_type"],
                    "impact_score": b["impact_score"],
                    "avg_duration_ms": b["avg_duration_ms"],
                    "p95_duration_ms": b["p95_duration_ms"],
                    "occurrences": bottleneck_counts.get(b["operation_type"], 0),
                    "optimizations": b["suggested_optimizations"]
                }
                for b in bottlenecks_result.data
            ],
            "optimization_priorities": self._prioritize_optimizations(bottlenecks_result.data),
            "performance_trends": self._analyze_performance_trends()
        }
        
        return report
    
    def _prioritize_optimizations(self, bottlenecks: List[Dict]) -> List[Dict[str, Any]]:
        """Prioritize optimization efforts based on impact."""
        priorities = []
        
        for bottleneck in bottlenecks[:5]:  # Top 5
            impact = bottleneck["impact_score"]
            
            if impact > 80:
                priority = "critical"
                timeline = "immediate"
            elif impact > 60:
                priority = "high"
                timeline = "this week"
            elif impact > 40:
                priority = "medium"
                timeline = "this month"
            else:
                priority = "low"
                timeline = "next quarter"
            
            priorities.append({
                "operation": bottleneck["operation_type"],
                "priority": priority,
                "timeline": timeline,
                "expected_improvement": f"{impact * 0.3:.0f}% reduction in response time",
                "effort_estimate": self._estimate_optimization_effort(bottleneck)
            })
        
        return priorities
    
    def _estimate_optimization_effort(self, bottleneck: Dict) -> str:
        """Estimate effort required for optimization."""
        operation = bottleneck["operation_type"].lower()
        
        if "cache" in operation:
            return "Low (1-2 days)"
        elif "index" in operation or "database" in operation:
            return "Medium (3-5 days)"
        elif "retrieval" in operation or "embedding" in operation:
            return "High (1-2 weeks)"
        else:
            return "Variable"
    
    def _analyze_performance_trends(self) -> Dict[str, str]:
        """Analyze performance trends from snapshot history."""
        if len(self.snapshot_history) < 10:
            return {"status": "Insufficient data for trend analysis"}
        
        snapshots = list(self.snapshot_history)
        response_times = [s.avg_response_time_ms for s in snapshots]
        memory_usage = [s.memory_usage_mb for s in snapshots]
        
        # Simple trend detection
        response_trend = "improving" if response_times[-5:] < response_times[:5] else "degrading"
        memory_trend = "stable" if np.std(memory_usage) < 50 else "increasing"
        
        return {
            "response_time_trend": response_trend,
            "memory_usage_trend": memory_trend,
            "recommendations": self._get_trend_recommendations(response_trend, memory_trend)
        }
    
    def _get_trend_recommendations(self, response_trend: str, memory_trend: str) -> List[str]:
        """Get recommendations based on performance trends."""
        recommendations = []
        
        if response_trend == "degrading":
            recommendations.append("Response times are increasing. Review recent changes and consider scaling resources.")
        
        if memory_trend == "increasing":
            recommendations.append("Memory usage is growing. Check for memory leaks and optimize caching strategy.")
        
        if not recommendations:
            recommendations.append("Performance is stable. Continue monitoring for changes.")
        
        return recommendations

# Convenience decorators for easy profiling
def profile_operation(profiler: PerformanceProfiler):
    """Decorator factory for profiling operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return profiler.profile_async(func)
        else:
            return profiler.profile_sync(func)
    return decorator 