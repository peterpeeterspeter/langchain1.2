"""
Comprehensive Monitoring System for RAG CMS
Implements the core monitoring and analytics system with real-time metrics collection,
alert management, and performance reporting.
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid
from supabase import Client

# Simple mock logger for monitoring (will be replaced with full logging later)
class MockLogger:
    def __init__(self, name, supabase_client=None):
        self.name = name
        self.client = supabase_client
    
    def debug(self, category, message, **kwargs):
        print(f"[DEBUG] {self.name}: {message}")
    
    def info(self, category, message, **kwargs):
        print(f"[INFO] {self.name}: {message}")
    
    def error(self, category, message, **kwargs):
        print(f"[ERROR] {self.name}: {message}")
    
    def warning(self, category, message, **kwargs):
        print(f"[WARNING] {self.name}: {message}")

class LogCategory:
    QUERY_PROCESSING = "query_processing"
    PERFORMANCE = "performance"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CONFIGURATION = "configuration"

def get_logger(name, supabase_client=None):
    return MockLogger(name, supabase_client)

# Import QueryType
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.prompt_config import QueryType


class MetricType(str, Enum):
    """Types of metrics collected."""
    CLASSIFICATION = "classification"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CACHE = "cache"
    ERROR = "error"
    USAGE = "usage"
    ALERT = "alert"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class QueryMetrics:
    """Structured metric storage for query processing."""
    query_id: str
    query_text: str
    query_type: QueryType
    timestamp: datetime
    
    # Classification metrics
    classification_confidence: float
    classification_time_ms: float = 0.0
    
    # Performance metrics
    response_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_tokens: int = 0
    
    # Quality metrics
    response_quality_score: float = 0.0
    relevance_scores: List[float] = field(default_factory=list)
    context_utilization_score: float = 0.0
    
    # Cache metrics
    cache_hit: bool = False
    cache_latency_ms: float = 0.0
    
    # Context metrics
    sources_count: int = 0
    context_length: int = 0
    
    # Error tracking
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['query_type'] = self.query_type.value if isinstance(self.query_type, QueryType) else self.query_type
        return data


@dataclass
class AlertThreshold:
    """Configuration for metric alerts."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "greater_than"  # greater_than, less_than, equals
    sample_size: int = 100  # Number of samples to consider
    cooldown_minutes: int = 15  # Minimum time between same alerts
    enabled: bool = True
    
    def evaluate(self, values: List[float]) -> Optional[AlertSeverity]:
        """Evaluate if alert should be triggered."""
        if not self.enabled or len(values) < min(self.sample_size, 10):
            return None
            
        # Use recent samples for evaluation
        recent_values = values[-self.sample_size:]
        avg_value = statistics.mean(recent_values)
        
        if self.comparison == "greater_than":
            if avg_value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif avg_value >= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison == "less_than":
            if avg_value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif avg_value <= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.comparison == "equals":
            if abs(avg_value - self.critical_threshold) < 0.01:
                return AlertSeverity.CRITICAL
            elif abs(avg_value - self.warning_threshold) < 0.01:
                return AlertSeverity.WARNING
                
        return None


@dataclass
class AlertInstance:
    """Individual alert instance."""
    id: str
    alert_type: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def acknowledge(self, user: str):
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()


class PromptAnalytics:
    """Main analytics system with buffered metrics collection and alert management."""
    
    def __init__(self, supabase_client: Client, buffer_size: int = 50):
        self.client = supabase_client
        self.buffer_size = buffer_size
        self.logger = get_logger("prompt_analytics", supabase_client)
        
        # Tables
        self.metrics_table = "prompt_metrics"
        self.alerts_table = "prompt_alerts"
        
        # In-memory buffers
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.recent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Alert thresholds
        self.alert_thresholds: Dict[str, AlertThreshold] = self._initialize_default_thresholds()
        
        # Background tasks
        self._flush_task = None
        self._analysis_task = None
        self._start_background_tasks()
    
    def _initialize_default_thresholds(self) -> Dict[str, AlertThreshold]:
        """Initialize default alert thresholds."""
        return {
            "avg_response_time": AlertThreshold(
                metric_name="avg_response_time_ms",
                warning_threshold=3000.0,  # 3 seconds
                critical_threshold=5000.0,  # 5 seconds
                comparison="greater_than",
                sample_size=100
            ),
            "error_rate": AlertThreshold(
                metric_name="error_rate",
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10,  # 10%
                comparison="greater_than",
                sample_size=50
            ),
            "quality_score": AlertThreshold(
                metric_name="avg_quality_score",
                warning_threshold=0.6,
                critical_threshold=0.4,
                comparison="less_than",
                sample_size=100
            ),
            "cache_hit_rate": AlertThreshold(
                metric_name="cache_hit_rate",
                warning_threshold=0.7,  # 70%
                critical_threshold=0.5,  # 50%
                comparison="less_than",
                sample_size=100
            ),
            "classification_confidence": AlertThreshold(
                metric_name="avg_classification_confidence",
                warning_threshold=0.7,
                critical_threshold=0.5,
                comparison="less_than",
                sample_size=100
            )
        }
    
    def _start_background_tasks(self):
        """Start background processing tasks."""
        loop = asyncio.get_event_loop()
        
        # Periodic buffer flushing
        self._flush_task = loop.create_task(self._periodic_flush())
        
        # Periodic analysis and alerting
        self._analysis_task = loop.create_task(self._periodic_analysis())
    
    async def _periodic_flush(self):
        """Periodically flush metrics buffer to database."""
        while True:
            try:
                await asyncio.sleep(30)  # Flush every 30 seconds
                await self.flush_metrics()
            except Exception as e:
                self.logger.error(
                    LogCategory.PERFORMANCE,
                    "Error in periodic flush",
                    error=e
                )
    
    async def _periodic_analysis(self):
        """Periodically analyze metrics and trigger alerts."""
        while True:
            try:
                await asyncio.sleep(60)  # Analyze every minute
                await self._analyze_and_alert()
            except Exception as e:
                self.logger.error(
                    LogCategory.PERFORMANCE,
                    "Error in periodic analysis", 
                    error=e
                )
    
    async def track_query_metrics(self, metrics: QueryMetrics):
        """Track metrics for a query."""
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        # Update recent metrics for real-time analysis
        self._update_recent_metrics(metrics)
        
        # Log the tracking
        self.logger.debug(
            LogCategory.PERFORMANCE,
            f"Tracked metrics for query {metrics.query_id}",
            data={
                "query_type": metrics.query_type.value if isinstance(metrics.query_type, QueryType) else metrics.query_type,
                "response_time_ms": metrics.response_time_ms,
                "quality_score": metrics.response_quality_score,
                "cache_hit": metrics.cache_hit
            }
        )
        
        # Auto-flush if buffer is full
        if len(self.metrics_buffer) >= self.buffer_size:
            await self.flush_metrics()
    
    def _update_recent_metrics(self, metrics: QueryMetrics):
        """Update recent metrics for real-time analysis."""
        timestamp = time.time()
        
        # Performance metrics
        self.recent_metrics["response_time_ms"].append(metrics.response_time_ms)
        self.recent_metrics["retrieval_time_ms"].append(metrics.retrieval_time_ms)
        self.recent_metrics["generation_time_ms"].append(metrics.generation_time_ms)
        
        # Quality metrics
        self.recent_metrics["quality_score"].append(metrics.response_quality_score)
        self.recent_metrics["classification_confidence"].append(metrics.classification_confidence)
        
        # Cache metrics
        self.recent_metrics["cache_hit"].append(1.0 if metrics.cache_hit else 0.0)
        
        # Error tracking
        self.recent_metrics["error"].append(1.0 if metrics.error else 0.0)
        
        # Query type distribution
        query_type = metrics.query_type.value if isinstance(metrics.query_type, QueryType) else metrics.query_type
        self.recent_metrics[f"query_type_{query_type}"].append(1.0)
    
    async def flush_metrics(self):
        """Flush metrics buffer to database."""
        if not self.metrics_buffer:
            return
        
        metrics_to_flush = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        
        try:
            # Prepare data for bulk insert
            db_records = []
            for metric in metrics_to_flush:
                db_records.append({
                    "query_id": metric.query_id,
                    "query_text": metric.query_text[:500],  # Truncate long queries
                    "query_type": metric.query_type.value if isinstance(metric.query_type, QueryType) else metric.query_type,
                    "metric_type": MetricType.PERFORMANCE.value,
                    "metric_value": metric.to_dict(),
                    "timestamp": metric.timestamp.isoformat()
                })
            
            # Bulk insert to Supabase
            self.client.table(self.metrics_table).insert(db_records).execute()
            
            self.logger.debug(
                LogCategory.DATABASE,
                f"Flushed {len(db_records)} metrics to database"
            )
            
        except Exception as e:
            # Put metrics back in buffer on failure
            for metric in reversed(metrics_to_flush):
                self.metrics_buffer.appendleft(metric)
            
            self.logger.error(
                LogCategory.DATABASE,
                "Failed to flush metrics to database",
                error=e
            )
    
    async def _analyze_and_alert(self):
        """Analyze recent metrics and trigger alerts."""
        current_time = datetime.utcnow()
        
        # Calculate current metric values
        metric_values = self._calculate_current_metrics()
        
        # Check each threshold
        for threshold_name, threshold in self.alert_thresholds.items():
            if not threshold.enabled:
                continue
                
            # Check cooldown
            cooldown_key = f"{threshold_name}_{threshold.metric_name}"
            if cooldown_key in self.alert_cooldowns:
                cooldown_end = self.alert_cooldowns[cooldown_key] + timedelta(minutes=threshold.cooldown_minutes)
                if current_time < cooldown_end:
                    continue
            
            # Get relevant metric values
            metric_key = self._get_metric_key_for_threshold(threshold)
            if metric_key not in metric_values:
                continue
                
            values = metric_values[metric_key]
            severity = threshold.evaluate(values)
            
            if severity:
                await self._trigger_alert(threshold, severity, values[-1] if values else 0.0)
                self.alert_cooldowns[cooldown_key] = current_time
    
    def _calculate_current_metrics(self) -> Dict[str, List[float]]:
        """Calculate current aggregated metrics."""
        metrics = {}
        
        # Response time metrics
        if "response_time_ms" in self.recent_metrics:
            metrics["avg_response_time_ms"] = list(self.recent_metrics["response_time_ms"])
        
        # Quality metrics  
        if "quality_score" in self.recent_metrics:
            metrics["avg_quality_score"] = list(self.recent_metrics["quality_score"])
            
        # Classification confidence
        if "classification_confidence" in self.recent_metrics:
            metrics["avg_classification_confidence"] = list(self.recent_metrics["classification_confidence"])
        
        # Error rate
        if "error" in self.recent_metrics:
            error_values = list(self.recent_metrics["error"])
            metrics["error_rate"] = error_values
        
        # Cache hit rate
        if "cache_hit" in self.recent_metrics:
            cache_values = list(self.recent_metrics["cache_hit"])
            metrics["cache_hit_rate"] = cache_values
            
        return metrics
    
    def _get_metric_key_for_threshold(self, threshold: AlertThreshold) -> str:
        """Get the metric key that corresponds to a threshold."""
        return threshold.metric_name
    
    async def _trigger_alert(self, threshold: AlertThreshold, severity: AlertSeverity, current_value: float):
        """Trigger an alert."""
        alert_id = str(uuid.uuid4())
        
        # Determine threshold value based on severity
        threshold_value = threshold.critical_threshold if severity == AlertSeverity.CRITICAL else threshold.warning_threshold
        
        # Create alert message
        message = self._create_alert_message(threshold, severity, current_value, threshold_value)
        
        # Create alert instance
        alert = AlertInstance(
            id=alert_id,
            alert_type=threshold.metric_name,
            severity=severity,
            metric_name=threshold.metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            metadata={
                "comparison": threshold.comparison,
                "sample_size": threshold.sample_size
            }
        )
        
        # Store in active alerts
        self.active_alerts[alert_id] = alert
        
        # Store in database
        try:
            await self._store_alert(alert)
        except Exception as e:
            self.logger.error(
                LogCategory.DATABASE,
                "Failed to store alert in database",
                error=e
            )
        
        # Log the alert
        log_level = LogCategory.PERFORMANCE
        self.logger.warning(
            log_level,
            f"Alert triggered: {message}",
            data={
                "alert_id": alert_id,
                "severity": severity.value,
                "metric_name": threshold.metric_name,
                "current_value": current_value,
                "threshold_value": threshold_value
            }
        )
    
    def _create_alert_message(self, threshold: AlertThreshold, severity: AlertSeverity, current_value: float, threshold_value: float) -> str:
        """Create human-readable alert message."""
        comparison_text = {
            "greater_than": "above",
            "less_than": "below", 
            "equals": "equal to"
        }.get(threshold.comparison, "compared to")
        
        return (
            f"{severity.value.upper()}: {threshold.metric_name} is {comparison_text} threshold. "
            f"Current: {current_value:.2f}, Threshold: {threshold_value:.2f}"
        )
    
    async def _store_alert(self, alert: AlertInstance):
        """Store alert in database."""
        alert_data = {
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "message": alert.message,
            "metadata": alert.metadata,
            "acknowledged": alert.acknowledged,
            "acknowledged_by": alert.acknowledged_by,
            "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            "created_at": alert.created_at.isoformat()
        }
        
        self.client.table(self.alerts_table).insert(alert_data).execute()
    
    async def get_real_time_metrics(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get real-time metrics for the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        # Calculate metrics from recent data
        metrics = {
            "window_minutes": window_minutes,
            "timestamp": datetime.utcnow().isoformat(),
            "performance": self._calculate_performance_metrics(),
            "quality": self._calculate_quality_metrics(),
            "cache": self._calculate_cache_metrics(),
            "classification": self._calculate_classification_metrics(),
            "errors": self._calculate_error_metrics(),
            "usage": self._calculate_usage_metrics()
        }
        
        return metrics
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance-related metrics."""
        response_times = list(self.recent_metrics.get("response_time_ms", []))
        retrieval_times = list(self.recent_metrics.get("retrieval_time_ms", []))
        generation_times = list(self.recent_metrics.get("generation_time_ms", []))
        
        if not response_times:
            return {"avg_response_time_ms": 0, "p95_response_time_ms": 0, "query_count": 0}
        
        return {
            "avg_response_time_ms": statistics.mean(response_times),
            "median_response_time_ms": statistics.median(response_times),
            "p95_response_time_ms": self._percentile(response_times, 95),
            "p99_response_time_ms": self._percentile(response_times, 99),
            "avg_retrieval_time_ms": statistics.mean(retrieval_times) if retrieval_times else 0,
            "avg_generation_time_ms": statistics.mean(generation_times) if generation_times else 0,
            "query_count": len(response_times),
            "trend": self._calculate_trend(response_times)
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality-related metrics."""
        quality_scores = list(self.recent_metrics.get("quality_score", []))
        confidence_scores = list(self.recent_metrics.get("classification_confidence", []))
        
        if not quality_scores:
            return {"avg_quality_score": 0, "avg_confidence": 0}
        
        return {
            "avg_quality_score": statistics.mean(quality_scores),
            "median_quality_score": statistics.median(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "quality_distribution": self._calculate_quality_distribution(quality_scores)
        }
    
    def _calculate_cache_metrics(self) -> Dict[str, Any]:
        """Calculate cache-related metrics."""
        cache_hits = list(self.recent_metrics.get("cache_hit", []))
        
        if not cache_hits:
            return {"overall_hit_rate": 0, "total_requests": 0}
        
        hit_rate = statistics.mean(cache_hits)
        
        return {
            "overall_hit_rate": hit_rate,
            "total_requests": len(cache_hits),
            "cache_hits": sum(cache_hits),
            "cache_misses": len(cache_hits) - sum(cache_hits)
        }
    
    def _calculate_classification_metrics(self) -> Dict[str, Any]:
        """Calculate query classification metrics."""
        confidence_scores = list(self.recent_metrics.get("classification_confidence", []))
        
        # Query type distribution
        query_type_counts = {}
        for key in self.recent_metrics:
            if key.startswith("query_type_"):
                query_type = key[11:]  # Remove "query_type_" prefix
                query_type_counts[query_type] = len(self.recent_metrics[key])
        
        return {
            "overall_accuracy": statistics.mean(confidence_scores) if confidence_scores else 0,
            "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0,
            "query_type_distribution": query_type_counts,
            "total_queries": len(confidence_scores)
        }
    
    def _calculate_error_metrics(self) -> Dict[str, Any]:
        """Calculate error-related metrics."""
        errors = list(self.recent_metrics.get("error", []))
        
        if not errors:
            return {"error_rate": 0, "total_errors": 0, "total_requests": 0}
        
        error_rate = statistics.mean(errors)
        total_errors = sum(errors)
        
        return {
            "error_rate": error_rate,
            "total_errors": int(total_errors),
            "total_requests": len(errors),
            "success_rate": 1.0 - error_rate
        }
    
    def _calculate_usage_metrics(self) -> Dict[str, Any]:
        """Calculate usage-related metrics."""
        # This would include user sessions, geographic distribution, etc.
        # For now, return basic metrics
        return {
            "active_sessions": len(set(self.recent_metrics.keys())),  # Simplified
            "queries_per_minute": len(self.recent_metrics.get("response_time_ms", [])) / 5,  # Assuming 5-minute window
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple trend calculation: compare first and last thirds
        first_third = values[:len(values)//3]
        last_third = values[-len(values)//3:]
        
        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)
        
        if last_avg > first_avg * 1.1:
            return "increasing"
        elif last_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_quality_distribution(self, quality_scores: List[float]) -> Dict[str, int]:
        """Calculate distribution of quality scores."""
        distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
        for score in quality_scores:
            if score >= 0.9:
                distribution["excellent"] += 1
            elif score >= 0.7:
                distribution["good"] += 1
            elif score >= 0.5:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        return distribution
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unacknowledged) alerts."""
        return [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "created_at": alert.created_at.isoformat(),
                "metadata": alert.metadata
            }
            for alert in self.active_alerts.values()
            if not alert.acknowledged
        ]
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge(user)
            
            # Update in database
            try:
                self.client.table(self.alerts_table).update({
                    "acknowledged": True,
                    "acknowledged_by": user,
                    "acknowledged_at": alert.acknowledged_at.isoformat()
                }).eq("id", alert_id).execute()
                
                self.logger.info(
                    LogCategory.PERFORMANCE,
                    f"Alert {alert_id} acknowledged by {user}"
                )
                return True
                
            except Exception as e:
                self.logger.error(
                    LogCategory.DATABASE,
                    "Failed to acknowledge alert in database",
                    error=e
                )
                return False
        
        return False
    
    def update_alert_threshold(self, threshold_name: str, **kwargs):
        """Update an alert threshold configuration."""
        if threshold_name in self.alert_thresholds:
            threshold = self.alert_thresholds[threshold_name]
            for key, value in kwargs.items():
                if hasattr(threshold, key):
                    setattr(threshold, key, value)
            
            self.logger.info(
                LogCategory.CONFIGURATION,
                f"Updated alert threshold: {threshold_name}",
                data=kwargs
            )
    
    def add_alert_threshold(self, threshold_name: str, threshold: AlertThreshold):
        """Add a new alert threshold."""
        self.alert_thresholds[threshold_name] = threshold
        
        self.logger.info(
            LogCategory.CONFIGURATION,
            f"Added new alert threshold: {threshold_name}",
            data={"metric_name": threshold.metric_name}
        )
    
    async def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Fetch metrics from database
        try:
            metrics_data = self.client.table(self.metrics_table).select(
                "*"
            ).gte("timestamp", cutoff_time.isoformat()).execute()
            
            # Process the data
            report = {
                "period": f"Last {hours} hours",
                "generated_at": datetime.utcnow().isoformat(),
                "summary": self._generate_report_summary(metrics_data.data),
                "trends": self._analyze_trends(metrics_data.data),
                "bottlenecks": self._identify_bottlenecks(metrics_data.data),
                "recommendations": self._generate_recommendations(metrics_data.data)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(
                LogCategory.DATABASE,
                "Failed to generate performance report",
                error=e
            )
            return {"error": "Failed to generate report"}
    
    def _generate_report_summary(self, metrics_data: List[Dict]) -> Dict[str, Any]:
        """Generate summary section of performance report."""
        if not metrics_data:
            return {"total_queries": 0}
        
        # Extract metric values
        response_times = []
        quality_scores = []
        error_count = 0
        cache_hits = 0
        
        for record in metrics_data:
            metric_value = record.get("metric_value", {})
            response_times.append(metric_value.get("response_time_ms", 0))
            quality_scores.append(metric_value.get("response_quality_score", 0))
            
            if metric_value.get("error"):
                error_count += 1
            if metric_value.get("cache_hit"):
                cache_hits += 1
        
        return {
            "total_queries": len(metrics_data),
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
            "error_rate": error_count / len(metrics_data) if metrics_data else 0,
            "cache_hit_rate": cache_hits / len(metrics_data) if metrics_data else 0
        }
    
    def _analyze_trends(self, metrics_data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in the metrics data."""
        # Simplified trend analysis
        return {
            "response_time_trend": "stable",  # Would implement proper trend analysis
            "quality_trend": "stable",
            "error_trend": "stable"
        }
    
    def _identify_bottlenecks(self, metrics_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Analyze retrieval vs generation times
        retrieval_times = []
        generation_times = []
        
        for record in metrics_data:
            metric_value = record.get("metric_value", {})
            retrieval_times.append(metric_value.get("retrieval_time_ms", 0))
            generation_times.append(metric_value.get("generation_time_ms", 0))
        
        if retrieval_times and generation_times:
            avg_retrieval = statistics.mean(retrieval_times)
            avg_generation = statistics.mean(generation_times)
            
            if avg_retrieval > avg_generation * 1.5:
                bottlenecks.append({
                    "component": "retrieval",
                    "avg_time_ms": avg_retrieval,
                    "impact": "high",
                    "description": "Retrieval operations are significantly slower than generation"
                })
            elif avg_generation > avg_retrieval * 1.5:
                bottlenecks.append({
                    "component": "generation", 
                    "avg_time_ms": avg_generation,
                    "impact": "high",
                    "description": "Generation operations are significantly slower than retrieval"
                })
        
        return bottlenecks
    
    def _generate_recommendations(self, metrics_data: List[Dict]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not metrics_data:
            return recommendations
        
        # Analyze cache performance
        cache_hits = sum(1 for record in metrics_data 
                        if record.get("metric_value", {}).get("cache_hit"))
        cache_hit_rate = cache_hits / len(metrics_data)
        
        if cache_hit_rate < 0.6:
            recommendations.append(
                "Consider optimizing cache strategy - current hit rate is below 60%"
            )
        
        # Analyze response times
        response_times = [record.get("metric_value", {}).get("response_time_ms", 0) 
                         for record in metrics_data]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        if avg_response_time > 2000:
            recommendations.append(
                "Average response time is above 2 seconds - consider performance optimization"
            )
        
        return recommendations
    
    def __del__(self):
        """Cleanup background tasks on destruction."""
        if self._flush_task:
            self._flush_task.cancel()
        if self._analysis_task:
            self._analysis_task.cancel()


# Singleton instance management
_analytics_instance: Optional[PromptAnalytics] = None

def get_analytics_instance(supabase_client: Client) -> PromptAnalytics:
    """Get or create analytics singleton instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = PromptAnalytics(supabase_client)
    return _analytics_instance


# Convenience function for tracking metrics
async def track_query_metrics(
    analytics: PromptAnalytics,
    query_id: str,
    query_text: str,
    query_type: Union[QueryType, str],
    classification_confidence: float,
    response_time_ms: float,
    retrieval_time_ms: float = 0.0,
    generation_time_ms: float = 0.0,
    total_tokens: int = 0,
    response_quality_score: float = 0.0,
    cache_hit: bool = False,
    cache_latency_ms: float = 0.0,
    sources_count: int = 0,
    context_length: int = 0,
    relevance_scores: Optional[List[float]] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Convenience function to track query metrics."""
    
    # Convert string to QueryType if needed
    if isinstance(query_type, str):
        try:
            query_type = QueryType(query_type)
        except ValueError:
            query_type = QueryType.GENERAL  # Default fallback
    
    metrics = QueryMetrics(
        query_id=query_id,
        query_text=query_text,
        query_type=query_type,
        timestamp=datetime.utcnow(),
        classification_confidence=classification_confidence,
        response_time_ms=response_time_ms,
        retrieval_time_ms=retrieval_time_ms,
        generation_time_ms=generation_time_ms,
        total_tokens=total_tokens,
        response_quality_score=response_quality_score,
        cache_hit=cache_hit,
        cache_latency_ms=cache_latency_ms,
        sources_count=sources_count,
        context_length=context_length,
        relevance_scores=relevance_scores or [],
        error=error,
        error_type=error_type,
        user_id=user_id,
        session_id=session_id
    )
    
    await analytics.track_query_metrics(metrics) 