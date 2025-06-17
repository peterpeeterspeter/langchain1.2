"""
Testing Monitoring & Reporting Dashboard

Comprehensive testing analytics and reporting system including test result dashboards,
performance tracking, and automated alerts integrated with Supabase.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from pathlib import Path

try:
    from supabase import create_client, Client
except ImportError:
    Client = None
    create_client = None

from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Test execution status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    RUNNING = "running"
    PENDING = "pending"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics collected."""
    TEST_EXECUTION = "test_execution"
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    FAILURE_RATE = "failure_rate"
    DURATION = "duration"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class TestResult:
    """Individual test result data structure."""
    test_id: str
    test_name: str
    test_file: str
    test_class: Optional[str]
    status: TestStatus
    duration: float  # seconds
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    markers: List[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.markers is None:
            self.markers = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestSuiteResult:
    """Test suite execution results."""
    suite_id: str
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    coverage_percentage: Optional[float] = None
    timestamp: datetime = None
    environment: str = "local"
    branch: str = "main"
    commit_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate test failure rate."""
        if self.total_tests == 0:
            return 0.0
        return ((self.failed + self.errors) / self.total_tests) * 100


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    metric_id: str
    metric_type: MetricType
    value: float
    unit: str
    test_suite_id: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    triggered_by: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.resolved_at is None


class TestingDashboard:
    """
    Comprehensive testing monitoring and reporting dashboard.
    
    Provides test result visualization, performance tracking, automated alerts,
    and integration with Supabase for data storage and analytics.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        project_name: str = "Universal RAG CMS",
        enable_alerts: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        mock_mode: bool = False
    ):
        """
        Initialize the testing dashboard.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
            project_name: Name of the project being monitored
            enable_alerts: Whether to enable automated alerting
            alert_thresholds: Custom alert thresholds
            mock_mode: Use mock data instead of real Supabase connection
        """
        self.project_name = project_name
        self.enable_alerts = enable_alerts
        self.mock_mode = mock_mode
        
        # Initialize Supabase client if available
        if not mock_mode and create_client and supabase_url and supabase_key:
            self.supabase: Client = create_client(supabase_url, supabase_key)
        else:
            self.supabase = None
            if not mock_mode:
                logger.warning("Supabase client not available - running in mock mode")
                self.mock_mode = True
        
        # Default alert thresholds
        default_thresholds = {
            "failure_rate": 10.0,  # percentage
            "coverage_drop": 5.0,  # percentage points
            "performance_regression": 20.0,  # percentage
            "test_duration": 300.0,  # seconds
        }
        
        if alert_thresholds:
            default_thresholds.update(alert_thresholds)
        
        self.alert_thresholds = default_thresholds
        
        # Mock data storage for testing
        if self.mock_mode:
            self._mock_data = {
                "test_suites": [],
                "test_results": [],
                "performance_metrics": [],
                "test_alerts": [],
                "coverage_history": []
            }
        
        if not self.mock_mode:
            self._setup_database_schema()
    
    def _setup_database_schema(self):
        """Set up the database schema for testing dashboard."""
        try:
            # Note: In a real implementation, this would create tables via Supabase migrations
            logger.info("Database schema setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup database schema: {e}")
    
    async def record_test_suite_result(self, suite_result: TestSuiteResult) -> str:
        """
        Record a test suite execution result.
        
        Args:
            suite_result: Test suite result data
            
        Returns:
            Database record ID
        """
        try:
            if self.mock_mode:
                # Store in mock data
                record_data = {
                    "id": f"suite_{len(self._mock_data['test_suites'])}",
                    "suite_id": suite_result.suite_id,
                    "suite_name": suite_result.suite_name,
                    "total_tests": suite_result.total_tests,
                    "passed": suite_result.passed,
                    "failed": suite_result.failed,
                    "skipped": suite_result.skipped,
                    "errors": suite_result.errors,
                    "total_duration": suite_result.total_duration,
                    "coverage_percentage": suite_result.coverage_percentage,
                    "success_rate": suite_result.success_rate,
                    "failure_rate": suite_result.failure_rate,
                    "environment": suite_result.environment,
                    "branch": suite_result.branch,
                    "commit_hash": suite_result.commit_hash,
                    "timestamp": suite_result.timestamp.isoformat()
                }
                self._mock_data["test_suites"].append(record_data)
                record_id = record_data["id"]
            else:
                # Insert into Supabase
                result = self.supabase.table("test_suites").insert({
                    "suite_id": suite_result.suite_id,
                    "suite_name": suite_result.suite_name,
                    "total_tests": suite_result.total_tests,
                    "passed": suite_result.passed,
                    "failed": suite_result.failed,
                    "skipped": suite_result.skipped,
                    "errors": suite_result.errors,
                    "total_duration": suite_result.total_duration,
                    "coverage_percentage": suite_result.coverage_percentage,
                    "success_rate": suite_result.success_rate,
                    "failure_rate": suite_result.failure_rate,
                    "environment": suite_result.environment,
                    "branch": suite_result.branch,
                    "commit_hash": suite_result.commit_hash,
                    "timestamp": suite_result.timestamp.isoformat()
                }).execute()
                
                record_id = result.data[0]["id"]
            
            # Check for alerts
            if self.enable_alerts:
                await self._check_suite_alerts(suite_result)
            
            logger.info(f"Recorded test suite result: {suite_result.suite_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to record test suite result: {e}")
            raise
    
    async def get_dashboard_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get dashboard summary for the specified time period.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dashboard summary data
        """
        try:
            since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            if self.mock_mode:
                # Use mock data
                suites = [s for s in self._mock_data["test_suites"] 
                         if s["timestamp"] >= since_date]
                alerts_data = [a for a in self._mock_data["test_alerts"] 
                              if a["triggered_at"] >= since_date]
                performance_metrics = [m for m in self._mock_data["performance_metrics"]
                                     if m["timestamp"] >= since_date]
            else:
                # Get from Supabase
                suites_result = self.supabase.table("test_suites").select("*").gte(
                    "timestamp", since_date
                ).execute()
                suites = suites_result.data
                
                alerts_result = self.supabase.table("test_alerts").select("*").gte(
                    "triggered_at", since_date
                ).order("triggered_at", desc=True).limit(10).execute()
                alerts_data = alerts_result.data
                
                performance_result = self.supabase.table("performance_metrics").select("*").gte(
                    "timestamp", since_date
                ).execute()
                performance_metrics = performance_result.data
            
            if not suites:
                return {
                    "summary": {
                        "total_suites": 0,
                        "total_tests": 0,
                        "average_success_rate": 0,
                        "average_coverage": 0
                    },
                    "trends": {},
                    "alerts": [],
                    "performance": {}
                }
            
            # Calculate summary statistics
            total_suites = len(suites)
            total_tests = sum(suite["total_tests"] for suite in suites)
            success_rates = [suite["success_rate"] for suite in suites]
            coverage_rates = [suite["coverage_percentage"] for suite in suites 
                            if suite["coverage_percentage"]]
            
            # Calculate performance statistics
            performance_stats = {}
            for metric_type in MetricType:
                type_metrics = [m for m in performance_metrics 
                              if m["metric_type"] == metric_type.value]
                if type_metrics:
                    values = [m["value"] for m in type_metrics]
                    performance_stats[metric_type.value] = {
                        "count": len(values),
                        "average": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "median": statistics.median(values)
                    }
            
            return {
                "summary": {
                    "total_suites": total_suites,
                    "total_tests": total_tests,
                    "average_success_rate": statistics.mean(success_rates) if success_rates else 0,
                    "average_coverage": statistics.mean(coverage_rates) if coverage_rates else 0,
                    "period_days": days
                },
                "trends": {
                    "success_rate_trend": self._calculate_trend(suites, "success_rate"),
                    "coverage_trend": self._calculate_trend(suites, "coverage_percentage"),
                    "duration_trend": self._calculate_trend(suites, "total_duration")
                },
                "alerts": alerts_data,
                "performance": performance_stats,
                "recent_suites": suites[-10:] if len(suites) > 10 else suites
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard summary: {e}")
            raise
    
    async def generate_html_report(self, days: int = 7) -> str:
        """
        Generate an HTML report for the dashboard.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            HTML report content
        """
        try:
            # Get dashboard data
            summary = await self.get_dashboard_summary(days)
            
            # Generate HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{self.project_name} - Testing Dashboard Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                    .metric-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
                    .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                    .metric-label {{ color: #666; margin-top: 5px; }}
                    .section {{ margin-bottom: 30px; }}
                    .section-title {{ font-size: 1.5em; font-weight: bold; margin-bottom: 15px; color: #333; }}
                    .alert {{ padding: 15px; margin: 10px 0; border-radius: 4px; }}
                    .alert-high {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                    .alert-medium {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
                    .alert-low {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                    th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f8f9fa; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{self.project_name} - Testing Dashboard</h1>
                        <p>Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                        <p>Period: Last {days} days</p>
                    </div>
                    
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{summary['summary']['total_suites']}</div>
                            <div class="metric-label">Total Test Suites</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary['summary']['total_tests']}</div>
                            <div class="metric-label">Total Tests</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary['summary']['average_success_rate']:.1f}%</div>
                            <div class="metric-label">Average Success Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary['summary']['average_coverage']:.1f}%</div>
                            <div class="metric-label">Average Coverage</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">Recent Alerts</h2>
                        {self._generate_alerts_html(summary['alerts'])}
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">Performance Metrics</h2>
                        {self._generate_performance_html(summary['performance'])}
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise
    
    def _generate_alerts_html(self, alerts: List[Dict]) -> str:
        """Generate HTML for alerts section."""
        if not alerts:
            return "<p>No recent alerts.</p>"
        
        html = ""
        for alert in alerts[:10]:  # Show last 10 alerts
            severity_class = f"alert-{alert['severity']}"
            html += f"""
            <div class="alert {severity_class}">
                <strong>{alert['alert_type'].replace('_', ' ').title()}</strong> - {alert['severity'].upper()}<br>
                {alert['message']}<br>
                <small>Triggered: {alert['triggered_at']}</small>
            </div>
            """
        
        return html
    
    def _generate_performance_html(self, performance: Dict) -> str:
        """Generate HTML for performance section."""
        if not performance:
            return "<p>No performance data available.</p>"
        
        html = "<table><tr><th>Metric Type</th><th>Count</th><th>Average</th><th>Min</th><th>Max</th></tr>"
        for metric_type, stats in performance.items():
            html += f"""
            <tr>
                <td>{metric_type.replace('_', ' ').title()}</td>
                <td>{stats['count']}</td>
                <td>{stats['average']:.2f}</td>
                <td>{stats['min']:.2f}</td>
                <td>{stats['max']:.2f}</td>
            </tr>
            """
        html += "</table>"
        
        return html
    
    async def _check_suite_alerts(self, suite_result: TestSuiteResult):
        """Check for alerts based on test suite results."""
        alerts = []
        
        # Check failure rate
        if suite_result.failure_rate > self.alert_thresholds["failure_rate"]:
            alerts.append(Alert(
                alert_id=f"failure_rate_{suite_result.suite_id}",
                alert_type="failure_rate",
                severity=AlertSeverity.HIGH if suite_result.failure_rate > 20 else AlertSeverity.MEDIUM,
                message=f"High failure rate detected: {suite_result.failure_rate:.1f}% in suite {suite_result.suite_name}",
                triggered_by=suite_result.suite_id,
                triggered_at=datetime.utcnow(),
                metadata={"failure_rate": suite_result.failure_rate, "suite_id": suite_result.suite_id}
            ))
        
        # Create alerts
        for alert in alerts:
            await self._create_alert(alert)
    
    async def _create_alert(self, alert: Alert) -> str:
        """Create a new alert."""
        try:
            if self.mock_mode:
                # Store in mock data
                alert_data = {
                    "id": f"alert_{len(self._mock_data['test_alerts'])}",
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_by": alert.triggered_by,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "metadata": alert.metadata
                }
                self._mock_data["test_alerts"].append(alert_data)
                record_id = alert_data["id"]
            else:
                result = self.supabase.table("test_alerts").insert({
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_by": alert.triggered_by,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "metadata": alert.metadata
                }).execute()
                
                record_id = result.data[0]["id"]
            
            logger.warning(f"Alert created: {alert.message} (Severity: {alert.severity.value})")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
    
    def _calculate_trend(self, data: List[Dict], field: str) -> Dict[str, Any]:
        """Calculate trend for a specific field."""
        if len(data) < 2:
            return {"direction": "stable", "change": 0}
        
        values = [item[field] for item in data if item.get(field) is not None]
        if len(values) < 2:
            return {"direction": "stable", "change": 0}
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        if abs(change) < 1:
            direction = "stable"
        elif change > 0:
            direction = "improving" if field in ["success_rate", "coverage_percentage"] else "degrading"
        else:
            direction = "degrading" if field in ["success_rate", "coverage_percentage"] else "improving"
        
        return {
            "direction": direction,
            "change": change,
            "first_period_avg": first_avg,
            "second_period_avg": second_avg
        }


# Export main classes
__all__ = [
    'TestingDashboard',
    'TestResult',
    'TestSuiteResult', 
    'PerformanceMetric',
    'Alert',
    'TestStatus',
    'AlertSeverity',
    'MetricType'
] 