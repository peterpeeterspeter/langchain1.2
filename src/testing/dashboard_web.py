"""
Testing Dashboard Web Interface

Flask-based web interface for the testing monitoring and reporting dashboard
with real-time updates, interactive charts, and comprehensive analytics.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

try:
    from flask import Flask, render_template, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    from flask_cors import CORS
except ImportError:
    Flask = None
    SocketIO = None
    CORS = None

from .dashboard import TestingDashboard, TestSuiteResult, TestResult, TestStatus, AlertSeverity


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestingDashboardWeb:
    """
    Web interface for the testing monitoring dashboard.
    
    Provides a Flask-based web application with real-time updates,
    interactive charts, and comprehensive testing analytics.
    """
    
    def __init__(
        self,
        dashboard: TestingDashboard,
        host: str = "localhost",
        port: int = 5000,
        debug: bool = False,
        enable_socketio: bool = True
    ):
        """
        Initialize the web dashboard.
        
        Args:
            dashboard: TestingDashboard instance
            host: Host to bind the web server
            port: Port to bind the web server
            debug: Enable Flask debug mode
            enable_socketio: Enable WebSocket support for real-time updates
        """
        self.dashboard = dashboard
        self.host = host
        self.port = port
        self.debug = debug
        self.enable_socketio = enable_socketio
        
        # Check if Flask dependencies are available
        if not Flask:
            logger.warning("Flask dependencies not available. Install with: pip install flask flask-socketio flask-cors")
            self.app = None
            self.socketio = None
            return
        
        # Initialize Flask app
        self.app = Flask(__name__, template_folder=self._get_template_dir())
        self.app.config['SECRET_KEY'] = 'testing-dashboard-secret-key'
        
        # Enable CORS for API endpoints
        CORS(self.app)
        
        # Initialize SocketIO if enabled
        if self.enable_socketio and SocketIO:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self._setup_socketio_handlers()
        else:
            self.socketio = None
        
        # Setup routes
        self._setup_routes()
        
        # Background task for real-time updates
        self._update_interval = 30  # seconds
        self._last_update = datetime.utcnow()
    
    def _get_template_dir(self) -> str:
        """Get the templates directory path."""
        current_dir = Path(__file__).parent
        template_dir = current_dir / "templates"
        
        # Create templates directory if it doesn't exist
        template_dir.mkdir(exist_ok=True)
        
        return str(template_dir)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return self._render_dashboard()
        
        @self.app.route('/api/summary')
        def api_summary():
            """API endpoint for dashboard summary."""
            days = request.args.get('days', 7, type=int)
            try:
                # Run async function in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                summary = loop.run_until_complete(self.dashboard.get_dashboard_summary(days))
                loop.close()
                
                return jsonify(summary)
            except Exception as e:
                logger.error(f"Failed to get dashboard summary: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "dashboard_mode": "mock" if self.dashboard.mock_mode else "live"
            })
    
    def _setup_socketio_handlers(self):
        """Setup SocketIO event handlers."""
        if not self.socketio:
            return
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info("Client connected to dashboard")
            emit('connected', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info("Client disconnected from dashboard")
    
    def _render_dashboard(self) -> str:
        """Render the main dashboard HTML."""
        try:
            # Get initial dashboard data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            summary = loop.run_until_complete(self.dashboard.get_dashboard_summary(7))
            loop.close()
            
            # Simple HTML template
            return self._generate_dashboard_html(summary)
        except Exception as e:
            logger.error(f"Failed to render dashboard: {e}")
            return self._render_error_page(str(e))
    
    def _generate_dashboard_html(self, summary: Dict) -> str:
        """Generate dashboard HTML."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.dashboard.project_name} - Testing Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .section-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        .section-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin: 0;
            color: #333;
        }}
        .section-content {{
            padding: 20px;
        }}
        .alert {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 4px solid;
        }}
        .alert-high {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }}
        .alert-medium {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
            color: #856404;
        }}
        .alert-low {{
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
            color: #0c5460;
        }}
        .no-data {{
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }}
        .performance-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .performance-table th,
        .performance-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .performance-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .refresh-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s ease;
        }}
        .refresh-btn:hover {{
            background: #5a6fd8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.dashboard.project_name} - Testing Dashboard</h1>
            <p>Real-time testing analytics and monitoring</p>
            <button class="refresh-btn" onclick="location.reload()">Refresh Dashboard</button>
        </div>

        <div class="metrics-grid">
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
            <div class="section-header">
                <h2 class="section-title">Recent Alerts</h2>
            </div>
            <div class="section-content">
                {self._generate_alerts_html(summary.get('alerts', []))}
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Performance Metrics</h2>
            </div>
            <div class="section-content">
                {self._generate_performance_table_html(summary.get('performance', {{}}))}
            </div>
        </div>
    </div>
</body>
</html>
        """
    
    def _generate_alerts_html(self, alerts: List[Dict]) -> str:
        """Generate HTML for alerts."""
        if not alerts:
            return '<div class="no-data">No recent alerts</div>'
        
        html = ""
        for alert in alerts[:5]:  # Show last 5 alerts
            severity_class = f"alert-{alert['severity']}"
            html += f"""
            <div class="alert {severity_class}">
                <strong>{alert['alert_type'].replace('_', ' ').title()}</strong> - {alert['severity'].upper()}<br>
                {alert['message']}<br>
                <small>Triggered: {alert['triggered_at']}</small>
            </div>
            """
        
        return html
    
    def _generate_performance_table_html(self, performance: Dict) -> str:
        """Generate HTML for performance table."""
        if not performance:
            return '<div class="no-data">No performance data available</div>'
        
        html = """
        <table class="performance-table">
            <thead>
                <tr>
                    <th>Metric Type</th>
                    <th>Count</th>
                    <th>Average</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
        """
        
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
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _render_error_page(self, error_message: str) -> str:
        """Render an error page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ background: #f8d7da; color: #721c24; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Testing Dashboard Error</h1>
            <div class="error">
                <h3>Error occurred while loading dashboard:</h3>
                <p>{error_message}</p>
            </div>
            <p><a href="/">Try again</a></p>
        </body>
        </html>
        """
    
    def run(self):
        """Run the web dashboard server."""
        if not self.app:
            logger.error("Flask not available - cannot start web dashboard")
            return
        
        logger.info(f"Starting Testing Dashboard Web Interface on {self.host}:{self.port}")
        
        if self.socketio:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
        else:
            self.app.run(host=self.host, port=self.port, debug=self.debug)


# Convenience function to create and run dashboard
def create_dashboard_app(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    project_name: str = "Universal RAG CMS",
    mock_mode: bool = True,
    host: str = "localhost",
    port: int = 5000,
    debug: bool = True
) -> TestingDashboardWeb:
    """
    Create and configure a testing dashboard web application.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase service key
        project_name: Project name for dashboard
        mock_mode: Use mock data instead of real Supabase
        host: Host to bind web server
        port: Port to bind web server
        debug: Enable debug mode
        
    Returns:
        Configured TestingDashboardWeb instance
    """
    # Create dashboard instance
    dashboard = TestingDashboard(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        project_name=project_name,
        mock_mode=mock_mode
    )
    
    # Create web interface
    web_dashboard = TestingDashboardWeb(
        dashboard=dashboard,
        host=host,
        port=port,
        debug=debug
    )
    
    return web_dashboard


# Export main classes
__all__ = [
    'TestingDashboardWeb',
    'create_dashboard_app'
] 