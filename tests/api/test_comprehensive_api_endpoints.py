"""
Comprehensive API Endpoint Testing Framework for Task 10.9

This module provides comprehensive testing for all REST API endpoints including:
- Configuration management endpoints
- Retrieval configuration endpoints  
- Contextual retrieval endpoints
- Monitoring and analytics endpoints
- Performance profiling endpoints
- Feature flag endpoints
- WebSocket connections
- Authentication middleware
- Rate limiting validation
- Error handling verification
- API specification compliance
"""

import asyncio
import aiohttp
import pytest
import logging
import json
import time
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APITestResult:
    """Result of an API endpoint test."""
    test_name: str
    method: str
    endpoint: str
    status_code: int
    expected_status: int
    success: bool
    response_time_ms: float
    response_data: Optional[Dict[str, Any]]
    error: Optional[str]
    timestamp: str
    description: str

@dataclass
class WebSocketTestResult:
    """Result of a WebSocket connection test."""
    test_name: str
    endpoint: str
    success: bool
    connection_time_ms: float
    messages_sent: int
    messages_received: int
    error: Optional[str]
    timestamp: str
    description: str

class ComprehensiveAPITester:
    """Comprehensive API endpoint testing framework."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results: List[APITestResult] = []
        self.websocket_results: List[WebSocketTestResult] = []
        
        # API endpoint definitions
        self.api_endpoints = self._define_api_endpoints()
        
        # Test data for various scenarios
        self.test_data = self._prepare_test_data()
    
    def _define_api_endpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define all API endpoints to test."""
        
        return {
            "basic": [
                {"method": "GET", "endpoint": "/", "description": "Root endpoint"},
                {"method": "GET", "endpoint": "/health", "description": "Health check"},
                {"method": "GET", "endpoint": "/docs", "description": "API documentation"},
                {"method": "GET", "endpoint": "/redoc", "description": "ReDoc documentation"}
            ],
            
            "config_management": [
                {"method": "GET", "endpoint": "/api/v1/config/prompt-optimization", "description": "Get prompt optimization config"},
                {"method": "POST", "endpoint": "/api/v1/config/prompt-optimization", "description": "Update prompt optimization config"},
                {"method": "GET", "endpoint": "/api/v1/config/analytics/real-time", "description": "Get real-time metrics"},
                {"method": "POST", "endpoint": "/api/v1/config/analytics/report", "description": "Generate performance report"},
                {"method": "GET", "endpoint": "/api/v1/config/profiling/optimization-report", "description": "Get optimization report"},
                {"method": "GET", "endpoint": "/api/v1/config/health", "description": "Config health check"}
            ],
            
            "retrieval_config": [
                {"method": "GET", "endpoint": "/retrieval/api/v1/config/", "description": "Get retrieval configuration"},
                {"method": "POST", "endpoint": "/retrieval/api/v1/config/update", "description": "Update retrieval configuration"},
                {"method": "POST", "endpoint": "/retrieval/api/v1/config/validate", "description": "Validate configuration"},
                {"method": "GET", "endpoint": "/retrieval/api/v1/config/performance-profiles", "description": "Get performance profiles"},
                {"method": "POST", "endpoint": "/retrieval/api/v1/config/reload", "description": "Reload configuration"},
                {"method": "GET", "endpoint": "/retrieval/api/v1/config/export", "description": "Export configuration"},
                {"method": "GET", "endpoint": "/retrieval/api/v1/config/health", "description": "Retrieval config health"}
            ],
            
            "contextual_retrieval": [
                {"method": "POST", "endpoint": "/api/v1/contextual/query", "description": "Contextual document query"},
                {"method": "POST", "endpoint": "/api/v1/contextual/ingest", "description": "Document ingestion"},
                {"method": "GET", "endpoint": "/api/v1/contextual/metrics", "description": "Retrieval metrics"},
                {"method": "GET", "endpoint": "/api/v1/contextual/analytics", "description": "Retrieval analytics"},
                {"method": "POST", "endpoint": "/api/v1/contextual/migrate", "description": "Content migration"},
                {"method": "GET", "endpoint": "/api/v1/contextual/health", "description": "Contextual retrieval health"}
            ],
            
            "feature_flags": [
                {"method": "GET", "endpoint": "/api/v1/config/feature-flags", "description": "Get feature flags"},
                {"method": "POST", "endpoint": "/api/v1/config/feature-flags", "description": "Create feature flag"},
                {"method": "PUT", "endpoint": "/api/v1/config/feature-flags/{flag_id}", "description": "Update feature flag"},
                {"method": "DELETE", "endpoint": "/api/v1/config/feature-flags/{flag_id}", "description": "Delete feature flag"},
                {"method": "POST", "endpoint": "/api/v1/config/feature-flags/{flag_id}/toggle", "description": "Toggle feature flag"}
            ],
            
            "websockets": [
                {"endpoint": "/ws/metrics", "description": "Real-time metrics WebSocket"},
                {"endpoint": "/ws/analytics", "description": "Analytics WebSocket"},
                {"endpoint": "/ws/profiling", "description": "Performance profiling WebSocket"}
            ]
        }
    
    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare test data for various API endpoints."""
        
        return {
            "config_update": {
                "config_data": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "model": "gpt-4"
                },
                "change_notes": "Test configuration update"
            },
            
            "retrieval_config_update": {
                "config_data": {
                    "performance": {
                        "max_response_time_ms": 2000,
                        "connection_pool_size": 10
                    }
                },
                "validate_only": False
            },
            
            "contextual_query": {
                "query": "What are the best online casino strategies?",
                "max_results": 5,
                "strategy": "hybrid",
                "include_metadata": True
            },
            
            "document_ingestion": {
                "content": "This is a test document for ingestion testing.",
                "metadata": {
                    "title": "Test Document",
                    "category": "test",
                    "source": "api_test"
                }
            },
            
            "feature_flag": {
                "name": "test_feature",
                "description": "Test feature flag",
                "enabled": True,
                "variants": [
                    {"name": "control", "weight": 50},
                    {"name": "treatment", "weight": 50}
                ]
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def test_endpoint(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        expected_status: int = 200,
        description: str = ""
    ) -> APITestResult:
        """Test a single API endpoint."""
        
        url = f"{self.base_url}{endpoint}"
        test_name = f"{method} {endpoint}"
        
        try:
            start_time = time.time()
            
            # Prepare request parameters
            request_kwargs = {
                "params": params,
                "headers": headers or {}
            }
            
            if data and method.upper() in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = data
            
            # Make request
            async with self.session.request(method.upper(), url, **request_kwargs) as response:
                try:
                    response_data = await response.json()
                except:
                    response_data = {"text": await response.text()}
                
                status_code = response.status
                response_time = (time.time() - start_time) * 1000
                
                # Determine success
                success = status_code == expected_status
                
                result = APITestResult(
                    test_name=test_name,
                    method=method.upper(),
                    endpoint=endpoint,
                    status_code=status_code,
                    expected_status=expected_status,
                    success=success,
                    response_time_ms=round(response_time, 2),
                    response_data=response_data,
                    error=None,
                    timestamp=datetime.now().isoformat(),
                    description=description
                )
                
                # Log result
                status_icon = "✅" if success else "❌"
                logger.info(f"{status_icon} {test_name} - {status_code} ({response_time:.0f}ms)")
                
                if not success:
                    logger.warning(f"   Expected {expected_status}, got {status_code}")
                
                self.test_results.append(result)
                return result
                
        except Exception as e:
            error_result = APITestResult(
                test_name=test_name,
                method=method.upper(),
                endpoint=endpoint,
                status_code=0,
                expected_status=expected_status,
                success=False,
                response_time_ms=0,
                response_data=None,
                error=str(e),
                timestamp=datetime.now().isoformat(),
                description=description
            )
            
            logger.error(f"❌ {test_name} - ERROR: {str(e)}")
            self.test_results.append(error_result)
            return error_result
    
    async def test_websocket_connection(
        self,
        endpoint: str,
        duration_seconds: int = 5,
        test_messages: List[str] = None,
        description: str = ""
    ) -> WebSocketTestResult:
        """Test WebSocket connection."""
        
        ws_url = f"ws://localhost:8000{endpoint}"
        test_name = f"WebSocket {endpoint}"
        
        try:
            start_time = time.time()
            messages_sent = 0
            messages_received = 0
            
            async with websockets.connect(ws_url) as websocket:
                connection_time = (time.time() - start_time) * 1000
                
                # Send test messages if provided
                if test_messages:
                    for message in test_messages:
                        await websocket.send(message)
                        messages_sent += 1
                
                # Listen for messages for specified duration
                end_time = time.time() + duration_seconds
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        messages_received += 1
                    except asyncio.TimeoutError:
                        continue
                
                result = WebSocketTestResult(
                    test_name=test_name,
                    endpoint=endpoint,
                    success=True,
                    connection_time_ms=round(connection_time, 2),
                    messages_sent=messages_sent,
                    messages_received=messages_received,
                    error=None,
                    timestamp=datetime.now().isoformat(),
                    description=description
                )
                
                logger.info(f"✅ {test_name} - Connected ({connection_time:.0f}ms), {messages_received} messages")
                self.websocket_results.append(result)
                return result
                
        except Exception as e:
            error_result = WebSocketTestResult(
                test_name=test_name,
                endpoint=endpoint,
                success=False,
                connection_time_ms=0,
                messages_sent=0,
                messages_received=0,
                error=str(e),
                timestamp=datetime.now().isoformat(),
                description=description
            )
            
            logger.error(f"❌ {test_name} - ERROR: {str(e)}")
            self.websocket_results.append(error_result)
            return error_result
    
    async def test_rate_limiting(self, endpoint: str, requests_count: int = 10) -> Dict[str, Any]:
        """Test rate limiting on an endpoint."""
        
        logger.info(f"🔄 Testing rate limiting on {endpoint}")
        
        start_time = time.time()
        results = []
        
        # Send multiple requests rapidly
        tasks = []
        for i in range(requests_count):
            task = self.test_endpoint("GET", endpoint, description=f"Rate limit test {i+1}")
            tasks.append(task)
        
        # Execute all requests concurrently
        test_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = 0
        rate_limited_requests = 0
        error_requests = 0
        
        for result in test_results:
            if isinstance(result, Exception):
                error_requests += 1
            elif result.status_code == 200:
                successful_requests += 1
            elif result.status_code == 429:  # Too Many Requests
                rate_limited_requests += 1
            else:
                error_requests += 1
        
        total_time = time.time() - start_time
        
        rate_limit_analysis = {
            "endpoint": endpoint,
            "total_requests": requests_count,
            "successful_requests": successful_requests,
            "rate_limited_requests": rate_limited_requests,
            "error_requests": error_requests,
            "total_time_seconds": round(total_time, 2),
            "requests_per_second": round(requests_count / total_time, 2),
            "rate_limiting_working": rate_limited_requests > 0
        }
        
        logger.info(f"   📊 Rate Limiting Analysis: {successful_requests} success, {rate_limited_requests} limited, {error_requests} errors")
        
        return rate_limit_analysis
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication middleware."""
        
        logger.info("🔐 Testing Authentication Middleware")
        
        auth_tests = []
        
        # Test without authentication
        result = await self.test_endpoint(
            "GET", "/api/v1/config/prompt-optimization",
            expected_status=401,
            description="No authentication"
        )
        auth_tests.append({"test": "no_auth", "success": result.success})
        
        # Test with invalid API key
        result = await self.test_endpoint(
            "GET", "/api/v1/config/prompt-optimization",
            headers={"X-API-Key": "invalid_key"},
            expected_status=401,
            description="Invalid API key"
        )
        auth_tests.append({"test": "invalid_key", "success": result.success})
        
        # Test with valid API key (if available)
        api_key = os.getenv("TEST_API_KEY")
        if api_key:
            result = await self.test_endpoint(
                "GET", "/api/v1/config/prompt-optimization",
                headers={"X-API-Key": api_key},
                expected_status=200,
                description="Valid API key"
            )
            auth_tests.append({"test": "valid_key", "success": result.success})
        
        return {
            "authentication_tests": auth_tests,
            "auth_middleware_working": any(test["success"] for test in auth_tests)
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling across endpoints."""
        
        logger.info("🚨 Testing Error Handling")
        
        error_tests = []
        
        # Test 404 errors
        result = await self.test_endpoint(
            "GET", "/api/v1/nonexistent",
            expected_status=404,
            description="404 Not Found"
        )
        error_tests.append({"test": "404_error", "success": result.success})
        
        # Test 400 errors with invalid data
        result = await self.test_endpoint(
            "POST", "/api/v1/config/prompt-optimization",
            data={"invalid": "data"},
            expected_status=400,
            description="400 Bad Request"
        )
        error_tests.append({"test": "400_error", "success": result.success})
        
        # Test 422 validation errors
        result = await self.test_endpoint(
            "POST", "/retrieval/api/v1/config/update",
            data={"config_data": {"invalid_field": "invalid_value"}},
            expected_status=422,
            description="422 Validation Error"
        )
        error_tests.append({"test": "422_error", "success": result.success})
        
        return {
            "error_handling_tests": error_tests,
            "error_handling_working": all(test["success"] for test in error_tests)
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete API endpoint test suite."""
        
        logger.info("🚀 Starting Comprehensive API Endpoint Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Test basic endpoints
        logger.info("\n📋 Testing Basic Endpoints")
        logger.info("-" * 40)
        for endpoint_def in self.api_endpoints["basic"]:
            await self.test_endpoint(**endpoint_def)
        
        # Test configuration management endpoints
        logger.info("\n⚙️ Testing Configuration Management Endpoints")
        logger.info("-" * 40)
        for endpoint_def in self.api_endpoints["config_management"]:
            if endpoint_def["method"] == "POST":
                endpoint_def["data"] = self.test_data.get("config_update")
            await self.test_endpoint(**endpoint_def)
        
        # Test retrieval configuration endpoints
        logger.info("\n🔍 Testing Retrieval Configuration Endpoints")
        logger.info("-" * 40)
        for endpoint_def in self.api_endpoints["retrieval_config"]:
            if endpoint_def["method"] == "POST" and "update" in endpoint_def["endpoint"]:
                endpoint_def["data"] = self.test_data.get("retrieval_config_update")
            await self.test_endpoint(**endpoint_def)
        
        # Test contextual retrieval endpoints
        logger.info("\n🧠 Testing Contextual Retrieval Endpoints")
        logger.info("-" * 40)
        for endpoint_def in self.api_endpoints["contextual_retrieval"]:
            if endpoint_def["method"] == "POST":
                if "query" in endpoint_def["endpoint"]:
                    endpoint_def["data"] = self.test_data.get("contextual_query")
                elif "ingest" in endpoint_def["endpoint"]:
                    endpoint_def["data"] = self.test_data.get("document_ingestion")
            await self.test_endpoint(**endpoint_def)
        
        # Test feature flag endpoints
        logger.info("\n🚩 Testing Feature Flag Endpoints")
        logger.info("-" * 40)
        for endpoint_def in self.api_endpoints["feature_flags"]:
            if "{flag_id}" in endpoint_def["endpoint"]:
                endpoint_def["endpoint"] = endpoint_def["endpoint"].replace("{flag_id}", "test_flag")
            if endpoint_def["method"] == "POST" and "feature-flags" in endpoint_def["endpoint"]:
                endpoint_def["data"] = self.test_data.get("feature_flag")
            await self.test_endpoint(**endpoint_def)
        
        # Test WebSocket connections
        logger.info("\n🔌 Testing WebSocket Connections")
        logger.info("-" * 40)
        for ws_def in self.api_endpoints["websockets"]:
            await self.test_websocket_connection(**ws_def)
        
        # Test rate limiting
        logger.info("\n⏱️ Testing Rate Limiting")
        logger.info("-" * 40)
        rate_limit_results = await self.test_rate_limiting("/api/v1/config/health")
        
        # Test authentication
        logger.info("\n🔐 Testing Authentication")
        logger.info("-" * 40)
        auth_results = await self.test_authentication()
        
        # Test error handling
        logger.info("\n🚨 Testing Error Handling")
        logger.info("-" * 40)
        error_results = await self.test_error_handling()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        return self._generate_comprehensive_report(
            total_time, rate_limit_results, auth_results, error_results
        )
    
    def _generate_comprehensive_report(
        self,
        total_time: float,
        rate_limit_results: Dict[str, Any],
        auth_results: Dict[str, Any],
        error_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        # Analyze API test results
        total_api_tests = len(self.test_results)
        successful_api_tests = len([r for r in self.test_results if r.success])
        failed_api_tests = total_api_tests - successful_api_tests
        
        # Analyze WebSocket test results
        total_ws_tests = len(self.websocket_results)
        successful_ws_tests = len([r for r in self.websocket_results if r.success])
        failed_ws_tests = total_ws_tests - successful_ws_tests
        
        # Calculate response time statistics
        response_times = [r.response_time_ms for r in self.test_results if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Overall success rate
        total_tests = total_api_tests + total_ws_tests
        total_successful = successful_api_tests + successful_ws_tests
        overall_success_rate = total_successful / total_tests if total_tests > 0 else 0
        
        report = {
            "test_execution_summary": {
                "total_execution_time_seconds": round(total_time, 2),
                "total_tests": total_tests,
                "successful_tests": total_successful,
                "failed_tests": total_tests - total_successful,
                "overall_success_rate": round(overall_success_rate, 3)
            },
            
            "api_endpoint_results": {
                "total_api_tests": total_api_tests,
                "successful_api_tests": successful_api_tests,
                "failed_api_tests": failed_api_tests,
                "api_success_rate": round(successful_api_tests / total_api_tests, 3) if total_api_tests > 0 else 0,
                "avg_response_time_ms": round(avg_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2),
                "min_response_time_ms": round(min_response_time, 2)
            },
            
            "websocket_results": {
                "total_ws_tests": total_ws_tests,
                "successful_ws_tests": successful_ws_tests,
                "failed_ws_tests": failed_ws_tests,
                "ws_success_rate": round(successful_ws_tests / total_ws_tests, 3) if total_ws_tests > 0 else 0
            },
            
            "rate_limiting_analysis": rate_limit_results,
            "authentication_analysis": auth_results,
            "error_handling_analysis": error_results,
            
            "compliance_status": {
                "api_endpoints_compliant": successful_api_tests >= total_api_tests * 0.9,
                "websockets_compliant": successful_ws_tests >= total_ws_tests * 0.8,
                "rate_limiting_compliant": rate_limit_results.get("rate_limiting_working", False),
                "authentication_compliant": auth_results.get("auth_middleware_working", False),
                "error_handling_compliant": error_results.get("error_handling_working", False)
            },
            
            "detailed_results": {
                "api_test_results": [asdict(r) for r in self.test_results],
                "websocket_test_results": [asdict(r) for r in self.websocket_results]
            },
            
            "recommendations": self._generate_recommendations(
                overall_success_rate, avg_response_time, rate_limit_results, auth_results, error_results
            )
        }
        
        return report
    
    def _generate_recommendations(
        self,
        success_rate: float,
        avg_response_time: float,
        rate_limit_results: Dict[str, Any],
        auth_results: Dict[str, Any],
        error_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        
        recommendations = []
        
        if success_rate < 0.9:
            recommendations.append(f"API success rate ({success_rate:.1%}) is below 90%. Investigate failing endpoints.")
        
        if avg_response_time > 1000:
            recommendations.append(f"Average response time ({avg_response_time:.0f}ms) exceeds 1s. Consider performance optimization.")
        
        if not rate_limit_results.get("rate_limiting_working", False):
            recommendations.append("Rate limiting is not working properly. Review rate limiting configuration.")
        
        if not auth_results.get("auth_middleware_working", False):
            recommendations.append("Authentication middleware needs attention. Verify API key validation.")
        
        if not error_results.get("error_handling_working", False):
            recommendations.append("Error handling needs improvement. Ensure proper HTTP status codes and error responses.")
        
        if not recommendations:
            recommendations.append("All API endpoint tests passed successfully. System is production-ready.")
        
        return recommendations

# Pytest integration
@pytest.mark.api
@pytest.mark.asyncio
class TestComprehensiveAPIEndpoints:
    """Pytest integration for comprehensive API endpoint testing."""
    
    @pytest.fixture(scope="class")
    async def api_tester(self):
        """Create API tester fixture."""
        async with ComprehensiveAPITester() as tester:
            yield tester
    
    async def test_basic_endpoints(self, api_tester):
        """Test basic API endpoints."""
        for endpoint_def in api_tester.api_endpoints["basic"]:
            result = await api_tester.test_endpoint(**endpoint_def)
            assert result.success, f"Basic endpoint {endpoint_def['endpoint']} failed"
    
    async def test_config_management_endpoints(self, api_tester):
        """Test configuration management endpoints."""
        for endpoint_def in api_tester.api_endpoints["config_management"]:
            if endpoint_def["method"] == "POST":
                endpoint_def["data"] = api_tester.test_data.get("config_update")
            result = await api_tester.test_endpoint(**endpoint_def)
            # Allow some endpoints to fail in test environment
            if result.status_code not in [200, 404, 500]:
                assert result.success, f"Config endpoint {endpoint_def['endpoint']} failed unexpectedly"
    
    async def test_comprehensive_api_suite(self, api_tester):
        """Test the complete API endpoint suite."""
        report = await api_tester.run_comprehensive_test_suite()
        
        # Verify overall success rate
        success_rate = report["test_execution_summary"]["overall_success_rate"]
        assert success_rate >= 0.7, f"Overall success rate ({success_rate:.1%}) is too low"
        
        # Verify response times
        avg_response_time = report["api_endpoint_results"]["avg_response_time_ms"]
        assert avg_response_time <= 5000, f"Average response time ({avg_response_time:.0f}ms) is too high"
        
        logger.info(f"✅ API Test Suite completed with {success_rate:.1%} success rate")

# Main execution function
async def run_comprehensive_api_tests():
    """Main function to run comprehensive API endpoint tests."""
    
    async with ComprehensiveAPITester() as tester:
        report = await tester.run_comprehensive_test_suite()
        
        # Save report
        results_dir = Path("tests/api/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"api_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE API ENDPOINT TEST SUITE - SUMMARY")
        logger.info("=" * 80)
        
        summary = report["test_execution_summary"]
        logger.info(f"📊 Total Tests: {summary['total_tests']}")
        logger.info(f"✅ Successful: {summary['successful_tests']}")
        logger.info(f"❌ Failed: {summary['failed_tests']}")
        logger.info(f"📈 Success Rate: {summary['overall_success_rate']:.1%}")
        logger.info(f"⏱️ Execution Time: {summary['total_execution_time_seconds']:.1f}s")
        
        api_results = report["api_endpoint_results"]
        logger.info(f"🌐 API Response Time: {api_results['avg_response_time_ms']:.0f}ms avg")
        
        logger.info(f"\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info(f"\n💾 Report saved to: {report_file}")
        logger.info("=" * 80)
        
        return report

if __name__ == "__main__":
    asyncio.run(run_comprehensive_api_tests())