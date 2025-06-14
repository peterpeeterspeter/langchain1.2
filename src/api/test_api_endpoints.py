"""
Comprehensive API Testing Script for Universal RAG CMS API

Tests all endpoints for:
- Configuration Management
- Contextual Retrieval Configuration  
- Monitoring & Analytics
- Performance Profiling
- Feature Flags & A/B Testing
- WebSocket Metrics Streaming

Usage:
    python src/api/test_api_endpoints.py
"""

import asyncio
import aiohttp
import json
import time
import websockets
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    """Comprehensive API testing class."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def test_endpoint(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        expected_status: int = 200,
        description: str = ""
    ) -> Dict[str, Any]:
        """Test a single API endpoint."""
        
        url = f"{self.base_url}{endpoint}"
        test_name = f"{method} {endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                async with self.session.get(url, params=params) as response:
                    response_data = await response.json()
                    status_code = response.status
            elif method.upper() == "POST":
                async with self.session.post(url, json=data, params=params) as response:
                    response_data = await response.json()
                    status_code = response.status
            elif method.upper() == "PUT":
                async with self.session.put(url, json=data, params=params) as response:
                    response_data = await response.json()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            
            # Determine test result
            success = status_code == expected_status
            
            result = {
                "test_name": test_name,
                "description": description,
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": status_code,
                "expected_status": expected_status,
                "success": success,
                "response_time_ms": round(response_time * 1000, 2),
                "response_data": response_data,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(result)
            
            status_emoji = "✅" if success else "❌"
            logger.info(f"{status_emoji} {test_name} - {status_code} ({response_time*1000:.1f}ms)")
            
            return result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "description": description,
                "method": method.upper(),
                "endpoint": endpoint,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(error_result)
            logger.error(f"❌ {test_name} - ERROR: {str(e)}")
            
            return error_result
    
    async def test_websocket_endpoint(self, endpoint: str, duration_seconds: int = 5) -> Dict[str, Any]:
        """Test WebSocket endpoint."""
        
        ws_url = f"ws://localhost:8000{endpoint}"
        test_name = f"WebSocket {endpoint}"
        
        try:
            start_time = time.time()
            messages_received = 0
            
            async with websockets.connect(ws_url) as websocket:
                # Listen for messages for specified duration
                end_time = start_time + duration_seconds
                
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        messages_received += 1
                        logger.debug(f"WebSocket message: {message[:100]}...")
                    except asyncio.TimeoutError:
                        continue
            
            response_time = time.time() - start_time
            success = messages_received > 0
            
            result = {
                "test_name": test_name,
                "description": f"WebSocket connection test ({duration_seconds}s)",
                "endpoint": endpoint,
                "success": success,
                "messages_received": messages_received,
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(result)
            
            status_emoji = "✅" if success else "❌"
            logger.info(f"{status_emoji} {test_name} - {messages_received} messages ({response_time*1000:.1f}ms)")
            
            return result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "description": f"WebSocket connection test ({duration_seconds}s)",
                "endpoint": endpoint,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(error_result)
            logger.error(f"❌ {test_name} - ERROR: {str(e)}")
            
            return error_result
    
    async def run_all_tests(self):
        """Run comprehensive test suite."""
        
        logger.info("🚀 Starting Universal RAG CMS API Test Suite")
        logger.info("=" * 60)
        
        # Test basic endpoints
        await self.test_basic_endpoints()
        
        # Test configuration management
        await self.test_configuration_management()
        
        # Test retrieval configuration
        await self.test_retrieval_configuration()
        
        # Test monitoring and analytics
        await self.test_monitoring_analytics()
        
        # Test performance profiling
        await self.test_performance_profiling()
        
        # Test feature flags
        await self.test_feature_flags()
        
        # Test WebSocket endpoints
        await self.test_websocket_endpoints()
        
        # Generate summary
        self.generate_test_summary()
    
    async def test_basic_endpoints(self):
        """Test basic API endpoints."""
        
        logger.info("\n📋 Testing Basic Endpoints")
        logger.info("-" * 30)
        
        await self.test_endpoint(
            "GET", "/",
            description="Root endpoint with API information"
        )
        
        await self.test_endpoint(
            "GET", "/health",
            description="General health check"
        )
    
    async def test_configuration_management(self):
        """Test configuration management endpoints."""
        
        logger.info("\n⚙️ Testing Configuration Management")
        logger.info("-" * 40)
        
        # Get current configuration
        await self.test_endpoint(
            "GET", "/api/v1/config/prompt-optimization",
            description="Get current prompt optimization configuration"
        )
        
        # Validate configuration
        test_config = {
            "config_data": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "system_prompt": "You are a helpful assistant"
            }
        }
        
        await self.test_endpoint(
            "POST", "/api/v1/config/prompt-optimization/validate",
            data=test_config,
            description="Validate configuration data"
        )
        
        # Get configuration history
        await self.test_endpoint(
            "GET", "/api/v1/config/prompt-optimization/history",
            params={"limit": 5},
            description="Get configuration history"
        )
        
        # Health check
        await self.test_endpoint(
            "GET", "/api/v1/config/health",
            description="Configuration system health check"
        )
    
    async def test_retrieval_configuration(self):
        """Test retrieval configuration endpoints."""
        
        logger.info("\n🎯 Testing Retrieval Configuration")
        logger.info("-" * 40)
        
        # Get retrieval configuration
        await self.test_endpoint(
            "GET", "/retrieval/api/v1/config/",
            params={"environment": "development", "include_sensitive": False},
            description="Get retrieval configuration"
        )
        
        # Validate retrieval configuration
        test_retrieval_config = {
            "retrieval_strategy": "hybrid",
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "mmr_lambda": 0.7,
            "max_results": 10
        }
        
        await self.test_endpoint(
            "POST", "/retrieval/api/v1/config/validate",
            data=test_retrieval_config,
            description="Validate retrieval configuration"
        )
        
        # Get performance profiles
        await self.test_endpoint(
            "GET", "/retrieval/api/v1/config/performance-profiles",
            description="Get available performance profiles"
        )
        
        # Get supported query types
        await self.test_endpoint(
            "GET", "/retrieval/api/v1/config/query-types",
            description="Get supported query types"
        )
        
        # Export configuration
        export_request = {
            "environment": "development",
            "include_sensitive": False,
            "format": "json"
        }
        
        await self.test_endpoint(
            "POST", "/retrieval/api/v1/config/export",
            data=export_request,
            description="Export configuration"
        )
        
        # Configuration health check
        await self.test_endpoint(
            "GET", "/retrieval/api/v1/config/health",
            description="Retrieval configuration health check"
        )
    
    async def test_monitoring_analytics(self):
        """Test monitoring and analytics endpoints."""
        
        logger.info("\n📊 Testing Monitoring & Analytics")
        logger.info("-" * 40)
        
        # Get real-time metrics
        await self.test_endpoint(
            "GET", "/api/v1/config/analytics/real-time",
            params={"window_minutes": 5},
            description="Get real-time metrics"
        )
        
        # Get active alerts
        await self.test_endpoint(
            "GET", "/api/v1/config/analytics/alerts",
            description="Get active alerts"
        )
        
        # Generate performance report
        start_date = (datetime.now() - timedelta(days=1)).isoformat()
        end_date = datetime.now().isoformat()
        
        await self.test_endpoint(
            "POST", "/api/v1/config/analytics/report",
            params={"start_date": start_date, "end_date": end_date},
            description="Generate performance report"
        )
    
    async def test_performance_profiling(self):
        """Test performance profiling endpoints."""
        
        logger.info("\n🎛️ Testing Performance Profiling")
        logger.info("-" * 40)
        
        # Get performance snapshot
        await self.test_endpoint(
            "GET", "/api/v1/config/profiling/snapshot",
            description="Get performance snapshot"
        )
        
        # Get optimization report
        await self.test_endpoint(
            "GET", "/api/v1/config/profiling/optimization-report",
            params={"hours": 24},
            description="Get optimization report"
        )
    
    async def test_feature_flags(self):
        """Test feature flags endpoints."""
        
        logger.info("\n🚀 Testing Feature Flags")
        logger.info("-" * 30)
        
        # List feature flags
        await self.test_endpoint(
            "GET", "/api/v1/config/feature-flags",
            description="List all feature flags"
        )
        
        # Create test feature flag
        test_flag = {
            "name": f"test_flag_{int(time.time())}",
            "description": "Test feature flag for API testing",
            "status": "disabled",
            "rollout_percentage": 0.0,
            "variants": [
                {
                    "name": "control",
                    "weight": 50.0,
                    "config_overrides": {}
                },
                {
                    "name": "test",
                    "weight": 50.0,
                    "config_overrides": {"test_mode": True}
                }
            ]
        }
        
        create_result = await self.test_endpoint(
            "POST", "/api/v1/config/feature-flags",
            data=test_flag,
            expected_status=201,
            description="Create test feature flag"
        )
        
        # If flag was created successfully, test getting it
        if create_result.get("success"):
            flag_name = test_flag["name"]
            
            await self.test_endpoint(
                "GET", f"/api/v1/config/feature-flags/{flag_name}",
                description=f"Get feature flag: {flag_name}"
            )
    
    async def test_websocket_endpoints(self):
        """Test WebSocket endpoints."""
        
        logger.info("\n🔌 Testing WebSocket Endpoints")
        logger.info("-" * 40)
        
        # Test metrics WebSocket
        await self.test_websocket_endpoint(
            "/api/v1/config/ws/metrics",
            duration_seconds=3
        )
    
    def generate_test_summary(self):
        """Generate comprehensive test summary."""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get("success", False)])
        failed_tests = total_tests - successful_tests
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Calculate average response time
        response_times = [r.get("response_time_ms", 0) for r in self.test_results if "response_time_ms" in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        logger.info(f"Average Response Time: {avg_response_time:.1f}ms")
        
        # Show failed tests
        if failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result.get("success", False):
                    test_name = result.get("test_name", "Unknown")
                    error = result.get("error", result.get("status_code", "Unknown error"))
                    logger.info(f"  - {test_name}: {error}")
        
        # Performance insights
        logger.info("\n⚡ PERFORMANCE INSIGHTS:")
        if avg_response_time < 100:
            logger.info("  - Excellent response times (< 100ms)")
        elif avg_response_time < 500:
            logger.info("  - Good response times (< 500ms)")
        elif avg_response_time < 1000:
            logger.info("  - Acceptable response times (< 1s)")
        else:
            logger.info("  - Slow response times (> 1s) - consider optimization")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"api_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "failed_tests": failed_tests,
                    "success_rate": success_rate,
                    "average_response_time_ms": avg_response_time,
                    "test_timestamp": datetime.now().isoformat()
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        logger.info(f"\n📄 Detailed results saved to: {results_file}")
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate
        }


async def main():
    """Main function to run API tests."""
    
    print("🧪 Universal RAG CMS API Test Suite")
    print("=" * 50)
    
    async with APITester() as tester:
        await tester.run_all_tests()
    
    print("\n✨ Testing completed!")


if __name__ == "__main__":
    asyncio.run(main()) 