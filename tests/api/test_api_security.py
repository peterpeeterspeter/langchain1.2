"""
API Security Testing Framework for Task 10.9

This module provides specialized security testing for API endpoints including:
- Authentication and authorization testing
- Input validation and injection attacks
- Rate limiting and DDoS protection
- CORS policy validation
- Security headers verification
- SQL injection and XSS protection
- API key management testing
- Session security validation
"""

import asyncio
import aiohttp
import pytest
import logging
import json
import time
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityTestResult:
    """Result of a security test."""
    test_name: str
    test_category: str
    endpoint: str
    vulnerability_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    success: bool
    details: str
    recommendation: str
    timestamp: str

class APISecurityTester:
    """Comprehensive API security testing framework."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.security_results: List[SecurityTestResult] = []
        
        # Security test payloads
        self.security_payloads = self._prepare_security_payloads()
        
        # Common endpoints to test
        self.test_endpoints = [
            "/api/v1/config/prompt-optimization",
            "/api/v1/contextual/query",
            "/retrieval/api/v1/config/",
            "/api/v1/config/feature-flags"
        ]
    
    def _prepare_security_payloads(self) -> Dict[str, List[str]]:
        """Prepare security test payloads for various attack vectors."""
        
        return {
            "sql_injection": [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM users --",
                "admin'--",
                "' OR 1=1#"
            ],
            
            "xss_payloads": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "';alert('XSS');//",
                "<svg onload=alert('XSS')>"
            ],
            
            "command_injection": [
                "; ls -la",
                "| whoami",
                "&& cat /etc/passwd",
                "`id`",
                "$(whoami)"
            ],
            
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "....//....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
            ],
            
            "ldap_injection": [
                "*)(uid=*",
                "*)(|(password=*))",
                "admin)(&(password=*))",
                "*))%00"
            ],
            
            "nosql_injection": [
                "{'$ne': null}",
                "{'$gt': ''}",
                "{'$where': 'this.password.match(/.*/)'}",
                "{'$regex': '.*'}"
            ]
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
    
    def _add_security_result(
        self,
        test_name: str,
        test_category: str,
        endpoint: str,
        vulnerability_type: str,
        severity: str,
        success: bool,
        details: str,
        recommendation: str
    ):
        """Add a security test result."""
        
        result = SecurityTestResult(
            test_name=test_name,
            test_category=test_category,
            endpoint=endpoint,
            vulnerability_type=vulnerability_type,
            severity=severity,
            success=success,
            details=details,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
        self.security_results.append(result)
        
        # Log result
        status_icon = "🔒" if success else "🚨"
        severity_icon = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "💀"}.get(severity, "⚪")
        
        logger.info(f"{status_icon} {severity_icon} {test_name} - {vulnerability_type}")
        if not success:
            logger.warning(f"   VULNERABILITY: {details}")
    
    async def test_authentication_bypass(self) -> None:
        """Test for authentication bypass vulnerabilities."""
        
        logger.info("🔐 Testing Authentication Bypass")
        
        for endpoint in self.test_endpoints:
            # Test without authentication
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        self._add_security_result(
                            test_name=f"Authentication Bypass - {endpoint}",
                            test_category="Authentication",
                            endpoint=endpoint,
                            vulnerability_type="Authentication Bypass",
                            severity="HIGH",
                            success=False,
                            details=f"Endpoint {endpoint} accessible without authentication",
                            recommendation="Implement proper authentication middleware"
                        )
                    else:
                        self._add_security_result(
                            test_name=f"Authentication Required - {endpoint}",
                            test_category="Authentication",
                            endpoint=endpoint,
                            vulnerability_type="Authentication Bypass",
                            severity="LOW",
                            success=True,
                            details=f"Endpoint {endpoint} properly requires authentication",
                            recommendation="Continue monitoring authentication requirements"
                        )
            except Exception as e:
                logger.error(f"Error testing authentication bypass on {endpoint}: {e}")
    
    async def test_authorization_flaws(self) -> None:
        """Test for authorization and privilege escalation flaws."""
        
        logger.info("👤 Testing Authorization Flaws")
        
        # Test with different user roles
        test_headers = [
            {"X-API-Key": "user_key", "X-User-Role": "user"},
            {"X-API-Key": "admin_key", "X-User-Role": "admin"},
            {"X-API-Key": "guest_key", "X-User-Role": "guest"}
        ]
        
        for endpoint in self.test_endpoints:
            for headers in test_headers:
                try:
                    async with self.session.get(f"{self.base_url}{endpoint}", headers=headers) as response:
                        # Check if low-privilege users can access admin endpoints
                        if "admin" in endpoint.lower() and headers.get("X-User-Role") != "admin" and response.status == 200:
                            self._add_security_result(
                                test_name=f"Privilege Escalation - {endpoint}",
                                test_category="Authorization",
                                endpoint=endpoint,
                                vulnerability_type="Privilege Escalation",
                                severity="HIGH",
                                success=False,
                                details=f"Non-admin user can access admin endpoint {endpoint}",
                                recommendation="Implement proper role-based access control"
                            )
                except Exception as e:
                    logger.error(f"Error testing authorization on {endpoint}: {e}")
    
    async def test_input_validation(self) -> None:
        """Test input validation against various injection attacks."""
        
        logger.info("🛡️ Testing Input Validation")
        
        for endpoint in self.test_endpoints:
            if "POST" in endpoint or "PUT" in endpoint:
                continue  # Skip for now, focus on GET parameters
            
            # Test SQL injection
            for payload in self.security_payloads["sql_injection"]:
                try:
                    params = {"query": payload, "search": payload, "filter": payload}
                    async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                        response_text = await response.text()
                        
                        # Check for SQL error messages
                        sql_errors = ["SQL syntax", "mysql_fetch", "ORA-", "PostgreSQL", "sqlite_"]
                        if any(error in response_text.lower() for error in sql_errors):
                            self._add_security_result(
                                test_name=f"SQL Injection - {endpoint}",
                                test_category="Input Validation",
                                endpoint=endpoint,
                                vulnerability_type="SQL Injection",
                                severity="CRITICAL",
                                success=False,
                                details=f"SQL injection vulnerability detected with payload: {payload}",
                                recommendation="Implement parameterized queries and input sanitization"
                            )
                except Exception as e:
                    logger.error(f"Error testing SQL injection on {endpoint}: {e}")
            
            # Test XSS
            for payload in self.security_payloads["xss_payloads"]:
                try:
                    params = {"message": payload, "content": payload, "description": payload}
                    async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                        response_text = await response.text()
                        
                        # Check if payload is reflected without encoding
                        if payload in response_text:
                            self._add_security_result(
                                test_name=f"XSS Vulnerability - {endpoint}",
                                test_category="Input Validation",
                                endpoint=endpoint,
                                vulnerability_type="Cross-Site Scripting",
                                severity="HIGH",
                                success=False,
                                details=f"XSS vulnerability detected with payload: {payload}",
                                recommendation="Implement proper output encoding and CSP headers"
                            )
                except Exception as e:
                    logger.error(f"Error testing XSS on {endpoint}: {e}")
    
    async def test_rate_limiting_security(self) -> None:
        """Test rate limiting for DDoS protection."""
        
        logger.info("⏱️ Testing Rate Limiting Security")
        
        for endpoint in self.test_endpoints:
            # Send rapid requests
            start_time = time.time()
            request_count = 50
            successful_requests = 0
            rate_limited_requests = 0
            
            tasks = []
            for i in range(request_count):
                task = self._make_request("GET", endpoint)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    if result.get("status") == 200:
                        successful_requests += 1
                    elif result.get("status") == 429:
                        rate_limited_requests += 1
            
            total_time = time.time() - start_time
            requests_per_second = request_count / total_time
            
            # Evaluate rate limiting effectiveness
            if rate_limited_requests == 0 and requests_per_second > 10:
                self._add_security_result(
                    test_name=f"Rate Limiting Bypass - {endpoint}",
                    test_category="Rate Limiting",
                    endpoint=endpoint,
                    vulnerability_type="DDoS Vulnerability",
                    severity="MEDIUM",
                    success=False,
                    details=f"No rate limiting detected. {requests_per_second:.1f} req/s allowed",
                    recommendation="Implement rate limiting to prevent DDoS attacks"
                )
            else:
                self._add_security_result(
                    test_name=f"Rate Limiting Active - {endpoint}",
                    test_category="Rate Limiting",
                    endpoint=endpoint,
                    vulnerability_type="DDoS Protection",
                    severity="LOW",
                    success=True,
                    details=f"Rate limiting working. {rate_limited_requests} requests limited",
                    recommendation="Continue monitoring rate limiting effectiveness"
                )
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a single request and return result."""
        try:
            async with self.session.request(method, f"{self.base_url}{endpoint}", **kwargs) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "text": await response.text()
                }
        except Exception as e:
            return {"error": str(e)}
    
    async def test_cors_policy(self) -> None:
        """Test CORS policy configuration."""
        
        logger.info("🌐 Testing CORS Policy")
        
        # Test CORS with different origins
        test_origins = [
            "https://malicious-site.com",
            "http://localhost:3000",
            "https://trusted-domain.com"
        ]
        
        for endpoint in self.test_endpoints:
            for origin in test_origins:
                try:
                    headers = {"Origin": origin}
                    async with self.session.options(f"{self.base_url}{endpoint}", headers=headers) as response:
                        cors_headers = response.headers
                        
                        # Check for overly permissive CORS
                        if cors_headers.get("Access-Control-Allow-Origin") == "*":
                            self._add_security_result(
                                test_name=f"Permissive CORS - {endpoint}",
                                test_category="CORS",
                                endpoint=endpoint,
                                vulnerability_type="CORS Misconfiguration",
                                severity="MEDIUM",
                                success=False,
                                details="CORS allows all origins (*)",
                                recommendation="Restrict CORS to specific trusted domains"
                            )
                        
                        # Check for credentials with wildcard origin
                        if (cors_headers.get("Access-Control-Allow-Origin") == "*" and 
                            cors_headers.get("Access-Control-Allow-Credentials") == "true"):
                            self._add_security_result(
                                test_name=f"CORS Credentials Risk - {endpoint}",
                                test_category="CORS",
                                endpoint=endpoint,
                                vulnerability_type="CORS Credentials Exposure",
                                severity="HIGH",
                                success=False,
                                details="CORS allows credentials with wildcard origin",
                                recommendation="Never allow credentials with wildcard origin"
                            )
                except Exception as e:
                    logger.error(f"Error testing CORS on {endpoint}: {e}")
    
    async def test_security_headers(self) -> None:
        """Test for security headers."""
        
        logger.info("🛡️ Testing Security Headers")
        
        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=",
            "Content-Security-Policy": "default-src"
        }
        
        for endpoint in self.test_endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    headers = response.headers
                    
                    for header_name, expected_value in required_headers.items():
                        header_value = headers.get(header_name, "")
                        
                        if not header_value:
                            self._add_security_result(
                                test_name=f"Missing Security Header - {header_name}",
                                test_category="Security Headers",
                                endpoint=endpoint,
                                vulnerability_type="Missing Security Header",
                                severity="MEDIUM",
                                success=False,
                                details=f"Missing {header_name} header",
                                recommendation=f"Add {header_name} header for security"
                            )
                        elif isinstance(expected_value, list):
                            if not any(val in header_value for val in expected_value):
                                self._add_security_result(
                                    test_name=f"Weak Security Header - {header_name}",
                                    test_category="Security Headers",
                                    endpoint=endpoint,
                                    vulnerability_type="Weak Security Header",
                                    severity="LOW",
                                    success=False,
                                    details=f"Weak {header_name}: {header_value}",
                                    recommendation=f"Strengthen {header_name} header"
                                )
                        elif expected_value not in header_value:
                            self._add_security_result(
                                test_name=f"Weak Security Header - {header_name}",
                                test_category="Security Headers",
                                endpoint=endpoint,
                                vulnerability_type="Weak Security Header",
                                severity="LOW",
                                success=False,
                                details=f"Weak {header_name}: {header_value}",
                                recommendation=f"Strengthen {header_name} header"
                            )
            except Exception as e:
                logger.error(f"Error testing security headers on {endpoint}: {e}")
    
    async def test_api_key_security(self) -> None:
        """Test API key security implementation."""
        
        logger.info("🔑 Testing API Key Security")
        
        # Test weak API keys
        weak_keys = [
            "123456",
            "password",
            "admin",
            "test",
            "key",
            "api_key"
        ]
        
        for endpoint in self.test_endpoints:
            for weak_key in weak_keys:
                try:
                    headers = {"X-API-Key": weak_key}
                    async with self.session.get(f"{self.base_url}{endpoint}", headers=headers) as response:
                        if response.status == 200:
                            self._add_security_result(
                                test_name=f"Weak API Key Accepted - {endpoint}",
                                test_category="API Key Security",
                                endpoint=endpoint,
                                vulnerability_type="Weak API Key",
                                severity="HIGH",
                                success=False,
                                details=f"Weak API key '{weak_key}' was accepted",
                                recommendation="Implement strong API key requirements and validation"
                            )
                except Exception as e:
                    logger.error(f"Error testing API key security on {endpoint}: {e}")
        
        # Test API key in URL (should be rejected)
        for endpoint in self.test_endpoints:
            try:
                params = {"api_key": "test_key", "key": "test_key"}
                async with self.session.get(f"{self.base_url}{endpoint}", params=params) as response:
                    if response.status == 200:
                        self._add_security_result(
                            test_name=f"API Key in URL - {endpoint}",
                            test_category="API Key Security",
                            endpoint=endpoint,
                            vulnerability_type="API Key Exposure",
                            severity="MEDIUM",
                            success=False,
                            details="API key accepted in URL parameters",
                            recommendation="Only accept API keys in headers, not URL parameters"
                        )
            except Exception as e:
                logger.error(f"Error testing API key in URL on {endpoint}: {e}")
    
    async def test_information_disclosure(self) -> None:
        """Test for information disclosure vulnerabilities."""
        
        logger.info("📋 Testing Information Disclosure")
        
        # Test for sensitive information in responses
        sensitive_patterns = [
            "password",
            "secret",
            "token",
            "key",
            "private",
            "confidential",
            "internal",
            "debug",
            "stack trace",
            "exception"
        ]
        
        for endpoint in self.test_endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    response_text = await response.text().lower()
                    
                    for pattern in sensitive_patterns:
                        if pattern in response_text:
                            self._add_security_result(
                                test_name=f"Information Disclosure - {endpoint}",
                                test_category="Information Disclosure",
                                endpoint=endpoint,
                                vulnerability_type="Sensitive Information Exposure",
                                severity="MEDIUM",
                                success=False,
                                details=f"Sensitive information '{pattern}' found in response",
                                recommendation="Remove sensitive information from API responses"
                            )
            except Exception as e:
                logger.error(f"Error testing information disclosure on {endpoint}: {e}")
    
    async def run_comprehensive_security_tests(self) -> Dict[str, Any]:
        """Run all security tests."""
        
        logger.info("🚀 Starting Comprehensive API Security Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all security tests
        await self.test_authentication_bypass()
        await self.test_authorization_flaws()
        await self.test_input_validation()
        await self.test_rate_limiting_security()
        await self.test_cors_policy()
        await self.test_security_headers()
        await self.test_api_key_security()
        await self.test_information_disclosure()
        
        total_time = time.time() - start_time
        
        return self._generate_security_report(total_time)
    
    def _generate_security_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        # Analyze results by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        category_counts = {}
        vulnerability_counts = {}
        
        for result in self.security_results:
            severity_counts[result.severity] += 1
            category_counts[result.test_category] = category_counts.get(result.test_category, 0) + 1
            vulnerability_counts[result.vulnerability_type] = vulnerability_counts.get(result.vulnerability_type, 0) + 1
        
        # Calculate security score
        total_tests = len(self.security_results)
        passed_tests = len([r for r in self.security_results if r.success])
        security_score = (passed_tests / total_tests * 100) if total_tests > 0 else 100
        
        # Determine security grade
        if security_score >= 95:
            security_grade = "A+"
        elif security_score >= 90:
            security_grade = "A"
        elif security_score >= 80:
            security_grade = "B"
        elif security_score >= 70:
            security_grade = "C"
        elif security_score >= 60:
            security_grade = "D"
        else:
            security_grade = "F"
        
        report = {
            "security_summary": {
                "total_execution_time_seconds": round(total_time, 2),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "security_score": round(security_score, 1),
                "security_grade": security_grade
            },
            
            "vulnerability_analysis": {
                "severity_breakdown": severity_counts,
                "category_breakdown": category_counts,
                "vulnerability_types": vulnerability_counts
            },
            
            "risk_assessment": {
                "critical_risk": severity_counts["CRITICAL"] > 0,
                "high_risk": severity_counts["HIGH"] > 0,
                "medium_risk": severity_counts["MEDIUM"] > 0,
                "overall_risk_level": self._calculate_risk_level(severity_counts)
            },
            
            "detailed_results": [asdict(r) for r in self.security_results],
            
            "security_recommendations": self._generate_security_recommendations(severity_counts),
            
            "compliance_status": {
                "owasp_top_10_compliant": severity_counts["CRITICAL"] == 0 and severity_counts["HIGH"] <= 2,
                "production_ready": security_score >= 80,
                "security_headers_compliant": self._check_security_headers_compliance(),
                "authentication_secure": self._check_authentication_security()
            }
        }
        
        return report
    
    def _calculate_risk_level(self, severity_counts: Dict[str, int]) -> str:
        """Calculate overall risk level."""
        if severity_counts["CRITICAL"] > 0:
            return "CRITICAL"
        elif severity_counts["HIGH"] > 2:
            return "HIGH"
        elif severity_counts["HIGH"] > 0 or severity_counts["MEDIUM"] > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _check_security_headers_compliance(self) -> bool:
        """Check if security headers are compliant."""
        header_failures = [r for r in self.security_results 
                          if r.test_category == "Security Headers" and not r.success]
        return len(header_failures) <= 2
    
    def _check_authentication_security(self) -> bool:
        """Check if authentication is secure."""
        auth_failures = [r for r in self.security_results 
                        if r.test_category == "Authentication" and not r.success and r.severity in ["HIGH", "CRITICAL"]]
        return len(auth_failures) == 0
    
    def _generate_security_recommendations(self, severity_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if severity_counts["CRITICAL"] > 0:
            recommendations.append("🚨 CRITICAL: Address critical vulnerabilities immediately before production deployment")
        
        if severity_counts["HIGH"] > 0:
            recommendations.append("🔴 HIGH: Fix high-severity vulnerabilities within 24 hours")
        
        if severity_counts["MEDIUM"] > 3:
            recommendations.append("🟠 MEDIUM: Address medium-severity vulnerabilities within 1 week")
        
        # Specific recommendations based on test results
        failed_categories = set()
        for result in self.security_results:
            if not result.success:
                failed_categories.add(result.test_category)
        
        if "Authentication" in failed_categories:
            recommendations.append("🔐 Implement robust authentication mechanisms")
        
        if "Input Validation" in failed_categories:
            recommendations.append("🛡️ Strengthen input validation and sanitization")
        
        if "Security Headers" in failed_categories:
            recommendations.append("📋 Add missing security headers")
        
        if "CORS" in failed_categories:
            recommendations.append("🌐 Review and tighten CORS policy")
        
        if not recommendations:
            recommendations.append("✅ Security posture is excellent. Continue regular security testing.")
        
        return recommendations

# Pytest integration
@pytest.mark.security
@pytest.mark.asyncio
class TestAPISecuritySuite:
    """Pytest integration for API security testing."""
    
    @pytest.fixture(scope="class")
    async def security_tester(self):
        """Create security tester fixture."""
        async with APISecurityTester() as tester:
            yield tester
    
    async def test_authentication_security(self, security_tester):
        """Test authentication security."""
        await security_tester.test_authentication_bypass()
        
        # Check for critical authentication failures
        auth_failures = [r for r in security_tester.security_results 
                        if r.test_category == "Authentication" and not r.success and r.severity == "CRITICAL"]
        assert len(auth_failures) == 0, "Critical authentication vulnerabilities found"
    
    async def test_input_validation_security(self, security_tester):
        """Test input validation security."""
        await security_tester.test_input_validation()
        
        # Check for critical injection vulnerabilities
        injection_failures = [r for r in security_tester.security_results 
                             if r.vulnerability_type in ["SQL Injection", "Cross-Site Scripting"] 
                             and not r.success and r.severity == "CRITICAL"]
        assert len(injection_failures) == 0, "Critical injection vulnerabilities found"
    
    async def test_comprehensive_security_suite(self, security_tester):
        """Test the complete security suite."""
        report = await security_tester.run_comprehensive_security_tests()
        
        # Verify security score
        security_score = report["security_summary"]["security_score"]
        assert security_score >= 70, f"Security score ({security_score}%) is too low"
        
        # Verify no critical vulnerabilities
        critical_count = report["vulnerability_analysis"]["severity_breakdown"]["CRITICAL"]
        assert critical_count == 0, f"Found {critical_count} critical vulnerabilities"
        
        logger.info(f"✅ Security Test Suite completed with {security_score}% score")

# Main execution function
async def run_comprehensive_security_tests():
    """Main function to run comprehensive API security tests."""
    
    async with APISecurityTester() as tester:
        report = await tester.run_comprehensive_security_tests()
        
        # Save report
        results_dir = Path("tests/api/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = results_dir / f"security_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE API SECURITY TEST SUITE - SUMMARY")
        logger.info("=" * 80)
        
        summary = report["security_summary"]
        logger.info(f"🛡️ Security Score: {summary['security_score']}% (Grade: {summary['security_grade']})")
        logger.info(f"📊 Total Tests: {summary['total_tests']}")
        logger.info(f"✅ Passed: {summary['passed_tests']}")
        logger.info(f"❌ Failed: {summary['failed_tests']}")
        
        severity = report["vulnerability_analysis"]["severity_breakdown"]
        logger.info(f"🚨 Critical: {severity['CRITICAL']}")
        logger.info(f"🔴 High: {severity['HIGH']}")
        logger.info(f"🟠 Medium: {severity['MEDIUM']}")
        logger.info(f"🟡 Low: {severity['LOW']}")
        
        logger.info(f"\n💡 Security Recommendations:")
        for i, rec in enumerate(report["security_recommendations"], 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info(f"\n💾 Report saved to: {report_file}")
        logger.info("=" * 80)
        
        return report

if __name__ == "__main__":
    asyncio.run(run_comprehensive_security_tests())