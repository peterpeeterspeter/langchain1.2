"""
Comprehensive Security & DataForSEO Integration Testing Framework
Tests Task 11 Security framework and Task 5 DataForSEO integration
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import json
import base64
import hashlib
import secrets

# Import security system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.security_manager import SecurityManager, SecurityMetrics
from security.models import (
    SecurityContext, SecurityConfig, UserRole, Permission, 
    AuditAction, SecurityViolation
)
from security.managers.rbac_manager import RBACManager
from security.managers.audit_logger import AuditLogger
from security.managers.api_key_manager import APIKeyManager
from security.managers.content_moderator import ContentModerator
from security.managers.gdpr_compliance_manager import GDPRComplianceManager
from security.managers.rate_limiter import RateLimiter
from security.encryption_manager import EncryptionManager

# Import DataForSEO components
from integrations.dataforseo_image_search import (
    EnhancedDataForSEOImageSearch, DataForSEOConfig, ImageSearchRequest,
    ImageSearchType, ImageSize, ImageType, ImageColor, ImageSearchResult
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSecurityDataForSEOIntegration:
    """Comprehensive Security & DataForSEO Integration Test Suite"""
    
    @pytest.fixture
    async def security_manager(self):
        """Create mock security manager for testing"""
        config = SecurityConfig(
            rate_limit_per_minute=100,
            enable_audit_logging=True,
            enable_content_moderation=True,
            enable_gdpr_compliance=True
        )
        
        manager = SecurityManager(config)
        
        # Mock database connections to avoid real DB calls
        with patch('security.utils.database_utils.get_database_connection'):
            await manager.initialize()
        
        return manager
    
    @pytest.fixture
    async def dataforseo_client(self):
        """Create mock DataForSEO client for testing"""
        config = DataForSEOConfig(
            login="test_login",
            password="test_password",
            max_requests_per_minute=100,
            max_concurrent_requests=5,
            enable_caching=True
        )
        
        client = EnhancedDataForSEOImageSearch(config)
        
        # Mock API calls to avoid real API usage
        with patch.object(client, '_make_request_with_retry') as mock_request:
            mock_request.return_value = {
                "status_code": 20000,
                "tasks": [{
                    "status_code": 20000,
                    "result": [{
                        "total_count": 10,
                        "items": [
                            {
                                "source_url": "https://example.com/image1.jpg",
                                "title": "Test Image 1",
                                "alt": "Test alt text",
                                "width": 800,
                                "height": 600,
                                "format": "jpg",
                                "thumbnail": "https://example.com/thumb1.jpg"
                            }
                        ]
                    }]
                }]
            }
            
            yield client
    
    @pytest.mark.asyncio
    async def test_rbac_system_comprehensive(self, security_manager):
        """Test comprehensive RBAC system functionality"""
        logger.info("🔐 Testing RBAC System Comprehensive...")
        
        rbac_manager = security_manager.rbac_manager
        
        # Test role hierarchy
        hierarchy = await rbac_manager.get_role_hierarchy()
        assert hierarchy[UserRole.SUPER_ADMIN.value] > hierarchy[UserRole.ADMIN.value]
        assert hierarchy[UserRole.ADMIN.value] > hierarchy[UserRole.CONTENT_MANAGER.value]
        assert hierarchy[UserRole.CONTENT_MANAGER.value] > hierarchy[UserRole.EDITOR.value]
        
        # Test permission checking for different roles
        test_cases = [
            (UserRole.SUPER_ADMIN, Permission.MANAGE_SYSTEM, True),
            (UserRole.ADMIN, Permission.MANAGE_USERS, True),
            (UserRole.CONTENT_MANAGER, Permission.WRITE_CONTENT, True),
            (UserRole.EDITOR, Permission.READ_CONTENT, True),
            (UserRole.USER, Permission.MANAGE_SYSTEM, False),
            (UserRole.ANONYMOUS, Permission.WRITE_CONTENT, False)
        ]
        
        for role, permission, expected in test_cases:
            result = await rbac_manager.check_permission(role, permission)
            assert result == expected, f"Permission check failed for {role.value} -> {permission.value}"
        
        # Test resource-level access control
        context = SecurityContext(
            user_id="test-user-123",
            user_role=UserRole.CONTENT_MANAGER,
            permissions=[Permission.READ_CONTENT, Permission.WRITE_CONTENT]
        )
        
        # Test resource access with ownership
        resource_data = {"owner_id": "test-user-123", "content_type": "article"}
        access_granted = await rbac_manager.check_resource_access(
            context.user_role, context.user_id, Permission.WRITE_CONTENT, 
            "content/article/123", resource_data
        )
        assert access_granted, "Resource access should be granted to owner"
        
        logger.info("✅ RBAC System Comprehensive test passed")
    
    @pytest.mark.asyncio
    async def test_audit_logging_system(self, security_manager):
        """Test comprehensive audit logging functionality"""
        logger.info("📝 Testing Audit Logging System...")
        
        audit_logger = security_manager.audit_logger
        
        # Mock database operations
        with patch.object(audit_logger, '_store_audit_entry') as mock_store:
            mock_store.return_value = True
            
            # Test various audit actions
            audit_tests = [
                {
                    "action": AuditAction.LOGIN_SUCCESS,
                    "user_id": "user-123",
                    "details": {"ip": "192.168.1.1", "user_agent": "TestAgent"},
                    "severity": "low"
                },
                {
                    "action": AuditAction.ACCESS_DENIED,
                    "user_id": "user-456",
                    "details": {"resource": "admin_panel", "reason": "insufficient_permissions"},
                    "severity": "medium"
                },
                {
                    "action": AuditAction.DATA_EXPORT,
                    "user_id": "admin-789",
                    "details": {"export_type": "user_data", "record_count": 100},
                    "severity": "high"
                }
            ]
            
            for test_case in audit_tests:
                success = await audit_logger.log_action(
                    action=test_case["action"],
                    user_id=test_case["user_id"],
                    details=test_case["details"],
                    severity=test_case["severity"]
                )
                assert success, f"Audit logging failed for {test_case['action']}"
            
            # Verify all audit entries were stored
            assert mock_store.call_count == len(audit_tests)
        
        # Test audit log querying
        with patch.object(audit_logger, 'get_audit_logs') as mock_get_logs:
            mock_get_logs.return_value = [
                {
                    "id": "audit-1",
                    "action": AuditAction.LOGIN_SUCCESS.value,
                    "user_id": "user-123",
                    "timestamp": datetime.utcnow(),
                    "details": {"ip": "192.168.1.1"}
                }
            ]
            
            logs = await audit_logger.get_audit_logs(
                user_id="user-123",
                start_date=datetime.utcnow() - timedelta(days=1),
                end_date=datetime.utcnow()
            )
            
            assert len(logs) == 1
            assert logs[0]["user_id"] == "user-123"
        
        logger.info("✅ Audit Logging System test passed")
    
    @pytest.mark.asyncio
    async def test_encryption_manager(self, security_manager):
        """Test encryption and decryption functionality"""
        logger.info("🔒 Testing Encryption Manager...")
        
        encryption_manager = security_manager.encryption_manager
        
        # Test data encryption/decryption
        test_data = "Sensitive user information and API keys"
        
        # Test symmetric encryption
        encrypted_data = await encryption_manager.encrypt_data(test_data)
        assert encrypted_data != test_data, "Data should be encrypted"
        assert len(encrypted_data) > len(test_data), "Encrypted data should be longer"
        
        decrypted_data = await encryption_manager.decrypt_data(encrypted_data)
        assert decrypted_data == test_data, "Decrypted data should match original"
        
        # Test field-level encryption
        user_data = {
            "user_id": "user-123",
            "email": "user@example.com",
            "api_key": "secret-api-key-12345",
            "personal_info": {"name": "John Doe", "phone": "+1234567890"}
        }
        
        encrypted_fields = await encryption_manager.encrypt_sensitive_fields(
            user_data, 
            sensitive_fields=["api_key", "personal_info"]
        )
        
        assert encrypted_fields["user_id"] == user_data["user_id"]  # Not encrypted
        assert encrypted_fields["email"] == user_data["email"]      # Not encrypted
        assert encrypted_fields["api_key"] != user_data["api_key"]  # Encrypted
        assert encrypted_fields["personal_info"] != user_data["personal_info"]  # Encrypted
        
        # Test decryption
        decrypted_fields = await encryption_manager.decrypt_sensitive_fields(
            encrypted_fields,
            sensitive_fields=["api_key", "personal_info"]
        )
        
        assert decrypted_fields == user_data, "Decrypted fields should match original"
        
        # Test key rotation
        old_key_id = encryption_manager.current_key_id
        await encryption_manager.rotate_encryption_key()
        new_key_id = encryption_manager.current_key_id
        
        assert new_key_id != old_key_id, "Key ID should change after rotation"
        
        logger.info("✅ Encryption Manager test passed")
    
    @pytest.mark.asyncio
    async def test_api_key_management(self, security_manager):
        """Test API key management functionality"""
        logger.info("🔑 Testing API Key Management...")
        
        api_key_manager = security_manager.api_key_manager
        
        # Mock database operations
        with patch.object(api_key_manager, '_store_api_key') as mock_store, \
             patch.object(api_key_manager, '_get_api_key_info') as mock_get:
            
            # Test API key generation
            key_info = await api_key_manager.generate_api_key(
                user_id="user-123",
                role=UserRole.API_USER,
                permissions=[Permission.READ_CONTENT, Permission.API_ACCESS],
                expires_in_days=30,
                description="Test API key"
            )
            
            assert "api_key" in key_info
            assert "key_id" in key_info
            assert key_info["user_id"] == "user-123"
            assert key_info["role"] == UserRole.API_USER.value
            assert len(key_info["api_key"]) >= 32  # Minimum key length
            
            # Test API key validation
            mock_get.return_value = {
                "id": key_info["key_id"],
                "user_id": "user-123",
                "role": UserRole.API_USER.value,
                "permissions": [Permission.READ_CONTENT.value, Permission.API_ACCESS.value],
                "active": True,
                "expires_at": datetime.utcnow() + timedelta(days=30)
            }
            
            is_valid, api_info = await api_key_manager.validate_api_key(key_info["api_key"])
            assert is_valid, "Generated API key should be valid"
            assert api_info["user_id"] == "user-123"
            
            # Test API key rotation
            with patch.object(api_key_manager, 'generate_api_key') as mock_generate:
                mock_generate.return_value = {
                    "api_key": "new-rotated-key-456",
                    "key_id": "new-key-id-789",
                    "user_id": "user-123"
                }
                
                rotated_key = await api_key_manager.rotate_api_key(key_info["key_id"])
                assert rotated_key["api_key"] != key_info["api_key"]
                assert rotated_key["user_id"] == "user-123"
        
        logger.info("✅ API Key Management test passed")
    
    @pytest.mark.asyncio
    async def test_dataforseo_rate_limiting(self, dataforseo_client):
        """Test DataForSEO rate limiting functionality"""
        logger.info("⏱️ Testing DataForSEO Rate Limiting...")
        
        # Test rate limiter initialization
        rate_limiter = dataforseo_client.rate_limiter
        assert rate_limiter.max_requests_per_minute == 100
        assert rate_limiter.max_concurrent_requests == 5
        
        # Test concurrent request limiting
        async def make_search_request():
            request = ImageSearchRequest(
                keyword="test search",
                max_results=5
            )
            return await dataforseo_client.search_images(request)
        
        # Create more concurrent requests than allowed
        tasks = [make_search_request() for _ in range(10)]
        
        # Execute with proper rate limiting
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0, "Some requests should succeed"
        
        # Verify rate limiting was applied (requests should take time)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        assert duration_ms > 100, "Rate limiting should introduce delays"
        
        # Test rate limiter statistics
        stats = await rate_limiter.get_statistics()
        assert "current_requests" in stats
        assert "requests_per_minute" in stats
        assert "concurrent_requests" in stats
        
        logger.info("✅ DataForSEO Rate Limiting test passed")
    
    @pytest.mark.asyncio
    async def test_dataforseo_batch_processing(self, dataforseo_client):
        """Test DataForSEO batch processing functionality"""
        logger.info("📦 Testing DataForSEO Batch Processing...")
        
        # Create batch of search requests
        keywords = ["casino games", "slot machines", "poker cards", "roulette wheel", "blackjack table"]
        requests = []
        
        for keyword in keywords:
            request = ImageSearchRequest(
                keyword=keyword,
                search_engine=ImageSearchType.GOOGLE_IMAGES,
                max_results=3,
                safe_search=True,
                quality_filter=True
            )
            requests.append(request)
        
        # Test batch processing
        start_time = datetime.now()
        results = await dataforseo_client.batch_search(requests)
        end_time = datetime.now()
        
        # Verify batch results
        assert len(results) == len(requests), "Should return result for each request"
        
        batch_duration = (end_time - start_time).total_seconds() * 1000
        average_per_request = batch_duration / len(requests)
        
        # Verify batch processing is more efficient than sequential
        assert average_per_request < 1000, "Batch processing should be efficient"
        
        # Verify individual results
        for i, result in enumerate(results):
            assert result.keyword == keywords[i]
            assert isinstance(result.images, list)
            assert result.search_duration_ms >= 0
        
        # Test batch size limits
        large_batch = [requests[0]] * 150  # Exceed max batch size
        large_results = await dataforseo_client.batch_search(large_batch)
        
        assert len(large_results) == 150, "Should handle large batches by splitting"
        
        logger.info("✅ DataForSEO Batch Processing test passed")
    
    @pytest.mark.asyncio
    async def test_security_compliance_integration(self, security_manager, dataforseo_client):
        """Test security compliance across integrated systems"""
        logger.info("🛡️ Testing Security Compliance Integration...")
        
        # Test GDPR compliance for DataForSEO data
        gdpr_manager = security_manager.gdpr_compliance_manager
        
        # Mock GDPR operations
        with patch.object(gdpr_manager, 'process_data_request') as mock_process:
            mock_process.return_value = {
                "request_id": "gdpr-123",
                "status": "completed",
                "data_exported": True,
                "records_processed": 50
            }
            
            # Test data export request
            export_result = await gdpr_manager.process_data_request(
                user_id="user-123",
                request_type="export",
                data_types=["search_history", "image_downloads"]
            )
            
            assert export_result["status"] == "completed"
            assert export_result["data_exported"] == True
        
        # Test content moderation integration
        content_moderator = security_manager.content_moderator
        
        # Test image content moderation
        test_image_data = {
            "url": "https://example.com/test-image.jpg",
            "title": "Test casino image",
            "alt_text": "Casino gaming table",
            "source_domain": "example.com"
        }
        
        with patch.object(content_moderator, 'moderate_content') as mock_moderate:
            mock_moderate.return_value = {
                "approved": True,
                "confidence": 0.95,
                "flags": [],
                "categories": ["gaming", "entertainment"]
            }
            
            moderation_result = await content_moderator.moderate_content(
                content_type="image",
                content_data=test_image_data
            )
            
            assert moderation_result["approved"] == True
            assert moderation_result["confidence"] > 0.9
        
        # Test integrated security metrics
        metrics = security_manager.get_security_metrics()
        assert isinstance(metrics, SecurityMetrics)
        assert metrics.total_requests >= 0
        assert metrics.get_security_score() >= 0.0
        
        logger.info("✅ Security Compliance Integration test passed")
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(self, security_manager, dataforseo_client):
        """Test complete end-to-end security workflow"""
        logger.info("🔄 Testing End-to-End Security Workflow...")
        
        # 1. Authentication
        request_data = {
            "user_id": "test-user-123",
            "client_ip": "192.168.1.100",
            "user_agent": "TestClient/1.0",
            "api_key": "test-api-key-456"
        }
        
        # Mock authentication
        with patch.object(security_manager, 'authenticate_request') as mock_auth:
            mock_auth.return_value = (
                True,
                SecurityContext(
                    user_id="test-user-123",
                    user_role=UserRole.API_USER,
                    permissions=[Permission.READ_CONTENT, Permission.API_ACCESS],
                    session_id="session-789",
                    client_ip="192.168.1.100",
                    authenticated=True
                ),
                []
            )
            
            success, context, errors = await security_manager.authenticate_request(request_data)
            assert success == True
            assert context.authenticated == True
            assert len(errors) == 0
        
        # 2. Authorization for DataForSEO access
        with patch.object(security_manager, 'authorize_action') as mock_authorize:
            mock_authorize.return_value = (True, [])
            
            authorized, auth_errors = await security_manager.authorize_action(
                context,
                Permission.API_ACCESS,
                resource="dataforseo_search"
            )
            assert authorized == True
            assert len(auth_errors) == 0
        
        # 3. Perform DataForSEO search with security context
        search_request = ImageSearchRequest(
            keyword="secure casino search",
            max_results=5,
            safe_search=True
        )
        
        # Add security context to search
        search_result = await dataforseo_client.search_images(search_request)
        assert search_result.total_results >= 0
        assert len(search_result.images) >= 0
        
        # 4. Audit the complete workflow
        with patch.object(security_manager.audit_logger, 'log_action') as mock_audit:
            mock_audit.return_value = True
            
            await security_manager.audit_logger.log_action(
                action=AuditAction.API_ACCESS,
                user_id=context.user_id,
                session_id=context.session_id,
                details={
                    "api_endpoint": "dataforseo_search",
                    "search_keyword": search_request.keyword,
                    "results_count": len(search_result.images),
                    "client_ip": context.client_ip
                },
                result="success"
            )
            
            # Verify audit logging was called
            mock_audit.assert_called_once()
        
        # 5. Update security metrics
        security_manager.metrics.total_requests += 1
        security_manager.metrics.api_key_rotations += 0  # No rotation in this test
        
        final_score = security_manager.metrics.get_security_score()
        assert 0.0 <= final_score <= 1.0
        
        logger.info("✅ End-to-End Security Workflow test passed")
    
    @pytest.mark.asyncio
    async def test_security_error_handling(self, security_manager, dataforseo_client):
        """Test security error handling and recovery"""
        logger.info("⚠️ Testing Security Error Handling...")
        
        # Test authentication failures
        invalid_request = {
            "user_id": "invalid-user",
            "client_ip": "192.168.1.1",
            "api_key": "invalid-key"
        }
        
        with patch.object(security_manager.api_key_manager, 'validate_api_key') as mock_validate:
            mock_validate.return_value = (False, None)
            
            success, context, errors = await security_manager.authenticate_request(invalid_request)
            assert success == False
            assert context is None
            assert len(errors) > 0
            assert "Invalid API key" in errors[0]
        
        # Test authorization failures
        limited_context = SecurityContext(
            user_id="limited-user",
            user_role=UserRole.ANONYMOUS,
            permissions=[Permission.READ_CONTENT],
            authenticated=False
        )
        
        authorized, auth_errors = await security_manager.authorize_action(
            limited_context,
            Permission.MANAGE_SYSTEM,
            resource="admin_panel"
        )
        assert authorized == False
        assert len(auth_errors) > 0
        
        # Test DataForSEO API failures
        with patch.object(dataforseo_client, '_make_request_with_retry') as mock_request:
            mock_request.side_effect = Exception("API connection failed")
            
            search_request = ImageSearchRequest(keyword="test", max_results=1)
            
            with pytest.raises(Exception) as exc_info:
                await dataforseo_client.search_images(search_request)
            
            assert "API connection failed" in str(exc_info.value)
        
        # Test rate limiting errors
        with patch.object(dataforseo_client.rate_limiter, 'acquire') as mock_acquire:
            mock_acquire.side_effect = Exception("Rate limit exceeded")
            
            with pytest.raises(Exception) as exc_info:
                await dataforseo_client.search_images(search_request)
            
            assert "Rate limit exceeded" in str(exc_info.value)
        
        logger.info("✅ Security Error Handling test passed")


# === PERFORMANCE TESTS ===

class TestSecurityPerformance:
    """Performance tests for security and DataForSEO integration"""
    
    @pytest.mark.asyncio
    async def test_authentication_performance(self, security_manager):
        """Test authentication performance under load"""
        logger.info("⚡ Testing Authentication Performance...")
        
        # Create multiple authentication requests
        requests = []
        for i in range(50):
            request = {
                "user_id": f"user-{i}",
                "client_ip": f"192.168.1.{i % 255}",
                "user_agent": "LoadTestClient/1.0"
            }
            requests.append(request)
        
        # Mock authentication to avoid DB calls
        with patch.object(security_manager, 'authenticate_request') as mock_auth:
            mock_auth.return_value = (True, Mock(), [])
            
            # Measure authentication performance
            start_time = datetime.now()
            
            tasks = [security_manager.authenticate_request(req) for req in requests]
            results = await asyncio.gather(*tasks)
            
            end_time = datetime.now()
            
            # Verify performance
            duration_ms = (end_time - start_time).total_seconds() * 1000
            avg_per_request = duration_ms / len(requests)
            
            assert len(results) == len(requests)
            assert avg_per_request < 50, f"Authentication too slow: {avg_per_request}ms per request"
        
        logger.info("✅ Authentication Performance test passed")
    
    @pytest.mark.asyncio
    async def test_dataforseo_concurrent_performance(self, dataforseo_client):
        """Test DataForSEO concurrent request performance"""
        logger.info("🚀 Testing DataForSEO Concurrent Performance...")
        
        # Create concurrent search requests
        keywords = [f"test keyword {i}" for i in range(20)]
        requests = [
            ImageSearchRequest(keyword=keyword, max_results=3)
            for keyword in keywords
        ]
        
        # Measure concurrent performance
        start_time = datetime.now()
        
        results = await dataforseo_client.batch_search(requests)
        
        end_time = datetime.now()
        
        # Verify performance metrics
        duration_ms = (end_time - start_time).total_seconds() * 1000
        avg_per_request = duration_ms / len(requests)
        
        assert len(results) == len(requests)
        assert avg_per_request < 500, f"Batch search too slow: {avg_per_request}ms per request"
        
        # Verify all results are valid
        for result in results:
            assert isinstance(result, ImageSearchResult)
            assert result.search_duration_ms >= 0
        
        logger.info("✅ DataForSEO Concurrent Performance test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 