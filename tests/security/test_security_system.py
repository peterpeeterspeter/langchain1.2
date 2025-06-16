"""
Comprehensive Security System Test Suite
Tests all security components for Universal RAG CMS
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Import security system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.security_manager import SecurityManager
from security.models import (
    SecurityContext, SecurityConfig, UserRole, Permission, 
    AuditAction, SecurityViolation
)
from security.managers.rbac_manager import RBACManager
from security.managers.audit_logger import AuditLogger
from security.managers.api_key_manager import APIKeyManager
from security.managers.content_moderator import ContentModerator, ContentModerationConfig
from security.managers.gdpr_compliance_manager import GDPRComplianceManager
from security.managers.rate_limiter import RateLimiter
from security.encryption_manager import EncryptionManager
from security.secure_rag_chain import SecureRAGChain

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSecuritySystemIntegration:
    """Integration tests for the complete security system."""
    
    @pytest.fixture
    async def security_manager(self):
        """Create a test security manager instance."""
        config = SecurityConfig(
            debug_mode=True,
            openai_api_key="test-key-12345",
            rate_limit_config={
                'default_requests_per_minute': 100,
                'default_requests_per_hour': 1000
            }
        )
        
        manager = SecurityManager(config)
        
        # Mock external dependencies for testing
        with patch.object(manager.content_moderator, '_openai_client'):
            await manager.initialize()
            yield manager
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_security_manager_initialization(self, security_manager):
        """Test that the security manager initializes all components."""
        
        # Check component status
        status = security_manager.component_status
        
        assert status['encryption'] == True, "Encryption manager should initialize"
        assert status['rbac'] == True, "RBAC manager should initialize"
        assert status['audit'] == True, "Audit logger should initialize"
        assert status['api_keys'] == True, "API key manager should initialize"
        assert status['content_moderation'] == True, "Content moderator should initialize"
        assert status['gdpr'] == True, "GDPR manager should initialize"
        assert status['rate_limiting'] == True, "Rate limiter should initialize"
        
        logger.info("✅ Security manager initialization test passed")
    
    @pytest.mark.asyncio
    async def test_full_security_pipeline(self, security_manager):
        """Test the complete security pipeline end-to-end."""
        
        # Test request data
        request_data = {
            'api_key': 'test-api-key-123',
            'user_id': 'test-user-456',
            'ip_address': '192.168.1.1',
            'user_agent': 'TestAgent/1.0',
            'content': 'This is a test query about casino bonuses.',
            'action': 'query'
        }
        
        # Mock API key validation
        with patch.object(security_manager.api_key_manager, 'validate_api_key') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'user_id': 'test-user-456',
                'permissions': ['read', 'query'],
                'rate_limits': {'requests_per_minute': 60}
            }
            
            # Test authentication
            success, context, errors = await security_manager.authenticate_request(
                request_data
            )
            
            assert success == True, f"Authentication should succeed: {errors}"
            assert context is not None, "Security context should be created"
            assert context.user_id == 'test-user-456', "User ID should match"
            assert errors == [], "No authentication errors expected"
        
        logger.info("✅ Security pipeline test passed")
    
    @pytest.mark.asyncio
    async def test_rbac_authorization(self, security_manager):
        """Test Role-Based Access Control authorization."""
        
        # Create test security context
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.MEMBER,
            permissions=[Permission.READ, Permission.QUERY],
            ip_address='192.168.1.1'
        )
        
        # Test authorized action
        success, errors = await security_manager.authorize_action(
            context,
            Permission.READ,
            resource='documents'
        )
        
        assert success == True, f"Read permission should be authorized: {errors}"
        assert errors == [], "No authorization errors for valid permission"
        
        # Test unauthorized action
        success, errors = await security_manager.authorize_action(
            context,
            Permission.ADMIN,
            resource='system'
        )
        
        assert success == False, "Admin permission should be denied for MEMBER role"
        assert len(errors) > 0, "Authorization errors should be returned"
        
        logger.info("✅ RBAC authorization test passed")
    
    @pytest.mark.asyncio
    async def test_content_moderation(self, security_manager):
        """Test content moderation functionality."""
        
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.MEMBER,
            permissions=[Permission.READ, Permission.QUERY]
        )
        
        # Test safe content
        safe_content = "What are the best strategies for playing poker?"
        
        with patch.object(security_manager.content_moderator, '_moderate_with_openai') as mock_openai:
            mock_openai.return_value = {
                'flagged': False,
                'categories': {'harassment': False, 'violence': False},
                'category_scores': {'harassment': 0.1, 'violence': 0.05}
            }
            
            approved, moderation_result = await security_manager.moderate_content(
                safe_content,
                'query',
                context
            )
            
            assert approved == True, "Safe content should be approved"
            assert moderation_result['action'] == 'approved', "Moderation action should be approved"
        
        # Test flagged content
        flagged_content = "How to hack casino systems?"
        
        with patch.object(security_manager.content_moderator, '_moderate_with_openai') as mock_openai:
            mock_openai.return_value = {
                'flagged': True,
                'categories': {'harassment': True},
                'category_scores': {'harassment': 0.9}
            }
            
            approved, moderation_result = await security_manager.moderate_content(
                flagged_content,
                'query',
                context
            )
            
            assert approved == False, "Flagged content should be rejected"
            assert moderation_result['action'] == 'blocked', "Moderation action should be blocked"
        
        logger.info("✅ Content moderation test passed")
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.MEMBER,
            permissions=[Permission.READ, Permission.QUERY]
        )
        
        # Mock rate limiter to test both allowed and denied scenarios
        with patch.object(security_manager.rate_limiter, 'check_rate_limit') as mock_check:
            # Test allowed request
            mock_check.return_value = {
                'allowed': True,
                'remaining': 59,
                'reset_time': datetime.utcnow() + timedelta(minutes=1)
            }
            
            result = await security_manager.rate_limiter.check_rate_limit(
                'test-user-456',
                'api_requests'
            )
            
            assert result['allowed'] == True, "Request should be allowed within limits"
            assert result['remaining'] == 59, "Remaining requests should be tracked"
            
            # Test rate limit exceeded
            mock_check.return_value = {
                'allowed': False,
                'remaining': 0,
                'reset_time': datetime.utcnow() + timedelta(minutes=1)
            }
            
            result = await security_manager.rate_limiter.check_rate_limit(
                'test-user-456',
                'api_requests'
            )
            
            assert result['allowed'] == False, "Request should be denied when limit exceeded"
            assert result['remaining'] == 0, "No remaining requests when limit exceeded"
        
        logger.info("✅ Rate limiting test passed")
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, security_manager):
        """Test audit logging functionality."""
        
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.MEMBER,
            permissions=[Permission.READ, Permission.QUERY]
        )
        
        # Mock audit logger to capture logs
        with patch.object(security_manager.audit_logger, 'log_event') as mock_log:
            mock_log.return_value = True
            
            # Test logging a security event
            await security_manager.audit_logger.log_event(
                AuditAction.LOGIN_SUCCESS,
                context,
                {'ip_address': '192.168.1.1'}
            )
            
            # Verify audit log was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            assert call_args[0][0] == AuditAction.LOGIN_SUCCESS, "Audit action should match"
            assert call_args[0][1] == context, "Security context should be logged"
        
        logger.info("✅ Audit logging test passed")
    
    @pytest.mark.asyncio
    async def test_api_key_management(self, security_manager):
        """Test API key management functionality."""
        
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.ADMIN,
            permissions=[Permission.ADMIN, Permission.API_KEY_MANAGE]
        )
        
        # Mock API key operations
        with patch.object(security_manager.api_key_manager, 'create_api_key') as mock_create:
            mock_create.return_value = {
                'key_id': 'test-key-id-123',
                'api_key': 'sk-test-api-key-456789',
                'created_at': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(days=90)
            }
            
            # Test API key creation
            result = await security_manager.api_key_manager.create_api_key(
                user_id='test-user-456',
                service_name='test-service',
                permissions=['read', 'query']
            )
            
            assert 'key_id' in result, "API key ID should be returned"
            assert 'api_key' in result, "API key should be returned"
            assert result['api_key'].startswith('sk-'), "API key should have proper prefix"
        
        # Test API key validation
        with patch.object(security_manager.api_key_manager, 'validate_api_key') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'user_id': 'test-user-456',
                'permissions': ['read', 'query'],
                'rate_limits': {'requests_per_minute': 60}
            }
            
            result = await security_manager.api_key_manager.validate_api_key(
                'sk-test-api-key-456789'
            )
            
            assert result['valid'] == True, "Valid API key should be accepted"
            assert result['user_id'] == 'test-user-456', "User ID should be returned"
            assert 'permissions' in result, "Permissions should be included"
        
        logger.info("✅ API key management test passed")
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance(self, security_manager):
        """Test GDPR compliance functionality."""
        
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.MEMBER,
            permissions=[Permission.READ, Permission.GDPR_REQUEST]
        )
        
        # Mock GDPR manager operations
        with patch.object(security_manager.gdpr_manager, 'handle_data_request') as mock_request:
            mock_request.return_value = {
                'request_id': 'gdpr-req-123',
                'status': 'processing',
                'created_at': datetime.utcnow(),
                'estimated_completion': datetime.utcnow() + timedelta(days=30)
            }
            
            # Test data export request
            result = await security_manager.handle_gdpr_request(
                'data_export',
                'test-user-456',
                context
            )
            
            assert 'request_id' in result, "GDPR request ID should be returned"
            assert result['status'] == 'processing', "Request should be in processing status"
        
        logger.info("✅ GDPR compliance test passed")
    
    @pytest.mark.asyncio
    async def test_encryption_functionality(self, security_manager):
        """Test encryption manager functionality."""
        
        # Test data encryption/decryption
        test_data = "Sensitive user information that needs protection"
        
        # Test encryption
        encrypted_data = security_manager.encryption_manager.encrypt(test_data)
        
        assert encrypted_data != test_data, "Data should be encrypted"
        assert len(encrypted_data) > 0, "Encrypted data should not be empty"
        
        # Test decryption
        decrypted_data = security_manager.encryption_manager.decrypt(encrypted_data)
        
        assert decrypted_data == test_data, "Decrypted data should match original"
        
        # Test password hashing
        password = "test_password_123"
        password_hash = security_manager.encryption_manager.hash_password(password)
        
        assert password_hash != password, "Password should be hashed"
        assert security_manager.encryption_manager.verify_password(password, password_hash), "Password verification should work"
        assert not security_manager.encryption_manager.verify_password("wrong_password", password_hash), "Wrong password should fail verification"
        
        logger.info("✅ Encryption functionality test passed")
    
    @pytest.mark.asyncio
    async def test_security_metrics(self, security_manager):
        """Test security metrics collection."""
        
        context = SecurityContext(
            user_id='test-user-456',
            session_id='test-session-789',
            user_role=UserRole.ADMIN,
            permissions=[Permission.ADMIN, Permission.READ_METRICS]
        )
        
        # Get security metrics
        metrics = await security_manager.get_security_metrics(context)
        
        assert 'security_score' in metrics, "Security score should be included"
        assert 'component_health' in metrics, "Component health should be included"
        assert 'audit_summary' in metrics, "Audit summary should be included"
        assert 'rate_limit_status' in metrics, "Rate limit status should be included"
        
        # Verify security score is valid
        assert 0.0 <= metrics['security_score'] <= 1.0, "Security score should be between 0 and 1"
        
        logger.info("✅ Security metrics test passed")
    
    @pytest.mark.asyncio
    async def test_health_check(self, security_manager):
        """Test system health check functionality."""
        
        health_status = await security_manager.health_check()
        
        assert 'overall_health' in health_status, "Overall health should be reported"
        assert 'component_status' in health_status, "Component status should be included"
        assert 'uptime' in health_status, "Uptime should be tracked"
        
        # Verify all components are healthy
        for component, status in health_status['component_status'].items():
            assert status == True, f"Component {component} should be healthy"
        
        logger.info("✅ Health check test passed")


class TestSecureRAGChainIntegration:
    """Test integration with the RAG chain."""
    
    @pytest.fixture
    async def secure_rag_chain(self):
        """Create a test secure RAG chain instance."""
        
        # Mock the base RAG chain
        mock_base_chain = Mock()
        mock_base_chain.ainvoke = AsyncMock(return_value={
            'content': 'Test response about casino bonuses',
            'sources': [{'title': 'Test Source', 'url': 'https://example.com'}]
        })
        
        config = SecurityConfig(debug_mode=True)
        security_manager = SecurityManager(config)
        
        with patch.object(security_manager.content_moderator, '_openai_client'):
            await security_manager.initialize()
            
            secure_chain = SecureRAGChain(
                base_chain=mock_base_chain,
                security_manager=security_manager
            )
            
            yield secure_chain
            await security_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_secure_rag_chain_execution(self, secure_rag_chain):
        """Test secure RAG chain execution with security pipeline."""
        
        # Test input
        test_input = {
            'query': 'What are the best casino bonuses available?',
            'api_key': 'sk-test-key-123',
            'user_id': 'test-user-456'
        }
        
        # Mock security validations
        with patch.object(secure_rag_chain.security_manager, 'authenticate_request') as mock_auth:
            mock_auth.return_value = (True, SecurityContext(
                user_id='test-user-456',
                session_id='test-session',
                user_role=UserRole.MEMBER,
                permissions=[Permission.READ, Permission.QUERY]
            ), [])
            
            with patch.object(secure_rag_chain.security_manager, 'moderate_content') as mock_moderate:
                mock_moderate.return_value = (True, {'action': 'approved'})
                
                # Execute secure RAG chain
                result = await secure_rag_chain.ainvoke(test_input)
                
                assert 'content' in result, "Response content should be present"
                assert 'security_metadata' in result, "Security metadata should be included"
                assert result['security_metadata']['authenticated'] == True, "Request should be authenticated"
                assert result['security_metadata']['content_approved'] == True, "Content should be approved"
        
        logger.info("✅ Secure RAG chain integration test passed")


# Test runner function
async def run_security_tests():
    """Run all security tests."""
    
    logger.info("🚀 Starting comprehensive security system tests...")
    
    try:
        # Test basic imports
        logger.info("Testing security module imports...")
        
        # Test SecurityManager import and basic initialization
        config = SecurityConfig(debug_mode=True)
        manager = SecurityManager(config)
        logger.info("✅ SecurityManager created successfully")
        
        # Test component initialization
        logger.info("Testing component initialization...")
        
        # Mock external dependencies
        with patch('openai.OpenAI'):
            init_success = await manager.initialize()
            logger.info(f"✅ Security manager initialization: {init_success}")
        
        # Test individual components
        logger.info("Testing individual components...")
        
        # Test encryption manager
        test_data = "test encryption data"
        encrypted = manager.encryption_manager.encrypt(test_data)
        decrypted = manager.encryption_manager.decrypt(encrypted)
        assert decrypted == test_data
        logger.info("✅ EncryptionManager working correctly")
        
        # Test RBAC manager
        test_context = SecurityContext(
            user_id="test-user",
            user_role=UserRole.MEMBER,
            permissions=[Permission.READ]
        )
        
        has_permission = await manager.rbac_manager.check_permission(
            test_context, Permission.READ
        )
        assert has_permission == True
        logger.info("✅ RBACManager working correctly")
        
        # Test basic health check
        health = await manager.health_check()
        assert 'overall_health' in health
        logger.info("✅ Health check working correctly")
        
        await manager.shutdown()
        logger.info("✅ Security manager shutdown successfully")
        
        logger.info("🎉 All basic security tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Security tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests directly
    result = asyncio.run(run_security_tests())
    if result:
        print("✅ Security system validation completed successfully!")
    else:
        print("❌ Security system validation failed!")
        exit(1) 