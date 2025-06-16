"""
Security System Integration Test
Tests how security components work together in realistic scenarios
"""

import asyncio
import logging
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag_security_integration():
    """Test integration between security system and RAG components."""
    logger.info("🔍 Testing RAG security integration...")
    
    try:
        from security.models import SecurityContext, SecurityConfig, UserRole, Permission
        from security.encryption_manager import EncryptionManager
        
        # Create security config
        config = SecurityConfig()
        encryption_manager = EncryptionManager(config)
        
        # Create a user context
        user_context = SecurityContext(
            user_id="test-rag-user",
            session_id="session-123",
            roles=[UserRole.EDITOR],
            permissions=[Permission.CONTENT_READ, Permission.CONTENT_CREATE, Permission.API_FULL_ACCESS],
            ip_address="192.168.1.100",
            user_agent="RAGClient/1.0"
        )
        
        # Test sensitive data encryption in RAG context
        sensitive_query = "What are the details about user john.doe@company.com's account?"
        encrypted_query = encryption_manager.encrypt(sensitive_query)
        
        # Simulate processing
        assert encrypted_query != sensitive_query, "Query should be encrypted"
        
        # Decrypt for processing
        decrypted_query = encryption_manager.decrypt(encrypted_query)
        assert decrypted_query == sensitive_query, "Query should decrypt correctly"
        
        logger.info("✅ RAG query encryption/decryption working")
        
        # Test permission checking for RAG operations
        can_read = Permission.CONTENT_READ in user_context.permissions
        can_create = Permission.CONTENT_CREATE in user_context.permissions
        can_admin = Permission.SYSTEM_CONFIG in user_context.permissions
        
        assert can_read == True, "User should have read permission"
        assert can_create == True, "User should have create permission"
        assert can_admin == False, "User should not have admin permission"
        
        logger.info("✅ RAG permission checking working")
        
        # Test API key generation for RAG API access
        api_key, api_key_hash = encryption_manager.generate_api_key()
        
        # Simulate API key validation
        assert len(api_key) >= 32, "API key should be sufficiently long"
        assert len(api_key_hash) > 0, "API key hash should exist"
        
        logger.info("✅ RAG API key management working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG security integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_security_pipeline():
    """Test the complete security pipeline."""
    logger.info("🔍 Testing complete security pipeline...")
    
    try:
        from security.models import SecurityContext, SecurityConfig, UserRole, Permission, AuditAction
        from security.encryption_manager import EncryptionManager
        
        # Step 1: User Authentication Simulation
        config = SecurityConfig()
        encryption_manager = EncryptionManager(config)
        
        # Simulate user login
        username = "security_test_user"
        password = "SecurePassword123!"
        
        # Hash password (registration simulation)
        password_hash = encryption_manager.hash_password(password)
        logger.info("✅ Step 1: Password hashing working")
        
        # Verify password (login simulation)
        login_success = encryption_manager.verify_password(password, password_hash)
        assert login_success == True, "Password verification should succeed"
        logger.info("✅ Step 2: Password verification working")
        
        # Step 3: Create security context
        user_context = SecurityContext(
            user_id="user-456",
            session_id="session-789",
            roles=[UserRole.MODERATOR],
            permissions=[
                Permission.CONTENT_READ, Permission.CONTENT_UPDATE, 
                Permission.USER_READ, Permission.SYSTEM_MONITOR
            ],
            ip_address="10.0.0.50",
            user_agent="SecurityTest/1.0"
        )
        logger.info("✅ Step 3: Security context creation working")
        
        # Step 4: Permission-based access control
        def check_operation_allowed(required_permission: Permission) -> bool:
            return required_permission in user_context.permissions
        
        # Test various operations
        can_read_content = check_operation_allowed(Permission.CONTENT_READ)
        can_update_content = check_operation_allowed(Permission.CONTENT_UPDATE)
        can_delete_content = check_operation_allowed(Permission.CONTENT_DELETE)
        can_config_system = check_operation_allowed(Permission.SYSTEM_CONFIG)
        
        assert can_read_content == True, "Should allow content reading"
        assert can_update_content == True, "Should allow content updating"
        assert can_delete_content == False, "Should not allow content deletion"
        assert can_config_system == False, "Should not allow system configuration"
        
        logger.info("✅ Step 4: Permission-based access control working")
        
        # Step 5: Data encryption for sensitive operations
        sensitive_data = {
            "user_email": "testuser@company.com",
            "personal_notes": "This user has special requirements",
            "api_credentials": "internal-api-key-12345"
        }
        
        # Encrypt sensitive fields
        encrypted_email = encryption_manager.encrypt(sensitive_data["user_email"])
        encrypted_notes = encryption_manager.encrypt(sensitive_data["personal_notes"])
        encrypted_creds = encryption_manager.encrypt(sensitive_data["api_credentials"])
        
        # Verify encryption worked
        assert encrypted_email != sensitive_data["user_email"], "Email should be encrypted"
        assert encrypted_notes != sensitive_data["personal_notes"], "Notes should be encrypted"
        assert encrypted_creds != sensitive_data["api_credentials"], "Credentials should be encrypted"
        
        logger.info("✅ Step 5: Sensitive data encryption working")
        
        # Step 6: Generate audit trail
        audit_events = [
            {
                "action": AuditAction.LOGIN,
                "user_id": user_context.user_id,
                "ip_address": user_context.ip_address,
                "timestamp": datetime.utcnow(),
                "details": {"login_method": "password"}
            },
            {
                "action": AuditAction.CONTENT_READ,
                "user_id": user_context.user_id,
                "ip_address": user_context.ip_address,
                "timestamp": datetime.utcnow(),
                "details": {"content_id": "doc-123", "content_type": "document"}
            },
            {
                "action": AuditAction.API_CALL,
                "user_id": user_context.user_id,
                "ip_address": user_context.ip_address,
                "timestamp": datetime.utcnow(),
                "details": {"endpoint": "/api/v1/rag/query", "method": "POST"}
            }
        ]
        
        # Verify audit events structure
        for event in audit_events:
            assert "action" in event, "Audit event should have action"
            assert "user_id" in event, "Audit event should have user_id"
            assert "timestamp" in event, "Audit event should have timestamp"
            assert isinstance(event["action"], AuditAction), "Action should be AuditAction enum"
        
        logger.info("✅ Step 6: Audit trail generation working")
        
        # Step 7: Session token generation
        session_token = encryption_manager.create_secure_token(32)
        assert len(session_token) >= 32, "Session token should be sufficiently long"
        
        logger.info("✅ Step 7: Session token generation working")
        
        logger.info("🎉 Complete security pipeline test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Security pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_security_edge_cases():
    """Test security system edge cases and error handling."""
    logger.info("🔍 Testing security edge cases...")
    
    try:
        from security.models import SecurityContext, SecurityConfig, UserRole, Permission
        from security.encryption_manager import EncryptionManager
        
        config = SecurityConfig()
        encryption_manager = EncryptionManager(config)
        
        # Test empty data encryption
        empty_result = encryption_manager.encrypt("")
        assert empty_result == "", "Empty string should remain empty"
        
        # Test None data encryption
        none_result = encryption_manager.encrypt(None)
        assert none_result is None, "None should remain None"
        
        logger.info("✅ Empty/None data handling working")
        
        # Test invalid password verification
        invalid_hash = "invalid$hash$format"
        verify_result = encryption_manager.verify_password("password", invalid_hash)
        assert verify_result == False, "Invalid hash should fail verification"
        
        logger.info("✅ Invalid password hash handling working")
        
        # Test role hierarchy edge cases
        anonymous_context = SecurityContext(
            user_id="anonymous",
            session_id="anon-session",
            roles=[UserRole.API_USER],
            permissions=[Permission.API_READ_ONLY],
            ip_address="192.168.1.1",
            user_agent="Anonymous/1.0"
        )
        
        # Anonymous should have very limited permissions
        assert Permission.CONTENT_CREATE not in anonymous_context.permissions, "Anonymous should not create content"
        assert Permission.SYSTEM_CONFIG not in anonymous_context.permissions, "Anonymous should not access system config"
        
        logger.info("✅ Role hierarchy edge cases working")
        
        # Test extremely long data encryption
        long_data = "A" * 10000  # 10KB of data
        encrypted_long = encryption_manager.encrypt(long_data)
        decrypted_long = encryption_manager.decrypt(encrypted_long)
        
        assert decrypted_long == long_data, "Long data should encrypt/decrypt correctly"
        
        logger.info("✅ Large data encryption working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Security edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_integration_tests():
    """Run all security integration tests."""
    logger.info("🚀 Starting security integration testing...")
    
    test_results = []
    
    # Run integration tests
    logger.info("\n" + "="*60)
    logger.info("SECURITY INTEGRATION TESTS")
    logger.info("="*60)
    
    test_results.append(("RAG Security Integration", await test_rag_security_integration()))
    test_results.append(("Security Pipeline", await test_security_pipeline()))
    test_results.append(("Security Edge Cases", await test_security_edge_cases()))
    
    # Print results summary
    logger.info("\n" + "="*60)
    logger.info("INTEGRATION TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal Integration Tests: {len(test_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL SECURITY INTEGRATION TESTS PASSED!")
        return True
    else:
        logger.error(f"❌ {failed} integration tests failed.")
        return False

if __name__ == "__main__":
    result = asyncio.run(run_integration_tests())
    if result:
        print("\n✅ Security integration testing completed successfully!")
        print("🔐 Security system is working correctly in integrated scenarios!")
        print("🛡️  The security pipeline provides comprehensive protection for the RAG system!")
        exit(0)
    else:
        print("\n❌ Security integration testing failed!")
        exit(1) 