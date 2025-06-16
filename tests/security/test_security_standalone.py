"""
Standalone Security System Tests
Tests security components that work without external dependencies
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_security_models():
    """Test security model functionality."""
    logger.info("🔍 Testing security models...")
    
    try:
        from security.models import (
            SecurityContext, SecurityConfig, UserRole, Permission, 
            AuditAction, SecurityViolation
        )
        
        # Test SecurityContext creation
        context = SecurityContext(
            user_id="test-user-123",
            session_id="test-session-456",
            roles=[UserRole.VIEWER],
            permissions=[Permission.CONTENT_READ, Permission.API_READ_ONLY],
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        
        assert context.user_id == "test-user-123", "User ID should be set correctly"
        assert UserRole.VIEWER in context.roles, "User role should be set correctly"
        assert Permission.CONTENT_READ in context.permissions, "Permissions should include READ"
        logger.info("✅ SecurityContext creation working")
        
        # Test SecurityConfig creation
        config = SecurityConfig(
            rate_limit_per_minute=100
        )
        
        assert config.rate_limit_per_minute == 100, "Rate limit should be set"
        logger.info("✅ SecurityConfig creation working")
        
        # Test role hierarchy
        assert UserRole.SUPER_ADMIN.level > UserRole.ADMIN.level, "Role hierarchy should work"
        assert UserRole.ADMIN.level > UserRole.VIEWER.level, "Role hierarchy should work"
        logger.info("✅ Role hierarchy working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Security models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encryption_manager():
    """Test encryption manager functionality."""
    logger.info("🔍 Testing EncryptionManager...")
    
    try:
        from security.encryption_manager import EncryptionManager
        from security.models import SecurityConfig
        
        # Create test config
        config = SecurityConfig()
        
        # Initialize encryption manager
        encryption_manager = EncryptionManager(config)
        
        # Test data encryption/decryption
        test_data = "This is sensitive data that needs encryption"
        
        # Test encryption
        encrypted_data = encryption_manager.encrypt(test_data)
        assert encrypted_data != test_data, "Data should be encrypted"
        assert len(encrypted_data) > 0, "Encrypted data should not be empty"
        logger.info("✅ Data encryption working")
        
        # Test decryption
        decrypted_data = encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == test_data, "Decrypted data should match original"
        logger.info("✅ Data decryption working")
        
        # Test password hashing
        password = "test_password_123"
        password_hash = encryption_manager.hash_password(password)
        
        assert password_hash != password, "Password should be hashed"
        assert len(password_hash) > 20, "Password hash should be substantial"
        logger.info("✅ Password hashing working")
        
        # Test password verification
        assert encryption_manager.verify_password(password, password_hash), "Correct password should verify"
        assert not encryption_manager.verify_password("wrong_password", password_hash), "Wrong password should fail"
        logger.info("✅ Password verification working")
        
        # Test API key generation
        api_key_result = encryption_manager.generate_api_key()
        assert isinstance(api_key_result, tuple), "API key generation should return a tuple"
        assert len(api_key_result) == 2, "Should return (api_key, api_key_hash)"
        
        api_key, api_key_hash = api_key_result
        assert len(api_key) >= 32, "API key should be sufficiently long"
        assert len(api_key_hash) > 0, "API key hash should not be empty"
        logger.info("✅ API key generation working")
        
        # Test secure token generation
        token = encryption_manager.create_secure_token(16)
        assert len(token) >= 16, "Secure token should be proper length"
        logger.info("✅ Secure token generation working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ EncryptionManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_enums():
    """Test security enum functionality."""
    logger.info("🔍 Testing security enums...")
    
    try:
        from security.models import UserRole, Permission, AuditAction
        
        # Test UserRole enum
        roles = [UserRole.SUPER_ADMIN, UserRole.ADMIN, UserRole.MODERATOR, UserRole.EDITOR, UserRole.VIEWER, UserRole.API_USER]
        assert len(roles) == 6, "Should have 6 user roles"
        
        # Test role hierarchy
        assert UserRole.SUPER_ADMIN.level == 100, "Super admin should have highest level"
        assert UserRole.VIEWER.level == 20, "Viewer should have low level"
        logger.info("✅ UserRole enum working")
        
        # Test Permission enum
        permissions = [
            Permission.CONTENT_CREATE, Permission.CONTENT_READ, Permission.CONTENT_UPDATE, Permission.CONTENT_DELETE,
            Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
            Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR, Permission.SYSTEM_AUDIT,
            Permission.API_FULL_ACCESS, Permission.API_READ_ONLY, Permission.API_RATE_LIMIT_EXEMPT
        ]
        assert len(permissions) >= 14, "Should have sufficient permissions"
        logger.info("✅ Permission enum working")
        
        # Test AuditAction enum
        audit_actions = [
            AuditAction.LOGIN, AuditAction.LOGOUT, AuditAction.LOGIN_FAILED,
            AuditAction.CONTENT_CREATE, AuditAction.API_CALL,
            AuditAction.SECURITY_VIOLATION, AuditAction.GDPR_REQUEST
        ]
        assert len(audit_actions) >= 7, "Should have sufficient audit actions"
        logger.info("✅ AuditAction enum working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Security enums test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all standalone security tests."""
    logger.info("🚀 Starting standalone security system validation...")
    
    test_results = []
    
    # Run basic tests
    logger.info("\n" + "="*50)
    logger.info("STANDALONE SECURITY TESTS")
    logger.info("="*50)
    
    test_results.append(("Security Models", test_security_models()))
    test_results.append(("Encryption Manager", test_encryption_manager()))
    test_results.append(("Security Enums", test_security_enums()))
    
    # Print results summary
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal Tests: {len(test_results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 ALL STANDALONE SECURITY TESTS PASSED!")
        return True
    else:
        logger.error(f"❌ {failed} tests failed. Security system needs attention.")
        return False

if __name__ == "__main__":
    result = run_all_tests()
    if result:
        print("\n✅ Standalone security system validation completed successfully!")
        print("✨ Core security components (models, encryption) are working correctly!")
        print("⚠️  Database-dependent components (RBAC, audit, etc.) need database setup for full testing.")
        exit(0)
    else:
        print("\n❌ Standalone security system validation failed!")
        exit(1) 