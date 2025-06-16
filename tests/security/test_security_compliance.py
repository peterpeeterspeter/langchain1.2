"""
Security Compliance Testing Framework
Comprehensive testing for GDPR, data protection, privacy controls, and regulatory compliance
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import json
import hashlib
import uuid

# Import security system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from security.security_manager import SecurityManager
from security.models import (
    SecurityContext, SecurityConfig, UserRole, Permission, 
    AuditAction, SecurityViolation
)
from security.managers.gdpr_compliance_manager import GDPRComplianceManager
from security.managers.audit_logger import AuditLogger
from security.encryption_manager import EncryptionManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGDPRCompliance:
    """Comprehensive GDPR compliance testing"""
    
    @pytest.fixture
    async def gdpr_manager(self):
        """Create GDPR compliance manager for testing"""
        config = SecurityConfig(enable_gdpr_compliance=True)
        manager = GDPRComplianceManager(config)
        
        # Mock database connections
        with patch('security.utils.database_utils.get_database_connection'):
            await manager.initialize()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_data_subject_rights(self, gdpr_manager):
        """Test GDPR data subject rights implementation"""
        logger.info("📋 Testing GDPR Data Subject Rights...")
        
        user_id = "test-user-123"
        
        # Test Right of Access (Article 15)
        with patch.object(gdpr_manager, '_export_user_data') as mock_export:
            mock_export.return_value = {
                "user_data": {
                    "user_id": user_id,
                    "email": "user@example.com",
                    "created_at": "2024-01-01T00:00:00Z",
                    "search_history": [
                        {"query": "casino games", "timestamp": "2024-01-15T10:00:00Z"},
                        {"query": "poker rules", "timestamp": "2024-01-16T11:00:00Z"}
                    ],
                    "downloaded_images": [
                        {"url": "https://example.com/image1.jpg", "downloaded_at": "2024-01-15T10:05:00Z"}
                    ]
                },
                "processing_activities": [
                    {"activity": "search_processing", "legal_basis": "legitimate_interest"},
                    {"activity": "image_storage", "legal_basis": "consent"}
                ]
            }
            
            access_result = await gdpr_manager.process_data_request(
                user_id=user_id,
                request_type="access",
                requester_email="user@example.com"
            )
            
            assert access_result["status"] == "completed"
            assert "user_data" in access_result
            assert access_result["user_data"]["user_id"] == user_id
        
        # Test Right to Rectification (Article 16)
        with patch.object(gdpr_manager, '_update_user_data') as mock_update:
            mock_update.return_value = {"updated": True, "fields_modified": ["email"]}
            
            rectification_result = await gdpr_manager.process_data_request(
                user_id=user_id,
                request_type="rectification",
                data_updates={"email": "newemail@example.com"},
                requester_email="user@example.com"
            )
            
            assert rectification_result["status"] == "completed"
            assert rectification_result["fields_modified"] == ["email"]
        
        # Test Right to Erasure (Article 17)
        with patch.object(gdpr_manager, '_delete_user_data') as mock_delete:
            mock_delete.return_value = {
                "deleted": True,
                "records_removed": 15,
                "anonymized_records": 5
            }
            
            erasure_result = await gdpr_manager.process_data_request(
                user_id=user_id,
                request_type="erasure",
                erasure_reason="withdrawal_of_consent",
                requester_email="user@example.com"
            )
            
            assert erasure_result["status"] == "completed"
            assert erasure_result["records_removed"] > 0
        
        # Test Right to Data Portability (Article 20)
        with patch.object(gdpr_manager, '_export_portable_data') as mock_portable:
            mock_portable.return_value = {
                "export_format": "json",
                "file_path": "/exports/user-123-data.json",
                "file_size_bytes": 2048,
                "data_categories": ["profile", "search_history", "preferences"]
            }
            
            portability_result = await gdpr_manager.process_data_request(
                user_id=user_id,
                request_type="portability",
                export_format="json",
                requester_email="user@example.com"
            )
            
            assert portability_result["status"] == "completed"
            assert portability_result["export_format"] == "json"
            assert portability_result["file_size_bytes"] > 0
        
        # Test Right to Object (Article 21)
        with patch.object(gdpr_manager, '_process_objection') as mock_object:
            mock_object.return_value = {
                "objection_processed": True,
                "processing_stopped": ["marketing", "profiling"],
                "processing_continued": ["service_delivery"]
            }
            
            objection_result = await gdpr_manager.process_data_request(
                user_id=user_id,
                request_type="objection",
                objection_scope=["marketing", "profiling"],
                requester_email="user@example.com"
            )
            
            assert objection_result["status"] == "completed"
            assert "marketing" in objection_result["processing_stopped"]
        
        logger.info("✅ GDPR Data Subject Rights test passed")
    
    @pytest.mark.asyncio
    async def test_consent_management(self, gdpr_manager):
        """Test GDPR consent management"""
        logger.info("✅ Testing GDPR Consent Management...")
        
        user_id = "test-user-456"
        
        # Test consent recording
        consent_data = {
            "user_id": user_id,
            "consent_type": "data_processing",
            "purpose": "image_search_and_storage",
            "legal_basis": "consent",
            "consent_given": True,
            "consent_method": "explicit_checkbox",
            "consent_timestamp": datetime.utcnow().isoformat(),
            "consent_version": "1.0"
        }
        
        with patch.object(gdpr_manager, '_record_consent') as mock_record:
            mock_record.return_value = {"consent_id": "consent-123", "recorded": True}
            
            consent_result = await gdpr_manager.record_consent(consent_data)
            
            assert consent_result["recorded"] == True
            assert "consent_id" in consent_result
        
        # Test consent withdrawal
        with patch.object(gdpr_manager, '_withdraw_consent') as mock_withdraw:
            mock_withdraw.return_value = {
                "withdrawn": True,
                "withdrawal_timestamp": datetime.utcnow().isoformat(),
                "data_processing_stopped": True
            }
            
            withdrawal_result = await gdpr_manager.withdraw_consent(
                user_id=user_id,
                consent_type="data_processing",
                withdrawal_reason="user_request"
            )
            
            assert withdrawal_result["withdrawn"] == True
            assert withdrawal_result["data_processing_stopped"] == True
        
        # Test consent history tracking
        with patch.object(gdpr_manager, '_get_consent_history') as mock_history:
            mock_history.return_value = [
                {
                    "consent_id": "consent-123",
                    "consent_given": True,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "version": "1.0"
                },
                {
                    "consent_id": "consent-124",
                    "consent_given": False,
                    "timestamp": "2024-01-15T12:00:00Z",
                    "version": "1.0",
                    "withdrawal_reason": "user_request"
                }
            ]
            
            history = await gdpr_manager.get_consent_history(user_id)
            
            assert len(history) == 2
            assert history[0]["consent_given"] == True
            assert history[1]["consent_given"] == False
        
        logger.info("✅ GDPR Consent Management test passed")
    
    @pytest.mark.asyncio
    async def test_data_retention_policies(self, gdpr_manager):
        """Test data retention and deletion policies"""
        logger.info("🗂️ Testing Data Retention Policies...")
        
        # Test retention policy configuration
        retention_policies = {
            "user_profiles": {"retention_days": 2555, "legal_basis": "contract"},  # 7 years
            "search_history": {"retention_days": 365, "legal_basis": "legitimate_interest"},  # 1 year
            "audit_logs": {"retention_days": 2555, "legal_basis": "legal_obligation"},  # 7 years
            "image_downloads": {"retention_days": 90, "legal_basis": "consent"},  # 3 months
            "api_keys": {"retention_days": 30, "legal_basis": "contract"}  # 30 days after expiry
        }
        
        with patch.object(gdpr_manager, '_apply_retention_policies') as mock_apply:
            mock_apply.return_value = {
                "policies_applied": 5,
                "records_reviewed": 1000,
                "records_deleted": 150,
                "records_anonymized": 50
            }
            
            retention_result = await gdpr_manager.apply_retention_policies(retention_policies)
            
            assert retention_result["policies_applied"] == 5
            assert retention_result["records_deleted"] > 0
        
        # Test automated cleanup
        with patch.object(gdpr_manager, '_run_automated_cleanup') as mock_cleanup:
            mock_cleanup.return_value = {
                "cleanup_run_id": "cleanup-789",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
                "categories_processed": ["expired_sessions", "old_search_history", "unused_api_keys"],
                "total_records_processed": 500,
                "records_deleted": 75,
                "records_anonymized": 25,
                "errors": []
            }
            
            cleanup_result = await gdpr_manager.run_automated_cleanup()
            
            assert cleanup_result["total_records_processed"] > 0
            assert cleanup_result["records_deleted"] > 0
            assert len(cleanup_result["errors"]) == 0
        
        logger.info("✅ Data Retention Policies test passed")
    
    @pytest.mark.asyncio
    async def test_privacy_by_design(self, gdpr_manager):
        """Test Privacy by Design principles implementation"""
        logger.info("🔒 Testing Privacy by Design...")
        
        # Test data minimization
        data_collection_request = {
            "user_registration": {
                "required_fields": ["email", "password"],
                "optional_fields": ["name", "preferences"],
                "prohibited_fields": ["ssn", "credit_card"]
            },
            "search_tracking": {
                "collect_queries": True,
                "collect_ip_hash": True,  # Hashed, not raw IP
                "collect_full_ip": False,
                "collect_user_agent": False
            }
        }
        
        with patch.object(gdpr_manager, '_validate_data_minimization') as mock_validate:
            mock_validate.return_value = {
                "compliant": True,
                "violations": [],
                "recommendations": ["Consider reducing optional fields"]
            }
            
            minimization_result = await gdpr_manager.validate_data_minimization(data_collection_request)
            
            assert minimization_result["compliant"] == True
            assert len(minimization_result["violations"]) == 0
        
        # Test purpose limitation
        processing_purposes = {
            "user_authentication": {
                "data_types": ["email", "password_hash"],
                "legal_basis": "contract",
                "retention_period": "account_lifetime"
            },
            "search_improvement": {
                "data_types": ["search_queries", "result_clicks"],
                "legal_basis": "legitimate_interest",
                "retention_period": "1_year"
            },
            "marketing": {
                "data_types": ["email", "preferences"],
                "legal_basis": "consent",
                "retention_period": "consent_duration"
            }
        }
        
        with patch.object(gdpr_manager, '_validate_purpose_limitation') as mock_purpose:
            mock_purpose.return_value = {
                "compliant": True,
                "purpose_violations": [],
                "cross_purpose_usage": []
            }
            
            purpose_result = await gdpr_manager.validate_purpose_limitation(processing_purposes)
            
            assert purpose_result["compliant"] == True
            assert len(purpose_result["purpose_violations"]) == 0
        
        # Test storage limitation
        with patch.object(gdpr_manager, '_check_storage_limitation') as mock_storage:
            mock_storage.return_value = {
                "compliant": True,
                "excessive_storage": [],
                "retention_violations": [],
                "total_data_size_mb": 150.5
            }
            
            storage_result = await gdpr_manager.check_storage_limitation()
            
            assert storage_result["compliant"] == True
            assert storage_result["total_data_size_mb"] > 0
        
        logger.info("✅ Privacy by Design test passed")


class TestDataProtection:
    """Data protection and security testing"""
    
    @pytest.fixture
    async def encryption_manager(self):
        """Create encryption manager for testing"""
        manager = EncryptionManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_data_encryption_at_rest(self, encryption_manager):
        """Test data encryption at rest"""
        logger.info("🔐 Testing Data Encryption at Rest...")
        
        # Test sensitive data encryption
        sensitive_data = {
            "user_id": "user-123",
            "email": "user@example.com",
            "api_key": "sk-1234567890abcdef",
            "search_history": [
                {"query": "casino games", "timestamp": "2024-01-01T10:00:00Z"},
                {"query": "poker strategy", "timestamp": "2024-01-01T11:00:00Z"}
            ],
            "personal_info": {
                "name": "John Doe",
                "phone": "+1234567890",
                "address": "123 Main St, City, State"
            }
        }
        
        # Define which fields should be encrypted
        sensitive_fields = ["api_key", "search_history", "personal_info"]
        
        # Test encryption
        encrypted_data = await encryption_manager.encrypt_sensitive_fields(
            sensitive_data, sensitive_fields
        )
        
        # Verify non-sensitive fields are unchanged
        assert encrypted_data["user_id"] == sensitive_data["user_id"]
        assert encrypted_data["email"] == sensitive_data["email"]
        
        # Verify sensitive fields are encrypted
        assert encrypted_data["api_key"] != sensitive_data["api_key"]
        assert encrypted_data["search_history"] != sensitive_data["search_history"]
        assert encrypted_data["personal_info"] != sensitive_data["personal_info"]
        
        # Test decryption
        decrypted_data = await encryption_manager.decrypt_sensitive_fields(
            encrypted_data, sensitive_fields
        )
        
        assert decrypted_data == sensitive_data
        
        logger.info("✅ Data Encryption at Rest test passed")
    
    @pytest.mark.asyncio
    async def test_data_encryption_in_transit(self, encryption_manager):
        """Test data encryption in transit"""
        logger.info("🌐 Testing Data Encryption in Transit...")
        
        # Test API payload encryption
        api_payload = {
            "user_id": "user-456",
            "search_request": {
                "keyword": "blackjack strategy",
                "filters": {"safe_search": True, "image_size": "large"}
            },
            "api_key": "sk-abcdef1234567890"
        }
        
        # Encrypt payload for transmission
        encrypted_payload = await encryption_manager.encrypt_api_payload(api_payload)
        
        assert "encrypted_data" in encrypted_payload
        assert "encryption_key_id" in encrypted_payload
        assert "timestamp" in encrypted_payload
        assert encrypted_payload["encrypted_data"] != json.dumps(api_payload)
        
        # Decrypt payload
        decrypted_payload = await encryption_manager.decrypt_api_payload(encrypted_payload)
        
        assert decrypted_payload == api_payload
        
        # Test TLS certificate validation (mock)
        with patch.object(encryption_manager, '_validate_tls_certificate') as mock_tls:
            mock_tls.return_value = {
                "valid": True,
                "issuer": "Let's Encrypt",
                "expires_at": "2024-12-31T23:59:59Z",
                "subject": "api.example.com"
            }
            
            tls_result = await encryption_manager.validate_tls_certificate("api.example.com")
            
            assert tls_result["valid"] == True
            assert "expires_at" in tls_result
        
        logger.info("✅ Data Encryption in Transit test passed")
    
    @pytest.mark.asyncio
    async def test_key_management(self, encryption_manager):
        """Test encryption key management"""
        logger.info("🔑 Testing Key Management...")
        
        # Test key generation
        key_info = await encryption_manager.generate_encryption_key(
            key_type="AES-256",
            purpose="data_encryption",
            expires_in_days=365
        )
        
        assert "key_id" in key_info
        assert "key_type" in key_info
        assert key_info["key_type"] == "AES-256"
        assert key_info["purpose"] == "data_encryption"
        
        # Test key rotation
        old_key_id = encryption_manager.current_key_id
        rotation_result = await encryption_manager.rotate_encryption_key()
        
        assert rotation_result["rotated"] == True
        assert rotation_result["old_key_id"] == old_key_id
        assert rotation_result["new_key_id"] != old_key_id
        
        # Test key backup and recovery
        with patch.object(encryption_manager, '_backup_key') as mock_backup:
            mock_backup.return_value = {
                "backup_id": "backup-123",
                "backup_location": "secure_vault",
                "backup_timestamp": datetime.utcnow().isoformat()
            }
            
            backup_result = await encryption_manager.backup_encryption_key(key_info["key_id"])
            
            assert backup_result["backup_id"] is not None
            assert backup_result["backup_location"] == "secure_vault"
        
        # Test key expiration handling
        with patch.object(encryption_manager, '_check_key_expiration') as mock_expiry:
            mock_expiry.return_value = {
                "expired_keys": ["key-old-1", "key-old-2"],
                "expiring_soon": ["key-warn-1"],
                "actions_taken": ["rotated_expired", "warned_expiring"]
            }
            
            expiry_result = await encryption_manager.check_key_expiration()
            
            assert len(expiry_result["expired_keys"]) >= 0
            assert "actions_taken" in expiry_result
        
        logger.info("✅ Key Management test passed")


class TestRegulatoryCompliance:
    """Regulatory compliance testing"""
    
    @pytest.mark.asyncio
    async def test_audit_trail_compliance(self):
        """Test audit trail compliance requirements"""
        logger.info("📊 Testing Audit Trail Compliance...")
        
        audit_logger = AuditLogger()
        
        # Mock database operations
        with patch.object(audit_logger, '_store_audit_entry') as mock_store:
            mock_store.return_value = True
            
            # Test comprehensive audit logging
            audit_events = [
                {
                    "action": AuditAction.LOGIN_SUCCESS,
                    "user_id": "user-123",
                    "session_id": "session-456",
                    "client_ip": "192.168.1.1",
                    "details": {"authentication_method": "api_key"},
                    "severity": "low"
                },
                {
                    "action": AuditAction.DATA_ACCESS,
                    "user_id": "user-123",
                    "session_id": "session-456",
                    "client_ip": "192.168.1.1",
                    "details": {"resource": "user_data", "operation": "read"},
                    "severity": "medium"
                },
                {
                    "action": AuditAction.DATA_MODIFICATION,
                    "user_id": "admin-789",
                    "session_id": "session-999",
                    "client_ip": "10.0.0.1",
                    "details": {"resource": "user_profile", "fields_modified": ["email"]},
                    "severity": "high"
                },
                {
                    "action": AuditAction.SYSTEM_CONFIGURATION,
                    "user_id": "admin-789",
                    "session_id": "session-999",
                    "client_ip": "10.0.0.1",
                    "details": {"configuration": "security_settings", "changes": ["rate_limit_updated"]},
                    "severity": "high"
                }
            ]
            
            # Log all events
            for event in audit_events:
                success = await audit_logger.log_action(**event)
                assert success == True
            
            # Verify all events were stored
            assert mock_store.call_count == len(audit_events)
        
        # Test audit log integrity
        with patch.object(audit_logger, '_verify_audit_integrity') as mock_verify:
            mock_verify.return_value = {
                "integrity_verified": True,
                "total_entries": 1000,
                "hash_mismatches": 0,
                "timestamp_anomalies": 0,
                "missing_entries": 0
            }
            
            integrity_result = await audit_logger.verify_audit_integrity()
            
            assert integrity_result["integrity_verified"] == True
            assert integrity_result["hash_mismatches"] == 0
        
        # Test audit log retention
        with patch.object(audit_logger, '_apply_retention_policy') as mock_retention:
            mock_retention.return_value = {
                "retention_applied": True,
                "entries_archived": 500,
                "entries_deleted": 100,
                "oldest_entry_date": "2023-01-01T00:00:00Z"
            }
            
            retention_result = await audit_logger.apply_retention_policy(
                retention_days=2555  # 7 years
            )
            
            assert retention_result["retention_applied"] == True
            assert retention_result["entries_archived"] > 0
        
        logger.info("✅ Audit Trail Compliance test passed")
    
    @pytest.mark.asyncio
    async def test_data_breach_response(self):
        """Test data breach detection and response procedures"""
        logger.info("🚨 Testing Data Breach Response...")
        
        security_manager = SecurityManager()
        
        # Mock breach detection
        with patch.object(security_manager, '_detect_potential_breach') as mock_detect:
            mock_detect.return_value = {
                "breach_detected": True,
                "breach_type": "unauthorized_access",
                "severity": "high",
                "affected_users": ["user-123", "user-456"],
                "data_types_affected": ["email", "search_history"],
                "detection_timestamp": datetime.utcnow().isoformat()
            }
            
            breach_detection = await security_manager.detect_potential_breach()
            
            assert breach_detection["breach_detected"] == True
            assert breach_detection["severity"] == "high"
            assert len(breach_detection["affected_users"]) > 0
        
        # Test breach notification
        with patch.object(security_manager, '_notify_breach') as mock_notify:
            mock_notify.return_value = {
                "notifications_sent": True,
                "users_notified": 2,
                "authorities_notified": True,
                "notification_timestamp": datetime.utcnow().isoformat(),
                "notification_methods": ["email", "system_alert"]
            }
            
            notification_result = await security_manager.notify_data_breach(
                breach_info=breach_detection,
                notification_within_hours=72  # GDPR requirement
            )
            
            assert notification_result["notifications_sent"] == True
            assert notification_result["authorities_notified"] == True
        
        # Test breach containment
        with patch.object(security_manager, '_contain_breach') as mock_contain:
            mock_contain.return_value = {
                "containment_successful": True,
                "actions_taken": [
                    "disabled_affected_api_keys",
                    "reset_user_sessions",
                    "increased_monitoring"
                ],
                "containment_timestamp": datetime.utcnow().isoformat()
            }
            
            containment_result = await security_manager.contain_data_breach(breach_detection)
            
            assert containment_result["containment_successful"] == True
            assert len(containment_result["actions_taken"]) > 0
        
        logger.info("✅ Data Breach Response test passed")
    
    @pytest.mark.asyncio
    async def test_compliance_reporting(self):
        """Test compliance reporting capabilities"""
        logger.info("📈 Testing Compliance Reporting...")
        
        security_manager = SecurityManager()
        
        # Test GDPR compliance report
        with patch.object(security_manager, '_generate_gdpr_report') as mock_gdpr:
            mock_gdpr.return_value = {
                "report_id": "gdpr-report-123",
                "report_period": "2024-Q1",
                "data_subject_requests": {
                    "total": 25,
                    "access_requests": 10,
                    "rectification_requests": 5,
                    "erasure_requests": 8,
                    "portability_requests": 2
                },
                "consent_management": {
                    "consents_recorded": 150,
                    "consents_withdrawn": 12,
                    "consent_compliance_rate": 0.98
                },
                "data_breaches": {
                    "total_incidents": 0,
                    "notifications_sent": 0,
                    "average_response_time_hours": 0
                },
                "compliance_score": 0.95
            }
            
            gdpr_report = await security_manager.generate_gdpr_compliance_report(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 31)
            )
            
            assert gdpr_report["compliance_score"] > 0.9
            assert gdpr_report["data_subject_requests"]["total"] > 0
        
        # Test security metrics report
        with patch.object(security_manager, '_generate_security_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "report_id": "security-metrics-456",
                "authentication_metrics": {
                    "total_attempts": 10000,
                    "successful_authentications": 9850,
                    "failed_authentications": 150,
                    "success_rate": 0.985
                },
                "authorization_metrics": {
                    "total_requests": 50000,
                    "authorized_requests": 49500,
                    "denied_requests": 500,
                    "authorization_rate": 0.99
                },
                "encryption_metrics": {
                    "data_encrypted_gb": 125.5,
                    "key_rotations": 4,
                    "encryption_failures": 0
                },
                "audit_metrics": {
                    "total_audit_entries": 75000,
                    "high_severity_events": 25,
                    "audit_integrity_score": 1.0
                }
            }
            
            security_report = await security_manager.generate_security_metrics_report()
            
            assert security_report["authentication_metrics"]["success_rate"] > 0.95
            assert security_report["audit_metrics"]["audit_integrity_score"] == 1.0
        
        logger.info("✅ Compliance Reporting test passed")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 