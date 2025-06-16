"""
GDPR Compliance Manager for Universal RAG CMS Security System

This module provides comprehensive GDPR compliance management including
data subject rights, consent management, data retention policies,
and privacy impact assessments.

Features:
- Data subject rights (access, rectification, erasure, portability)
- Consent management and tracking
- Data retention policy enforcement
- Privacy impact assessments
- Data breach notifications
- Cookie consent management
- Cross-border data transfer compliance
- Audit trail for compliance activities
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from ..models import AuditAction, SecurityViolation
from .audit_logger import AuditLogger
from ..utils.database_utils import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)


class DataSubjectRight(Enum):
    """GDPR Data Subject Rights"""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 (Right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Article 18
    DATA_PORTABILITY = "data_portability"  # Article 20
    OBJECT = "object"  # Article 21
    WITHDRAW_CONSENT = "withdraw_consent"  # Article 7


class ConsentType(Enum):
    """Types of consent for data processing"""
    NECESSARY = "necessary"
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PERSONALIZATION = "personalization"


class ConsentStatus(Enum):
    """Consent status values"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


class ProcessingLawfulBasis(Enum):
    """GDPR Article 6 lawful basis for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataCategory(Enum):
    """Categories of personal data"""
    BASIC_IDENTITY = "basic_identity"
    CONTACT_INFO = "contact_info"
    DEMOGRAPHIC = "demographic"
    FINANCIAL = "financial"
    BEHAVIORAL = "behavioral"
    PREFERENCES = "preferences"
    TECHNICAL = "technical"
    SPECIAL_CATEGORY = "special_category"  # Article 9 data


@dataclass
class ConsentRecord:
    """Record of user consent"""
    consent_id: str
    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    lawful_basis: ProcessingLawfulBasis
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    consent_text: str = ""
    purpose: str = ""
    data_categories: List[DataCategory] = field(default_factory=list)
    third_parties: List[str] = field(default_factory=list)
    retention_period: Optional[int] = None  # in days
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSubjectRequest:
    """Data subject rights request"""
    request_id: str
    user_id: str
    user_email: str
    request_type: DataSubjectRight
    status: str  # pending, in_progress, completed, rejected
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    verification_status: str = "pending"
    request_details: Dict[str, Any] = field(default_factory=dict)
    response_data: Optional[Dict[str, Any]] = None
    processing_notes: List[str] = field(default_factory=list)


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition"""
    policy_id: str
    data_category: DataCategory
    retention_period_days: int
    lawful_basis: ProcessingLawfulBasis
    deletion_method: str  # secure_delete, anonymize, archive
    exceptions: List[str] = field(default_factory=list)
    review_frequency_days: int = 90
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA)"""
    pia_id: str
    project_name: str
    data_controller: str
    processing_purpose: str
    data_categories: List[DataCategory]
    lawful_basis: ProcessingLawfulBasis
    risk_level: str  # low, medium, high
    mitigation_measures: List[str]
    completed_at: datetime
    review_date: datetime
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


class GDPRComplianceManager:
    """
    Comprehensive GDPR compliance management system.
    
    Handles all aspects of GDPR compliance including data subject rights,
    consent management, data retention, and privacy assessments.
    """
    
    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize GDPR Compliance Manager"""
        
        self.db = database_manager
        self.audit_logger = audit_logger
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default retention policies
        self.default_retention_policies = self._initialize_default_policies()
        
        # Consent management settings
        self.consent_expiry_days = self.config.get('consent_expiry_days', 365)
        self.verification_required_for_requests = self.config.get('verification_required', True)
        
        # Data processing settings
        self.auto_delete_expired_data = self.config.get('auto_delete_expired_data', True)
        self.data_retention_check_frequency = self.config.get('retention_check_frequency', 24)  # hours
        
        # Compliance metrics
        self.metrics = {
            'consent_records': 0,
            'data_subject_requests': 0,
            'completed_requests': 0,
            'data_deletions': 0,
            'consent_withdrawals': 0,
            'privacy_assessments': 0,
            'retention_violations': 0
        }
        
        self.logger.info("GDPR Compliance Manager initialized")
    
    def _initialize_default_policies(self) -> List[DataRetentionPolicy]:
        """Initialize default data retention policies"""
        
        return [
            DataRetentionPolicy(
                policy_id="basic_identity_policy",
                data_category=DataCategory.BASIC_IDENTITY,
                retention_period_days=2555,  # 7 years
                lawful_basis=ProcessingLawfulBasis.CONTRACT,
                deletion_method="secure_delete"
            ),
            DataRetentionPolicy(
                policy_id="contact_info_policy",
                data_category=DataCategory.CONTACT_INFO,
                retention_period_days=1095,  # 3 years
                lawful_basis=ProcessingLawfulBasis.LEGITIMATE_INTERESTS,
                deletion_method="secure_delete"
            ),
            DataRetentionPolicy(
                policy_id="behavioral_policy",
                data_category=DataCategory.BEHAVIORAL,
                retention_period_days=730,  # 2 years
                lawful_basis=ProcessingLawfulBasis.CONSENT,
                deletion_method="anonymize"
            ),
            DataRetentionPolicy(
                policy_id="technical_policy",
                data_category=DataCategory.TECHNICAL,
                retention_period_days=180,  # 6 months
                lawful_basis=ProcessingLawfulBasis.LEGITIMATE_INTERESTS,
                deletion_method="secure_delete"
            ),
            DataRetentionPolicy(
                policy_id="marketing_policy",
                data_category=DataCategory.PREFERENCES,
                retention_period_days=1095,  # 3 years
                lawful_basis=ProcessingLawfulBasis.CONSENT,
                deletion_method="secure_delete"
            )
        ]
    
    async def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        status: ConsentStatus,
        lawful_basis: ProcessingLawfulBasis,
        consent_text: str,
        purpose: str,
        data_categories: List[DataCategory],
        retention_period: Optional[int] = None,
        third_parties: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record user consent for data processing.
        
        Args:
            user_id: Unique user identifier
            consent_type: Type of consent being recorded
            status: Consent status (granted/denied)
            lawful_basis: GDPR Article 6 lawful basis
            consent_text: Text of consent request shown to user
            purpose: Purpose of data processing
            data_categories: Categories of data being processed
            retention_period: Data retention period in days
            third_parties: List of third parties data may be shared with
            metadata: Additional metadata
            
        Returns:
            Consent record ID
        """
        
        try:
            consent_id = str(uuid.uuid4())
            
            # Calculate expiry date
            expires_at = None
            if retention_period or self.consent_expiry_days:
                days = retention_period or self.consent_expiry_days
                expires_at = datetime.utcnow() + timedelta(days=days)
            
            consent_record = ConsentRecord(
                consent_id=consent_id,
                user_id=user_id,
                consent_type=consent_type,
                status=status,
                lawful_basis=lawful_basis,
                granted_at=datetime.utcnow(),
                expires_at=expires_at,
                consent_text=consent_text,
                purpose=purpose,
                data_categories=data_categories,
                third_parties=third_parties or [],
                retention_period=retention_period,
                metadata=metadata or {}
            )
            
            # Store in database
            if self.db:
                await self._store_consent_record(consent_record)
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event(
                    action=AuditAction.CONSENT_RECORDED,
                    user_id=user_id,
                    resource_type="consent",
                    resource_id=consent_id,
                    details={
                        'consent_type': consent_type.value,
                        'status': status.value,
                        'lawful_basis': lawful_basis.value,
                        'purpose': purpose,
                        'data_categories': [cat.value for cat in data_categories],
                        'expires_at': expires_at.isoformat() if expires_at else None
                    }
                )
            
            self.metrics['consent_records'] += 1
            
            self.logger.info(f"Consent recorded: {consent_id} for user {user_id}")
            return consent_id
            
        except Exception as e:
            self.logger.error(f"Failed to record consent: {e}")
            raise SecurityViolation(f"Consent recording failed: {str(e)}")
    
    async def withdraw_consent(
        self,
        user_id: str,
        consent_id: Optional[str] = None,
        consent_type: Optional[ConsentType] = None
    ) -> List[str]:
        """
        Withdraw user consent for data processing.
        
        Args:
            user_id: User identifier
            consent_id: Specific consent to withdraw (optional)
            consent_type: Type of consent to withdraw (optional)
            
        Returns:
            List of withdrawn consent IDs
        """
        
        try:
            withdrawn_consents = []
            
            if consent_id:
                # Withdraw specific consent
                withdrawn_consents = await self._withdraw_specific_consent(user_id, consent_id)
            elif consent_type:
                # Withdraw all consents of specific type
                withdrawn_consents = await self._withdraw_consent_by_type(user_id, consent_type)
            else:
                # Withdraw all consents for user
                withdrawn_consents = await self._withdraw_all_consent(user_id)
            
            # Trigger data processing based on consent withdrawal
            await self._process_consent_withdrawal(user_id, withdrawn_consents)
            
            # Audit logging
            if self.audit_logger:
                for consent_id in withdrawn_consents:
                    await self.audit_logger.log_event(
                        action=AuditAction.CONSENT_WITHDRAWN,
                        user_id=user_id,
                        resource_type="consent",
                        resource_id=consent_id,
                        details={
                            'withdrawal_type': 'specific' if consent_id else 'bulk',
                            'consent_type': consent_type.value if consent_type else 'all'
                        }
                    )
            
            self.metrics['consent_withdrawals'] += len(withdrawn_consents)
            
            return withdrawn_consents
            
        except Exception as e:
            self.logger.error(f"Failed to withdraw consent: {e}")
            raise SecurityViolation(f"Consent withdrawal failed: {str(e)}")
    
    async def submit_data_subject_request(
        self,
        user_id: str,
        user_email: str,
        request_type: DataSubjectRight,
        request_details: Optional[Dict[str, Any]] = None,
        verification_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a data subject rights request.
        
        Args:
            user_id: User identifier
            user_email: User email for verification
            request_type: Type of request (access, erasure, etc.)
            request_details: Additional request details
            verification_data: Identity verification data
            
        Returns:
            Request ID
        """
        
        try:
            request_id = str(uuid.uuid4())
            
            # Verify user identity if required
            verification_status = "verified"
            if self.verification_required_for_requests:
                verification_status = await self._verify_user_identity(
                    user_id, user_email, verification_data
                )
            
            request = DataSubjectRequest(
                request_id=request_id,
                user_id=user_id,
                user_email=user_email,
                request_type=request_type,
                status="pending",
                submitted_at=datetime.utcnow(),
                verification_status=verification_status,
                request_details=request_details or {}
            )
            
            # Store request
            if self.db:
                await self._store_data_subject_request(request)
            
            # Auto-process certain types of requests
            if request_type in [DataSubjectRight.ACCESS, DataSubjectRight.DATA_PORTABILITY]:
                await self._auto_process_request(request)
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event(
                    action=AuditAction.DATA_SUBJECT_REQUEST,
                    user_id=user_id,
                    resource_type="data_request",
                    resource_id=request_id,
                    details={
                        'request_type': request_type.value,
                        'verification_status': verification_status,
                        'auto_processed': request_type in [DataSubjectRight.ACCESS, DataSubjectRight.DATA_PORTABILITY]
                    }
                )
            
            self.metrics['data_subject_requests'] += 1
            
            self.logger.info(f"Data subject request submitted: {request_id}")
            return request_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit data subject request: {e}")
            raise SecurityViolation(f"Data subject request failed: {str(e)}")
    
    async def process_erasure_request(
        self,
        request_id: str,
        verification_confirmed: bool = True
    ) -> Dict[str, Any]:
        """
        Process a data erasure (right to be forgotten) request.
        
        Args:
            request_id: Request identifier
            verification_confirmed: Whether identity verification is confirmed
            
        Returns:
            Processing results
        """
        
        try:
            if not self.db:
                raise SecurityViolation("Database connection required for data erasure")
            
            # Get request details
            request = await self._get_data_subject_request(request_id)
            if not request or request.request_type != DataSubjectRight.ERASURE:
                raise SecurityViolation("Invalid erasure request")
            
            if not verification_confirmed:
                raise SecurityViolation("Identity verification required for data erasure")
            
            # Identify data to be erased
            data_inventory = await self._get_user_data_inventory(request.user_id)
            
            # Check for legal holds or exceptions
            legal_holds = await self._check_legal_holds(request.user_id)
            if legal_holds:
                request.status = "partially_completed"
                request.processing_notes.append(f"Legal holds prevent full erasure: {legal_holds}")
            
            # Perform data erasure
            erasure_results = await self._perform_data_erasure(request.user_id, data_inventory, legal_holds)
            
            # Update request status
            request.status = "completed" if not legal_holds else "partially_completed"
            request.completed_at = datetime.utcnow()
            request.response_data = erasure_results
            
            # Store updated request
            await self._update_data_subject_request(request)
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event(
                    action=AuditAction.DATA_ERASED,
                    user_id=request.user_id,
                    resource_type="user_data",
                    resource_id=request.user_id,
                    details={
                        'request_id': request_id,
                        'erasure_scope': erasure_results,
                        'legal_holds': legal_holds,
                        'full_erasure': not legal_holds
                    }
                )
            
            self.metrics['data_deletions'] += 1
            self.metrics['completed_requests'] += 1
            
            return erasure_results
            
        except Exception as e:
            self.logger.error(f"Failed to process erasure request: {e}")
            raise SecurityViolation(f"Data erasure failed: {str(e)}")
    
    async def generate_data_export(
        self,
        user_id: str,
        request_id: Optional[str] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate data export for data portability or access requests.
        
        Args:
            user_id: User identifier
            request_id: Associated request ID
            format: Export format (json, csv, xml)
            
        Returns:
            Exported data package
        """
        
        try:
            # Get comprehensive user data
            user_data = await self._collect_user_data(user_id)
            
            # Format data according to GDPR requirements
            formatted_data = {
                'user_id': user_id,
                'export_date': datetime.utcnow().isoformat(),
                'request_id': request_id,
                'data_categories': {},
                'consent_history': await self._get_user_consent_history(user_id),
                'processing_activities': await self._get_user_processing_activities(user_id)
            }
            
            # Organize data by category
            for category, data in user_data.items():
                if data:  # Only include non-empty categories
                    formatted_data['data_categories'][category] = data
            
            # Add metadata
            formatted_data['metadata'] = {
                'export_format': format,
                'gdpr_compliance': True,
                'retention_policies': await self._get_applicable_retention_policies(user_id),
                'lawful_basis': await self._get_processing_lawful_basis(user_id)
            }
            
            # Convert to requested format
            if format.lower() == "csv":
                formatted_data = await self._convert_to_csv(formatted_data)
            elif format.lower() == "xml":
                formatted_data = await self._convert_to_xml(formatted_data)
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event(
                    action=AuditAction.DATA_EXPORTED,
                    user_id=user_id,
                    resource_type="user_data",
                    resource_id=user_id,
                    details={
                        'request_id': request_id,
                        'export_format': format,
                        'data_categories_count': len(formatted_data.get('data_categories', {}))
                    }
                )
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate data export: {e}")
            raise SecurityViolation(f"Data export failed: {str(e)}")
    
    async def enforce_data_retention(self) -> Dict[str, Any]:
        """
        Enforce data retention policies by identifying and processing expired data.
        
        Returns:
            Retention enforcement results
        """
        
        try:
            enforcement_results = {
                'processed_policies': 0,
                'expired_records': 0,
                'deleted_records': 0,
                'anonymized_records': 0,
                'archived_records': 0,
                'errors': []
            }
            
            # Get all active retention policies
            policies = await self._get_active_retention_policies()
            
            for policy in policies:
                try:
                    # Find expired data for this policy
                    expired_data = await self._find_expired_data(policy)
                    
                    if expired_data:
                        # Process expired data according to policy
                        if policy.deletion_method == "secure_delete":
                            deleted = await self._secure_delete_data(expired_data)
                            enforcement_results['deleted_records'] += deleted
                        elif policy.deletion_method == "anonymize":
                            anonymized = await self._anonymize_data(expired_data)
                            enforcement_results['anonymized_records'] += anonymized
                        elif policy.deletion_method == "archive":
                            archived = await self._archive_data(expired_data)
                            enforcement_results['archived_records'] += archived
                        
                        enforcement_results['expired_records'] += len(expired_data)
                    
                    enforcement_results['processed_policies'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process policy {policy.policy_id}: {str(e)}"
                    enforcement_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event(
                    action=AuditAction.DATA_RETENTION_ENFORCED,
                    user_id="system",
                    resource_type="retention_policy",
                    details=enforcement_results
                )
            
            return enforcement_results
            
        except Exception as e:
            self.logger.error(f"Data retention enforcement failed: {e}")
            raise SecurityViolation(f"Retention enforcement failed: {str(e)}")
    
    async def create_privacy_impact_assessment(
        self,
        project_name: str,
        data_controller: str,
        processing_purpose: str,
        data_categories: List[DataCategory],
        lawful_basis: ProcessingLawfulBasis,
        risk_assessment: Dict[str, Any]
    ) -> str:
        """
        Create a Privacy Impact Assessment (PIA).
        
        Args:
            project_name: Name of the project/processing activity
            data_controller: Data controller responsible
            processing_purpose: Purpose of data processing
            data_categories: Categories of personal data involved
            lawful_basis: Legal basis for processing
            risk_assessment: Risk assessment results
            
        Returns:
            PIA ID
        """
        
        try:
            pia_id = str(uuid.uuid4())
            
            # Assess risk level
            risk_level = self._assess_privacy_risk(data_categories, risk_assessment)
            
            # Generate mitigation measures
            mitigation_measures = self._generate_mitigation_measures(risk_level, data_categories)
            
            pia = PrivacyImpactAssessment(
                pia_id=pia_id,
                project_name=project_name,
                data_controller=data_controller,
                processing_purpose=processing_purpose,
                data_categories=data_categories,
                lawful_basis=lawful_basis,
                risk_level=risk_level,
                mitigation_measures=mitigation_measures,
                completed_at=datetime.utcnow(),
                review_date=datetime.utcnow() + timedelta(days=365),  # Annual review
                metadata=risk_assessment
            )
            
            # Store PIA
            if self.db:
                await self._store_privacy_impact_assessment(pia)
            
            # Audit logging
            if self.audit_logger:
                await self.audit_logger.log_event(
                    action=AuditAction.PRIVACY_ASSESSMENT_CREATED,
                    user_id="system",
                    resource_type="privacy_assessment",
                    resource_id=pia_id,
                    details={
                        'project_name': project_name,
                        'risk_level': risk_level,
                        'data_categories': [cat.value for cat in data_categories],
                        'lawful_basis': lawful_basis.value
                    }
                )
            
            self.metrics['privacy_assessments'] += 1
            
            return pia_id
            
        except Exception as e:
            self.logger.error(f"Failed to create PIA: {e}")
            raise SecurityViolation(f"PIA creation failed: {str(e)}")
    
    # Helper methods (implementation details)
    
    async def _store_consent_record(self, consent: ConsentRecord):
        """Store consent record in database"""
        # Implementation would store in actual database
        pass
    
    async def _withdraw_specific_consent(self, user_id: str, consent_id: str) -> List[str]:
        """Withdraw specific consent"""
        # Implementation would update database
        return [consent_id]
    
    async def _withdraw_consent_by_type(self, user_id: str, consent_type: ConsentType) -> List[str]:
        """Withdraw all consents of specific type"""
        # Implementation would query and update database
        return []
    
    async def _withdraw_all_consent(self, user_id: str) -> List[str]:
        """Withdraw all consents for user"""
        # Implementation would update all user consents
        return []
    
    async def _process_consent_withdrawal(self, user_id: str, consent_ids: List[str]):
        """Process implications of consent withdrawal"""
        # Implementation would handle data processing changes
        pass
    
    async def _verify_user_identity(
        self, 
        user_id: str, 
        email: str, 
        verification_data: Optional[Dict[str, Any]]
    ) -> str:
        """Verify user identity for data subject requests"""
        # Implementation would perform identity verification
        return "verified"
    
    async def _store_data_subject_request(self, request: DataSubjectRequest):
        """Store data subject request in database"""
        # Implementation would store in database
        pass
    
    async def _get_data_subject_request(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Get data subject request from database"""
        # Implementation would retrieve from database
        return None
    
    async def _auto_process_request(self, request: DataSubjectRequest):
        """Auto-process simple requests"""
        # Implementation would handle automatic processing
        pass
    
    async def _get_user_data_inventory(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive inventory of user data"""
        # Implementation would collect all user data
        return {}
    
    async def _check_legal_holds(self, user_id: str) -> List[str]:
        """Check for legal holds preventing data deletion"""
        # Implementation would check for legal requirements
        return []
    
    async def _perform_data_erasure(
        self, 
        user_id: str, 
        data_inventory: Dict[str, Any], 
        legal_holds: List[str]
    ) -> Dict[str, Any]:
        """Perform actual data erasure"""
        # Implementation would delete user data
        return {"erased": True}
    
    async def _update_data_subject_request(self, request: DataSubjectRequest):
        """Update data subject request in database"""
        # Implementation would update database
        pass
    
    async def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all user data for export"""
        # Implementation would gather comprehensive user data
        return {}
    
    async def _get_user_consent_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's consent history"""
        # Implementation would retrieve consent history
        return []
    
    async def _get_user_processing_activities(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's data processing activities"""
        # Implementation would retrieve processing activities
        return []
    
    async def _get_applicable_retention_policies(self, user_id: str) -> List[Dict[str, Any]]:
        """Get retention policies applicable to user"""
        # Implementation would retrieve applicable policies
        return []
    
    async def _get_processing_lawful_basis(self, user_id: str) -> Dict[str, str]:
        """Get lawful basis for each type of processing"""
        # Implementation would retrieve lawful basis
        return {}
    
    async def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert data to CSV format"""
        # Implementation would convert to CSV
        return "csv_data"
    
    async def _convert_to_xml(self, data: Dict[str, Any]) -> str:
        """Convert data to XML format"""
        # Implementation would convert to XML
        return "xml_data"
    
    async def _get_active_retention_policies(self) -> List[DataRetentionPolicy]:
        """Get all active retention policies"""
        return self.default_retention_policies
    
    async def _find_expired_data(self, policy: DataRetentionPolicy) -> List[Dict[str, Any]]:
        """Find data that has expired according to policy"""
        # Implementation would query for expired data
        return []
    
    async def _secure_delete_data(self, data_records: List[Dict[str, Any]]) -> int:
        """Securely delete data records"""
        # Implementation would perform secure deletion
        return len(data_records)
    
    async def _anonymize_data(self, data_records: List[Dict[str, Any]]) -> int:
        """Anonymize data records"""
        # Implementation would anonymize data
        return len(data_records)
    
    async def _archive_data(self, data_records: List[Dict[str, Any]]) -> int:
        """Archive data records"""
        # Implementation would archive data
        return len(data_records)
    
    async def _store_privacy_impact_assessment(self, pia: PrivacyImpactAssessment):
        """Store PIA in database"""
        # Implementation would store in database
        pass
    
    def _assess_privacy_risk(
        self, 
        data_categories: List[DataCategory], 
        risk_assessment: Dict[str, Any]
    ) -> str:
        """Assess privacy risk level"""
        
        if DataCategory.SPECIAL_CATEGORY in data_categories:
            return "high"
        elif len(data_categories) > 3:
            return "medium"
        else:
            return "low"
    
    def _generate_mitigation_measures(
        self, 
        risk_level: str, 
        data_categories: List[DataCategory]
    ) -> List[str]:
        """Generate privacy risk mitigation measures"""
        
        measures = [
            "Implement data minimization principles",
            "Ensure explicit consent for processing",
            "Regular privacy training for staff",
            "Conduct periodic compliance audits"
        ]
        
        if risk_level == "high":
            measures.extend([
                "Implement additional encryption measures",
                "Conduct regular penetration testing",
                "Appoint Data Protection Officer",
                "Implement privacy by design principles"
            ])
        
        if DataCategory.SPECIAL_CATEGORY in data_categories:
            measures.extend([
                "Implement additional safeguards for special category data",
                "Ensure explicit consent for special category processing",
                "Regular review of special category data necessity"
            ])
        
        return measures
    
    def get_compliance_metrics(self) -> Dict[str, Any]:
        """Get GDPR compliance metrics"""
        
        return {
            'consent_records': self.metrics['consent_records'],
            'data_subject_requests': self.metrics['data_subject_requests'],
            'completed_requests': self.metrics['completed_requests'],
            'request_completion_rate': (
                self.metrics['completed_requests'] / self.metrics['data_subject_requests'] * 100
                if self.metrics['data_subject_requests'] > 0 else 0
            ),
            'data_deletions': self.metrics['data_deletions'],
            'consent_withdrawals': self.metrics['consent_withdrawals'],
            'privacy_assessments': self.metrics['privacy_assessments'],
            'retention_violations': self.metrics['retention_violations'],
            'compliance_health': self._calculate_compliance_health()
        }
    
    def _calculate_compliance_health(self) -> str:
        """Calculate overall compliance health score"""
        
        # Simple health calculation based on metrics
        total_requests = self.metrics['data_subject_requests']
        if total_requests == 0:
            return "excellent"
        
        completion_rate = self.metrics['completed_requests'] / total_requests
        
        if completion_rate >= 0.95:
            return "excellent"
        elif completion_rate >= 0.85:
            return "good"
        elif completion_rate >= 0.70:
            return "fair"
        else:
            return "needs_improvement"
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform GDPR compliance system health check"""
        
        return {
            'status': 'healthy',
            'database_connected': self.db is not None,
            'audit_logging_enabled': self.audit_logger is not None,
            'auto_deletion_enabled': self.auto_delete_expired_data,
            'verification_required': self.verification_required_for_requests,
            'retention_policies_count': len(self.default_retention_policies),
            'metrics': self.get_compliance_metrics()
        }


# Factory function
def create_gdpr_compliance_manager(
    database_manager: Optional[DatabaseManager] = None,
    audit_logger: Optional[AuditLogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> GDPRComplianceManager:
    """Factory function to create a configured GDPR Compliance Manager"""
    
    return GDPRComplianceManager(
        database_manager=database_manager,
        audit_logger=audit_logger,
        config=config or {}
    )


# Export all necessary components
__all__ = [
    'GDPRComplianceManager',
    'ConsentRecord',
    'DataSubjectRequest',
    'DataRetentionPolicy',
    'PrivacyImpactAssessment',
    'DataSubjectRight',
    'ConsentType',
    'ConsentStatus',
    'ProcessingLawfulBasis',
    'DataCategory',
    'create_gdpr_compliance_manager'
] 