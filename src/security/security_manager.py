"""
Universal RAG CMS Security Manager
Main orchestrator for enterprise-grade security and compliance

This module provides the central SecurityManager that coordinates:
- Role-Based Access Control (RBAC)
- Content Moderation and Filtering
- Audit Logging and Compliance
- API Key Management
- GDPR Compliance
- Rate Limiting and Security Monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import secrets

from .models import (
    SecurityContext, SecurityConfig, UserRole, Permission,
    AuditAction, SecurityViolation
)
from .encryption_manager import EncryptionManager
from .managers.rbac_manager import RBACManager
from .managers.audit_logger import AuditLogger
from .managers.api_key_manager import APIKeyManager
from .managers.content_moderator import ContentModerator, ContentModerationConfig
from .managers.gdpr_compliance_manager import GDPRComplianceManager
from .managers.rate_limiter import RateLimiter, RateLimitScope

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class SecurityMetrics:
    """Security system performance and threat metrics."""
    total_requests: int = 0
    blocked_requests: int = 0
    failed_authentications: int = 0
    content_violations: int = 0
    rate_limit_hits: int = 0
    gdpr_requests: int = 0
    api_key_rotations: int = 0
    audit_entries: int = 0
    
    def get_security_score(self) -> float:
        """Calculate overall security health score (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 1.0
            
        threat_ratio = (
            self.blocked_requests + 
            self.failed_authentications + 
            self.content_violations
        ) / self.total_requests
        
        # Lower threat ratio = higher security score
        return max(0.0, 1.0 - min(1.0, threat_ratio * 2))


class SecurityManager:
    """
    Main Security Manager orchestrating all security components.
    
    Provides enterprise-grade security for Universal RAG CMS including:
    - Centralized security policy enforcement
    - Multi-component security orchestration
    - Real-time threat detection and response
    - Compliance monitoring and reporting
    - Security metrics and analytics
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize the Security Manager with all components."""
        
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core security components
        self.encryption_manager = EncryptionManager()
        self.audit_logger = AuditLogger()
        self.rbac_manager = RBACManager(audit_logger=self.audit_logger)
        self.api_key_manager = APIKeyManager(audit_logger=self.audit_logger)
        
        # Initialize advanced security components
        content_config = ContentModerationConfig(
            enable_openai_moderation=True,
            enable_custom_rules=True,
            enable_content_sanitization=True,
            allow_gambling_content=True,
            gambling_age_verification_required=True
        )
        self.content_moderator = ContentModerator(
            openai_api_key=self.config.openai_api_key,
            config=content_config,
            audit_logger=self.audit_logger
        )
        
        self.gdpr_manager = GDPRComplianceManager(
            audit_logger=self.audit_logger,
            config=self.config.gdpr_config
        )
        
        self.rate_limiter = RateLimiter(
            audit_logger=self.audit_logger,
            config=self.config.rate_limit_config
        )
        
        # Security metrics tracking
        self.metrics = SecurityMetrics()
        
        # Security incident tracking
        self.recent_incidents = []
        self.security_alerts = []
        
        # Component health status
        self.component_status = {
            'encryption': True,
            'rbac': True,
            'audit': True,
            'api_keys': True,
            'content_moderation': True,
            'gdpr': True,
            'rate_limiting': True
        }
        
        self.logger.info("SecurityManager initialized with all components")
    
    async def initialize(self) -> bool:
        """Initialize all security components and validate system health."""
        
        try:
            # Initialize components in order of dependency
            init_tasks = [
                self.encryption_manager.initialize(),
                self.audit_logger.initialize(),
                self.rbac_manager.initialize(),
                self.api_key_manager.initialize(),
                self.content_moderator.initialize(),
                self.gdpr_manager.initialize(),
                self.rate_limiter.initialize()
            ]
            
            # Execute initialization in parallel where possible
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Check initialization results
            component_names = [
                'encryption', 'audit', 'rbac', 'api_keys', 
                'content_moderation', 'gdpr', 'rate_limiting'
            ]
            
            for i, result in enumerate(results):
                component_name = component_names[i]
                if isinstance(result, Exception):
                    self.component_status[component_name] = False
                    self.logger.error(f"Failed to initialize {component_name}: {result}")
                else:
                    self.component_status[component_name] = result
            
            # Check overall system health
            healthy_components = sum(1 for status in self.component_status.values() if status)
            total_components = len(self.component_status)
            
            if healthy_components == total_components:
                self.logger.info("All security components initialized successfully")
                return True
            else:
                self.logger.warning(
                    f"Security system partially initialized: {healthy_components}/{total_components} components healthy"
                )
                return healthy_components >= (total_components * 0.7)  # 70% threshold
        
        except Exception as e:
            self.logger.error(f"Security system initialization failed: {e}")
            return False
    
    async def authenticate_request(
        self, 
        request_data: Dict[str, Any],
        context: Optional[SecurityContext] = None
    ) -> tuple[bool, Optional[SecurityContext], List[str]]:
        """
        Authenticate and authorize a request through the security pipeline.
        
        Args:
            request_data: Request data including headers, user info, etc.
            context: Optional existing security context
            
        Returns:
            Tuple of (success, security_context, error_messages)
        """
        
        start_time = datetime.utcnow()
        errors = []
        
        try:
            self.metrics.total_requests += 1
            
            # Extract request information
            user_id = request_data.get('user_id')
            api_key = request_data.get('api_key')
            client_ip = request_data.get('client_ip', 'unknown')
            user_agent = request_data.get('user_agent', 'unknown')
            
            # 1. Rate limiting check
            rate_limit_key = f"{client_ip}:{user_id or 'anonymous'}"
            if not await self.rate_limiter.check_rate_limit(rate_limit_key):
                self.metrics.rate_limit_hits += 1
                await self._log_security_incident(
                    'rate_limit_exceeded',
                    {'client_ip': client_ip, 'user_id': user_id}
                )
                return False, None, ['Rate limit exceeded']
            
            # 2. API Key validation (if provided)
            if api_key:
                api_key_valid, api_key_info = await self.api_key_manager.validate_api_key(api_key)
                if not api_key_valid:
                    self.metrics.failed_authentications += 1
                    await self._log_security_incident(
                        'invalid_api_key',
                        {'client_ip': client_ip, 'api_key_hint': api_key[:8] + '...'}
                    )
                    return False, None, ['Invalid API key']
                
                # Create context from API key
                security_context = SecurityContext(
                    user_id=api_key_info.get('user_id'),
                    user_role=UserRole(api_key_info.get('role', 'API_USER')),
                    permissions=api_key_info.get('permissions', []),
                    session_id=self._generate_session_id(),
                    client_ip=client_ip,
                    user_agent=user_agent,
                    authenticated=True,
                    api_key_id=api_key_info.get('id')
                )
            
            # 3. User authentication (if user_id provided)
            elif user_id:
                # Get user role and permissions
                user_role, permissions = await self.rbac_manager.get_user_permissions(user_id)
                
                if not user_role:
                    self.metrics.failed_authentications += 1
                    return False, None, ['User not found or inactive']
                
                security_context = SecurityContext(
                    user_id=user_id,
                    user_role=user_role,
                    permissions=permissions,
                    session_id=self._generate_session_id(),
                    client_ip=client_ip,
                    user_agent=user_agent,
                    authenticated=True
                )
            
            # 4. Anonymous access
            else:
                security_context = SecurityContext(
                    user_role=UserRole.ANONYMOUS,
                    permissions=[Permission.READ_CONTENT],
                    session_id=self._generate_session_id(),
                    client_ip=client_ip,
                    user_agent=user_agent,
                    authenticated=False
                )
            
            # 5. Additional security checks
            additional_checks = await self._perform_additional_security_checks(
                security_context, request_data
            )
            
            if not additional_checks['passed']:
                errors.extend(additional_checks['errors'])
                return False, security_context, errors
            
            # 6. Log successful authentication
            await self.audit_logger.log_action(
                AuditAction.LOGIN_SUCCESS,
                user_id=security_context.user_id,
                details={
                    'client_ip': client_ip,
                    'user_agent': user_agent,
                    'session_id': security_context.session_id,
                    'authentication_method': 'api_key' if api_key else 'user_auth' if user_id else 'anonymous'
                }
            )
            
            return True, security_context, []
        
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            await self._log_security_incident('authentication_error', {'error': str(e)})
            return False, None, ['Authentication system error']
    
    async def authorize_action(
        self,
        security_context: SecurityContext,
        action: Permission,
        resource: Optional[str] = None,
        resource_data: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Authorize a specific action based on user permissions and context.
        
        Args:
            security_context: Current security context
            action: Permission being requested
            resource: Optional resource identifier
            resource_data: Optional resource-specific data
            
        Returns:
            Tuple of (authorized, error_messages)
        """
        
        try:
            # Check basic permission
            if not await self.rbac_manager.check_permission(
                security_context.user_role, action
            ):
                await self.audit_logger.log_action(
                    AuditAction.ACCESS_DENIED,
                    user_id=security_context.user_id,
                    details={
                        'requested_permission': action.value,
                        'user_role': security_context.user_role.value,
                        'resource': resource
                    }
                )
                return False, [f'Insufficient permissions for {action.value}']
            
            # Resource-specific authorization
            if resource:
                resource_authorized = await self._check_resource_authorization(
                    security_context, action, resource, resource_data
                )
                if not resource_authorized:
                    return False, [f'Access denied to resource: {resource}']
            
            # Log successful authorization
            await self.audit_logger.log_action(
                AuditAction.ACCESS_GRANTED,
                user_id=security_context.user_id,
                details={
                    'granted_permission': action.value,
                    'resource': resource
                }
            )
            
            return True, []
        
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return False, ['Authorization system error']
    
    async def moderate_content(
        self,
        content: str,
        content_type: str,
        security_context: SecurityContext
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Moderate content through security filters.
        
        Args:
            content: Content to moderate
            content_type: Type of content (e.g., 'query', 'response', 'comment')
            security_context: Current security context
            
        Returns:
            Tuple of (approved, moderation_result)
        """
        
        try:
            # Perform content moderation
            moderation_result = await self.content_moderator.moderate_content(
                content, content_type, security_context
            )
            
            # Check if content is approved
            approved = moderation_result.get('approved', False)
            
            # Log content moderation
            await self.audit_logger.log_action(
                AuditAction.CONTENT_MODERATED,
                user_id=security_context.user_id,
                details={
                    'content_type': content_type,
                    'approved': approved,
                    'moderation_score': moderation_result.get('score', 0),
                    'violations': moderation_result.get('violations', [])
                }
            )
            
            # Track violations
            if not approved:
                self.metrics.content_violations += 1
                await self._log_security_incident(
                    'content_violation',
                    {
                        'user_id': security_context.user_id,
                        'content_type': content_type,
                        'violations': moderation_result.get('violations', [])
                    }
                )
            
            return approved, moderation_result
        
        except Exception as e:
            self.logger.error(f"Content moderation error: {e}")
            return False, {'error': 'Content moderation system error'}
    
    async def handle_gdpr_request(
        self,
        request_type: str,
        user_id: str,
        security_context: SecurityContext,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle GDPR compliance requests.
        
        Args:
            request_type: Type of GDPR request ('access', 'delete', 'rectify', 'port')
            user_id: User ID for the request
            security_context: Current security context
            additional_data: Additional request data
            
        Returns:
            Dict containing request results
        """
        
        try:
            self.metrics.gdpr_requests += 1
            
            # Delegate to GDPR manager
            result = await self.gdpr_manager.handle_request(
                request_type, user_id, security_context, additional_data
            )
            
            # Log GDPR request
            await self.audit_logger.log_action(
                AuditAction.GDPR_REQUEST,
                user_id=security_context.user_id,
                details={
                    'request_type': request_type,
                    'target_user_id': user_id,
                    'result': result.get('status', 'unknown')
                }
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"GDPR request error: {e}")
            return {'status': 'error', 'message': 'GDPR system error'}
    
    async def rotate_api_keys(
        self,
        security_context: SecurityContext,
        key_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Rotate API keys for security maintenance.
        
        Args:
            security_context: Current security context
            key_ids: Optional specific key IDs to rotate
            
        Returns:
            Dict containing rotation results
        """
        
        try:
            # Check authorization for key rotation
            authorized, auth_errors = await self.authorize_action(
                security_context, Permission.MANAGE_API_KEYS
            )
            
            if not authorized:
                return {'status': 'error', 'message': 'Unauthorized for key rotation'}
            
            # Perform key rotation
            rotation_result = await self.api_key_manager.rotate_keys(key_ids)
            
            self.metrics.api_key_rotations += len(rotation_result.get('rotated_keys', []))
            
            # Log key rotation
            await self.audit_logger.log_action(
                AuditAction.API_KEY_ROTATED,
                user_id=security_context.user_id,
                details={
                    'rotated_count': len(rotation_result.get('rotated_keys', [])),
                    'key_ids': key_ids or 'all_eligible'
                }
            )
            
            return rotation_result
        
        except Exception as e:
            self.logger.error(f"API key rotation error: {e}")
            return {'status': 'error', 'message': 'Key rotation system error'}
    
    async def get_security_metrics(
        self,
        security_context: SecurityContext,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive security metrics and analytics.
        
        Args:
            security_context: Current security context
            time_range: Optional time range for metrics
            
        Returns:
            Dict containing security metrics
        """
        
        try:
            # Check authorization for metrics access
            authorized, auth_errors = await self.authorize_action(
                security_context, Permission.VIEW_SECURITY_METRICS
            )
            
            if not authorized:
                return {'error': 'Unauthorized for security metrics'}
            
            # Gather metrics from all components
            metrics_data = {
                'overall': {
                    'security_score': self.metrics.get_security_score(),
                    'total_requests': self.metrics.total_requests,
                    'blocked_requests': self.metrics.blocked_requests,
                    'failed_authentications': self.metrics.failed_authentications,
                    'content_violations': self.metrics.content_violations,
                    'rate_limit_hits': self.metrics.rate_limit_hits,
                    'gdpr_requests': self.metrics.gdpr_requests,
                    'api_key_rotations': self.metrics.api_key_rotations
                },
                'component_health': self.component_status,
                'recent_incidents': self.recent_incidents[-10:],  # Last 10 incidents
                'active_alerts': self.security_alerts,
                'audit_summary': await self.audit_logger.get_audit_summary(time_range),
                'rate_limiting': await self.rate_limiter.get_metrics(),
                'content_moderation': await self.content_moderator.get_metrics(),
                'api_keys': await self.api_key_manager.get_key_metrics()
            }
            
            return metrics_data
        
        except Exception as e:
            self.logger.error(f"Security metrics error: {e}")
            return {'error': 'Security metrics system error'}
    
    async def _perform_additional_security_checks(
        self,
        security_context: SecurityContext,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform additional security checks beyond basic authentication."""
        
        checks = {'passed': True, 'errors': []}
        
        try:
            # IP-based security checks
            client_ip = security_context.client_ip
            if await self._is_ip_blocked(client_ip):
                checks['passed'] = False
                checks['errors'].append('IP address blocked')
            
            # User agent analysis
            user_agent = security_context.user_agent
            if await self._is_suspicious_user_agent(user_agent):
                checks['passed'] = False
                checks['errors'].append('Suspicious user agent detected')
            
            # Request pattern analysis
            if await self._detect_suspicious_patterns(security_context, request_data):
                checks['passed'] = False
                checks['errors'].append('Suspicious request pattern detected')
            
            return checks
        
        except Exception as e:
            self.logger.error(f"Additional security checks error: {e}")
            checks['passed'] = False
            checks['errors'].append('Security check system error')
            return checks
    
    async def _check_resource_authorization(
        self,
        security_context: SecurityContext,
        action: Permission,
        resource: str,
        resource_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Check resource-specific authorization rules."""
        
        try:
            # Delegate to RBAC manager for resource-specific checks
            return await self.rbac_manager.check_resource_access(
                security_context.user_role,
                security_context.user_id,
                action,
                resource,
                resource_data
            )
        
        except Exception as e:
            self.logger.error(f"Resource authorization error: {e}")
            return False
    
    async def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is in blocked list."""
        # In a real implementation, this would check against a blacklist database
        return False
    
    async def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Analyze user agent for suspicious patterns."""
        suspicious_patterns = ['bot', 'crawler', 'scanner', 'hack']
        return any(pattern in user_agent.lower() for pattern in suspicious_patterns)
    
    async def _detect_suspicious_patterns(
        self,
        security_context: SecurityContext,
        request_data: Dict[str, Any]
    ) -> bool:
        """Detect suspicious request patterns."""
        # Simple pattern detection - could be enhanced with ML
        return False
    
    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return secrets.token_urlsafe(32)
    
    async def _log_security_incident(
        self,
        incident_type: str,
        details: Dict[str, Any]
    ):
        """Log a security incident for monitoring and analysis."""
        
        incident = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': incident_type,
            'details': details
        }
        
        self.recent_incidents.append(incident)
        
        # Keep only recent incidents (last 100)
        if len(self.recent_incidents) > 100:
            self.recent_incidents = self.recent_incidents[-100:]
        
        # Log to audit system
        await self.audit_logger.log_action(
            AuditAction.SECURITY_INCIDENT,
            details={
                'incident_type': incident_type,
                'incident_details': details
            }
        )
        
        self.logger.warning(f"Security incident: {incident_type}", extra=details)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive security system health check."""
        
        health_data = {
            'overall_status': 'healthy',
            'components': {},
            'metrics': {
                'security_score': self.metrics.get_security_score(),
                'total_requests': self.metrics.total_requests,
                'error_rate': 0.0
            },
            'alerts': len(self.security_alerts),
            'last_check': datetime.utcnow().isoformat()
        }
        
        # Check each component
        for component, status in self.component_status.items():
            health_data['components'][component] = 'healthy' if status else 'unhealthy'
        
        # Calculate error rate
        if self.metrics.total_requests > 0:
            errors = (
                self.metrics.blocked_requests +
                self.metrics.failed_authentications +
                self.metrics.content_violations
            )
            health_data['metrics']['error_rate'] = errors / self.metrics.total_requests
        
        # Determine overall status
        unhealthy_components = sum(1 for status in self.component_status.values() if not status)
        if unhealthy_components > 0:
            health_data['overall_status'] = 'degraded' if unhealthy_components <= 2 else 'unhealthy'
        
        if health_data['metrics']['error_rate'] > 0.1:  # 10% error threshold
            health_data['overall_status'] = 'degraded'
        
        return health_data
    
    async def shutdown(self):
        """Gracefully shutdown all security components."""
        
        try:
            self.logger.info("Shutting down SecurityManager...")
            
            # Shutdown components
            shutdown_tasks = [
                self.rate_limiter.shutdown(),
                self.content_moderator.shutdown(),
                self.gdpr_manager.shutdown(),
                self.api_key_manager.shutdown(),
                self.audit_logger.shutdown(),
                self.rbac_manager.shutdown()
            ]
            
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            self.logger.info("SecurityManager shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during SecurityManager shutdown: {e}") 