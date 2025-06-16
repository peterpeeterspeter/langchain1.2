"""
Secure RAG Chain
Enterprise-grade security wrapper for Universal RAG CMS

Extends the IntegratedRAGChain with comprehensive security features:
- Request authentication and authorization
- Input validation and sanitization
- Content moderation and filtering
- Audit logging for all operations
- Rate limiting and abuse prevention
- GDPR compliance tracking
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import bleach

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

from .security_manager import SecurityManager
from .models import SecurityContext, UserRole, Permission, AuditAction
from ..chains.integrated_rag_chain import IntegratedRAGChain
from ..chains.enhanced_confidence_scoring_system import EnhancedRAGResponse

logger = logging.getLogger(__name__)


class SecurityCallbackHandler(BaseCallbackHandler):
    """Callback handler for security audit logging."""
    
    def __init__(self, security_manager: SecurityManager, security_context: SecurityContext):
        super().__init__()
        self.security_manager = security_manager
        self.security_context = security_context
        self.start_time = None
        self.query_hash = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Log chain start."""
        self.start_time = datetime.utcnow()
        
        # Create query hash for privacy
        query = inputs.get('query', '')
        self.query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        asyncio.create_task(
            self.security_manager.audit_logger.log_action(
                action=AuditAction.QUERY_STARTED,
                user_id=self.security_context.user_id,
                session_id=self.security_context.session_id,
                client_ip=self.security_context.client_ip,
                details={
                    'query_hash': self.query_hash,
                    'input_keys': list(inputs.keys())
                },
                result='success',
                severity='low'
            )
        )
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Log chain completion."""
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            
            asyncio.create_task(
                self.security_manager.audit_logger.log_action(
                    action=AuditAction.QUERY_COMPLETED,
                    user_id=self.security_context.user_id,
                    session_id=self.security_context.session_id,
                    client_ip=self.security_context.client_ip,
                    details={
                        'query_hash': self.query_hash,
                        'duration_seconds': duration,
                        'output_keys': list(outputs.keys())
                    },
                    result='success',
                    severity='low'
                )
            )
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Log chain errors."""
        asyncio.create_task(
            self.security_manager.audit_logger.log_action(
                action=AuditAction.QUERY_FAILED,
                user_id=self.security_context.user_id,
                session_id=self.security_context.session_id,
                client_ip=self.security_context.client_ip,
                details={
                    'query_hash': self.query_hash,
                    'error_type': type(error).__name__,
                    'error_message': str(error)
                },
                result='failure',
                severity='medium'
            )
        )


class SecureRAGChain:
    """
    Secure wrapper for IntegratedRAGChain with enterprise security features.
    
    Provides comprehensive security including:
    - Authentication and authorization
    - Input validation and sanitization
    - Content moderation
    - Audit logging
    - Rate limiting
    - GDPR compliance
    """
    
    def __init__(self, base_chain: IntegratedRAGChain, security_manager: SecurityManager):
        """
        Initialize Secure RAG Chain.
        
        Args:
            base_chain: The underlying IntegratedRAGChain
            security_manager: Security manager instance
        """
        
        self.logger = logging.getLogger(__name__)
        self.base_chain = base_chain
        self.security_manager = security_manager
        
        # Security configuration
        self.max_query_length = 10000
        self.allowed_html_tags = ['b', 'i', 'em', 'strong', 'u']
        self.blocked_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',                # JavaScript URLs
            r'on\w+\s*=',                 # Event handlers
            r'eval\s*\(',                 # eval() calls
            r'document\.',                # DOM access
            r'window\.',                  # Window object access
        ]
        
        # Performance metrics
        self.metrics = {
            'requests_processed': 0,
            'requests_blocked': 0,
            'validation_failures': 0,
            'content_moderation_flags': 0,
            'rate_limit_hits': 0,
            'security_violations': 0
        }
    
    async def invoke(
        self,
        query: str,
        security_context: SecurityContext,
        additional_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[EnhancedRAGResponse, Dict[str, Any]]:
        """
        Secure invoke method with full security pipeline.
        
        Args:
            query: User query
            security_context: Security context for the request
            additional_context: Optional additional context
            **kwargs: Additional arguments
            
        Returns:
            Enhanced RAG response or error response
        """
        
        try:
            self.metrics['requests_processed'] += 1
            
            # 1. Authentication & Authorization
            auth_result = await self._authenticate_request(security_context)
            if not auth_result['success']:
                self.metrics['requests_blocked'] += 1
                return self._create_error_response(auth_result['message'], 'authentication_failed')
            
            # 2. Rate Limiting
            rate_limit_result = await self._check_rate_limit(security_context)
            if not rate_limit_result['allowed']:
                self.metrics['rate_limit_hits'] += 1
                return self._create_error_response('Rate limit exceeded', 'rate_limit_exceeded')
            
            # 3. Input Validation & Sanitization
            validation_result = await self._validate_and_sanitize_input(query, security_context)
            if not validation_result['valid']:
                self.metrics['validation_failures'] += 1
                return self._create_error_response(validation_result['message'], 'validation_failed')
            
            sanitized_query = validation_result['sanitized_query']
            
            # 4. Content Moderation (Pre-processing)
            moderation_result = await self._moderate_content(sanitized_query, security_context)
            if not moderation_result['approved']:
                self.metrics['content_moderation_flags'] += 1
                return self._create_error_response(moderation_result['message'], 'content_moderated')
            
            # 5. Execute Base Chain with Security Callback
            security_callback = SecurityCallbackHandler(self.security_manager, security_context)
            
            response = await self.base_chain.ainvoke(
                {
                    'query': sanitized_query,
                    'additional_context': additional_context
                },
                config={'callbacks': [security_callback]},
                **kwargs
            )
            
            # 6. Post-processing Security Checks
            if isinstance(response, EnhancedRAGResponse):
                # Content moderation on response
                response_moderation = await self._moderate_content(response.answer, security_context)
                if not response_moderation['approved']:
                    self.metrics['content_moderation_flags'] += 1
                    return self._create_error_response('Response filtered by content moderation', 'response_moderated')
                
                # Add security metadata
                response.metadata = response.metadata or {}
                response.metadata.update({
                    'security_context': security_context.user_id,
                    'request_id': security_context.session_id,
                    'security_level': 'enterprise',
                    'content_moderated': True,
                    'audit_logged': True
                })
            
            # 7. GDPR Compliance Tracking
            await self._track_gdpr_processing(security_context, sanitized_query)
            
            # 8. Success Audit Log
            await self.security_manager.audit_logger.log_action(
                action=AuditAction.QUERY_SUCCESS,
                user_id=security_context.user_id,
                session_id=security_context.session_id,
                client_ip=security_context.client_ip,
                details={
                    'query_length': len(sanitized_query),
                    'response_type': type(response).__name__,
                    'security_checks_passed': True
                },
                result='success',
                severity='low'
            )
            
            return response
        
        except Exception as e:
            self.logger.error(f"Secure RAG Chain error: {e}")
            self.metrics['security_violations'] += 1
            
            # Log security incident
            await self.security_manager.audit_logger.log_security_event(
                event_type='rag_chain_error',
                security_context=security_context,
                details={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                severity='high'
            )
            
            return self._create_error_response('Internal security error', 'internal_error')
    
    async def _authenticate_request(self, security_context: SecurityContext) -> Dict[str, Any]:
        """Authenticate and authorize the request."""
        
        try:
            # Check if user has required permissions
            required_permissions = [Permission.READ_CONTENT, Permission.API_ACCESS]
            
            has_permissions = await self.security_manager.rbac_manager.check_permissions(
                user_id=security_context.user_id,
                permissions=required_permissions,
                resource="rag_query"
            )
            
            if not has_permissions:
                await self.security_manager.audit_logger.log_action(
                    action=AuditAction.ACCESS_DENIED,
                    user_id=security_context.user_id,
                    session_id=security_context.session_id,
                    client_ip=security_context.client_ip,
                    details={'required_permissions': [p.value for p in required_permissions]},
                    result='failure',
                    severity='medium'
                )
                
                return {'success': False, 'message': 'Insufficient permissions'}
            
            return {'success': True, 'message': 'Authentication successful'}
        
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return {'success': False, 'message': 'Authentication failed'}
    
    async def _check_rate_limit(self, security_context: SecurityContext) -> Dict[str, Any]:
        """Check rate limiting for the request."""
        
        try:
            # Get user's rate limit from API key or role
            rate_limit_key = f"rag_query:{security_context.user_id}:{security_context.client_ip}"
            
            allowed = await self.security_manager.rate_limiter.check_rate_limit(
                key=rate_limit_key,
                user_id=security_context.user_id
            )
            
            if not allowed:
                await self.security_manager.audit_logger.log_action(
                    action=AuditAction.RATE_LIMIT_EXCEEDED,
                    user_id=security_context.user_id,
                    session_id=security_context.session_id,
                    client_ip=security_context.client_ip,
                    details={'rate_limit_key': rate_limit_key},
                    result='warning',
                    severity='medium'
                )
            
            return {'allowed': allowed, 'key': rate_limit_key}
        
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return {'allowed': False, 'error': str(e)}
    
    async def _validate_and_sanitize_input(
        self, 
        query: str, 
        security_context: SecurityContext
    ) -> Dict[str, Any]:
        """Validate and sanitize user input."""
        
        try:
            # Length validation
            if len(query) > self.max_query_length:
                return {
                    'valid': False,
                    'message': f'Query too long (max {self.max_query_length} characters)',
                    'sanitized_query': None
                }
            
            # Check for malicious patterns
            import re
            for pattern in self.blocked_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    await self.security_manager.audit_logger.log_security_event(
                        event_type='malicious_input_detected',
                        security_context=security_context,
                        details={
                            'pattern_matched': pattern,
                            'query_hash': hashlib.sha256(query.encode()).hexdigest()[:16]
                        },
                        severity='high'
                    )
                    
                    return {
                        'valid': False,
                        'message': 'Query contains potentially harmful content',
                        'sanitized_query': None
                    }
            
            # HTML sanitization
            sanitized_query = bleach.clean(
                query,
                tags=self.allowed_html_tags,
                strip=True
            )
            
            # Additional sanitization
            sanitized_query = sanitized_query.strip()
            
            return {
                'valid': True,
                'message': 'Input validation successful',
                'sanitized_query': sanitized_query
            }
        
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return {
                'valid': False,
                'message': 'Input validation failed',
                'sanitized_query': None
            }
    
    async def _moderate_content(
        self, 
        content: str, 
        security_context: SecurityContext
    ) -> Dict[str, Any]:
        """Moderate content using the content moderator."""
        
        try:
            moderation_result = await self.security_manager.content_moderator.moderate_text(
                content=content,
                user_id=security_context.user_id,
                context_type='rag_query'
            )
            
            if not moderation_result['approved']:
                await self.security_manager.audit_logger.log_action(
                    action=AuditAction.CONTENT_MODERATED,
                    user_id=security_context.user_id,
                    session_id=security_context.session_id,
                    client_ip=security_context.client_ip,
                    details={
                        'moderation_flags': moderation_result.get('flags', []),
                        'confidence_score': moderation_result.get('confidence', 0)
                    },
                    result='warning',
                    severity='medium'
                )
            
            return moderation_result
        
        except Exception as e:
            self.logger.error(f"Content moderation error: {e}")
            # Fail open with logging
            return {'approved': True, 'message': 'Moderation check bypassed due to error'}
    
    async def _track_gdpr_processing(self, security_context: SecurityContext, query: str):
        """Track GDPR compliance for data processing."""
        
        try:
            await self.security_manager.gdpr_manager.log_data_processing(
                user_id=security_context.user_id,
                processing_type='rag_query',
                data_categories=['query_text', 'response_generation'],
                legal_basis='legitimate_interest',
                purpose='ai_assistance',
                retention_period_days=90,
                details={
                    'session_id': security_context.session_id,
                    'query_length': len(query),
                    'processing_timestamp': datetime.utcnow().isoformat()
                }
            )
        
        except Exception as e:
            self.logger.error(f"GDPR tracking error: {e}")
    
    def _create_error_response(self, message: str, error_type: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        
        return {
            'success': False,
            'error': {
                'type': error_type,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            },
            'answer': f"I'm sorry, I cannot process your request: {message}",
            'sources': [],
            'metadata': {
                'security_filtered': True,
                'error_type': error_type
            }
        }
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring."""
        
        return {
            'chain_metrics': self.metrics,
            'security_manager_metrics': await self.security_manager.get_security_health(),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    async def validate_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration and readiness."""
        
        try:
            validation_results = {
                'security_manager': await self.security_manager.health_check(),
                'base_chain': hasattr(self.base_chain, 'ainvoke'),
                'content_moderation': self.security_manager.content_moderator is not None,
                'rate_limiting': self.security_manager.rate_limiter is not None,
                'audit_logging': self.security_manager.audit_logger is not None,
                'rbac': self.security_manager.rbac_manager is not None
            }
            
            all_valid = all(validation_results.values())
            
            return {
                'valid': all_valid,
                'components': validation_results,
                'recommendations': self._get_security_recommendations(validation_results)
            }
        
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'components': {},
                'recommendations': ['Fix security configuration errors']
            }
    
    def _get_security_recommendations(self, validation_results: Dict[str, bool]) -> List[str]:
        """Get security configuration recommendations."""
        
        recommendations = []
        
        if not validation_results.get('content_moderation'):
            recommendations.append('Enable content moderation for production use')
        
        if not validation_results.get('rate_limiting'):
            recommendations.append('Configure rate limiting to prevent abuse')
        
        if not validation_results.get('audit_logging'):
            recommendations.append('Enable audit logging for compliance')
        
        if not validation_results.get('rbac'):
            recommendations.append('Configure RBAC for access control')
        
        if not recommendations:
            recommendations.append('Security configuration is optimal')
        
        return recommendations 