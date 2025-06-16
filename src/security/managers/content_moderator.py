"""
Content Moderator for Universal RAG CMS Security System

This module provides comprehensive content moderation using OpenAI's moderation API
and custom filtering rules for detecting harmful content, inappropriate language,
and compliance violations.

Features:
- OpenAI Moderation API integration
- Custom content filtering rules
- Severity assessment and scoring
- Content categorization and flagging
- Integration with audit logging
- Real-time content analysis
- Batch moderation for efficiency
- Content sanitization and filtering
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from collections import defaultdict

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

from ..models import AuditAction, SecurityViolation
from .audit_logger import AuditLogger

# Configure logging
logger = logging.getLogger(__name__)


class ContentSeverity(Enum):
    """Content moderation severity levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentCategory(Enum):
    """Content moderation categories"""
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    COPYRIGHT = "copyright"
    INAPPROPRIATE_LANGUAGE = "inappropriate_language"
    GAMBLING_VIOLATION = "gambling_violation"
    SAFE = "safe"


@dataclass
class ModerationResult:
    """Result of content moderation analysis"""
    is_flagged: bool
    severity: ContentSeverity
    categories: List[ContentCategory]
    confidence_score: float
    explanation: str
    suggestions: List[str] = field(default_factory=list)
    
    # OpenAI moderation results
    openai_flagged: bool = False
    openai_categories: Dict[str, bool] = field(default_factory=dict)
    openai_scores: Dict[str, float] = field(default_factory=dict)
    
    # Custom rule results
    custom_violations: List[str] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    
    # Processing metadata
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContentModerationConfig:
    """Configuration for content moderation"""
    enable_openai_moderation: bool = True
    enable_custom_rules: bool = True
    enable_content_sanitization: bool = True
    
    # Thresholds
    auto_block_threshold: float = 0.8
    review_threshold: float = 0.5
    warning_threshold: float = 0.3
    
    # OpenAI settings
    openai_model: str = "text-moderation-latest"
    openai_timeout: int = 10
    
    # Rate limiting
    max_requests_per_minute: int = 100
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Custom rules settings
    blocked_words_threshold: int = 2
    severity_multiplier: float = 1.5
    
    # Gambling-specific settings (for casino content)
    allow_gambling_content: bool = True
    gambling_age_verification_required: bool = True
    responsible_gambling_notices: bool = True


class ContentModerator:
    """
    Comprehensive content moderation system using OpenAI API and custom rules.
    
    Provides real-time content analysis, filtering, and sanitization to ensure
    content compliance with platform policies and legal requirements.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[ContentModerationConfig] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        """Initialize the Content Moderator"""
        
        self.config = config or ContentModerationConfig()
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE and openai_api_key and self.config.enable_openai_moderation:
            try:
                self.openai_client = AsyncOpenAI(api_key=openai_api_key)
                self.logger.info("OpenAI moderation client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Content filtering rules
        self.content_rules = self._initialize_content_rules()
        
        # Rate limiting tracking
        self.request_history = defaultdict(list)
        
        # Moderation cache
        self.moderation_cache = {}
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'flagged_content': 0,
            'cache_hits': 0,
            'openai_requests': 0,
            'custom_rule_violations': 0,
            'average_processing_time': 0.0
        }
        
        self.logger.info("ContentModerator initialized")
    
    def _initialize_content_rules(self) -> Dict[str, Any]:
        """Initialize custom content filtering rules"""
        
        return {
            'blocked_words': {
                'hate_speech': [
                    # Hate speech patterns (sanitized examples)
                    'hate', 'racist', 'bigot', 'extremist'
                ],
                'harassment': [
                    'bully', 'harass', 'stalk', 'threaten'
                ],
                'inappropriate': [
                    'vulgar', 'profanity', 'obscene', 'offensive'
                ],
                'spam': [
                    'spam', 'scam', 'fraud', 'fake'
                ]
            },
            
            'pattern_rules': {
                'personal_info': [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b'  # Phone number
                ],
                'urls': [
                    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                ],
                'gambling_violations': [
                    r'\b(guaranteed\s+win|sure\s+bet|risk\s+free)\b',
                    r'\b(addiction|problem\s+gambling)\b'
                ]
            },
            
            'severity_indicators': {
                'critical': ['illegal', 'criminal', 'terrorist', 'violence'],
                'high': ['threat', 'harm', 'dangerous', 'explicit'],
                'medium': ['inappropriate', 'offensive', 'violation'],
                'low': ['questionable', 'borderline', 'concern']
            },
            
            'gambling_compliance': {
                'required_disclaimers': [
                    'gambling can be addictive',
                    'play responsibly',
                    'age verification required'
                ],
                'prohibited_claims': [
                    'guaranteed win',
                    'sure thing',
                    'risk free',
                    'easy money'
                ]
            }
        }
    
    async def moderate_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> ModerationResult:
        """
        Perform comprehensive content moderation analysis.
        
        Args:
            content: Text content to moderate
            context: Additional context for moderation
            user_id: User ID for audit logging
            
        Returns:
            ModerationResult with analysis and recommendations
        """
        
        start_time = time.time()
        
        try:
            # Check rate limiting
            if not await self._check_rate_limit():
                raise SecurityViolation("Rate limit exceeded for content moderation")
            
            # Check cache first
            cache_key = self._generate_cache_key(content)
            if self.config.enable_caching and cache_key in self.moderation_cache:
                cached_result = self.moderation_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.metrics['cache_hits'] += 1
                    return cached_result
            
            # Initialize result
            result = ModerationResult(
                is_flagged=False,
                severity=ContentSeverity.SAFE,
                categories=[ContentCategory.SAFE],
                confidence_score=0.0,
                explanation="Content appears safe"
            )
            
            # Parallel moderation tasks
            moderation_tasks = []
            
            # OpenAI moderation
            if self.openai_client and self.config.enable_openai_moderation:
                moderation_tasks.append(self._openai_moderation(content))
            
            # Custom rules analysis
            if self.config.enable_custom_rules:
                moderation_tasks.append(self._custom_rules_analysis(content, context))
            
            # Execute moderation tasks
            if moderation_tasks:
                moderation_results = await asyncio.gather(*moderation_tasks, return_exceptions=True)
                
                # Process OpenAI results
                if len(moderation_results) > 0 and not isinstance(moderation_results[0], Exception):
                    openai_result = moderation_results[0]
                    result.openai_flagged = openai_result.get('flagged', False)
                    result.openai_categories = openai_result.get('categories', {})
                    result.openai_scores = openai_result.get('category_scores', {})
                
                # Process custom rules results
                if len(moderation_results) > 1 and not isinstance(moderation_results[1], Exception):
                    custom_result = moderation_results[1]
                    result.custom_violations = custom_result.get('violations', [])
            
            # Aggregate results and determine final verdict
            result = await self._aggregate_moderation_results(result, content)
            
            # Content sanitization if needed
            if self.config.enable_content_sanitization and result.is_flagged:
                result.sanitized_content = await self._sanitize_content(content, result)
            
            # Generate suggestions
            result.suggestions = self._generate_moderation_suggestions(result)
            
            # Update metrics
            self.metrics['total_requests'] += 1
            if result.is_flagged:
                self.metrics['flagged_content'] += 1
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            self.metrics['average_processing_time'] = (
                (self.metrics['average_processing_time'] * (self.metrics['total_requests'] - 1) + 
                 result.processing_time_ms) / self.metrics['total_requests']
            )
            
            # Cache result
            if self.config.enable_caching:
                self.moderation_cache[cache_key] = result
            
            # Audit logging
            if self.audit_logger and result.is_flagged:
                await self._log_moderation_event(result, content, user_id, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content moderation failed: {e}")
            
            # Return safe fallback
            return ModerationResult(
                is_flagged=False,
                severity=ContentSeverity.SAFE,
                categories=[ContentCategory.SAFE],
                confidence_score=0.0,
                explanation=f"Moderation failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _openai_moderation(self, content: str) -> Dict[str, Any]:
        """Perform OpenAI content moderation"""
        
        try:
            response = await self.openai_client.moderations.create(
                input=content,
                model=self.config.openai_model
            )
            
            self.metrics['openai_requests'] += 1
            
            result = response.results[0]
            return {
                'flagged': result.flagged,
                'categories': dict(result.categories),
                'category_scores': dict(result.category_scores)
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI moderation failed: {e}")
            return {'flagged': False, 'categories': {}, 'category_scores': {}}
    
    async def _custom_rules_analysis(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze content using custom rules"""
        
        violations = []
        content_lower = content.lower()
        
        # Check blocked words
        for category, words in self.content_rules['blocked_words'].items():
            word_count = sum(1 for word in words if word in content_lower)
            if word_count >= self.config.blocked_words_threshold:
                violations.append(f"{category}_violation")
        
        # Check pattern rules
        for rule_type, patterns in self.content_rules['pattern_rules'].items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(f"{rule_type}_detected")
        
        # Gambling-specific checks
        if context and context.get('content_type') == 'gambling':
            violations.extend(await self._check_gambling_compliance(content))
        
        self.metrics['custom_rule_violations'] += len(violations)
        
        return {'violations': violations}
    
    async def _check_gambling_compliance(self, content: str) -> List[str]:
        """Check gambling-specific compliance rules"""
        
        violations = []
        content_lower = content.lower()
        
        # Check for prohibited claims
        for claim in self.content_rules['gambling_compliance']['prohibited_claims']:
            if claim in content_lower:
                violations.append(f"prohibited_gambling_claim: {claim}")
        
        # Check for required disclaimers in long content
        if len(content) > 500:  # Long content should have disclaimers
            has_disclaimer = any(
                disclaimer in content_lower 
                for disclaimer in self.content_rules['gambling_compliance']['required_disclaimers']
            )
            if not has_disclaimer:
                violations.append("missing_responsible_gambling_disclaimer")
        
        return violations
    
    async def _aggregate_moderation_results(
        self, 
        result: ModerationResult, 
        content: str
    ) -> ModerationResult:
        """Aggregate OpenAI and custom rule results into final verdict"""
        
        # Start with safe assumption
        max_confidence = 0.0
        detected_categories = []
        
        # Process OpenAI results
        if result.openai_flagged:
            max_confidence = max(max_confidence, 0.8)  # High confidence for OpenAI flags
            
            # Map OpenAI categories to our categories
            openai_mapping = {
                'hate': ContentCategory.HATE_SPEECH,
                'harassment': ContentCategory.HARASSMENT,
                'violence': ContentCategory.VIOLENCE,
                'sexual': ContentCategory.SEXUAL_CONTENT,
                'self-harm': ContentCategory.SELF_HARM
            }
            
            for openai_cat, flagged in result.openai_categories.items():
                if flagged and openai_cat in openai_mapping:
                    detected_categories.append(openai_mapping[openai_cat])
        
        # Process custom rule violations
        if result.custom_violations:
            max_confidence = max(max_confidence, 0.6)  # Medium confidence for custom rules
            
            # Map violations to categories
            violation_mapping = {
                'hate_speech_violation': ContentCategory.HATE_SPEECH,
                'harassment_violation': ContentCategory.HARASSMENT,
                'spam_violation': ContentCategory.SPAM,
                'personal_info_detected': ContentCategory.PRIVACY_VIOLATION,
                'gambling_violation': ContentCategory.GAMBLING_VIOLATION
            }
            
            for violation in result.custom_violations:
                for pattern, category in violation_mapping.items():
                    if pattern in violation:
                        detected_categories.append(category)
        
        # Determine severity
        severity = ContentSeverity.SAFE
        if max_confidence >= self.config.auto_block_threshold:
            severity = ContentSeverity.CRITICAL
        elif max_confidence >= self.config.review_threshold:
            severity = ContentSeverity.HIGH
        elif max_confidence >= self.config.warning_threshold:
            severity = ContentSeverity.MEDIUM
        elif max_confidence > 0:
            severity = ContentSeverity.LOW
        
        # Determine if flagged
        is_flagged = max_confidence >= self.config.warning_threshold
        
        # Generate explanation
        explanation = self._generate_moderation_explanation(
            is_flagged, detected_categories, result.openai_flagged, result.custom_violations
        )
        
        # Update result
        result.is_flagged = is_flagged
        result.severity = severity
        result.categories = detected_categories if detected_categories else [ContentCategory.SAFE]
        result.confidence_score = max_confidence
        result.explanation = explanation
        
        return result
    
    def _generate_moderation_explanation(
        self,
        is_flagged: bool,
        categories: List[ContentCategory],
        openai_flagged: bool,
        custom_violations: List[str]
    ) -> str:
        """Generate human-readable explanation of moderation results"""
        
        if not is_flagged:
            return "Content appears to be safe and compliant with platform policies."
        
        explanation_parts = ["Content flagged for the following reasons:"]
        
        if openai_flagged:
            explanation_parts.append("- Detected by OpenAI moderation system")
        
        if custom_violations:
            explanation_parts.append(f"- Custom rule violations: {', '.join(custom_violations)}")
        
        if categories and categories != [ContentCategory.SAFE]:
            category_names = [cat.value.replace('_', ' ').title() for cat in categories]
            explanation_parts.append(f"- Categories: {', '.join(category_names)}")
        
        return " ".join(explanation_parts)
    
    async def _sanitize_content(self, content: str, result: ModerationResult) -> str:
        """Sanitize flagged content by removing or replacing problematic parts"""
        
        sanitized = content
        
        # Remove personal information patterns
        for pattern in self.content_rules['pattern_rules']['personal_info']:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        # Replace blocked words with asterisks
        for category, words in self.content_rules['blocked_words'].items():
            for word in words:
                if word in sanitized.lower():
                    replacement = '*' * len(word)
                    sanitized = re.sub(rf'\b{re.escape(word)}\b', replacement, sanitized, flags=re.IGNORECASE)
        
        # Add responsible gambling disclaimers if needed
        if ContentCategory.GAMBLING_VIOLATION in result.categories:
            disclaimer = "\n\n[Disclaimer: Gambling can be addictive. Please play responsibly.]"
            sanitized += disclaimer
        
        return sanitized
    
    def _generate_moderation_suggestions(self, result: ModerationResult) -> List[str]:
        """Generate actionable suggestions based on moderation results"""
        
        suggestions = []
        
        if not result.is_flagged:
            suggestions.append("Content is approved for publication")
            return suggestions
        
        if result.severity == ContentSeverity.CRITICAL:
            suggestions.append("Content should be blocked immediately")
            suggestions.append("Review content manually before any publication")
        elif result.severity == ContentSeverity.HIGH:
            suggestions.append("Content requires manual review before publication")
            suggestions.append("Consider content modification or removal")
        elif result.severity == ContentSeverity.MEDIUM:
            suggestions.append("Content should be reviewed and potentially modified")
        else:
            suggestions.append("Minor content adjustments may be beneficial")
        
        # Category-specific suggestions
        if ContentCategory.PRIVACY_VIOLATION in result.categories:
            suggestions.append("Remove or redact personal information")
        
        if ContentCategory.GAMBLING_VIOLATION in result.categories:
            suggestions.append("Add responsible gambling disclaimers")
            suggestions.append("Ensure age verification requirements are mentioned")
        
        if ContentCategory.HATE_SPEECH in result.categories:
            suggestions.append("Remove discriminatory language")
        
        if ContentCategory.SPAM in result.categories:
            suggestions.append("Reduce promotional language")
        
        if result.sanitized_content:
            suggestions.append("Sanitized version available for review")
        
        return suggestions
    
    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        recent_requests = [req_time for req_time in self.request_history['moderation'] if req_time > minute_ago]
        self.request_history['moderation'] = recent_requests
        
        # Check limit
        if len(recent_requests) >= self.config.max_requests_per_minute:
            return False
        
        # Add current request
        self.request_history['moderation'].append(current_time)
        return True
    
    def _generate_cache_key(self, content: str) -> str:
        """Generate cache key for content"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: ModerationResult) -> bool:
        """Check if cached result is still valid"""
        
        cache_age = datetime.utcnow() - cached_result.timestamp
        return cache_age < timedelta(hours=self.config.cache_ttl_hours)
    
    async def _log_moderation_event(
        self,
        result: ModerationResult,
        content: str,
        user_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ):
        """Log moderation event for audit purposes"""
        
        if not self.audit_logger:
            return
        
        try:
            audit_data = {
                'action': AuditAction.CONTENT_MODERATED.value,
                'user_id': user_id,
                'resource_type': 'content',
                'resource_id': context.get('content_id') if context else None,
                'details': {
                    'is_flagged': result.is_flagged,
                    'severity': result.severity.value,
                    'categories': [cat.value for cat in result.categories],
                    'confidence_score': result.confidence_score,
                    'explanation': result.explanation,
                    'openai_flagged': result.openai_flagged,
                    'custom_violations': result.custom_violations,
                    'content_preview': content[:100] + "..." if len(content) > 100 else content,
                    'processing_time_ms': result.processing_time_ms
                },
                'ip_address': context.get('ip_address') if context else None,
                'user_agent': context.get('user_agent') if context else None
            }
            
            await self.audit_logger.log_event(**audit_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log moderation event: {e}")
    
    async def batch_moderate_content(
        self,
        content_list: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[ModerationResult]:
        """
        Moderate multiple content items in batch for efficiency.
        
        Args:
            content_list: List of content dictionaries with 'content' and optional 'context'
            user_id: User ID for audit logging
            
        Returns:
            List of ModerationResult objects
        """
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        results = []
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [
                self.moderate_content(
                    content=item['content'],
                    context=item.get('context'),
                    user_id=user_id
                )
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in batch_results if isinstance(r, ModerationResult)]
            results.extend(valid_results)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(content_list):
                await asyncio.sleep(0.1)
        
        return results
    
    def get_moderation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive moderation metrics"""
        
        return {
            'total_requests': self.metrics['total_requests'],
            'flagged_content': self.metrics['flagged_content'],
            'flagged_percentage': (
                self.metrics['flagged_content'] / self.metrics['total_requests'] * 100
                if self.metrics['total_requests'] > 0 else 0
            ),
            'cache_hits': self.metrics['cache_hits'],
            'cache_hit_rate': (
                self.metrics['cache_hits'] / self.metrics['total_requests'] * 100
                if self.metrics['total_requests'] > 0 else 0
            ),
            'openai_requests': self.metrics['openai_requests'],
            'custom_rule_violations': self.metrics['custom_rule_violations'],
            'average_processing_time_ms': self.metrics['average_processing_time'],
            'openai_available': self.openai_client is not None,
            'cache_size': len(self.moderation_cache),
            'rate_limit_remaining': (
                self.config.max_requests_per_minute - 
                len(self.request_history.get('moderation', []))
            )
        }
    
    async def clear_cache(self) -> int:
        """Clear the moderation cache"""
        
        cache_size = len(self.moderation_cache)
        self.moderation_cache.clear()
        return cache_size
    
    def update_config(self, new_config: ContentModerationConfig):
        """Update moderation configuration"""
        
        self.config = new_config
        self.logger.info("Content moderation configuration updated")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on moderation system"""
        
        health_status = {
            'status': 'healthy',
            'openai_available': self.openai_client is not None,
            'custom_rules_loaded': bool(self.content_rules),
            'cache_enabled': self.config.enable_caching,
            'audit_logging_enabled': self.audit_logger is not None,
            'last_request_time': max(self.request_history.get('moderation', [0])),
            'metrics': self.get_moderation_metrics()
        }
        
        # Test OpenAI connection if available
        if self.openai_client:
            try:
                test_result = await self._openai_moderation("test content")
                health_status['openai_connection'] = 'working'
            except Exception as e:
                health_status['openai_connection'] = f'error: {str(e)}'
                health_status['status'] = 'degraded'
        
        return health_status


# Factory function
def create_content_moderator(
    openai_api_key: Optional[str] = None,
    config: Optional[ContentModerationConfig] = None,
    audit_logger: Optional[AuditLogger] = None
) -> ContentModerator:
    """Factory function to create a configured ContentModerator"""
    
    return ContentModerator(
        openai_api_key=openai_api_key,
        config=config or ContentModerationConfig(),
        audit_logger=audit_logger
    )


# Export all necessary components
__all__ = [
    'ContentModerator',
    'ModerationResult',
    'ContentModerationConfig',
    'ContentSeverity',
    'ContentCategory',
    'create_content_moderator'
] 