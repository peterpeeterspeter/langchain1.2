"""
Security Models for Universal RAG CMS
Data classes and enums for security system
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

# === ENUMS ===

class UserRole(Enum):
    """User role hierarchy"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    EDITOR = "editor"
    VIEWER = "viewer"
    API_USER = "api_user"
    
    @property
    def level(self) -> int:
        """Role hierarchy level for comparison"""
        levels = {
            self.SUPER_ADMIN: 100,
            self.ADMIN: 80,
            self.MODERATOR: 60,
            self.EDITOR: 40,
            self.VIEWER: 20,
            self.API_USER: 10
        }
        return levels.get(self, 0)

class Permission(Enum):
    """System permissions"""
    # Content permissions
    CONTENT_CREATE = "content:create"
    CONTENT_READ = "content:read"
    CONTENT_UPDATE = "content:update"
    CONTENT_DELETE = "content:delete"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_AUDIT = "system:audit"
    
    # API permissions
    API_FULL_ACCESS = "api:full"
    API_READ_ONLY = "api:read"
    API_RATE_LIMIT_EXEMPT = "api:no_limit"

class AuditAction(Enum):
    """Auditable actions"""
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    
    CONTENT_CREATE = "content.create"
    CONTENT_READ = "content.read"
    CONTENT_UPDATE = "content.update"
    CONTENT_DELETE = "content.delete"
    
    API_CALL = "api.call"
    API_KEY_CREATE = "api.key_create"
    API_KEY_ROTATE = "api.key_rotate"
    API_KEY_REVOKE = "api.key_revoke"
    
    PERMISSION_GRANT = "permission.grant"
    PERMISSION_REVOKE = "permission.revoke"
    
    SECURITY_VIOLATION = "security.violation"
    RATE_LIMIT_EXCEEDED = "security.rate_limit"
    
    GDPR_REQUEST = "gdpr.request"
    GDPR_EXPORT = "gdpr.export"
    GDPR_DELETE = "gdpr.delete"

# === DATA CLASSES ===

@dataclass
class SecurityConfig:
    """Security configuration with all settings"""
    
    # Encryption settings
    encryption_key: str = field(default_factory=lambda: os.environ.get("ENCRYPTION_KEY", ""))
    salt_value: str = field(default_factory=lambda: os.environ.get("SALT_VALUE", ""))
    
    # JWT settings
    jwt_secret: str = field(default_factory=lambda: os.environ.get("JWT_SECRET", ""))
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    
    # API key settings
    api_key_rotation_days: int = 90
    api_key_length: int = 32
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    burst_limit: int = 100
    
    # Session settings
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5
    
    # Password policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_count: int = 5
    
    # GDPR settings
    data_retention_days: int = 365
    consent_expiry_days: int = 730
    
    # Content moderation
    moderation_threshold: float = 0.8
    moderation_categories: List[str] = field(default_factory=lambda: [
        "hate", "hate/threatening", "harassment", "harassment/threatening",
        "self-harm", "self-harm/intent", "self-harm/instructions",
        "sexual", "sexual/minors", "violence", "violence/graphic"
    ])
    
    # Audit settings
    audit_retention_days: int = 2555  # 7 years
    audit_batch_size: int = 1000
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()
            print("⚠️ Generated new encryption key - save to environment!")
        
        if not self.salt_value:
            self.salt_value = secrets.token_hex(16)
            print("⚠️ Generated new salt value - save to environment!")
        
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)
            print("⚠️ Generated new JWT secret - save to environment!")

# === PYDANTIC MODELS ===

class SecurityContext(BaseModel):
    """Security context for requests"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    roles: List[UserRole]
    permissions: List[Permission]
    api_key_id: Optional[str] = None
    request_id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SecurityViolation(BaseModel):
    """Security violation record"""
    violation_type: str
    severity: str  # low, medium, high, critical
    user_id: Optional[str]
    ip_address: str
    details: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow) 