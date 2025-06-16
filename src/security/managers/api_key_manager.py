"""
API Key Manager
Secure API key generation, validation, and lifecycle management

Features:
- Secure API key generation with configurable entropy
- Automatic key rotation and expiry
- Usage tracking and rate limiting
- Scope-based permissions
- Audit trail integration
"""

import asyncio
import logging
import secrets
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..models import UserRole, Permission
from ..utils.database_utils import get_database_connection

logger = logging.getLogger(__name__)


class APIKeyStatus(Enum):
    """API Key status types."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class APIKeyInfo:
    """API Key information structure."""
    id: str
    key_hash: str
    user_id: str
    role: UserRole
    permissions: List[Permission]
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit: int
    scopes: List[str]
    name: Optional[str]
    description: Optional[str]


class APIKeyManager:
    """
    Enterprise API Key Manager for secure key lifecycle management.
    
    Provides comprehensive API key management including:
    - Secure key generation and storage
    - Automatic rotation and expiry
    - Usage tracking and analytics
    - Scope-based access control
    - Rate limiting integration
    """
    
    def __init__(self):
        """Initialize API Key Manager."""
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.default_expiry_days = 90
        self.key_length = 64  # characters
        self.hash_algorithm = 'sha256'
        self.max_keys_per_user = 10
        
        # Rate limiting defaults
        self.default_rate_limits = {
            UserRole.SUPER_ADMIN: 10000,  # requests per hour
            UserRole.ADMIN: 5000,
            UserRole.CONTENT_MANAGER: 2000,
            UserRole.EDITOR: 1000,
            UserRole.USER: 500,
            UserRole.API_USER: 1000,
            UserRole.ANONYMOUS: 100
        }
        
        # Key prefixes for identification
        self.key_prefixes = {
            UserRole.SUPER_ADMIN: 'rag_sa_',
            UserRole.ADMIN: 'rag_admin_',
            UserRole.CONTENT_MANAGER: 'rag_cm_',
            UserRole.EDITOR: 'rag_edit_',
            UserRole.USER: 'rag_user_',
            UserRole.API_USER: 'rag_api_',
            UserRole.ANONYMOUS: 'rag_anon_'
        }
        
        # Performance metrics
        self.metrics = {
            'keys_generated': 0,
            'keys_validated': 0,
            'keys_rotated': 0,
            'keys_revoked': 0,
            'validation_failures': 0,
            'usage_tracking_updates': 0
        }
        
        # Cache for recently validated keys
        self.validation_cache = {}
        self.cache_ttl = timedelta(minutes=5)
    
    async def initialize(self) -> bool:
        """Initialize API Key Manager."""
        
        try:
            # Verify database schema
            await self._verify_api_key_schema()
            
            # Clean up expired keys
            await self.cleanup_expired_keys()
            
            self.logger.info("API Key Manager initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"API Key Manager initialization failed: {e}")
            return False
    
    async def generate_api_key(
        self,
        user_id: str,
        role: UserRole,
        permissions: Optional[List[Permission]] = None,
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        rate_limit: Optional[int] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Generate a new API key for a user.
        
        Args:
            user_id: User identifier
            role: User role for the key
            permissions: Optional specific permissions (defaults to role permissions)
            expires_in_days: Expiry in days (default: 90)
            scopes: Optional access scopes
            name: Optional key name
            description: Optional key description
            rate_limit: Optional rate limit override
            
        Returns:
            Tuple of (success, api_key, key_id)
        """
        
        try:
            # Check if user has reached key limit
            if await self._check_key_limit(user_id):
                return False, None, "User has reached maximum number of API keys"
            
            # Generate secure API key
            api_key = self._generate_secure_key(role)
            key_hash = self._hash_key(api_key)
            key_id = secrets.token_urlsafe(16)
            
            # Set expiry
            expires_at = None
            if expires_in_days or self.default_expiry_days:
                days = expires_in_days or self.default_expiry_days
                expires_at = datetime.utcnow() + timedelta(days=days)
            
            # Set permissions
            if permissions is None:
                permissions = await self._get_default_permissions(role)
            
            # Set rate limit
            if rate_limit is None:
                rate_limit = self.default_rate_limits.get(role, 1000)
            
            # Store in database
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    insert_query = """
                        INSERT INTO api_keys 
                        (id, key_hash, user_id, role, permissions, status, created_at, 
                         expires_at, rate_limit, scopes, name, description)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s)
                    """
                    
                    await cursor.execute(insert_query, (
                        key_id,
                        key_hash,
                        user_id,
                        role.value,
                        ','.join([p.value for p in permissions]),
                        APIKeyStatus.ACTIVE.value,
                        expires_at,
                        rate_limit,
                        ','.join(scopes or []),
                        name,
                        description
                    ))
                    
                    await conn.commit()
            
            self.metrics['keys_generated'] += 1
            
            self.logger.info(f"API key generated for user {user_id}, role {role.value}")
            return True, api_key, key_id
        
        except Exception as e:
            self.logger.error(f"API key generation failed: {e}")
            return False, None, f"Key generation failed: {str(e)}"
    
    async def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate an API key and return key information.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (valid, key_info_dict)
        """
        
        try:
            self.metrics['keys_validated'] += 1
            
            # Check validation cache first
            key_hash = self._hash_key(api_key)
            cached_info = self._get_cached_validation(key_hash)
            if cached_info is not None:
                if cached_info['valid']:
                    await self._update_key_usage(cached_info['key_id'])
                return cached_info['valid'], cached_info.get('info')
            
            # Query database
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT id, user_id, role, permissions, status, created_at, 
                               expires_at, last_used_at, usage_count, rate_limit, 
                               scopes, name, description
                        FROM api_keys 
                        WHERE key_hash = %s
                    """
                    
                    await cursor.execute(query, (key_hash,))
                    result = await cursor.fetchone()
                    
                    if not result:
                        self.metrics['validation_failures'] += 1
                        self._cache_validation_result(key_hash, False, None)
                        return False, None
                    
                    # Parse result
                    (key_id, user_id, role_str, permissions_str, status_str, 
                     created_at, expires_at, last_used_at, usage_count, 
                     rate_limit, scopes_str, name, description) = result
                    
                    # Check if key is active
                    if status_str != APIKeyStatus.ACTIVE.value:
                        self.metrics['validation_failures'] += 1
                        self._cache_validation_result(key_hash, False, None)
                        return False, None
                    
                    # Check if key has expired
                    if expires_at and datetime.utcnow() > expires_at:
                        # Mark as expired
                        await self._update_key_status(key_id, APIKeyStatus.EXPIRED)
                        self.metrics['validation_failures'] += 1
                        self._cache_validation_result(key_hash, False, None)
                        return False, None
                    
                    # Build key info
                    key_info = {
                        'id': key_id,
                        'user_id': user_id,
                        'role': role_str,
                        'permissions': permissions_str.split(',') if permissions_str else [],
                        'status': status_str,
                        'created_at': created_at.isoformat(),
                        'expires_at': expires_at.isoformat() if expires_at else None,
                        'last_used_at': last_used_at.isoformat() if last_used_at else None,
                        'usage_count': usage_count,
                        'rate_limit': rate_limit,
                        'scopes': scopes_str.split(',') if scopes_str else [],
                        'name': name,
                        'description': description
                    }
                    
                    # Cache result
                    self._cache_validation_result(key_hash, True, key_info)
                    
                    # Update usage
                    await self._update_key_usage(key_id)
                    
                    return True, key_info
        
        except Exception as e:
            self.logger.error(f"API key validation error: {e}")
            self.metrics['validation_failures'] += 1
            return False, None
    
    async def revoke_api_key(
        self,
        key_id: str,
        revoked_by: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            revoked_by: User ID performing revocation
            reason: Optional revocation reason
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    update_query = """
                        UPDATE api_keys 
                        SET status = %s, revoked_at = NOW(), revoked_by = %s, 
                            revocation_reason = %s
                        WHERE id = %s
                    """
                    
                    await cursor.execute(update_query, (
                        APIKeyStatus.REVOKED.value,
                        revoked_by,
                        reason,
                        key_id
                    ))
                    
                    await conn.commit()
                    
                    if cursor.rowcount > 0:
                        self.metrics['keys_revoked'] += 1
                        
                        # Clear from cache
                        self._clear_key_from_cache(key_id)
                        
                        self.logger.info(f"API key {key_id} revoked by {revoked_by}")
                        return True
                    else:
                        return False
        
        except Exception as e:
            self.logger.error(f"API key revocation failed: {e}")
            return False
    
    async def rotate_keys(
        self,
        key_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        older_than_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rotate API keys based on criteria.
        
        Args:
            key_ids: Specific key IDs to rotate
            user_id: Rotate keys for specific user
            older_than_days: Rotate keys older than specified days
            
        Returns:
            Dict containing rotation results
        """
        
        try:
            rotated_keys = []
            failed_rotations = []
            
            # Build query criteria
            where_clauses = ["status = %s"]
            params = [APIKeyStatus.ACTIVE.value]
            
            if key_ids:
                key_placeholders = ','.join(['%s'] * len(key_ids))
                where_clauses.append(f"id IN ({key_placeholders})")
                params.extend(key_ids)
            
            if user_id:
                where_clauses.append("user_id = %s")
                params.append(user_id)
            
            if older_than_days:
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                where_clauses.append("created_at < %s")
                params.append(cutoff_date)
            
            where_clause = " AND ".join(where_clauses)
            
            # Get keys to rotate
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    select_query = f"""
                        SELECT id, user_id, role, permissions, rate_limit, 
                               scopes, name, description
                        FROM api_keys 
                        WHERE {where_clause}
                    """
                    
                    await cursor.execute(select_query, params)
                    keys_to_rotate = await cursor.fetchall()
                    
                    # Rotate each key
                    for key_data in keys_to_rotate:
                        (old_key_id, user_id, role_str, permissions_str, 
                         rate_limit, scopes_str, name, description) = key_data
                        
                        try:
                            # Generate new key
                            role = UserRole(role_str)
                            permissions = [Permission(p) for p in permissions_str.split(',') if p]
                            scopes = scopes_str.split(',') if scopes_str else []
                            
                            success, new_api_key, new_key_id = await self.generate_api_key(
                                user_id=user_id,
                                role=role,
                                permissions=permissions,
                                scopes=scopes,
                                name=f"{name} (Rotated)" if name else "Rotated Key",
                                description=f"Rotated from {old_key_id}",
                                rate_limit=rate_limit
                            )
                            
                            if success:
                                # Revoke old key
                                await self.revoke_api_key(
                                    old_key_id, 
                                    "system", 
                                    "Automatic rotation"
                                )
                                
                                rotated_keys.append({
                                    'old_key_id': old_key_id,
                                    'new_key_id': new_key_id,
                                    'new_api_key': new_api_key,
                                    'user_id': user_id
                                })
                                
                                self.metrics['keys_rotated'] += 1
                            else:
                                failed_rotations.append({
                                    'key_id': old_key_id,
                                    'error': 'Failed to generate new key'
                                })
                        
                        except Exception as e:
                            failed_rotations.append({
                                'key_id': old_key_id,
                                'error': str(e)
                            })
                    
                    return {
                        'status': 'completed',
                        'rotated_keys': rotated_keys,
                        'failed_rotations': failed_rotations,
                        'total_processed': len(keys_to_rotate),
                        'successful': len(rotated_keys),
                        'failed': len(failed_rotations)
                    }
        
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'rotated_keys': [],
                'failed_rotations': []
            }
    
    async def get_user_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all API keys for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of key information dictionaries
        """
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT id, role, permissions, status, created_at, expires_at, 
                               last_used_at, usage_count, rate_limit, scopes, name, description
                        FROM api_keys 
                        WHERE user_id = %s 
                        ORDER BY created_at DESC
                    """
                    
                    await cursor.execute(query, (user_id,))
                    results = await cursor.fetchall()
                    
                    keys = []
                    for row in results:
                        key_info = {
                            'id': row[0],
                            'role': row[1],
                            'permissions': row[2].split(',') if row[2] else [],
                            'status': row[3],
                            'created_at': row[4].isoformat(),
                            'expires_at': row[5].isoformat() if row[5] else None,
                            'last_used_at': row[6].isoformat() if row[6] else None,
                            'usage_count': row[7],
                            'rate_limit': row[8],
                            'scopes': row[9].split(',') if row[9] else [],
                            'name': row[10],
                            'description': row[11]
                        }
                        keys.append(key_info)
                    
                    return keys
        
        except Exception as e:
            self.logger.error(f"Error getting user keys: {e}")
            return []
    
    async def cleanup_expired_keys(self) -> int:
        """
        Clean up expired API keys.
        
        Returns:
            Number of keys cleaned up
        """
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Update expired keys
                    update_query = """
                        UPDATE api_keys 
                        SET status = %s 
                        WHERE expires_at < NOW() AND status = %s
                    """
                    
                    await cursor.execute(update_query, (
                        APIKeyStatus.EXPIRED.value,
                        APIKeyStatus.ACTIVE.value
                    ))
                    
                    expired_count = cursor.rowcount
                    await conn.commit()
                    
                    if expired_count > 0:
                        self.logger.info(f"Marked {expired_count} keys as expired")
                    
                    return expired_count
        
        except Exception as e:
            self.logger.error(f"Key cleanup failed: {e}")
            return 0
    
    def _generate_secure_key(self, role: UserRole) -> str:
        """Generate a secure API key with role prefix."""
        
        prefix = self.key_prefixes.get(role, 'rag_')
        random_part = secrets.token_urlsafe(self.key_length)
        
        return f"{prefix}{random_part}"
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        
        return hashlib.new(self.hash_algorithm, api_key.encode()).hexdigest()
    
    async def _check_key_limit(self, user_id: str) -> bool:
        """Check if user has reached key limit."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT COUNT(*) FROM api_keys 
                        WHERE user_id = %s AND status = %s
                    """
                    
                    await cursor.execute(query, (user_id, APIKeyStatus.ACTIVE.value))
                    result = await cursor.fetchone()
                    
                    return result[0] >= self.max_keys_per_user
        
        except Exception as e:
            self.logger.error(f"Error checking key limit: {e}")
            return True  # Fail safe
    
    async def _get_default_permissions(self, role: UserRole) -> List[Permission]:
        """Get default permissions for a role."""
        
        # This would typically come from the RBAC manager
        # For now, returning basic permissions based on role
        role_permissions = {
            UserRole.SUPER_ADMIN: [p for p in Permission],
            UserRole.ADMIN: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT, 
                Permission.DELETE_CONTENT, Permission.MANAGE_USERS,
                Permission.VIEW_ANALYTICS, Permission.API_ACCESS
            ],
            UserRole.CONTENT_MANAGER: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT, 
                Permission.DELETE_CONTENT, Permission.API_ACCESS
            ],
            UserRole.EDITOR: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT, 
                Permission.API_ACCESS
            ],
            UserRole.USER: [Permission.READ_CONTENT, Permission.API_ACCESS],
            UserRole.API_USER: [Permission.READ_CONTENT, Permission.API_ACCESS],
            UserRole.ANONYMOUS: [Permission.READ_CONTENT]
        }
        
        return role_permissions.get(role, [Permission.READ_CONTENT])
    
    async def _update_key_usage(self, key_id: str):
        """Update key usage statistics."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    update_query = """
                        UPDATE api_keys 
                        SET usage_count = usage_count + 1, last_used_at = NOW()
                        WHERE id = %s
                    """
                    
                    await cursor.execute(update_query, (key_id,))
                    await conn.commit()
                    
                    self.metrics['usage_tracking_updates'] += 1
        
        except Exception as e:
            self.logger.error(f"Usage tracking update failed: {e}")
    
    async def _update_key_status(self, key_id: str, status: APIKeyStatus):
        """Update key status."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    update_query = "UPDATE api_keys SET status = %s WHERE id = %s"
                    await cursor.execute(update_query, (status.value, key_id))
                    await conn.commit()
        
        except Exception as e:
            self.logger.error(f"Status update failed: {e}")
    
    def _get_cached_validation(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached validation result."""
        
        if key_hash in self.validation_cache:
            cached_time, result = self.validation_cache[key_hash]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                return result
            else:
                del self.validation_cache[key_hash]
        
        return None
    
    def _cache_validation_result(self, key_hash: str, valid: bool, info: Optional[Dict[str, Any]]):
        """Cache validation result."""
        
        self.validation_cache[key_hash] = (
            datetime.utcnow(),
            {'valid': valid, 'info': info}
        )
    
    def _clear_key_from_cache(self, key_id: str):
        """Clear specific key from validation cache."""
        
        # In a real implementation, you'd maintain a reverse mapping
        # For now, just clear entire cache when keys are revoked
        self.validation_cache.clear()
    
    async def _verify_api_key_schema(self):
        """Verify API key table schema exists."""
        
        async with get_database_connection() as conn:
            async with conn.cursor() as cursor:
                check_query = """
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = 'api_keys'
                """
                
                await cursor.execute(check_query)
                result = await cursor.fetchone()
                
                if result[0] == 0:
                    raise Exception("API keys table does not exist. Run security migration first.")
    
    async def get_key_metrics(self) -> Dict[str, Any]:
        """Get API key management metrics."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Get key statistics
                    stats_query = """
                        SELECT 
                            status,
                            COUNT(*) as count,
                            AVG(usage_count) as avg_usage
                        FROM api_keys 
                        GROUP BY status
                    """
                    
                    await cursor.execute(stats_query)
                    status_stats = await cursor.fetchall()
                    
                    # Get expiry statistics
                    expiry_query = """
                        SELECT 
                            COUNT(*) as expiring_soon
                        FROM api_keys 
                        WHERE expires_at BETWEEN NOW() AND NOW() + INTERVAL '7 days'
                        AND status = %s
                    """
                    
                    await cursor.execute(expiry_query, (APIKeyStatus.ACTIVE.value,))
                    expiring_result = await cursor.fetchone()
                    
                    return {
                        'total_metrics': self.metrics,
                        'key_statistics': {
                            'by_status': {row[0]: {'count': row[1], 'avg_usage': float(row[2] or 0)} 
                                        for row in status_stats},
                            'expiring_soon': expiring_result[0] if expiring_result else 0
                        },
                        'cache_size': len(self.validation_cache),
                        'configuration': {
                            'default_expiry_days': self.default_expiry_days,
                            'max_keys_per_user': self.max_keys_per_user,
                            'key_length': self.key_length
                        }
                    }
        
        except Exception as e:
            self.logger.error(f"Error getting key metrics: {e}")
            return {'error': 'Failed to get metrics'}
    
    async def shutdown(self):
        """Shutdown API Key Manager."""
        
        try:
            # Clear caches
            self.validation_cache.clear()
            
            self.logger.info("API Key Manager shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during API Key Manager shutdown: {e}") 