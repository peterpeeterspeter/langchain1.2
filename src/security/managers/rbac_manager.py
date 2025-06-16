"""
Role-Based Access Control (RBAC) Manager
Comprehensive permission and role management for Universal RAG CMS

Features:
- Hierarchical role system (6 levels: SUPER_ADMIN to ANONYMOUS)
- Granular permission management
- Resource-level access control
- Dynamic permission evaluation
- Role inheritance and delegation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum

from ..models import UserRole, Permission, SecurityContext
from ..utils.database_utils import get_database_connection

logger = logging.getLogger(__name__)


class AccessDecision(Enum):
    """Access control decision types."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class RBACManager:
    """
    Role-Based Access Control Manager.
    
    Manages user roles, permissions, and access control decisions
    with support for hierarchical roles and resource-level permissions.
    """
    
    def __init__(self):
        """Initialize RBAC Manager with role hierarchy and permissions."""
        
        self.logger = logging.getLogger(__name__)
        
        # Role hierarchy (higher number = more privileges)
        self.role_hierarchy = {
            UserRole.SUPER_ADMIN: 100,
            UserRole.ADMIN: 80,
            UserRole.CONTENT_MANAGER: 60,
            UserRole.EDITOR: 40,
            UserRole.USER: 20,
            UserRole.API_USER: 15,
            UserRole.ANONYMOUS: 0
        }
        
        # Default permissions for each role
        self.default_role_permissions = {
            UserRole.SUPER_ADMIN: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT, Permission.DELETE_CONTENT,
                Permission.MANAGE_USERS, Permission.MANAGE_ROLES, Permission.MANAGE_SYSTEM,
                Permission.VIEW_ANALYTICS, Permission.MANAGE_API_KEYS, Permission.VIEW_AUDIT_LOGS,
                Permission.MANAGE_SECURITY, Permission.VIEW_SECURITY_METRICS, Permission.EXPORT_DATA
            ],
            UserRole.ADMIN: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT, Permission.DELETE_CONTENT,
                Permission.MANAGE_USERS, Permission.VIEW_ANALYTICS, Permission.MANAGE_API_KEYS,
                Permission.VIEW_AUDIT_LOGS, Permission.VIEW_SECURITY_METRICS
            ],
            UserRole.CONTENT_MANAGER: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT, Permission.DELETE_CONTENT,
                Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA
            ],
            UserRole.EDITOR: [
                Permission.READ_CONTENT, Permission.WRITE_CONTENT
            ],
            UserRole.USER: [
                Permission.READ_CONTENT
            ],
            UserRole.API_USER: [
                Permission.READ_CONTENT, Permission.API_ACCESS
            ],
            UserRole.ANONYMOUS: [
                Permission.READ_CONTENT
            ]
        }
        
        # Resource-specific access rules
        self.resource_access_rules = {}
        
        # Permission cache for performance
        self.permission_cache = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Permission dependencies
        self.permission_dependencies = {
            Permission.DELETE_CONTENT: [Permission.WRITE_CONTENT, Permission.READ_CONTENT],
            Permission.MANAGE_USERS: [Permission.READ_CONTENT],
            Permission.MANAGE_SYSTEM: [Permission.MANAGE_USERS],
            Permission.VIEW_SECURITY_METRICS: [Permission.VIEW_ANALYTICS]
        }
    
    async def initialize(self) -> bool:
        """Initialize RBAC system and load configuration."""
        
        try:
            # Load role permissions from database
            await self._load_role_permissions()
            
            # Load resource access rules
            await self._load_resource_rules()
            
            # Validate role hierarchy
            self._validate_role_hierarchy()
            
            self.logger.info("RBAC Manager initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"RBAC initialization failed: {e}")
            return False
    
    async def check_permission(
        self,
        user_role: UserRole,
        permission: Permission,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Check if a role has a specific permission.
        
        Args:
            user_role: User's role
            permission: Permission to check
            user_id: Optional user ID for user-specific checks
            
        Returns:
            True if permission is granted, False otherwise
        """
        
        try:
            # Check cache first
            cache_key = f"{user_role.value}:{permission.value}:{user_id or 'none'}"
            cached_result = self._get_cached_permission(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Check base role permissions
            role_permissions = await self._get_role_permissions(user_role)
            
            if permission in role_permissions:
                result = True
            else:
                # Check if higher role in hierarchy has permission
                result = await self._check_inherited_permission(user_role, permission)
            
            # Check permission dependencies
            if result:
                result = await self._check_permission_dependencies(user_role, permission, user_id)
            
            # Check user-specific overrides if user_id provided
            if user_id:
                user_override = await self._check_user_permission_override(user_id, permission)
                if user_override is not None:
                    result = user_override
            
            # Cache the result
            self._cache_permission(cache_key, result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
            return False
    
    async def check_resource_access(
        self,
        user_role: UserRole,
        user_id: Optional[str],
        permission: Permission,
        resource: str,
        resource_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check access to a specific resource.
        
        Args:
            user_role: User's role
            user_id: User ID
            permission: Permission being requested
            resource: Resource identifier
            resource_data: Optional resource metadata
            
        Returns:
            True if access is granted, False otherwise
        """
        
        try:
            # First check basic permission
            if not await self.check_permission(user_role, permission, user_id):
                return False
            
            # Check resource-specific rules
            resource_rules = self.resource_access_rules.get(resource, {})
            
            # Check role-based resource access
            if 'role_restrictions' in resource_rules:
                role_restrictions = resource_rules['role_restrictions']
                if user_role.value in role_restrictions.get('denied_roles', []):
                    return False
                
                if role_restrictions.get('required_roles'):
                    if user_role.value not in role_restrictions['required_roles']:
                        return False
            
            # Check ownership-based access
            if resource_data and 'owner_id' in resource_data:
                if await self._check_ownership_access(user_id, resource_data['owner_id'], permission):
                    return True
            
            # Check time-based restrictions
            if 'time_restrictions' in resource_rules:
                if not await self._check_time_restrictions(resource_rules['time_restrictions']):
                    return False
            
            # Check custom access rules
            if 'custom_rules' in resource_rules:
                custom_result = await self._evaluate_custom_rules(
                    resource_rules['custom_rules'],
                    user_role, user_id, permission, resource_data
                )
                if not custom_result:
                    return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Resource access check error: {e}")
            return False
    
    async def get_user_permissions(self, user_id: str) -> Tuple[Optional[UserRole], List[Permission]]:
        """
        Get user's role and effective permissions.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (user_role, permissions_list)
        """
        
        try:
            async with get_database_connection() as conn:
                # Get user role
                user_query = """
                    SELECT role FROM user_roles 
                    WHERE user_id = %s AND active = true
                """
                
                async with conn.cursor() as cursor:
                    await cursor.execute(user_query, (user_id,))
                    user_result = await cursor.fetchone()
                    
                    if not user_result:
                        return None, []
                    
                    user_role = UserRole(user_result[0])
                    
                    # Get role permissions
                    role_permissions = await self._get_role_permissions(user_role)
                    
                    # Get user-specific permission overrides
                    override_query = """
                        SELECT permission, granted 
                        FROM user_permission_overrides 
                        WHERE user_id = %s AND active = true
                    """
                    
                    await cursor.execute(override_query, (user_id,))
                    overrides = await cursor.fetchall()
                    
                    # Apply overrides
                    effective_permissions = set(role_permissions)
                    
                    for permission_str, granted in overrides:
                        permission = Permission(permission_str)
                        if granted:
                            effective_permissions.add(permission)
                        else:
                            effective_permissions.discard(permission)
                    
                    return user_role, list(effective_permissions)
        
        except Exception as e:
            self.logger.error(f"Error getting user permissions: {e}")
            return None, []
    
    async def assign_role(
        self,
        user_id: str,
        role: UserRole,
        assigned_by: str,
        expiry_date: Optional[datetime] = None
    ) -> bool:
        """
        Assign a role to a user.
        
        Args:
            user_id: User to assign role to
            role: Role to assign
            assigned_by: User ID of role assigner
            expiry_date: Optional role expiry date
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Deactivate existing roles
                    deactivate_query = """
                        UPDATE user_roles 
                        SET active = false, updated_at = NOW()
                        WHERE user_id = %s
                    """
                    await cursor.execute(deactivate_query, (user_id,))
                    
                    # Insert new role
                    insert_query = """
                        INSERT INTO user_roles 
                        (user_id, role, assigned_by, assigned_at, expires_at, active)
                        VALUES (%s, %s, %s, NOW(), %s, true)
                    """
                    
                    await cursor.execute(insert_query, (
                        user_id, role.value, assigned_by, expiry_date
                    ))
                    
                    await conn.commit()
                    
                    # Clear cache for this user
                    self._clear_user_cache(user_id)
                    
                    self.logger.info(f"Role {role.value} assigned to user {user_id}")
                    return True
        
        except Exception as e:
            self.logger.error(f"Role assignment error: {e}")
            return False
    
    async def revoke_role(self, user_id: str, revoked_by: str) -> bool:
        """
        Revoke user's current role.
        
        Args:
            user_id: User whose role to revoke
            revoked_by: User ID of role revoker
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    update_query = """
                        UPDATE user_roles 
                        SET active = false, revoked_by = %s, revoked_at = NOW()
                        WHERE user_id = %s AND active = true
                    """
                    
                    await cursor.execute(update_query, (revoked_by, user_id))
                    await conn.commit()
                    
                    # Clear cache for this user
                    self._clear_user_cache(user_id)
                    
                    self.logger.info(f"Role revoked for user {user_id}")
                    return True
        
        except Exception as e:
            self.logger.error(f"Role revocation error: {e}")
            return False
    
    async def grant_permission_override(
        self,
        user_id: str,
        permission: Permission,
        granted: bool,
        granted_by: str,
        expiry_date: Optional[datetime] = None
    ) -> bool:
        """
        Grant or deny a specific permission override for a user.
        
        Args:
            user_id: User ID
            permission: Permission to override
            granted: True to grant, False to deny
            granted_by: User ID of granter
            expiry_date: Optional expiry date
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Deactivate existing override for this permission
                    deactivate_query = """
                        UPDATE user_permission_overrides 
                        SET active = false 
                        WHERE user_id = %s AND permission = %s
                    """
                    await cursor.execute(deactivate_query, (user_id, permission.value))
                    
                    # Insert new override
                    insert_query = """
                        INSERT INTO user_permission_overrides 
                        (user_id, permission, granted, granted_by, granted_at, expires_at, active)
                        VALUES (%s, %s, %s, %s, NOW(), %s, true)
                    """
                    
                    await cursor.execute(insert_query, (
                        user_id, permission.value, granted, granted_by, expiry_date
                    ))
                    
                    await conn.commit()
                    
                    # Clear cache for this user
                    self._clear_user_cache(user_id)
                    
                    action = "granted" if granted else "denied"
                    self.logger.info(f"Permission {permission.value} {action} for user {user_id}")
                    return True
        
        except Exception as e:
            self.logger.error(f"Permission override error: {e}")
            return False
    
    async def _load_role_permissions(self):
        """Load role permissions from database."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT role, permission 
                        FROM role_permissions 
                        WHERE active = true
                    """
                    
                    await cursor.execute(query)
                    results = await cursor.fetchall()
                    
                    # Group permissions by role
                    db_permissions = {}
                    for role_str, permission_str in results:
                        role = UserRole(role_str)
                        permission = Permission(permission_str)
                        
                        if role not in db_permissions:
                            db_permissions[role] = []
                        db_permissions[role].append(permission)
                    
                    # Merge with default permissions
                    for role, permissions in self.default_role_permissions.items():
                        if role in db_permissions:
                            # Combine and deduplicate
                            combined = set(permissions) | set(db_permissions[role])
                            self.default_role_permissions[role] = list(combined)
                        # else keep defaults
        
        except Exception as e:
            self.logger.warning(f"Could not load role permissions from database: {e}")
            # Continue with default permissions
    
    async def _load_resource_rules(self):
        """Load resource access rules from database."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT resource_id, rule_type, rule_data 
                        FROM resource_access_rules 
                        WHERE active = true
                    """
                    
                    await cursor.execute(query)
                    results = await cursor.fetchall()
                    
                    for resource_id, rule_type, rule_data in results:
                        if resource_id not in self.resource_access_rules:
                            self.resource_access_rules[resource_id] = {}
                        
                        self.resource_access_rules[resource_id][rule_type] = rule_data
        
        except Exception as e:
            self.logger.warning(f"Could not load resource rules: {e}")
    
    def _validate_role_hierarchy(self):
        """Validate that role hierarchy is properly configured."""
        
        for role in UserRole:
            if role not in self.role_hierarchy:
                self.logger.warning(f"Role {role.value} not in hierarchy")
            
            if role not in self.default_role_permissions:
                self.logger.warning(f"No default permissions for role {role.value}")
    
    async def _get_role_permissions(self, role: UserRole) -> List[Permission]:
        """Get permissions for a specific role."""
        return self.default_role_permissions.get(role, [])
    
    async def _check_inherited_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if permission is inherited from higher roles."""
        
        user_role_level = self.role_hierarchy.get(user_role, 0)
        
        for role, level in self.role_hierarchy.items():
            if level > user_role_level:
                role_permissions = await self._get_role_permissions(role)
                if permission in role_permissions:
                    return True
        
        return False
    
    async def _check_permission_dependencies(
        self,
        user_role: UserRole,
        permission: Permission,
        user_id: Optional[str]
    ) -> bool:
        """Check if user has required dependent permissions."""
        
        dependencies = self.permission_dependencies.get(permission, [])
        
        for dep_permission in dependencies:
            if not await self.check_permission(user_role, dep_permission, user_id):
                return False
        
        return True
    
    async def _check_user_permission_override(
        self,
        user_id: str,
        permission: Permission
    ) -> Optional[bool]:
        """Check user-specific permission overrides."""
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    query = """
                        SELECT granted 
                        FROM user_permission_overrides 
                        WHERE user_id = %s AND permission = %s 
                        AND active = true 
                        AND (expires_at IS NULL OR expires_at > NOW())
                    """
                    
                    await cursor.execute(query, (user_id, permission.value))
                    result = await cursor.fetchone()
                    
                    return result[0] if result else None
        
        except Exception as e:
            self.logger.error(f"Error checking permission override: {e}")
            return None
    
    async def _check_ownership_access(
        self,
        user_id: Optional[str],
        owner_id: str,
        permission: Permission
    ) -> bool:
        """Check if user has access based on resource ownership."""
        
        if not user_id:
            return False
        
        # Owner always has access to their own resources
        if user_id == owner_id:
            return True
        
        # Could add delegation logic here
        return False
    
    async def _check_time_restrictions(self, time_restrictions: Dict[str, Any]) -> bool:
        """Check time-based access restrictions."""
        
        current_time = datetime.utcnow()
        
        # Check time windows
        if 'allowed_hours' in time_restrictions:
            current_hour = current_time.hour
            allowed_hours = time_restrictions['allowed_hours']
            if current_hour not in allowed_hours:
                return False
        
        # Check date ranges
        if 'allowed_date_range' in time_restrictions:
            date_range = time_restrictions['allowed_date_range']
            start_date = datetime.fromisoformat(date_range['start'])
            end_date = datetime.fromisoformat(date_range['end'])
            
            if not (start_date <= current_time <= end_date):
                return False
        
        return True
    
    async def _evaluate_custom_rules(
        self,
        custom_rules: List[Dict[str, Any]],
        user_role: UserRole,
        user_id: Optional[str],
        permission: Permission,
        resource_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate custom access rules."""
        
        # This would implement custom rule evaluation logic
        # For now, returning True (permissive)
        return True
    
    def _get_cached_permission(self, cache_key: str) -> Optional[bool]:
        """Get cached permission result."""
        
        if cache_key in self.permission_cache:
            cached_time, result = self.permission_cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                return result
            else:
                del self.permission_cache[cache_key]
        
        return None
    
    def _cache_permission(self, cache_key: str, result: bool):
        """Cache permission result."""
        self.permission_cache[cache_key] = (datetime.utcnow(), result)
    
    def _clear_user_cache(self, user_id: str):
        """Clear cached permissions for a specific user."""
        
        keys_to_remove = [
            key for key in self.permission_cache.keys()
            if f":{user_id}" in key
        ]
        
        for key in keys_to_remove:
            del self.permission_cache[key]
    
    async def get_role_hierarchy(self) -> Dict[str, int]:
        """Get the role hierarchy mapping."""
        return {role.value: level for role, level in self.role_hierarchy.items()}
    
    async def get_effective_permissions(self, user_role: UserRole) -> List[Permission]:
        """Get all effective permissions for a role including inherited ones."""
        
        permissions = set()
        user_role_level = self.role_hierarchy.get(user_role, 0)
        
        # Add permissions from all roles at or below this level
        for role, level in self.role_hierarchy.items():
            if level <= user_role_level:
                role_permissions = await self._get_role_permissions(role)
                permissions.update(role_permissions)
        
        return list(permissions)
    
    async def shutdown(self):
        """Shutdown RBAC manager and clean up resources."""
        
        try:
            # Clear caches
            self.permission_cache.clear()
            self.resource_access_rules.clear()
            
            self.logger.info("RBAC Manager shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during RBAC shutdown: {e}") 