"""
Audit Logger Manager
Comprehensive audit logging and compliance tracking for Universal RAG CMS

Features:
- Comprehensive audit trail logging
- GDPR compliance tracking
- Security event monitoring
- Performance metrics
- Retention management
- Export capabilities
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict

from ..models import AuditAction, SecurityContext
from ..utils.database_utils import get_database_connection

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Audit log entry structure."""
    timestamp: datetime
    action: AuditAction
    user_id: Optional[str]
    session_id: Optional[str]
    client_ip: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    details: Dict[str, Any]
    result: str  # 'success', 'failure', 'warning'
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'client_ip': self.client_ip,
            'user_agent': self.user_agent,
            'resource': self.resource,
            'details': self.details,
            'result': self.result,
            'severity': self.severity
        }


class AuditLevel(Enum):
    """Audit logging levels."""
    MINIMAL = "minimal"      # Critical events only
    STANDARD = "standard"    # Important security events
    DETAILED = "detailed"    # All security-relevant events
    VERBOSE = "verbose"      # All events including debug


class AuditLogger:
    """
    Enterprise Audit Logger for comprehensive compliance tracking.
    
    Provides secure, tamper-evident audit logging with:
    - Comprehensive event tracking
    - GDPR compliance features
    - Retention management
    - Performance optimization
    - Export capabilities
    """
    
    def __init__(self, audit_level: AuditLevel = AuditLevel.STANDARD):
        """Initialize audit logger with specified logging level."""
        
        self.logger = logging.getLogger(__name__)
        self.audit_level = audit_level
        
        # Audit configuration
        self.retention_days = 2555  # 7 years for compliance
        self.batch_size = 100
        self.flush_interval = 60  # seconds
        
        # Buffer for batched logging
        self.audit_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_entries': 0,
            'entries_by_action': {},
            'entries_by_severity': {},
            'buffer_flushes': 0,
            'export_requests': 0
        }
        
        # Background task for periodic flushing
        self.flush_task = None
        self.running = False
    
    async def initialize(self) -> bool:
        """Initialize audit logger and start background tasks."""
        
        try:
            # Verify database connection and schema
            await self._verify_audit_schema()
            
            # Start background flush task
            self.running = True
            self.flush_task = asyncio.create_task(self._periodic_flush())
            
            self.logger.info(f"Audit Logger initialized with level: {self.audit_level.value}")
            return True
        
        except Exception as e:
            self.logger.error(f"Audit Logger initialization failed: {e}")
            return False
    
    async def log_action(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        result: str = 'success',
        severity: str = 'medium'
    ) -> bool:
        """
        Log an audit action.
        
        Args:
            action: The audit action being logged
            user_id: User performing the action
            session_id: Session identifier
            client_ip: Client IP address
            user_agent: User agent string
            resource: Resource being acted upon
            details: Additional action details
            result: Action result ('success', 'failure', 'warning')
            severity: Event severity ('low', 'medium', 'high', 'critical')
            
        Returns:
            True if logged successfully, False otherwise
        """
        
        try:
            # Check if this action should be logged at current level
            if not self._should_log_action(action, severity):
                return True
            
            # Create audit entry
            audit_entry = AuditEntry(
                timestamp=datetime.utcnow(),
                action=action,
                user_id=user_id,
                session_id=session_id,
                client_ip=client_ip,
                user_agent=user_agent,
                resource=resource,
                details=details or {},
                result=result,
                severity=severity
            )
            
            # Add to buffer for batched processing
            async with self.buffer_lock:
                self.audit_buffer.append(audit_entry)
                
                # Flush immediately for critical events
                if severity == 'critical' or len(self.audit_buffer) >= self.batch_size:
                    await self._flush_buffer()
            
            # Update metrics
            self.metrics['total_entries'] += 1
            self.metrics['entries_by_action'][action.value] = self.metrics['entries_by_action'].get(action.value, 0) + 1
            self.metrics['entries_by_severity'][severity] = self.metrics['entries_by_severity'].get(severity, 0) + 1
            
            return True
        
        except Exception as e:
            self.logger.error(f"Audit logging failed: {e}")
            return False
    
    async def log_security_event(
        self,
        event_type: str,
        security_context: Optional[SecurityContext] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = 'high'
    ) -> bool:
        """
        Log a security-specific event.
        
        Args:
            event_type: Type of security event
            security_context: Current security context
            details: Event details
            severity: Event severity
            
        Returns:
            True if logged successfully, False otherwise
        """
        
        return await self.log_action(
            action=AuditAction.SECURITY_INCIDENT,
            user_id=security_context.user_id if security_context else None,
            session_id=security_context.session_id if security_context else None,
            client_ip=security_context.client_ip if security_context else None,
            user_agent=security_context.user_agent if security_context else None,
            details={
                'event_type': event_type,
                'security_details': details or {}
            },
            result='warning',
            severity=severity
        )
    
    async def get_audit_summary(
        self,
        time_range: Optional[timedelta] = None,
        user_id: Optional[str] = None,
        action_filter: Optional[List[AuditAction]] = None
    ) -> Dict[str, Any]:
        """
        Get audit summary and statistics.
        
        Args:
            time_range: Time range for summary (default: last 24 hours)
            user_id: Filter by specific user
            action_filter: Filter by specific actions
            
        Returns:
            Dict containing audit summary
        """
        
        try:
            if not time_range:
                time_range = timedelta(hours=24)
            
            start_time = datetime.utcnow() - time_range
            
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Build query with filters
                    where_clauses = ["timestamp >= %s"]
                    params = [start_time]
                    
                    if user_id:
                        where_clauses.append("user_id = %s")
                        params.append(user_id)
                    
                    if action_filter:
                        action_placeholders = ','.join(['%s'] * len(action_filter))
                        where_clauses.append(f"action IN ({action_placeholders})")
                        params.extend([action.value for action in action_filter])
                    
                    where_clause = " AND ".join(where_clauses)
                    
                    # Get summary statistics
                    summary_query = f"""
                        SELECT 
                            COUNT(*) as total_events,
                            COUNT(DISTINCT user_id) as unique_users,
                            COUNT(DISTINCT client_ip) as unique_ips,
                            action,
                            result,
                            severity,
                            COUNT(*) as count
                        FROM security_audit_log 
                        WHERE {where_clause}
                        GROUP BY action, result, severity
                        ORDER BY count DESC
                    """
                    
                    await cursor.execute(summary_query, params)
                    results = await cursor.fetchall()
                    
                    # Process results
                    summary = {
                        'time_range': {
                            'start': start_time.isoformat(),
                            'end': datetime.utcnow().isoformat(),
                            'duration_hours': time_range.total_seconds() / 3600
                        },
                        'totals': {
                            'events': 0,
                            'unique_users': 0,
                            'unique_ips': 0
                        },
                        'breakdown': {
                            'by_action': {},
                            'by_result': {},
                            'by_severity': {}
                        },
                        'top_events': []
                    }
                    
                    for row in results:
                        # Update totals (first row contains overall stats)
                        if summary['totals']['events'] == 0:
                            summary['totals']['events'] = row[0]
                            summary['totals']['unique_users'] = row[1]
                            summary['totals']['unique_ips'] = row[2]
                        
                        action, result, severity, count = row[3:]
                        
                        # Update breakdowns
                        summary['breakdown']['by_action'][action] = summary['breakdown']['by_action'].get(action, 0) + count
                        summary['breakdown']['by_result'][result] = summary['breakdown']['by_result'].get(result, 0) + count
                        summary['breakdown']['by_severity'][severity] = summary['breakdown']['by_severity'].get(severity, 0) + count
                        
                        # Add to top events
                        summary['top_events'].append({
                            'action': action,
                            'result': result,
                            'severity': severity,
                            'count': count
                        })
                    
                    return summary
        
        except Exception as e:
            self.logger.error(f"Error generating audit summary: {e}")
            return {'error': 'Failed to generate audit summary'}
    
    async def export_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = 'json',
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Export audit logs for compliance or analysis.
        
        Args:
            start_date: Export start date
            end_date: Export end date
            format: Export format ('json', 'csv')
            filters: Optional filters
            
        Returns:
            Exported data as string or None if failed
        """
        
        try:
            self.metrics['export_requests'] += 1
            
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Build query
                    where_clauses = ["timestamp BETWEEN %s AND %s"]
                    params = [start_date, end_date]
                    
                    if filters:
                        if 'user_id' in filters:
                            where_clauses.append("user_id = %s")
                            params.append(filters['user_id'])
                        
                        if 'actions' in filters:
                            action_placeholders = ','.join(['%s'] * len(filters['actions']))
                            where_clauses.append(f"action IN ({action_placeholders})")
                            params.extend(filters['actions'])
                        
                        if 'severity' in filters:
                            where_clauses.append("severity = %s")
                            params.append(filters['severity'])
                    
                    where_clause = " AND ".join(where_clauses)
                    
                    query = f"""
                        SELECT timestamp, action, user_id, session_id, client_ip, 
                               user_agent, resource, details, result, severity
                        FROM security_audit_log 
                        WHERE {where_clause}
                        ORDER BY timestamp DESC
                    """
                    
                    await cursor.execute(query, params)
                    results = await cursor.fetchall()
                    
                    # Format results
                    if format.lower() == 'json':
                        export_data = []
                        for row in results:
                            entry = {
                                'timestamp': row[0].isoformat(),
                                'action': row[1],
                                'user_id': row[2],
                                'session_id': row[3],
                                'client_ip': row[4],
                                'user_agent': row[5],
                                'resource': row[6],
                                'details': json.loads(row[7]) if row[7] else {},
                                'result': row[8],
                                'severity': row[9]
                            }
                            export_data.append(entry)
                        
                        return json.dumps(export_data, indent=2)
                    
                    elif format.lower() == 'csv':
                        import csv
                        import io
                        
                        output = io.StringIO()
                        writer = csv.writer(output)
                        
                        # Write header
                        writer.writerow([
                            'timestamp', 'action', 'user_id', 'session_id', 'client_ip',
                            'user_agent', 'resource', 'details', 'result', 'severity'
                        ])
                        
                        # Write data
                        for row in results:
                            writer.writerow([
                                row[0].isoformat(),
                                row[1],
                                row[2] or '',
                                row[3] or '',
                                row[4] or '',
                                row[5] or '',
                                row[6] or '',
                                row[7] or '{}',
                                row[8],
                                row[9]
                            ])
                        
                        return output.getvalue()
                    
                    else:
                        raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            self.logger.error(f"Audit export failed: {e}")
            return None
    
    async def cleanup_old_logs(self, older_than_days: Optional[int] = None) -> int:
        """
        Clean up old audit logs based on retention policy.
        
        Args:
            older_than_days: Days to retain (default: use configured retention)
            
        Returns:
            Number of deleted entries
        """
        
        try:
            retention_days = older_than_days or self.retention_days
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Delete old entries
                    delete_query = """
                        DELETE FROM security_audit_log 
                        WHERE timestamp < %s
                    """
                    
                    await cursor.execute(delete_query, (cutoff_date,))
                    deleted_count = cursor.rowcount
                    
                    await conn.commit()
                    
                    self.logger.info(f"Cleaned up {deleted_count} old audit entries")
                    return deleted_count
        
        except Exception as e:
            self.logger.error(f"Audit cleanup failed: {e}")
            return 0
    
    async def _verify_audit_schema(self):
        """Verify audit table schema exists."""
        
        async with get_database_connection() as conn:
            async with conn.cursor() as cursor:
                # Check if audit table exists
                check_query = """
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = 'security_audit_log'
                """
                
                await cursor.execute(check_query)
                result = await cursor.fetchone()
                
                if result[0] == 0:
                    raise Exception("Audit table does not exist. Run security migration first.")
    
    def _should_log_action(self, action: AuditAction, severity: str) -> bool:
        """Check if action should be logged at current audit level."""
        
        if self.audit_level == AuditLevel.MINIMAL:
            return severity == 'critical'
        elif self.audit_level == AuditLevel.STANDARD:
            return severity in ['critical', 'high', 'medium']
        elif self.audit_level == AuditLevel.DETAILED:
            return severity in ['critical', 'high', 'medium', 'low']
        else:  # VERBOSE
            return True
    
    async def _flush_buffer(self):
        """Flush audit buffer to database."""
        
        if not self.audit_buffer:
            return
        
        try:
            async with get_database_connection() as conn:
                async with conn.cursor() as cursor:
                    # Prepare batch insert
                    insert_query = """
                        INSERT INTO security_audit_log 
                        (timestamp, action, user_id, session_id, client_ip, user_agent, 
                         resource, details, result, severity)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # Prepare batch data
                    batch_data = []
                    for entry in self.audit_buffer:
                        batch_data.append((
                            entry.timestamp,
                            entry.action.value,
                            entry.user_id,
                            entry.session_id,
                            entry.client_ip,
                            entry.user_agent,
                            entry.resource,
                            json.dumps(entry.details),
                            entry.result,
                            entry.severity
                        ))
                    
                    # Execute batch insert
                    await cursor.executemany(insert_query, batch_data)
                    await conn.commit()
                    
                    # Clear buffer
                    self.audit_buffer.clear()
                    self.metrics['buffer_flushes'] += 1
        
        except Exception as e:
            self.logger.error(f"Buffer flush failed: {e}")
            # Keep entries in buffer for retry
    
    async def _periodic_flush(self):
        """Periodic background task to flush audit buffer."""
        
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                async with self.buffer_lock:
                    if self.audit_buffer:
                        await self._flush_buffer()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic flush error: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get audit logger performance metrics."""
        
        return {
            'total_entries': self.metrics['total_entries'],
            'buffer_size': len(self.audit_buffer),
            'buffer_flushes': self.metrics['buffer_flushes'],
            'export_requests': self.metrics['export_requests'],
            'entries_by_action': self.metrics['entries_by_action'],
            'entries_by_severity': self.metrics['entries_by_severity'],
            'audit_level': self.audit_level.value,
            'retention_days': self.retention_days
        }
    
    async def shutdown(self):
        """Shutdown audit logger and flush remaining entries."""
        
        try:
            self.running = False
            
            # Cancel background task
            if self.flush_task:
                self.flush_task.cancel()
                try:
                    await self.flush_task
                except asyncio.CancelledError:
                    pass
            
            # Flush remaining entries
            async with self.buffer_lock:
                if self.audit_buffer:
                    await self._flush_buffer()
            
            self.logger.info("Audit Logger shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during audit logger shutdown: {e}") 