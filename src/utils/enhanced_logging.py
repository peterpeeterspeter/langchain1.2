"""
Enhanced Logging System for RAG CMS
Provides structured logging, query pipeline tracking, and error pattern analysis.
"""

import logging
import json
import sys
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import asyncio
from functools import wraps
from supabase import Client
import hashlib

class LogLevel(str, Enum):
    """Log levels with additional granularity."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRACE = "TRACE"  # More detailed than DEBUG

class LogCategory(str, Enum):
    """Categories for log classification."""
    QUERY_PROCESSING = "query_processing"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CACHE = "cache"
    API = "api"
    DATABASE = "database"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    FEATURE_FLAG = "feature_flag"

@dataclass
class LogContext:
    """Context information for structured logging."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            k: v for k, v in asdict(self).items() 
            if v is not None and (not isinstance(v, (list, dict)) or v)
        }

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    context: LogContext
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        entry_dict = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "data": self.data
        }
        
        if self.error:
            entry_dict["error"] = self.error
        if self.duration_ms is not None:
            entry_dict["duration_ms"] = self.duration_ms
            
        return json.dumps(entry_dict)

class StructuredLogger:
    """Enhanced logger with structured output and analysis capabilities."""
    
    def __init__(self, name: str, supabase_client: Optional[Client] = None):
        self.name = name
        self.client = supabase_client
        self.logs_table = "application_logs"
        self.error_patterns_table = "error_patterns"
        
        # Python logger setup
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # JSON formatter for structured logs
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._create_json_formatter())
        self.logger.addHandler(handler)
        
        # In-memory buffers
        self.log_buffer: deque = deque(maxlen=1000)
        self.error_patterns: defaultdict = defaultdict(int)
        self.performance_traces: Dict[str, List[float]] = defaultdict(list)
        
        # Context management
        self._context_stack: List[LogContext] = []
        
        if self.client:
            self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure logging tables exist in Supabase."""
        # These would be in migration files
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS application_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            timestamp TIMESTAMPTZ NOT NULL,
            level TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            query_id TEXT,
            user_id TEXT,
            session_id TEXT,
            request_id TEXT,
            context JSONB DEFAULT '{}',
            data JSONB DEFAULT '{}',
            error JSONB,
            duration_ms FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_logs_timestamp ON application_logs(timestamp DESC);
        CREATE INDEX idx_logs_query_id ON application_logs(query_id);
        CREATE INDEX idx_logs_level ON application_logs(level);
        CREATE INDEX idx_logs_category ON application_logs(category);
        CREATE INDEX idx_logs_error ON application_logs((error IS NOT NULL));
        
        CREATE TABLE IF NOT EXISTS error_patterns (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pattern_hash TEXT NOT NULL UNIQUE,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            stack_trace TEXT,
            occurrence_count INTEGER DEFAULT 1,
            first_seen TIMESTAMPTZ DEFAULT NOW(),
            last_seen TIMESTAMPTZ DEFAULT NOW(),
            affected_queries JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}'
        );
        
        CREATE INDEX idx_error_patterns_type ON error_patterns(error_type);
        CREATE INDEX idx_error_patterns_count ON error_patterns(occurrence_count DESC);
        """
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                if hasattr(record, 'context'):
                    log_obj['context'] = record.context
                
                if hasattr(record, 'data'):
                    log_obj['data'] = record.data
                
                if record.exc_info:
                    log_obj['error'] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info)
                    }
                
                return json.dumps(log_obj)
        
        return JSONFormatter()
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for adding context to logs."""
        context = LogContext(**kwargs)
        self._context_stack.append(context)
        
        try:
            yield context
        finally:
            self._context_stack.pop()
    
    @property
    def current_context(self) -> Optional[LogContext]:
        """Get current logging context."""
        return self._context_stack[-1] if self._context_stack else None
    
    def _log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration_ms: Optional[float] = None
    ):
        """Internal logging method."""
        context = self.current_context or LogContext()
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            message=message,
            context=context,
            data=data or {},
            error=self._format_error(error) if error else None,
            duration_ms=duration_ms
        )
        
        # Add to buffer
        self.log_buffer.append(entry)
        
        # Log to Python logger
        log_record = logging.LogRecord(
            name=self.name,
            level=getattr(logging, level.value),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        log_record.context = context.to_dict()
        log_record.data = data or {}
        
        self.logger.handle(log_record)
        
        # Store in Supabase if available
        if self.client:
            asyncio.create_task(self._store_log_entry(entry))
        
        # Track error patterns
        if error:
            self._track_error_pattern(error, context)
    
    async def _store_log_entry(self, entry: LogEntry):
        """Store log entry in Supabase."""
        try:
            log_data = {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,
                "category": entry.category.value,
                "message": entry.message,
                "query_id": entry.context.query_id,
                "user_id": entry.context.user_id,
                "session_id": entry.context.session_id,
                "request_id": entry.context.request_id,
                "context": entry.context.to_dict(),
                "data": entry.data,
                "error": entry.error,
                "duration_ms": entry.duration_ms
            }
            
            self.client.table(self.logs_table).insert(log_data).execute()
        except Exception as e:
            # Fallback to stderr if storage fails
            print(f"Failed to store log: {e}", file=sys.stderr)
    
    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """Format exception for logging."""
        return {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(
                type(error), error, error.__traceback__
            ),
            "attributes": {
                k: str(v) for k, v in vars(error).items()
                if not k.startswith('_')
            }
        }
    
    def _track_error_pattern(self, error: Exception, context: LogContext):
        """Track error patterns for analysis."""
        # Create pattern hash
        error_type = type(error).__name__
        error_msg = str(error)
        pattern_hash = hashlib.md5(
            f"{error_type}:{error_msg}".encode()
        ).hexdigest()
        
        # Update in-memory tracking
        self.error_patterns[pattern_hash] += 1
        
        # Store in database if available
        if self.client:
            asyncio.create_task(
                self._update_error_pattern(
                    pattern_hash, error_type, error_msg, context
                )
            )
    
    async def _update_error_pattern(
        self,
        pattern_hash: str,
        error_type: str,
        error_message: str,
        context: LogContext
    ):
        """Update error pattern in database."""
        try:
            # Check if pattern exists
            existing = self.client.table(self.error_patterns_table).select(
                "id, occurrence_count, affected_queries"
            ).eq("pattern_hash", pattern_hash).execute()
            
            if existing.data:
                # Update existing pattern
                pattern = existing.data[0]
                affected_queries = pattern["affected_queries"]
                if context.query_id not in affected_queries:
                    affected_queries.append(context.query_id)
                
                self.client.table(self.error_patterns_table).update({
                    "occurrence_count": pattern["occurrence_count"] + 1,
                    "last_seen": datetime.utcnow().isoformat(),
                    "affected_queries": affected_queries[-100:]  # Keep last 100
                }).eq("id", pattern["id"]).execute()
            else:
                # Create new pattern
                self.client.table(self.error_patterns_table).insert({
                    "pattern_hash": pattern_hash,
                    "error_type": error_type,
                    "error_message": error_message[:500],
                    "affected_queries": [context.query_id]
                }).execute()
        except Exception:
            pass  # Ignore storage errors
    
    # Convenience methods for different log levels
    def trace(self, category: LogCategory, message: str, **kwargs):
        """Log trace level message."""
        self._log(LogLevel.TRACE, category, message, **kwargs)
    
    def debug(self, category: LogCategory, message: str, **kwargs):
        """Log debug level message."""
        self._log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, category: LogCategory, message: str, **kwargs):
        """Log info level message."""
        self._log(LogLevel.INFO, category, message, **kwargs)
    
    def warning(self, category: LogCategory, message: str, **kwargs):
        """Log warning level message."""
        self._log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, category: LogCategory, message: str, **kwargs):
        """Log error level message."""
        self._log(LogLevel.ERROR, category, message, **kwargs)
    
    def critical(self, category: LogCategory, message: str, **kwargs):
        """Log critical level message."""
        self._log(LogLevel.CRITICAL, category, message, **kwargs)
    
    # Performance tracking
    @contextmanager
    def track_performance(self, operation: str, category: LogCategory):
        """Track performance of an operation."""
        start_time = datetime.utcnow()
        
        try:
            yield
        finally:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.performance_traces[operation].append(duration_ms)
            
            self.info(
                category,
                f"Completed {operation}",
                data={"operation": operation},
                duration_ms=duration_ms
            )
    
    # Decorators for easy integration
    def log_function(self, category: LogCategory, level: LogLevel = LogLevel.DEBUG):
        """Decorator to log function calls."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                self.debug(
                    category,
                    f"Calling {func.__name__}",
                    data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                )
                
                try:
                    with self.track_performance(func.__name__, category):
                        result = await func(*args, **kwargs)
                    
                    self.debug(
                        category,
                        f"Completed {func.__name__}",
                        data={"result_type": type(result).__name__}
                    )
                    return result
                
                except Exception as e:
                    self.error(
                        category,
                        f"Error in {func.__name__}",
                        error=e
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                self.debug(
                    category,
                    f"Calling {func.__name__}",
                    data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
                )
                
                start_time = datetime.utcnow()
                
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    self.debug(
                        category,
                        f"Completed {func.__name__}",
                        data={"result_type": type(result).__name__},
                        duration_ms=duration_ms
                    )
                    return result
                
                except Exception as e:
                    self.error(
                        category,
                        f"Error in {func.__name__}",
                        error=e
                    )
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    # Analysis methods
    async def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.client:
            return {"error": "No database connection"}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get error patterns
        patterns = self.client.table(self.error_patterns_table).select(
            "*"
        ).gte("last_seen", cutoff_time.isoformat()).order(
            "occurrence_count", desc=True
        ).limit(20).execute()
        
        # Get error logs
        error_logs = self.client.table(self.logs_table).select(
            "category, error"
        ).eq("level", LogLevel.ERROR.value).gte(
            "timestamp", cutoff_time.isoformat()
        ).execute()
        
        # Categorize errors
        errors_by_category = defaultdict(int)
        for log in error_logs.data:
            errors_by_category[log["category"]] += 1
        
        return {
            "period": f"Last {hours} hours",
            "total_errors": len(error_logs.data),
            "unique_patterns": len(patterns.data),
            "top_patterns": [
                {
                    "type": p["error_type"],
                    "message": p["error_message"],
                    "count": p["occurrence_count"],
                    "first_seen": p["first_seen"],
                    "last_seen": p["last_seen"]
                }
                for p in patterns.data[:10]
            ],
            "errors_by_category": dict(errors_by_category),
            "recommendations": self._generate_error_recommendations(patterns.data)
        }
    
    def _generate_error_recommendations(self, patterns: List[Dict]) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        
        for pattern in patterns[:5]:
            error_type = pattern["error_type"]
            count = pattern["occurrence_count"]
            
            if "timeout" in error_type.lower():
                recommendations.append(
                    f"Frequent timeouts detected ({count} occurrences). "
                    "Consider increasing timeout values or optimizing slow operations."
                )
            elif "connection" in error_type.lower():
                recommendations.append(
                    f"Connection errors detected ({count} occurrences). "
                    "Check network stability and implement retry logic."
                )
            elif "validation" in error_type.lower():
                recommendations.append(
                    f"Validation errors detected ({count} occurrences). "
                    "Review input validation rules and error messages."
                )
        
        return recommendations
    
    async def get_query_trace(self, query_id: str) -> List[Dict[str, Any]]:
        """Get complete trace for a query."""
        if not self.client:
            return []
        
        logs = self.client.table(self.logs_table).select(
            "*"
        ).eq("query_id", query_id).order("timestamp").execute()
        
        return [
            {
                "timestamp": log["timestamp"],
                "level": log["level"],
                "category": log["category"],
                "message": log["message"],
                "duration_ms": log.get("duration_ms"),
                "data": log.get("data", {}),
                "error": log.get("error")
            }
            for log in logs.data
        ]

# Singleton logger instances
_loggers: Dict[str, StructuredLogger] = {}

def get_logger(name: str, supabase_client: Optional[Client] = None) -> StructuredLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, supabase_client)
    return _loggers[name]

# Usage example for RAG pipeline
class RAGPipelineLogger:
    """Specialized logger for RAG pipeline with stage tracking."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.stages = [
            "query_classification",
            "retrieval",
            "context_formatting",
            "generation",
            "post_processing"
        ]
        self.current_stage = None
    
    def start_pipeline(self, query_id: str, query_text: str, user_context: Dict[str, Any]):
        """Start logging a RAG pipeline execution."""
        return self.logger.context(
            query_id=query_id,
            user_id=user_context.get("user_id"),
            session_id=user_context.get("session_id"),
            tags=["rag_pipeline"],
            metadata={"query_text": query_text[:200]}
        )
    
    @contextmanager
    def stage(self, stage_name: str):
        """Track a pipeline stage."""
        self.current_stage = stage_name
        
        self.logger.info(
            LogCategory.QUERY_PROCESSING,
            f"Starting stage: {stage_name}",
            data={"stage": stage_name}
        )
        
        with self.logger.track_performance(f"stage_{stage_name}", LogCategory.QUERY_PROCESSING):
            try:
                yield
                self.logger.info(
                    LogCategory.QUERY_PROCESSING,
                    f"Completed stage: {stage_name}",
                    data={"stage": stage_name}
                )
            except Exception as e:
                self.logger.error(
                    LogCategory.QUERY_PROCESSING,
                    f"Failed stage: {stage_name}",
                    data={"stage": stage_name},
                    error=e
                )
                raise 