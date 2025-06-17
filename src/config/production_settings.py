"""
Production Settings Configuration for Universal RAG CMS

Manages environment-specific configuration for production deployment:
- Environment variable management
- Security settings
- Performance optimization
- Monitoring configuration
- Database connection settings
- API rate limiting
- Logging configuration
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from datetime import timedelta

class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ProductionSettings(BaseSettings):
    """Production configuration settings."""
    
    # Environment Configuration
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    api_port: int = Field(
        default=8000,
        ge=1000,
        le=65535,
        description="API port number"
    )
    api_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of API workers"
    )
    api_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="API request timeout in seconds"
    )
    
    # Security Configuration
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for encryption"
    )
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="Allowed host origins"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # Database Configuration
    supabase_url: str = Field(
        ...,
        description="Supabase project URL"
    )
    supabase_key: str = Field(
        ...,
        description="Supabase service role key"
    )
    database_pool_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Database connection pool size"
    )
    database_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Database query timeout in seconds"
    )
    
    # AI Model Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model to use"
    )
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="LLM model for query expansion"
    )
    
    # Performance Configuration
    max_concurrent_requests: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum concurrent requests"
    )
    request_rate_limit: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Requests per minute rate limit"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Cache TTL in hours"
    )
    batch_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Default batch processing size"
    )
    
    # Retrieval Configuration
    default_retrieval_strategy: str = Field(
        default="hybrid",
        description="Default retrieval strategy"
    )
    max_results_per_query: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum results per query"
    )
    context_window_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Context window size for embeddings"
    )
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter"
    )
    
    # Monitoring Configuration
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1000,
        le=65535,
        description="Metrics server port"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # Health Check Configuration
    health_check_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Health check interval in seconds"
    )
    health_check_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Health check timeout in seconds"
    )
    
    # Storage Configuration
    temp_dir: str = Field(
        default="/tmp/rag_cms",
        description="Temporary directory for processing"
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum file size in MB"
    )
    
    # Feature Flags
    enable_contextual_retrieval: bool = Field(
        default=True,
        description="Enable contextual retrieval features"
    )
    enable_multi_query: bool = Field(
        default=True,
        description="Enable multi-query retrieval"
    )
    enable_self_query: bool = Field(
        default=True,
        description="Enable self-query filtering"
    )
    enable_performance_optimization: bool = Field(
        default=True,
        description="Enable performance optimization"
    )
    enable_real_time_monitoring: bool = Field(
        default=True,
        description="Enable real-time monitoring"
    )
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v, values):
        """Validate secret key in production."""
        env = values.get("environment", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and v == "your-secret-key-change-in-production":
            raise ValueError("Secret key must be changed in production")
        return v
    
    @validator("allowed_hosts")
    def validate_allowed_hosts(cls, v, values):
        """Validate allowed hosts in production."""
        env = values.get("environment", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and "*" in v:
            logging.warning("Using wildcard '*' for allowed hosts in production is not recommended")
        return v
    
    @validator("temp_dir")
    def validate_temp_dir(cls, v):
        """Ensure temp directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable prefixes
        env_prefix = "RAG_CMS_"
        
        # Field aliases for common environment variables
        fields = {
            "supabase_url": {"env": ["RAG_CMS_SUPABASE_URL", "SUPABASE_URL"]},
            "supabase_key": {"env": ["RAG_CMS_SUPABASE_KEY", "SUPABASE_KEY"]},
            "openai_api_key": {"env": ["RAG_CMS_OPENAI_API_KEY", "OPENAI_API_KEY"]},
            "anthropic_api_key": {"env": ["RAG_CMS_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"]},
            "secret_key": {"env": ["RAG_CMS_SECRET_KEY", "SECRET_KEY"]},
        }
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return f"{self.supabase_url}/rest/v1/"
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level.value,
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.log_level.value,
                    "formatter": "detailed",
                    "filename": f"{self.temp_dir}/rag_cms.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                },
            },
            "loggers": {
                "": {
                    "level": self.log_level.value,
                    "handlers": ["console", "file"],
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "fastapi": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return {
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_rate_limit": self.request_rate_limit,
            "cache_ttl_hours": self.cache_ttl_hours,
            "batch_size": self.batch_size,
            "database_pool_size": self.database_pool_size,
            "database_timeout": self.database_timeout,
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval system configuration."""
        return {
            "default_strategy": self.default_retrieval_strategy,
            "max_results": self.max_results_per_query,
            "context_window_size": self.context_window_size,
            "mmr_lambda": self.mmr_lambda,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flag configuration."""
        return {
            "contextual_retrieval": self.enable_contextual_retrieval,
            "multi_query": self.enable_multi_query,
            "self_query": self.enable_self_query,
            "performance_optimization": self.enable_performance_optimization,
            "real_time_monitoring": self.enable_real_time_monitoring,
        }
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check required API keys
        if self.enable_contextual_retrieval:
            if not self.openai_api_key and not self.anthropic_api_key:
                issues.append("At least one AI API key (OpenAI or Anthropic) is required")
        
        # Check production settings
        if self.is_production():
            if self.debug:
                issues.append("Debug mode should be disabled in production")
            
            if self.secret_key == "your-secret-key-change-in-production":
                issues.append("Secret key must be changed in production")
            
            if "*" in self.allowed_hosts:
                issues.append("Wildcard allowed hosts not recommended in production")
        
        # Check resource limits
        if self.max_concurrent_requests > 500 and self.database_pool_size < 50:
            issues.append("Database pool size may be too small for high concurrency")
        
        return issues

# Global settings instance
_settings: Optional[ProductionSettings] = None

def get_settings() -> ProductionSettings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = ProductionSettings()
        
        # Validate configuration
        issues = _settings.validate_configuration()
        if issues:
            logger = logging.getLogger(__name__)
            logger.warning(f"Configuration issues found: {issues}")
    
    return _settings

def reload_settings() -> ProductionSettings:
    """Reload settings from environment."""
    global _settings
    _settings = None
    return get_settings()

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    Environment.DEVELOPMENT: {
        "debug": True,
        "log_level": LogLevel.DEBUG,
        "api_workers": 1,
        "max_concurrent_requests": 10,
        "request_rate_limit": 100,
        "database_pool_size": 5,
    },
    Environment.STAGING: {
        "debug": False,
        "log_level": LogLevel.INFO,
        "api_workers": 2,
        "max_concurrent_requests": 50,
        "request_rate_limit": 500,
        "database_pool_size": 10,
    },
    Environment.PRODUCTION: {
        "debug": False,
        "log_level": LogLevel.WARNING,
        "api_workers": 4,
        "max_concurrent_requests": 100,
        "request_rate_limit": 1000,
        "database_pool_size": 20,
    },
}

def get_environment_config(environment: Environment) -> Dict[str, Any]:
    """Get environment-specific configuration overrides."""
    return ENVIRONMENT_CONFIGS.get(environment, {})

def apply_environment_config(settings: ProductionSettings) -> ProductionSettings:
    """Apply environment-specific configuration."""
    env_config = get_environment_config(settings.environment)
    
    for key, value in env_config.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    return settings 