"""
Enhanced Configuration System for RAG CMS
Provides runtime configuration management with validation and dynamic updates.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from enum import Enum
import json
import hashlib
from supabase import Client

class QueryType(str, Enum):
    """Supported query types for classification."""
    CASINO_REVIEW = "casino_review"
    NEWS = "news"
    PRODUCT_REVIEW = "product_review"
    TECHNICAL_DOC = "technical_doc"
    GENERAL = "general"
    GUIDE = "guide"
    FAQ = "faq"

class CacheConfig(BaseModel):
    """Cache TTL configuration by query type."""
    casino_review_ttl: int = Field(default=24, description="Hours to cache casino reviews")
    news_ttl: int = Field(default=2, description="Hours to cache news content")
    product_review_ttl: int = Field(default=12, description="Hours to cache product reviews")
    technical_doc_ttl: int = Field(default=168, description="Hours to cache technical docs (1 week)")
    general_ttl: int = Field(default=6, description="Hours to cache general content")
    guide_ttl: int = Field(default=48, description="Hours to cache guides")
    faq_ttl: int = Field(default=72, description="Hours to cache FAQs")
    
    def get_ttl(self, query_type: QueryType) -> int:
        """Get TTL in hours for a specific query type."""
        ttl_map = {
            QueryType.CASINO_REVIEW: self.casino_review_ttl,
            QueryType.NEWS: self.news_ttl,
            QueryType.PRODUCT_REVIEW: self.product_review_ttl,
            QueryType.TECHNICAL_DOC: self.technical_doc_ttl,
            QueryType.GENERAL: self.general_ttl,
            QueryType.GUIDE: self.guide_ttl,
            QueryType.FAQ: self.faq_ttl
        }
        return ttl_map.get(query_type, self.general_ttl)

class QueryClassificationConfig(BaseModel):
    """Configuration for query classification."""
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    fallback_type: QueryType = Field(default=QueryType.GENERAL)
    enable_multi_classification: bool = Field(default=False)
    max_classification_attempts: int = Field(default=2, ge=1, le=5)
    
    @validator('confidence_threshold')
    def validate_threshold(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError("Confidence threshold should be between 0.5 and 0.95 for optimal performance")
        return v

class ContextFormattingConfig(BaseModel):
    """Configuration for context formatting."""
    max_context_length: int = Field(default=3000, ge=500, le=10000)
    quality_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    freshness_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    relevance_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = Field(default=True)
    max_chunks_per_source: int = Field(default=3, ge=1, le=10)
    chunk_overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    
    @validator('relevance_weight')
    def validate_weights(cls, v, values):
        if 'freshness_weight' in values:
            if abs(v + values['freshness_weight'] - 1.0) > 0.01:
                raise ValueError("Freshness and relevance weights must sum to 1.0")
        return v

class PerformanceConfig(BaseModel):
    """Performance monitoring and alerting configuration."""
    enable_monitoring: bool = Field(default=True)
    enable_profiling: bool = Field(default=False)
    response_time_warning_ms: int = Field(default=2000, ge=100, le=10000)
    response_time_critical_ms: int = Field(default=5000, ge=1000, le=30000)
    error_rate_warning_percent: float = Field(default=5.0, ge=0.1, le=20.0)
    error_rate_critical_percent: float = Field(default=10.0, ge=1.0, le=30.0)
    min_samples_for_alerts: int = Field(default=100, ge=10, le=1000)
    alert_cooldown_minutes: int = Field(default=15, ge=5, le=60)

class BasicFeatureFlags(BaseModel):
    """Basic feature flags for gradual rollout (legacy system)."""
    enable_contextual_retrieval: bool = Field(default=True)
    enable_hybrid_search: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=False)
    enable_response_caching: bool = Field(default=True)
    enable_semantic_cache: bool = Field(default=True)
    enable_auto_retry: bool = Field(default=True)
    enable_cost_optimization: bool = Field(default=True)
    ab_test_percentage: float = Field(default=100.0, ge=0.0, le=100.0)
    
    @validator('ab_test_percentage')
    def validate_percentage(cls, v):
        return round(v, 2)

class PromptOptimizationConfig(BaseModel):
    """Main configuration class for prompt optimization."""
    query_classification: QueryClassificationConfig = Field(default_factory=QueryClassificationConfig)
    context_formatting: ContextFormattingConfig = Field(default_factory=ContextFormattingConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    feature_flags: BasicFeatureFlags = Field(default_factory=BasicFeatureFlags)
    
    # Metadata
    version: str = Field(default="1.0.0")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = Field(default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptOptimizationConfig':
        """Create configuration from dictionary."""
        return cls.model_validate(data)
    
    def get_hash(self) -> str:
        """Get hash of current configuration for change detection."""
        config_dict = self.model_dump(exclude={'last_updated', 'updated_by'})
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

class ConfigurationManager:
    """Manages configuration storage and retrieval from Supabase."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_name = "prompt_configurations"
        self.cache_duration = timedelta(minutes=5)
        self._cached_config: Optional[PromptOptimizationConfig] = None
        self._cache_timestamp: Optional[datetime] = None
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure configuration table exists in Supabase."""
        # This would be in a migration file
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS prompt_configurations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            config_data JSONB NOT NULL,
            config_hash TEXT NOT NULL UNIQUE,
            version TEXT NOT NULL,
            is_active BOOLEAN DEFAULT false,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            updated_by TEXT,
            rollback_to UUID REFERENCES prompt_configurations(id),
            change_notes TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_prompt_config_active ON prompt_configurations(is_active);
        CREATE INDEX IF NOT EXISTS idx_prompt_config_version ON prompt_configurations(version);
        CREATE INDEX IF NOT EXISTS idx_prompt_config_created ON prompt_configurations(created_at DESC);
        """
    
    async def get_active_config(self, force_refresh: bool = False) -> PromptOptimizationConfig:
        """Get the active configuration with caching."""
        # Check cache first
        if not force_refresh and self._is_cache_valid():
            return self._cached_config
        
        try:
            # Fetch from Supabase
            result = self.client.table(self.table_name).select(
                "config_data"
            ).eq("is_active", True).single().execute()
            
            if result.data:
                config = PromptOptimizationConfig.from_dict(result.data["config_data"])
                self._update_cache(config)
                return config
            else:
                # Return default config if none exists
                default_config = PromptOptimizationConfig()
                await self.save_config(default_config, "System", "Initial default configuration")
                return default_config
        except Exception as e:
            # Fallback to default configuration on error
            print(f"Error fetching configuration: {e}")
            return PromptOptimizationConfig()
    
    async def save_config(
        self, 
        config: PromptOptimizationConfig, 
        updated_by: str,
        change_notes: Optional[str] = None
    ) -> str:
        """Save a new configuration version."""
        config.updated_by = updated_by
        config.last_updated = datetime.utcnow()
        config_hash = config.get_hash()
        
        try:
            # Check if this config already exists
            existing = self.client.table(self.table_name).select("id").eq(
                "config_hash", config_hash
            ).execute()
            
            if existing.data:
                return existing.data[0]["id"]
            
            # Deactivate current active config
            self.client.table(self.table_name).update(
                {"is_active": False}
            ).eq("is_active", True).execute()
            
            # Insert new config
            new_config = {
                "config_data": config.to_dict(),
                "config_hash": config_hash,
                "version": config.version,
                "is_active": True,
                "updated_by": updated_by,
                "change_notes": change_notes
            }
            
            result = self.client.table(self.table_name).insert(new_config).execute()
            
            # Update cache
            self._update_cache(config)
            
            return result.data[0]["id"]
        except Exception as e:
            print(f"Error saving configuration: {e}")
            raise
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration without saving."""
        try:
            validated_config = PromptOptimizationConfig.from_dict(config)
            return {
                "valid": True,
                "config": validated_config.to_dict(),
                "hash": validated_config.get_hash()
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "details": self._extract_validation_errors(e)
            }
    
    async def rollback_config(self, version_id: str, updated_by: str) -> bool:
        """Rollback to a previous configuration version."""
        try:
            # Get the target config
            target = self.client.table(self.table_name).select(
                "config_data"
            ).eq("id", version_id).single().execute()
            
            if not target.data:
                return False
            
            # Create new config as rollback
            config = PromptOptimizationConfig.from_dict(target.data["config_data"])
            rollback_id = await self.save_config(
                config, 
                updated_by, 
                f"Rollback to version {version_id}"
            )
            
            # Update rollback reference
            self.client.table(self.table_name).update(
                {"rollback_to": version_id}
            ).eq("id", rollback_id).execute()
            
            return True
        except Exception as e:
            print(f"Error rolling back configuration: {e}")
            return False
    
    async def get_config_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        try:
            result = self.client.table(self.table_name).select(
                "id, version, created_at, updated_by, change_notes, is_active"
            ).order("created_at", desc=True).limit(limit).execute()
            
            return result.data
        except Exception as e:
            print(f"Error fetching configuration history: {e}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """Check if cached configuration is still valid."""
        if not self._cached_config or not self._cache_timestamp:
            return False
        
        return datetime.utcnow() - self._cache_timestamp < self.cache_duration
    
    def _update_cache(self, config: PromptOptimizationConfig):
        """Update the cached configuration."""
        self._cached_config = config
        self._cache_timestamp = datetime.utcnow()
    
    def _extract_validation_errors(self, exception: Exception) -> List[Dict[str, str]]:
        """Extract detailed validation errors from Pydantic exceptions."""
        errors = []
        if hasattr(exception, 'errors'):
            for error in exception.errors():
                errors.append({
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
        return errors

# Singleton instance management
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(supabase_client: Client) -> ConfigurationManager:
    """Get or create configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(supabase_client)
    return _config_manager 