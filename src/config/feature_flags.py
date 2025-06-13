"""
Feature Flags and A/B Testing Infrastructure
Enables gradual rollout and experimentation with RAG system features.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import json
import random
from abc import ABC, abstractmethod
import numpy as np
from supabase import Client

# Handle scipy import with graceful fallback
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Provide simple fallback for basic statistical functions
    class stats:
        class norm:
            @staticmethod
            def cdf(x):
                # Simple approximation for normal CDF
                return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi))) 

class FeatureStatus(str, Enum):
    """Feature flag status options."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    GRADUAL_ROLLOUT = "gradual_rollout"
    AB_TEST = "ab_test"
    CANARY = "canary"

class SegmentationType(str, Enum):
    """User segmentation strategies."""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    USER_ATTRIBUTE = "user_attribute"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"

@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    name: str
    description: str
    status: FeatureStatus
    rollout_percentage: float = 0.0
    segments: Dict[str, Any] = field(default_factory=dict)
    variants: List['FeatureVariant'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if feature flag has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

@dataclass
class FeatureVariant:
    """A/B test variant configuration."""
    name: str
    weight: float  # 0-100
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentMetrics:
    """Metrics tracked for A/B testing."""
    variant_name: str
    impressions: int = 0
    conversions: int = 0
    errors: int = 0
    avg_response_time_ms: float = 0.0
    avg_quality_score: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        return self.conversions / self.impressions if self.impressions > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.errors / self.impressions if self.impressions > 0 else 0.0

class SegmentationStrategy(ABC):
    """Base class for user segmentation strategies."""
    
    @abstractmethod
    def should_enable_feature(self, user_context: Dict[str, Any], percentage: float) -> bool:
        """Determine if feature should be enabled for user."""
        pass
    
    @abstractmethod
    def get_variant(self, user_context: Dict[str, Any], variants: List[FeatureVariant]) -> FeatureVariant:
        """Select variant for user."""
        pass

class HashBasedSegmentation(SegmentationStrategy):
    """Deterministic hash-based segmentation."""
    
    def __init__(self, salt: str = "rag-cms-features"):
        self.salt = salt
    
    def _get_user_hash(self, user_context: Dict[str, Any]) -> float:
        """Generate consistent hash for user."""
        user_id = user_context.get("user_id", user_context.get("session_id", "anonymous"))
        hash_input = f"{self.salt}:{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        return (hash_value % 10000) / 100.0  # 0-100 range
    
    def should_enable_feature(self, user_context: Dict[str, Any], percentage: float) -> bool:
        """Check if user falls within rollout percentage."""
        user_hash = self._get_user_hash(user_context)
        return user_hash < percentage
    
    def get_variant(self, user_context: Dict[str, Any], variants: List[FeatureVariant]) -> FeatureVariant:
        """Select variant based on user hash."""
        if not variants:
            raise ValueError("No variants configured")
        
        user_hash = self._get_user_hash(user_context)
        cumulative_weight = 0.0
        
        for variant in variants:
            cumulative_weight += variant.weight
            if user_hash < cumulative_weight:
                return variant
        
        return variants[-1]  # Fallback to last variant

class RandomSegmentation(SegmentationStrategy):
    """Random segmentation for each request."""
    
    def should_enable_feature(self, user_context: Dict[str, Any], percentage: float) -> bool:
        """Randomly decide feature enablement."""
        return random.random() * 100 < percentage
    
    def get_variant(self, user_context: Dict[str, Any], variants: List[FeatureVariant]) -> FeatureVariant:
        """Randomly select variant based on weights."""
        if not variants:
            raise ValueError("No variants configured")
        
        weights = [v.weight for v in variants]
        return random.choices(variants, weights=weights)[0]

class FeatureFlagManager:
    """Manages feature flags and A/B testing."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.flags_table = "feature_flags"
        self.experiments_table = "ab_test_experiments"
        self.metrics_table = "ab_test_metrics"
        
        # Segmentation strategies
        self.segmentation_strategies = {
            SegmentationType.HASH_BASED: HashBasedSegmentation(),
            SegmentationType.RANDOM: RandomSegmentation()
        }
        
        # Cache for feature flags
        self._flag_cache: Dict[str, FeatureFlag] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update = datetime.utcnow()
        
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self):
        """Ensure feature flag tables exist in Supabase.
        
        Note: Actual table creation is handled by migration files.
        This method is kept for compatibility but tables should be
        created via migrations/004_create_feature_flags_tables.sql
        """
        # Tables are created via migration files
        # See migrations/004_create_feature_flags_tables.sql
        pass
    
    async def is_feature_enabled(
        self,
        feature_name: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if a feature is enabled for the given user context."""
        flag = await self.get_feature_flag(feature_name)
        
        if not flag or flag.is_expired():
            return False
        
        if flag.status == FeatureStatus.DISABLED:
            return False
        
        if flag.status == FeatureStatus.ENABLED:
            return True
        
        # For gradual rollout or A/B testing
        if flag.status in [FeatureStatus.GRADUAL_ROLLOUT, FeatureStatus.AB_TEST]:
            user_context = user_context or {}
            strategy = self._get_segmentation_strategy(flag)
            return strategy.should_enable_feature(user_context, flag.rollout_percentage)
        
        return False
    
    async def get_variant(
        self,
        feature_name: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[FeatureVariant]:
        """Get the variant for a user in an A/B test."""
        flag = await self.get_feature_flag(feature_name)
        
        if not flag or flag.status != FeatureStatus.AB_TEST or not flag.variants:
            return None
        
        user_context = user_context or {}
        strategy = self._get_segmentation_strategy(flag)
        
        # Check if user is in the test
        if not strategy.should_enable_feature(user_context, flag.rollout_percentage):
            return None
        
        return strategy.get_variant(user_context, flag.variants)
    
    async def get_feature_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name."""
        # Check cache first
        if self._is_cache_valid() and name in self._flag_cache:
            return self._flag_cache[name]
        
        # Fetch from database
        result = self.client.table(self.flags_table).select("*").eq("name", name).single().execute()
        
        if result.data:
            flag = self._parse_feature_flag(result.data)
            self._flag_cache[name] = flag
            return flag
        
        return None
    
    async def list_feature_flags(self) -> List[FeatureFlag]:
        """List all feature flags."""
        try:
            result = self.client.table(self.flags_table).select("*").order("created_at", desc=True).execute()
            return [self._parse_feature_flag(flag_data) for flag_data in result.data]
        except Exception as e:
            print(f"Error listing feature flags: {e}")
            return []
    
    async def create_feature_flag(self, flag: FeatureFlag) -> str:
        """Create a new feature flag."""
        flag_data = {
            "name": flag.name,
            "description": flag.description,
            "status": flag.status.value,
            "rollout_percentage": flag.rollout_percentage,
            "segments": flag.segments,
            "variants": [self._variant_to_dict(v) for v in flag.variants],
            "metadata": flag.metadata,
            "expires_at": flag.expires_at.isoformat() if flag.expires_at else None
        }
        
        result = self.client.table(self.flags_table).insert(flag_data).execute()
        
        # Clear cache
        self._flag_cache.pop(flag.name, None)
        
        return result.data[0]["id"]
    
    async def update_feature_flag(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing feature flag."""
        # Validate status if provided
        if "status" in updates and isinstance(updates["status"], str):
            updates["status"] = FeatureStatus(updates["status"]).value
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        result = self.client.table(self.flags_table).update(updates).eq("name", name).execute()
        
        # Clear cache
        self._flag_cache.pop(name, None)
        
        return len(result.data) > 0
    
    async def create_experiment(
        self,
        feature_flag_name: str,
        experiment_name: str,
        hypothesis: str,
        success_metrics: List[str],
        duration_days: int = 14
    ) -> str:
        """Create a new A/B test experiment."""
        flag = await self.get_feature_flag(feature_flag_name)
        if not flag:
            raise ValueError(f"Feature flag '{feature_flag_name}' not found")
        
        experiment_data = {
            "feature_flag_id": await self._get_flag_id(feature_flag_name),
            "name": experiment_name,
            "hypothesis": hypothesis,
            "success_metrics": success_metrics,
            "end_date": (datetime.utcnow() + timedelta(days=duration_days)).isoformat()
        }
        
        result = self.client.table(self.experiments_table).insert(experiment_data).execute()
        
        return result.data[0]["id"]
    
    async def track_metric(
        self,
        experiment_id: str,
        variant_name: str,
        metric_type: str,
        metric_value: float,
        user_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track a metric for an A/B test."""
        user_context = user_context or {}
        
        metric_data = {
            "experiment_id": experiment_id,
            "variant_name": variant_name,
            "user_id": user_context.get("user_id"),
            "session_id": user_context.get("session_id"),
            "metric_type": metric_type,
            "metric_value": metric_value,
            "metadata": metadata or {}
        }
        
        self.client.table(self.metrics_table).insert(metric_data).execute()
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for an A/B test experiment."""
        # Get experiment details
        experiment = self.client.table(self.experiments_table).select("*").eq(
            "id", experiment_id
        ).single().execute()
        
        if not experiment.data:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        
        # Get metrics
        metrics_result = self.client.table(self.metrics_table).select("*").eq(
            "experiment_id", experiment_id
        ).execute()
        
        # Aggregate metrics by variant
        variant_metrics = defaultdict(lambda: ExperimentMetrics(variant_name=""))
        
        for metric in metrics_result.data:
            variant = metric["variant_name"]
            if not variant_metrics[variant].variant_name:
                variant_metrics[variant].variant_name = variant
            
            vm = variant_metrics[variant]
            
            if metric["metric_type"] == "impression":
                vm.impressions += 1
            elif metric["metric_type"] == "conversion":
                vm.conversions += 1
            elif metric["metric_type"] == "error":
                vm.errors += 1
            elif metric["metric_type"] == "response_time_ms":
                # Running average
                vm.avg_response_time_ms = (
                    (vm.avg_response_time_ms * (vm.impressions - 1) + metric["metric_value"]) 
                    / vm.impressions
                )
            elif metric["metric_type"] == "quality_score":
                vm.avg_quality_score = (
                    (vm.avg_quality_score * (vm.impressions - 1) + metric["metric_value"]) 
                    / vm.impressions
                )
            else:
                # Custom metrics
                if metric["metric_type"] not in vm.custom_metrics:
                    vm.custom_metrics[metric["metric_type"]] = []
                vm.custom_metrics[metric["metric_type"]].append(metric["metric_value"])
        
        # Calculate statistical significance
        results = {
            "experiment": experiment.data,
            "variants": {},
            "statistical_analysis": {}
        }
        
        for variant_name, metrics in variant_metrics.items():
            results["variants"][variant_name] = {
                "impressions": metrics.impressions,
                "conversions": metrics.conversions,
                "conversion_rate": metrics.conversion_rate,
                "error_rate": metrics.error_rate,
                "avg_response_time_ms": metrics.avg_response_time_ms,
                "avg_quality_score": metrics.avg_quality_score,
                "custom_metrics": {
                    k: np.mean(v) for k, v in metrics.custom_metrics.items()
                }
            }
        
        # Perform statistical tests if we have enough data
        if len(variant_metrics) == 2 and all(m.impressions > 30 for m in variant_metrics.values()):
            results["statistical_analysis"] = self._calculate_statistical_significance(
                list(variant_metrics.values())
            )
        
        return results
    
    def _calculate_statistical_significance(self, variants: List[ExperimentMetrics]) -> Dict[str, Any]:
        """Calculate statistical significance between variants."""
        if len(variants) != 2:
            return {"error": "Statistical analysis requires exactly 2 variants"}
        
        control, treatment = variants
        
        # Conversion rate significance (using normal approximation)
        control_rate = control.conversion_rate
        treatment_rate = treatment.conversion_rate
        
        # Standard error
        se_control = np.sqrt(control_rate * (1 - control_rate) / control.impressions)
        se_treatment = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment.impressions)
        se_diff = np.sqrt(se_control**2 + se_treatment**2)
        
        # Z-score
        z_score = (treatment_rate - control_rate) / se_diff if se_diff > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        margin_of_error = 1.96 * se_diff  # 95% confidence
        ci_lower = (treatment_rate - control_rate) - margin_of_error
        ci_upper = (treatment_rate - control_rate) + margin_of_error
        
        return {
            "control_conversion_rate": control_rate,
            "treatment_conversion_rate": treatment_rate,
            "relative_improvement": ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_interval_95": {
                "lower": ci_lower,
                "upper": ci_upper
            },
            "recommendation": self._get_experiment_recommendation(p_value, treatment_rate - control_rate),
            "scipy_available": SCIPY_AVAILABLE
        }
    
    def _get_experiment_recommendation(self, p_value: float, effect_size: float) -> str:
        """Generate recommendation based on experiment results."""
        if p_value < 0.05:
            if effect_size > 0:
                return "Treatment variant shows statistically significant improvement. Consider full rollout."
            else:
                return "Control variant performs significantly better. Do not roll out treatment."
        elif p_value < 0.1:
            return "Results are marginally significant. Consider extending the experiment."
        else:
            return "No significant difference detected. More data needed or effect size too small."
    
    def _get_segmentation_strategy(self, flag: FeatureFlag) -> SegmentationStrategy:
        """Get the appropriate segmentation strategy for a flag."""
        strategy_type = flag.segments.get("type", SegmentationType.HASH_BASED)
        return self.segmentation_strategies.get(
            strategy_type,
            self.segmentation_strategies[SegmentationType.HASH_BASED]
        )
    
    def _is_cache_valid(self) -> bool:
        """Check if the flag cache is still valid."""
        return datetime.utcnow() - self._last_cache_update < self._cache_ttl
    
    def _parse_feature_flag(self, data: Dict[str, Any]) -> FeatureFlag:
        """Parse feature flag from database record."""
        variants = [
            FeatureVariant(**v) for v in data.get("variants", [])
        ]
        
        return FeatureFlag(
            name=data["name"],
            description=data.get("description", ""),
            status=FeatureStatus(data["status"]),
            rollout_percentage=data.get("rollout_percentage", 0.0),
            segments=data.get("segments", {}),
            variants=variants,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        )
    
    def _variant_to_dict(self, variant: FeatureVariant) -> Dict[str, Any]:
        """Convert variant to dictionary for storage."""
        return {
            "name": variant.name,
            "weight": variant.weight,
            "config_overrides": variant.config_overrides,
            "metadata": variant.metadata
        }
    
    async def _get_flag_id(self, flag_name: str) -> str:
        """Get flag ID by name."""
        result = self.client.table(self.flags_table).select("id").eq("name", flag_name).single().execute()
        if result.data:
            return result.data["id"]
        raise ValueError(f"Feature flag '{flag_name}' not found")

# Decorator for feature flag checking
def feature_flag(flag_name: str, default: bool = False):
    """Decorator to conditionally execute functions based on feature flags."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract feature flag manager from arguments
            manager = None
            for arg in args:
                if hasattr(arg, 'feature_flag_manager'):
                    manager = arg.feature_flag_manager
                    break
            
            if manager:
                user_context = kwargs.get('user_context', {})
                if await manager.is_feature_enabled(flag_name, user_context):
                    return await func(*args, **kwargs)
            
            # Feature not enabled or manager not found
            if default:
                return await func(*args, **kwargs)
            else:
                return None
        
        return wrapper
    return decorator 