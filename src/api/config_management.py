"""
REST API endpoints for configuration management, monitoring, and A/B testing.
Integrates with FastAPI for the RAG CMS system.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import os

from supabase import Client, create_client
from src.config.prompt_config import (
    PromptOptimizationConfig,
    ConfigurationManager,
    QueryType,
    get_config_manager
)
from src.monitoring.prompt_analytics import PromptAnalytics
from src.monitoring.performance_profiler import PerformanceProfiler
from src.config.feature_flags import (
    FeatureFlagManager,
    FeatureFlag,
    FeatureStatus,
    FeatureVariant
)

# Pydantic models for API requests/responses
class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""
    config_data: Dict[str, Any]
    change_notes: Optional[str] = None

class ConfigValidationRequest(BaseModel):
    """Request model for configuration validation."""
    config_data: Dict[str, Any]

class FeatureFlagRequest(BaseModel):
    """Request model for feature flag creation/update."""
    name: str
    description: str
    status: str = FeatureStatus.DISABLED.value
    rollout_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    segments: Dict[str, Any] = Field(default_factory=dict)
    variants: List[Dict[str, Any]] = Field(default_factory=list)
    expires_at: Optional[datetime] = None

class ExperimentRequest(BaseModel):
    """Request model for A/B test experiment creation."""
    feature_flag_name: str
    experiment_name: str
    hypothesis: str
    success_metrics: List[str]
    duration_days: int = Field(default=14, ge=1, le=90)

class MetricTrackingRequest(BaseModel):
    """Request model for tracking metrics."""
    experiment_id: str
    variant_name: str
    metric_type: str
    metric_value: float
    user_context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class AlertAcknowledgeRequest(BaseModel):
    """Request model for acknowledging alerts."""
    alert_id: str
    acknowledged_by: str

# Create router
router = APIRouter(prefix="/api/v1/config", tags=["configuration"])

# Dependency injection for managers
async def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=500,
            detail="Supabase credentials not configured"
        )
    
    return create_client(supabase_url, supabase_key)

async def get_config_manager_dep(
    client: Client = Depends(get_supabase_client)
) -> ConfigurationManager:
    """Dependency for configuration manager."""
    return get_config_manager(client)

async def get_analytics_dep(
    client: Client = Depends(get_supabase_client)
) -> PromptAnalytics:
    """Dependency for analytics."""
    return PromptAnalytics(client)

async def get_profiler_dep(
    client: Client = Depends(get_supabase_client)
) -> PerformanceProfiler:
    """Dependency for profiler."""
    return PerformanceProfiler(client)

async def get_feature_flags_dep(
    client: Client = Depends(get_supabase_client)
) -> FeatureFlagManager:
    """Dependency for feature flags."""
    return FeatureFlagManager(client)

# Configuration Management Endpoints
@router.get("/prompt-optimization")
async def get_current_config(
    config_manager: ConfigurationManager = Depends(get_config_manager_dep)
) -> Dict[str, Any]:
    """Get current prompt optimization configuration."""
    try:
        config = await config_manager.get_active_config()
        return {
            "status": "success",
            "config": config.to_dict(),
            "hash": config.get_hash()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/prompt-optimization")
async def update_config(
    request: ConfigUpdateRequest,
    updated_by: str = Query(..., description="User making the update"),
    config_manager: ConfigurationManager = Depends(get_config_manager_dep)
) -> Dict[str, Any]:
    """Update prompt optimization configuration."""
    try:
        # Validate configuration first
        validation_result = await config_manager.validate_config(request.config_data)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid configuration",
                    "details": validation_result["details"]
                }
            )
        
        # Create new config from data
        new_config = PromptOptimizationConfig.from_dict(request.config_data)
        
        # Save configuration
        config_id = await config_manager.save_config(
            new_config,
            updated_by,
            request.change_notes
        )
        
        return {
            "status": "success",
            "config_id": config_id,
            "message": "Configuration updated successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompt-optimization/validate")
async def validate_config(
    request: ConfigValidationRequest,
    config_manager: ConfigurationManager = Depends(get_config_manager_dep)
) -> Dict[str, Any]:
    """Validate configuration without saving."""
    try:
        result = await config_manager.validate_config(request.config_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompt-optimization/history")
async def get_config_history(
    limit: int = Query(default=10, ge=1, le=100),
    config_manager: ConfigurationManager = Depends(get_config_manager_dep)
) -> Dict[str, Any]:
    """Get configuration change history."""
    try:
        history = await config_manager.get_config_history(limit)
        return {
            "status": "success",
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prompt-optimization/rollback/{version_id}")
async def rollback_config(
    version_id: str,
    updated_by: str = Query(..., description="User performing rollback"),
    config_manager: ConfigurationManager = Depends(get_config_manager_dep)
) -> Dict[str, Any]:
    """Rollback to a previous configuration version."""
    try:
        success = await config_manager.rollback_config(version_id, updated_by)
        if success:
            return {
                "status": "success",
                "message": f"Successfully rolled back to version {version_id}"
            }
        else:
            raise HTTPException(status_code=404, detail="Version not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring and Analytics Endpoints
@router.get("/analytics/real-time")
async def get_real_time_metrics(
    window_minutes: int = Query(default=5, ge=1, le=60),
    analytics: PromptAnalytics = Depends(get_analytics_dep)
) -> Dict[str, Any]:
    """Get real-time performance metrics."""
    try:
        metrics = await analytics.get_real_time_metrics(window_minutes)
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/alerts")
async def get_active_alerts(
    analytics: PromptAnalytics = Depends(get_analytics_dep)
) -> Dict[str, Any]:
    """Get all active alerts."""
    try:
        alerts = await analytics.get_active_alerts()
        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/alerts/acknowledge")
async def acknowledge_alert(
    request: AlertAcknowledgeRequest,
    analytics: PromptAnalytics = Depends(get_analytics_dep)
) -> Dict[str, Any]:
    """Acknowledge an alert."""
    try:
        success = await analytics.acknowledge_alert(
            request.alert_id,
            request.acknowledged_by
        )
        if success:
            return {
                "status": "success",
                "message": "Alert acknowledged"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/report")
async def generate_performance_report(
    start_date: datetime = Query(..., description="Report start date"),
    end_date: datetime = Query(..., description="Report end date"),
    analytics: PromptAnalytics = Depends(get_analytics_dep)
) -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    try:
        if end_date <= start_date:
            raise HTTPException(
                status_code=400,
                detail="End date must be after start date"
            )
        
        report = await analytics.create_performance_report(start_date, end_date)
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Performance Profiling Endpoints
@router.get("/profiling/optimization-report")
async def get_optimization_report(
    hours: int = Query(default=24, ge=1, le=168),
    profiler: PerformanceProfiler = Depends(get_profiler_dep)
) -> Dict[str, Any]:
    """Get performance optimization report."""
    try:
        report = await profiler.get_optimization_report(hours)
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiling/snapshot")
async def get_performance_snapshot(
    profiler: PerformanceProfiler = Depends(get_profiler_dep)
) -> Dict[str, Any]:
    """Get current performance snapshot."""
    try:
        snapshot = await profiler.capture_performance_snapshot()
        return {
            "status": "success",
            "snapshot": {
                "timestamp": snapshot.timestamp.isoformat(),
                "active_operations": snapshot.active_operations,
                "memory_usage_mb": snapshot.memory_usage_mb,
                "cpu_percent": snapshot.cpu_percent,
                "pending_tasks": snapshot.pending_tasks,
                "avg_response_time_ms": snapshot.avg_response_time_ms
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feature Flag Endpoints
@router.get("/feature-flags")
async def list_feature_flags(
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """List all feature flags."""
    try:
        flags = await feature_flags.list_feature_flags()
        return {
            "status": "success",
            "flags": [
                {
                    "name": flag.name,
                    "description": flag.description,
                    "status": flag.status.value,
                    "rollout_percentage": flag.rollout_percentage,
                    "created_at": flag.created_at.isoformat() if flag.created_at else None,
                    "expires_at": flag.expires_at.isoformat() if flag.expires_at else None
                }
                for flag in flags
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-flags/{flag_name}")
async def get_feature_flag(
    flag_name: str,
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """Get a specific feature flag."""
    try:
        flag = await feature_flags.get_feature_flag(flag_name)
        if flag:
            return {
                "status": "success",
                "flag": {
                    "name": flag.name,
                    "description": flag.description,
                    "status": flag.status.value,
                    "rollout_percentage": flag.rollout_percentage,
                    "segments": flag.segments,
                    "variants": [
                        {
                            "name": v.name,
                            "weight": v.weight,
                            "config_overrides": v.config_overrides
                        }
                        for v in flag.variants
                    ],
                    "expires_at": flag.expires_at.isoformat() if flag.expires_at else None
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Feature flag not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feature-flags")
async def create_feature_flag(
    request: FeatureFlagRequest,
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """Create a new feature flag."""
    try:
        # Create FeatureFlag object
        flag = FeatureFlag(
            name=request.name,
            description=request.description,
            status=FeatureStatus(request.status),
            rollout_percentage=request.rollout_percentage,
            segments=request.segments,
            variants=[FeatureVariant(**v) for v in request.variants],
            expires_at=request.expires_at
        )
        
        flag_id = await feature_flags.create_feature_flag(flag)
        
        return {
            "status": "success",
            "flag_id": flag_id,
            "message": "Feature flag created successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/feature-flags/{flag_name}")
async def update_feature_flag(
    flag_name: str,
    updates: Dict[str, Any] = Body(...),
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """Update a feature flag."""
    try:
        success = await feature_flags.update_feature_flag(flag_name, updates)
        if success:
            return {
                "status": "success",
                "message": "Feature flag updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Feature flag not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# A/B Testing Endpoints
@router.post("/experiments")
async def create_experiment(
    request: ExperimentRequest,
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """Create a new A/B test experiment."""
    try:
        experiment_id = await feature_flags.create_experiment(
            request.feature_flag_name,
            request.experiment_name,
            request.hypothesis,
            request.success_metrics,
            request.duration_days
        )
        
        return {
            "status": "success",
            "experiment_id": experiment_id,
            "message": "Experiment created successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/experiments/track-metric")
async def track_experiment_metric(
    request: MetricTrackingRequest,
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """Track a metric for an experiment."""
    try:
        await feature_flags.track_metric(
            request.experiment_id,
            request.variant_name,
            request.metric_type,
            request.metric_value,
            request.user_context,
            request.metadata
        )
        
        return {
            "status": "success",
            "message": "Metric tracked successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    feature_flags: FeatureFlagManager = Depends(get_feature_flags_dep)
) -> Dict[str, Any]:
    """Get results for an A/B test experiment."""
    try:
        results = await feature_flags.get_experiment_results(experiment_id)
        return {
            "status": "success",
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# WebSocket endpoint for real-time monitoring
@router.websocket("/ws/metrics")
async def websocket_metrics(
    websocket: WebSocket,
    analytics: PromptAnalytics = Depends(get_analytics_dep)
):
    """WebSocket endpoint for real-time metrics streaming."""
    await websocket.accept()
    
    try:
        while True:
            # Send metrics every 5 seconds
            metrics = await analytics.get_real_time_metrics(window_minutes=1)
            await websocket.send_json({
                "type": "metrics_update",
                "data": metrics,
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1000, reason=str(e)) 