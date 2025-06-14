"""
FastAPI Router for Retrieval Configuration Management

Provides REST API endpoints for managing contextual retrieval system configuration:
- Configuration CRUD operations
- Environment-specific configuration management
- Hot-reload configuration updates
- Configuration validation and export
- Performance profile management
- Query-type specific configuration optimization
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query, Path as PathParam
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from ..config.retrieval_settings import (
    RetrievalSettings,
    ConfigurationManager,
    create_retrieval_settings,
    DeploymentEnvironment,
    PerformanceProfile,
    RetrievalStrategy
)

# API Models
class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""
    
    config_data: Dict[str, Any] = Field(
        ...,
        description="Configuration data to update"
    )
    validate_only: bool = Field(
        default=False,
        description="Only validate, don't apply changes"
    )
    force_update: bool = Field(
        default=False,
        description="Force update despite warnings"
    )

class ConfigValidationResponse(BaseModel):
    """Response model for configuration validation."""
    
    is_valid: bool = Field(description="Whether configuration is valid")
    issues: List[str] = Field(description="List of validation issues")
    warnings: List[str] = Field(description="List of validation warnings")
    recommendations: List[str] = Field(description="List of optimization recommendations")

class ConfigExportRequest(BaseModel):
    """Request model for configuration export."""
    
    environment: Optional[str] = Field(
        default=None,
        description="Environment to export configuration for"
    )
    include_sensitive: bool = Field(
        default=False,
        description="Include sensitive information in export"
    )
    format: str = Field(
        default="json",
        regex="^(json|yaml|env)$",
        description="Export format"
    )

class PerformanceProfileUpdateRequest(BaseModel):
    """Request model for performance profile updates."""
    
    profile: PerformanceProfile = Field(
        description="Performance profile to apply"
    )
    custom_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom overrides"
    )

class QueryTypeOptimizationRequest(BaseModel):
    """Request model for query-type specific optimization."""
    
    query_type: str = Field(
        description="Query type to optimize for"
    )
    sample_queries: Optional[List[str]] = Field(
        default=None,
        description="Sample queries for optimization analysis"
    )

# API Router
router = APIRouter(
    prefix="/api/v1/config",
    tags=["Configuration Management"],
    responses={
        404: {"description": "Configuration not found"},
        422: {"description": "Validation error"}
    }
)

# Configuration manager instance
config_manager = ConfigurationManager()

# Dependency for getting current configuration
async def get_current_config() -> RetrievalSettings:
    """Get the current configuration."""
    try:
        return config_manager.load_config()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load configuration: {str(e)}"
        )

@router.get("/", response_model=Dict[str, Any])
async def get_configuration(
    environment: Optional[str] = Query(
        default=None,
        description="Environment to get configuration for"
    ),
    include_sensitive: bool = Query(
        default=False,
        description="Include sensitive information"
    )
):
    """Get current retrieval configuration."""
    
    try:
        config = config_manager.load_config(environment=environment)
        config_dict = config.dict()
        
        # Remove sensitive information if not requested
        if not include_sensitive:
            sensitive_keys = ['openai_api_key', 'anthropic_api_key', 'supabase_key']
            for key in sensitive_keys:
                if key in config_dict:
                    config_dict[key] = "***REDACTED***"
                # Also check nested configs
                for section in config_dict.values():
                    if isinstance(section, dict) and key in section:
                        section[key] = "***REDACTED***"
        
        return {
            "config": config_dict,
            "environment": config.environment.value,
            "loaded_at": datetime.now().isoformat(),
            "performance_profile": config.performance_profile.value
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration: {str(e)}"
        )

@router.put("/", response_model=Dict[str, Any])
async def update_configuration(
    request: ConfigUpdateRequest,
    current_config: RetrievalSettings = Depends(get_current_config)
):
    """Update retrieval configuration."""
    
    try:
        # Create new configuration with updates
        current_dict = current_config.dict()
        current_dict.update(request.config_data)
        
        new_config = RetrievalSettings(**current_dict)
        new_config.apply_performance_profile()
        
        # Validate configuration
        issues = config_manager.validate_config(new_config)
        
        if issues and not request.force_update:
            return {
                "success": False,
                "message": "Configuration validation failed",
                "issues": issues,
                "suggestion": "Use force_update=true to apply despite issues"
            }
        
        if request.validate_only:
            return {
                "success": True,
                "message": "Configuration validation passed",
                "issues": issues,
                "config_preview": new_config.dict()
            }
        
        # Save configuration
        config_path = "config/retrieval_settings.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(new_config.dict(), f, indent=2, default=str)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "issues": issues,
            "updated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to update configuration: {str(e)}"
        )

@router.post("/validate", response_model=ConfigValidationResponse)
async def validate_configuration(
    config_data: Optional[Dict[str, Any]] = None,
    current_config: RetrievalSettings = Depends(get_current_config)
):
    """Validate configuration and provide recommendations."""
    
    try:
        # Use provided config or current config
        if config_data:
            current_dict = current_config.dict()
            current_dict.update(config_data)
            config = RetrievalSettings(**current_dict)
        else:
            config = current_config
        
        # Validate configuration
        issues = config_manager.validate_config(config)
        warnings = []
        recommendations = []
        
        # Check for warnings and recommendations
        if config.performance.max_response_time_ms > 10000:
            warnings.append("Response time threshold is very high (>10s)")
        
        if config.performance.connection_pool_size > 50:
            warnings.append("Large connection pool may consume excessive resources")
        
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.monitoring.enable_metrics_collection:
                recommendations.append("Enable metrics collection for production monitoring")
            
            if not config.api.enable_api_key_auth:
                recommendations.append("Enable API key authentication for production security")
        
        # Performance recommendations
        if config.performance_profile == PerformanceProfile.BALANCED:
            if config.performance.max_response_time_ms < 1000:
                recommendations.append("Consider SPEED_OPTIMIZED profile for sub-1s requirements")
            elif config.performance.max_response_time_ms > 5000:
                recommendations.append("Consider QUALITY_OPTIMIZED profile for >5s tolerance")
        
        return ConfigValidationResponse(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to validate configuration: {str(e)}"
        )

@router.post("/export")
async def export_configuration(
    request: ConfigExportRequest,
    current_config: RetrievalSettings = Depends(get_current_config)
):
    """Export configuration to file."""
    
    try:
        config = config_manager.load_config(environment=request.environment)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_suffix = f"_{request.environment}" if request.environment else ""
        filename = f"retrieval_config{env_suffix}_{timestamp}.{request.format}"
        output_path = f"exports/{filename}"
        
        if request.format == "json":
            config_manager.export_config(config, output_path)
        
        elif request.format == "yaml":
            import yaml
            config_dict = config.dict()
            if not request.include_sensitive:
                # Remove sensitive keys
                sensitive_keys = ['openai_api_key', 'anthropic_api_key', 'supabase_key']
                for key in sensitive_keys:
                    if key in config_dict:
                        config_dict[key] = "***REDACTED***"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, indent=2, default_flow_style=False)
        
        elif request.format == "env":
            # Generate .env format
            config_dict = config.dict()
            env_lines = []
            
            def flatten_dict(d, prefix=""):
                for key, value in d.items():
                    env_key = f"{prefix}{key.upper()}"
                    if isinstance(value, dict):
                        flatten_dict(value, f"{env_key}_")
                    else:
                        if isinstance(value, bool):
                            value = str(value).lower()
                        env_lines.append(f"{env_key}={value}")
            
            flatten_dict(config_dict, "RETRIEVAL_")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(env_lines))
        
        return FileResponse(
            path=output_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export configuration: {str(e)}"
        )

@router.put("/performance-profile", response_model=Dict[str, Any])
async def update_performance_profile(
    request: PerformanceProfileUpdateRequest,
    current_config: RetrievalSettings = Depends(get_current_config)
):
    """Update performance profile and apply optimizations."""
    
    try:
        # Update performance profile
        current_dict = current_config.dict()
        current_dict["performance_profile"] = request.profile.value
        
        # Apply custom overrides if provided
        if request.custom_overrides:
            current_dict.update(request.custom_overrides)
        
        new_config = RetrievalSettings(**current_dict)
        new_config.apply_performance_profile()
        
        # Validate new configuration
        issues = config_manager.validate_config(new_config)
        
        # Save configuration
        config_path = "config/retrieval_settings.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(new_config.dict(), f, indent=2, default=str)
        
        return {
            "success": True,
            "message": f"Performance profile updated to {request.profile.value}",
            "profile": request.profile.value,
            "validation_issues": issues,
            "updated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to update performance profile: {str(e)}"
        )

@router.get("/performance-profiles", response_model=Dict[str, Any])
async def get_performance_profiles():
    """Get available performance profiles and their descriptions."""
    
    profiles = {
        "SPEED_OPTIMIZED": {
            "description": "Optimized for fastest response times",
            "use_cases": ["Real-time chat", "Interactive applications", "High-frequency queries"]
        },
        "QUALITY_OPTIMIZED": {
            "description": "Optimized for highest result quality",
            "use_cases": ["Research applications", "Complex analysis", "High-stakes decisions"]
        },
        "BALANCED": {
            "description": "Balanced performance and quality",
            "use_cases": ["General purpose", "Mixed workloads", "Standard applications"]
        },
        "RESOURCE_OPTIMIZED": {
            "description": "Minimized resource consumption",
            "use_cases": ["Limited resources", "Cost optimization", "Light workloads"]
        }
    }
    
    return {
        "profiles": profiles,
        "current_profile": (await get_current_config()).performance_profile.value
    }

@router.post("/optimize-for-query-type", response_model=Dict[str, Any])
async def optimize_for_query_type(
    request: QueryTypeOptimizationRequest,
    current_config: RetrievalSettings = Depends(get_current_config)
):
    """Optimize configuration for specific query type."""
    
    try:
        # Get query-type specific configuration
        optimized_config = current_config.get_query_type_config(request.query_type)
        
        # Analyze sample queries if provided
        analysis_results = {}
        if request.sample_queries:
            analysis_results = _analyze_sample_queries(request.sample_queries, request.query_type)
        
        return {
            "query_type": request.query_type,
            "optimized_config": optimized_config,
            "optimizations_applied": _get_query_type_optimizations(request.query_type),
            "sample_analysis": analysis_results,
            "recommendation": f"Configuration optimized for {request.query_type} queries"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to optimize for query type: {str(e)}"
        )

@router.get("/query-types", response_model=Dict[str, Any])
async def get_supported_query_types():
    """Get supported query types and their optimizations."""
    
    query_types = {
        "factual": {
            "description": "Factual information queries",
            "optimizations": _get_query_type_optimizations("factual"),
            "examples": ["What is the capital of France?", "How does photosynthesis work?"]
        },
        "comparison": {
            "description": "Comparison and analysis queries",
            "optimizations": _get_query_type_optimizations("comparison"),
            "examples": ["Compare Python vs Java", "Pros and cons of renewable energy"]
        },
        "tutorial": {
            "description": "How-to and tutorial queries",
            "optimizations": _get_query_type_optimizations("tutorial"),
            "examples": ["How to bake a cake", "Tutorial on machine learning"]
        },
        "news": {
            "description": "News and current events queries",
            "optimizations": _get_query_type_optimizations("news"),
            "examples": ["Latest technology news", "Current market trends"]
        }
    }
    
    return {"supported_query_types": query_types}

@router.post("/reload", response_model=Dict[str, Any])
async def reload_configuration():
    """Force reload configuration from file."""
    
    try:
        current_config = await get_current_config()
        new_config = config_manager.reload_config_if_needed(current_config)
        
        config_changed = new_config != current_config
        
        return {
            "success": True,
            "config_changed": config_changed,
            "message": "Configuration reloaded successfully" if config_changed else "No configuration changes detected",
            "reloaded_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload configuration: {str(e)}"
        )

@router.get("/health", response_model=Dict[str, Any])
async def configuration_health_check():
    """Health check for configuration system."""
    
    try:
        config = await get_current_config()
        issues = config_manager.validate_config(config)
        
        health_status = "healthy" if len(issues) == 0 else "unhealthy"
        
        return {
            "status": health_status,
            "environment": config.environment.value,
            "performance_profile": config.performance_profile.value,
            "issues_count": len(issues),
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "checked_at": datetime.now().isoformat()
        }

# Helper functions
def _get_profile_optimizations(profile: PerformanceProfile) -> List[str]:
    """Get list of optimizations for a performance profile."""
    
    optimizations = {
        PerformanceProfile.SPEED_OPTIMIZED: [
            "Reduced response time threshold to 1000ms",
            "Aggressive caching strategy",
            "Reduced reranking candidates to 20",
            "Limited query variations to 2",
            "Minimal context window size"
        ],
        PerformanceProfile.QUALITY_OPTIMIZED: [
            "Extended response time threshold to 5000ms",
            "Conservative caching for freshness",
            "Increased reranking candidates to 100",
            "More query variations (up to 5)",
            "Larger context window for better understanding"
        ],
        PerformanceProfile.BALANCED: [
            "Standard response time threshold (2000ms)",
            "Adaptive caching strategy",
            "Moderate reranking (50 candidates)",
            "Standard query variations (3)",
            "Balanced context window size"
        ],
        PerformanceProfile.RESOURCE_OPTIMIZED: [
            "Reduced connection pool size",
            "Smaller batch sizes",
            "Fewer worker threads",
            "Limited reranking candidates",
            "Minimal resource consumption"
        ]
    }
    
    return optimizations.get(profile, [])

def _get_query_type_optimizations(query_type: str) -> List[str]:
    """Get list of optimizations for a query type."""
    
    optimizations = {
        "factual": [
            "Increased dense search weight (0.8)",
            "Higher relevance preference (MMR lambda 0.8)",
            "Enabled metadata filtering",
            "Focused on precision over diversity"
        ],
        "comparison": [
            "Balanced dense/sparse weights (0.6/0.4)",
            "Moderate diversity (MMR lambda 0.6)",
            "Increased query variations (4)",
            "Enhanced result diversity"
        ],
        "tutorial": [
            "Larger context window (3 sentences)",
            "Extended cache TTL (48 hours)",
            "Moderate relevance preference",
            "Optimized for sequential content"
        ],
        "news": [
            "Short cache TTL (2 hours)",
            "Enabled reranking for freshness",
            "High relevance preference (MMR lambda 0.9)",
            "Optimized for recent content"
        ]
    }
    
    return optimizations.get(query_type, [])

def _analyze_sample_queries(queries: List[str], query_type: str) -> Dict[str, Any]:
    """Analyze sample queries and provide optimization insights."""
    
    # Simple analysis - in production this could use ML models
    analysis = {
        "query_count": len(queries),
        "avg_length": sum(len(q.split()) for q in queries) / len(queries),
        "complexity_score": _estimate_query_complexity(queries),
        "recommendations": []
    }
    
    # Add recommendations based on analysis
    if analysis["avg_length"] > 20:
        analysis["recommendations"].append("Consider larger context window for complex queries")
    
    if analysis["complexity_score"] > 0.7:
        analysis["recommendations"].append("Enable advanced query expansion for complex topics")
    
    return analysis

def _estimate_query_complexity(queries: List[str]) -> float:
    """Estimate average complexity of queries (0.0 to 1.0)."""
    
    complexity_indicators = [
        "compare", "contrast", "analyze", "explain", "why", "how",
        "difference", "advantage", "disadvantage", "relationship"
    ]
    
    total_score = 0
    for query in queries:
        query_lower = query.lower()
        score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        score += len(query.split()) / 50  # Length factor
        total_score += min(score / 3, 1.0)  # Normalize to 0-1
    
    return total_score / len(queries) if queries else 0.0 