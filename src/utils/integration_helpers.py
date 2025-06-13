"""
Integration Helper Functions for RAG CMS
Provides utilities for setting up and managing the integrated monitoring and configuration system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import os
from contextlib import asynccontextmanager

from supabase import Client, create_client

# Import all necessary components
from src.config.prompt_config import (
    ConfigurationManager, get_config_manager,
    PromptOptimizationConfig, QueryType
)
from src.monitoring.prompt_analytics import PromptAnalytics
from src.monitoring.performance_profiler import PerformanceProfiler
from src.config.feature_flags import (
    FeatureFlagManager, FeatureFlag, FeatureStatus, FeatureVariant
)
from src.utils.enhanced_logging import get_logger, StructuredLogger

logger = logging.getLogger(__name__)

# Global manager instances for singleton pattern
_managers: Dict[str, Any] = {}

class IntegrationSetup:
    """Helper class for setting up the integrated RAG system."""
    
    @staticmethod
    async def initialize_all_systems(
        supabase_url: str,
        supabase_key: str,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize all integrated systems with a single call.
        
        Returns:
            Dict containing all initialized managers
        """
        
        # Create Supabase client
        supabase_client = create_client(supabase_url, supabase_key)
        
        # Initialize configuration manager
        config_manager = get_config_manager(supabase_client)
        
        # Initialize analytics
        analytics = PromptAnalytics(supabase_client)
        
        # Initialize profiler
        profiler = PerformanceProfiler(supabase_client, enable_profiling=False)
        
        # Initialize feature flags
        feature_flags = FeatureFlagManager(supabase_client)
        
        # Initialize logger
        structured_logger = get_logger("rag_cms", supabase_client)
        
        # Apply initial configuration
        if config_overrides:
            config = PromptOptimizationConfig.from_dict(config_overrides)
            await config_manager.save_config(
                config,
                "System",
                "Initial configuration with overrides"
            )
        
        # Store managers globally
        managers = {
            'supabase_client': supabase_client,
            'config_manager': config_manager,
            'analytics': analytics,
            'profiler': profiler,
            'feature_flags': feature_flags,
            'logger': structured_logger
        }
        
        # Update global registry
        _managers.update(managers)
        
        logger.info("All integrated systems initialized successfully")
        
        return managers
    
    @staticmethod
    async def create_default_configuration() -> PromptOptimizationConfig:
        """Create default configuration for RAG system."""
        
        config = PromptOptimizationConfig()
        
        # Customize default values for RAG CMS
        config.query_classification.confidence_threshold = 0.75
        config.context_formatting.max_context_length = 4000
        config.cache_config.casino_review_ttl = 48
        config.cache_config.news_ttl = 2
        config.performance.response_time_warning_ms = 2000
        config.feature_flags.enable_contextual_retrieval = True
        config.feature_flags.enable_hybrid_search = False  # Start disabled
        
        return config
    
    @staticmethod
    async def setup_default_feature_flags(feature_flags: FeatureFlagManager) -> List[str]:
        """Create default feature flags for the system."""
        
        flags_created = []
        
        # Define default feature flags
        default_flags = [
            FeatureFlag(
                name="enable_hybrid_search",
                description="Enable hybrid search combining vector and keyword search",
                status=FeatureStatus.GRADUAL_ROLLOUT,
                rollout_percentage=20.0,
                variants=[
                    FeatureVariant(name="control", weight=80.0),
                    FeatureVariant(name="hybrid", weight=20.0)
                ]
            ),
            FeatureFlag(
                name="enable_advanced_prompts",
                description="Enable advanced prompt optimization system",
                status=FeatureStatus.DISABLED,
                rollout_percentage=0.0
            ),
            FeatureFlag(
                name="enable_streaming_responses",
                description="Enable streaming response generation",
                status=FeatureStatus.AB_TEST,
                rollout_percentage=50.0,
                variants=[
                    FeatureVariant(name="batch", weight=50.0),
                    FeatureVariant(name="streaming", weight=50.0)
                ]
            ),
            FeatureFlag(
                name="enable_multilingual_support",
                description="Enable multilingual query support",
                status=FeatureStatus.DISABLED,
                rollout_percentage=0.0
            )
        ]
        
        # Create flags
        for flag in default_flags:
            try:
                flag_id = await feature_flags.create_feature_flag(flag)
                flags_created.append(flag.name)
                logger.info(f"Created feature flag: {flag.name}")
            except Exception as e:
                logger.warning(f"Failed to create flag {flag.name}: {e}")
        
        return flags_created


class DependencyInjector:
    """Dependency injection utilities for the integrated system."""
    
    @staticmethod
    def get_managers() -> Dict[str, Any]:
        """Get all initialized managers."""
        return _managers
    
    @staticmethod
    def get_manager(manager_name: str) -> Optional[Any]:
        """Get a specific manager by name."""
        return _managers.get(manager_name)
    
    @staticmethod
    def inject_managers(func):
        """Decorator to inject managers into function arguments."""
        async def wrapper(*args, **kwargs):
            # Inject managers into kwargs
            kwargs['managers'] = _managers
            return await func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    @asynccontextmanager
    async def managed_chain_context(supabase_client: Client):
        """Context manager for managed chain execution."""
        
        # Initialize managers
        managers = await IntegrationSetup.initialize_all_systems(
            supabase_client.url,
            supabase_client.options.headers.get('apikey')
        )
        
        try:
            yield managers
        finally:
            # Cleanup if needed
            pass


class ConfigurationValidator:
    """Validate configuration for the integrated system."""
    
    @staticmethod
    def validate_integration_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate integration configuration.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        
        issues = []
        
        # Check required fields
        required_fields = [
            'supabase_url',
            'supabase_key',
            'enable_monitoring',
            'enable_profiling',
            'enable_feature_flags'
        ]
        
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Validate Supabase configuration
        if 'supabase_url' in config:
            url = config['supabase_url']
            if not url.startswith('http'):
                issues.append("Supabase URL must start with http:// or https://")
        
        # Validate feature toggles
        for toggle in ['enable_monitoring', 'enable_profiling', 'enable_feature_flags']:
            if toggle in config and not isinstance(config[toggle], bool):
                issues.append(f"{toggle} must be a boolean value")
        
        # Validate optional configurations
        if 'cache_ttl_hours' in config:
            ttl = config['cache_ttl_hours']
            if not isinstance(ttl, (int, float)) or ttl <= 0:
                issues.append("cache_ttl_hours must be a positive number")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_runtime_environment() -> Tuple[bool, List[str]]:
        """Validate runtime environment for integration."""
        
        issues = []
        
        # Check environment variables
        required_env_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
        
        for var in required_env_vars:
            if not os.getenv(var):
                issues.append(f"Missing environment variable: {var}")
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            issues.append("Python 3.8 or higher is required")
        
        return len(issues) == 0, issues


class IntegrationHealthChecker:
    """Health check utilities for the integrated system."""
    
    @staticmethod
    async def check_all_systems(managers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform health checks on all integrated systems.
        
        Returns:
            Dict with health status for each system
        """
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': 'healthy',
            'systems': {}
        }
        
        # Check Supabase connection
        try:
            supabase_client = managers.get('supabase_client')
            if supabase_client:
                # Simple query to test connection
                result = supabase_client.table('prompt_configurations').select('id').limit(1).execute()
                health_status['systems']['supabase'] = {
                    'status': 'healthy',
                    'message': 'Connected successfully'
                }
            else:
                health_status['systems']['supabase'] = {
                    'status': 'unavailable',
                    'message': 'Client not initialized'
                }
        except Exception as e:
            health_status['systems']['supabase'] = {
                'status': 'unhealthy',
                'message': str(e)
            }
            health_status['overall_health'] = 'degraded'
        
        # Check Configuration Manager
        try:
            config_manager = managers.get('config_manager')
            if config_manager:
                config = await config_manager.get_active_config()
                health_status['systems']['configuration'] = {
                    'status': 'healthy',
                    'message': f'Active config version: {config.version}'
                }
            else:
                health_status['systems']['configuration'] = {
                    'status': 'unavailable',
                    'message': 'Manager not initialized'
                }
        except Exception as e:
            health_status['systems']['configuration'] = {
                'status': 'unhealthy',
                'message': str(e)
            }
            health_status['overall_health'] = 'degraded'
        
        # Check Analytics
        try:
            analytics = managers.get('analytics')
            if analytics:
                metrics = await analytics.get_real_time_metrics(window_minutes=1)
                health_status['systems']['analytics'] = {
                    'status': 'healthy',
                    'message': 'Collecting metrics successfully'
                }
            else:
                health_status['systems']['analytics'] = {
                    'status': 'unavailable',
                    'message': 'Analytics not initialized'
                }
        except Exception as e:
            health_status['systems']['analytics'] = {
                'status': 'unhealthy',
                'message': str(e)
            }
            health_status['overall_health'] = 'degraded'
        
        # Check Feature Flags
        try:
            feature_flags = managers.get('feature_flags')
            if feature_flags:
                # Try to check a flag
                test_flag = await feature_flags.is_feature_enabled('test_flag', {})
                health_status['systems']['feature_flags'] = {
                    'status': 'healthy',
                    'message': 'Feature flags operational'
                }
            else:
                health_status['systems']['feature_flags'] = {
                    'status': 'unavailable',
                    'message': 'Feature flags not initialized'
                }
        except Exception as e:
            health_status['systems']['feature_flags'] = {
                'status': 'unhealthy',
                'message': str(e)
            }
            health_status['overall_health'] = 'degraded'
        
        return health_status


class MigrationHelper:
    """Helper for migrating from UniversalRAGChain to IntegratedRAGChain."""
    
    @staticmethod
    def create_migration_guide() -> Dict[str, Any]:
        """Create a migration guide for upgrading existing code."""
        
        return {
            'migration_steps': [
                {
                    'step': 1,
                    'description': 'Update imports',
                    'old_code': 'from src.chains.universal_rag_lcel import UniversalRAGChain',
                    'new_code': 'from src.chains.integrated_rag_chain import IntegratedRAGChain'
                },
                {
                    'step': 2,
                    'description': 'Add Supabase client',
                    'old_code': 'chain = UniversalRAGChain(model_name="gpt-4")',
                    'new_code': '''
supabase_client = create_client(url, key)
chain = IntegratedRAGChain(
    model_name="gpt-4",
    supabase_client=supabase_client,
    enable_monitoring=True
)'''
                },
                {
                    'step': 3,
                    'description': 'Update method calls',
                    'old_code': 'response = await chain.ainvoke(query)',
                    'new_code': '''
response = await chain.ainvoke(
    query,
    user_context={"user_id": "user123"}  # Optional
)'''
                }
            ],
            'new_features': [
                'Automatic performance tracking',
                'Dynamic configuration updates',
                'A/B testing support',
                'Real-time monitoring',
                'Feature flag control'
            ],
            'backward_compatibility': True,
            'breaking_changes': []
        }
    
    @staticmethod
    async def migrate_chain_instance(
        old_chain: Any,
        supabase_client: Client
    ) -> 'IntegratedRAGChain':
        """Migrate an existing UniversalRAGChain instance to IntegratedRAGChain."""
        
        from src.chains.integrated_rag_chain import IntegratedRAGChain
        
        # Extract configuration from old chain
        config = {
            'model_name': getattr(old_chain, 'model_name', 'gpt-4'),
            'temperature': getattr(old_chain, 'temperature', 0.1),
            'enable_caching': getattr(old_chain, 'enable_caching', True),
            'enable_contextual_retrieval': getattr(old_chain, 'enable_contextual_retrieval', True),
            'enable_prompt_optimization': getattr(old_chain, 'enable_prompt_optimization', False),
            'enable_enhanced_confidence': getattr(old_chain, 'enable_enhanced_confidence', True),
            'vector_store': getattr(old_chain, 'vector_store', None)
        }
        
        # Create new integrated chain
        new_chain = IntegratedRAGChain(
            **config,
            supabase_client=supabase_client,
            enable_monitoring=True,
            enable_profiling=False,  # Start with profiling disabled
            enable_feature_flags=True,
            enable_configuration=True
        )
        
        logger.info("Successfully migrated chain to IntegratedRAGChain")
        
        return new_chain


class PerformanceOptimizer:
    """Utilities for optimizing integrated system performance."""
    
    @staticmethod
    async def analyze_and_optimize(
        analytics: PromptAnalytics,
        profiler: PerformanceProfiler,
        config_manager: ConfigurationManager
    ) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations."""
        
        # Get performance metrics
        metrics = await analytics.get_real_time_metrics(window_minutes=60)
        optimization_report = await profiler.get_optimization_report(hours=24)
        
        suggestions = []
        
        # Analyze response times
        avg_response_time = metrics.get('performance', {}).get('avg_response_time_ms', 0)
        if avg_response_time > 3000:
            suggestions.append({
                'issue': 'High average response time',
                'suggestion': 'Enable more aggressive caching or reduce context size',
                'config_change': {
                    'context_formatting.max_context_length': 3000,
                    'cache_config.general_ttl': 48
                }
            })
        
        # Analyze cache performance
        cache_hit_rate = metrics.get('cache', {}).get('overall_hit_rate', 0)
        if cache_hit_rate < 0.3:
            suggestions.append({
                'issue': 'Low cache hit rate',
                'suggestion': 'Increase cache TTL or improve cache key generation',
                'config_change': {
                    'feature_flags.enable_semantic_cache': True,
                    'cache_config.general_ttl': 72
                }
            })
        
        # Analyze quality scores
        avg_quality = metrics.get('quality', {}).get('avg_quality_score', 0)
        if avg_quality < 0.7:
            suggestions.append({
                'issue': 'Low average quality scores',
                'suggestion': 'Enable contextual retrieval or increase retrieval count',
                'config_change': {
                    'feature_flags.enable_contextual_retrieval': True,
                    'context_formatting.max_chunks_per_source': 5
                }
            })
        
        # Apply suggested optimizations if requested
        optimizations_applied = []
        if suggestions and config_manager:
            # Get current config
            current_config = await config_manager.get_active_config()
            config_dict = current_config.to_dict()
            
            # Apply high-priority suggestions
            for suggestion in suggestions[:2]:  # Apply top 2 suggestions
                for key, value in suggestion['config_change'].items():
                    keys = key.split('.')
                    target = config_dict
                    for k in keys[:-1]:
                        target = target[k]
                    target[keys[-1]] = value
                    optimizations_applied.append(f"{key} = {value}")
            
            # Save optimized config
            if optimizations_applied:
                optimized_config = PromptOptimizationConfig.from_dict(config_dict)
                await config_manager.save_config(
                    optimized_config,
                    "PerformanceOptimizer",
                    f"Applied optimizations: {', '.join(optimizations_applied)}"
                )
        
        return {
            'current_metrics': metrics,
            'suggestions': suggestions,
            'optimizations_applied': optimizations_applied,
            'bottlenecks': optimization_report.get('top_bottlenecks', [])
        }


# Convenience functions for quick setup
async def quick_setup_integrated_rag(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None
) -> Dict[str, Any]:
    """Quick setup function for integrated RAG system."""
    
    # Use environment variables if not provided
    supabase_url = supabase_url or os.getenv('SUPABASE_URL')
    supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials required")
    
    # Initialize all systems
    managers = await IntegrationSetup.initialize_all_systems(
        supabase_url, supabase_key
    )
    
    # Create default configuration
    config = await IntegrationSetup.create_default_configuration()
    await managers['config_manager'].save_config(
        config, "System", "Default configuration"
    )
    
    # Setup default feature flags
    await IntegrationSetup.setup_default_feature_flags(
        managers['feature_flags']
    )
    
    # Perform health check
    health = await IntegrationHealthChecker.check_all_systems(managers)
    
    logger.info(f"Quick setup completed. Health status: {health['overall_health']}")
    
    return managers


def get_integrated_chain_from_managers(managers: Dict[str, Any], **kwargs):
    """Create an integrated chain from initialized managers."""
    
    from src.chains.integrated_rag_chain import IntegratedRAGChain
    
    return IntegratedRAGChain(
        supabase_client=managers['supabase_client'],
        enable_monitoring=True,
        enable_profiling=kwargs.get('enable_profiling', False),
        enable_feature_flags=True,
        enable_configuration=True,
        **kwargs
    ) 