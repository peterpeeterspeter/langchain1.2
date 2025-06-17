"""
Integrated Universal RAG Chain with Full Monitoring and Configuration
Extends the existing UniversalRAGChain with comprehensive monitoring, configuration, and feature flags.

This implementation provides:
- Runtime configuration management
- Real-time performance monitoring
- Feature flag-based gradual rollout
- Comprehensive profiling and analytics
- Backward compatibility with existing code
"""

import asyncio
import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import hashlib
import json

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

# Import the existing Universal RAG Chain
from .universal_rag_lcel import (
    UniversalRAGChain, RAGResponse, RAGMetricsCallback,
    RetrievalException, GenerationException, ValidationException,
    EnhancedVectorStore, QueryAwareCache
)

# Import configuration and monitoring components from Task 2.5
from config.prompt_config import (
    ConfigurationManager, get_config_manager,
    PromptOptimizationConfig, QueryType as ConfigQueryType
)
from monitoring.prompt_analytics import (
    PromptAnalytics, track_query_metrics, QueryMetrics
)
from monitoring.performance_profiler import (
    PerformanceProfiler, profile_operation
)
from config.feature_flags import (
    FeatureFlagManager, FeatureFlag, FeatureStatus,
    feature_flag
)
from utils.enhanced_logging import (
    get_logger, StructuredLogger, LogCategory, LogContext,
    RAGPipelineLogger
)

# Import advanced prompt system if available
try:
    from .advanced_prompt_system import QueryType as PromptQueryType
except ImportError:
    PromptQueryType = None

logger = logging.getLogger(__name__)

class IntegratedRAGChain(UniversalRAGChain):
    """
    Enhanced Universal RAG Chain with integrated monitoring, configuration, and feature flags.
    
    This class extends UniversalRAGChain to add:
    - Dynamic configuration management
    - Comprehensive performance monitoring
    - Feature flag-based control
    - Real-time analytics and profiling
    - Enhanced logging and observability
    """
    
    def __init__(
        self,
        # Original parameters
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        enable_caching: bool = True,
        enable_contextual_retrieval: bool = True,
        enable_prompt_optimization: bool = False,
        enable_enhanced_confidence: bool = True,
        vector_store = None,
        
        # New integrated parameters
        supabase_client = None,
        enable_monitoring: bool = True,
        enable_profiling: bool = False,
        enable_feature_flags: bool = True,
        enable_configuration: bool = True,
        config_override: Optional[Dict[str, Any]] = None,
        
        **kwargs
    ):
        """
        Initialize the Integrated RAG Chain with monitoring and configuration.
        
        Args:
            supabase_client: Supabase client for persistence
            enable_monitoring: Enable analytics and monitoring
            enable_profiling: Enable performance profiling
            enable_feature_flags: Enable feature flag support
            enable_configuration: Enable dynamic configuration
            config_override: Override specific configuration values
        """
        
        # Store integration settings
        self.supabase_client = supabase_client
        self.enable_monitoring = enable_monitoring
        self.enable_profiling = enable_profiling
        self.enable_feature_flags = enable_feature_flags
        self.enable_configuration = enable_configuration
        
        # Initialize managers
        self._init_managers()
        
        # Apply configuration if enabled
        if self.enable_configuration and self.config_manager:
            config = self._apply_configuration(config_override)
            
            # Override initialization parameters from configuration
            temperature = config.query_classification.confidence_threshold
            enable_caching = config.cache_config.general_ttl > 0
            enable_contextual_retrieval = config.feature_flags.enable_contextual_retrieval
            enable_prompt_optimization = config.feature_flags.enable_query_expansion
            enable_enhanced_confidence = config.feature_flags.enable_response_caching
        
        # Initialize parent class with potentially modified parameters
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            enable_caching=enable_caching,
            enable_contextual_retrieval=enable_contextual_retrieval,
            enable_prompt_optimization=enable_prompt_optimization,
            enable_enhanced_confidence=enable_enhanced_confidence,
            vector_store=vector_store,
            **kwargs
        )
        
        # Create integrated chain with monitoring
        self.chain = self._create_integrated_chain()
        
        logger.info(
            f"IntegratedRAGChain initialized with monitoring={enable_monitoring}, "
            f"profiling={enable_profiling}, feature_flags={enable_feature_flags}"
        )
    
    def _init_managers(self):
        """Initialize configuration, monitoring, and feature flag managers."""
        
        # Configuration Manager
        if self.enable_configuration and self.supabase_client:
            self.config_manager = get_config_manager(self.supabase_client)
        else:
            self.config_manager = None
        
        # Analytics Manager
        if self.enable_monitoring and self.supabase_client:
            self.analytics = PromptAnalytics(self.supabase_client)
        else:
            self.analytics = None
        
        # Performance Profiler
        if self.enable_profiling and self.supabase_client:
            self.profiler = PerformanceProfiler(self.supabase_client, enable_profiling=True)
        else:
            self.profiler = None
        
        # Feature Flag Manager
        if self.enable_feature_flags and self.supabase_client:
            self.feature_flags = FeatureFlagManager(self.supabase_client)
        else:
            self.feature_flags = None
        
        # Enhanced Logger
        if self.supabase_client:
            self.logger = get_logger("integrated_rag_chain", self.supabase_client)
            self.pipeline_logger = RAGPipelineLogger(self.logger)
        else:
            self.logger = get_logger("integrated_rag_chain")
            self.pipeline_logger = RAGPipelineLogger(self.logger)
    
    def _apply_configuration(self, config_override: Optional[Dict[str, Any]] = None) -> PromptOptimizationConfig:
        """Apply configuration from ConfigurationManager."""
        try:
            # Get active configuration
            config = asyncio.run(self.config_manager.get_active_config())
            
            # Apply any overrides
            if config_override:
                config_dict = config.to_dict()
                config_dict.update(config_override)
                config = PromptOptimizationConfig.from_dict(config_dict)
            
            # Apply configuration to instance
            self._apply_config_to_instance(config)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to apply configuration: {e}")
            return PromptOptimizationConfig()  # Return default config
    
    def _apply_config_to_instance(self, config: PromptOptimizationConfig):
        """Apply configuration values to instance attributes."""
        
        # Apply query classification settings
        if hasattr(self, 'confidence_threshold'):
            self.confidence_threshold = config.query_classification.confidence_threshold
        
        # Apply context formatting settings
        if hasattr(self, 'max_context_length'):
            self.max_context_length = config.context_formatting.max_context_length
        
        # Apply cache settings
        if self.cache and hasattr(self.cache, 'ttl_mapping'):
            # Update cache TTL mapping
            self.cache.ttl_mapping = {
                'casino_review': config.cache_config.casino_review_ttl,
                'news': config.cache_config.news_ttl,
                'product_review': config.cache_config.product_review_ttl,
                'technical_doc': config.cache_config.technical_doc_ttl,
                'general': config.cache_config.general_ttl,
                'guide': config.cache_config.guide_ttl,
                'faq': config.cache_config.faq_ttl
            }
        
        # Apply feature flags
        if config.feature_flags:
            self.enable_contextual_retrieval = config.feature_flags.enable_contextual_retrieval
            self.enable_hybrid_search = config.feature_flags.enable_hybrid_search
            self.enable_response_caching = config.feature_flags.enable_response_caching
    
    def _create_integrated_chain(self):
        """Create LCEL chain with integrated monitoring and profiling."""
        
        if self.enable_prompt_optimization:
            # Enhanced chain with monitoring at each step
            chain = (
                RunnablePassthrough.assign(
                    query_analysis=RunnableLambda(self._analyze_query_with_monitoring),
                    retrieval_result=RunnableLambda(self._retrieve_with_monitoring),
                    context=RunnableLambda(self._extract_context_with_monitoring)
                )
                | RunnableLambda(self._generate_with_monitoring)
                | RunnableLambda(self._enhance_with_monitoring)
            )
        else:
            # Standard chain with monitoring
            chain = (
                RunnablePassthrough.assign(
                    context=RunnableLambda(self._retrieve_and_format_with_monitoring)
                )
                | self._create_standard_prompt()
                | self.llm
                | StrOutputParser()
            )
        
        return chain
    
    async def _analyze_query_with_monitoring(self, inputs: Dict[str, Any]) -> Any:
        """Analyze query with monitoring."""
        if self.profiler:
            async with self.profiler.profile("query_analysis"):
                return await self._analyze_query(inputs)
        else:
            return await self._analyze_query(inputs)
    
    async def _retrieve_with_monitoring(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve documents with monitoring."""
        query = inputs.get("question", "")
        
        if self.profiler:
            async with self.profiler.profile("document_retrieval", query=query):
                result = await self._retrieve_with_docs(inputs)
        else:
            result = await self._retrieve_with_docs(inputs)
        
        # Log retrieval metrics
        if self.logger:
            self.logger.info(
                LogCategory.RETRIEVAL,
                f"Retrieved {len(result.get('documents', []))} documents",
                data={
                    "document_count": len(result.get("documents", [])),
                    "query": query[:100]
                }
            )
        
        return result
    
    async def _extract_context_with_monitoring(self, inputs: Dict[str, Any]) -> str:
        """Extract context with monitoring."""
        if self.profiler:
            async with self.profiler.profile("context_extraction"):
                return await self._extract_context_from_retrieval(inputs)
        else:
            return await self._extract_context_from_retrieval(inputs)
    
    async def _generate_with_monitoring(self, inputs: Dict[str, Any]) -> str:
        """Generate response with monitoring."""
        query = inputs.get("question", "")
        
        if self.profiler:
            async with self.profiler.profile("response_generation", query=query):
                response = await self._select_prompt_and_generate(inputs)
        else:
            response = await self._select_prompt_and_generate(inputs)
        
        # Log generation metrics
        if self.logger:
            self.logger.info(
                LogCategory.GENERATION,
                "Response generated",
                data={
                    "response_length": len(response),
                    "query": query[:100]
                }
            )
        
        return response
    
    async def _enhance_with_monitoring(self, response: str) -> str:
        """Enhance response with monitoring."""
        if self.profiler:
            async with self.profiler.profile("response_enhancement"):
                return await self._enhance_response(response)
        else:
            return await self._enhance_response(response)
    
    async def _retrieve_and_format_with_monitoring(self, inputs: Dict[str, Any]) -> str:
        """Standard retrieval with monitoring."""
        query = inputs.get("question", "")
        
        # Check feature flag for hybrid search
        use_hybrid_search = False
        if self.feature_flags:
            user_context = inputs.get("user_context", {})
            use_hybrid_search = await self.feature_flags.is_feature_enabled(
                "enable_hybrid_search", user_context
            )
        
        if self.profiler:
            async with self.profiler.profile("retrieval_and_formatting", query=query):
                if use_hybrid_search:
                    # Use enhanced retrieval method
                    return await self._hybrid_retrieve_and_format(inputs)
                else:
                    return await self._retrieve_and_format(inputs)
        else:
            if use_hybrid_search:
                return await self._hybrid_retrieve_and_format(inputs)
            else:
                return await self._retrieve_and_format(inputs)
    
    async def _hybrid_retrieve_and_format(self, inputs: Dict[str, Any]) -> str:
        """Hybrid retrieval combining vector and keyword search."""
        query = inputs.get("question", "")
        
        if not self.vector_store:
            return "No vector store configured."
        
        try:
            # Perform both vector and keyword search
            vector_docs = await self.vector_store.asimilarity_search_with_score(query, k=3)
            
            # Combine and deduplicate results
            seen_content = set()
            combined_docs = []
            
            for doc, score in vector_docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    combined_docs.append((doc, score))
            
            # Format combined results
            context_parts = []
            for i, (doc, score) in enumerate(combined_docs[:5], 1):
                context_parts.append(f"Source {i} (relevance: {score:.2f}): {doc.page_content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return await self._retrieve_and_format(inputs)  # Fallback to standard
    
    async def ainvoke(self, input, **kwargs) -> RAGResponse:
        """
        Enhanced async invoke with full monitoring and configuration integration.
        
        This method extends the parent ainvoke with:
        - Query tracking and analytics
        - Performance profiling
        - Feature flag checks
        - Configuration-based behavior
        - Enhanced logging
        """
        
        # Extract query from inputs (handle both dict and string)
        if isinstance(input, dict):
            query = input.get('query', input.get('question', ''))
        else:
            query = str(input)
            
        # Generate query ID for tracking
        query_id = str(uuid.uuid4())
        user_context = kwargs.get('user_context', {})
        
        # Start pipeline tracking
        if self.pipeline_logger:
            pipeline_context = self.pipeline_logger.start_pipeline(query_id, query, user_context)
        else:
            pipeline_context = None
        
        # Track start time
        start_time = time.time()
        
        try:
            # Check feature flags
            skip_cache = False
            if self.feature_flags:
                # Check if caching is disabled via feature flag
                cache_enabled = await self.feature_flags.is_feature_enabled(
                    "enable_response_caching", user_context
                )
                skip_cache = not cache_enabled
            
            # Override cache check if needed
            if skip_cache:
                cached_response = None
            else:
                # Let parent handle cache check
                cached_response = None  # Parent will check
            
            # Call parent ainvoke with monitoring (pass original input format)
            if self.profiler:
                async with self.profiler.profile("rag_pipeline", query_id=query_id):
                    if pipeline_context:
                        async with pipeline_context:
                            response = await super().ainvoke(input, **kwargs)
                    else:
                        response = await super().ainvoke(input, **kwargs)
            else:
                if pipeline_context:
                    async with pipeline_context:
                        response = await super().ainvoke(input, **kwargs)
                else:
                    response = await super().ainvoke(input, **kwargs)
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            
            # Track metrics if analytics enabled
            if self.analytics and not response.cached:
                # Determine query type for analytics
                query_type = self._determine_query_type(query, response)
                
                await track_query_metrics(
                    self.analytics,
                    query_id=query_id,
                    query_text=query,
                    query_type=query_type,
                    classification_confidence=response.metadata.get('classification_confidence', 0.75),
                    response_time_ms=total_time,
                    retrieval_time_ms=response.metadata.get('retrieval_time_ms', 0),
                    generation_time_ms=response.metadata.get('generation_time_ms', 0),
                    total_tokens=response.token_usage.get('total_tokens', 0) if response.token_usage else 0,
                    response_quality_score=response.confidence_score,
                    cache_hit=response.cached,
                    sources_count=len(response.sources),
                    relevance_scores=[s.get('similarity_score', 0) for s in response.sources]
                )
            
            # Track A/B test metrics if applicable
            if self.feature_flags and user_context:
                variant = await self.feature_flags.get_variant(
                    "rag_optimization_experiment", user_context
                )
                if variant:
                    # Find active experiment ID
                    experiment_id = await self._get_active_experiment_id()
                    if experiment_id:
                        await self.feature_flags.track_metric(
                            experiment_id=experiment_id,
                            variant_name=variant.name,
                            metric_type="response_quality",
                            metric_value=response.confidence_score,
                            user_context=user_context
                        )
                        
                        await self.feature_flags.track_metric(
                            experiment_id=experiment_id,
                            variant_name=variant.name,
                            metric_type="response_time_ms",
                            metric_value=total_time,
                            user_context=user_context
                        )
            
            # Add monitoring metadata to response
            response.metadata.update({
                'query_id': query_id,
                'monitoring_enabled': self.enable_monitoring,
                'profiling_enabled': self.enable_profiling,
                'feature_flags_enabled': self.enable_feature_flags,
                'total_pipeline_time_ms': total_time
            })
            
            return response
            
        except Exception as e:
            # Log error with full context
            if self.logger:
                self.logger.error(
                    LogCategory.QUERY_PROCESSING,
                    f"Pipeline failed for query: {query}",
                    error=e,
                    data={
                        'query_id': query_id,
                        'query': query[:200],
                        'error_type': type(e).__name__
                    }
                )
            
            # Track error metric
            if self.analytics:
                await track_query_metrics(
                    self.analytics,
                    query_id=query_id,
                    query_text=query,
                    query_type="unknown",
                    classification_confidence=0,
                    response_time_ms=(time.time() - start_time) * 1000,
                    retrieval_time_ms=0,
                    generation_time_ms=0,
                    total_tokens=0,
                    response_quality_score=0,
                    cache_hit=False,
                    error=str(e)
                )
            
            raise
    
    def _determine_query_type(self, query: str, response: RAGResponse) -> str:
        """Determine query type for analytics."""
        
        # Check if query analysis was performed
        if response.query_analysis:
            query_type_value = response.query_analysis.get('query_type')
            if query_type_value:
                return query_type_value
        
        # Fallback to simple pattern matching
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['casino', 'slot', 'poker', 'blackjack']):
            return 'casino_review'
        elif any(word in query_lower for word in ['how', 'guide', 'tutorial', 'step']):
            return 'game_guide'
        elif any(word in query_lower for word in ['bonus', 'promotion', 'offer']):
            return 'promotion_analysis'
        elif any(word in query_lower for word in ['news', 'update', 'latest']):
            return 'news_update'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison'
        else:
            return 'general_info'
    
    async def _get_active_experiment_id(self) -> Optional[str]:
        """Get active experiment ID for A/B testing."""
        # This would query the experiments table to find active experiments
        # For now, return a placeholder
        return "default_experiment_id"
    
    async def reload_configuration(self) -> bool:
        """Reload configuration from database."""
        if not self.config_manager:
            return False
        
        try:
            config = await self.config_manager.get_active_config(force_refresh=True)
            self._apply_config_to_instance(config)
            
            if self.logger:
                self.logger.info(
                    LogCategory.CONFIGURATION,
                    "Configuration reloaded successfully",
                    data={'version': config.version}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        stats = {
            'cache_stats': self.get_cache_stats() if self.cache else {},
            'monitoring_enabled': self.enable_monitoring,
            'profiling_enabled': self.enable_profiling,
            'feature_flags_enabled': self.enable_feature_flags
        }
        
        if self.analytics:
            # Get recent metrics from analytics
            stats['recent_metrics'] = asyncio.run(
                self.analytics.get_real_time_metrics(window_minutes=5)
            )
        
        return stats
    
    async def check_feature_flag(self, flag_name: str, user_context: Optional[Dict] = None) -> bool:
        """Check if a feature flag is enabled."""
        if not self.feature_flags:
            return False
        
        return await self.feature_flags.is_feature_enabled(flag_name, user_context or {})
    
    async def get_optimization_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance optimization report."""
        if not self.profiler:
            return {"error": "Profiling not enabled"}
        
        return await self.profiler.get_optimization_report(hours)


# Factory function for creating integrated chain
def create_integrated_rag_chain(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    supabase_client = None,
    enable_all_features: bool = True,
    **kwargs
) -> IntegratedRAGChain:
    """
    Factory function to create an Integrated RAG Chain with monitoring.
    
    Args:
        model_name: LLM model to use
        temperature: Temperature for generation
        supabase_client: Supabase client for persistence
        enable_all_features: Enable all integration features
        **kwargs: Additional parameters
        
    Returns:
        Configured IntegratedRAGChain instance
    """
    
    if enable_all_features and supabase_client:
        return IntegratedRAGChain(
            model_name=model_name,
            temperature=temperature,
            supabase_client=supabase_client,
            enable_monitoring=True,
            enable_profiling=True,
            enable_feature_flags=True,
            enable_configuration=True,
            **kwargs
        )
    else:
        # Create with minimal features
        return IntegratedRAGChain(
            model_name=model_name,
            temperature=temperature,
            supabase_client=supabase_client,
            enable_monitoring=kwargs.get('enable_monitoring', False),
            enable_profiling=kwargs.get('enable_profiling', False),
            enable_feature_flags=kwargs.get('enable_feature_flags', False),
            enable_configuration=kwargs.get('enable_configuration', False),
            **kwargs
        )


# Backward compatibility wrapper
class MonitoredUniversalRAGChain(IntegratedRAGChain):
    """Alias for backward compatibility."""
    pass


# Example usage and migration guide
if __name__ == "__main__":
    async def example_usage():
        """Example of using the Integrated RAG Chain."""
        
        # Initialize Supabase client
        from supabase import create_client
        import os
        
        supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        # Create integrated chain with all features
        chain = create_integrated_rag_chain(
            model_name="gpt-4",
            supabase_client=supabase_client,
            enable_all_features=True,
            vector_store=None  # Add your vector store
        )
        
        # Example 1: Basic query with monitoring
        response = await chain.ainvoke(
            "What are the best online casinos for beginners?",
            user_context={"user_id": "test_user_123"}
        )
        
        print(f"Response: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence_score}")
        print(f"Query ID: {response.metadata.get('query_id')}")
        
        # Example 2: Check feature flag
        hybrid_search_enabled = await chain.check_feature_flag(
            "enable_hybrid_search",
            {"user_id": "test_user_123"}
        )
        print(f"Hybrid search enabled: {hybrid_search_enabled}")
        
        # Example 3: Reload configuration
        await chain.reload_configuration()
        
        # Example 4: Get monitoring stats
        stats = chain.get_monitoring_stats()
        print(f"Cache hit rate: {stats['cache_stats'].get('hit_rate', 0):.2%}")
        
        # Example 5: Get optimization report
        report = await chain.get_optimization_report(hours=24)
        print(f"Top bottlenecks: {report.get('top_bottlenecks', [])}")
    
    # Run example
    # asyncio.run(example_usage()) 