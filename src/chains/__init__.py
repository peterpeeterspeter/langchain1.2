# RAG Chains package
# Contains universal RAG chain and advanced prompt optimization

from .advanced_prompt_system import (
    QueryType,
    ExpertiseLevel,
    ResponseFormat,
    QueryAnalysis,
    QueryClassifier,
    AdvancedContextFormatter,
    EnhancedSourceFormatter,
    DomainSpecificPrompts,
    OptimizedPromptManager
)

from .universal_rag_lcel import (
    UniversalRAGChain,
    create_universal_rag_chain,
    RAGResponse
)

# Integrated RAG Chain with monitoring and configuration
# TODO: Fix utils.enhanced_logging import before enabling
# from .integrated_rag_chain import (
#     IntegratedRAGChain,
#     create_integrated_rag_chain,  
#     MonitoredUniversalRAGChain
# )

# Enhanced Confidence Scoring System
from .enhanced_confidence_scoring_system import (
    # Core models and enums
    EnhancedRAGResponse,
    ConfidenceFactors,
    SourceQualityTier,
    ResponseQualityLevel,
    CacheStrategy,
    ConfidenceFactorType,
    CacheEntry,
    SystemConfiguration,
    PerformanceTracker,
    
    # Utility functions
    calculate_quality_tier,
    generate_query_hash,
    normalize_score,
    performance_tracker,
    
    # Source Quality Analysis
    SourceQualityAnalyzer,
    
    # Intelligent Caching System
    IntelligentCache,
    intelligent_cache,
    
    # Response Validation Framework
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    ValidationMetrics,
    ResponseValidator,
    ValidationIntegrator,
    validate_and_enhance_response,
    response_validator,
)

__all__ = [
    # Advanced Prompt System
    "QueryType",
    "ExpertiseLevel", 
    "ResponseFormat",
    "QueryAnalysis",
    "QueryClassifier",
    "AdvancedContextFormatter",
    "EnhancedSourceFormatter",
    "DomainSpecificPrompts",
    "OptimizedPromptManager",
    
    # Universal RAG Chain
    "UniversalRAGChain",
    "create_universal_rag_chain",
    "RAGResponse",
    
    # Integrated RAG Chain with monitoring and configuration
    # "IntegratedRAGChain",
    # "create_integrated_rag_chain", 
    # "MonitoredUniversalRAGChain",
    
    # Core models and enums
    "EnhancedRAGResponse",
    "ConfidenceFactors", 
    "SourceQualityTier",
    "ResponseQualityLevel",
    "CacheStrategy",
    "ConfidenceFactorType",
    "CacheEntry",
    "SystemConfiguration",
    "PerformanceTracker",
    
    # Utility functions
    "calculate_quality_tier",
    "generate_query_hash", 
    "normalize_score",
    "performance_tracker",
    
    # Source Quality Analysis
    "SourceQualityAnalyzer",
    
    # Intelligent Caching System
    "IntelligentCache",
    "intelligent_cache",
    
    # Response Validation Framework
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationMetrics",
    "ResponseValidator",
    "ValidationIntegrator",
    "validate_and_enhance_response",
    "response_validator",
] 