# API Reference - Enhanced Confidence Scoring System

## Table of Contents

1. [EnhancedConfidenceCalculator](#enhancedconfidencecalculator)
2. [SourceQualityAnalyzer](#sourcequalityanalyzer)
3. [IntelligentCache](#intelligentcache)
4. [ResponseValidator](#responsevalidator)
5. [ConfidenceIntegrator](#confidenceintegrator)
6. [Universal RAG Chain Integration](#universal-rag-chain-integration)
7. [Models and Enums](#models-and-enums)
8. [Configuration](#configuration)
9. [Utility Functions](#utility-functions)

---

## EnhancedConfidenceCalculator

Main orchestrator class for the enhanced confidence scoring system.

### Class Definition

```python
class EnhancedConfidenceCalculator:
    """
    Advanced confidence calculator with 4-factor analysis, intelligent caching,
    and comprehensive quality assessment.
    """
```

### Constructor

```python
def __init__(
    self,
    source_analyzer: Optional[SourceQualityAnalyzer] = None,
    cache_system: Optional[IntelligentCache] = None,
    validator: Optional[ResponseValidator] = None,
    integrator: Optional[ConfidenceIntegrator] = None
)
```

**Parameters:**
- `source_analyzer` (optional): Custom source quality analyzer instance
- `cache_system` (optional): Custom intelligent cache instance  
- `validator` (optional): Custom response validator instance
- `integrator` (optional): Custom confidence integrator instance

### Methods

#### calculate_enhanced_confidence

```python
async def calculate_enhanced_confidence(
    self,
    response: RAGResponse,
    query: str,
    query_type: str = "general",
    sources: List[Document] = None,
    generation_metadata: Dict[str, Any] = None
) -> Tuple[ConfidenceBreakdown, EnhancedRAGResponse]
```

Calculate comprehensive confidence score with breakdown.

**Parameters:**
- `response`: RAG response to analyze
- `query`: Original user query
- `query_type`: Type of query (factual, tutorial, comparison, review)
- `sources`: Source documents used
- `generation_metadata`: Additional metadata from generation process

**Returns:**
- `ConfidenceBreakdown`: Detailed confidence analysis
- `EnhancedRAGResponse`: Enhanced response with metadata

**Example:**
```python
calculator = EnhancedConfidenceCalculator()
breakdown, enhanced_response = await calculator.calculate_enhanced_confidence(
    response=response,
    query="How to optimize RAG systems?",
    query_type="tutorial",
    sources=source_docs,
    generation_metadata={"model": "gpt-4", "temperature": 0.7}
)
```

#### get_quality_level

```python
def get_quality_level(self, confidence_score: float) -> ResponseQualityLevel
```

Classify response quality based on confidence score.

**Parameters:**
- `confidence_score`: Overall confidence score (0.0-1.0)

**Returns:**
- `ResponseQualityLevel`: Quality classification (EXCELLENT, GOOD, ACCEPTABLE, POOR)

#### should_regenerate

```python
def should_regenerate(
    self,
    confidence_breakdown: ConfidenceBreakdown,
    threshold: float = 0.4
) -> bool
```

Determine if response should be regenerated based on quality.

**Parameters:**
- `confidence_breakdown`: Detailed confidence analysis
- `threshold`: Minimum acceptable confidence score

**Returns:**
- `bool`: True if regeneration is recommended

---

## SourceQualityAnalyzer

Analyzes and scores the quality of source documents.

### Class Definition

```python
class SourceQualityAnalyzer:
    """
    Multi-tier source quality assessment with comprehensive quality indicators.
    """
```

### Methods

#### analyze_sources

```python
def analyze_sources(self, sources: List[Document]) -> SourceQualityAnalysis
```

Analyze quality of source documents.

**Parameters:**
- `sources`: List of source documents to analyze

**Returns:**
- `SourceQualityAnalysis`: Comprehensive quality assessment

#### get_source_tier

```python
def get_source_tier(self, source: Document) -> SourceTier
```

Classify individual source into quality tier.

**Parameters:**
- `source`: Document to classify

**Returns:**
- `SourceTier`: Quality tier (PREMIUM, HIGH, MEDIUM, LOW, POOR)

#### calculate_quality_indicators

```python
def calculate_quality_indicators(self, source: Document) -> Dict[str, float]
```

Calculate detailed quality indicators for a source.

**Parameters:**
- `source`: Document to analyze

**Returns:**
- `Dict[str, float]`: Quality indicators (authority, credibility, expertise, recency, detail)

**Example:**
```python
analyzer = SourceQualityAnalyzer()
analysis = analyzer.analyze_sources(source_documents)
print(f"Overall Quality: {analysis.overall_quality:.3f}")
print(f"Quality Distribution: {analysis.quality_distribution}")
```

---

## IntelligentCache

Learning-based caching system with adaptive strategies.

### Class Definition

```python
class IntelligentCache:
    """
    Intelligent caching system with multiple strategies and adaptive optimization.
    """
```

### Constructor

```python
def __init__(
    self,
    strategy: CacheStrategy = CacheStrategy.BALANCED,
    max_size: int = 1000,
    default_ttl: int = 3600
)
```

**Parameters:**
- `strategy`: Caching strategy (CONSERVATIVE, BALANCED, AGGRESSIVE, ADAPTIVE)
- `max_size`: Maximum cache size
- `default_ttl`: Default TTL in seconds

### Methods

#### get

```python
async def get(self, cache_key: str) -> Optional[CacheEntry]
```

Retrieve cached response.

**Parameters:**
- `cache_key`: Unique cache key

**Returns:**
- `CacheEntry`: Cached entry if exists and valid, None otherwise

#### put

```python
async def put(
    self,
    cache_key: str,
    response: RAGResponse,
    confidence_score: float,
    metadata: Dict[str, Any] = None
) -> bool
```

Store response in cache if it meets quality criteria.

**Parameters:**
- `cache_key`: Unique cache key
- `response`: Response to cache
- `confidence_score`: Quality score for admission control
- `metadata`: Additional metadata

**Returns:**
- `bool`: True if cached successfully

#### get_performance_metrics

```python
def get_performance_metrics(self) -> CachePerformanceMetrics
```

Get current cache performance statistics.

**Returns:**
- `CachePerformanceMetrics`: Performance statistics

**Example:**
```python
cache = IntelligentCache(strategy=CacheStrategy.ADAPTIVE)

# Try to get cached response
cached_entry = await cache.get(cache_key)
if cached_entry:
    return cached_entry.response

# Cache new response
await cache.put(cache_key, response, confidence_score, metadata)
```

---

## ResponseValidator

Validates responses across multiple quality dimensions.

### Class Definition

```python
class ResponseValidator:
    """
    Comprehensive validation framework with extensible criteria.
    """
```

### Methods

#### validate_response

```python
def validate_response(
    self,
    response: RAGResponse,
    query: str,
    sources: List[Document] = None
) -> ValidationResult
```

Perform comprehensive response validation.

**Parameters:**
- `response`: Response to validate
- `query`: Original query
- `sources`: Source documents used

**Returns:**
- `ValidationResult`: Comprehensive validation assessment

#### validate_format

```python
def validate_format(self, response: RAGResponse) -> FormatValidation
```

Validate response format and structure.

**Parameters:**
- `response`: Response to validate

**Returns:**
- `FormatValidation`: Format validation results

#### validate_content

```python
def validate_content(
    self,
    response: RAGResponse,
    query: str
) -> ContentValidation
```

Validate content quality and relevance.

**Parameters:**
- `response`: Response to validate
- `query`: Original query

**Returns:**
- `ContentValidation`: Content validation results

**Example:**
```python
validator = ResponseValidator()
validation = validator.validate_response(response, query, sources)

if validation.overall_score < 0.6:
    print("Response quality issues detected:")
    for issue in validation.issues:
        print(f"- {issue.category}: {issue.description}")
```

---

## ConfidenceIntegrator

Unified interface for integrating confidence scoring across components.

### Class Definition

```python
class ConfidenceIntegrator:
    """
    Unified confidence integration with adaptive weight management.
    """
```

### Methods

#### integrate_confidence

```python
def integrate_confidence(
    self,
    content_score: float,
    source_score: float,
    query_match_score: float,
    technical_score: float,
    query_type: str = "general"
) -> float
```

Calculate final confidence score with query-type specific weights.

**Parameters:**
- `content_score`: Content quality score (0.0-1.0)
- `source_score`: Source quality score (0.0-1.0)
- `query_match_score`: Query matching score (0.0-1.0)
- `technical_score`: Technical factors score (0.0-1.0)
- `query_type`: Type of query for weight adjustment

**Returns:**
- `float`: Final integrated confidence score

#### get_query_type_weights

```python
def get_query_type_weights(self, query_type: str) -> Dict[str, float]
```

Get weight configuration for specific query type.

**Parameters:**
- `query_type`: Query type identifier

**Returns:**
- `Dict[str, float]`: Weight configuration

---

## Universal RAG Chain Integration

Enhanced Universal RAG Chain with confidence scoring integration.

### Factory Function

```python
def create_universal_rag_chain(
    model_name: str = "gpt-3.5-turbo",
    enable_enhanced_confidence: bool = False,
    enable_prompt_optimization: bool = False,
    enable_caching: bool = False,
    confidence_config: Optional[Dict[str, Any]] = None,
    vector_store: Optional[VectorStore] = None,
    **kwargs
) -> UniversalRAGChain
```

Create enhanced Universal RAG Chain with confidence scoring.

**Parameters:**
- `model_name`: LLM model to use
- `enable_enhanced_confidence`: Enable 4-factor confidence scoring
- `enable_prompt_optimization`: Enable prompt optimization
- `enable_caching`: Enable intelligent caching
- `confidence_config`: Custom confidence configuration
- `vector_store`: Vector store for document retrieval
- `**kwargs`: Additional chain configuration

**Returns:**
- `UniversalRAGChain`: Configured chain instance

### Chain Methods

#### ainvoke

```python
async def ainvoke(
    self,
    input_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Union[RAGResponse, EnhancedRAGResponse]
```

Asynchronously invoke the RAG chain.

**Parameters:**
- `input_data`: Input containing query and optional parameters
- `config`: Runtime configuration

**Returns:**
- Enhanced response with confidence metadata if enabled

#### get_enhanced_system_status

```python
def get_enhanced_system_status(self) -> SystemStatus
```

Get comprehensive system status and performance metrics.

**Returns:**
- `SystemStatus`: System health and performance data

**Example:**
```python
# Create enhanced chain
chain = create_universal_rag_chain(
    model_name="gpt-4",
    enable_enhanced_confidence=True,
    enable_caching=True,
    confidence_config={
        'quality_threshold': 0.75,
        'cache_strategy': 'ADAPTIVE'
    }
)

# Use the chain
response = await chain.ainvoke({
    "query": "How to implement RAG systems?",
    "query_type": "tutorial"
})
```

---

## Models and Enums

### Core Response Models

#### RAGResponse

```python
class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
```

#### EnhancedRAGResponse

```python
class EnhancedRAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    confidence_breakdown: Dict[str, Dict[str, float]]
    quality_level: ResponseQualityLevel
    quality_flags: List[str]
    improvement_suggestions: List[str]
    cached: bool
    response_time: float
    metadata: Dict[str, Any]
```

#### ConfidenceBreakdown

```python
class ConfidenceBreakdown(BaseModel):
    overall_score: float
    content_quality: Dict[str, float]
    source_quality: Dict[str, float]
    query_matching: Dict[str, float]
    technical_factors: Dict[str, float]
    quality_flags: List[str]
    improvement_suggestions: List[str]
    quality_level: ResponseQualityLevel
```

### Enums

#### ResponseQualityLevel

```python
class ResponseQualityLevel(str, Enum):
    EXCELLENT = "excellent"    # 0.8-1.0
    GOOD = "good"             # 0.6-0.79
    ACCEPTABLE = "acceptable"  # 0.4-0.59
    POOR = "poor"             # 0.0-0.39
```

#### SourceTier

```python
class SourceTier(str, Enum):
    PREMIUM = "premium"       # 0.9-1.0
    HIGH = "high"            # 0.7-0.89
    MEDIUM = "medium"        # 0.5-0.69
    LOW = "low"              # 0.3-0.49
    POOR = "poor"            # 0.0-0.29
```

#### CacheStrategy

```python
class CacheStrategy(str, Enum):
    CONSERVATIVE = "conservative"  # >0.85 confidence
    BALANCED = "balanced"         # >0.70 confidence
    AGGRESSIVE = "aggressive"     # >0.50 confidence
    ADAPTIVE = "adaptive"         # Dynamic threshold
```

### Analysis Models

#### SourceQualityAnalysis

```python
class SourceQualityAnalysis(BaseModel):
    overall_quality: float
    quality_distribution: Dict[SourceTier, int]
    quality_indicators: Dict[str, float]
    source_scores: List[Dict[str, Any]]
    recommendations: List[str]
```

#### ValidationResult

```python
class ValidationResult(BaseModel):
    overall_score: float
    format_validation: FormatValidation
    content_validation: ContentValidation
    source_validation: SourceValidation
    issues: List[ValidationIssue]
    recommendations: List[str]
```

#### CachePerformanceMetrics

```python
class CachePerformanceMetrics(BaseModel):
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    average_response_time: float
    cache_size: int
    cache_utilization: float
    quality_distribution: Dict[str, int]
```

---

## Configuration

### Confidence Configuration

```python
class ConfidenceConfig(BaseModel):
    quality_threshold: float = 0.70
    regeneration_threshold: float = 0.40
    max_regeneration_attempts: int = 2
    cache_strategy: CacheStrategy = CacheStrategy.BALANCED
    query_type_weights: Optional[Dict[str, Dict[str, float]]] = None
    enable_quality_flags: bool = True
    enable_improvement_suggestions: bool = True
```

### Cache Configuration

```python
class CacheConfig(BaseModel):
    strategy: CacheStrategy = CacheStrategy.BALANCED
    max_size: int = 1000
    default_ttl: int = 3600
    quality_threshold: float = 0.70
    adaptive_threshold_range: Tuple[float, float] = (0.6, 0.85)
    performance_monitor_interval: int = 300
```

### Validation Configuration

```python
class ValidationConfig(BaseModel):
    min_length: int = 10
    max_length: int = 5000
    min_relevance_score: float = 0.5
    enable_format_validation: bool = True
    enable_content_validation: bool = True
    enable_source_validation: bool = True
    critical_issue_threshold: float = 0.3
```

---

## Utility Functions

### Query Type Detection

```python
def detect_query_type(query: str) -> str
```

Automatically detect query type based on content and patterns.

**Parameters:**
- `query`: User query to analyze

**Returns:**
- `str`: Detected query type (factual, tutorial, comparison, review, general)

### Cache Key Generation

```python
def generate_cache_key(
    query: str,
    model_name: str,
    additional_params: Dict[str, Any] = None
) -> str
```

Generate unique cache key for query and parameters.

**Parameters:**
- `query`: User query
- `model_name`: LLM model name
- `additional_params`: Additional parameters affecting response

**Returns:**
- `str`: Unique cache key

### Quality Score Calculation

```python
def calculate_weighted_score(
    scores: Dict[str, float],
    weights: Dict[str, float]
) -> float
```

Calculate weighted average of component scores.

**Parameters:**
- `scores`: Component scores
- `weights`: Component weights

**Returns:**
- `float`: Weighted average score

---

## Error Handling

### Custom Exceptions

#### ConfidenceCalculationError

```python
class ConfidenceCalculationError(Exception):
    """Raised when confidence calculation fails."""
    pass
```

#### CacheError

```python
class CacheError(Exception):
    """Raised when cache operations fail."""
    pass
```

#### ValidationError

```python
class ValidationError(Exception):
    """Raised when response validation fails."""
    pass
```

### Error Recovery

All components implement graceful degradation:

- **Confidence Calculator**: Falls back to basic scoring if enhanced calculation fails
- **Cache System**: Continues operation without caching if cache fails
- **Validator**: Logs issues but continues processing if validation fails
- **Source Analyzer**: Uses default quality scores if analysis fails

---

## Performance Considerations

### Async Operations

All heavy operations are async for optimal performance:

```python
# Parallel confidence calculation
async def calculate_enhanced_confidence(self, ...):
    tasks = [
        self._calculate_content_quality(response, query),
        self._calculate_source_quality(sources),
        self._calculate_query_matching(response, query),
        self._calculate_technical_factors(metadata)
    ]
    results = await asyncio.gather(*tasks)
```

### Caching Strategies

- **CONSERVATIVE**: Best for quality-focused applications
- **BALANCED**: Optimal for most use cases
- **AGGRESSIVE**: Best for high-traffic scenarios
- **ADAPTIVE**: Self-optimizing based on performance

### Memory Management

- Automatic cache eviction based on LRU and quality
- Configurable cache size limits
- Memory-efficient storage of confidence breakdowns

---

For more information, see:
- [System Overview](./enhanced_confidence_scoring_system.md)
- [Quick Start Guide](./quick_start.md)
- [Production Deployment Guide](./production_deployment.md) 