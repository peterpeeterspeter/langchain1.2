"""
Self-Query Metadata Filtering System for Universal RAG CMS
==========================================================

Task 3.4: Self-Query Metadata Filtering with intelligent query analysis
and metadata-based filtering capabilities.

Key Features:
- Natural language query parsing and intent detection
- Metadata extraction and constraint application
- Integration with hybrid search and multi-query systems
- Dynamic filtering rules based on query context
- Performance optimization with caching
"""

import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict

from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Import hybrid search and multi-query components
from .hybrid_search import (
    HybridSearchEngine, ContextualHybridSearch, HybridSearchResults,
    SearchType, SearchResult, HybridSearchConfig
)
from .multi_query import (
    MultiQueryRetriever, QueryExpansionStrategy, QueryType,
    MultiQueryResults
)

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Filter operators for metadata filtering."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    BETWEEN = "between"


class FilterScope(Enum):
    """Scope of filtering application."""
    PRE_SEARCH = "pre_search"    # Filter before search
    POST_SEARCH = "post_search"  # Filter after search
    BOTH = "both"               # Filter before and after


class MetadataFieldType(Enum):
    """Types of metadata fields for proper filtering."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"
    NESTED = "nested"


@dataclass
class MetadataFilter:
    """Individual metadata filter with operator and value."""
    field: str
    operator: FilterOperator
    value: Any
    field_type: MetadataFieldType = MetadataFieldType.STRING
    scope: FilterScope = FilterScope.BOTH
    confidence: float = 1.0
    source: str = "user"  # "user", "inferred", "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def applies_to_document(self, document: Document) -> bool:
        """Check if filter applies to a document."""
        try:
            doc_metadata = document.metadata or {}
            field_value = self._get_nested_value(doc_metadata, self.field)
            
            if field_value is None:
                return False
            
            return self._evaluate_condition(field_value, self.operator, self.value)
            
        except Exception as e:
            logger.debug(f"Error evaluating filter {self.field}: {e}")
            return False
    
    def _get_nested_value(self, metadata: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested metadata using dot notation."""
        keys = field_path.split('.')
        value = metadata
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _evaluate_condition(self, field_value: Any, operator: FilterOperator, filter_value: Any) -> bool:
        """Evaluate filter condition."""
        try:
            # Type conversion based on field type
            if self.field_type == MetadataFieldType.DATE:
                field_value = self._parse_date(field_value)
                filter_value = self._parse_date(filter_value)
            elif self.field_type == MetadataFieldType.INTEGER:
                field_value = int(field_value)
                filter_value = int(filter_value)
            elif self.field_type == MetadataFieldType.FLOAT:
                field_value = float(field_value)
                filter_value = float(filter_value)
            
            # Apply operator
            if operator == FilterOperator.EQUALS:
                return field_value == filter_value
            elif operator == FilterOperator.NOT_EQUALS:
                return field_value != filter_value
            elif operator == FilterOperator.GREATER_THAN:
                return field_value > filter_value
            elif operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return field_value >= filter_value
            elif operator == FilterOperator.LESS_THAN:
                return field_value < filter_value
            elif operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return field_value <= filter_value
            elif operator == FilterOperator.IN:
                return field_value in filter_value
            elif operator == FilterOperator.NOT_IN:
                return field_value not in filter_value
            elif operator == FilterOperator.CONTAINS:
                return str(filter_value).lower() in str(field_value).lower()
            elif operator == FilterOperator.NOT_CONTAINS:
                return str(filter_value).lower() not in str(field_value).lower()
            elif operator == FilterOperator.STARTS_WITH:
                return str(field_value).lower().startswith(str(filter_value).lower())
            elif operator == FilterOperator.ENDS_WITH:
                return str(field_value).lower().endswith(str(filter_value).lower())
            elif operator == FilterOperator.REGEX:
                return bool(re.search(str(filter_value), str(field_value), re.IGNORECASE))
            elif operator == FilterOperator.BETWEEN:
                if isinstance(filter_value, (list, tuple)) and len(filter_value) == 2:
                    return filter_value[0] <= field_value <= filter_value[1]
            
            return False
            
        except Exception as e:
            logger.debug(f"Error evaluating condition: {e}")
            return False
    
    def _parse_date(self, date_value: Any) -> datetime:
        """Parse date value to datetime object."""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {date_value}")
        else:
            raise ValueError(f"Invalid date type: {type(date_value)}")


@dataclass
class QueryAnalysis:
    """Analysis results from natural language query parsing."""
    
    # Core query information
    original_query: str
    intent: str  # "find", "filter", "compare", "list", etc.
    main_topic: str
    
    # Extracted filters
    metadata_filters: List[MetadataFilter]
    temporal_constraints: List[MetadataFilter]
    categorical_constraints: List[MetadataFilter]
    
    # Query characteristics
    query_type: QueryType
    complexity_level: str  # "simple", "moderate", "complex"
    confidence: float = 0.0
    
    # Processing metadata
    extraction_method: str = "rule_based"  # "rule_based", "llm_based", "hybrid"
    processing_time_ms: float = 0.0
    ambiguities: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SelfQueryResults:
    """Results from self-query metadata filtering with detailed metrics."""
    
    # Core results
    filtered_results: List[SearchResult]
    original_results: List[SearchResult]
    
    # Query analysis
    query_analysis: QueryAnalysis
    applied_filters: List[MetadataFilter]
    
    # Performance metrics
    total_documents_before: int
    total_documents_after: int
    filter_effectiveness: float  # Reduction ratio
    
    # Filter breakdown
    pre_search_filters: List[MetadataFilter]
    post_search_filters: List[MetadataFilter]
    
    # Quality metrics and optional fields with defaults
    processing_time_ms: float = 0.0
    relevance_improvement: float = 0.0
    precision_score: float = 0.0
    recall_impact: float = 0.0


class SelfQueryConfig(BaseModel):
    """Configuration for self-query metadata filtering."""
    
    # Analysis settings
    enable_llm_analysis: bool = Field(default=True, description="Enable LLM-based query analysis")
    llm_model: str = Field(default="gpt-4.1-mini", description="LLM model for query analysis")
    analysis_confidence_threshold: float = Field(default=0.7, description="Minimum confidence for filter application")
    
    # Filter settings
    max_filters_per_query: int = Field(default=10, description="Maximum filters per query")
    enable_fuzzy_matching: bool = Field(default=True, description="Enable fuzzy value matching")
    fuzzy_threshold: float = Field(default=0.8, description="Fuzzy matching threshold")
    
    # Performance settings
    cache_query_analysis: bool = Field(default=True, description="Cache query analysis results")
    enable_parallel_filtering: bool = Field(default=True, description="Enable parallel filter processing")
    max_concurrent_filters: int = Field(default=5, description="Maximum concurrent filter operations")
    
    # Quality settings
    filter_validation: bool = Field(default=True, description="Validate filters before application")
    require_filter_confidence: bool = Field(default=True, description="Require minimum confidence for filters")
    auto_correction: bool = Field(default=True, description="Auto-correct common filter mistakes")
    
    # Integration settings
    integrate_with_multi_query: bool = Field(default=True, description="Integrate with multi-query retrieval")
    preserve_query_expansion: bool = Field(default=True, description="Preserve query expansion results")


class QueryAnalyzer:
    """Intelligent query analyzer for extracting metadata filters."""
    
    def __init__(self, config: SelfQueryConfig):
        """Initialize the query analyzer."""
        self.config = config
        
        # Initialize LLM for query analysis
        if config.enable_llm_analysis:
            if "gpt" in config.llm_model.lower():
                self.llm = ChatOpenAI(
                    model=config.llm_model,
                    temperature=0.1,  # Low temperature for consistent analysis
                    max_tokens=1000
                )
            elif "claude" in config.llm_model.lower():
                self.llm = ChatAnthropic(
                    model=config.llm_model,
                    temperature=0.1,
                    max_tokens=1000
                )
            else:
                self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        else:
            self.llm = None
        
        # Query analysis cache
        self.analysis_cache = {}
        
        # Rule-based patterns for filter extraction
        self.filter_patterns = {
            'date_patterns': [
                r'after (\d{4})',
                r'before (\d{4})',
                r'from (\d{4})',
                r'since (\d{4})',
                r'published in (\d{4})',
                r'(\d{4}) onwards',
                r'recent|latest|new',
                r'old|older|previous'
            ],
            'type_patterns': [
                r'type:(\w+)',
                r'category:(\w+)',
                r'kind of (\w+)',
                r'(\w+) type',
                r'(\w+) category'
            ],
            'source_patterns': [
                r'from (\w+\.com|\w+\.org|\w+\.edu)',
                r'source:(\w+)',
                r'site:(\w+)',
                r'domain:(\w+)',
                r'published by (\w+)'
            ],
            'quality_patterns': [
                r'high quality',
                r'best rated',
                r'top rated',
                r'highly rated',
                r'expert level',
                r'professional',
                r'authoritative'
            ],
            'language_patterns': [
                r'in (\w+) language',
                r'(\w+) language',
                r'lang:(\w+)'
            ],
            'content_length_patterns': [
                r'detailed',
                r'brief',
                r'comprehensive',
                r'summary',
                r'in-depth',
                r'short',
                r'long'
            ]
        }
        
        # LLM analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert query analyzer for a RAG system. Analyze the user query and extract structured metadata filters.

User Query: {query}

Extract the following information:

1. Main Intent: What is the user trying to accomplish? (find, filter, compare, list, etc.)
2. Main Topic: What is the core subject matter?
3. Metadata Filters: Extract specific filters the user wants applied

For each filter, identify:
- Field name (e.g., "type", "date", "source", "language", "quality_score")
- Operator (equals, contains, greater_than, less_than, in, between)
- Value(s)
- Confidence (0.0-1.0)

Common filter categories:
- Temporal: publication dates, last updated, recency
- Content type: article, review, tutorial, news, academic
- Source quality: authority score, credibility, expertise level  
- Domain/Source: specific websites, organizations, authors
- Language: content language
- Content characteristics: length, detail level, complexity

Return your analysis in this JSON format:
{{
    "intent": "string",
    "main_topic": "string",
    "filters": [
        {{
            "field": "string",
            "operator": "string", 
            "value": "any",
            "confidence": 0.0-1.0,
            "reasoning": "string"
        }}
    ],
    "query_type": "factual|comparison|tutorial|review|general",
    "complexity": "simple|moderate|complex",
    "overall_confidence": 0.0-1.0
}}

Be conservative with filter extraction - only extract filters you are confident about.
""")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query and extract metadata filters."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.config.cache_query_analysis:
                cache_key = self._get_cache_key(query)
                if cache_key in self.analysis_cache:
                    cached_analysis = self.analysis_cache[cache_key]
                    cached_analysis.processing_time_ms = (time.time() - start_time) * 1000
                    return cached_analysis
            
            # Parallel analysis approaches
            analysis_tasks = [
                self._rule_based_analysis(query),
                self._llm_based_analysis(query) if self.llm else self._create_empty_llm_analysis()
            ]
            
            rule_analysis, llm_analysis = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(rule_analysis, Exception):
                rule_analysis = self._create_fallback_analysis(query)
            if isinstance(llm_analysis, Exception):
                llm_analysis = self._create_empty_llm_analysis()
            
            # Combine analyses
            combined_analysis = await self._combine_analyses(query, rule_analysis, llm_analysis)
            
            # Calculate processing time
            combined_analysis.processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            if self.config.cache_query_analysis:
                self.analysis_cache[cache_key] = combined_analysis
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._create_fallback_analysis(query)
    
    async def _rule_based_analysis(self, query: str) -> Dict[str, Any]:
        """Perform rule-based query analysis."""
        
        query_lower = query.lower()
        filters = []
        
        # Extract date filters
        for pattern in self.filter_patterns['date_patterns']:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match.isdigit():  # Year
                    year = int(match)
                    if 'after' in pattern or 'since' in pattern or 'from' in pattern:
                        filters.append({
                            'field': 'published_date',
                            'operator': 'gte',
                            'value': f"{year}-01-01",
                            'confidence': 0.9,
                            'source': 'rule_based'
                        })
                    elif 'before' in pattern:
                        filters.append({
                            'field': 'published_date',
                            'operator': 'lt',
                            'value': f"{year}-01-01",
                            'confidence': 0.9,
                            'source': 'rule_based'
                        })
                elif pattern in ['recent', 'latest', 'new']:
                    # Recent content filter
                    filters.append({
                        'field': 'published_date',
                        'operator': 'gte',
                        'value': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        'confidence': 0.8,
                        'source': 'rule_based'
                    })
        
        # Extract type/category filters
        for pattern in self.filter_patterns['type_patterns']:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                filters.append({
                    'field': 'content_type',
                    'operator': 'eq',
                    'value': match,
                    'confidence': 0.8,
                    'source': 'rule_based'
                })
        
        # Extract source filters
        for pattern in self.filter_patterns['source_patterns']:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                filters.append({
                    'field': 'source_domain',
                    'operator': 'contains',
                    'value': match,
                    'confidence': 0.9,
                    'source': 'rule_based'
                })
        
        # Extract quality filters
        for pattern in self.filter_patterns['quality_patterns']:
            if pattern in query_lower:
                filters.append({
                    'field': 'quality_score',
                    'operator': 'gte',
                    'value': 0.8,
                    'confidence': 0.7,
                    'source': 'rule_based'
                })
                break
        
        # Determine intent and topic
        intent = self._determine_intent(query_lower)
        main_topic = self._extract_main_topic(query)
        
        return {
            'intent': intent,
            'main_topic': main_topic,
            'filters': filters,
            'method': 'rule_based',
            'confidence': 0.7
        }
    
    async def _llm_based_analysis(self, query: str) -> Dict[str, Any]:
        """Perform LLM-based query analysis."""
        
        if not self.llm:
            return self._create_empty_llm_analysis()
        
        try:
            # Generate analysis prompt
            prompt = self.analysis_prompt.format(query=query)
            
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            
            # Parse JSON response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
                analysis_data['method'] = 'llm_based'
                return analysis_data
            else:
                logger.warning("No JSON found in LLM response")
                return self._create_empty_llm_analysis()
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._create_empty_llm_analysis()
    
    def _create_empty_llm_analysis(self) -> Dict[str, Any]:
        """Create empty LLM analysis structure."""
        return {
            'intent': 'find',
            'main_topic': '',
            'filters': [],
            'method': 'llm_based',
            'confidence': 0.0
        }
    
    async def _combine_analyses(
        self, 
        query: str, 
        rule_analysis: Dict[str, Any], 
        llm_analysis: Dict[str, Any]
    ) -> QueryAnalysis:
        """Combine rule-based and LLM-based analyses."""
        
        # Merge filters, preferring higher confidence ones
        all_filters = []
        
        # Add rule-based filters
        for filter_data in rule_analysis.get('filters', []):
            metadata_filter = MetadataFilter(
                field=filter_data['field'],
                operator=FilterOperator(filter_data['operator']),
                value=filter_data['value'],
                confidence=filter_data['confidence'],
                source=filter_data.get('source', 'rule_based')
            )
            all_filters.append(metadata_filter)
        
        # Add LLM filters (if confidence is high enough)
        for filter_data in llm_analysis.get('filters', []):
            if filter_data.get('confidence', 0) >= self.config.analysis_confidence_threshold:
                metadata_filter = MetadataFilter(
                    field=filter_data['field'],
                    operator=FilterOperator(filter_data['operator']),
                    value=filter_data['value'],
                    confidence=filter_data['confidence'],
                    source='llm_based'
                )
                all_filters.append(metadata_filter)
        
        # Remove duplicates and limit count
        unique_filters = self._deduplicate_filters(all_filters)
        limited_filters = unique_filters[:self.config.max_filters_per_query]
        
        # Categorize filters
        temporal_constraints = [f for f in limited_filters if f.field in ['published_date', 'last_updated', 'created_date']]
        categorical_constraints = [f for f in limited_filters if f.field in ['content_type', 'category', 'source_domain']]
        other_filters = [f for f in limited_filters if f not in temporal_constraints + categorical_constraints]
        
        # Determine best intent and topic
        intent = llm_analysis.get('intent', rule_analysis.get('intent', 'find'))
        main_topic = llm_analysis.get('main_topic', rule_analysis.get('main_topic', query))
        
        # Determine query type
        query_type = self._determine_query_type(query, intent)
        
        # Calculate overall confidence
        if limited_filters:
            avg_confidence = sum(f.confidence for f in limited_filters) / len(limited_filters)
        else:
            avg_confidence = 0.5
        
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            main_topic=main_topic,
            metadata_filters=limited_filters,
            temporal_constraints=temporal_constraints,
            categorical_constraints=categorical_constraints,
            query_type=query_type,
            complexity_level=self._determine_complexity(query, limited_filters),
            confidence=avg_confidence,
            extraction_method="hybrid" if self.llm else "rule_based"
        )
    
    def _deduplicate_filters(self, filters: List[MetadataFilter]) -> List[MetadataFilter]:
        """Remove duplicate filters, keeping highest confidence ones."""
        
        # Group by field and operator
        filter_groups = defaultdict(list)
        for filter_obj in filters:
            key = f"{filter_obj.field}_{filter_obj.operator.value}"
            filter_groups[key].append(filter_obj)
        
        # Keep best filter from each group
        unique_filters = []
        for group in filter_groups.values():
            best_filter = max(group, key=lambda f: f.confidence)
            unique_filters.append(best_filter)
        
        return unique_filters
    
    def _determine_intent(self, query_lower: str) -> str:
        """Determine query intent from text analysis."""
        
        intent_patterns = {
            'find': ['find', 'search', 'look for', 'get', 'retrieve'],
            'filter': ['filter', 'narrow down', 'refine', 'limit to'],
            'compare': ['compare', 'vs', 'versus', 'difference between'],
            'list': ['list', 'show all', 'enumerate', 'what are'],
            'analyze': ['analyze', 'analysis', 'examine', 'study'],
            'explain': ['explain', 'how', 'why', 'what is', 'describe']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent
        
        return 'find'  # Default intent
    
    def _extract_main_topic(self, query: str) -> str:
        """Extract main topic from query."""
        
        # Remove common filter words and operators
        filter_words = ['find', 'search', 'get', 'show', 'list', 'from', 'after', 'before', 'recent', 'latest']
        words = query.lower().split()
        topic_words = [w for w in words if w not in filter_words and len(w) > 2]
        
        # Return first few significant words as topic
        return ' '.join(topic_words[:5]) if topic_words else query
    
    def _determine_query_type(self, query: str, intent: str) -> QueryType:
        """Determine QueryType based on query and intent."""
        
        query_lower = query.lower()
        
        if intent == 'compare' or any(word in query_lower for word in ['vs', 'versus', 'compare', 'difference']):
            return QueryType.COMPARISON
        elif intent == 'explain' or any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'steps']):
            return QueryType.TUTORIAL
        elif any(word in query_lower for word in ['review', 'rating', 'opinion', 'recommend']):
            return QueryType.REVIEW
        elif any(word in query_lower for word in ['problem', 'issue', 'error', 'fix', 'troubleshoot']):
            return QueryType.TROUBLESHOOTING
        elif any(word in query_lower for word in ['news', 'latest', 'breaking', 'update']):
            return QueryType.NEWS
        elif any(word in query_lower for word in ['research', 'study', 'analysis', 'data']):
            return QueryType.RESEARCH
        elif any(word in query_lower for word in ['what', 'define', 'meaning', 'explanation']):
            return QueryType.FACTUAL
        else:
            return QueryType.GENERAL
    
    def _determine_complexity(self, query: str, filters: List[MetadataFilter]) -> str:
        """Determine query complexity level."""
        
        complexity_score = 0
        
        # Query length factor
        word_count = len(query.split())
        if word_count > 10:
            complexity_score += 1
        if word_count > 20:
            complexity_score += 1
        
        # Number of filters factor
        filter_count = len(filters)
        if filter_count > 2:
            complexity_score += 1
        if filter_count > 5:
            complexity_score += 1
        
        # Complex operators factor
        complex_operators = [FilterOperator.BETWEEN, FilterOperator.REGEX, FilterOperator.NOT_IN]
        if any(f.operator in complex_operators for f in filters):
            complexity_score += 1
        
        # Multiple filter types factor
        filter_types = set(f.field for f in filters)
        if len(filter_types) > 3:
            complexity_score += 1
        
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "simple"
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query analysis."""
        import hashlib
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _create_fallback_analysis(self, query: str) -> QueryAnalysis:
        """Create fallback analysis when other methods fail."""
        return QueryAnalysis(
            original_query=query,
            intent="find",
            main_topic=query,
            metadata_filters=[],
            temporal_constraints=[],
            categorical_constraints=[],
            query_type=QueryType.GENERAL,
            complexity_level="simple",
            confidence=0.1,
            extraction_method="fallback"
        )


class MetadataFilterEngine:
    """Engine for applying metadata filters to search results."""
    
    def __init__(self, config: SelfQueryConfig):
        """Initialize the metadata filter engine."""
        self.config = config
        
    async def apply_filters(
        self, 
        documents: List[Document], 
        filters: List[MetadataFilter],
        scope: FilterScope = FilterScope.BOTH
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Apply metadata filters to documents."""
        
        start_time = time.time()
        
        try:
            # Filter documents based on scope
            applicable_filters = [f for f in filters if f.scope == scope or f.scope == FilterScope.BOTH]
            
            if not applicable_filters:
                return documents, {'filters_applied': 0, 'processing_time_ms': 0}
            
            # Apply filters
            if self.config.enable_parallel_filtering and len(applicable_filters) > 1:
                filtered_docs = await self._apply_filters_parallel(documents, applicable_filters)
            else:
                filtered_docs = await self._apply_filters_sequential(documents, applicable_filters)
            
            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            
            metrics = {
                'filters_applied': len(applicable_filters),
                'documents_before': len(documents),
                'documents_after': len(filtered_docs),
                'reduction_ratio': 1.0 - (len(filtered_docs) / len(documents)) if documents else 0.0,
                'processing_time_ms': processing_time_ms,
                'filter_details': [
                    {
                        'field': f.field,
                        'operator': f.operator.value,
                        'value': f.value,
                        'confidence': f.confidence
                    }
                    for f in applicable_filters
                ]
            }
            
            return filtered_docs, metrics
            
        except Exception as e:
            logger.error(f"Filter application failed: {e}")
            return documents, {'error': str(e), 'filters_applied': 0}
    
    async def _apply_filters_parallel(
        self, 
        documents: List[Document], 
        filters: List[MetadataFilter]
    ) -> List[Document]:
        """Apply filters in parallel for better performance."""
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_filters)
        
        # Apply each filter
        async def apply_single_filter(filter_obj: MetadataFilter) -> Set[int]:
            async with semaphore:
                matching_indices = set()
                for i, doc in enumerate(documents):
                    if filter_obj.applies_to_document(doc):
                        matching_indices.add(i)
                return matching_indices
        
        # Execute all filters in parallel
        filter_tasks = [apply_single_filter(f) for f in filters]
        filter_results = await asyncio.gather(*filter_tasks)
        
        # Find intersection of all matching document indices
        if filter_results:
            matching_indices = filter_results[0]
            for result in filter_results[1:]:
                matching_indices &= result
        else:
            matching_indices = set()
        
        # Return filtered documents
        return [documents[i] for i in sorted(matching_indices)]
    
    async def _apply_filters_sequential(
        self, 
        documents: List[Document], 
        filters: List[MetadataFilter]
    ) -> List[Document]:
        """Apply filters sequentially."""
        
        current_docs = documents
        
        for filter_obj in filters:
            current_docs = [
                doc for doc in current_docs 
                if filter_obj.applies_to_document(doc)
            ]
        
        return current_docs


class SelfQueryRetriever:
    """
    Main self-query retrieval system with intelligent metadata filtering.
    
    Integrates query analysis, metadata filtering, and existing retrieval systems.
    """
    
    def __init__(
        self,
        hybrid_search_engine: Union[HybridSearchEngine, ContextualHybridSearch],
        multi_query_retriever: Optional[MultiQueryRetriever] = None,
        config: Optional[SelfQueryConfig] = None
    ):
        """Initialize the self-query retriever."""
        
        self.hybrid_search = hybrid_search_engine
        self.multi_query_retriever = multi_query_retriever
        self.config = config or SelfQueryConfig()
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer(self.config)
        self.filter_engine = MetadataFilterEngine(self.config)
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'successful_analyses': 0,
            'filters_applied': 0,
            'avg_filter_effectiveness': 0.0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.info("SelfQueryRetriever initialized")
    
    async def retrieve(
        self,
        query: str,
        max_results: int = 20,
        search_type: SearchType = SearchType.HYBRID,
        enable_multi_query: bool = None
    ) -> SelfQueryResults:
        """
        Perform self-query retrieval with intelligent metadata filtering.
        
        Args:
            query: Natural language query
            max_results: Maximum results to return
            search_type: Type of search to perform
            enable_multi_query: Whether to use multi-query expansion
            
        Returns:
            SelfQueryResults with filtered results and analysis
        """
        
        start_time = time.time()
        
        try:
            # Determine if multi-query should be used
            if enable_multi_query is None:
                enable_multi_query = (
                    self.config.integrate_with_multi_query and 
                    self.multi_query_retriever is not None
                )
            
            # Step 1: Analyze query to extract metadata filters
            query_analysis = await self.query_analyzer.analyze_query(query)
            
            # Step 2: Separate pre-search and post-search filters
            pre_search_filters = [
                f for f in query_analysis.metadata_filters 
                if f.scope in [FilterScope.PRE_SEARCH, FilterScope.BOTH]
            ]
            post_search_filters = [
                f for f in query_analysis.metadata_filters
                if f.scope in [FilterScope.POST_SEARCH, FilterScope.BOTH]
            ]
            
            # Step 3: Perform search (with or without multi-query)
            if enable_multi_query:
                search_results = await self._search_with_multi_query(
                    query, query_analysis, max_results, search_type
                )
            else:
                search_results = await self._search_basic(
                    query, max_results, search_type
                )
            
            # Step 4: Apply post-search filters
            if post_search_filters:
                # Convert search results to documents for filtering
                result_documents = [result.document for result in search_results.results]
                
                # Apply filters
                filtered_documents, filter_metrics = await self.filter_engine.apply_filters(
                    result_documents, post_search_filters, FilterScope.POST_SEARCH
                )
                
                # Reconstruct search results
                doc_to_result = {id(result.document): result for result in search_results.results}
                filtered_results = [
                    doc_to_result[id(doc)] for doc in filtered_documents 
                    if id(doc) in doc_to_result
                ]
            else:
                filtered_results = search_results.results
                filter_metrics = {'filters_applied': 0}
            
            # Step 5: Calculate performance metrics
            total_time_ms = (time.time() - start_time) * 1000
            
            # Step 6: Build results
            self_query_results = SelfQueryResults(
                filtered_results=filtered_results,
                original_results=search_results.results,
                query_analysis=query_analysis,
                applied_filters=query_analysis.metadata_filters,
                total_documents_before=len(search_results.results),
                total_documents_after=len(filtered_results),
                filter_effectiveness=filter_metrics.get('reduction_ratio', 0.0),
                processing_time_ms=total_time_ms,
                pre_search_filters=pre_search_filters,
                post_search_filters=post_search_filters,
                relevance_improvement=self._calculate_relevance_improvement(
                    search_results.results, filtered_results
                ),
                precision_score=self._calculate_precision_score(filtered_results),
                recall_impact=self._calculate_recall_impact(
                    search_results.results, filtered_results
                )
            )
            
            # Update performance statistics
            await self._update_performance_stats(self_query_results)
            
            return self_query_results
            
        except Exception as e:
            logger.error(f"Self-query retrieval failed: {e}")
            
            # Fallback to basic search
            basic_results = await self._search_basic(query, max_results, search_type)
            
            return SelfQueryResults(
                filtered_results=basic_results.results,
                original_results=basic_results.results,
                query_analysis=QueryAnalysis(
                    original_query=query,
                    intent="find",
                    main_topic=query,
                    metadata_filters=[],
                    temporal_constraints=[],
                    categorical_constraints=[],
                    query_type=QueryType.GENERAL,
                    complexity_level="simple",
                    confidence=0.0
                ),
                applied_filters=[],
                total_documents_before=len(basic_results.results),
                total_documents_after=len(basic_results.results),
                filter_effectiveness=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                pre_search_filters=[],
                post_search_filters=[]
            )
    
    async def _search_with_multi_query(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        max_results: int,
        search_type: SearchType
    ) -> MultiQueryResults:
        """Perform search using multi-query retrieval."""
        
        return await self.multi_query_retriever.retrieve(
            query=query,
            query_type=query_analysis.query_type,
            search_type=search_type,
            max_results=max_results
        )
    
    async def _search_basic(
        self,
        query: str,
        max_results: int,
        search_type: SearchType
    ) -> HybridSearchResults:
        """Perform basic hybrid search."""
        
        return await self.hybrid_search.search(
            query=query,
            search_type=search_type,
            max_results=max_results
        )
    
    def _calculate_relevance_improvement(
        self, 
        original_results: List[SearchResult], 
        filtered_results: List[SearchResult]
    ) -> float:
        """Calculate relevance improvement from filtering."""
        
        if not filtered_results or not original_results:
            return 0.0
        
        # Calculate average scores
        original_avg = sum(r.hybrid_score for r in original_results) / len(original_results)
        filtered_avg = sum(r.hybrid_score for r in filtered_results) / len(filtered_results)
        
        # Return relative improvement
        if original_avg > 0:
            return (filtered_avg - original_avg) / original_avg
        else:
            return 0.0
    
    def _calculate_precision_score(self, filtered_results: List[SearchResult]) -> float:
        """Calculate precision score based on result quality."""
        
        if not filtered_results:
            return 0.0
        
        # Use hybrid scores as proxy for precision
        scores = [result.hybrid_score for result in filtered_results]
        return sum(scores) / len(scores)
    
    def _calculate_recall_impact(
        self, 
        original_results: List[SearchResult], 
        filtered_results: List[SearchResult]
    ) -> float:
        """Calculate impact on recall from filtering."""
        
        if not original_results:
            return 0.0
        
        # Simple recall impact based on result count reduction
        recall_ratio = len(filtered_results) / len(original_results)
        return 1.0 - recall_ratio  # Higher values indicate more recall impact
    
    async def _update_performance_stats(self, results: SelfQueryResults):
        """Update performance statistics."""
        
        self.performance_stats['total_queries'] += 1
        
        if results.query_analysis.confidence > 0.5:
            self.performance_stats['successful_analyses'] += 1
        
        if results.applied_filters:
            self.performance_stats['filters_applied'] += len(results.applied_filters)
        
        # Update running averages
        total = self.performance_stats['total_queries']
        
        self.performance_stats['avg_filter_effectiveness'] = (
            (self.performance_stats['avg_filter_effectiveness'] * (total - 1) + 
             results.filter_effectiveness) / total
        )
        
        self.performance_stats['avg_processing_time_ms'] = (
            (self.performance_stats['avg_processing_time_ms'] * (total - 1) + 
             results.processing_time_ms) / total
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        total = self.performance_stats['total_queries']
        
        return {
            'total_queries': total,
            'successful_analyses': self.performance_stats['successful_analyses'],
            'analysis_success_rate': (
                self.performance_stats['successful_analyses'] / total if total > 0 else 0.0
            ),
            'total_filters_applied': self.performance_stats['filters_applied'],
            'avg_filters_per_query': (
                self.performance_stats['filters_applied'] / total if total > 0 else 0.0
            ),
            'avg_filter_effectiveness': self.performance_stats['avg_filter_effectiveness'],
            'avg_processing_time_ms': self.performance_stats['avg_processing_time_ms']
        }
    
    async def clear_caches(self):
        """Clear all caches."""
        
        self.query_analyzer.analysis_cache.clear()
        logger.info("Self-query caches cleared") 