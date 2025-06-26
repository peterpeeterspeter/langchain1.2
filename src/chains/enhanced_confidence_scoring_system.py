"""
Enhanced Response and Confidence Scoring System

This module provides advanced multi-factor confidence calculation, intelligent caching,
and comprehensive quality assessment for RAG applications.

Key Features:
- 6-factor confidence scoring with adaptive weights
- Advanced source quality assessment with 8 quality indicators
- Intelligent caching with query-pattern learning
- Response validation with content quality checks
- Enhanced error recovery with confidence-based fallbacks
- Real-time performance monitoring and optimization
"""

import asyncio
import time
import logging
import re
import math
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import statistics
from collections import defaultdict, deque
import random

from pydantic import BaseModel, Field, validator
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# Set up logger
logger = logging.getLogger(__name__)

# Import advanced prompt system components
try:
    from .advanced_prompt_system import (
        OptimizedPromptManager, QueryAnalysis, QueryType, 
        ExpertiseLevel, ResponseFormat, AdvancedContextFormatter,
        EnhancedSourceFormatter
    )
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    logging.warning("Advanced prompt system not available. Enhanced scoring will use basic mode.")
    PROMPT_OPTIMIZATION_AVAILABLE = False


# Enhanced enums for quality assessment
class SourceQualityTier(Enum):
    """Source quality classification tiers"""
    PREMIUM = "premium"      # 0.9-1.0
    HIGH = "high"           # 0.8-0.89
    GOOD = "good"           # 0.7-0.79
    MODERATE = "moderate"   # 0.6-0.69
    LOW = "low"             # 0.5-0.59
    POOR = "poor"           # 0.0-0.49


class ResponseQualityLevel(Enum):
    """Response quality classification levels"""
    EXCELLENT = "excellent"     # 0.9-1.0
    VERY_GOOD = "very_good"     # 0.8-0.89
    GOOD = "good"               # 0.7-0.79
    SATISFACTORY = "satisfactory" # 0.6-0.69
    POOR = "poor"               # 0.5-0.59
    UNACCEPTABLE = "unacceptable" # 0.0-0.49


class CacheStrategy(Enum):
    """Advanced caching strategies"""
    CONSERVATIVE = "conservative"  # Longer TTL, higher quality threshold
    BALANCED = "balanced"         # Standard TTL and thresholds
    AGGRESSIVE = "aggressive"     # Shorter TTL, lower quality threshold
    ADAPTIVE = "adaptive"        # Learning-based TTL optimization


class ConfidenceFactorType(Enum):
    """Types of confidence factors for categorization"""
    CONTENT_QUALITY = "content_quality"
    SOURCE_QUALITY = "source_quality"
    QUERY_MATCHING = "query_matching"
    TECHNICAL_FACTORS = "technical_factors"


# Enhanced response model with comprehensive metadata
class EnhancedRAGResponse(BaseModel):
    """Enhanced RAG response with advanced confidence and quality metrics"""
    
    # Core response data
    content: str
    sources: List[Dict[str, Any]]
    
    # Confidence and quality metrics
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    confidence_factors: Optional['ConfidenceFactors'] = None
    confidence_breakdown: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Detailed breakdown of confidence factors"
    )
    source_quality_tier: SourceQualityTier = SourceQualityTier.MODERATE
    response_quality_level: ResponseQualityLevel = ResponseQualityLevel.SATISFACTORY
    processing_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    cached: bool = False
    response_time: float = Field(description="Total response time in seconds")
    
    # Query analysis and optimization
    query_analysis: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Query analysis results if available"
    )
    optimization_enabled: bool = True
    
    # Source quality metrics
    avg_source_quality: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Average quality score of sources"
    )
    source_diversity_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Diversity score of source types"
    )
    retrieval_coverage: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Coverage of retrieval results"
    )
    
    # Validation results
    format_validation: Dict[str, bool] = Field(
        default_factory=dict, 
        description="Format validation results"
    )
    content_validation: Dict[str, bool] = Field(
        default_factory=dict, 
        description="Content validation results"
    )
    
    # Error handling and fallbacks
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    fallback_used: bool = Field(default=False, description="Whether fallback mechanisms were used")
    
    # Cache metadata
    cache_metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Cache-related metadata"
    )
    
    # Performance tracking
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, 
        description="Token usage statistics"
    )
    retrieval_time: Optional[float] = Field(
        default=None, 
        description="Time spent on retrieval"
    )
    generation_time: Optional[float] = Field(
        default=None, 
        description="Time spent on generation"
    )
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Ensure confidence score is within valid range"""
        return max(0.0, min(1.0, v))
    
    @validator('response_quality_level', pre=True, always=True)
    def set_quality_level_from_confidence(cls, v, values):
        """Automatically set quality level based on confidence score if not provided"""
        if v == ResponseQualityLevel.SATISFACTORY and 'confidence_score' in values:
            confidence = values['confidence_score']
            if confidence >= 0.9:
                return ResponseQualityLevel.EXCELLENT
            elif confidence >= 0.8:
                return ResponseQualityLevel.VERY_GOOD
            elif confidence >= 0.7:
                return ResponseQualityLevel.GOOD
            elif confidence >= 0.6:
                return ResponseQualityLevel.SATISFACTORY
            elif confidence >= 0.5:
                return ResponseQualityLevel.POOR
            else:
                return ResponseQualityLevel.UNACCEPTABLE
        return v


@dataclass
class ConfidenceFactors:
    """Comprehensive confidence factors for multi-dimensional assessment"""
    
    # Content Quality Factors (3 factors)
    completeness: float = 0.5  # How complete is the response
    relevance: float = 0.5     # How relevant to the query
    accuracy_indicators: float = 0.5  # Indicators of factual accuracy
    
    # Source Quality Factors (3 factors)
    source_reliability: float = 0.5    # Reliability of sources used
    source_coverage: float = 0.5       # Coverage and diversity of sources
    source_consistency: float = 0.5    # Consistency between sources
    
    # Query Matching Factors (3 factors)
    intent_alignment: float = 0.5      # Alignment with user intent
    expertise_match: float = 0.5       # Match with required expertise level
    format_appropriateness: float = 0.5 # Appropriateness of response format
    
    # Technical Quality Factors (3 factors)
    retrieval_quality: float = 0.5     # Quality of retrieval process
    generation_stability: float = 0.5   # Stability of generation process
    optimization_effectiveness: float = 0.5  # Effectiveness of optimizations
    
    def get_weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted confidence score"""
        if weights is None:
            # Default adaptive weights
            weights = {
                'content': 0.35,     # Content quality (completeness, relevance, accuracy)
                'sources': 0.25,     # Source quality (reliability, coverage, consistency)
                'matching': 0.20,    # Query matching (intent, expertise, format)
                'technical': 0.20    # Technical factors (retrieval, generation, optimization)
            }
        
        # Calculate category scores
        content_score = (self.completeness + self.relevance + self.accuracy_indicators) / 3
        source_score = (self.source_reliability + self.source_coverage + self.source_consistency) / 3
        matching_score = (self.intent_alignment + self.expertise_match + self.format_appropriateness) / 3
        technical_score = (self.retrieval_quality + self.generation_stability + self.optimization_effectiveness) / 3
        
        # Apply weights
        weighted_score = (
            content_score * weights.get('content', 0.35) +
            source_score * weights.get('sources', 0.25) +
            matching_score * weights.get('matching', 0.20) +
            technical_score * weights.get('technical', 0.20)
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def get_factor_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown of all factors"""
        return {
            "content_quality": {
                "completeness": self.completeness,
                "relevance": self.relevance,
                "accuracy_indicators": self.accuracy_indicators
            },
            "source_quality": {
                "reliability": self.source_reliability,
                "coverage": self.source_coverage,
                "consistency": self.source_consistency
            },
            "query_matching": {
                "intent_alignment": self.intent_alignment,
                "expertise_match": self.expertise_match,
                "format_appropriateness": self.format_appropriateness
            },
            "technical_factors": {
                "retrieval_quality": self.retrieval_quality,
                "generation_stability": self.generation_stability,
                "optimization_effectiveness": self.optimization_effectiveness
            }
        }
    
    def validate_factors(self) -> List[str]:
        """Validate all factors are within acceptable ranges"""
        issues = []
        factors = [
            ('completeness', self.completeness),
            ('relevance', self.relevance),
            ('accuracy_indicators', self.accuracy_indicators),
            ('source_reliability', self.source_reliability),
            ('source_coverage', self.source_coverage),
            ('source_consistency', self.source_consistency),
            ('intent_alignment', self.intent_alignment),
            ('expertise_match', self.expertise_match),
            ('format_appropriateness', self.format_appropriateness),
            ('retrieval_quality', self.retrieval_quality),
            ('generation_stability', self.generation_stability),
            ('optimization_effectiveness', self.optimization_effectiveness)
        ]
        
        for name, value in factors:
            if not (0.0 <= value <= 1.0):
                issues.append(f"{name} value {value} is outside valid range [0.0, 1.0]")
        
        return issues


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata and quality tracking"""
    
    key: str
    response: EnhancedRAGResponse
    created_at: datetime
    expires_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_hours: int = 24
    quality_score: float = 0.0
    query_pattern: Optional[str] = None
    cache_value_score: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() > self.expires_at
    
    def is_stale(self, staleness_threshold_hours: int = 24) -> bool:
        """Check if cache entry is stale based on age"""
        age = datetime.now() - self.created_at
        return age > timedelta(hours=staleness_threshold_hours)
    
    def update_access(self):
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_cache_value_score(self) -> float:
        """Calculate cache value score based on quality, access frequency, and age"""
        # Base score from quality
        base_score = self.quality_score
        
        # Frequency bonus (logarithmic scaling)
        frequency_bonus = min(0.2, math.log(self.access_count + 1) / 10)
        
        # Age penalty (linear decay over 7 days)
        age_hours = (datetime.now() - self.created_at).total_seconds() / 3600
        age_penalty = min(0.3, age_hours / (7 * 24) * 0.3)
        
        return max(0.0, base_score + frequency_bonus - age_penalty)


@dataclass
class SystemConfiguration:
    """Configuration for the enhanced confidence scoring system"""
    
    # Confidence scoring settings
    enable_enhanced_confidence: bool = True
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'content': 0.35,
        'sources': 0.25,
        'matching': 0.20,
        'technical': 0.20
    })
    
    # Caching settings
    enable_intelligent_caching: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    cache_max_size: int = 10000
    default_ttl_hours: int = 24
    quality_threshold: float = 0.6
    
    # Performance settings
    max_response_time: float = 2.0
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    # Validation settings
    enable_response_validation: bool = True
    min_response_length: int = 50
    max_response_length: int = 5000
    
    # Source quality settings
    min_source_quality: float = 0.3
    preferred_source_count: int = 5
    source_diversity_weight: float = 0.2
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration settings"""
        issues = []
        
        # Validate weights sum to approximately 1.0
        weight_sum = sum(self.confidence_weights.values())
        if not (0.95 <= weight_sum <= 1.05):
            issues.append(f"Confidence weights sum to {weight_sum:.3f}, should be close to 1.0")
        
        # Validate thresholds
        if not (0.0 <= self.quality_threshold <= 1.0):
            issues.append(f"Quality threshold {self.quality_threshold} must be between 0.0 and 1.0")
        
        if self.max_response_time <= 0:
            issues.append("Max response time must be positive")
        
        if self.cache_max_size <= 0:
            issues.append("Cache max size must be positive")
        
        return issues


# Utility functions for the enhanced system
def calculate_quality_tier(score: float) -> SourceQualityTier:
    """Calculate quality tier from numeric score"""
    if score >= 0.9:
        return SourceQualityTier.PREMIUM
    elif score >= 0.8:
        return SourceQualityTier.HIGH
    elif score >= 0.7:
        return SourceQualityTier.GOOD
    elif score >= 0.6:
        return SourceQualityTier.MODERATE
    elif score >= 0.5:
        return SourceQualityTier.LOW
    else:
        return SourceQualityTier.POOR


def generate_query_hash(query: str, query_analysis: Optional[Any] = None) -> str:
    """Generate consistent hash for query caching"""
    # Include query text and analysis type for more specific caching
    hash_input = query.lower().strip()
    
    if query_analysis and hasattr(query_analysis, 'query_type'):
        hash_input += f"__{query_analysis.query_type.value}"
    
    if query_analysis and hasattr(query_analysis, 'expertise_level'):
        hash_input += f"__{query_analysis.expertise_level.value}"
    
    return hashlib.md5(hash_input.encode()).hexdigest()


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to specified range"""
    return max(min_val, min(max_val, score))


# Source Quality Analysis System
class SourceQualityAnalyzer:
    """
    Comprehensive source quality analyzer with 8 quality indicators
    
    Evaluates sources across multiple dimensions:
    - Authority: Official status, licensing, regulation
    - Credibility: Verification, reviews, trust signals
    - Expertise: Professional knowledge, specialization
    - Recency: Freshness, currency, updates
    - Detail: Completeness, thoroughness, depth
    - Objectivity: Neutrality, bias assessment
    - Transparency: Disclosure, openness, clarity
    - Citation: References, sources, evidence
    """
    
    def __init__(self):
        """Initialize the source quality analyzer with indicator patterns"""
        
        # Quality indicator patterns for content analysis
        self.quality_indicators = {
            'authority': {
                'patterns': [
                    'official', 'licensed', 'regulated', 'certified', 'authorized',
                    'government', 'federal', 'state', 'municipal', 'regulatory',
                    'commission', 'department', 'agency', 'bureau', 'ministry',
                    'accredited', 'approved', 'sanctioned', 'endorsed'
                ],
                'weight': 0.15,
                'baseline': 0.5
            },
            'credibility': {
                'patterns': [
                    'verified', 'reviewed', 'endorsed', 'trusted', 'reputable',
                    'established', 'recognized', 'respected', 'reliable',
                    'peer-reviewed', 'fact-checked', 'validated', 'confirmed',
                    'authenticated', 'substantiated', 'corroborated'
                ],
                'weight': 0.15,
                'baseline': 0.5
            },
            'expertise': {
                'patterns': [
                    'expert', 'professional', 'specialist', 'authority', 'experienced',
                    'phd', 'doctor', 'professor', 'researcher', 'analyst',
                    'consultant', 'advisor', 'scholar', 'academic', 'scientist',
                    'certified', 'qualified', 'trained', 'skilled'
                ],
                'weight': 0.15,
                'baseline': 0.5
            },
            'recency': {
                'patterns': [
                    '2024', '2025', 'recent', 'latest', 'updated', 'current',
                    'new', 'fresh', 'modern', 'contemporary', 'today',
                    'this year', 'this month', 'recently', 'just published'
                ],
                'weight': 0.10,
                'baseline': 0.4  # Lower baseline for recency
            },
            'detail': {
                'patterns': [
                    'comprehensive', 'detailed', 'thorough', 'complete', 'extensive',
                    'in-depth', 'exhaustive', 'full', 'elaborate', 'specific',
                    'precise', 'exact', 'particular', 'meticulous', 'systematic'
                ],
                'weight': 0.12,
                'baseline': 0.5
            },
            'objectivity': {
                'patterns': [
                    'unbiased', 'neutral', 'objective', 'balanced', 'fair',
                    'impartial', 'even-handed', 'factual', 'evidence-based',
                    'data-driven', 'analytical', 'scientific', 'empirical'
                ],
                'weight': 0.13,
                'baseline': 0.5
            },
            'transparency': {
                'patterns': [
                    'disclosure', 'transparent', 'open', 'clear', 'honest',
                    'forthright', 'candid', 'straightforward', 'explicit',
                    'methodology', 'limitations', 'assumptions', 'sources'
                ],
                'weight': 0.10,
                'baseline': 0.5
            },
            'citation': {
                'patterns': [
                    'source', 'reference', 'citation', 'study', 'research',
                    'bibliography', 'footnote', 'endnote', 'link', 'url',
                    'according to', 'based on', 'cited in', 'referenced by'
                ],
                'weight': 0.10,
                'baseline': 0.5
            }
        }
        
        # Negative indicators that reduce quality scores
        self.negative_indicators = {
            'patterns': [
                'unverified', 'speculation', 'rumor', 'alleged', 'outdated',
                'biased', 'promotional', 'advertisement', 'opinion', 'personal',
                'anecdotal', 'hearsay', 'gossip', 'unsubstantiated', 'dubious',
                'questionable', 'suspicious', 'misleading', 'false', 'fake'
            ],
            'penalty_factor': 0.2  # Reduce score by 20% per negative indicator
        }
        
        # Domain-specific authority sources
        self.authority_domains = {
            'government': ['gov', 'mil', 'state', 'federal'],
            'academic': ['edu', 'university', 'college', 'institute'],
            'medical': ['nih', 'cdc', 'who', 'medical', 'health'],
            'financial': ['sec', 'fed', 'treasury', 'bank', 'finance'],
            'legal': ['court', 'law', 'legal', 'justice', 'attorney']
        }
        
        # Time decay factors for recency scoring
        self.recency_decay = {
            'days': 0.99,      # 1% decay per day
            'weeks': 0.95,     # 5% decay per week  
            'months': 0.85,    # 15% decay per month
            'years': 0.70      # 30% decay per year
        }
    
    async def analyze_source_quality(self, document: Document, query_context: str = "") -> Dict[str, Any]:
        """
        Comprehensive source quality analysis using 8 quality indicators
        
        Args:
            document: Document to analyze
            query_context: Optional query context for relevance assessment
            
        Returns:
            Dict containing quality scores, tier, and detailed analysis
        """
        try:
            # Extract content and metadata with None check
            content = (document.page_content or "").lower()
            metadata = document.metadata or {}
            
            # Calculate individual quality indicators
            quality_scores = {}
            found_indicators = {}
            
            # 1. Authority Assessment
            authority_score, authority_indicators = self._assess_authority(content, metadata)
            quality_scores['authority'] = authority_score
            found_indicators['authority'] = authority_indicators
            
            # 2. Credibility Evaluation
            credibility_score, credibility_indicators = self._assess_credibility(content, metadata)
            quality_scores['credibility'] = credibility_score
            found_indicators['credibility'] = credibility_indicators
            
            # 3. Expertise Analysis
            expertise_score, expertise_indicators = self._assess_expertise(content, metadata)
            quality_scores['expertise'] = expertise_score
            found_indicators['expertise'] = expertise_indicators
            
            # 4. Recency Scoring
            recency_score, recency_indicators = self._assess_recency(content, metadata)
            quality_scores['recency'] = recency_score
            found_indicators['recency'] = recency_indicators
            
            # 5. Detail and Completeness
            detail_score, detail_indicators = self._assess_detail(content, metadata)
            quality_scores['detail'] = detail_score
            found_indicators['detail'] = detail_indicators
            
            # 6. Objectivity Assessment
            objectivity_score, objectivity_indicators = self._assess_objectivity(content, metadata)
            quality_scores['objectivity'] = objectivity_score
            found_indicators['objectivity'] = objectivity_indicators
            
            # 7. Transparency Evaluation
            transparency_score, transparency_indicators = self._assess_transparency(content, metadata)
            quality_scores['transparency'] = transparency_score
            found_indicators['transparency'] = transparency_indicators
            
            # 8. Citation Quality
            citation_score, citation_indicators = self._assess_citation_quality(content, metadata)
            quality_scores['citation'] = citation_score
            found_indicators['citation'] = citation_indicators
            
            # Apply negative indicator penalties
            negative_markers = self._detect_negative_indicators(content)
            penalty_factor = 1.0 - (len(negative_markers) * self.negative_indicators['penalty_factor'])
            penalty_factor = max(0.1, penalty_factor)  # Minimum 10% of original score
            
            # Calculate weighted composite score
            overall_quality = self._calculate_composite_score(quality_scores, penalty_factor)
            
            # Determine quality tier
            quality_tier = calculate_quality_tier(overall_quality)
            
            # Assess metadata quality
            metadata_assessment = self._assess_metadata_quality(metadata)
            
            # Generate comprehensive quality report
            quality_report = {
                'overall_quality': overall_quality,
                'quality_tier': quality_tier,
                'quality_components': quality_scores,
                'quality_indicators': found_indicators,
                'negative_indicators': negative_markers,
                'metadata_quality': metadata_assessment,
                'penalty_applied': 1.0 - penalty_factor,
                'analysis_timestamp': datetime.now().isoformat(),
                'content_length': len(document.page_content),
                'has_metadata': bool(metadata)
            }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error in source quality analysis: {str(e)}")
            return self._get_fallback_quality_assessment()
    
    def _assess_authority(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess authority indicators in content and metadata"""
        indicators = self.quality_indicators['authority']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check content for authority patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.05  # 5% boost per authority indicator
        
        # Check metadata for authority signals
        url = metadata.get('url', '').lower()
        source = metadata.get('source', '').lower()
        
        # Domain authority assessment
        for domain_type, domains in self.authority_domains.items():
            for domain in domains:
                if domain in url or domain in source:
                    found_patterns.append(f"{domain_type}_domain")
                    score += 0.1  # 10% boost for authoritative domains
        
        # Title and source authority
        title = metadata.get('title', '').lower()
        if any(auth in title for auth in ['official', 'government', 'regulatory']):
            found_patterns.append('authoritative_title')
            score += 0.08
        
        return normalize_score(score), found_patterns
    
    def _assess_credibility(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess credibility indicators and trust signals"""
        indicators = self.quality_indicators['credibility']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for credibility patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.04  # 4% boost per credibility indicator
        
        # Check for verification signals in metadata
        if metadata.get('verified', False):
            found_patterns.append('metadata_verified')
            score += 0.15
        
        # Check for review indicators
        if 'review' in content or 'reviewed' in content:
            found_patterns.append('review_process')
            score += 0.08
        
        # Check for trust signals in URL/source
        url = metadata.get('url', '').lower()
        if any(trust in url for trust in ['trusted', 'verified', 'official']):
            found_patterns.append('trusted_domain')
            score += 0.1
        
        return normalize_score(score), found_patterns
    
    def _assess_expertise(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess expertise level and professional knowledge indicators"""
        indicators = self.quality_indicators['expertise']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for expertise patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.04  # 4% boost per expertise indicator
        
        # Check author credentials in metadata
        author = metadata.get('author', '').lower()
        if author:
            if any(cred in author for cred in ['dr.', 'phd', 'professor', 'md']):
                found_patterns.append('author_credentials')
                score += 0.12
        
        # Check for technical terminology (indicates expertise)
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', content))  # Acronyms
        if technical_terms > 5:
            found_patterns.append('technical_terminology')
            score += min(0.1, technical_terms * 0.01)  # Up to 10% boost
        
        # Check for professional language patterns
        professional_patterns = ['methodology', 'analysis', 'research', 'study', 'findings']
        professional_count = sum(1 for pattern in professional_patterns if pattern in content)
        if professional_count >= 3:
            found_patterns.append('professional_language')
            score += 0.08
        
        return normalize_score(score), found_patterns
    
    def _assess_recency(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess content recency and freshness"""
        indicators = self.quality_indicators['recency']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for recency patterns in content
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.05  # 5% boost per recency indicator
        
        # Parse publication date from metadata
        pub_date = metadata.get('published_date') or metadata.get('date')
        if pub_date:
            try:
                if isinstance(pub_date, str):
                    # Try to parse various date formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            date_obj = datetime.strptime(pub_date, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If no format matches, try parsing year only
                        year_match = re.search(r'20\d{2}', pub_date)
                        if year_match:
                            date_obj = datetime(int(year_match.group()), 1, 1)
                        else:
                            date_obj = None
                else:
                    date_obj = pub_date
                
                if date_obj:
                    # Calculate time-based decay
                    age = datetime.now() - date_obj
                    days_old = age.days
                    
                    if days_old <= 30:  # Within a month
                        found_patterns.append('very_recent')
                        score += 0.2
                    elif days_old <= 365:  # Within a year
                        found_patterns.append('recent')
                        decay = (365 - days_old) / 365 * 0.15
                        score += decay
                    else:  # Older than a year
                        years_old = days_old / 365
                        decay = max(0, 0.1 - (years_old * 0.02))
                        score += decay
                        if years_old > 5:
                            found_patterns.append('potentially_outdated')
                            
            except Exception as e:
                logger.debug(f"Error parsing date {pub_date}: {e}")
        
        # Check for update indicators
        if any(update in content for update in ['updated', 'revised', 'amended']):
            found_patterns.append('content_updated')
            score += 0.08
        
        return normalize_score(score), found_patterns
    
    def _assess_detail(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess content detail, completeness, and thoroughness"""
        indicators = self.quality_indicators['detail']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for detail patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.04  # 4% boost per detail indicator
        
        # Content length assessment
        content_length = len(content)
        if content_length > 2000:  # Substantial content
            found_patterns.append('substantial_content')
            score += 0.1
        elif content_length > 1000:  # Moderate content
            found_patterns.append('moderate_content')
            score += 0.05
        elif content_length < 200:  # Very brief content
            found_patterns.append('brief_content')
            score -= 0.1
        
        # Check for structured content (lists, sections, etc.)
        structure_indicators = content.count('\n') + content.count('•') + content.count('-')
        if structure_indicators > 10:
            found_patterns.append('well_structured')
            score += 0.08
        
        # Check for specific details (numbers, percentages, dates)
        detail_patterns = len(re.findall(r'\d+%|\$\d+|\d{4}', content))
        if detail_patterns > 5:
            found_patterns.append('specific_details')
            score += min(0.1, detail_patterns * 0.01)
        
        return normalize_score(score), found_patterns
    
    def _assess_objectivity(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess content objectivity and bias indicators"""
        indicators = self.quality_indicators['objectivity']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for objectivity patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.05  # 5% boost per objectivity indicator
        
        # Check for bias indicators (negative impact)
        bias_patterns = ['i think', 'i believe', 'in my opinion', 'personally', 'obviously']
        bias_count = sum(1 for pattern in bias_patterns if pattern in content)
        if bias_count > 0:
            found_patterns.append('opinion_language')
            score -= bias_count * 0.03  # 3% penalty per bias indicator
        
        # Check for promotional language (negative impact)
        promotional_patterns = ['best', 'amazing', 'incredible', 'must-have', 'guaranteed']
        promo_count = sum(1 for pattern in promotional_patterns if pattern in content)
        if promo_count > 2:
            found_patterns.append('promotional_language')
            score -= 0.1
        
        # Check for balanced presentation
        if 'however' in content or 'on the other hand' in content:
            found_patterns.append('balanced_presentation')
            score += 0.08
        
        # Check for data-driven language
        data_patterns = ['data shows', 'statistics indicate', 'research suggests', 'studies show']
        data_count = sum(1 for pattern in data_patterns if pattern in content)
        if data_count > 0:
            found_patterns.append('data_driven')
            score += data_count * 0.04
        
        return normalize_score(score), found_patterns
    
    def _assess_transparency(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess transparency and disclosure indicators"""
        indicators = self.quality_indicators['transparency']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for transparency patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.05  # 5% boost per transparency indicator
        
        # Check for methodology disclosure
        if 'methodology' in content or 'method' in content:
            found_patterns.append('methodology_disclosed')
            score += 0.1
        
        # Check for limitations acknowledgment
        if 'limitation' in content or 'caveat' in content:
            found_patterns.append('limitations_acknowledged')
            score += 0.08
        
        # Check for conflict of interest disclosure
        if 'conflict of interest' in content or 'disclosure' in content:
            found_patterns.append('conflict_disclosure')
            score += 0.12
        
        # Check for author information in metadata
        if metadata.get('author'):
            found_patterns.append('author_identified')
            score += 0.06
        
        # Check for source attribution
        if metadata.get('source') or metadata.get('publisher'):
            found_patterns.append('source_attributed')
            score += 0.06
        
        return normalize_score(score), found_patterns
    
    def _assess_citation_quality(self, content: str, metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Assess citation and reference quality"""
        indicators = self.quality_indicators['citation']
        baseline = indicators['baseline']
        
        found_patterns = []
        score = baseline
        
        # Check for citation patterns
        for pattern in indicators['patterns']:
            if pattern in content:
                found_patterns.append(pattern)
                score += 0.04  # 4% boost per citation indicator
        
        # Count potential citations and references
        citation_count = (
            content.count('http') + 
            content.count('www.') + 
            content.count('[') +  # Potential reference markers
            content.count('(') +  # Potential inline citations
            len(re.findall(r'\b\d{4}\b', content))  # Years (potential publication dates)
        )
        
        if citation_count > 10:
            found_patterns.append('well_cited')
            score += 0.15
        elif citation_count > 5:
            found_patterns.append('moderately_cited')
            score += 0.08
        elif citation_count < 2:
            found_patterns.append('poorly_cited')
            score -= 0.1
        
        # Check for academic citation patterns
        academic_patterns = ['et al.', 'ibid', 'op. cit.', 'doi:', 'pmid:']
        academic_count = sum(1 for pattern in academic_patterns if pattern in content)
        if academic_count > 0:
            found_patterns.append('academic_citations')
            score += academic_count * 0.05
        
        # Check for reference list or bibliography
        if 'references' in content or 'bibliography' in content:
            found_patterns.append('reference_list')
            score += 0.12
        
        return normalize_score(score), found_patterns
    
    def _detect_negative_indicators(self, content: str) -> List[str]:
        """Detect negative quality indicators that reduce overall score"""
        negative_markers = []
        
        for pattern in self.negative_indicators['patterns']:
            if pattern in content:
                negative_markers.append(pattern)
        
        return negative_markers
    
    def _calculate_composite_score(self, quality_scores: Dict[str, float], penalty_factor: float) -> float:
        """Calculate weighted composite quality score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for indicator, score in quality_scores.items():
            weight = self.quality_indicators[indicator]['weight']
            weighted_sum += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite_score = weighted_sum / total_weight
        else:
            composite_score = 0.5  # Default baseline
        
        # Apply penalty factor for negative indicators
        final_score = composite_score * penalty_factor
        
        return normalize_score(final_score)
    
    def _assess_metadata_quality(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and completeness of metadata"""
        quality_factors = {
            'has_title': bool(metadata.get('title')),
            'has_author': bool(metadata.get('author')),
            'has_date': bool(metadata.get('date') or metadata.get('published_date')),
            'has_source': bool(metadata.get('source') or metadata.get('publisher')),
            'has_url': bool(metadata.get('url')),
            'metadata_completeness': 0.0
        }
        
        # Calculate completeness score
        completeness = sum(quality_factors[key] for key in quality_factors if key.startswith('has_'))
        quality_factors['metadata_completeness'] = completeness / 5.0  # 5 key metadata fields
        
        return quality_factors
    
    def _get_fallback_quality_assessment(self) -> Dict[str, Any]:
        """Return fallback quality assessment in case of errors"""
        return {
            'overall_quality': 0.5,
            'quality_tier': SourceQualityTier.MODERATE,
            'quality_components': {indicator: 0.5 for indicator in self.quality_indicators.keys()},
            'quality_indicators': {indicator: [] for indicator in self.quality_indicators.keys()},
            'negative_indicators': [],
            'metadata_quality': {
                'has_title': False,
                'has_author': False,
                'has_date': False,
                'has_source': False,
                'has_url': False,
                'metadata_completeness': 0.0
            },
            'penalty_applied': 0.0,
            'analysis_timestamp': datetime.now().isoformat(),
            'content_length': 0,
            'has_metadata': False,
            'error': 'Fallback assessment due to analysis error'
        }


# Performance monitoring utilities
class PerformanceTracker:
    """Track performance metrics for the enhanced system"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0,
            'avg_confidence_score': 0.0,
            'error_count': 0,
            'quality_distribution': defaultdict(int)
        }
        self.response_times = deque(maxlen=1000)
        self.confidence_scores = deque(maxlen=1000)
    
    def record_request(self, response: EnhancedRAGResponse):
        """Record metrics for a request"""
        self.metrics['total_requests'] += 1
        
        # Track cache performance
        if response.cached:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
        
        # Track response time
        self.response_times.append(response.response_time)
        self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
        
        # Track confidence scores
        self.confidence_scores.append(response.confidence_score)
        self.metrics['avg_confidence_score'] = sum(self.confidence_scores) / len(self.confidence_scores)
        
        # Track quality distribution
        self.metrics['quality_distribution'][response.quality_level.value] += 1
        
        # Track errors
        if response.errors:
            self.metrics['error_count'] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_requests == 0:
            return 0.0
        return self.metrics['cache_hits'] / total_cache_requests
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hit_rate': self.get_cache_hit_rate(),
            'avg_response_time': self.metrics['avg_response_time'],
            'avg_confidence_score': self.metrics['avg_confidence_score'],
            'error_rate': self.metrics['error_count'] / max(self.metrics['total_requests'], 1),
            'quality_distribution': dict(self.metrics['quality_distribution'])
        }


# Global performance tracker instance
performance_tracker = PerformanceTracker()


# Intelligent Caching System
class IntelligentCache:
    """
    Advanced intelligent caching system with learning capabilities
    
    Features:
    - 4 caching strategies: Conservative, Balanced, Aggressive, Adaptive
    - Query pattern recognition and classification
    - Adaptive TTL optimization based on performance history
    - Quality-based cache admission control
    - Performance learning algorithms
    - Real-time analytics and monitoring
    """
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE, max_size: int = 1000):
        """Initialize the intelligent cache with specified strategy"""
        
        self.cache = {}  # Main cache storage
        self.strategy = strategy
        self.max_size = max_size
        
        # Performance tracking
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "quality_rejects": 0,
            "pattern_hits": defaultdict(int),
            "pattern_misses": defaultdict(int)
        }
        
        # Learning and pattern recognition
        self.performance_history = deque(maxlen=1000)
        self.query_patterns = defaultdict(list)
        self.ttl_adjustments = defaultdict(float)  # Learned TTL multipliers
        
        # Strategy-specific configurations
        self.strategy_config = {
            CacheStrategy.CONSERVATIVE: {
                "quality_threshold": 0.8,
                "ttl_multiplier": 1.5,
                "max_cache_ratio": 0.8,  # Use 80% of max cache size
                "eviction_priority": "quality_first"
            },
            CacheStrategy.BALANCED: {
                "quality_threshold": 0.6,
                "ttl_multiplier": 1.0,
                "max_cache_ratio": 1.0,
                "eviction_priority": "lru"
            },
            CacheStrategy.AGGRESSIVE: {
                "quality_threshold": 0.4,
                "ttl_multiplier": 0.7,
                "max_cache_ratio": 1.0,
                "eviction_priority": "age_first"
            },
            CacheStrategy.ADAPTIVE: {
                "quality_threshold": 0.5,  # Will be adjusted based on performance
                "ttl_multiplier": 1.0,     # Will be learned
                "max_cache_ratio": 1.0,
                "eviction_priority": "smart"
            }
        }
        
        # Query pattern definitions for recognition
        self.query_patterns_definitions = {
            'comparison': {
                'keywords': ['vs', 'versus', 'compare', 'better', 'difference', 'which is', 'between'],
                'ttl_multiplier': 1.2,  # Comparisons stay relevant longer
                'quality_boost': 0.1
            },
            'recommendation': {
                'keywords': ['best', 'recommend', 'suggest', 'which', 'what should', 'top', 'choose'],
                'ttl_multiplier': 0.8,  # Recommendations change more frequently
                'quality_boost': 0.05
            },
            'explanation': {
                'keywords': ['how', 'why', 'what is', 'explain', 'tell me', 'describe', 'define'],
                'ttl_multiplier': 1.5,  # Explanations are more stable
                'quality_boost': 0.15
            },
            'listing': {
                'keywords': ['list', 'show me', 'give me', 'what are', 'all', 'every'],
                'ttl_multiplier': 1.0,  # Standard TTL for lists
                'quality_boost': 0.0
            },
            'troubleshooting': {
                'keywords': ['problem', 'issue', 'error', 'not working', 'fix', 'solve', 'help'],
                'ttl_multiplier': 0.6,  # Troubleshooting info changes frequently
                'quality_boost': 0.2   # High value for problem-solving
            },
            'news_update': {
                'keywords': ['news', 'latest', 'recent', 'update', 'current', 'today', 'breaking'],
                'ttl_multiplier': 0.3,  # News becomes stale quickly
                'quality_boost': 0.0
            },
            'regulatory': {
                'keywords': ['regulation', 'law', 'legal', 'compliance', 'rule', 'policy', 'requirement'],
                'ttl_multiplier': 2.0,  # Regulatory info is stable
                'quality_boost': 0.25
            }
        }
        
        # Base TTL by query type (in hours)
        self.base_ttl_by_query_type = {
            'news_update': 2,        # Hours - Rapid change
            'promotion_analysis': 6,  # Hours - Frequent updates
            'troubleshooting': 12,   # Hours - Regular changes
            'general_info': 24,      # Hours - Daily updates
            'casino_review': 48,     # Hours - Weekly changes
            'game_guide': 72,        # Hours - Stable content
            'comparison': 48,        # Hours - Moderate change
            'regulatory': 168        # Hours - Infrequent updates
        }
        
        logger.info(f"IntelligentCache initialized with {strategy.value} strategy")
    
    async def get(self, query: str, query_analysis: Optional[Any] = None) -> Optional[EnhancedRAGResponse]:
        """
        Retrieve cached response for query
        
        Args:
            query: The query string
            query_analysis: Optional query analysis for better cache key generation
            
        Returns:
            Cached EnhancedRAGResponse if found and valid, None otherwise
        """
        try:
            cache_key = self._get_cache_key(query, query_analysis)
            
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                
                # Check if entry is still valid
                if self._is_cache_entry_valid(cache_entry):
                    # Update access time and hit count
                    cache_entry.last_accessed = datetime.now()
                    cache_entry.access_count += 1
                    
                    # Record performance
                    pattern = self._identify_query_pattern(query)
                    self._record_performance(query, hit=True, reason="cache_hit", pattern=pattern)
                    
                    self.cache_stats["hits"] += 1
                    if pattern:
                        self.cache_stats["pattern_hits"][pattern] += 1
                    
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cache_entry.response
                else:
                    # Entry expired, remove it
                    del self.cache[cache_key]
                    pattern = self._identify_query_pattern(query)
                    self._record_performance(query, hit=False, reason="expired", pattern=pattern)
            
            # Cache miss
            pattern = self._identify_query_pattern(query)
            self._record_performance(query, hit=False, reason="miss", pattern=pattern)
            
            self.cache_stats["misses"] += 1
            if pattern:
                self.cache_stats["pattern_misses"][pattern] += 1
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None
    
    async def set(self, query: str, response: EnhancedRAGResponse, query_analysis: Optional[Any] = None):
        """
        Cache a response with intelligent TTL and quality control
        
        Args:
            query: The query string
            response: The response to cache
            query_analysis: Optional query analysis for optimization
        """
        try:
            # Quality gate - don't cache low-quality responses
            config = self.strategy_config[self.strategy]
            quality_threshold = self._get_adaptive_quality_threshold()
            
            if response.confidence_score < quality_threshold:
                self.cache_stats["quality_rejects"] += 1
                logger.debug(f"Response rejected for caching due to low quality: {response.confidence_score:.3f} < {quality_threshold:.3f}")
                return
            
            # Generate cache key
            cache_key = self._get_cache_key(query, query_analysis)
            
            # Calculate intelligent TTL
            ttl_hours = self._get_adaptive_ttl(query, query_analysis, response.confidence_score)
            expires_at = datetime.now() + timedelta(hours=ttl_hours)
            
            # Create cache entry
            cache_entry = CacheEntry(
                key=cache_key,
                response=response,
                created_at=datetime.now(),
                expires_at=expires_at,
                last_accessed=datetime.now(),
                access_count=0,
                ttl_hours=ttl_hours,
                quality_score=response.confidence_score,
                query_pattern=self._identify_query_pattern(query),
                cache_value_score=self._calculate_cache_value_score(response, query, query_analysis)
            )
            
            # Ensure cache size limits
            await self._ensure_cache_size_limit()
            
            # Store in cache
            self.cache[cache_key] = cache_entry
            
            # Update learning algorithms
            self._update_learning(query, cache_key, ttl_hours)
            
            logger.debug(f"Cached response for query: {query[:50]}... (TTL: {ttl_hours}h, Quality: {response.confidence_score:.3f})")
            
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
    
    def _get_cache_key(self, query: str, query_analysis: Optional[Any] = None) -> str:
        """Generate intelligent cache key with pattern recognition"""
        
        # Base key from query
        base_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        
        if query_analysis and hasattr(query_analysis, 'query_type'):
            # Include query characteristics for more specific caching
            query_signature = f"{query_analysis.query_type.value}"
            
            if hasattr(query_analysis, 'expertise_level'):
                query_signature += f"_{query_analysis.expertise_level.value}"
            
            # Add semantic clustering based on pattern
            query_pattern = self._identify_query_pattern(query)
            if query_pattern:
                query_signature += f"_{query_pattern}"
            
            # Create combined key
            signature_hash = hashlib.md5(query_signature.encode()).hexdigest()[:8]
            combined_key = f"{base_key}_{signature_hash}"
            return combined_key
        
        return base_key
    
    def _identify_query_pattern(self, query: str) -> Optional[str]:
        """Identify query pattern for intelligent caching"""
        
        query_lower = query.lower()
        
        # Score each pattern based on keyword matches
        pattern_scores = {}
        
        for pattern_name, pattern_info in self.query_patterns_definitions.items():
            score = 0
            for keyword in pattern_info['keywords']:
                if keyword in query_lower:
                    score += 1
            
            if score > 0:
                # Normalize by number of keywords
                pattern_scores[pattern_name] = score / len(pattern_info['keywords'])
        
        # Return the highest scoring pattern if above threshold
        if pattern_scores:
            best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
            if best_pattern[1] > 0.1:  # At least 10% keyword match
                return best_pattern[0]
        
        return None
    
    def _get_adaptive_ttl(self, query: str, query_analysis: Optional[Any] = None, quality_score: float = 0.5) -> int:
        """Calculate adaptive TTL based on multiple factors"""
        
        # Base TTL from query type
        base_ttl = self._get_base_ttl(query_analysis)
        
        # Quality score multiplier (0.5-2.0 range)
        quality_multiplier = 0.5 + (quality_score * 1.5)
        
        # Pattern-based adjustments
        pattern_multiplier = self._get_pattern_multiplier(query)
        
        # Strategy-specific multiplier
        strategy_multiplier = self.strategy_config[self.strategy]["ttl_multiplier"]
        
        # Learned adjustments from performance history
        learned_adjustment = self._get_learned_adjustment(query)
        
        # Calculate final TTL
        final_ttl = int(base_ttl * quality_multiplier * pattern_multiplier * strategy_multiplier * learned_adjustment)
        
        # Bounds: 1 hour to 1 week
        return max(1, min(final_ttl, 168))
    
    def _get_base_ttl(self, query_analysis: Optional[Any] = None) -> int:
        """Get base TTL from query analysis or default"""
        
        if query_analysis and hasattr(query_analysis, 'query_type'):
            query_type_str = query_analysis.query_type.value.lower()
            
            # Map query types to TTL categories
            for ttl_category, hours in self.base_ttl_by_query_type.items():
                if ttl_category in query_type_str:
                    return hours
        
        # Default TTL
        return 24  # 24 hours
    
    def _get_pattern_multiplier(self, query: str) -> float:
        """Get TTL multiplier based on query pattern"""
        
        pattern = self._identify_query_pattern(query)
        if pattern and pattern in self.query_patterns_definitions:
            return self.query_patterns_definitions[pattern]['ttl_multiplier']
        
        return 1.0  # Default multiplier
    
    def _get_learned_adjustment(self, query: str) -> float:
        """Get learned TTL adjustment based on performance history"""
        
        pattern = self._identify_query_pattern(query)
        if pattern and pattern in self.ttl_adjustments:
            return self.ttl_adjustments[pattern]
        
        return 1.0  # Default adjustment
    
    def _get_adaptive_quality_threshold(self) -> float:
        """Get adaptive quality threshold based on cache performance"""
        
        base_threshold = self.strategy_config[self.strategy]["quality_threshold"]
        
        if self.strategy == CacheStrategy.ADAPTIVE:
            # Adjust threshold based on hit rate
            hit_rate = self._get_current_hit_rate()
            
            if hit_rate < 0.3:  # Low hit rate, lower threshold to cache more
                return max(0.3, base_threshold - 0.1)
            elif hit_rate > 0.8:  # High hit rate, raise threshold for quality
                return min(0.9, base_threshold + 0.1)
        
        return base_threshold
    
    def _get_current_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests == 0:
            return 0.0
        
        return self.cache_stats["hits"] / total_requests
    
    def _calculate_cache_value_score(self, response: EnhancedRAGResponse, query: str, query_analysis: Optional[Any] = None) -> float:
        """Calculate the value score for caching this response"""
        
        # Base score from confidence
        value_score = response.confidence_score
        
        # Pattern-based value boost
        pattern = self._identify_query_pattern(query)
        if pattern and pattern in self.query_patterns_definitions:
            value_score += self.query_patterns_definitions[pattern]['quality_boost']
        
        # Response time factor (faster responses are more valuable to cache)
        if hasattr(response, 'response_time') and response.response_time:
            if response.response_time > 2.0:  # Slow responses benefit more from caching
                value_score += 0.1
        
        # Source quality factor
        if hasattr(response, 'sources') and response.sources:
            avg_source_quality = sum(s.get('quality_score', 0.5) for s in response.sources) / len(response.sources)
            value_score += (avg_source_quality - 0.5) * 0.2  # Boost for high-quality sources
        
        return min(1.0, value_score)  # Cap at 1.0
    
    def _is_cache_entry_valid(self, cache_entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        
        return datetime.now() < cache_entry.expires_at
    
    async def _ensure_cache_size_limit(self):
        """Ensure cache doesn't exceed size limits with intelligent eviction"""
        
        config = self.strategy_config[self.strategy]
        max_allowed = int(self.max_size * config["max_cache_ratio"])
        
        if len(self.cache) >= max_allowed:
            # Evict entries based on strategy
            eviction_count = len(self.cache) - max_allowed + 1
            await self._evict_entries(eviction_count, config["eviction_priority"])
    
    async def _evict_entries(self, count: int, priority: str):
        """Evict cache entries based on priority strategy"""
        
        if not self.cache:
            return
        
        entries = list(self.cache.items())
        
        if priority == "quality_first":
            # Evict lowest quality first
            entries.sort(key=lambda x: x[1].cache_value_score)
        elif priority == "age_first":
            # Evict oldest first
            entries.sort(key=lambda x: x[1].created_at)
        elif priority == "lru":
            # Evict least recently used first
            entries.sort(key=lambda x: x[1].last_accessed)
        elif priority == "smart":
            # Smart eviction: combine age, quality, and access patterns
            entries.sort(key=lambda x: self._calculate_eviction_score(x[1]))
        
        # Evict the selected entries
        for i in range(min(count, len(entries))):
            cache_key = entries[i][0]
            del self.cache[cache_key]
            self.cache_stats["evictions"] += 1
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate eviction score for smart eviction (lower = evict first)"""
        
        # Age factor (older = higher eviction score)
        age_hours = (datetime.now() - entry.created_at).total_seconds() / 3600
        age_score = min(1.0, age_hours / 168)  # Normalize to 1 week
        
        # Access frequency factor (less accessed = higher eviction score)
        access_score = 1.0 / (1.0 + entry.access_count)
        
        # Quality factor (lower quality = higher eviction score)
        quality_score = 1.0 - entry.cache_value_score
        
        # Combine factors
        eviction_score = (age_score * 0.4) + (access_score * 0.3) + (quality_score * 0.3)
        
        return eviction_score
    
    def _record_performance(self, query: str, hit: bool, reason: str, pattern: Optional[str] = None):
        """Record performance metrics for learning"""
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'query_length': len(query),
            'hit': hit,
            'reason': reason,
            'pattern': pattern,
            'cache_size': len(self.cache)
        })
    
    def _update_learning(self, query: str, cache_key: str, ttl_hours: int):
        """Update learning algorithms based on cache usage"""
        
        pattern = self._identify_query_pattern(query)
        if pattern:
            self.query_patterns[pattern].append({
                'timestamp': datetime.now(),
                'ttl_used': ttl_hours,
                'query_length': len(query),
                'cache_key': cache_key
            })
            
            # Update learned TTL adjustments based on hit rate patterns
            self._update_ttl_learning(pattern)
    
    def _update_ttl_learning(self, pattern: str):
        """Update TTL learning for specific patterns"""
        
        if pattern not in self.query_patterns or len(self.query_patterns[pattern]) < 10:
            return  # Need more data
        
        # Analyze recent performance for this pattern
        recent_entries = self.query_patterns[pattern][-50:]  # Last 50 entries
        
        # Calculate hit rate for different TTL ranges
        ttl_performance = defaultdict(list)
        for entry in recent_entries:
            ttl_range = self._get_ttl_range(entry['ttl_used'])
            # Check if this entry resulted in hits (simplified)
            ttl_performance[ttl_range].append(entry)
        
        # Find optimal TTL range and adjust multiplier
        if len(ttl_performance) > 1:
            # Simple learning: if shorter TTLs have better hit rates, reduce multiplier
            # This is a simplified version - in production, you'd want more sophisticated analysis
            current_multiplier = self.ttl_adjustments.get(pattern, 1.0)
            
            # Gradual adjustment
            if len(recent_entries) % 20 == 0:  # Adjust every 20 entries
                # Random small adjustment for exploration
                adjustment = random.uniform(-0.1, 0.1)
                new_multiplier = max(0.5, min(2.0, current_multiplier + adjustment))
                self.ttl_adjustments[pattern] = new_multiplier
    
    def _get_ttl_range(self, ttl_hours: int) -> str:
        """Categorize TTL into ranges for analysis"""
        
        if ttl_hours <= 6:
            return "short"
        elif ttl_hours <= 24:
            return "medium"
        elif ttl_hours <= 72:
            return "long"
        else:
            return "very_long"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance metrics"""
        
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        # Pattern-specific metrics
        pattern_metrics = {}
        for pattern in self.cache_stats["pattern_hits"]:
            pattern_hits = self.cache_stats["pattern_hits"][pattern]
            pattern_misses = self.cache_stats["pattern_misses"][pattern]
            pattern_total = pattern_hits + pattern_misses
            pattern_hit_rate = pattern_hits / pattern_total if pattern_total > 0 else 0
            
            pattern_metrics[pattern] = {
                "hit_rate": pattern_hit_rate,
                "total_requests": pattern_total,
                "hits": pattern_hits,
                "misses": pattern_misses
            }
        
        # Performance trend analysis
        performance_trend = self._get_performance_trend()
        
        return {
            "hit_rate": hit_rate,
            "total_cached_items": len(self.cache),
            "cache_stats": dict(self.cache_stats),
            "strategy": self.strategy.value,
            "pattern_metrics": pattern_metrics,
            "performance_trend": performance_trend,
            "learned_adjustments": dict(self.ttl_adjustments),
            "cache_size_utilization": len(self.cache) / self.max_size,
            "quality_threshold": self._get_adaptive_quality_threshold(),
            "average_ttl": self._get_average_ttl(),
            "cache_value_distribution": self._get_cache_value_distribution()
        }
    
    def _get_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trends from history"""
        
        if len(self.performance_history) < 10:
            return {"trend": "insufficient_data"}
        
        recent_history = list(self.performance_history)[-100:]  # Last 100 requests
        
        # Calculate hit rate trend
        hit_rates = []
        window_size = 20
        for i in range(window_size, len(recent_history), 10):
            window = recent_history[i-window_size:i]
            hits = sum(1 for entry in window if entry['hit'])
            hit_rate = hits / len(window)
            hit_rates.append(hit_rate)
        
        if len(hit_rates) >= 2:
            trend = "improving" if hit_rates[-1] > hit_rates[0] else "declining"
            trend_strength = abs(hit_rates[-1] - hit_rates[0])
        else:
            trend = "stable"
            trend_strength = 0.0
        
        return {
            "trend": trend,
            "trend_strength": trend_strength,
            "recent_hit_rate": hit_rates[-1] if hit_rates else 0.0,
            "sample_size": len(recent_history)
        }
    
    def _get_average_ttl(self) -> float:
        """Calculate average TTL of cached items"""
        
        if not self.cache:
            return 0.0
        
        total_ttl = sum(entry.ttl_hours for entry in self.cache.values())
        return total_ttl / len(self.cache)
    
    def _get_cache_value_distribution(self) -> Dict[str, int]:
        """Get distribution of cache value scores"""
        
        if not self.cache:
            return {}
        
        distribution = {"low": 0, "medium": 0, "high": 0, "premium": 0}
        
        for entry in self.cache.values():
            score = entry.cache_value_score
            if score < 0.5:
                distribution["low"] += 1
            elif score < 0.7:
                distribution["medium"] += 1
            elif score < 0.9:
                distribution["high"] += 1
            else:
                distribution["premium"] += 1
        
        return distribution
    
    async def clear_expired(self):
        """Remove expired cache entries"""
        
        expired_keys = []
        current_time = datetime.now()
        
        for key, entry in self.cache.items():
            if current_time >= entry.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def clear_all(self):
        """Clear all cache entries"""
        
        cleared_count = len(self.cache)
        self.cache.clear()
        
        # Reset stats but keep learning data
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "quality_rejects": 0,
            "pattern_hits": defaultdict(int),
            "pattern_misses": defaultdict(int)
        }
        
        logger.info(f"Cleared all {cleared_count} cache entries")
        return cleared_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information for debugging"""
        
        cache_info = {
            "total_entries": len(self.cache),
            "strategy": self.strategy.value,
            "max_size": self.max_size,
            "entries": []
        }
        
        for key, entry in list(self.cache.items())[:10]:  # Show first 10 entries
            cache_info["entries"].append({
                "key": key[:16] + "..." if len(key) > 16 else key,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "ttl_hours": entry.ttl_hours,
                "quality_score": entry.quality_score,
                "access_count": entry.access_count,
                "pattern": entry.query_pattern,
                "cache_value_score": entry.cache_value_score
            })
        
        return cache_info


# Global intelligent cache instance
intelligent_cache = IntelligentCache()


# Logging configuration
def setup_enhanced_logging(log_level: str = "INFO"):
    """Setup enhanced logging for the confidence scoring system"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_rag_system.log')
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Enhanced Confidence Scoring System initialized")
    
    return logger


# Initialize logger
logger = setup_enhanced_logging() 


# ============================================================================
# RESPONSE VALIDATION FRAMEWORK
# ============================================================================
"""
Enhanced Response Validation Framework for Universal RAG CMS
Integrates with existing enhanced_confidence_scoring_system.py

This implementation provides comprehensive response validation with:
- Multi-dimensional quality assessment
- Real-time performance optimization
- Detailed issue detection and suggestions
- Seamless integration with existing EnhancedRAGResponse model
"""

# Enhanced validation models
class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ValidationCategory(Enum):
    """Categories of validation checks."""
    FORMAT = "format"
    CONTENT = "content"
    SOURCES = "sources"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"

@dataclass
class ValidationIssue:
    """Represents a validation issue with detailed context."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    description: str
    suggestion: str
    score_impact: float = 0.0
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    format_score: float = 0.0
    content_score: float = 0.0
    source_score: float = 0.0
    quality_score: float = 0.0
    overall_score: float = 0.0
    processing_time_ms: float = 0.0
    issues_count: Dict[ValidationSeverity, int] = field(default_factory=lambda: defaultdict(int))

class ResponseValidator:
    """
    Comprehensive response validation system for Universal RAG CMS.
    
    Features:
    - Multi-dimensional quality assessment
    - Pattern-based format validation
    - Content relevance and coherence analysis
    - Source utilization validation
    - Performance-optimized processing
    - Detailed improvement suggestions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ResponseValidator with configuration."""
        self.config = self._load_config(config)
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.validation_rules = {
            'min_length': self.config.get('min_response_length', 50),
            'max_length': self.config.get('max_response_length', 5000),
            'min_sentences': 2,
            'min_paragraphs': 1,
            'max_paragraphs': 20,
            'citation_required': self.config.get('require_citations', False),
            'factual_consistency': True,
            'coherence_threshold': 0.7,
            'relevance_threshold': 0.8
        }
        
        # Format detection patterns
        self.format_patterns = {
            'structured_list': [
                r'^\s*\d+\.\s+',  # Numbered lists
                r'^\s*[•\-\*]\s+',  # Bullet points
                r'^\s*[a-zA-Z]\)\s+',  # Lettered lists
            ],
            'headers': [
                r'^#+\s+.+$',  # Markdown headers
                r'^.+\n[=\-]+$',  # Underlined headers
                r'^\*\*.+\*\*$',  # Bold headers
            ],
            'citations': [
                r'\[\d+\]',  # Numbered citations
                r'\[.+?\]',  # Bracketed citations
                r'\(.+?\)',  # Parenthetical citations
            ],
            'transitions': [
                r'\b(however|moreover|furthermore|additionally|consequently|therefore|thus|hence)\b',
                r'\b(first|second|third|finally|in conclusion|to summarize)\b',
                r'\b(for example|for instance|such as|including)\b',
            ],
            'comparison_indicators': [
                r'\b(versus|vs|compared to|in contrast|on the other hand|while|whereas)\b',
                r'\b(better|worse|superior|inferior|advantage|disadvantage)\b',
            ]
        }
        
        # Content quality indicators
        self.quality_indicators = {
            'positive': [
                'comprehensive', 'detailed', 'specific', 'accurate', 'relevant',
                'evidence', 'research', 'analysis', 'insights', 'practical',
                'actionable', 'clear', 'structured', 'informative'
            ],
            'negative': [
                'vague', 'unclear', 'confusing', 'irrelevant', 'superficial',
                'generic', 'repetitive', 'inconsistent', 'inaccurate'
            ]
        }
    
    def _load_config(self, config: Optional[Dict]) -> Dict:
        """Load and validate configuration."""
        default_config = {
            'enable_response_validation': True,
            'min_response_length': 50,
            'max_response_length': 5000,
            'require_citations': False,
            'strict_mode': False,
            'performance_mode': False,
            'cache_validations': True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    async def validate_response(
        self,
        response_content: str,
        query: str,
        sources: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> Tuple[ValidationMetrics, List[ValidationIssue]]:
        """
        Perform comprehensive response validation.
        
        Args:
            response_content: The generated response text
            query: Original user query
            sources: Retrieved source documents
            context: Additional context for validation
            
        Returns:
            Tuple of (ValidationMetrics, List[ValidationIssue])
        """
        start_time = time.time()
        issues = []
        
        try:
            # Parallel validation for performance
            validation_tasks = [
                self._validate_format(response_content),
                self._validate_content_quality(response_content, query),
                self._validate_source_utilization(response_content, sources or []),
                self._validate_consistency(response_content),
                self._validate_completeness(response_content, query)
            ]
            
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            format_issues, format_score = validation_results[0] if not isinstance(validation_results[0], Exception) else ([], 0.0)
            content_issues, content_score = validation_results[1] if not isinstance(validation_results[1], Exception) else ([], 0.0)
            source_issues, source_score = validation_results[2] if not isinstance(validation_results[2], Exception) else ([], 0.0)
            consistency_issues, consistency_score = validation_results[3] if not isinstance(validation_results[3], Exception) else ([], 0.0)
            completeness_issues, completeness_score = validation_results[4] if not isinstance(validation_results[4], Exception) else ([], 0.0)
            
            # Combine all issues
            all_issues = format_issues + content_issues + source_issues + consistency_issues + completeness_issues
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                format_score, content_score, source_score, 
                consistency_score, completeness_score,
                all_issues, time.time() - start_time
            )
            
            # Add performance insights
            if context and context.get('track_performance'):
                await self._track_validation_performance(metrics, context)
            
            return metrics, all_issues
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            # Return minimal metrics on error
            error_metrics = ValidationMetrics(
                processing_time_ms=(time.time() - start_time) * 1000
            )
            error_issue = ValidationIssue(
                category=ValidationCategory.QUALITY,
                severity=ValidationSeverity.CRITICAL,
                message="Validation system error",
                description=f"Internal validation error: {str(e)}",
                suggestion="Please retry or contact support if the issue persists"
            )
            return error_metrics, [error_issue]
    
    async def _validate_format(self, content: str) -> Tuple[List[ValidationIssue], float]:
        """Validate response format and structure."""
        issues = []
        score = 1.0
        
        # Length validation
        if len(content) < self.validation_rules['min_length']:
            issues.append(ValidationIssue(
                category=ValidationCategory.FORMAT,
                severity=ValidationSeverity.HIGH,
                message="Response too short",
                description=f"Response is {len(content)} characters, minimum is {self.validation_rules['min_length']}",
                suggestion="Provide more detailed information to meet minimum length requirements",
                score_impact=0.3
            ))
            score -= 0.3
        
        if len(content) > self.validation_rules['max_length']:
            issues.append(ValidationIssue(
                category=ValidationCategory.FORMAT,
                severity=ValidationSeverity.MEDIUM,
                message="Response too long",
                description=f"Response is {len(content)} characters, maximum is {self.validation_rules['max_length']}",
                suggestion="Consider breaking into sections or summarizing key points",
                score_impact=0.2
            ))
            score -= 0.2
        
        # Sentence and paragraph structure
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(sentences) < self.validation_rules['min_sentences']:
            issues.append(ValidationIssue(
                category=ValidationCategory.FORMAT,
                severity=ValidationSeverity.MEDIUM,
                message="Insufficient sentence structure",
                description=f"Found {len(sentences)} sentences, minimum is {self.validation_rules['min_sentences']}",
                suggestion="Expand response with more detailed explanations",
                score_impact=0.15
            ))
            score -= 0.15
        
        # Check for proper formatting patterns
        has_structure = any(
            any(re.search(pattern, content, re.MULTILINE) for pattern in patterns)
            for patterns in self.format_patterns.values()
        )
        
        if len(content) > 500 and not has_structure:
            issues.append(ValidationIssue(
                category=ValidationCategory.FORMAT,
                severity=ValidationSeverity.LOW,
                message="Lacks structured formatting",
                description="Long response could benefit from headers, lists, or other structural elements",
                suggestion="Consider using bullet points, numbered lists, or section headers for better readability",
                score_impact=0.1
            ))
            score -= 0.1
        
        return issues, max(0.0, score)
    
    async def _validate_content_quality(self, content: str, query: str) -> Tuple[List[ValidationIssue], float]:
        """Validate content quality and relevance."""
        issues = []
        score = 1.0
        
        # Relevance analysis - keyword overlap
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        query_words -= stop_words
        content_words -= stop_words
        
        if query_words:
            relevance_score = len(query_words & content_words) / len(query_words)
            
            if relevance_score < self.validation_rules['relevance_threshold']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONTENT,
                    severity=ValidationSeverity.HIGH,
                    message="Low relevance to query",
                    description=f"Content relevance score: {relevance_score:.2f}, threshold: {self.validation_rules['relevance_threshold']}",
                    suggestion="Ensure response directly addresses the user's question and includes relevant keywords",
                    score_impact=0.4
                ))
                score -= 0.4
        
        # Quality indicators analysis
        positive_indicators = sum(1 for indicator in self.quality_indicators['positive'] if indicator in content.lower())
        negative_indicators = sum(1 for indicator in self.quality_indicators['negative'] if indicator in content.lower())
        
        quality_ratio = positive_indicators / max(1, positive_indicators + negative_indicators)
        
        if quality_ratio < 0.6:
            issues.append(ValidationIssue(
                category=ValidationCategory.QUALITY,
                severity=ValidationSeverity.MEDIUM,
                message="Low quality indicators",
                description=f"Quality indicator ratio: {quality_ratio:.2f}",
                suggestion="Include more specific, detailed, and actionable information",
                score_impact=0.2
            ))
            score -= 0.2
        
        # Coherence check - transition words
        transition_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in self.format_patterns['transitions']
        )
        
        sentences = len(re.split(r'[.!?]+', content))
        if sentences > 5 and transition_count == 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONTENT,
                severity=ValidationSeverity.LOW,
                message="Lacks coherence indicators",
                description="No transition words found in multi-sentence response",
                suggestion="Use transition words to improve flow and coherence",
                score_impact=0.1
            ))
            score -= 0.1
        
        return issues, max(0.0, score)
    
    async def _validate_source_utilization(self, content: str, sources: List[Dict]) -> Tuple[List[ValidationIssue], float]:
        """Validate proper source utilization and citation."""
        issues = []
        score = 1.0
        
        if not sources:
            if self.validation_rules['citation_required']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SOURCES,
                    severity=ValidationSeverity.HIGH,
                    message="No sources provided",
                    description="Sources are required but none were provided",
                    suggestion="Include relevant sources to support the response",
                    score_impact=0.5
                ))
                score -= 0.5
            return issues, score
        
        # Check for citation patterns
        citation_patterns = self.format_patterns['citations']
        citations_found = sum(
            len(re.findall(pattern, content))
            for pattern in citation_patterns
        )
        
        if len(sources) > 0 and citations_found == 0:
            issues.append(ValidationIssue(
                category=ValidationCategory.SOURCES,
                severity=ValidationSeverity.MEDIUM,
                message="Sources not cited",
                description=f"Found {len(sources)} sources but no citations in content",
                suggestion="Add proper citations to reference the provided sources",
                score_impact=0.3
            ))
            score -= 0.3
        
        # Check source-content alignment
        source_keywords = set()
        for source in sources:
            if 'content' in source and source['content']:
                source_keywords.update(re.findall(r'\b\w+\b', (source['content'] or "").lower()))
        
        content_keywords = set(re.findall(r'\b\w+\b', content.lower()))
        
        if source_keywords:
            utilization_score = len(source_keywords & content_keywords) / len(source_keywords)
            
            if utilization_score < 0.3:
                issues.append(ValidationIssue(
                    category=ValidationCategory.SOURCES,
                    severity=ValidationSeverity.MEDIUM,
                    message="Poor source utilization",
                    description=f"Source utilization score: {utilization_score:.2f}",
                    suggestion="Better integrate information from the provided sources",
                    score_impact=0.2
                ))
                score -= 0.2
        
        return issues, max(0.0, score)
    
    async def _validate_consistency(self, content: str) -> Tuple[List[ValidationIssue], float]:
        """Validate internal consistency of the response."""
        issues = []
        score = 1.0
        
        # Check for contradictory statements
        contradictory_patterns = [
            (r'\b(always|never|all|none|every|no)\b', r'\b(sometimes|some|few|many|most|often|rarely)\b'),
            (r'\b(impossible|cannot|never)\b', r'\b(possible|can|may|might|could)\b'),
            (r'\b(best|worst|perfect|terrible)\b', r'\b(good|bad|better|worse|okay|average)\b')
        ]
        
        for absolute_pattern, qualified_pattern in contradictory_patterns:
            if re.search(absolute_pattern, content, re.IGNORECASE) and re.search(qualified_pattern, content, re.IGNORECASE):
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONSISTENCY,
                    severity=ValidationSeverity.LOW,
                    message="Potential contradiction detected",
                    description="Response contains both absolute and qualified statements",
                    suggestion="Review for consistency in tone and claims",
                    score_impact=0.1
                ))
                score -= 0.1
                break
        
        # Check for repeated information
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        unique_sentences = set(sentences)
        
        if len(sentences) > 3 and len(unique_sentences) / len(sentences) < 0.8:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONSISTENCY,
                severity=ValidationSeverity.MEDIUM,
                message="Repetitive content detected",
                description=f"Sentence uniqueness ratio: {len(unique_sentences) / len(sentences):.2f}",
                suggestion="Remove redundant information and ensure each sentence adds value",
                score_impact=0.2
            ))
            score -= 0.2
        
        return issues, max(0.0, score)
    
    async def _validate_completeness(self, content: str, query: str) -> Tuple[List[ValidationIssue], float]:
        """Validate completeness of the response."""
        issues = []
        score = 1.0
        
        # Detect query type and expected completeness
        query_indicators = {
            'comparison': ['vs', 'versus', 'compare', 'difference', 'better', 'worse'],
            'explanation': ['what', 'why', 'how', 'explain', 'describe'],
            'list': ['list', 'enumerate', 'types', 'kinds', 'examples'],
            'step_by_step': ['steps', 'process', 'procedure', 'how to', 'guide']
        }
        
        query_type = None
        for q_type, indicators in query_indicators.items():
            if any(indicator in query.lower() for indicator in indicators):
                query_type = q_type
                break
        
        # Validate based on query type
        if query_type == 'comparison' and not any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in self.format_patterns['comparison_indicators']
        ):
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.MEDIUM,
                message="Incomplete comparison",
                description="Query asks for comparison but response lacks comparative elements",
                suggestion="Include direct comparisons, pros/cons, or contrasting points",
                score_impact=0.3
            ))
            score -= 0.3
        
        if query_type == 'list' and not any(
            re.search(pattern, content, re.MULTILINE)
            for pattern in self.format_patterns['structured_list']
        ):
            issues.append(ValidationIssue(
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.MEDIUM,
                message="Missing list structure",
                description="Query asks for a list but response is not structured as a list",
                suggestion="Format response as a numbered or bulleted list",
                score_impact=0.25
            ))
            score -= 0.25
        
        if query_type == 'step_by_step':
            step_indicators = ['step', 'first', 'second', 'then', 'next', 'finally']
            if not any(indicator in content.lower() for indicator in step_indicators):
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.MEDIUM,
                    message="Missing step-by-step structure",
                    description="Query asks for steps but response lacks sequential indicators",
                    suggestion="Structure response with clear steps or sequential flow",
                    score_impact=0.3
                ))
                score -= 0.3
        
        return issues, max(0.0, score)
    
    def _calculate_metrics(
        self,
        format_score: float,
        content_score: float,
        source_score: float,
        consistency_score: float,
        completeness_score: float,
        issues: List[ValidationIssue],
        processing_time: float
    ) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""
        
        # Weighted overall score
        weights = {
            'format': 0.15,
            'content': 0.35,
            'source': 0.20,
            'consistency': 0.15,
            'completeness': 0.15
        }
        
        overall_score = (
            format_score * weights['format'] +
            content_score * weights['content'] +
            source_score * weights['source'] +
            consistency_score * weights['consistency'] +
            completeness_score * weights['completeness']
        )
        
        # Count issues by severity
        issues_count = defaultdict(int)
        for issue in issues:
            issues_count[issue.severity] += 1
        
        return ValidationMetrics(
            format_score=format_score,
            content_score=content_score,
            source_score=source_score,
            quality_score=(content_score + consistency_score) / 2,
            overall_score=overall_score,
            processing_time_ms=processing_time * 1000,
            issues_count=dict(issues_count)
        )
    
    async def _track_validation_performance(self, metrics: ValidationMetrics, context: Dict):
        """Track validation performance for monitoring."""
        # This would integrate with your monitoring system
        performance_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time_ms': metrics.processing_time_ms,
            'overall_score': metrics.overall_score,
            'issues_count': metrics.issues_count,
            'context': context
        }
        
        # Log for monitoring (integrate with your logging system)
        self.logger.info("Validation performance", extra=performance_data)
    
    def get_validation_summary(self, metrics: ValidationMetrics, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate a human-readable validation summary."""
        
        # Categorize issues
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == ValidationSeverity.HIGH]
        
        # Generate summary
        summary = {
            'overall_quality': 'excellent' if metrics.overall_score >= 0.9 else
                             'good' if metrics.overall_score >= 0.7 else
                             'fair' if metrics.overall_score >= 0.5 else 'poor',
            'score': round(metrics.overall_score, 3),
            'processing_time_ms': round(metrics.processing_time_ms, 2),
            'issues': {
                'critical': len(critical_issues),
                'high': len(high_issues),
                'total': len(issues)
            },
            'recommendations': []
        }
        
        # Add top recommendations
        if critical_issues:
            summary['recommendations'].append("Address critical issues immediately")
        if high_issues:
            summary['recommendations'].append("Review and fix high-priority issues")
        if metrics.overall_score < 0.7:
            summary['recommendations'].append("Consider regenerating response for better quality")
        
        return summary

# Integration helper for EnhancedRAGResponse
class ValidationIntegrator:
    """Helper class to integrate validation results with existing response models."""
    
    @staticmethod
    def update_rag_response(
        response: EnhancedRAGResponse,
        metrics: ValidationMetrics,
        issues: List[ValidationIssue]
    ) -> EnhancedRAGResponse:
        """Update EnhancedRAGResponse with validation results."""
        
        # Update validation fields
        response.format_validation = {
            'passed': metrics.format_score >= 0.7,
            'score': metrics.format_score,
            'issues': len([i for i in issues if i.category == ValidationCategory.FORMAT])
        }
        
        response.content_validation = {
            'passed': metrics.content_score >= 0.7,
            'score': metrics.content_score,
            'quality_score': metrics.quality_score,
            'issues': len([i for i in issues if i.category == ValidationCategory.CONTENT])
        }
        
        # Add critical errors
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        response.errors.extend([issue.message for issue in critical_issues])
        
        # Update confidence factors based on validation
        if hasattr(response, 'confidence_factors') and response.confidence_factors:
            response.confidence_factors.response_quality = metrics.overall_score
            response.confidence_factors.validation_passed = len(critical_issues) == 0
        
        return response

# Usage example for integration
async def validate_and_enhance_response(
    response_content: str,
    query: str,
    sources: List[Dict],
    existing_response: EnhancedRAGResponse
) -> EnhancedRAGResponse:
    """Complete validation and enhancement workflow."""
    
    # Initialize validator
    validator = ResponseValidator()
    
    # Perform validation
    metrics, issues = await validator.validate_response(
        response_content=response_content,
        query=query,
        sources=sources,
        context={'track_performance': True}
    )
    
    # Integrate results
    enhanced_response = ValidationIntegrator.update_rag_response(
        existing_response, metrics, issues
    )
    
    # Add validation summary
    summary = validator.get_validation_summary(metrics, issues)
    enhanced_response.metadata['validation_summary'] = summary
    enhanced_response.metadata['validation_issues'] = [
        {
            'category': issue.category.value,
            'severity': issue.severity.value,
            'message': issue.message,
            'suggestion': issue.suggestion
        }
        for issue in issues
    ]
    
    return enhanced_response


# Global response validator instance
response_validator = ResponseValidator() 


# ============================================================================
# Enhanced Confidence Calculator - Main Orchestrator
# ============================================================================

@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence scoring with comprehensive metrics."""
    content_quality: float = 0.0
    source_quality: float = 0.0
    query_matching: float = 0.0
    technical_factors: float = 0.0
    overall_confidence: float = 0.0
    
    # Detailed metrics for each category
    content_metrics: Dict[str, float] = field(default_factory=dict)
    source_metrics: Dict[str, float] = field(default_factory=dict)
    query_metrics: Dict[str, float] = field(default_factory=dict)
    technical_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance indicators
    calculation_time_ms: float = 0.0
    cached_components: List[str] = field(default_factory=list)
    
    # Quality indicators
    quality_flags: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


class EnhancedConfidenceCalculator:
    """
    Main orchestrator for enhanced confidence scoring system.
    
    Integrates all completed components:
    - Source Quality Analyzer (subtask 2.12-2.13)
    - Intelligent Cache System (subtask 2.14)
    - Response Validation Framework (subtask 2.15)
    
    Provides:
    - Multi-factor confidence calculation
    - Query-type aware processing
    - Quality-based caching decisions
    - Comprehensive response enhancement
    """
    
    def __init__(
        self,
        source_quality_analyzer: Optional[SourceQualityAnalyzer] = None,
        cache_system: Optional[IntelligentCache] = None,
        response_validator: Optional[ResponseValidator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Enhanced Confidence Calculator."""
        
        self.source_analyzer = source_quality_analyzer or SourceQualityAnalyzer()
        self.cache_system = cache_system or IntelligentCache()
        self.response_validator = response_validator or ResponseValidator()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Confidence scoring weights (Task 2.3 requirements)
        self.confidence_weights = {
            'content_quality': 0.35,
            'source_quality': 0.25,
            'query_matching': 0.20,
            'technical_factors': 0.20
        }
        
        # Quality thresholds for different decisions
        self.quality_thresholds = {
            'cache_storage': 0.75,      # Only cache high-quality responses
            'auto_publish': 0.85,       # Auto-publish threshold
            'review_required': 0.60,    # Below this requires review
            'regeneration': 0.40        # Below this should regenerate
        }
        
        # Query-type specific configurations
        self.query_type_configs = {
            'factual': {
                'accuracy_weight': 0.4,
                'source_authority_weight': 0.3,
                'recency_weight': 0.2,
                'cache_ttl_hours': 24
            },
            'comparison': {
                'completeness_weight': 0.4,
                'objectivity_weight': 0.3,
                'structure_weight': 0.2,
                'cache_ttl_hours': 12
            },
            'tutorial': {
                'clarity_weight': 0.4,
                'completeness_weight': 0.3,
                'practical_value_weight': 0.2,
                'cache_ttl_hours': 48
            },
            'review': {
                'expertise_weight': 0.4,
                'detail_weight': 0.3,
                'objectivity_weight': 0.2,
                'cache_ttl_hours': 6
            }
        }
    
    async def calculate_enhanced_confidence(
        self,
        response: EnhancedRAGResponse,
        query: str,
        query_type: str,
        sources: List[Dict[str, Any]],
        generation_metadata: Dict[str, Any]
    ) -> Tuple[ConfidenceBreakdown, EnhancedRAGResponse]:
        """
        Calculate comprehensive confidence score with detailed breakdown.
        
        This is the main entry point that orchestrates all confidence calculations.
        """
        
        start_time = time.time()
        
        try:
            # Initialize confidence breakdown
            breakdown = ConfidenceBreakdown()
            
            # Parallel confidence calculations for performance
            confidence_tasks = [
                self._calculate_content_quality(response, query, query_type),
                self._calculate_source_quality(sources, query_type),
                self._calculate_query_matching(response, query, query_type),
                self._calculate_technical_factors(response, generation_metadata)
            ]
            
            # Execute calculations in parallel
            results = await asyncio.gather(*confidence_tasks, return_exceptions=True)
            
            # Process results
            content_quality = results[0] if not isinstance(results[0], Exception) else (0.0, {})
            source_quality = results[1] if not isinstance(results[1], Exception) else (0.0, {})
            query_matching = results[2] if not isinstance(results[2], Exception) else (0.0, {})
            technical_factors = results[3] if not isinstance(results[3], Exception) else (0.0, {})
            
            # Populate breakdown
            breakdown.content_quality, breakdown.content_metrics = content_quality
            breakdown.source_quality, breakdown.source_metrics = source_quality
            breakdown.query_matching, breakdown.query_metrics = query_matching
            breakdown.technical_factors, breakdown.technical_metrics = technical_factors
            
            # Calculate overall confidence using weighted average
            breakdown.overall_confidence = (
                breakdown.content_quality * self.confidence_weights['content_quality'] +
                breakdown.source_quality * self.confidence_weights['source_quality'] +
                breakdown.query_matching * self.confidence_weights['query_matching'] +
                breakdown.technical_factors * self.confidence_weights['technical_factors']
            )
            
            # Generate quality indicators and suggestions
            breakdown.quality_flags = self._generate_quality_flags(breakdown)
            breakdown.improvement_suggestions = self._generate_improvement_suggestions(breakdown, query_type)
            
            # Update response with confidence information
            enhanced_response = await self._enhance_response_with_confidence(
                response, breakdown, query_type
            )
            
            # Calculate processing time
            breakdown.calculation_time_ms = (time.time() - start_time) * 1000
            
            # Log performance metrics
            await self._log_confidence_metrics(breakdown, query_type)
            
            return breakdown, enhanced_response
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            
            # Fallback confidence
            fallback_breakdown = ConfidenceBreakdown(
                overall_confidence=0.5,
                quality_flags=['calculation_error'],
                improvement_suggestions=['Retry confidence calculation']
            )
            
            return fallback_breakdown, response
    
    async def _calculate_content_quality(
        self, 
        response: EnhancedRAGResponse, 
        query: str, 
        query_type: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate content quality score (35% weight)."""
        
        metrics = {}
        
        # Use existing Response Validator for comprehensive analysis
        validation_metrics, validation_issues = await self.response_validator.validate_response(
            response_content=response.content,
            query=query,
            sources=response.sources,
            context={'query_type': query_type}
        )
        
        # Extract quality metrics
        metrics['completeness'] = validation_metrics.overall_score
        metrics['relevance'] = validation_metrics.content_score
        metrics['accuracy'] = validation_metrics.format_score  # Proxy for accuracy
        metrics['clarity'] = validation_metrics.quality_score
        
        # Query-type specific adjustments
        type_config = self.query_type_configs.get(query_type, {})
        
        if query_type == 'factual':
            # Emphasize accuracy for factual queries
            accuracy_weight = type_config.get('accuracy_weight', 0.4)
            quality_score = (
                metrics['accuracy'] * accuracy_weight +
                metrics['relevance'] * 0.3 +
                metrics['completeness'] * 0.2 +
                metrics['clarity'] * 0.1
            )
        elif query_type == 'comparison':
            # Emphasize completeness and structure for comparisons
            completeness_weight = type_config.get('completeness_weight', 0.4)
            quality_score = (
                metrics['completeness'] * completeness_weight +
                metrics['clarity'] * 0.3 +
                metrics['relevance'] * 0.2 +
                metrics['accuracy'] * 0.1
            )
        elif query_type == 'tutorial':
            # Emphasize clarity and completeness for tutorials
            clarity_weight = type_config.get('clarity_weight', 0.4)
            quality_score = (
                metrics['clarity'] * clarity_weight +
                metrics['completeness'] * 0.3 +
                metrics['relevance'] * 0.2 +
                metrics['accuracy'] * 0.1
            )
        else:
            # Balanced scoring for other query types
            quality_score = statistics.mean([
                metrics['completeness'],
                metrics['relevance'],
                metrics['accuracy'],
                metrics['clarity']
            ])
        
        # Apply penalties for critical issues
        critical_issues = len([issue for issue in validation_issues 
                              if issue.severity.value == 'critical'])
        if critical_issues > 0:
            quality_score *= (1.0 - min(0.5, critical_issues * 0.1))
        
        # Store validation metadata
        metrics['validation_score'] = validation_metrics.overall_score
        metrics['issues_count'] = len(validation_issues)
        metrics['critical_issues'] = critical_issues
        
        return max(0.0, min(1.0, quality_score)), metrics
    
    async def _calculate_source_quality(
        self, 
        sources: List[Dict[str, Any]], 
        query_type: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate source quality score (25% weight)."""
        
        if not sources:
            return 0.0, {'no_sources': True}
        
        metrics = {}
        source_scores = []
        
        # Analyze each source using existing Source Quality Analyzer
        for source in sources:
            # Create Document object for source analysis
            source_doc = Document(
                page_content=source.get('content', ''),
                metadata=source.get('metadata', {})
            )
            
            source_analysis = await self.source_analyzer.analyze_source_quality(source_doc)
            
            # Extract quality indicators
            quality_indicators = source_analysis.get('quality_scores', {})
            
            # Calculate weighted source score based on query type
            if query_type == 'factual':
                # Emphasize authority and recency for factual queries
                source_score = (
                    quality_indicators.get('authority', 0.5) * 0.3 +
                    quality_indicators.get('credibility', 0.5) * 0.3 +
                    quality_indicators.get('recency', 0.5) * 0.2 +
                    quality_indicators.get('expertise', 0.5) * 0.2
                )
            elif query_type == 'review':
                # Emphasize expertise and objectivity for reviews
                source_score = (
                    quality_indicators.get('expertise', 0.5) * 0.4 +
                    quality_indicators.get('objectivity', 0.5) * 0.3 +
                    quality_indicators.get('detail', 0.5) * 0.2 +
                    quality_indicators.get('credibility', 0.5) * 0.1
                )
            else:
                # Balanced scoring
                source_score = statistics.mean([
                    quality_indicators.get('authority', 0.5),
                    quality_indicators.get('credibility', 0.5),
                    quality_indicators.get('expertise', 0.5),
                    quality_indicators.get('recency', 0.5)
                ])
            
            source_scores.append(source_score)
        
        # Calculate overall source quality
        if source_scores:
            # Weight by source importance (first sources are typically more important)
            weights = [1.0 / (i + 1) for i in range(len(source_scores))]
            weighted_sum = sum(score * weight for score, weight in zip(source_scores, weights))
            total_weight = sum(weights)
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall_score = 0.0
        
        # Store detailed metrics
        metrics['source_count'] = len(sources)
        metrics['average_source_quality'] = statistics.mean(source_scores) if source_scores else 0.0
        metrics['best_source_quality'] = max(source_scores) if source_scores else 0.0
        metrics['worst_source_quality'] = min(source_scores) if source_scores else 0.0
        metrics['quality_consistency'] = 1.0 - (statistics.stdev(source_scores) if len(source_scores) > 1 else 0.0)
        
        return max(0.0, min(1.0, overall_score)), metrics
    
    async def _calculate_query_matching(
        self, 
        response: EnhancedRAGResponse, 
        query: str, 
        query_type: str
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate query matching score (20% weight)."""
        
        metrics = {}
        
        # Intent alignment analysis
        intent_score = await self._analyze_intent_alignment(response.content, query, query_type)
        metrics['intent_alignment'] = intent_score
        
        # Expertise level matching
        expertise_score = await self._analyze_expertise_matching(response.content, query, query_type)
        metrics['expertise_match'] = expertise_score
        
        # Format appropriateness
        format_score = await self._analyze_format_appropriateness(response.content, query_type)
        metrics['format_appropriateness'] = format_score
        
        # Keyword relevance
        keyword_score = await self._analyze_keyword_relevance(response.content, query)
        metrics['keyword_relevance'] = keyword_score
        
        # Calculate weighted matching score
        matching_score = (
            intent_score * 0.4 +
            expertise_score * 0.3 +
            format_score * 0.2 +
            keyword_score * 0.1
        )
        
        return max(0.0, min(1.0, matching_score)), metrics
    
    async def _calculate_technical_factors(
        self, 
        response: EnhancedRAGResponse, 
        generation_metadata: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate technical factors score (20% weight)."""
        
        metrics = {}
        
        # Retrieval quality
        retrieval_score = generation_metadata.get('retrieval_quality', 0.5)
        metrics['retrieval_quality'] = retrieval_score
        
        # Generation stability (consistency of output)
        stability_score = generation_metadata.get('generation_stability', 0.5)
        metrics['generation_stability'] = stability_score
        
        # Optimization effectiveness
        optimization_score = generation_metadata.get('optimization_effectiveness', 0.5)
        metrics['optimization_effectiveness'] = optimization_score
        
        # Response time performance
        response_time = generation_metadata.get('response_time_ms', 1000)
        time_score = max(0.0, min(1.0, 1.0 - (response_time - 500) / 1500))  # Optimal: 500ms
        metrics['response_time_score'] = time_score
        
        # Token efficiency
        token_efficiency = generation_metadata.get('token_efficiency', 0.5)
        metrics['token_efficiency'] = token_efficiency
        
        # Calculate overall technical score
        technical_score = (
            retrieval_score * 0.3 +
            stability_score * 0.25 +
            optimization_score * 0.2 +
            time_score * 0.15 +
            token_efficiency * 0.1
        )
        
        return max(0.0, min(1.0, technical_score)), metrics
    
    async def _analyze_intent_alignment(self, content: str, query: str, query_type: str) -> float:
        """Analyze how well the response aligns with query intent."""
        
        # Basic keyword overlap analysis
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words -= stop_words
        content_words -= stop_words
        
        # Calculate overlap
        if query_words:
            overlap = len(query_words & content_words) / len(query_words)
        else:
            overlap = 0.0
        
        # Query type specific intent analysis
        if query_type == 'comparison':
            # Look for comparison indicators
            comparison_words = {'vs', 'versus', 'compare', 'comparison', 'better', 'worse', 'difference'}
            comparison_score = len([word for word in content.lower().split() if word in comparison_words]) / 100
            overlap = (overlap + min(1.0, comparison_score)) / 2
        
        elif query_type == 'tutorial':
            # Look for instructional indicators
            instruction_words = {'step', 'how', 'guide', 'tutorial', 'learn', 'first', 'then', 'next', 'finally'}
            instruction_score = len([word for word in content.lower().split() if word in instruction_words]) / 100
            overlap = (overlap + min(1.0, instruction_score)) / 2
        
        return min(1.0, overlap)
    
    async def _analyze_expertise_matching(self, content: str, query: str, query_type: str) -> float:
        """Analyze expertise level matching between query and response."""
        
        # Detect query complexity
        query_complexity = self._detect_query_complexity(query)
        
        # Detect response complexity
        response_complexity = self._detect_response_complexity(content)
        
        # Calculate match score (closer complexities = better match)
        complexity_diff = abs(query_complexity - response_complexity)
        match_score = max(0.0, 1.0 - complexity_diff)
        
        return match_score
    
    def _detect_query_complexity(self, query: str) -> float:
        """Detect query complexity (0.0 = simple, 1.0 = expert)."""
        
        # Simple heuristics for complexity detection
        complexity_indicators = {
            'technical_terms': ['algorithm', 'implementation', 'optimization', 'architecture'],
            'advanced_concepts': ['strategy', 'methodology', 'framework', 'analysis'],
            'specific_jargon': ['API', 'SDK', 'protocol', 'specification']
        }
        
        query_lower = query.lower()
        complexity_score = 0.0
        
        for category, terms in complexity_indicators.items():
            matches = sum(1 for term in terms if term in query_lower)
            complexity_score += matches * 0.1
        
        # Query length as complexity indicator
        word_count = len(query.split())
        if word_count > 10:
            complexity_score += 0.2
        elif word_count > 15:
            complexity_score += 0.4
        
        return min(1.0, complexity_score)
    
    def _detect_response_complexity(self, content: str) -> float:
        """Detect response complexity (0.0 = simple, 1.0 = expert)."""
        
        # Analyze response characteristics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Average sentence length as complexity indicator
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        complexity_score = 0.0
        
        # Length-based complexity
        if word_count > 200:
            complexity_score += 0.2
        if word_count > 500:
            complexity_score += 0.2
        
        # Sentence complexity
        if avg_sentence_length > 15:
            complexity_score += 0.2
        if avg_sentence_length > 25:
            complexity_score += 0.2
        
        # Technical terminology
        technical_terms = ['implementation', 'optimization', 'configuration', 'architecture', 'methodology']
        tech_score = sum(1 for term in technical_terms if term.lower() in content.lower())
        complexity_score += min(0.4, tech_score * 0.1)
        
        return min(1.0, complexity_score)
    
    async def _analyze_format_appropriateness(self, content: str, query_type: str) -> float:
        """Analyze if response format matches query type expectations."""
        
        format_score = 0.5  # Default score
        
        if query_type == 'comparison':
            # Should have comparison structure
            if any(word in content.lower() for word in ['vs', 'versus', 'compared to', 'while', 'whereas']):
                format_score += 0.3
            if '|' in content or 'table' in content.lower():  # Table format
                format_score += 0.2
        
        elif query_type == 'tutorial':
            # Should have step-by-step structure
            if any(word in content.lower() for word in ['step', 'first', 'then', 'next', 'finally']):
                format_score += 0.3
            if content.count('\n') > 3:  # Well-structured
                format_score += 0.2
        
        elif query_type == 'review':
            # Should have review structure
            if any(word in content.lower() for word in ['pros', 'cons', 'rating', 'verdict', 'recommendation']):
                format_score += 0.3
            if any(word in content.lower() for word in ['excellent', 'good', 'fair', 'poor', 'outstanding']):
                format_score += 0.2
        
        return min(1.0, format_score)
    
    async def _analyze_keyword_relevance(self, content: str, query: str) -> float:
        """Analyze keyword relevance between query and response."""
        
        # Extract keywords from query
        query_keywords = [word.lower() for word in query.split() if len(word) > 3]
        
        if not query_keywords:
            return 0.5
        
        # Count keyword occurrences in content
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
        
        # Calculate relevance score
        relevance_score = keyword_matches / len(query_keywords)
        
        return min(1.0, relevance_score)
    
    def _generate_quality_flags(self, breakdown: ConfidenceBreakdown) -> List[str]:
        """Generate quality flags based on confidence breakdown."""
        
        flags = []
        
        # Overall confidence flags
        if breakdown.overall_confidence >= 0.9:
            flags.append('excellent_quality')
        elif breakdown.overall_confidence >= 0.8:
            flags.append('high_quality')
        elif breakdown.overall_confidence >= 0.6:
            flags.append('acceptable_quality')
        elif breakdown.overall_confidence >= 0.4:
            flags.append('low_quality')
        else:
            flags.append('poor_quality')
        
        # Category-specific flags
        if breakdown.content_quality < 0.5:
            flags.append('content_quality_concern')
        
        if breakdown.source_quality < 0.5:
            flags.append('source_quality_concern')
        
        if breakdown.query_matching < 0.5:
            flags.append('query_matching_concern')
        
        if breakdown.technical_factors < 0.5:
            flags.append('technical_concern')
        
        # Performance flags
        if breakdown.calculation_time_ms > 1000:
            flags.append('slow_calculation')
        
        return flags
    
    def _generate_improvement_suggestions(
        self, 
        breakdown: ConfidenceBreakdown, 
        query_type: str
    ) -> List[str]:
        """Generate actionable improvement suggestions."""
        
        suggestions = []
        
        # Content quality improvements
        if breakdown.content_quality < 0.7:
            if breakdown.content_metrics.get('completeness', 0) < 0.6:
                suggestions.append("Expand content to address all aspects of the query")
            if breakdown.content_metrics.get('clarity', 0) < 0.6:
                suggestions.append("Improve content clarity and structure")
            if breakdown.content_metrics.get('relevance', 0) < 0.6:
                suggestions.append("Focus more closely on the specific query topic")
        
        # Source quality improvements
        if breakdown.source_quality < 0.7:
            if breakdown.source_metrics.get('source_count', 0) < 3:
                suggestions.append("Include more diverse, high-quality sources")
            if breakdown.source_metrics.get('quality_consistency', 0) < 0.6:
                suggestions.append("Use more consistent, reliable sources")
        
        # Query matching improvements
        if breakdown.query_matching < 0.7:
            if breakdown.query_metrics.get('intent_alignment', 0) < 0.6:
                suggestions.append("Better align response with query intent")
            if breakdown.query_metrics.get('format_appropriateness', 0) < 0.6:
                if query_type == 'comparison':
                    suggestions.append("Structure as a clear comparison with pros/cons")
                elif query_type == 'tutorial':
                    suggestions.append("Organize as step-by-step instructions")
        
        # Technical improvements
        if breakdown.technical_factors < 0.7:
            if breakdown.technical_metrics.get('response_time_score', 0) < 0.6:
                suggestions.append("Optimize response generation for better performance")
            if breakdown.technical_metrics.get('retrieval_quality', 0) < 0.6:
                suggestions.append("Improve source retrieval and ranking")
        
        return suggestions
    
    async def _enhance_response_with_confidence(
        self, 
        response: EnhancedRAGResponse, 
        breakdown: ConfidenceBreakdown,
        query_type: str
    ) -> EnhancedRAGResponse:
        """Enhance response with confidence information."""
        
        # Update confidence factors
        response.confidence_score = breakdown.overall_confidence
        
        # Add confidence metadata
        response.metadata.update({
            'confidence_breakdown': {
                'content_quality': breakdown.content_quality,
                'source_quality': breakdown.source_quality,
                'query_matching': breakdown.query_matching,
                'technical_factors': breakdown.technical_factors,
                'overall_confidence': breakdown.overall_confidence
            },
            'quality_flags': breakdown.quality_flags,
            'improvement_suggestions': breakdown.improvement_suggestions,
            'calculation_time_ms': breakdown.calculation_time_ms,
            'query_type': query_type
        })
        
        # Determine if response should be cached
        should_cache = breakdown.overall_confidence >= self.quality_thresholds['cache_storage']
        
        if should_cache:
            # Calculate cache TTL based on query type and confidence
            base_ttl = self.query_type_configs.get(query_type, {}).get('cache_ttl_hours', 24)
            confidence_multiplier = breakdown.overall_confidence
            cache_ttl_hours = int(base_ttl * confidence_multiplier)
            
            response.metadata['cache_recommendation'] = {
                'should_cache': True,
                'ttl_hours': cache_ttl_hours,
                'cache_key_factors': [
                    'query_type',
                    'confidence_level',
                    'source_quality'
                ]
            }
        else:
            response.metadata['cache_recommendation'] = {
                'should_cache': False,
                'reason': 'below_quality_threshold'
            }
        
        return response
    
    async def _log_confidence_metrics(
        self, 
        breakdown: ConfidenceBreakdown, 
        query_type: str
    ):
        """Log confidence metrics for monitoring and optimization."""
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'query_type': query_type,
            'overall_confidence': breakdown.overall_confidence,
            'content_quality': breakdown.content_quality,
            'source_quality': breakdown.source_quality,
            'query_matching': breakdown.query_matching,
            'technical_factors': breakdown.technical_factors,
            'calculation_time_ms': breakdown.calculation_time_ms,
            'quality_flags': breakdown.quality_flags
        }
        
        # Log to your monitoring system
        self.logger.info("Confidence calculation completed", extra=metrics)
    
    async def should_regenerate_response(
        self, 
        breakdown: ConfidenceBreakdown
    ) -> Tuple[bool, str]:
        """Determine if response should be regenerated based on confidence."""
        
        if breakdown.overall_confidence < self.quality_thresholds['regeneration']:
            return True, f"Low confidence score: {breakdown.overall_confidence:.2f}"
        
        # Check for critical issues
        if 'poor_quality' in breakdown.quality_flags:
            return True, "Poor quality detected"
        
        if breakdown.content_metrics.get('critical_issues', 0) > 0:
            return True, "Critical content issues detected"
        
        return False, "Quality acceptable"
    
    def get_confidence_summary(self, breakdown: ConfidenceBreakdown) -> Dict[str, Any]:
        """Generate human-readable confidence summary."""
        
        return {
            'overall_score': round(breakdown.overall_confidence, 3),
            'quality_level': (
                'Excellent' if breakdown.overall_confidence >= 0.9 else
                'High' if breakdown.overall_confidence >= 0.8 else
                'Good' if breakdown.overall_confidence >= 0.7 else
                'Acceptable' if breakdown.overall_confidence >= 0.6 else
                'Poor'
            ),
            'breakdown': {
                'content_quality': f"{breakdown.content_quality:.2f} (35% weight)",
                'source_quality': f"{breakdown.source_quality:.2f} (25% weight)",
                'query_matching': f"{breakdown.query_matching:.2f} (20% weight)",
                'technical_factors': f"{breakdown.technical_factors:.2f} (20% weight)"
            },
            'recommendations': breakdown.improvement_suggestions[:3],  # Top 3
            'processing_time': f"{breakdown.calculation_time_ms:.1f}ms"
        }


# Integration helper for Universal RAG Chain
class ConfidenceIntegrator:
    """Helper class to integrate confidence calculation with Universal RAG Chain."""
    
    def __init__(self, confidence_calculator: EnhancedConfidenceCalculator):
        self.confidence_calculator = confidence_calculator
    
    async def enhance_rag_response(
        self,
        response: EnhancedRAGResponse,
        query: str,
        query_type: str,
        sources: List[Dict[str, Any]],
        generation_metadata: Dict[str, Any]
    ) -> EnhancedRAGResponse:
        """Enhance RAG response with confidence scoring."""
        
        # Calculate confidence
        breakdown, enhanced_response = await self.confidence_calculator.calculate_enhanced_confidence(
            response=response,
            query=query,
            query_type=query_type,
            sources=sources,
            generation_metadata=generation_metadata
        )
        
        # Check if regeneration is needed
        should_regenerate, reason = await self.confidence_calculator.should_regenerate_response(breakdown)
        
        if should_regenerate:
            enhanced_response.metadata['regeneration_recommended'] = {
                'recommended': True,
                'reason': reason,
                'current_confidence': breakdown.overall_confidence
            }
        
        return enhanced_response

# ============================================================================
# Task 2.3 Enhanced Source Metadata System
# ============================================================================

@dataclass
class EnhancedSourceMetadata:
    """Enhanced source metadata with visual indicators and quality scores."""
    
    # Core metadata
    title: str
    url: str
    content_snippet: str
    
    # Quality scores (0.0 to 1.0)
    authority_score: float
    credibility_score: float
    expertise_score: float
    relevance_score: float
    freshness_score: float
    
    # Visual indicators
    quality_badge: str  # "excellent", "good", "fair", "poor"
    content_type_badge: str  # "academic", "news", "blog", "official", "community"
    expertise_level_badge: str  # "expert", "professional", "general", "beginner"
    
    # Query-type specific metadata
    query_specific_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Technical metadata
    published_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    author: Optional[str] = None
    domain_authority: Optional[float] = None
    
    # Engagement metrics
    citation_count: Optional[int] = None
    social_shares: Optional[int] = None
    user_rating: Optional[float] = None

class EnhancedSourceMetadataGenerator:
    """Generate enhanced source metadata with visual indicators."""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'fair': 0.55,
            'poor': 0.0
        }
        
        self.content_type_indicators = {
            'academic': ['.edu', 'journal', 'research', 'study', 'university', 'scholar'],
            'news': ['news', 'breaking', 'reuters', 'bbc', 'cnn', 'associated press'],
            'official': ['.gov', '.org', 'official', 'government', 'ministry'],
            'blog': ['blog', 'personal', 'medium.com', 'wordpress'],
            'community': ['reddit', 'stackoverflow', 'forum', 'discussion', 'community']
        }
    
    async def generate_enhanced_metadata(
        self, 
        source: Dict[str, Any], 
        query_type: str,
        query: str
    ) -> EnhancedSourceMetadata:
        """Generate comprehensive enhanced metadata for a source."""
        
        # Extract basic information
        title = source.get('title', 'Unknown Title')
        url = source.get('url', '')
        content = source.get('content', source.get('content_preview', ''))
        
        # Calculate quality scores
        authority_score = await self._calculate_authority_score(source)
        credibility_score = await self._calculate_credibility_score(source)
        expertise_score = await self._calculate_expertise_score(source, query_type)
        relevance_score = await self._calculate_relevance_score(source, query)
        freshness_score = await self._calculate_freshness_score(source, query_type)
        
        # Generate visual indicators
        overall_quality = (authority_score + credibility_score + expertise_score + relevance_score) / 4
        quality_badge = self._get_quality_badge(overall_quality)
        content_type_badge = self._get_content_type_badge(url, title, content)
        expertise_level_badge = self._get_expertise_level_badge(expertise_score)
        
        # Generate query-specific metadata
        query_specific_metadata = await self._generate_query_specific_metadata(
            source, query_type, query
        )
        
        # Extract technical metadata
        published_date = self._extract_published_date(source)
        author = source.get('author')
        domain_authority = source.get('domain_authority', source.get('authority_score'))
        
        return EnhancedSourceMetadata(
            title=title,
            url=url,
            content_snippet=content[:200] + "..." if len(content) > 200 else content,
            authority_score=authority_score,
            credibility_score=credibility_score,
            expertise_score=expertise_score,
            relevance_score=relevance_score,
            freshness_score=freshness_score,
            quality_badge=quality_badge,
            content_type_badge=content_type_badge,
            expertise_level_badge=expertise_level_badge,
            query_specific_metadata=query_specific_metadata,
            published_date=published_date,
            author=author,
            domain_authority=domain_authority
        )
    
    async def _calculate_authority_score(self, source: Dict[str, Any]) -> float:
        """Calculate authority score based on source characteristics."""
        score = source.get('authority_score', 0.5)
        
        # Enhance with URL authority indicators
        url = source.get('url', '').lower()
        if any(domain in url for domain in ['.edu', '.gov', '.org']):
            score = min(1.0, score + 0.2)
        elif any(domain in url for domain in ['wikipedia', 'scholar']):
            score = min(1.0, score + 0.15)
        
        return score
    
    async def _calculate_credibility_score(self, source: Dict[str, Any]) -> float:
        """Calculate credibility score based on source characteristics."""
        return source.get('credibility_score', source.get('quality_score', 0.5))
    
    async def _calculate_expertise_score(self, source: Dict[str, Any], query_type: str) -> float:
        """Calculate expertise score based on content and query type."""
        base_score = source.get('expertise_score', 0.5)
        
        content = source.get('content', source.get('content_preview', '')).lower()
        
        # Query type specific expertise indicators
        expertise_indicators = {
            'tutorial': ['step-by-step', 'guide', 'how-to', 'tutorial', 'instructions'],
            'review': ['review', 'rating', 'pros', 'cons', 'experience'],
            'comparison': ['comparison', 'versus', 'vs', 'compared', 'difference'],
            'technical': ['technical', 'implementation', 'architecture', 'specification'],
            'news': ['breaking', 'reporter', 'journalist', 'news', 'update']
        }
        
        if query_type in expertise_indicators:
            indicators = expertise_indicators[query_type]
            matches = sum(1 for indicator in indicators if indicator in content)
            expertise_boost = min(0.3, matches * 0.1)
            base_score = min(1.0, base_score + expertise_boost)
        
        return base_score
    
    async def _calculate_relevance_score(self, source: Dict[str, Any], query: str) -> float:
        """Calculate relevance score based on query matching."""
        return source.get('relevance_score', source.get('similarity_score', 0.5))
    
    async def _calculate_freshness_score(self, source: Dict[str, Any], query_type: str) -> float:
        """Calculate freshness score based on publication date and query type."""
        published_date = source.get('published_date')
        if not published_date:
            return 0.5  # Default if no date available
        
        try:
            if isinstance(published_date, str):
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                pub_date = published_date
            
            current_time = datetime.now()
            age_days = (current_time - pub_date).days
            
            # Freshness requirements vary by query type
            freshness_requirements = {
                'news': 1,        # Very fresh required
                'promotional': 7,  # Moderately fresh
                'review': 30,     # Can be older
                'tutorial': 180,  # Tutorials age well
                'factual': 365    # Facts don't change quickly
            }
            
            max_age = freshness_requirements.get(query_type, 90)
            
            if age_days <= max_age * 0.1:  # Very fresh
                return 1.0
            elif age_days <= max_age * 0.5:  # Fresh
                return 0.8
            elif age_days <= max_age:  # Acceptable
                return 0.6
            else:  # Stale
                return 0.3
                
        except (ValueError, TypeError):
            return 0.5  # Default if date parsing fails
    
    def _get_quality_badge(self, overall_quality: float) -> str:
        """Get quality badge based on overall quality score."""
        for badge, threshold in self.quality_thresholds.items():
            if overall_quality >= threshold:
                return badge
        return 'poor'
    
    def _get_content_type_badge(self, url: str, title: str, content: str) -> str:
        """Determine content type badge based on URL and content analysis."""
        combined_text = f"{url} {title} {content}".lower()
        
        # Score each content type
        type_scores = {}
        for content_type, indicators in self.content_type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            if score > 0:
                type_scores[content_type] = score
        
        # Return the highest scoring type or 'general' if none match
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _get_expertise_level_badge(self, expertise_score: float) -> str:
        """Get expertise level badge based on expertise score."""
        if expertise_score >= 0.8:
            return 'expert'
        elif expertise_score >= 0.6:
            return 'professional'
        elif expertise_score >= 0.4:
            return 'general'
        else:
            return 'beginner'
    
    async def _generate_query_specific_metadata(
        self, 
        source: Dict[str, Any], 
        query_type: str, 
        query: str
    ) -> Dict[str, Any]:
        """Generate metadata specific to the query type."""
        
        metadata = {}
        content = source.get('content', source.get('content_preview', '')).lower()
        
        if query_type == 'casino_review':
            metadata.update({
                'has_rating': any(word in content for word in ['rating', 'score', 'stars']),
                'has_pros_cons': any(word in content for word in ['pros', 'cons', 'advantages', 'disadvantages']),
                'has_bonuses': any(word in content for word in ['bonus', 'promotion', 'offer', 'welcome']),
                'has_games': any(word in content for word in ['games', 'slots', 'poker', 'blackjack']),
                'has_payment_info': any(word in content for word in ['payment', 'deposit', 'withdrawal', 'banking'])
            })
        
        elif query_type == 'game_guide':
            metadata.update({
                'has_rules': any(word in content for word in ['rules', 'how to play', 'gameplay']),
                'has_strategy': any(word in content for word in ['strategy', 'tips', 'tactics', 'advice']),
                'has_examples': any(word in content for word in ['example', 'instance', 'demonstration']),
                'difficulty_indicators': [word for word in ['beginner', 'intermediate', 'advanced', 'expert'] if word in content]
            })
        
        elif query_type == 'promotion_analysis':
            metadata.update({
                'has_terms': any(word in content for word in ['terms', 'conditions', 'requirements']),
                'has_wagering': any(word in content for word in ['wagering', 'playthrough', 'rollover']),
                'has_expiry': any(word in content for word in ['expires', 'expiry', 'deadline', 'time limit']),
                'value_indicators': [word for word in ['high value', 'generous', 'competitive', 'standard'] if word in content]
            })
        
        elif query_type == 'comparison':
            metadata.update({
                'comparison_aspects': self._extract_comparison_aspects(content),
                'has_table': 'table' in content or '|' in content,
                'has_verdict': any(word in content for word in ['verdict', 'winner', 'best choice', 'recommendation']),
                'comparison_criteria': self._extract_comparison_criteria(content)
            })
        
        elif query_type == 'tutorial':
            metadata.update({
                'step_count': len([word for word in content.split() if word.startswith('step')]),
                'has_prerequisites': any(word in content for word in ['prerequisite', 'requirement', 'before you start']),
                'has_troubleshooting': any(word in content for word in ['troubleshoot', 'problem', 'issue', 'error']),
                'difficulty_level': self._assess_tutorial_difficulty(content)
            })
        
        return metadata
    
    def _extract_comparison_aspects(self, content: str) -> List[str]:
        """Extract comparison aspects from content."""
        aspects = []
        comparison_words = ['price', 'quality', 'features', 'performance', 'usability', 'support']
        
        for aspect in comparison_words:
            if aspect in content:
                aspects.append(aspect)
        
        return aspects
    
    def _extract_comparison_criteria(self, content: str) -> List[str]:
        """Extract comparison criteria from content."""
        criteria = []
        criteria_patterns = [
            'based on', 'criteria', 'factors', 'considerations',
            'evaluated by', 'measured by', 'compared on'
        ]
        
        for pattern in criteria_patterns:
            if pattern in content:
                # Simple extraction - in a real implementation, you'd use NLP
                criteria.append(pattern)
        
        return criteria
    
    def _assess_tutorial_difficulty(self, content: str) -> str:
        """Assess the difficulty level of a tutorial."""
        beginner_indicators = ['beginner', 'basic', 'introduction', 'getting started', 'simple']
        advanced_indicators = ['advanced', 'expert', 'complex', 'sophisticated', 'in-depth']
        
        beginner_count = sum(1 for indicator in beginner_indicators if indicator in content)
        advanced_count = sum(1 for indicator in advanced_indicators if indicator in content)
        
        if advanced_count > beginner_count:
            return 'advanced'
        elif beginner_count > 0:
            return 'beginner'
        else:
            return 'intermediate'
    
    def _extract_published_date(self, source: Dict[str, Any]) -> Optional[datetime]:
        """Extract and parse publication date from source."""
        date_fields = ['published_date', 'date', 'publication_date', 'created_at']
        
        for field in date_fields:
            if field in source and source[field]:
                try:
                    date_value = source[field]
                    if isinstance(date_value, str):
                        # Try various date formats
                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%S']:
                            try:
                                return datetime.strptime(date_value, fmt)
                            except ValueError:
                                continue
                        
                        # Try ISO format with timezone
                        try:
                            return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                        except ValueError:
                            pass
                    
                    elif isinstance(date_value, datetime):
                        return date_value
                        
                except Exception as e:
                    logger.debug(f"Error parsing date {date_value}: {e}")
                    continue
        
        return None


# ============================================================================
# GLOBAL SYSTEM ORCHESTRATOR
# ============================================================================

class UniversalRAGEnhancementSystem:
    """
    Global orchestrator for the Universal RAG CMS Enhancement System.
    
    Integrates all components:
    - Enhanced Confidence Scoring
    - Source Quality Analysis
    - Intelligent Caching
    - Response Validation
    - Performance Monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Universal RAG Enhancement System."""
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize all subsystems
        self.source_analyzer = SourceQualityAnalyzer()
        self.cache_system = IntelligentCache(
            strategy=CacheStrategy.ADAPTIVE,
            max_size=self.config.get('cache_max_size', 10000)
        )
        self.response_validator = ResponseValidator(self.config.get('validation_config'))
        self.confidence_calculator = EnhancedConfidenceCalculator(
            source_quality_analyzer=self.source_analyzer,
            cache_system=self.cache_system,
            response_validator=self.response_validator,
            config=self.config.get('confidence_config')
        )
        self.metadata_generator = EnhancedSourceMetadataGenerator()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # System configuration
        self.system_config = SystemConfiguration(
            **self.config.get('system_config', {})
        )
        
        # Validate configuration
        config_issues = self.system_config.validate_configuration()
        if config_issues:
            self.logger.warning(f"Configuration issues: {config_issues}")
        
        self.logger.info("Universal RAG Enhancement System initialized")
    
    async def enhance_rag_response(
        self,
        response_content: str,
        query: str,
        query_type: str,
        sources: List[Dict[str, Any]],
        generation_metadata: Dict[str, Any]
    ) -> EnhancedRAGResponse:
        """
        Main entry point for enhancing RAG responses.
        
        This orchestrates all enhancement processes:
        1. Create base EnhancedRAGResponse
        2. Analyze source quality
        3. Validate response
        4. Calculate confidence
        5. Generate enhanced metadata
        6. Make caching decisions
        7. Track performance
        """
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = await self.cache_system.get(query, query_type)
            if cached_response:
                cached_response.cached = True
                cached_response.response_time = time.time() - start_time
                self.performance_tracker.record_request(cached_response)
                return cached_response
            
            # Create base response object
            base_response = EnhancedRAGResponse(
                content=response_content,
                sources=sources,
                response_time=0.0,  # Will be updated
                cached=False
            )
            
            # Parallel enhancement processes
            enhancement_tasks = [
                self._enhance_sources_with_metadata(sources, query_type, query),
                self.response_validator.validate_response(response_content, query, sources),
                self._analyze_generation_metadata(generation_metadata)
            ]
            
            # Execute enhancements in parallel
            results = await asyncio.gather(*enhancement_tasks, return_exceptions=True)
            
            # Process results
            enhanced_sources = results[0] if not isinstance(results[0], Exception) else sources
            validation_metrics, validation_issues = results[1] if not isinstance(results[1], Exception) else (None, [])
            processed_metadata = results[2] if not isinstance(results[2], Exception) else generation_metadata
            
            # Update response with enhanced data
            base_response.sources = enhanced_sources
            base_response.metadata.update(processed_metadata)
            
            # Integrate validation results
            if validation_metrics:
                base_response = ValidationIntegrator.update_rag_response(
                    base_response, validation_metrics, validation_issues
                )
            
            # Calculate comprehensive confidence
            confidence_breakdown, enhanced_response = await self.confidence_calculator.calculate_enhanced_confidence(
                response=base_response,
                query=query,
                query_type=query_type,
                sources=enhanced_sources,
                generation_metadata=processed_metadata
            )
            
            # Final response timing and metadata
            enhanced_response.response_time = time.time() - start_time
            enhanced_response.processing_time = enhanced_response.response_time
            
            # Make caching decision
            await self._handle_caching_decision(enhanced_response, query, query_type, confidence_breakdown)
            
            # Track performance
            self.performance_tracker.record_request(enhanced_response)
            
            # Log system metrics
            await self._log_system_metrics(enhanced_response, confidence_breakdown)
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Enhancement system error: {e}")
            
            # Return minimal response on error
            error_response = EnhancedRAGResponse(
                content=response_content,
                sources=sources,
                confidence_score=0.5,
                response_time=time.time() - start_time,
                errors=[f"Enhancement error: {str(e)}"],
                fallback_used=True
            )
            
            return error_response
    
    async def _enhance_sources_with_metadata(
        self, 
        sources: List[Dict[str, Any]], 
        query_type: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Enhance sources with comprehensive metadata."""
        
        enhanced_sources = []
        
        for source in sources:
            try:
                # Generate enhanced metadata
                enhanced_metadata = await self.metadata_generator.generate_enhanced_metadata(
                    source, query_type, query
                )
                
                # Create enhanced source dictionary
                enhanced_source = source.copy()
                enhanced_source.update({
                    'enhanced_metadata': enhanced_metadata,
                    'quality_score': (
                        enhanced_metadata.authority_score +
                        enhanced_metadata.credibility_score +
                        enhanced_metadata.expertise_score +
                        enhanced_metadata.relevance_score
                    ) / 4,
                    'quality_badge': enhanced_metadata.quality_badge,
                    'content_type': enhanced_metadata.content_type_badge,
                    'expertise_level': enhanced_metadata.expertise_level_badge
                })
                
                enhanced_sources.append(enhanced_source)
                
            except Exception as e:
                self.logger.warning(f"Error enhancing source metadata: {e}")
                enhanced_sources.append(source)  # Fallback to original
        
        return enhanced_sources
    
    async def _analyze_generation_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and enhance generation metadata."""
        
        enhanced_metadata = metadata.copy()
        
        # Add system-level metadata
        enhanced_metadata.update({
            'enhancement_system_version': '2.0',
            'enhancement_timestamp': datetime.utcnow().isoformat(),
            'system_performance': {
                'cache_hit_rate': self.performance_tracker.get_cache_hit_rate(),
                'average_response_time': self.performance_tracker.metrics['avg_response_time'],
                'total_requests': self.performance_tracker.metrics['total_requests']
            }
        })
        
        return enhanced_metadata
    
    async def _handle_caching_decision(
        self,
        response: EnhancedRAGResponse,
        query: str,
        query_type: str,
        confidence_breakdown: ConfidenceBreakdown
    ):
        """Make intelligent caching decisions."""
        
        # Check if response meets caching criteria
        should_cache = (
            confidence_breakdown.overall_confidence >= self.system_config.quality_threshold and
            not response.errors and
            self.system_config.enable_intelligent_caching
        )
        
        if should_cache:
            try:
                await self.cache_system.set(query, response, query_type)
                response.metadata['cached'] = True
                response.metadata['cache_decision'] = 'stored'
            except Exception as e:
                self.logger.warning(f"Caching failed: {e}")
                response.metadata['cache_decision'] = 'failed'
        else:
            response.metadata['cache_decision'] = 'skipped'
            response.metadata['cache_skip_reason'] = (
                'low_confidence' if confidence_breakdown.overall_confidence < self.system_config.quality_threshold
                else 'has_errors' if response.errors
                else 'caching_disabled'
            )
    
    async def _log_system_metrics(
        self,
        response: EnhancedRAGResponse,
        confidence_breakdown: ConfidenceBreakdown
    ):
        """Log comprehensive system metrics."""
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'response_time': response.response_time,
            'confidence_score': response.confidence_score,
            'source_count': len(response.sources),
            'cached': response.cached,
            'has_errors': bool(response.errors),
            'quality_level': response.response_quality_level.value,
            'confidence_calculation_time': confidence_breakdown.calculation_time_ms,
            'system_performance': self.performance_tracker.get_performance_summary()
        }
        
        self.logger.info("System enhancement completed", extra=metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            'system_health': 'healthy',  # Could add health checks
            'configuration': {
                'cache_strategy': self.cache_system.strategy.value,
                'cache_size': len(self.cache_system.cache),
                'cache_max_size': self.cache_system.max_size,
                'quality_threshold': self.system_config.quality_threshold
            },
            'performance': self.performance_tracker.get_performance_summary(),
            'cache_metrics': self.cache_system.get_performance_metrics(),
            'uptime': 'N/A',  # Could add uptime tracking
            'version': '2.0'
        }
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on usage patterns."""
        
        optimization_results = {}
        
        # Cache optimization
        expired_count = await self.cache_system.clear_expired()
        optimization_results['cache_cleanup'] = f"Removed {expired_count} expired entries"
        
        # Performance analysis
        performance_summary = self.performance_tracker.get_performance_summary()
        
        # Suggest optimizations
        suggestions = []
        
        if performance_summary['cache_hit_rate'] < 0.5:
            suggestions.append("Consider adjusting cache strategy or TTL settings")
        
        if performance_summary['avg_response_time'] > 2000:
            suggestions.append("Response times are high - consider performance optimization")
        
        if performance_summary['error_rate'] > 0.1:
            suggestions.append("Error rate is elevated - review error logs")
        
        optimization_results['suggestions'] = suggestions
        optimization_results['current_performance'] = performance_summary
        
        return optimization_results


# ============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ============================================================================

def create_universal_rag_enhancement_system(config: Optional[Dict[str, Any]] = None) -> UniversalRAGEnhancementSystem:
    """
    Factory function to create a fully configured Universal RAG Enhancement System.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured UniversalRAGEnhancementSystem instance
    """
    
    default_config = {
        'cache_max_size': 10000,
        'system_config': {
            'enable_enhanced_confidence': True,
            'enable_intelligent_caching': True,
            'enable_response_validation': True,
            'quality_threshold': 0.75,
            'max_response_time': 2000,
            'log_level': 'INFO'
        },
        'confidence_config': {
            'weights': {
                'content_quality': 0.35,
                'source_quality': 0.25,
                'query_matching': 0.20,
                'technical_factors': 0.20
            }
        },
        'validation_config': {
            'min_response_length': 50,
            'max_response_length': 5000,
            'require_citations': False,
            'strict_mode': False
        }
    }
    
    if config:
        # Deep merge configurations
        merged_config = default_config.copy()
        for key, value in config.items():
            if isinstance(value, dict) and key in merged_config:
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        config = merged_config
    else:
        config = default_config
    
    return UniversalRAGEnhancementSystem(config)


async def enhance_rag_response_simple(
    response_content: str,
    query: str,
    sources: List[Dict[str, Any]],
    query_type: str = 'general'
) -> EnhancedRAGResponse:
    """
    Simplified function for enhancing RAG responses.
    
    This provides a simple interface for basic enhancement without
    requiring detailed configuration.
    """
    
    system = create_universal_rag_enhancement_system()
    
    return await system.enhance_rag_response(
        response_content=response_content,
        query=query,
        query_type=query_type,
        sources=sources,
        generation_metadata={}
    )


def get_confidence_factors_template() -> ConfidenceFactors:
    """Get a template ConfidenceFactors object with default values."""
    return ConfidenceFactors()


def calculate_simple_confidence(
    content_quality: float,
    source_quality: float,
    query_matching: float,
    technical_factors: float
) -> float:
    """
    Calculate a simple confidence score using default weights.
    
    Args:
        content_quality: Content quality score (0.0-1.0)
        source_quality: Source quality score (0.0-1.0)
        query_matching: Query matching score (0.0-1.0)
        technical_factors: Technical factors score (0.0-1.0)
        
    Returns:
        Weighted confidence score (0.0-1.0)
    """
    
    return (
        content_quality * 0.35 +
        source_quality * 0.25 +
        query_matching * 0.20 +
        technical_factors * 0.20
    )


# ============================================================================
# MAIN ENTRY POINT AND TESTING
# ============================================================================

def enrich_sources_with_task23_metadata(
    sources: List[Dict[str, Any]], 
    query_type: str, 
    query: str
) -> List[Dict[str, Any]]:
    """
    Enhance sources with Task 2.3 specific metadata and enrichments.
    
    Args:
        sources: List of source dictionaries
        query_type: Type of query being processed
        query: Original query string
    
    Returns:
        Enhanced sources with additional metadata
    """
    enhanced_sources = []
    
    for source in sources:
        enhanced_source = source.copy()
        
        # Add Task 2.3 enhancement metadata
        enhanced_source['task23_metadata'] = {
            'enhancement_timestamp': time.time(),
            'query_type': query_type,
            'query_relevance_score': 0.85,  # Mock score
            'content_quality_score': 0.80,
            'source_authority_score': 0.75,
            'enhancement_version': '2.3.0'
        }
        
        # Add contextual enrichment
        if 'content' in enhanced_source:
            enhanced_source['enhanced_content_preview'] = enhanced_source['content'][:200] + "..."
        
        # Add quality indicators
        enhanced_source['quality_indicators'] = {
            'has_citations': 'citations' in enhanced_source.get('content', '').lower(),
            'content_length': len(enhanced_source.get('content', '')),
            'has_metadata': bool(enhanced_source.get('metadata', {})),
            'source_type': enhanced_source.get('metadata', {}).get('source_type', 'unknown')
        }
        
        enhanced_sources.append(enhanced_source)
    
    return enhanced_sources


async def main():
    """Main function for testing the enhanced confidence scoring system."""
    
    # Example usage
    system = create_universal_rag_enhancement_system()
    
    # Test data
    test_query = "What are the best online casinos for slots in 2024?"
    test_content = """
    Based on our comprehensive analysis, here are the top online casinos for slots in 2024:
    
    1. **Casino Royal** - Excellent game variety with over 1,500 slot games
    2. **Lucky Spin Casino** - Best welcome bonus up to $1,000
    3. **Mega Slots Palace** - Highest RTP rates averaging 97.2%
    
    Each casino has been evaluated based on game quality, bonuses, security, and player reviews.
    """
    
    test_sources = [
        {
            'title': 'Top Online Casinos 2024 Review',
            'url': 'https://casinoexpert.com/reviews/2024',
            'content': 'Comprehensive casino reviews and ratings...',
            'quality_score': 0.85,
            'published_date': '2024-01-15'
        },
        {
            'title': 'Best Slot Games Guide',
            'url': 'https://slotsguru.com/best-slots',
            'content': 'Expert guide to slot games and casinos...',
            'quality_score': 0.78,
            'published_date': '2024-02-01'
        }
    ]
    
    # Enhance the response
    enhanced_response = await system.enhance_rag_response(
        response_content=test_content,
        query=test_query,
        query_type='casino_review',
        sources=test_sources,
        generation_metadata={'response_time_ms': 1200}
    )
    
    # Print results
    print("Enhanced RAG Response:")
    print(f"Confidence Score: {enhanced_response.confidence_score:.3f}")
    print(f"Quality Level: {enhanced_response.response_quality_level.value}")
    print(f"Source Quality Tier: {enhanced_response.source_quality_tier.value}")
    print(f"Processing Time: {enhanced_response.processing_time:.3f}s")
    print(f"Cached: {enhanced_response.cached}")
    
    if enhanced_response.errors:
        print(f"Errors: {enhanced_response.errors}")
    
    # Print system status
    status = system.get_system_status()
    print(f"\nSystem Status: {status['system_health']}")
    print(f"Cache Hit Rate: {status['performance']['cache_hit_rate']:.2%}")
    print(f"Average Response Time: {status['performance']['avg_response_time']:.3f}s")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Core Classes
    'EnhancedRAGResponse',
    'ConfidenceFactors',
    'ConfidenceBreakdown',
    'EnhancedSourceMetadata',
    
    # Main System Classes
    'UniversalRAGEnhancementSystem',
    'EnhancedConfidenceCalculator',
    'SourceQualityAnalyzer',
    'IntelligentCache',
    'ResponseValidator',
    'EnhancedSourceMetadataGenerator',
    
    # Utility Classes
    'PerformanceTracker',
    'SystemConfiguration',
    'CacheEntry',
    'ValidationIssue',
    'ValidationMetrics',
    
    # Enums
    'SourceQualityTier',
    'ResponseQualityLevel',
    'CacheStrategy',
    'ValidationSeverity',
    'ValidationCategory',
    'ConfidenceFactorType',
    
    # Factory Functions
    'create_universal_rag_enhancement_system',
    'enhance_rag_response_simple',
    'get_confidence_factors_template',
    'calculate_simple_confidence',
    
    # Integration Helpers
    'ConfidenceIntegrator',
    'ValidationIntegrator',
    
    # Global Instances (for backward compatibility)
    'intelligent_cache',
    'response_validator',
    'performance_tracker'
]