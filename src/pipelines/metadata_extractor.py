#!/usr/bin/env python3
"""
Metadata Extraction Pipeline - Task 4.4
Comprehensive metadata extraction for enhanced content understanding and retrieval

✅ INTEGRATIONS:
- Task 1: Supabase foundation with proper schema
- Task 2: Enhanced confidence scoring system (CORRECTED IMPORTS)
- Task 3: Contextual retrieval integration
- Task 4.2: Content type detection integration
"""

import asyncio
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

# NLP and text processing
import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
from langdetect import detect, DetectorFactory
from collections import Counter, defaultdict

# ✅ CORRECTED IMPORTS - Using actual file structure
from src.chains.enhanced_confidence_scoring_system import (
    SourceQualityAnalyzer,
    IntelligentCache,
    EnhancedRAGResponse
)
from src.pipelines.content_type_detector import ContentTypeDetector, ContentType

# Set up logging
logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

class MetadataCategory(Enum):
    """Categories of metadata that can be extracted."""
    BASIC = "basic"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    QUALITY = "quality"
    DOMAIN_SPECIFIC = "domain_specific"
    TECHNICAL = "technical"

@dataclass
class ExtractedMetadata:
    """Comprehensive metadata extracted from content."""
    
    # Basic metadata
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    language: Optional[str] = None
    word_count: int = 0
    character_count: int = 0
    
    # Structural metadata
    paragraph_count: int = 0
    sentence_count: int = 0
    heading_count: int = 0
    list_count: int = 0
    link_count: int = 0
    image_count: int = 0
    
    # Semantic metadata
    keywords: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    sentiment_score: Optional[float] = None
    
    # Quality metadata
    readability_score: Optional[float] = None
    reading_grade_level: Optional[float] = None
    content_quality_score: Optional[float] = None
    completeness_score: Optional[float] = None
    
    # Domain-specific metadata
    domain_specific: Dict[str, Any] = field(default_factory=dict)
    
    # Technical metadata
    content_hash: Optional[str] = None
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    extraction_confidence: float = 0.0
    processing_time_ms: float = 0.0
    
    # Source information
    source_url: Optional[str] = None
    source_type: Optional[str] = None
    content_type: Optional[ContentType] = None

class MetadataExtractor:
    """
    Comprehensive metadata extraction system for diverse content types.
    
    Features:
    - Multi-category metadata extraction
    - Content type-aware processing
    - Pluggable extractors for different domains
    - Quality scoring and validation
    - Performance optimization with caching
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the MetadataExtractor with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.content_detector = ContentTypeDetector()
        self.source_analyzer = SourceQualityAnalyzer()
        self.cache = IntelligentCache()
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Extraction strategies by content type
        self.extraction_strategies = {
            ContentType.ARTICLE: self._extract_article_metadata,
            ContentType.REVIEW: self._extract_review_metadata,
            ContentType.TECHNICAL_DOC: self._extract_technical_metadata,
            ContentType.NEWS: self._extract_news_metadata,
            ContentType.BLOG_POST: self._extract_blog_metadata,
            ContentType.ACADEMIC: self._extract_academic_metadata,
            ContentType.LEGAL: self._extract_legal_metadata,
            ContentType.MARKETING: self._extract_marketing_metadata
        }
        
        # Domain-specific extractors
        self.domain_extractors = {
            'casino': self._extract_casino_metadata,
            'gaming': self._extract_gaming_metadata,
            'finance': self._extract_finance_metadata,
            'health': self._extract_health_metadata,
            'technology': self._extract_technology_metadata
        }
        
        self.logger.info("MetadataExtractor initialized successfully")
    
    def _initialize_nlp_models(self):
        """Initialize NLP models and resources."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found. Some features will be limited.")
                self.nlp = None
            
            # Initialize NLTK components
            from nltk.corpus import stopwords
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            self.stop_words = set(stopwords.words('english'))
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP models: {e}")
            self.nlp = None
            self.stop_words = set()
            self.sentiment_analyzer = None
    
    async def extract_metadata(
        self,
        content: str,
        source_url: Optional[str] = None,
        content_type: Optional[ContentType] = None,
        domain: Optional[str] = None
    ) -> ExtractedMetadata:
        """
        Extract comprehensive metadata from content.
        
        Args:
            content: The text content to analyze
            source_url: Optional source URL for additional context
            content_type: Optional content type (will be detected if not provided)
            domain: Optional domain for domain-specific extraction
            
        Returns:
            ExtractedMetadata object with comprehensive metadata
        """
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(content, source_url, content_type, domain)
            cached_metadata = await self.cache.get(cache_key)
            if cached_metadata:
                return cached_metadata.metadata.get('extracted_metadata')
            
            # Detect content type if not provided
            if not content_type:
                detection_result = await self.content_detector.detect_content_type(content, source_url)
                content_type = detection_result.content_type
            
            # Initialize metadata object
            metadata = ExtractedMetadata(
                content_hash=hashlib.md5(content.encode()).hexdigest(),
                source_url=source_url,
                content_type=content_type,
                extraction_timestamp=start_time
            )
            
            # Parallel extraction tasks for performance
            extraction_tasks = [
                self._extract_basic_metadata(content, metadata),
                self._extract_structural_metadata(content, metadata),
                self._extract_semantic_metadata(content, metadata),
                self._extract_quality_metadata(content, metadata)
            ]
            
            # Execute extractions in parallel
            await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            # Content type-specific extraction
            if content_type in self.extraction_strategies:
                await self.extraction_strategies[content_type](content, metadata)
            
            # Domain-specific extraction
            if domain and domain in self.domain_extractors:
                await self.domain_extractors[domain](content, metadata)
            
            # Calculate processing time and confidence
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            metadata.processing_time_ms = processing_time
            metadata.extraction_confidence = self._calculate_extraction_confidence(metadata)
            
            # Cache the result
            await self._cache_metadata(cache_key, metadata)
            
            self.logger.debug(f"Metadata extraction completed in {processing_time:.2f}ms")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return self._create_fallback_metadata(content, source_url, content_type)
    
    async def _extract_basic_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract basic metadata like title, author, dates, language."""
        
        # Extract title (first heading or first line)
        title_patterns = [
            r'^#\s+(.+)$',  # Markdown heading
            r'^(.+)\n[=\-]+$',  # Underlined heading
            r'<h[1-6][^>]*>([^<]+)</h[1-6]>',  # HTML heading
            r'^\*\*(.+)\*\*$',  # Bold title
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                metadata.title = match.group(1).strip()
                break
        
        # If no title found, use first sentence
        if not metadata.title:
            sentences = re.split(r'[.!?]+', content)
            if sentences:
                metadata.title = sentences[0].strip()[:100]  # Limit length
        
        # Extract author information
        author_patterns = [
            r'(?:by|author|written by)[\s:]+([^\n\r]+)',
            r'@([a-zA-Z0-9_]+)',  # Social media handle
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Name pattern
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata.author = match.group(1).strip()
                break
        
        # Extract dates
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # ISO format
            r'(\d{1,2}/\d{1,2}/\d{4})',  # US format
            r'(\d{1,2}-\d{1,2}-\d{4})',  # European format
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})',  # Month format
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    # Try to parse the first date found
                    date_str = matches[0]
                    metadata.creation_date = self._parse_date(date_str)
                    break
                except:
                    continue
        
        # Detect language
        try:
            if len(content) > 50:  # Need sufficient text for detection
                detected_lang = detect(content)
                metadata.language = detected_lang
        except:
            metadata.language = 'en'  # Default to English
        
        # Basic counts
        metadata.word_count = len(content.split())
        metadata.character_count = len(content)
    
    async def _extract_structural_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract structural metadata about content organization."""
        
        # Count paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        metadata.paragraph_count = len(paragraphs)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        metadata.sentence_count = len(sentences)
        
        # Count headings
        heading_patterns = [
            r'^#+\s+.+$',  # Markdown headings
            r'^.+\n[=\-]+$',  # Underlined headings
            r'<h[1-6][^>]*>.*?</h[1-6]>',  # HTML headings
            r'^\*\*.+\*\*$',  # Bold headings
        ]
        
        heading_count = 0
        for pattern in heading_patterns:
            heading_count += len(re.findall(pattern, content, re.MULTILINE | re.IGNORECASE))
        metadata.heading_count = heading_count
        
        # Count lists
        list_patterns = [
            r'^\s*[\*\-\+]\s+',  # Bullet lists
            r'^\s*\d+\.\s+',  # Numbered lists
            r'<[uo]l>.*?</[uo]l>',  # HTML lists
        ]
        
        list_count = 0
        for pattern in list_patterns:
            list_count += len(re.findall(pattern, content, re.MULTILINE | re.IGNORECASE))
        metadata.list_count = list_count
        
        # Count links
        link_patterns = [
            r'https?://[^\s]+',  # URLs
            r'\[([^\]]+)\]\([^\)]+\)',  # Markdown links
            r'<a[^>]*>.*?</a>',  # HTML links
        ]
        
        link_count = 0
        for pattern in link_patterns:
            link_count += len(re.findall(pattern, content, re.IGNORECASE))
        metadata.link_count = link_count
        
        # Count images
        image_patterns = [
            r'!\[([^\]]*)\]\([^\)]+\)',  # Markdown images
            r'<img[^>]*>',  # HTML images
        ]
        
        image_count = 0
        for pattern in image_patterns:
            image_count += len(re.findall(pattern, content, re.IGNORECASE))
        metadata.image_count = image_count
    
    async def _extract_semantic_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract semantic metadata like keywords, entities, topics."""
        
        # Extract keywords using frequency analysis
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        words = [word for word in words if word not in self.stop_words]
        word_freq = Counter(words)
        
        # Top keywords (excluding very common words)
        metadata.keywords = [word for word, freq in word_freq.most_common(20) if freq > 1]
        
        # Extract key phrases (2-3 word combinations)
        if self.nlp:
            try:
                doc = self.nlp(content[:1000000])  # Limit for performance
                
                # Extract noun phrases as key phrases
                noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                              if len(chunk.text.split()) <= 3 and len(chunk.text) > 3]
                phrase_freq = Counter(noun_phrases)
                metadata.key_phrases = [phrase for phrase, freq in phrase_freq.most_common(15)]
                
                # Extract named entities
                entities = defaultdict(list)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        entities[ent.label_].append(ent.text)
                
                metadata.entities = dict(entities)
                
            except Exception as e:
                self.logger.debug(f"spaCy processing failed: {e}")
        
        # Generate summary (first few sentences or key sentences)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        
        if sentences:
            # Simple extractive summary - first 2-3 sentences
            summary_sentences = sentences[:min(3, len(sentences))]
            metadata.summary = '. '.join(summary_sentences) + '.'
        
        # Topic extraction (simple keyword-based)
        topic_keywords = {
            'technology': ['software', 'computer', 'digital', 'tech', 'programming', 'code'],
            'business': ['company', 'market', 'business', 'revenue', 'profit', 'strategy'],
            'health': ['health', 'medical', 'doctor', 'treatment', 'patient', 'medicine'],
            'finance': ['money', 'investment', 'financial', 'bank', 'economy', 'market'],
            'gaming': ['game', 'casino', 'bet', 'play', 'gambling', 'poker', 'slots'],
            'education': ['learn', 'education', 'student', 'school', 'university', 'course'],
            'travel': ['travel', 'trip', 'vacation', 'hotel', 'destination', 'tourism'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'cuisine']
        }
        
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score >= 2:  # Threshold for topic detection
                detected_topics.append(topic)
        
        metadata.topics = detected_topics
        
        # Sentiment analysis
        if self.sentiment_analyzer:
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(content[:5000])
                metadata.sentiment_score = sentiment_scores['compound']
            except Exception as e:
                self.logger.debug(f"Sentiment analysis failed: {e}")
    
    async def _extract_quality_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract quality-related metadata."""
        
        try:
            # Readability scores
            if len(content) > 100:  # Need sufficient text
                metadata.readability_score = flesch_reading_ease(content)
                metadata.reading_grade_level = flesch_kincaid_grade(content)
        except Exception as e:
            self.logger.debug(f"Readability calculation failed: {e}")
        
        # Content quality heuristics
        quality_indicators = {
            'has_structure': metadata.heading_count > 0 or metadata.list_count > 0,
            'adequate_length': metadata.word_count >= 100,
            'has_links': metadata.link_count > 0,
            'proper_sentences': metadata.sentence_count >= 3,
            'varied_vocabulary': len(set(metadata.keywords)) > 5
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        metadata.content_quality_score = quality_score
        
        # Completeness score based on metadata richness
        completeness_factors = {
            'has_title': metadata.title is not None,
            'has_author': metadata.author is not None,
            'has_date': metadata.creation_date is not None,
            'has_keywords': len(metadata.keywords) > 0,
            'has_summary': metadata.summary is not None,
            'has_topics': len(metadata.topics) > 0
        }
        
        completeness_score = sum(completeness_factors.values()) / len(completeness_factors)
        metadata.completeness_score = completeness_score
    
    async def _extract_article_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract article-specific metadata."""
        metadata.domain_specific['article'] = {
            'estimated_read_time': max(1, metadata.word_count // 200),  # ~200 WPM
            'has_byline': metadata.author is not None,
            'has_publication_date': metadata.creation_date is not None,
            'article_structure_score': self._calculate_article_structure_score(metadata)
        }
    
    async def _extract_review_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract review-specific metadata."""
        
        # Look for rating indicators
        rating_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*(\d+)',  # "4.5 out of 5"
            r'(\d+(?:\.\d+)?)\s*stars?',  # "4.5 stars"
            r'rating:?\s*(\d+(?:\.\d+)?)',  # "Rating: 4.5"
        ]
        
        rating = None
        for pattern in rating_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    rating = float(match.group(1))
                    break
                except:
                    continue
        
        # Look for pros/cons
        has_pros = bool(re.search(r'\b(?:pros?|advantages?|benefits?)\b', content, re.IGNORECASE))
        has_cons = bool(re.search(r'\b(?:cons?|disadvantages?|drawbacks?)\b', content, re.IGNORECASE))
        
        metadata.domain_specific['review'] = {
            'rating': rating,
            'has_pros_cons': has_pros and has_cons,
            'has_rating': rating is not None,
            'review_completeness': self._calculate_review_completeness(content, rating, has_pros, has_cons)
        }
    
    async def _extract_casino_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract casino-specific metadata."""
        
        casino_indicators = {
            'games': ['slots', 'poker', 'blackjack', 'roulette', 'baccarat', 'craps'],
            'bonuses': ['bonus', 'welcome', 'deposit', 'free spins', 'cashback'],
            'payments': ['visa', 'mastercard', 'paypal', 'bitcoin', 'withdrawal', 'deposit'],
            'features': ['live dealer', 'mobile', 'app', 'customer support', 'license']
        }
        
        content_lower = content.lower()
        casino_data = {}
        
        for category, terms in casino_indicators.items():
            found_terms = [term for term in terms if term in content_lower]
            casino_data[f'{category}_mentioned'] = found_terms
            casino_data[f'{category}_count'] = len(found_terms)
        
        # Look for specific casino information
        license_match = re.search(r'licen[sc]ed?\s+(?:by|in)\s+([^.\n]+)', content, re.IGNORECASE)
        if license_match:
            casino_data['license_info'] = license_match.group(1).strip()
        
        metadata.domain_specific['casino'] = casino_data
    
    def _calculate_article_structure_score(self, metadata: ExtractedMetadata) -> float:
        """Calculate article structure quality score."""
        factors = {
            'has_headings': metadata.heading_count > 0,
            'adequate_paragraphs': metadata.paragraph_count >= 3,
            'good_length': 300 <= metadata.word_count <= 3000,
            'has_links': metadata.link_count > 0
        }
        return sum(factors.values()) / len(factors)
    
    def _calculate_review_completeness(self, content: str, rating: Optional[float], 
                                     has_pros: bool, has_cons: bool) -> float:
        """Calculate review completeness score."""
        factors = {
            'has_rating': rating is not None,
            'has_pros': has_pros,
            'has_cons': has_cons,
            'has_verdict': bool(re.search(r'\b(?:verdict|conclusion|recommendation)\b', content, re.IGNORECASE)),
            'adequate_length': len(content.split()) >= 200
        }
        return sum(factors.values()) / len(factors)
    
    def _calculate_extraction_confidence(self, metadata: ExtractedMetadata) -> float:
        """Calculate confidence score for the extraction process."""
        
        confidence_factors = {
            'has_basic_info': (metadata.title is not None) * 0.2,
            'language_detected': (metadata.language is not None) * 0.1,
            'has_structure': (metadata.paragraph_count > 0) * 0.15,
            'has_keywords': (len(metadata.keywords) > 0) * 0.15,
            'has_summary': (metadata.summary is not None) * 0.1,
            'quality_calculated': (metadata.content_quality_score is not None) * 0.1,
            'processing_successful': 0.2  # Base score for successful processing
        }
        
        return sum(confidence_factors.values())
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats."""
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%B %d %Y',
            '%b %d %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse date: {date_str}")
    
    def _generate_cache_key(self, content: str, source_url: Optional[str], 
                          content_type: Optional[ContentType], domain: Optional[str]) -> str:
        """Generate cache key for metadata extraction."""
        key_components = [
            hashlib.md5(content.encode()).hexdigest()[:16],
            source_url or 'no_url',
            content_type.value if content_type else 'unknown',
            domain or 'no_domain'
        ]
        return '_'.join(key_components)
    
    async def _cache_metadata(self, cache_key: str, metadata: ExtractedMetadata):
        """Cache extracted metadata."""
        try:
            # Create a simple response object for caching
            cache_response = EnhancedRAGResponse(
                content="metadata_cache",
                sources=[],
                confidence_score=metadata.extraction_confidence,
                metadata={'extracted_metadata': metadata}
            )
            
            await self.cache.set(cache_key, cache_response)
        except Exception as e:
            self.logger.debug(f"Caching failed: {e}")
    
    def _create_fallback_metadata(self, content: str, source_url: Optional[str], 
                                content_type: Optional[ContentType]) -> ExtractedMetadata:
        """Create fallback metadata when extraction fails."""
        return ExtractedMetadata(
            title="Unknown Title",
            word_count=len(content.split()) if content else 0,
            character_count=len(content) if content else 0,
            language='en',
            source_url=source_url,
            content_type=content_type,
            extraction_confidence=0.1,  # Low confidence for fallback
            content_hash=hashlib.md5(content.encode()).hexdigest() if content else None
        )
    
    # Additional domain-specific extractors
    async def _extract_gaming_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract gaming-specific metadata."""
        gaming_terms = {
            'game_types': ['slots', 'poker', 'blackjack', 'roulette', 'sports betting'],
            'features': ['multiplayer', 'single player', 'online', 'mobile', 'VR'],
            'platforms': ['PC', 'console', 'mobile', 'web', 'iOS', 'Android']
        }
        
        content_lower = content.lower()
        gaming_data = {}
        
        for category, terms in gaming_terms.items():
            found = [term for term in terms if term in content_lower]
            gaming_data[category] = found
        
        metadata.domain_specific['gaming'] = gaming_data
    
    async def _extract_technical_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract technical document metadata."""
        tech_indicators = {
            'code_blocks': len(re.findall(r'```[\s\S]*?```|`[^`]+`', content)),
            'api_references': len(re.findall(r'\bAPI\b|\bendpoint\b|\bREST\b', content, re.IGNORECASE)),
            'version_numbers': len(re.findall(r'v?\d+\.\d+(?:\.\d+)?', content)),
            'technical_terms': len(re.findall(r'\b(?:function|method|class|variable|parameter)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['technical'] = tech_indicators
    
    async def _extract_news_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract news-specific metadata."""
        news_indicators = {
            'has_dateline': bool(re.search(r'^[A-Z][A-Z\s]+ - ', content, re.MULTILINE)),
            'has_quotes': len(re.findall(r'"[^"]{20,}"', content)),
            'breaking_news': bool(re.search(r'\bbreaking\b|\burgent\b', content, re.IGNORECASE)),
            'news_agencies': len(re.findall(r'\b(?:Reuters|AP|AFP|Bloomberg|CNN|BBC)\b', content))
        }
        
        metadata.domain_specific['news'] = news_indicators
    
    async def _extract_blog_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract blog-specific metadata."""
        blog_indicators = {
            'personal_pronouns': len(re.findall(r'\b(?:I|my|me|we|our|us)\b', content, re.IGNORECASE)),
            'call_to_action': bool(re.search(r'\b(?:subscribe|follow|share|comment|like)\b', content, re.IGNORECASE)),
            'social_links': len(re.findall(r'(?:twitter|facebook|instagram|linkedin)\.com', content, re.IGNORECASE)),
            'tags_mentioned': len(re.findall(r'#\w+', content))
        }
        
        metadata.domain_specific['blog'] = blog_indicators
    
    async def _extract_academic_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract academic document metadata."""
        academic_indicators = {
            'citations': len(re.findall(r'\[\d+\]|\([^)]*\d{4}[^)]*\)', content)),
            'references_section': bool(re.search(r'\b(?:references|bibliography)\b', content, re.IGNORECASE)),
            'abstract': bool(re.search(r'\babstract\b', content, re.IGNORECASE)),
            'methodology': bool(re.search(r'\b(?:methodology|methods?)\b', content, re.IGNORECASE)),
            'results': bool(re.search(r'\b(?:results?|findings?)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['academic'] = academic_indicators
    
    async def _extract_legal_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract legal document metadata."""
        legal_indicators = {
            'legal_terms': len(re.findall(r'\b(?:whereas|therefore|pursuant|herein|thereof)\b', content, re.IGNORECASE)),
            'sections': len(re.findall(r'Section \d+|§\d+', content)),
            'definitions': bool(re.search(r'\bdefinitions?\b', content, re.IGNORECASE)),
            'parties': len(re.findall(r'\b(?:plaintiff|defendant|party|parties)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['legal'] = legal_indicators
    
    async def _extract_marketing_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract marketing content metadata."""
        marketing_indicators = {
            'cta_count': len(re.findall(r'\b(?:buy now|sign up|get started|learn more|contact us)\b', content, re.IGNORECASE)),
            'benefits_mentioned': len(re.findall(r'\b(?:benefit|advantage|feature|solution)\b', content, re.IGNORECASE)),
            'urgency_words': len(re.findall(r'\b(?:limited|exclusive|now|today|hurry)\b', content, re.IGNORECASE)),
            'testimonials': bool(re.search(r'\b(?:testimonial|review|customer says)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['marketing'] = marketing_indicators
    
    async def _extract_finance_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract finance-specific metadata."""
        finance_indicators = {
            'financial_terms': len(re.findall(r'\b(?:investment|portfolio|dividend|stock|bond|fund)\b', content, re.IGNORECASE)),
            'currency_mentions': len(re.findall(r'\$[\d,]+|\b(?:USD|EUR|GBP|JPY)\b', content)),
            'percentages': len(re.findall(r'\d+(?:\.\d+)?%', content)),
            'financial_advice': bool(re.search(r'\b(?:invest|financial advice|portfolio)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['finance'] = finance_indicators
    
    async def _extract_health_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract health-specific metadata."""
        health_indicators = {
            'medical_terms': len(re.findall(r'\b(?:diagnosis|treatment|symptom|medication|therapy)\b', content, re.IGNORECASE)),
            'disclaimers': bool(re.search(r'\b(?:consult|doctor|medical advice|disclaimer)\b', content, re.IGNORECASE)),
            'conditions': len(re.findall(r'\b(?:diabetes|cancer|heart disease|depression|anxiety)\b', content, re.IGNORECASE)),
            'treatments': len(re.findall(r'\b(?:surgery|medication|therapy|treatment)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['health'] = health_indicators
    
    async def _extract_technology_metadata(self, content: str, metadata: ExtractedMetadata):
        """Extract technology-specific metadata."""
        tech_indicators = {
            'programming_languages': len(re.findall(r'\b(?:Python|JavaScript|Java|C\+\+|Ruby|PHP)\b', content)),
            'frameworks': len(re.findall(r'\b(?:React|Angular|Django|Flask|Spring|Laravel)\b', content)),
            'tech_companies': len(re.findall(r'\b(?:Google|Microsoft|Apple|Amazon|Facebook|Netflix)\b', content)),
            'tech_concepts': len(re.findall(r'\b(?:AI|machine learning|blockchain|cloud|API|database)\b', content, re.IGNORECASE))
        }
        
        metadata.domain_specific['technology'] = tech_indicators
    
    def get_extraction_summary(self, metadata: ExtractedMetadata) -> Dict[str, Any]:
        """Generate a human-readable summary of extracted metadata."""
        
        return {
            'basic_info': {
                'title': metadata.title,
                'author': metadata.author,
                'language': metadata.language,
                'word_count': metadata.word_count,
                'creation_date': metadata.creation_date.isoformat() if metadata.creation_date else None
            },
            'content_structure': {
                'paragraphs': metadata.paragraph_count,
                'sentences': metadata.sentence_count,
                'headings': metadata.heading_count,
                'lists': metadata.list_count,
                'links': metadata.link_count
            },
            'semantic_analysis': {
                'top_keywords': metadata.keywords[:10],
                'key_phrases': metadata.key_phrases[:5],
                'topics': metadata.topics,
                'sentiment': metadata.sentiment_score,
                'entities': dict(list(metadata.entities.items())[:3]) if metadata.entities else {}
            },
            'quality_metrics': {
                'readability_score': metadata.readability_score,
                'content_quality': metadata.content_quality_score,
                'completeness': metadata.completeness_score,
                'extraction_confidence': metadata.extraction_confidence
            },
            'domain_specific': metadata.domain_specific,
            'processing_info': {
                'processing_time_ms': metadata.processing_time_ms,
                'content_type': metadata.content_type.value if metadata.content_type else None,
                'extraction_timestamp': metadata.extraction_timestamp.isoformat()
            }
        }


# Global metadata extractor instance
metadata_extractor = MetadataExtractor()


# Utility functions
async def extract_metadata_simple(content: str, source_url: Optional[str] = None) -> ExtractedMetadata:
    """Simple function to extract metadata from content."""
    return await metadata_extractor.extract_metadata(content, source_url)


def create_metadata_extractor(config: Optional[Dict] = None) -> MetadataExtractor:
    """Factory function to create a configured MetadataExtractor."""
    return MetadataExtractor(config)


# Export main classes and functions
__all__ = [
    'MetadataExtractor',
    'ExtractedMetadata',
    'MetadataCategory',
    'metadata_extractor',
    'extract_metadata_simple',
    'create_metadata_extractor'
] 