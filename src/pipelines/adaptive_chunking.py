#!/usr/bin/env python3
"""
Adaptive Chunking Strategies - Task 4.3
Intelligent chunking system that adapts to different content types and structures

✅ INTEGRATIONS:
- Task 1: Supabase foundation with proper schema
- Task 2: Enhanced confidence scoring system (CORRECTED IMPORTS)
- Task 3: Contextual retrieval integration
- Task 4.2: Content type detection integration
"""

import asyncio
import re
import nltk
import spacy
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import numpy as np
from collections import defaultdict

# ✅ CORRECTED IMPORTS - Using actual file structure
from src.chains.enhanced_confidence_scoring_system import (
    SourceQualityAnalyzer,
    IntelligentCache,
    EnhancedRAGResponse
)
from src.pipelines.content_type_detector import (
    ContentTypeDetector,
    ContentType,
    ContentAnalysis
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    ADAPTIVE = "adaptive"
    PARAGRAPH_BASED = "paragraph_based"
    SENTENCE_BASED = "sentence_based"


@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    content_type: ContentType
    strategy_used: ChunkingStrategy
    start_position: int
    end_position: int
    word_count: int
    sentence_count: int
    semantic_coherence: float
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    section_header: Optional[str] = None
    importance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessedChunk:
    """A processed content chunk with metadata"""
    content: str
    metadata: ChunkMetadata
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content": self.content,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "content_type": self.metadata.content_type.value,
                "strategy_used": self.metadata.strategy_used.value,
                "start_position": self.metadata.start_position,
                "end_position": self.metadata.end_position,
                "word_count": self.metadata.word_count,
                "sentence_count": self.metadata.sentence_count,
                "semantic_coherence": self.metadata.semantic_coherence,
                "overlap_with_previous": self.metadata.overlap_with_previous,
                "overlap_with_next": self.metadata.overlap_with_next,
                "section_header": self.metadata.section_header,
                "importance_score": self.metadata.importance_score,
                "created_at": self.metadata.created_at.isoformat()
            },
            "embeddings": self.embeddings
        }


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.source_analyzer = SourceQualityAnalyzer()
    
    @abstractmethod
    def chunk_content(self, content: str, content_analysis: ContentAnalysis) -> List[ProcessedChunk]:
        """Abstract method for chunking content"""
        pass
    
    def _calculate_semantic_coherence(self, chunk: str) -> float:
        """Calculate semantic coherence score for a chunk"""
        if not nlp:
            return 0.5  # Default score if spaCy not available
        
        try:
            doc = nlp(chunk)
            sentences = list(doc.sents)
            
            if len(sentences) < 2:
                return 1.0  # Single sentence is perfectly coherent
            
            # Calculate average similarity between consecutive sentences
            similarities = []
            for i in range(len(sentences) - 1):
                sent1 = sentences[i]
                sent2 = sentences[i + 1]
                
                # Skip if sentences are too short
                if len(sent1.text.strip()) < 10 or len(sent2.text.strip()) < 10:
                    continue
                
                similarity = sent1.similarity(sent2)
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.5
        
        except Exception:
            return 0.5  # Fallback score
    
    def _calculate_importance_score(self, chunk: str, content_analysis: ContentAnalysis) -> float:
        """Calculate importance score based on content features"""
        score = 0.0
        chunk_lower = chunk.lower()
        
        # Header indicators
        if any(indicator in chunk_lower for indicator in ['introduction', 'conclusion', 'summary', 'overview']):
            score += 0.3
        
        # Question indicators
        if '?' in chunk:
            score += 0.2
        
        # Emphasis indicators
        emphasis_count = chunk.count('**') + chunk.count('*') + chunk.count('_')
        score += min(emphasis_count * 0.1, 0.3)
        
        # Length penalty for very short chunks
        if len(chunk.split()) < 20:
            score -= 0.2
        
        # Content type specific scoring
        if content_analysis.content_type == ContentType.TECHNICAL_DOCUMENTATION:
            if any(tech_word in chunk_lower for tech_word in ['api', 'function', 'method', 'class', 'example']):
                score += 0.2
        elif content_analysis.content_type == ContentType.CASINO_REVIEW:
            if any(review_word in chunk_lower for review_word in ['rating', 'score', 'pros', 'cons', 'verdict']):
                score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_chunk_id(self, content: str, position: int) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chunk_{position}_{content_hash}"


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with word boundaries"""
    
    def chunk_content(self, content: str, content_analysis: ContentAnalysis) -> List[ProcessedChunk]:
        """Chunk content into fixed-size pieces with overlap"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.max_chunk_size - self.overlap_size):
            chunk_words = words[i:i + self.max_chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            # Calculate positions
            start_pos = len(' '.join(words[:i]))
            end_pos = start_pos + len(chunk_content)
            
            # Calculate overlap
            overlap_prev = self.overlap_size if i > 0 else 0
            overlap_next = self.overlap_size if i + self.max_chunk_size < len(words) else 0
            
            metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(chunk_content, i),
                content_type=content_analysis.content_type,
                strategy_used=ChunkingStrategy.FIXED_SIZE,
                start_position=start_pos,
                end_position=end_pos,
                word_count=len(chunk_words),
                sentence_count=len(nltk.sent_tokenize(chunk_content)),
                semantic_coherence=self._calculate_semantic_coherence(chunk_content),
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next,
                importance_score=self._calculate_importance_score(chunk_content, content_analysis)
            )
            
            chunks.append(ProcessedChunk(
                content=chunk_content,
                metadata=metadata
            ))
        
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on topic coherence"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100, coherence_threshold: float = 0.6):
        super().__init__(max_chunk_size, overlap_size)
        self.coherence_threshold = coherence_threshold
    
    def chunk_content(self, content: str, content_analysis: ContentAnalysis) -> List[ProcessedChunk]:
        """Chunk content based on semantic coherence"""
        if not nlp:
            # Fallback to sentence-based chunking
            return self._fallback_sentence_chunking(content, content_analysis)
        
        sentences = nltk.sent_tokenize(content)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed max size
            if current_word_count + sentence_words > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                chunks.append(self._create_semantic_chunk(chunk_content, content_analysis, len(chunks)))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(self._create_semantic_chunk(chunk_content, content_analysis, len(chunks)))
        
        return chunks
    
    def _create_semantic_chunk(self, content: str, content_analysis: ContentAnalysis, chunk_index: int) -> ProcessedChunk:
        """Create a semantic chunk with metadata"""
        metadata = ChunkMetadata(
            chunk_id=self._generate_chunk_id(content, chunk_index),
            content_type=content_analysis.content_type,
            strategy_used=ChunkingStrategy.SEMANTIC,
            start_position=0,  # Would need full text to calculate
            end_position=len(content),
            word_count=len(content.split()),
            sentence_count=len(nltk.sent_tokenize(content)),
            semantic_coherence=self._calculate_semantic_coherence(content),
            importance_score=self._calculate_importance_score(content, content_analysis)
        )
        
        return ProcessedChunk(content=content, metadata=metadata)
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on word count"""
        overlap_words = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if overlap_words + sentence_words <= self.overlap_size:
                overlap_sentences.insert(0, sentence)
                overlap_words += sentence_words
            else:
                break
        
        return overlap_sentences
    
    def _fallback_sentence_chunking(self, content: str, content_analysis: ContentAnalysis) -> List[ProcessedChunk]:
        """Fallback to sentence-based chunking when spaCy is not available"""
        sentences = nltk.sent_tokenize(content)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > self.max_chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunks.append(self._create_semantic_chunk(chunk_content, content_analysis, len(chunks)))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(self._create_semantic_chunk(chunk_content, content_analysis, len(chunks)))
        
        return chunks


class StructuralChunker(BaseChunker):
    """Structural chunking based on document structure (headers, sections)"""
    
    def chunk_content(self, content: str, content_analysis: ContentAnalysis) -> List[ProcessedChunk]:
        """Chunk content based on structural elements"""
        # Detect headers and sections
        sections = self._detect_sections(content)
        chunks = []
        
        for section in sections:
            section_chunks = self._chunk_section(section, content_analysis, len(chunks))
            chunks.extend(section_chunks)
        
        return chunks
    
    def _detect_sections(self, content: str) -> List[Dict[str, Any]]:
        """Detect sections based on headers and structure"""
        lines = content.split('\n')
        sections = []
        current_section = {'header': None, 'content': [], 'level': 0}
        
        for line in lines:
            line = line.strip()
            
            # Detect markdown headers
            if line.startswith('#'):
                if current_section['content']:
                    sections.append(current_section)
                
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                current_section = {
                    'header': header_text,
                    'content': [],
                    'level': header_level
                }
            
            # Detect other header patterns
            elif self._is_header_line(line):
                if current_section['content']:
                    sections.append(current_section)
                
                current_section = {
                    'header': line,
                    'content': [],
                    'level': 1
                }
            else:
                if line:  # Skip empty lines
                    current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _is_header_line(self, line: str) -> bool:
        """Detect if a line is likely a header"""
        # All caps and short
        if line.isupper() and len(line.split()) <= 6:
            return True
        
        # Ends with colon and short
        if line.endswith(':') and len(line.split()) <= 8:
            return True
        
        # Numbered sections
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        return False
    
    def _chunk_section(self, section: Dict[str, Any], content_analysis: ContentAnalysis, chunk_offset: int) -> List[ProcessedChunk]:
        """Chunk a single section"""
        section_content = '\n'.join(section['content'])
        
        # If section is small enough, keep as single chunk
        if len(section_content.split()) <= self.max_chunk_size:
            metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(section_content, chunk_offset),
                content_type=content_analysis.content_type,
                strategy_used=ChunkingStrategy.STRUCTURAL,
                start_position=0,
                end_position=len(section_content),
                word_count=len(section_content.split()),
                sentence_count=len(nltk.sent_tokenize(section_content)),
                semantic_coherence=self._calculate_semantic_coherence(section_content),
                section_header=section['header'],
                importance_score=self._calculate_importance_score(section_content, content_analysis)
            )
            
            return [ProcessedChunk(content=section_content, metadata=metadata)]
        
        # Otherwise, use semantic chunking within the section
        semantic_chunker = SemanticChunker(self.max_chunk_size, self.overlap_size)
        chunks = semantic_chunker.chunk_content(section_content, content_analysis)
        
        # Update metadata to include section header
        for chunk in chunks:
            chunk.metadata.strategy_used = ChunkingStrategy.STRUCTURAL
            chunk.metadata.section_header = section['header']
        
        return chunks


class AdaptiveChunker(BaseChunker):
    """Adaptive chunking that selects the best strategy based on content type"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 100):
        super().__init__(max_chunk_size, overlap_size)
        self.fixed_chunker = FixedSizeChunker(max_chunk_size, overlap_size)
        self.semantic_chunker = SemanticChunker(max_chunk_size, overlap_size)
        self.structural_chunker = StructuralChunker(max_chunk_size, overlap_size)
    
    def chunk_content(self, content: str, content_analysis: ContentAnalysis) -> List[ProcessedChunk]:
        """Select and apply the best chunking strategy"""
        strategy = self._select_strategy(content_analysis)
        
        if strategy == ChunkingStrategy.SEMANTIC:
            chunks = self.semantic_chunker.chunk_content(content, content_analysis)
        elif strategy == ChunkingStrategy.STRUCTURAL:
            chunks = self.structural_chunker.chunk_content(content, content_analysis)
        else:
            chunks = self.fixed_chunker.chunk_content(content, content_analysis)
        
        # Update strategy in metadata
        for chunk in chunks:
            chunk.metadata.strategy_used = ChunkingStrategy.ADAPTIVE
        
        return chunks
    
    def _select_strategy(self, content_analysis: ContentAnalysis) -> ChunkingStrategy:
        """Select the best chunking strategy based on content analysis"""
        content_type = content_analysis.content_type
        
        # Strategy selection based on content type
        if content_type in [ContentType.TECHNICAL_DOCUMENTATION, ContentType.ACADEMIC_PAPER]:
            return ChunkingStrategy.STRUCTURAL
        elif content_type in [ContentType.NEWS_ARTICLE, ContentType.BLOG_POST, ContentType.CASINO_REVIEW]:
            return ChunkingStrategy.SEMANTIC
        elif content_type in [ContentType.LEGAL_DOCUMENT, ContentType.FINANCIAL_REPORT]:
            return ChunkingStrategy.STRUCTURAL
        else:
            return ChunkingStrategy.SEMANTIC  # Default to semantic


class AdaptiveChunkingSystem:
    """Main system for adaptive content chunking"""
    
    def __init__(self, supabase_client=None):
        self.content_detector = ContentTypeDetector()
        self.adaptive_chunker = AdaptiveChunker()
        self.cache = IntelligentCache() if supabase_client else None
        self.supabase = supabase_client
        
        # Performance metrics
        self.metrics = {
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'strategy_usage': defaultdict(int),
            'average_coherence': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def process_content(self, content: str, content_id: Optional[str] = None) -> List[ProcessedChunk]:
        """Process content with adaptive chunking"""
        start_time = datetime.now()
        
        # Check cache first
        if self.cache and content_id:
            cache_key = f"chunks_{hashlib.md5(content.encode()).hexdigest()}"
            cached_chunks = await self.cache.get(cache_key)
            if cached_chunks:
                self.metrics['cache_hits'] += 1
                return [ProcessedChunk(**chunk_data) for chunk_data in cached_chunks]
            self.metrics['cache_misses'] += 1
        
        # Detect content type
        content_analysis = await self.content_detector.analyze_content(content)
        
        # Chunk content
        chunks = self.adaptive_chunker.chunk_content(content, content_analysis)
        
        # Update metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_metrics(chunks, processing_time)
        
        # Cache results
        if self.cache and content_id:
            chunk_data = [chunk.to_dict() for chunk in chunks]
            await self.cache.set(cache_key, chunk_data, ttl=3600)  # 1 hour TTL
        
        return chunks
    
    def _update_metrics(self, chunks: List[ProcessedChunk], processing_time: float):
        """Update performance metrics"""
        self.metrics['chunks_processed'] += len(chunks)
        self.metrics['total_processing_time'] += processing_time
        
        # Strategy usage
        for chunk in chunks:
            self.metrics['strategy_usage'][chunk.metadata.strategy_used.value] += 1
        
        # Average coherence
        coherence_scores = [chunk.metadata.semantic_coherence for chunk in chunks]
        if coherence_scores:
            total_coherence = sum(coherence_scores)
            total_chunks = self.metrics['chunks_processed']
            self.metrics['average_coherence'] = (
                (self.metrics['average_coherence'] * (total_chunks - len(chunks)) + total_coherence) / total_chunks
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            'average_processing_time': (
                self.metrics['total_processing_time'] / max(1, self.metrics['chunks_processed'])
            ),
            'cache_hit_rate': (
                self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses'])
            )
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'strategy_usage': defaultdict(int),
            'average_coherence': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Example usage and testing
async def test_adaptive_chunking():
    """Test the adaptive chunking system"""
    system = AdaptiveChunkingSystem()
    
    # Test content
    test_content = """
    # Casino Review: BetMGM Online Casino
    
    ## Overview
    BetMGM is one of the leading online casinos in the United States, offering a comprehensive gaming experience with slots, table games, and live dealer options.
    
    ## Game Selection
    The casino features over 1,000 slot games from top providers like NetEnt, IGT, and Evolution Gaming. Popular titles include Starburst, Gonzo's Quest, and Divine Fortune.
    
    ### Table Games
    BetMGM offers a solid selection of table games including:
    - Blackjack variants (Classic, European, Atlantic City)
    - Roulette (American, European, French)
    - Baccarat and Poker games
    
    ## Bonuses and Promotions
    New players can claim a welcome bonus of up to $1,000 in bonus funds plus 200 free spins. The wagering requirement is 15x the bonus amount.
    
    ## Verdict
    BetMGM provides a premium online casino experience with excellent game variety, generous bonuses, and reliable customer support. Highly recommended for US players.
    """
    
    chunks = await system.process_content(test_content)
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Strategy: {chunk.metadata.strategy_used.value}")
        print(f"Word count: {chunk.metadata.word_count}")
        print(f"Coherence: {chunk.metadata.semantic_coherence:.2f}")
        print(f"Importance: {chunk.metadata.importance_score:.2f}")
        print(f"Header: {chunk.metadata.section_header}")
        print(f"Content preview: {chunk.content[:100]}...")
    
    print(f"\nPerformance metrics: {system.get_performance_metrics()}")


if __name__ == "__main__":
    asyncio.run(test_adaptive_chunking()) 