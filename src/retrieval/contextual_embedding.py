"""
Task 3.1: Contextual Embedding System Implementation

This module implements the foundational contextual embedding system that enhances
document chunks with surrounding context before embedding generation.

Key features:
- ContextualChunk class with intelligent context aggregation
- Document structure awareness (titles, sections, hierarchy)
- Configurable context window extraction
- Enhanced metadata generation
- Integration with existing embedding pipelines
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, validator

# Set up logger
logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Strategies for context extraction and aggregation."""
    SURROUNDING = "surrounding"  # Include chunks before and after
    HIERARCHICAL = "hierarchical"  # Include section headers and document structure
    SEMANTIC = "semantic"  # Include semantically related chunks
    COMBINED = "combined"  # Combine multiple strategies


class ContentType(Enum):
    """Content type classification for context optimization."""
    ARTICLE = "article"
    REVIEW = "review"
    TUTORIAL = "tutorial"
    DOCUMENTATION = "documentation"
    NEWS = "news"
    BLOG = "blog"
    ACADEMIC = "academic"
    FORUM = "forum"
    FAQ = "faq"
    UNKNOWN = "unknown"


@dataclass
class RetrievalConfig:
    """Configuration for the contextual retrieval system."""
    
    # Context extraction settings
    context_window_size: int = 2  # Number of chunks before/after for context
    include_document_title: bool = True
    include_section_headers: bool = True
    include_breadcrumbs: bool = True
    context_strategy: ContextStrategy = ContextStrategy.COMBINED
    
    # Content processing settings
    max_context_length: int = 1000  # Maximum characters for context
    min_chunk_length: int = 50  # Minimum chunk length to process
    overlap_size: int = 50  # Overlap between chunks
    
    # Metadata enhancement settings
    extract_content_type: bool = True
    calculate_document_position: bool = True
    include_semantic_tags: bool = True
    
    # Performance settings
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Quality control
    filter_low_quality_chunks: bool = True
    quality_threshold: float = 0.3
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings and return any issues."""
        issues = []
        
        if self.context_window_size < 0:
            issues.append("context_window_size must be non-negative")
        
        if self.max_context_length < 100:
            issues.append("max_context_length should be at least 100 characters")
        
        if self.min_chunk_length <= 0:
            issues.append("min_chunk_length must be positive")
        
        if not (0.0 <= self.quality_threshold <= 1.0):
            issues.append("quality_threshold must be between 0.0 and 1.0")
        
        return issues


@dataclass
class ContextualChunk:
    """
    Enhanced chunk with contextual information for improved embedding quality.
    
    This class represents a document chunk enriched with surrounding context,
    document structure information, and enhanced metadata.
    """
    
    # Core content
    text: str
    context: str
    chunk_index: int
    document_id: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Document structure
    document_title: Optional[str] = None
    section_header: Optional[str] = None
    subsection_header: Optional[str] = None
    breadcrumbs: List[str] = field(default_factory=list)
    
    # Position and hierarchy
    document_position: float = 0.0  # 0.0 to 1.0 position in document
    total_chunks: int = 1
    hierarchy_level: int = 0
    
    # Content analysis
    content_type: ContentType = ContentType.UNKNOWN
    semantic_tags: List[str] = field(default_factory=list)
    quality_score: float = 0.5
    
    # Context metadata
    context_source: str = "surrounding"  # Where context came from
    context_chunks_used: List[int] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        """
        Generate the complete contextualized text for embedding.
        
        This property intelligently combines document metadata, structural
        information, context, and the main chunk text into a rich representation
        suitable for embedding generation.
        """
        parts = []
        
        # Add document title if available
        if self.document_title and self.document_title.strip():
            parts.append(f"Document: {self.document_title.strip()}")
        
        # Add section hierarchy
        if self.section_header and self.section_header.strip():
            parts.append(f"Section: {self.section_header.strip()}")
        
        if self.subsection_header and self.subsection_header.strip():
            parts.append(f"Subsection: {self.subsection_header.strip()}")
        
        # Add breadcrumbs for navigation context
        if self.breadcrumbs:
            breadcrumb_path = " > ".join(self.breadcrumbs)
            parts.append(f"Context Path: {breadcrumb_path}")
        
        # Add content type for domain awareness
        if self.content_type != ContentType.UNKNOWN:
            parts.append(f"Content Type: {self.content_type.value}")
        
        # Add semantic tags if available
        if self.semantic_tags:
            tags = ", ".join(self.semantic_tags[:5])  # Limit to top 5 tags
            parts.append(f"Topics: {tags}")
        
        # Add surrounding context
        if self.context and self.context.strip():
            parts.append(f"Context: {self.context.strip()}")
        
        # Add the main chunk content
        parts.append(f"Content: {self.text.strip()}")
        
        # Add position information for document structure awareness
        if self.total_chunks > 1:
            position_info = f"Position: {self.chunk_index + 1}/{self.total_chunks}"
            if self.document_position > 0:
                position_info += f" ({self.document_position:.1%} through document)"
            parts.append(position_info)
        
        return "\n".join(parts)
    
    @property
    def context_hash(self) -> str:
        """Generate a hash of the contextual content for caching."""
        content_to_hash = f"{self.text}|{self.context}|{self.document_title}|{self.section_header}"
        return hashlib.md5(content_to_hash.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ContextualChunk to dictionary for serialization."""
        return {
            "text": self.text,
            "context": self.context,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "document_title": self.document_title,
            "section_header": self.section_header,
            "subsection_header": self.subsection_header,
            "breadcrumbs": self.breadcrumbs,
            "document_position": self.document_position,
            "total_chunks": self.total_chunks,
            "hierarchy_level": self.hierarchy_level,
            "content_type": self.content_type.value,
            "semantic_tags": self.semantic_tags,
            "quality_score": self.quality_score,
            "context_source": self.context_source,
            "context_chunks_used": self.context_chunks_used,
            "full_text": self.full_text,
            "context_hash": self.context_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextualChunk":
        """Create ContextualChunk from dictionary."""
        # Handle enum conversion
        if isinstance(data.get("content_type"), str):
            try:
                data["content_type"] = ContentType(data["content_type"])
            except ValueError:
                data["content_type"] = ContentType.UNKNOWN
        
        # Remove computed properties
        data.pop("full_text", None)
        data.pop("context_hash", None)
        
        return cls(**data)
    
    def validate_quality(self) -> Tuple[bool, List[str]]:
        """Validate chunk quality and return issues if any."""
        issues = []
        
        if len(self.text.strip()) < 10:
            issues.append("Text content too short")
        
        if not self.text.strip():
            issues.append("Empty text content")
        
        if self.quality_score < 0.3:
            issues.append("Quality score below threshold")
        
        if self.chunk_index < 0:
            issues.append("Invalid chunk index")
        
        if not self.document_id:
            issues.append("Missing document ID")
        
        return len(issues) == 0, issues


class DocumentProcessor:
    """
    Document processing utilities for chunk extraction and structure analysis.
    """
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".DocumentProcessor")
    
    def extract_document_structure(self, document: Document) -> Dict[str, Any]:
        """
        Extract document structure including headers, sections, and hierarchy.
        """
        content = document.page_content
        metadata = document.metadata
        
        structure = {
            "title": self._extract_title(content, metadata),
            "sections": self._extract_sections(content),
            "headers": self._extract_headers(content),
            "content_type": self._classify_content_type(content, metadata),
            "breadcrumbs": self._extract_breadcrumbs(metadata),
            "semantic_tags": self._extract_semantic_tags(content)
        }
        
        return structure
    
    def _extract_title(self, content: str, metadata: Dict[str, Any]) -> str:
        """Extract document title from content or metadata."""
        # Try metadata first
        title = metadata.get("title", "")
        if title and title.strip():
            return title.strip()
        
        # Try to extract from content
        lines = content.split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and (line.startswith("#") or len(line.split()) <= 15):
                # Likely a title (markdown header or short line)
                return line.lstrip("# ").strip()
        
        return ""
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract section information from content."""
        sections = []
        lines = content.split("\n")
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for markdown headers
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                section_title = line.lstrip("# ").strip()
                
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "title": section_title,
                    "level": level,
                    "start_line": i,
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _extract_headers(self, content: str) -> List[str]:
        """Extract all headers from content."""
        headers = []
        
        # Markdown headers
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
            headers.append(match.group(2).strip())
        
        # HTML headers (basic)
        for match in re.finditer(r'<h[1-6][^>]*>([^<]+)</h[1-6]>', content, re.IGNORECASE):
            headers.append(match.group(1).strip())
        
        return headers
    
    def _classify_content_type(self, content: str, metadata: Dict[str, Any]) -> ContentType:
        """Classify content type based on content and metadata analysis."""
        # Check metadata first
        content_type = metadata.get("content_type", "").lower()
        if content_type:
            try:
                return ContentType(content_type)
            except ValueError:
                pass
        
        # Analyze content patterns
        content_lower = content.lower()
        
        # Academic patterns
        if any(term in content_lower for term in ["abstract", "methodology", "references", "bibliography"]):
            return ContentType.ACADEMIC
        
        # Tutorial patterns
        if any(term in content_lower for term in ["step 1", "tutorial", "how to", "guide", "instructions"]):
            return ContentType.TUTORIAL
        
        # Review patterns
        if any(term in content_lower for term in ["rating", "review", "pros and cons", "verdict"]):
            return ContentType.REVIEW
        
        # Documentation patterns
        if any(term in content_lower for term in ["api", "function", "parameter", "returns", "example"]):
            return ContentType.DOCUMENTATION
        
        # News patterns
        if any(term in content_lower for term in ["breaking", "reported", "according to", "spokesman"]):
            return ContentType.NEWS
        
        # FAQ patterns
        if any(term in content_lower for term in ["frequently asked", "q:", "a:", "question"]):
            return ContentType.FAQ
        
        return ContentType.UNKNOWN
    
    def _extract_breadcrumbs(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract breadcrumb navigation from metadata."""
        breadcrumbs = []
        
        # Check various metadata fields
        if "breadcrumbs" in metadata:
            if isinstance(metadata["breadcrumbs"], list):
                breadcrumbs = metadata["breadcrumbs"]
            elif isinstance(metadata["breadcrumbs"], str):
                breadcrumbs = metadata["breadcrumbs"].split(" > ")
        
        # Check for category hierarchy
        if "category" in metadata:
            breadcrumbs.append(metadata["category"])
        
        if "subcategory" in metadata:
            breadcrumbs.append(metadata["subcategory"])
        
        return [b.strip() for b in breadcrumbs if b.strip()]
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from content analysis."""
        tags = []
        content_lower = content.lower()
        
        # Domain-specific keyword mapping
        tag_keywords = {
            "casino": ["casino", "gambling", "bet", "poker", "slots"],
            "sports": ["sports", "football", "basketball", "soccer", "tennis"],
            "finance": ["finance", "money", "investment", "stock", "trading"],
            "technology": ["technology", "software", "programming", "computer"],
            "health": ["health", "medical", "wellness", "fitness", "nutrition"],
            "travel": ["travel", "vacation", "hotel", "flight", "destination"],
            "food": ["food", "recipe", "cooking", "restaurant", "cuisine"]
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        return tags


class ContextualEmbeddingSystem:
    """
    Main system for creating contextual embeddings with enhanced document understanding.
    
    This system processes documents by:
    1. Extracting document structure and metadata
    2. Creating contextual chunks with surrounding information
    3. Generating embeddings for the contextualized content
    4. Maintaining quality control and validation
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.document_processor = DocumentProcessor(self.config)
        self.logger = logging.getLogger(__name__ + ".ContextualEmbeddingSystem")
        
        # Validate configuration
        config_issues = self.config.validate_config()
        if config_issues:
            self.logger.warning(f"Configuration issues: {config_issues}")
    
    def create_contextual_chunks(
        self,
        document: Document,
        chunks: List[str],
        chunk_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[ContextualChunk]:
        """
        Create contextual chunks with enhanced metadata and surrounding context.
        
        Args:
            document: Source document
            chunks: List of text chunks
            chunk_metadata: Optional metadata for each chunk
            
        Returns:
            List of ContextualChunk objects with rich context
        """
        if not chunks:
            self.logger.warning("No chunks provided for contextual processing")
            return []
        
        # Extract document structure
        doc_structure = self.document_processor.extract_document_structure(document)
        
        # Initialize chunk metadata if not provided
        if chunk_metadata is None:
            chunk_metadata = [{"chunk_index": i} for i in range(len(chunks))]
        
        contextual_chunks = []
        
        for i, (chunk_text, metadata) in enumerate(zip(chunks, chunk_metadata)):
            # Skip low-quality chunks if filtering enabled
            if self.config.filter_low_quality_chunks:
                if len(chunk_text.strip()) < self.config.min_chunk_length:
                    self.logger.debug(f"Skipping short chunk {i}: {len(chunk_text)} chars")
                    continue
            
            # Extract context for this chunk
            context = self._extract_chunk_context(chunks, i, doc_structure)
            
            # Calculate document position
            doc_position = i / len(chunks) if len(chunks) > 1 else 0.0
            
            # Determine section header
            section_header = self._find_section_for_chunk(i, doc_structure)
            
            # Calculate quality score
            quality_score = self._calculate_chunk_quality(chunk_text, context, metadata)
            
            # Create contextual chunk
            contextual_chunk = ContextualChunk(
                text=chunk_text,
                context=context,
                chunk_index=i,
                document_id=document.metadata.get("id", str(hash(document.page_content))),
                metadata={
                    **metadata,
                    **document.metadata,
                    "original_chunk_index": i,
                    "processing_timestamp": datetime.now().isoformat()
                },
                document_title=doc_structure["title"],
                section_header=section_header,
                breadcrumbs=doc_structure["breadcrumbs"],
                document_position=doc_position,
                total_chunks=len(chunks),
                content_type=doc_structure["content_type"],
                semantic_tags=doc_structure["semantic_tags"],
                quality_score=quality_score,
                context_source=self.config.context_strategy.value,
                context_chunks_used=self._get_context_chunk_indices(i, len(chunks))
            )
            
            # Validate chunk quality
            is_valid, validation_issues = contextual_chunk.validate_quality()
            if not is_valid:
                self.logger.warning(f"Chunk {i} validation issues: {validation_issues}")
                if self.config.filter_low_quality_chunks:
                    continue
            
            contextual_chunks.append(contextual_chunk)
        
        self.logger.info(f"Created {len(contextual_chunks)} contextual chunks from {len(chunks)} original chunks")
        return contextual_chunks
    
    def _extract_chunk_context(
        self,
        chunks: List[str],
        chunk_index: int,
        doc_structure: Dict[str, Any]
    ) -> str:
        """Extract surrounding context for a specific chunk."""
        context_parts = []
        
        if self.config.context_strategy in [ContextStrategy.SURROUNDING, ContextStrategy.COMBINED]:
            # Add surrounding chunks
            start_idx = max(0, chunk_index - self.config.context_window_size)
            end_idx = min(len(chunks), chunk_index + self.config.context_window_size + 1)
            
            for i in range(start_idx, end_idx):
                if i != chunk_index:  # Don't include the chunk itself
                    context_parts.append(chunks[i])
        
        if self.config.context_strategy in [ContextStrategy.HIERARCHICAL, ContextStrategy.COMBINED]:
            # Add hierarchical context (section headers, etc.)
            if doc_structure["sections"]:
                relevant_section = self._find_relevant_section(chunk_index, doc_structure["sections"])
                if relevant_section:
                    context_parts.append(f"Section context: {relevant_section['title']}")
        
        # Combine and truncate context
        context = " ".join(context_parts)
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length] + "..."
        
        return context
    
    def _find_section_for_chunk(self, chunk_index: int, doc_structure: Dict[str, Any]) -> Optional[str]:
        """Find the most relevant section header for a chunk."""
        sections = doc_structure.get("sections", [])
        if not sections:
            return None
        
        # Simple heuristic: find the section that would contain this chunk
        # based on document position
        total_chunks = chunk_index + 1  # Approximate
        
        for section in sections:
            # This is a simplified approach - in practice, you'd want
            # more sophisticated mapping between chunks and sections
            if section.get("title"):
                return section["title"]
        
        return None
    
    def _find_relevant_section(self, chunk_index: int, sections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the section most relevant to the given chunk."""
        if not sections:
            return None
        
        # Simple approach: return first section with content
        # In practice, you'd map chunk positions to section boundaries
        for section in sections:
            if section.get("content"):
                return section
        
        return sections[0] if sections else None
    
    def _calculate_chunk_quality(self, chunk_text: str, context: str, metadata: Dict[str, Any]) -> float:
        """Calculate quality score for a chunk."""
        score = 0.5  # Base score
        
        # Length-based quality
        text_length = len(chunk_text.strip())
        if text_length > 100:
            score += 0.2
        elif text_length < 50:
            score -= 0.2
        
        # Context availability
        if context and len(context) > 50:
            score += 0.1
        
        # Metadata richness
        if len(metadata) > 3:
            score += 0.1
        
        # Content quality indicators
        if any(indicator in chunk_text.lower() for indicator in ["however", "therefore", "specifically", "importantly"]):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_context_chunk_indices(self, chunk_index: int, total_chunks: int) -> List[int]:
        """Get indices of chunks used for context."""
        start_idx = max(0, chunk_index - self.config.context_window_size)
        end_idx = min(total_chunks, chunk_index + self.config.context_window_size + 1)
        
        return [i for i in range(start_idx, end_idx) if i != chunk_index]
    
    async def embed_contextual_chunks(
        self,
        chunks: List[ContextualChunk],
        embeddings: Embeddings
    ) -> List[Tuple[ContextualChunk, List[float]]]:
        """
        Generate embeddings for contextual chunks using their full_text property.
        
        Args:
            chunks: List of ContextualChunk objects
            embeddings: Embedding model to use
            
        Returns:
            List of tuples containing (chunk, embedding_vector)
        """
        if not chunks:
            return []
        
        # Get full contextualized texts
        texts_to_embed = [chunk.full_text for chunk in chunks]
        
        self.logger.info(f"Generating embeddings for {len(texts_to_embed)} contextual chunks")
        
        try:
            # Generate embeddings (async if supported)
            if hasattr(embeddings, 'aembed_documents'):
                chunk_embeddings = await embeddings.aembed_documents(texts_to_embed)
            else:
                chunk_embeddings = embeddings.embed_documents(texts_to_embed)
            
            # Combine chunks with their embeddings
            result = list(zip(chunks, chunk_embeddings))
            
            self.logger.info(f"Successfully generated {len(result)} contextual embeddings")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating contextual embeddings: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and configuration info."""
        return {
            "config": {
                "context_window_size": self.config.context_window_size,
                "context_strategy": self.config.context_strategy.value,
                "max_context_length": self.config.max_context_length,
                "include_document_title": self.config.include_document_title,
                "include_section_headers": self.config.include_section_headers
            },
            "system_info": {
                "version": "1.0.0",
                "component": "ContextualEmbeddingSystem",
                "task": "3.1"
            }
        }


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    config = RetrievalConfig(
        context_window_size=2,
        include_document_title=True,
        include_section_headers=True
    )
    
    system = ContextualEmbeddingSystem(config)
    
    # Example document
    doc = Document(
        page_content="This is a sample document with multiple sections...",
        metadata={"title": "Sample Document", "source": "test.pdf"}
    )
    
    chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
    
    contextual_chunks = system.create_contextual_chunks(doc, chunks)
    
    for chunk in contextual_chunks:
        print(f"Chunk {chunk.chunk_index}:")
        print(f"  Original: {chunk.text[:50]}...")
        print(f"  Contextual: {chunk.full_text[:100]}...")
        print(f"  Quality: {chunk.quality_score:.2f}")
        print() 