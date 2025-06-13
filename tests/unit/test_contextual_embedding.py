"""
Unit tests for Task 3.1: Contextual Embedding System

This test suite validates:
- ContextualChunk creation and properties
- RetrievalConfig validation
- DocumentProcessor structure extraction
- ContextualEmbeddingSystem functionality
- Quality control and validation
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import our implementation
from src.retrieval.contextual_embedding import (
    ContextualChunk,
    ContextualEmbeddingSystem,
    RetrievalConfig,
    DocumentProcessor,
    ContextStrategy,
    ContentType
)


class TestRetrievalConfig:
    """Test RetrievalConfig validation and settings."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RetrievalConfig()
        
        assert config.context_window_size == 2
        assert config.include_document_title is True
        assert config.include_section_headers is True
        assert config.context_strategy == ContextStrategy.COMBINED
        assert config.max_context_length == 1000
        assert config.min_chunk_length == 50
        assert config.quality_threshold == 0.3
    
    def test_config_validation_valid(self):
        """Test valid configuration passes validation."""
        config = RetrievalConfig(
            context_window_size=3,
            max_context_length=1500,
            min_chunk_length=100,
            quality_threshold=0.5
        )
        
        issues = config.validate_config()
        assert len(issues) == 0
    
    def test_config_validation_invalid(self):
        """Test invalid configuration raises validation issues."""
        config = RetrievalConfig(
            context_window_size=-1,
            max_context_length=50,
            min_chunk_length=0,
            quality_threshold=1.5
        )
        
        issues = config.validate_config()
        assert len(issues) > 0
        assert any("context_window_size must be non-negative" in issue for issue in issues)
        assert any("max_context_length should be at least 100" in issue for issue in issues)
        assert any("min_chunk_length must be positive" in issue for issue in issues)
        assert any("quality_threshold must be between 0.0 and 1.0" in issue for issue in issues)


class TestContextualChunk:
    """Test ContextualChunk creation and properties."""
    
    def test_basic_chunk_creation(self):
        """Test basic ContextualChunk creation."""
        chunk = ContextualChunk(
            text="This is a test chunk.",
            context="This is context.",
            chunk_index=0,
            document_id="test_doc_1"
        )
        
        assert chunk.text == "This is a test chunk."
        assert chunk.context == "This is context."
        assert chunk.chunk_index == 0
        assert chunk.document_id == "test_doc_1"
        assert chunk.quality_score == 0.5  # default
    
    def test_full_text_property_basic(self):
        """Test full_text property with basic content."""
        chunk = ContextualChunk(
            text="Main content here.",
            context="Some context.",
            chunk_index=0,
            document_id="test_doc"
        )
        
        full_text = chunk.full_text
        assert "Content: Main content here." in full_text
        assert "Context: Some context." in full_text
    
    def test_full_text_property_with_metadata(self):
        """Test full_text property with rich metadata."""
        chunk = ContextualChunk(
            text="Main content here.",
            context="Some context.",
            chunk_index=0,
            document_id="test_doc",
            document_title="Test Document",
            section_header="Introduction",
            breadcrumbs=["Home", "Docs", "Tutorial"],
            content_type=ContentType.TUTORIAL,
            semantic_tags=["programming", "tutorial"],
            total_chunks=5,
            document_position=0.2
        )
        
        full_text = chunk.full_text
        assert "Document: Test Document" in full_text
        assert "Section: Introduction" in full_text
        assert "Context Path: Home > Docs > Tutorial" in full_text
        assert "Content Type: tutorial" in full_text
        assert "Topics: programming, tutorial" in full_text
        assert "Position: 1/5 (20.0% through document)" in full_text
        assert "Content: Main content here." in full_text
    
    def test_context_hash(self):
        """Test context hash generation."""
        chunk = ContextualChunk(
            text="Test content",
            context="Test context",
            chunk_index=0,
            document_id="test_doc",
            document_title="Test Title",
            section_header="Test Section"
        )
        
        hash1 = chunk.context_hash
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length
        
        # Same content should produce same hash
        chunk2 = ContextualChunk(
            text="Test content",
            context="Test context", 
            chunk_index=0,
            document_id="test_doc",
            document_title="Test Title",
            section_header="Test Section"
        )
        
        assert chunk2.context_hash == hash1
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original_chunk = ContextualChunk(
            text="Test content",
            context="Test context",
            chunk_index=1,
            document_id="test_doc",
            document_title="Test Title",
            content_type=ContentType.REVIEW,
            semantic_tags=["test", "review"],
            quality_score=0.8
        )
        
        # Convert to dict
        chunk_dict = original_chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["text"] == "Test content"
        assert chunk_dict["content_type"] == "review"
        
        # Convert back from dict
        restored_chunk = ContextualChunk.from_dict(chunk_dict)
        assert restored_chunk.text == original_chunk.text
        assert restored_chunk.context == original_chunk.context
        assert restored_chunk.content_type == original_chunk.content_type
        assert restored_chunk.semantic_tags == original_chunk.semantic_tags
    
    def test_validate_quality_valid(self):
        """Test quality validation for valid chunk."""
        chunk = ContextualChunk(
            text="This is a sufficiently long test chunk with good content.",
            context="Good context",
            chunk_index=0,
            document_id="test_doc",
            quality_score=0.8
        )
        
        is_valid, issues = chunk.validate_quality()
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_quality_invalid(self):
        """Test quality validation for invalid chunk."""
        chunk = ContextualChunk(
            text="",  # Empty text
            context="",
            chunk_index=-1,  # Invalid index
            document_id="",  # Empty ID
            quality_score=0.1  # Low quality
        )
        
        is_valid, issues = chunk.validate_quality()
        assert is_valid is False
        assert len(issues) > 0


class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""
    
    def test_extract_title_from_metadata(self):
        """Test title extraction from metadata."""
        config = RetrievalConfig()
        processor = DocumentProcessor(config)
        
        content = "Some content without clear title"
        metadata = {"title": "Document Title from Metadata"}
        
        title = processor._extract_title(content, metadata)
        assert title == "Document Title from Metadata"
    
    def test_extract_title_from_content(self):
        """Test title extraction from content."""
        config = RetrievalConfig()
        processor = DocumentProcessor(config)
        
        content = "# Main Title\nThis is the content of the document..."
        metadata = {}
        
        title = processor._extract_title(content, metadata)
        assert title == "Main Title"
    
    def test_extract_sections(self):
        """Test section extraction from markdown content."""
        config = RetrievalConfig()
        processor = DocumentProcessor(config)
        
        content = """# Main Title
This is intro content.

## Section 1
Content for section 1.

### Subsection 1.1
More detailed content.

## Section 2
Content for section 2."""
        
        sections = processor._extract_sections(content)
        assert len(sections) >= 3
        
        # Check that we extracted the sections correctly
        section_titles = [s["title"] for s in sections]
        assert "Section 1" in section_titles
        assert "Subsection 1.1" in section_titles
        assert "Section 2" in section_titles
    
    def test_classify_content_type(self):
        """Test content type classification."""
        config = RetrievalConfig()
        processor = DocumentProcessor(config)
        
        # Test tutorial content
        tutorial_content = "# How to Build a Web App\nStep 1: Install the requirements..."
        content_type = processor._classify_content_type(tutorial_content, {})
        assert content_type == ContentType.TUTORIAL
        
        # Test review content
        review_content = "This product has a rating of 4.5 stars. Pros and cons are..."
        content_type = processor._classify_content_type(review_content, {})
        assert content_type == ContentType.REVIEW
        
        # Test academic content
        academic_content = "Abstract: This paper presents a methodology for..."
        content_type = processor._classify_content_type(academic_content, {})
        assert content_type == ContentType.ACADEMIC
    
    def test_extract_semantic_tags(self):
        """Test semantic tag extraction."""
        config = RetrievalConfig()
        processor = DocumentProcessor(config)
        
        content = "This guide covers casino games and poker strategies for gambling enthusiasts."
        tags = processor._extract_semantic_tags(content)
        
        assert "casino" in tags
    
    def test_extract_document_structure(self):
        """Test complete document structure extraction."""
        config = RetrievalConfig()
        processor = DocumentProcessor(config)
        
        document = Document(
            page_content="""# Casino Guide
            
## Poker Strategies
Content about poker...

## Slot Machine Tips  
Content about slots...""",
            metadata={"title": "Ultimate Casino Guide", "category": "gambling"}
        )
        
        structure = processor.extract_document_structure(document)
        
        assert structure["title"] == "Ultimate Casino Guide"
        assert len(structure["sections"]) >= 2
        assert "casino" in structure["semantic_tags"]
        assert structure["content_type"] == ContentType.UNKNOWN  # Would need more specific patterns


class TestContextualEmbeddingSystem:
    """Test ContextualEmbeddingSystem functionality."""
    
    def test_system_initialization(self):
        """Test system initialization with default config."""
        system = ContextualEmbeddingSystem()
        
        assert system.config is not None
        assert isinstance(system.document_processor, DocumentProcessor)
        assert system.config.context_window_size == 2
    
    def test_system_initialization_with_config(self):
        """Test system initialization with custom config."""
        config = RetrievalConfig(context_window_size=3, max_context_length=2000)
        system = ContextualEmbeddingSystem(config)
        
        assert system.config.context_window_size == 3
        assert system.config.max_context_length == 2000
    
    def test_create_contextual_chunks_basic(self):
        """Test basic contextual chunk creation."""
        system = ContextualEmbeddingSystem()
        
        document = Document(
            page_content="This is a test document with multiple sections.",
            metadata={"id": "test_doc_1", "title": "Test Document"}
        )
        
        chunks = [
            "First chunk of content here.",
            "Second chunk with different content.",
            "Third chunk to complete the test."
        ]
        
        contextual_chunks = system.create_contextual_chunks(document, chunks)
        
        assert len(contextual_chunks) == 3
        assert all(isinstance(chunk, ContextualChunk) for chunk in contextual_chunks)
        assert contextual_chunks[0].chunk_index == 0
        assert contextual_chunks[1].chunk_index == 1
        assert contextual_chunks[2].chunk_index == 2
        
        # Check that document title is propagated
        assert all(chunk.document_title == "Test Document" for chunk in contextual_chunks)
    
    def test_create_contextual_chunks_with_metadata(self):
        """Test contextual chunk creation with custom metadata."""
        system = ContextualEmbeddingSystem()
        
        document = Document(
            page_content="Test document content.",
            metadata={"id": "test_doc_1", "title": "Test Document"}
        )
        
        chunks = ["Chunk 1", "Chunk 2"]
        chunk_metadata = [
            {"custom_field": "value1"},
            {"custom_field": "value2"}
        ]
        
        contextual_chunks = system.create_contextual_chunks(
            document, chunks, chunk_metadata
        )
        
        assert len(contextual_chunks) == 2
        assert contextual_chunks[0].metadata["custom_field"] == "value1"
        assert contextual_chunks[1].metadata["custom_field"] == "value2"
    
    def test_create_contextual_chunks_empty_input(self):
        """Test handling of empty input."""
        system = ContextualEmbeddingSystem()
        
        document = Document(page_content="", metadata={})
        chunks = []
        
        contextual_chunks = system.create_contextual_chunks(document, chunks)
        assert len(contextual_chunks) == 0
    
    def test_create_contextual_chunks_with_filtering(self):
        """Test chunk filtering based on quality."""
        config = RetrievalConfig(
            filter_low_quality_chunks=True,
            min_chunk_length=20
        )
        system = ContextualEmbeddingSystem(config)
        
        document = Document(
            page_content="Test document",
            metadata={"id": "test_doc"}
        )
        
        chunks = [
            "This is a good chunk with sufficient length for processing.",
            "Short",  # Should be filtered out
            "Another good chunk that meets the minimum length requirement."
        ]
        
        contextual_chunks = system.create_contextual_chunks(document, chunks)
        
        # Should have filtered out the short chunk
        assert len(contextual_chunks) == 2
        assert contextual_chunks[0].text.startswith("This is a good chunk")
        assert contextual_chunks[1].text.startswith("Another good chunk")
    
    def test_extract_chunk_context_surrounding(self):
        """Test surrounding context extraction."""
        config = RetrievalConfig(
            context_strategy=ContextStrategy.SURROUNDING,
            context_window_size=1
        )
        system = ContextualEmbeddingSystem(config)
        
        chunks = ["Chunk 0", "Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
        doc_structure = {"sections": []}
        
        # Test context for middle chunk
        context = system._extract_chunk_context(chunks, 2, doc_structure)
        assert "Chunk 1" in context
        assert "Chunk 3" in context
        assert "Chunk 2" not in context  # Should not include the chunk itself
        
        # Test context for first chunk
        context = system._extract_chunk_context(chunks, 0, doc_structure)
        assert "Chunk 1" in context
        assert "Chunk 2" not in context  # Outside window
    
    def test_get_context_chunk_indices(self):
        """Test context chunk index calculation."""
        system = ContextualEmbeddingSystem()
        
        # Test middle chunk
        indices = system._get_context_chunk_indices(2, 5)
        expected = [0, 1, 3, 4]  # window size 2, excluding chunk 2 itself
        assert sorted(indices) == sorted(expected)
        
        # Test edge chunk
        indices = system._get_context_chunk_indices(0, 5)
        expected = [1, 2]  # Can't go below 0
        assert sorted(indices) == sorted(expected)
    
    @pytest.mark.asyncio
    async def test_embed_contextual_chunks(self):
        """Test contextual chunk embedding generation."""
        system = ContextualEmbeddingSystem()
        
        # Create mock embedding model
        mock_embeddings = AsyncMock(spec=Embeddings)
        mock_embeddings.aembed_documents = AsyncMock(return_value=[
            [0.1, 0.2, 0.3],  # Embedding for chunk 1
            [0.4, 0.5, 0.6]   # Embedding for chunk 2
        ])
        
        chunks = [
            ContextualChunk("Chunk 1", "Context 1", 0, "doc1"),
            ContextualChunk("Chunk 2", "Context 2", 1, "doc1")
        ]
        
        result = await system.embed_contextual_chunks(chunks, mock_embeddings)
        
        assert len(result) == 2
        assert result[0][0] == chunks[0]  # First chunk
        assert result[0][1] == [0.1, 0.2, 0.3]  # First embedding
        assert result[1][0] == chunks[1]  # Second chunk
        assert result[1][1] == [0.4, 0.5, 0.6]  # Second embedding
    
    @pytest.mark.asyncio 
    async def test_embed_contextual_chunks_sync_fallback(self):
        """Test fallback to sync embedding when async not available."""
        system = ContextualEmbeddingSystem()
        
        # Create mock embedding model without async support
        mock_embeddings = Mock(spec=Embeddings)
        mock_embeddings.embed_documents = Mock(return_value=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        
        chunks = [
            ContextualChunk("Chunk 1", "Context 1", 0, "doc1"),
            ContextualChunk("Chunk 2", "Context 2", 1, "doc1")
        ]
        
        result = await system.embed_contextual_chunks(chunks, mock_embeddings)
        
        assert len(result) == 2
        mock_embeddings.embed_documents.assert_called_once()
    
    def test_get_system_stats(self):
        """Test system statistics generation."""
        config = RetrievalConfig(context_window_size=3)
        system = ContextualEmbeddingSystem(config)
        
        stats = system.get_system_stats()
        
        assert "config" in stats
        assert "system_info" in stats
        assert stats["config"]["context_window_size"] == 3
        assert stats["system_info"]["component"] == "ContextualEmbeddingSystem"
        assert stats["system_info"]["task"] == "3.1"


@pytest.mark.integration
class TestContextualEmbeddingIntegration:
    """Integration tests for the contextual embedding system."""
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end contextual processing."""
        # Setup
        config = RetrievalConfig(
            context_window_size=1,
            include_document_title=True,
            include_section_headers=True
        )
        system = ContextualEmbeddingSystem(config)
        
        # Create a realistic document
        document = Document(
            page_content="""# Casino Gaming Guide
            
This guide covers the fundamentals of casino gaming.

## Poker Basics

Poker is a card game that combines skill and chance. 
Understanding the basic rules is essential for success.

## Blackjack Strategy

Blackjack requires mathematical thinking and strategy.
Card counting can improve your odds significantly.""",
            metadata={
                "id": "casino_guide_001",
                "title": "Ultimate Casino Guide",
                "category": "gambling",
                "author": "Gaming Expert"
            }
        )
        
        # Split into chunks (simplified)
        chunks = [
            "This guide covers the fundamentals of casino gaming.",
            "Poker is a card game that combines skill and chance.",
            "Understanding the basic rules is essential for success.",
            "Blackjack requires mathematical thinking and strategy.",
            "Card counting can improve your odds significantly."
        ]
        
        # Process chunks
        contextual_chunks = system.create_contextual_chunks(document, chunks)
        
        # Verify results
        assert len(contextual_chunks) == 5
        
        # Check first chunk
        first_chunk = contextual_chunks[0]
        assert first_chunk.document_title == "Ultimate Casino Guide"
        assert "casino" in first_chunk.semantic_tags
        assert first_chunk.document_position == 0.0
        
        # Check that context is properly added
        full_text = first_chunk.full_text
        assert "Document: Ultimate Casino Guide" in full_text
        assert "Content Type:" in full_text or "Topics:" in full_text
        assert "Content: This guide covers" in full_text
        
        # Check middle chunk has surrounding context
        middle_chunk = contextual_chunks[2]
        assert middle_chunk.context  # Should have context from surrounding chunks
        
        # Verify quality scores are assigned
        assert all(0.0 <= chunk.quality_score <= 1.0 for chunk in contextual_chunks)
        
        # Verify chunk validation
        for chunk in contextual_chunks:
            is_valid, issues = chunk.validate_quality()
            assert is_valid, f"Chunk validation failed: {issues}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 