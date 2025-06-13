#!/usr/bin/env python3
"""
Task 3.1: Contextual Embedding System Demo

This script demonstrates the capabilities of the contextual embedding system
including chunk creation, context extraction, and quality assessment.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from retrieval.contextual_embedding import (
    ContextualChunk,
    ContextualEmbeddingSystem,
    RetrievalConfig,
    DocumentProcessor,
    ContextStrategy,
    ContentType
)


def demo_basic_contextual_chunk():
    """Demonstrate basic ContextualChunk functionality."""
    print("=== Basic ContextualChunk Demo ===")
    
    chunk = ContextualChunk(
        text="This casino offers excellent poker games and slot machines.",
        context="The previous section discussed table games. The next section covers promotions.",
        chunk_index=2,
        document_id="casino_review_001",
        document_title="Best Online Casinos 2024",
        section_header="Game Selection",
        breadcrumbs=["Home", "Reviews", "Online Casinos"],
        content_type=ContentType.REVIEW,
        semantic_tags=["casino", "poker", "gaming"],
        total_chunks=10,
        document_position=0.3,
        quality_score=0.85
    )
    
    print(f"Original text: {chunk.text}")
    print(f"Context: {chunk.context}")
    print(f"Quality score: {chunk.quality_score}")
    print()
    print("Full contextualized text for embedding:")
    print("-" * 50)
    print(chunk.full_text)
    print("-" * 50)
    print()
    
    # Demonstrate validation
    is_valid, issues = chunk.validate_quality()
    print(f"Chunk validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        print(f"Issues: {issues}")
    print()


def demo_document_processing():
    """Demonstrate document structure extraction."""
    print("=== Document Processing Demo ===")
    
    # Create a realistic casino review document
    document = Document(
        page_content="""# Ultimate Casino Guide 2024

This comprehensive guide reviews the top online casinos.

## Game Selection

### Slot Machines
Modern casinos offer hundreds of slot games with various themes.
Progressive jackpots can reach millions of dollars.

### Table Games
Classic games like blackjack and poker remain popular.
Live dealer games provide authentic casino experience.

## Bonuses and Promotions

Welcome bonuses can double your initial deposit.
Regular players get loyalty rewards and cashback offers.""",
        metadata={
            "title": "Ultimate Casino Guide 2024",
            "author": "Casino Expert",
            "category": "gambling",
            "subcategory": "reviews",
            "source": "casinoreviews.com"
        }
    )
    
    config = RetrievalConfig()
    processor = DocumentProcessor(config)
    
    structure = processor.extract_document_structure(document)
    
    print(f"Extracted title: {structure['title']}")
    print(f"Content type: {structure['content_type']}")
    print(f"Semantic tags: {structure['semantic_tags']}")
    print(f"Number of sections: {len(structure['sections'])}")
    
    if structure['sections']:
        print("Sections found:")
        for i, section in enumerate(structure['sections'][:3]):  # Show first 3
            print(f"  {i+1}. {section['title']} (level {section['level']})")
    
    print()


def demo_contextual_embedding_system():
    """Demonstrate the full contextual embedding system."""
    print("=== Contextual Embedding System Demo ===")
    
    # Create configuration
    config = RetrievalConfig(
        context_window_size=2,
        include_document_title=True,
        include_section_headers=True,
        context_strategy=ContextStrategy.COMBINED,
        max_context_length=800,
        filter_low_quality_chunks=True,
        min_chunk_length=30
    )
    
    # Initialize system
    system = ContextualEmbeddingSystem(config)
    
    # Create a sample document
    document = Document(
        page_content="""# Casino Bonus Guide

Learn how to maximize your casino bonuses and promotions.

## Welcome Bonuses

Most online casinos offer generous welcome bonuses to new players.
These bonuses typically match your first deposit by 100% or more.
Always read the wagering requirements before claiming any bonus.

## Loyalty Programs

Regular players can join VIP programs for exclusive benefits.
Higher tier members receive faster withdrawals and personal account managers.
Some casinos offer cashback on losses and reload bonuses.""",
        metadata={
            "id": "bonus_guide_001",
            "title": "Casino Bonus Strategies",
            "category": "gambling",
            "author": "Bonus Expert"
        }
    )
    
    # Split into chunks (simplified chunking for demo)
    chunks = [
        "Learn how to maximize your casino bonuses and promotions.",
        "Most online casinos offer generous welcome bonuses to new players.",
        "These bonuses typically match your first deposit by 100% or more.",
        "Always read the wagering requirements before claiming any bonus.",
        "Regular players can join VIP programs for exclusive benefits.",
        "Higher tier members receive faster withdrawals and personal account managers.",
        "Some casinos offer cashback on losses and reload bonuses."
    ]
    
    print(f"Processing {len(chunks)} chunks with config:")
    print(f"  Context window size: {config.context_window_size}")
    print(f"  Strategy: {config.context_strategy.value}")
    print(f"  Include document title: {config.include_document_title}")
    print()
    
    # Create contextual chunks
    contextual_chunks = system.create_contextual_chunks(document, chunks)
    
    print(f"Created {len(contextual_chunks)} contextual chunks:")
    print()
    
    # Show details for a few chunks
    for i, chunk in enumerate(contextual_chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"Original: {chunk.text}")
        print(f"Quality: {chunk.quality_score:.2f}")
        print(f"Context source: {chunk.context_source}")
        print(f"Context chunks used: {chunk.context_chunks_used}")
        print(f"Document position: {chunk.document_position:.1%}")
        print()
        print("Contextualized text:")
        print(chunk.full_text[:300] + "..." if len(chunk.full_text) > 300 else chunk.full_text)
        print("-" * 50)
        print()
    
    # Show system statistics
    stats = system.get_system_stats()
    print("System Statistics:")
    print(f"  Version: {stats['system_info']['version']}")
    print(f"  Component: {stats['system_info']['component']}")
    print(f"  Task: {stats['system_info']['task']}")
    print()


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("=== Configuration Options Demo ===")
    
    # Test different strategies
    strategies = [
        (ContextStrategy.SURROUNDING, "Uses surrounding chunks for context"),
        (ContextStrategy.HIERARCHICAL, "Uses document structure for context"),
        (ContextStrategy.COMBINED, "Combines multiple context sources")
    ]
    
    document = Document(
        page_content="Sample document for testing different strategies.",
        metadata={"title": "Test Document"}
    )
    
    chunks = ["First chunk", "Second chunk", "Third chunk"]
    
    for strategy, description in strategies:
        print(f"Strategy: {strategy.value}")
        print(f"Description: {description}")
        
        config = RetrievalConfig(context_strategy=strategy)
        system = ContextualEmbeddingSystem(config)
        
        contextual_chunks = system.create_contextual_chunks(document, chunks)
        
        print(f"Generated {len(contextual_chunks)} chunks")
        if contextual_chunks:
            sample_chunk = contextual_chunks[1]  # Middle chunk
            print(f"Sample context: {sample_chunk.context[:100]}...")
        print()
    
    # Test configuration validation
    print("Configuration Validation:")
    
    # Valid config
    valid_config = RetrievalConfig()
    issues = valid_config.validate_config()
    print(f"Valid config issues: {len(issues)}")
    
    # Invalid config
    invalid_config = RetrievalConfig(
        context_window_size=-1,
        max_context_length=50,
        quality_threshold=1.5
    )
    issues = invalid_config.validate_config()
    print(f"Invalid config issues: {len(issues)}")
    for issue in issues:
        print(f"  - {issue}")
    print()


async def demo_embedding_integration():
    """Demonstrate integration with embedding models (mock)."""
    print("=== Embedding Integration Demo ===")
    
    # Create mock embedding model
    class MockEmbeddings:
        def embed_documents(self, texts):
            # Return mock embeddings (random-like vectors)
            import random
            return [[random.random() for _ in range(384)] for _ in texts]
        
        async def aembed_documents(self, texts):
            # Simulate async embedding
            await asyncio.sleep(0.1)
            return self.embed_documents(texts)
    
    system = ContextualEmbeddingSystem()
    embeddings = MockEmbeddings()
    
    # Create sample chunks
    chunks = [
        ContextualChunk(
            text="Poker is a popular card game in casinos.",
            context="Previous discussion about casino games.",
            chunk_index=0,
            document_id="poker_guide"
        ),
        ContextualChunk(
            text="Blackjack requires strategic thinking.",
            context="Card games section continues.",
            chunk_index=1,
            document_id="poker_guide"
        )
    ]
    
    print(f"Generating embeddings for {len(chunks)} contextual chunks...")
    
    # Generate embeddings
    chunk_embeddings = await system.embed_contextual_chunks(chunks, embeddings)
    
    print(f"Generated {len(chunk_embeddings)} embedding pairs")
    for i, (chunk, embedding) in enumerate(chunk_embeddings):
        print(f"Chunk {i+1}: {chunk.text[:50]}...")
        print(f"Embedding size: {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        print()


def main():
    """Run all demonstrations."""
    print("🎰 Task 3.1: Contextual Embedding System Demonstration")
    print("=" * 60)
    print()
    
    # Run demonstrations
    demo_basic_contextual_chunk()
    demo_document_processing()
    demo_contextual_embedding_system()
    demo_configuration_options()
    
    # Run async demo
    asyncio.run(demo_embedding_integration())
    
    print("✅ Task 3.1 Implementation Complete!")
    print()
    print("Key features demonstrated:")
    print("- ContextualChunk with rich metadata and full_text generation")
    print("- DocumentProcessor for structure extraction")
    print("- ContextualEmbeddingSystem for chunk processing")
    print("- Configurable context strategies and quality control")
    print("- Integration with embedding models")
    print("- Comprehensive validation and error handling")


if __name__ == "__main__":
    main() 