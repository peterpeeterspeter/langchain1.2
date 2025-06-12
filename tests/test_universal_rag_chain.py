import asyncio
import pytest

from langchain_core.documents import Document

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from chains import create_universal_rag_chain, RAGResponse


class MockVectorStore:
    """Simple in-memory mock vector store for testing."""

    async def asimilarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        docs = [
            Document(page_content="Mock content 1 about casinos", metadata={"title": "Doc1"}),
            Document(page_content="Mock content 2 about blackjack", metadata={"title": "Doc2"}),
        ]
        # Return tuples of (Document, similarity_score)
        return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]


class FailingVectorStore:
    async def asimilarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        raise RuntimeError("Forced failure for testing")


@pytest.mark.asyncio
async def test_basic_chain_initialization():
    chain = create_universal_rag_chain(
        enable_prompt_optimization=False,
        enable_caching=False,
        enable_contextual_retrieval=False,
        vector_store=MockVectorStore(),
    )
    assert hasattr(chain, "ainvoke")


@pytest.mark.asyncio
async def test_optimized_chain_generation():
    chain = create_universal_rag_chain(
        enable_prompt_optimization=True,
        enable_caching=False,
        enable_contextual_retrieval=False,
        vector_store=MockVectorStore(),
    )
    response: RAGResponse = await chain.ainvoke("Which casino is safest for beginners?")
    assert isinstance(response, RAGResponse)
    assert response.answer
    assert response.sources  # should contain mocked sources
    assert response.confidence_score >= 0.0


@pytest.mark.asyncio
async def test_error_handling_fallback():
    chain = create_universal_rag_chain(
        enable_prompt_optimization=True,
        enable_caching=False,
        enable_contextual_retrieval=False,
        vector_store=FailingVectorStore(),
    )
    response = await chain.ainvoke("trigger failure")
    assert "error" in response.answer.lower() or response.confidence_score == 0.0


@pytest.mark.asyncio
async def test_caching_behavior():
    chain = create_universal_rag_chain(
        enable_prompt_optimization=False,
        enable_caching=True,
        enable_contextual_retrieval=False,
        vector_store=MockVectorStore(),
    )
    # First call – should not be cached
    resp1 = await chain.ainvoke("test cache")
    assert resp1.cached is False

    # Second identical call – should be cached
    resp2 = await chain.ainvoke("test cache")
    assert resp2.cached is True
    stats = chain.get_cache_stats()
    assert stats.get("hit_rate", 0) >= 0.5 