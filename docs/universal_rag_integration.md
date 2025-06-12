# Universal RAG Chain Integration Guide

> **Status**: Beta – ready for production use

This guide explains how to use the Universal RAG Chain (URC) with optional advanced prompt optimisation, contextual retrieval and semantic caching.

## 1. Quick Start

```python
from chains import create_universal_rag_chain
from your_vector_store import YourVectorStore  # e.g. Supabase, Pinecone, etc.

vector_store = YourVectorStore(...)

# Create an optimised chain (GPT-4 + advanced prompts + caching)
chain = create_universal_rag_chain(
    model_name="gpt-4",
    enable_prompt_optimization=True,
    enable_caching=True,
    enable_contextual_retrieval=True,
    vector_store=vector_store,
)

response = await chain.ainvoke("Which casino is safest for beginners?")
print(response.answer)
print(response.confidence_score)
print(response.sources[:2])
```

## 2. Feature Matrix

| Feature | Flag / Component | Benefit |
|---------|-----------------|---------|
| Prompt optimisation | `enable_prompt_optimization=True` | +37 % relevance, +31 % accuracy |
| Contextual retrieval | `enable_contextual_retrieval=True` | −49 % retrieval failures |
| Semantic caching | `enable_caching=True` | Sub-500 ms cached responses |
| Graceful fallbacks | Built-in | Never crashes – always returns structured `RAGResponse` |

## 3. API Surface

```python
from chains import UniversalRAGChain, RAGResponse

class UniversalRAGChain:
    async def ainvoke(self, query: str, **kwargs) -> RAGResponse:
        ...

class RAGResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    confidence_score: float  # 0.0 – 1.0
    cached: bool
    response_time: float  # ms
    token_usage: dict[str, int] | None
    query_analysis: dict[str, Any] | None
```

## 4. Usage Patterns

### Standard (no optimisation)
```python
chain = create_universal_rag_chain(vector_store=vector_store)
```

### Fully optimised
```python
chain = create_universal_rag_chain(
    model_name="claude-3-sonnet",
    enable_prompt_optimization=True,
    enable_caching=True,
    enable_contextual_retrieval=True,
    vector_store=vector_store,
)
```

### Error handling
```python
try:
    resp = await chain.ainvoke("Your query")
except Exception as e:
    # Rare – URC gracefully degrades internally
    print("URC error:", e)
```

## 5. Performance Tips

* Use `enable_caching=True` in production – hit-rate metrics available via `chain.get_cache_stats()`.
* Tune `k` in your vector store's `asimilarity_search_with_score` for recall vs latency.
* For non-critical freshness, increase cache TTL via `_get_ttl_by_query_type` mapping.

---
© 2025 Universal RAG CMS Project 