"""Example usages of the Universal RAG Chain"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from chains import create_universal_rag_chain


class DummyVectorStore:
    async def asimilarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        from langchain_core.documents import Document
        docs = [Document(page_content="Dummy content 1", metadata={"title": "Doc1"})]
        return [(docs[0], 0.9)]


def run_basic_usage():
    chain = create_universal_rag_chain(vector_store=DummyVectorStore())
    response = asyncio.run(chain.ainvoke("What is a casino bonus?"))
    print("Answer:", response.answer)


def run_optimized_usage():
    chain = create_universal_rag_chain(
        enable_prompt_optimization=True,
        enable_contextual_retrieval=True,
        vector_store=DummyVectorStore(),
    )
    resp = asyncio.run(chain.ainvoke("Which casino is safest for beginners?"))
    print("Answer:", resp.answer)
    print("Confidence:", resp.confidence_score)
    print("Sources:", resp.sources)


def run_error_handling():
    class FailingVS(DummyVectorStore):
        async def asimilarity_search_with_score(self, query: str, k: int = 4, **kwargs):
            raise RuntimeError("Vector store failure")

    chain = create_universal_rag_chain(vector_store=FailingVS())
    resp = asyncio.run(chain.ainvoke("trigger error"))
    print("Graceful response:", resp.answer)


if __name__ == "__main__":
    run_basic_usage()
    run_optimized_usage()
    run_error_handling() 