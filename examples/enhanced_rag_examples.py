"""
Enhanced RAG Examples - Practical Usage Demonstrations

This file provides practical examples of using the Enhanced Confidence Scoring System.
"""

import asyncio
import time
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from chains.universal_rag_lcel import create_universal_rag_chain
from chains.enhanced_confidence_scoring_system import (
    EnhancedConfidenceCalculator,
    SourceQualityAnalyzer,
    ResponseValidator
)


class EnhancedRAGExamples:
    """Practical examples for Enhanced RAG features."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.sample_documents = self._create_sample_documents()
        self.vector_store = self._create_vector_store()
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for examples."""
        return [
            Document(
                page_content="""RAG (Retrieval-Augmented Generation) combines information retrieval 
                with text generation. It works by retrieving relevant documents and using them as 
                context for generating responses. This improves accuracy and reduces hallucinations.""",
                metadata={"source": "rag_guide.pdf", "type": "technical", "quality_tier": "HIGH"}
            ),
            Document(
                page_content="""Vector databases store embeddings for efficient similarity search. 
                Popular options include Pinecone, Weaviate, and Chroma. Each has different 
                strengths for various use cases.""",
                metadata={"source": "vector_db_guide.md", "type": "tutorial", "quality_tier": "HIGH"}
            ),
            Document(
                page_content="""I tried RAG and it's okay. Sometimes works well, sometimes not. 
                Setup can be tricky.""",
                metadata={"source": "user_review.txt", "type": "review", "quality_tier": "LOW"}
            )
        ]
    
    def _create_vector_store(self) -> Chroma:
        """Create vector store with sample documents."""
        return Chroma.from_documents(
            documents=self.sample_documents,
            embedding=self.embeddings,
            persist_directory="./examples_chroma_db"
        )

    async def basic_usage_example(self):
        """Example 1: Basic Enhanced RAG usage"""
        print("🔹 Example 1: Basic Enhanced RAG Usage")
        print("=" * 50)
        
        chain = create_universal_rag_chain(
            model_name="gpt-3.5-turbo",
            enable_enhanced_confidence=True,
            vector_store=self.vector_store
        )
        
        query = "What is RAG and how does it work?"
        print(f"Query: {query}")
        
        response = await chain.ainvoke({"query": query, "query_type": "factual"})
        
        print(f"\nAnswer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Quality Level: {response.quality_level}")
        print(f"Response Time: {response.response_time:.3f}s")
        
        return response

    async def custom_configuration_example(self):
        """Example 2: Custom confidence configuration"""
        print("\n🔹 Example 2: Custom Configuration")
        print("=" * 50)
        
        confidence_config = {
            'quality_threshold': 0.80,
            'cache_strategy': 'CONSERVATIVE',
            'query_type_weights': {
                'factual': {
                    'content_quality': 0.40,
                    'source_quality': 0.35,
                    'query_matching': 0.15,
                    'technical_factors': 0.10
                }
            }
        }
        
        chain = create_universal_rag_chain(
            model_name="gpt-4",
            enable_enhanced_confidence=True,
            confidence_config=confidence_config,
            vector_store=self.vector_store
        )
        
        query = "How do vector databases improve search performance?"
        response = await chain.ainvoke({"query": query, "query_type": "factual"})
        
        print(f"Query: {query}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Quality Flags: {', '.join(response.quality_flags)}")

    async def query_types_example(self):
        """Example 3: Different query types"""
        print("\n🔹 Example 3: Query Type Optimization")
        print("=" * 50)
        
        chain = create_universal_rag_chain(
            enable_enhanced_confidence=True,
            vector_store=self.vector_store
        )
        
        queries = [
            ("How to set up a vector database?", "tutorial"),
            ("Compare Pinecone vs Chroma", "comparison"),
            ("What is the latest in RAG research?", "factual"),
            ("Should I use RAG for my project?", "review")
        ]
        
        for query, query_type in queries:
            response = await chain.ainvoke({"query": query, "query_type": query_type})
            print(f"\nType: {query_type} | Confidence: {response.confidence_score:.3f}")

    async def caching_example(self):
        """Example 4: Intelligent caching"""
        print("\n🔹 Example 4: Caching Performance")
        print("=" * 50)
        
        chain = create_universal_rag_chain(
            enable_enhanced_confidence=True,
            enable_caching=True,
            confidence_config={'cache_strategy': 'ADAPTIVE'},
            vector_store=self.vector_store
        )
        
        query = "What are the benefits of RAG systems?"
        
        # First request (miss)
        start = time.time()
        response1 = await chain.ainvoke({"query": query})
        time1 = time.time() - start
        
        # Second request (potential hit)
        start = time.time()
        response2 = await chain.ainvoke({"query": query})
        time2 = time.time() - start
        
        print(f"First request: {time1:.3f}s (Cached: {response1.cached})")
        print(f"Second request: {time2:.3f}s (Cached: {response2.cached})")
        print(f"Speed improvement: {((time1-time2)/time1*100):.1f}%")

    async def source_quality_example(self):
        """Example 5: Source quality analysis"""
        print("\n🔹 Example 5: Source Quality Analysis")
        print("=" * 50)
        
        analyzer = SourceQualityAnalyzer()
        analysis = analyzer.analyze_sources(self.sample_documents)
        
        print(f"Overall Quality: {analysis.overall_quality:.3f}")
        print(f"Quality Distribution: {analysis.quality_distribution}")
        
        for i, doc in enumerate(self.sample_documents):
            tier = analyzer.get_source_tier(doc)
            print(f"Source {i+1}: {tier} quality")

    async def validation_example(self):
        """Example 6: Response validation"""
        print("\n🔹 Example 6: Response Validation")
        print("=" * 50)
        
        validator = ResponseValidator()
        
        from models import RAGResponse
        
        # High quality response
        good_response = RAGResponse(
            answer="RAG combines retrieval with generation for better accuracy.",
            sources=[{"content": "technical docs"}]
        )
        
        # Low quality response
        poor_response = RAGResponse(
            answer="RAG is okay.",
            sources=[{"content": "brief note"}]
        )
        
        for i, response in enumerate([good_response, poor_response], 1):
            validation = validator.validate_response(
                response, "What is RAG?", self.sample_documents
            )
            print(f"Response {i} Score: {validation.overall_score:.3f}")

    async def performance_monitoring_example(self):
        """Example 7: Performance monitoring"""
        print("\n🔹 Example 7: Performance Monitoring")
        print("=" * 50)
        
        chain = create_universal_rag_chain(
            enable_enhanced_confidence=True,
            enable_caching=True,
            vector_store=self.vector_store
        )
        
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What is AI?",
            "How to implement RAG?"
        ]
        
        results = []
        for query in queries:
            start = time.time()
            response = await chain.ainvoke({"query": query})
            duration = time.time() - start
            
            results.append({
                'confidence': response.confidence_score,
                'time': duration,
                'cached': response.cached
            })
        
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        cache_rate = sum(1 for r in results if r['cached']) / len(results)
        
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Time: {avg_time:.3f}s")
        print(f"Cache Hit Rate: {cache_rate:.1%}")

    async def comparison_example(self):
        """Example 8: Basic vs Enhanced RAG"""
        print("\n🔹 Example 8: Basic vs Enhanced Comparison")
        print("=" * 50)
        
        query = "How do vector databases work?"
        
        # Basic RAG
        basic_chain = create_universal_rag_chain(
            enable_enhanced_confidence=False,
            vector_store=self.vector_store
        )
        
        # Enhanced RAG
        enhanced_chain = create_universal_rag_chain(
            enable_enhanced_confidence=True,
            vector_store=self.vector_store
        )
        
        basic_response = await basic_chain.ainvoke({"query": query})
        enhanced_response = await enhanced_chain.ainvoke({"query": query})
        
        print(f"Basic Confidence: {getattr(basic_response, 'confidence_score', 'N/A')}")
        print(f"Enhanced Confidence: {enhanced_response.confidence_score:.3f}")
        print(f"Enhanced Quality: {enhanced_response.quality_level}")

    async def run_all_examples(self):
        """Run all examples in sequence."""
        print("🚀 Enhanced RAG Examples - Complete Demo")
        print("=" * 60)
        
        examples = [
            self.basic_usage_example,
            self.custom_configuration_example,
            self.query_types_example,
            self.caching_example,
            self.source_quality_example,
            self.validation_example,
            self.performance_monitoring_example,
            self.comparison_example
        ]
        
        for example in examples:
            try:
                await example()
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"❌ Error in {example.__name__}: {e}")
        
        print(f"\n✅ Demo completed at: {datetime.now().strftime('%H:%M:%S')}")


# Quick demo functions
async def quick_demo():
    """Quick demonstration."""
    examples = EnhancedRAGExamples()
    await examples.basic_usage_example()
    await examples.source_quality_example()

async def caching_demo():
    """Caching demonstration."""
    examples = EnhancedRAGExamples()
    await examples.caching_example()

async def performance_demo():
    """Performance demonstration."""
    examples = EnhancedRAGExamples()
    await examples.performance_monitoring_example()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        
        if demo_type == "quick":
            asyncio.run(quick_demo())
        elif demo_type == "caching":
            asyncio.run(caching_demo())
        elif demo_type == "performance":
            asyncio.run(performance_demo())
        else:
            print("Available: quick, caching, performance")
    else:
        examples = EnhancedRAGExamples()
        asyncio.run(examples.run_all_examples())


"""
Usage:
1. All examples: python examples/enhanced_rag_examples.py
2. Quick demo: python examples/enhanced_rag_examples.py quick
3. Caching demo: python examples/enhanced_rag_examples.py caching
4. Performance demo: python examples/enhanced_rag_examples.py performance
""" 