# Quick Start Guide - Enhanced Confidence Scoring System

Get started with the Enhanced Confidence Scoring System in just a few minutes. This guide will walk you through the complete setup process and your first implementation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Configuration](#basic-configuration)
4. [First Implementation](#first-implementation)
5. [Common Use Cases](#common-use-cases)
6. [Verification and Testing](#verification-and-testing)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, ensure you have:

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended for production)
- **Storage**: At least 1GB free space

### Required Dependencies
- `langchain-core`: Latest version
- `langchain`: Latest version
- `pydantic`: 2.0+
- `asyncio`: Built-in with Python 3.7+

### API Keys (Optional but Recommended)
- **OpenAI API Key**: For advanced LLM models
- **Anthropic API Key**: For Claude models
- **Vector Store**: Pinecone, Weaviate, or local Chroma

---

## Installation

### Step 1: Clone or Download the System

If you're integrating into an existing project:

```bash
# Navigate to your project directory
cd your-rag-project

# Copy the enhanced confidence scoring system files
# (Assuming you have the files from the repository)
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install langchain langchain-core pydantic

# For enhanced features (optional)
pip install openai anthropic pinecone-client chromadb

# For development and testing
pip install pytest pytest-asyncio
```

### Step 3: Verify Installation

```python
# Test basic imports
try:
    from chains.enhanced_confidence_scoring_system import EnhancedConfidenceCalculator
    from chains.universal_rag_lcel import create_universal_rag_chain
    print("✅ Installation successful!")
except ImportError as e:
    print(f"❌ Installation issue: {e}")
```

---

## Basic Configuration

### Step 1: Environment Setup

Create a `.env` file in your project root:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Vector store configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment
```

### Step 2: Basic Configuration File

Create `config.py`:

```python
# config.py
import os
from typing import Dict, Any

# Basic configuration
CONFIG = {
    # Model settings
    "model_name": "gpt-3.5-turbo",  # or "gpt-4", "claude-3-sonnet"
    
    # Enhanced confidence settings
    "enable_enhanced_confidence": True,
    "enable_prompt_optimization": True,
    "enable_caching": True,
    
    # Confidence thresholds
    "quality_threshold": 0.70,
    "regeneration_threshold": 0.40,
    
    # Cache settings
    "cache_strategy": "BALANCED",  # CONSERVATIVE, BALANCED, AGGRESSIVE, ADAPTIVE
    "cache_max_size": 1000,
    
    # API keys
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
}

def get_confidence_config() -> Dict[str, Any]:
    """Get confidence scoring configuration."""
    return {
        'quality_threshold': CONFIG['quality_threshold'],
        'regeneration_threshold': CONFIG['regeneration_threshold'],
        'cache_strategy': CONFIG['cache_strategy'],
        'max_regeneration_attempts': 2,
        'enable_quality_flags': True,
        'enable_improvement_suggestions': True,
    }
```

---

## First Implementation

### Step 1: Create a Simple RAG Chain

Create `basic_rag.py`:

```python
# basic_rag.py
import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from chains.universal_rag_lcel import create_universal_rag_chain
from config import CONFIG, get_confidence_config

async def setup_basic_rag():
    """Set up a basic RAG system with enhanced confidence scoring."""
    
    # Step 1: Create embeddings (you can use any embeddings)
    embeddings = OpenAIEmbeddings(
        api_key=CONFIG['openai_api_key']
    )
    
    # Step 2: Create or load vector store
    # For this example, we'll create a simple in-memory store
    sample_documents = [
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
            metadata={"source": "rag_guide.pdf", "type": "technical"}
        ),
        Document(
            page_content="Confidence scoring helps assess the quality and reliability of RAG responses.",
            metadata={"source": "confidence_paper.pdf", "type": "academic"}
        ),
        Document(
            page_content="Vector databases store embeddings for efficient similarity search in RAG systems.",
            metadata={"source": "vector_db_tutorial.md", "type": "tutorial"}
        )
    ]
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=sample_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Step 3: Create enhanced RAG chain
    chain = create_universal_rag_chain(
        model_name=CONFIG['model_name'],
        enable_enhanced_confidence=CONFIG['enable_enhanced_confidence'],
        enable_prompt_optimization=CONFIG['enable_prompt_optimization'],
        enable_caching=CONFIG['enable_caching'],
        confidence_config=get_confidence_config(),
        vector_store=vector_store
    )
    
    return chain

async def test_basic_query():
    """Test the RAG system with a basic query."""
    
    # Set up the chain
    chain = await setup_basic_rag()
    
    # Test query
    test_query = "What is RAG and how does it work?"
    
    print(f"🔍 Query: {test_query}")
    print("⏳ Processing...")
    
    # Get response
    response = await chain.ainvoke({
        "query": test_query,
        "query_type": "factual"  # Optional: helps with confidence scoring
    })
    
    # Display results
    print("\n📝 Response:")
    print(f"Answer: {response.answer}")
    print(f"\n📊 Confidence Score: {response.confidence_score:.3f}")
    print(f"🏆 Quality Level: {response.quality_level}")
    print(f"⚡ Response Time: {response.response_time:.2f}s")
    print(f"💾 Cached: {response.cached}")
    
    if response.quality_flags:
        print(f"\n🚩 Quality Flags: {', '.join(response.quality_flags)}")
    
    if response.improvement_suggestions:
        print(f"\n💡 Improvement Suggestions:")
        for suggestion in response.improvement_suggestions:
            print(f"  • {suggestion}")
    
    return response

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_basic_query())
```

### Step 2: Run Your First Query

```bash
# Run the basic example
python basic_rag.py
```

Expected output:
```
🔍 Query: What is RAG and how does it work?
⏳ Processing...

📝 Response:
Answer: RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It works by first retrieving relevant documents from a knowledge base, then using that information to generate accurate and contextual responses.

📊 Confidence Score: 0.782
🏆 Quality Level: good
⚡ Response Time: 1.45s
💾 Cached: False

🚩 Quality Flags: good_quality

💡 Improvement Suggestions:
  • Consider adding more diverse sources for better coverage
  • Include more recent information about RAG developments
```

---

## Common Use Cases

### Use Case 1: High-Quality Question Answering

```python
# high_quality_qa.py
import asyncio
from chains.universal_rag_lcel import create_universal_rag_chain

async def create_qa_system():
    """Create a high-quality Q&A system."""
    
    chain = create_universal_rag_chain(
        model_name="gpt-4",  # Higher quality model
        enable_enhanced_confidence=True,
        confidence_config={
            'quality_threshold': 0.80,      # Higher quality threshold
            'cache_strategy': 'CONSERVATIVE', # Only cache best responses
            'regeneration_threshold': 0.50,  # Regenerate low-quality responses
        }
    )
    
    return chain

async def qa_example():
    """Example Q&A session."""
    chain = await create_qa_system()
    
    questions = [
        "What are the best practices for implementing RAG systems?",
        "How do you evaluate the performance of a RAG system?",
        "What are common challenges in RAG implementation?"
    ]
    
    for question in questions:
        response = await chain.ainvoke({
            "query": question,
            "query_type": "factual"
        })
        
        print(f"\n🤔 Q: {question}")
        print(f"💬 A: {response.answer}")
        print(f"📊 Confidence: {response.confidence_score:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(qa_example())
```

### Use Case 2: Tutorial Generation

```python
# tutorial_generator.py
import asyncio
from chains.universal_rag_lcel import create_universal_rag_chain

async def create_tutorial_system():
    """Create a system optimized for tutorials."""
    
    chain = create_universal_rag_chain(
        model_name="gpt-4",
        enable_enhanced_confidence=True,
        confidence_config={
            'query_type_weights': {
                'tutorial': {
                    'content_quality': 0.45,    # Emphasize clarity
                    'source_quality': 0.20,    
                    'query_matching': 0.25,    # Match learning style
                    'technical_factors': 0.10
                }
            }
        }
    )
    
    return chain

async def tutorial_example():
    """Generate a tutorial response."""
    chain = await create_tutorial_system()
    
    response = await chain.ainvoke({
        "query": "How to build a RAG system step by step for beginners?",
        "query_type": "tutorial"
    })
    
    print(f"📚 Tutorial Response:")
    print(f"{response.answer}")
    print(f"\n📊 Tutorial Quality Score: {response.confidence_score:.3f}")

if __name__ == "__main__":
    asyncio.run(tutorial_example())
```

### Use Case 3: Performance Monitoring

```python
# performance_monitor.py
import asyncio
from chains.universal_rag_lcel import create_universal_rag_chain

async def monitor_performance():
    """Monitor system performance over multiple queries."""
    
    chain = create_universal_rag_chain(
        enable_enhanced_confidence=True,
        enable_caching=True,
        confidence_config={'cache_strategy': 'ADAPTIVE'}
    )
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain natural language processing",
        "What is computer vision?"
    ]
    
    results = []
    
    for query in test_queries:
        response = await chain.ainvoke({"query": query})
        results.append({
            'query': query,
            'confidence': response.confidence_score,
            'quality_level': response.quality_level,
            'response_time': response.response_time,
            'cached': response.cached
        })
    
    # Performance summary
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_response_time = sum(r['response_time'] for r in results) / len(results)
    cache_hit_rate = sum(1 for r in results if r['cached']) / len(results)
    
    print("📈 Performance Summary:")
    print(f"Average Confidence: {avg_confidence:.3f}")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print(f"Cache Hit Rate: {cache_hit_rate:.1%}")
    
    # Get system status
    if hasattr(chain, 'get_enhanced_system_status'):
        status = chain.get_enhanced_system_status()
        print(f"\n🔧 System Status:")
        print(f"Cache Performance: {status.get('cache_performance', {})}")
        print(f"Confidence Metrics: {status.get('confidence_metrics', {})}")

if __name__ == "__main__":
    asyncio.run(monitor_performance())
```

---

## Verification and Testing

### Step 1: Run System Health Check

```python
# health_check.py
import asyncio
from chains.enhanced_confidence_scoring_system import EnhancedConfidenceCalculator
from chains.universal_rag_lcel import create_universal_rag_chain

async def health_check():
    """Perform comprehensive system health check."""
    
    print("🔍 Enhanced Confidence Scoring System Health Check")
    print("=" * 50)
    
    # Test 1: Component initialization
    try:
        calculator = EnhancedConfidenceCalculator()
        print("✅ EnhancedConfidenceCalculator: OK")
    except Exception as e:
        print(f"❌ EnhancedConfidenceCalculator: {e}")
    
    # Test 2: Chain creation
    try:
        chain = create_universal_rag_chain(
            enable_enhanced_confidence=True,
            enable_caching=True
        )
        print("✅ Universal RAG Chain: OK")
    except Exception as e:
        print(f"❌ Universal RAG Chain: {e}")
    
    # Test 3: Basic functionality
    try:
        # Create a minimal test
        from langchain_core.documents import Document
        from models import RAGResponse
        
        test_response = RAGResponse(
            answer="Test response",
            sources=[{"content": "test", "metadata": {}}]
        )
        
        breakdown, enhanced = await calculator.calculate_enhanced_confidence(
            response=test_response,
            query="test query",
            sources=[Document(page_content="test", metadata={})]
        )
        
        print(f"✅ Confidence Calculation: OK (Score: {breakdown.overall_score:.3f})")
    except Exception as e:
        print(f"❌ Confidence Calculation: {e}")
    
    # Test 4: Cache functionality
    try:
        from chains.enhanced_confidence_scoring_system import IntelligentCache
        cache = IntelligentCache()
        metrics = cache.get_performance_metrics()
        print("✅ Intelligent Cache: OK")
    except Exception as e:
        print(f"❌ Intelligent Cache: {e}")
    
    print("\n🎉 Health check complete!")

if __name__ == "__main__":
    asyncio.run(health_check())
```

### Step 2: Run Performance Benchmarks

```python
# benchmark.py
import asyncio
import time
from statistics import mean, stdev

async def benchmark_system():
    """Run performance benchmarks."""
    
    from chains.universal_rag_lcel import create_universal_rag_chain
    
    # Create chain with different configurations
    configurations = [
        ("Basic", {"enable_enhanced_confidence": False}),
        ("Enhanced", {"enable_enhanced_confidence": True}),
        ("Full Features", {
            "enable_enhanced_confidence": True,
            "enable_caching": True,
            "enable_prompt_optimization": True
        })
    ]
    
    test_query = "What are the benefits of using RAG systems?"
    
    for config_name, config in configurations:
        print(f"\n📊 Benchmarking: {config_name}")
        
        chain = create_universal_rag_chain(**config)
        
        # Run multiple times for average
        times = []
        for i in range(5):
            start_time = time.time()
            response = await chain.ainvoke({"query": test_query})
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = mean(times)
        std_time = stdev(times) if len(times) > 1 else 0
        
        print(f"  Average Time: {avg_time:.3f}s (±{std_time:.3f}s)")
        if hasattr(response, 'confidence_score'):
            print(f"  Confidence Score: {response.confidence_score:.3f}")

if __name__ == "__main__":
    asyncio.run(benchmark_system())
```

---

## Next Steps

### 1. Production Configuration

Once you've verified the system works, configure it for production:

```python
# production_config.py
PRODUCTION_CONFIG = {
    # Use more powerful models
    "model_name": "gpt-4",
    
    # Optimize for production performance
    "confidence_config": {
        'quality_threshold': 0.75,
        'cache_strategy': 'ADAPTIVE',
        'max_regeneration_attempts': 1,  # Limit regeneration in production
    },
    
    # Enable monitoring
    "enable_monitoring": True,
    "log_level": "INFO",
}
```

### 2. Advanced Features

Explore advanced features:

- **Custom Query Types**: Define domain-specific query types with custom weights
- **Source Quality Customization**: Implement domain-specific source quality indicators
- **Advanced Caching**: Use Redis or other external cache systems
- **Analytics Integration**: Connect to monitoring and analytics platforms

### 3. Integration with Existing Systems

- **API Integration**: Wrap the system in a REST API using FastAPI
- **Database Integration**: Connect to your existing knowledge bases
- **Authentication**: Add user authentication and access controls
- **Scalability**: Deploy with load balancing and horizontal scaling

### 4. Monitoring and Maintenance

Set up ongoing monitoring:

```python
# monitoring.py
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

# Regular health checks
async def daily_health_check():
    """Daily system health check."""
    # Implementation here
    pass

# Performance monitoring
async def monitor_performance():
    """Monitor and alert on performance issues."""
    # Implementation here
    pass
```

---

## Common Issues and Solutions

### Issue 1: Low Confidence Scores

**Problem**: Getting consistently low confidence scores

**Solutions:**
- Check source quality - ensure you have high-quality documents
- Verify query-response relevance
- Consider lowering quality thresholds for your domain
- Review and adjust query type weights

### Issue 2: Slow Response Times

**Problem**: System responds slowly

**Solutions:**
- Enable caching with appropriate strategy
- Use faster models for non-critical queries
- Optimize vector store configuration
- Consider parallel processing optimizations

### Issue 3: Cache Issues

**Problem**: Cache not working effectively

**Solutions:**
- Check cache strategy matches your use case
- Verify quality thresholds are appropriate
- Monitor cache hit rates and adjust accordingly
- Ensure sufficient cache size for your query volume

### Issue 4: Model Integration Issues

**Problem**: Issues with specific AI models

**Solutions:**
- Verify API keys are correct and active
- Check model availability and quotas
- Use fallback models for reliability
- Review model-specific configuration requirements

---

## Support and Resources

### Documentation
- [System Overview](./enhanced_confidence_scoring_system.md)
- [API Reference](./api_reference.md)
- [Production Deployment Guide](./production_deployment.md)

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences
- Examples: Check the examples directory for more use cases

### Professional Support
For enterprise deployments or custom integrations, consider:
- Professional services for implementation
- Custom training for your team
- Ongoing support and maintenance contracts

---

**🎉 Congratulations!** You've successfully set up the Enhanced Confidence Scoring System. You're now ready to build high-quality, confident RAG applications with advanced quality assessment and intelligent caching. 