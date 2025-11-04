# LangSmith Integration Guide

Complete guide to LangSmith observability, evaluation, and prompt testing integration in Universal RAG CMS.

## Table of Contents

1. [Overview](#overview)
2. [Setup & Configuration](#setup--configuration)
3. [Observability & Tracing](#observability--tracing)
4. [Evaluation Framework](#evaluation-framework)
5. [Prompt Testing](#prompt-testing)
6. [API Middleware](#api-middleware)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

LangSmith provides comprehensive tools for developing, debugging, and deploying LLM applications. This integration adds:

- **Observability**: Visual tracing of all chain executions
- **Evaluation**: Systematic quality assessment with datasets
- **Prompt Testing**: Version control and A/B testing for prompts
- **Production Monitoring**: Automatic tracing in production environments

### Benefits

- 🔍 **Better Debugging**: Visual trace UI for complex chains
- 📊 **Systematic Evaluation**: Track quality metrics over time
- 🚀 **Production Monitoring**: Built-in monitoring and alerting
- 🔄 **Prompt Optimization**: Data-driven prompt iteration
- 📉 **Reduced Maintenance**: Less custom monitoring code

---

## Setup & Configuration

### 1. Install LangSmith

LangSmith is already included in `requirements.txt`:

```bash
pip install langsmith>=0.1.0
```

### 2. Get Your API Key

1. Sign up at [smith.langchain.com](https://smith.langchain.com) (free account)
2. Go to **Settings** → **API Keys** → **Create API Key**
3. Copy your API key

### 3. Configure Environment Variables

Add to your `.env` file:

```bash
# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=universal-rag-cms

# Optional: Custom endpoint
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

### 4. Verify Configuration

The system automatically detects LangSmith configuration. Check logs for:

```
✅ LangSmith tracing enabled for project: universal-rag-cms
```

---

## Observability & Tracing

### Universal RAG Chain Tracing

All chain invocations are automatically traced when LangSmith is enabled:

```python
from src.chains.universal_rag_lcel import create_universal_rag_chain

# Create chain (tracing enabled automatically)
chain = create_universal_rag_chain()

# Execute with automatic tracing
response = await chain.ainvoke({"query": "Betway Casino Review"})

# View traces at: https://smith.langchain.com/projects/universal-rag-cms
```

**What Gets Traced:**
- Query input and processing
- Retrieval operations
- LLM calls with prompts and responses
- Chain composition steps
- Error handling and retries
- Performance metrics (latency, tokens)

### Agent Orchestrator Tracing

Agent workflows are automatically traced:

```python
from src.agents.factory import create_agent_based_cms

cms = create_agent_based_cms(
    enable_research=True,
    enable_writing=True,
    enable_publishing=True
)

# Workflow execution is automatically traced
result = await cms.run(
    query="CyBet Casino Review 2025",
    target_sites=["crashcasino"]
)
```

**Traced Workflow Steps:**
- Research agent execution
- Writing agent execution
- Affiliate link integration
- Image processing
- WordPress publishing

### Custom Metadata

Additional metadata is automatically included in traces:

```python
# Universal RAG Chain includes:
{
    "query": "your query",
    "publish_to_wordpress": True,
    "chain_type": "universal_rag"
}

# Agent Orchestrator includes:
{
    "query": "your query",
    "target_sites": ["site1", "site2"],
    "workflow_type": "cms_orchestrator",
    "checkpoints_enabled": True
}
```

---

## Evaluation Framework

### Creating Evaluation Datasets

Use pre-configured datasets or create custom ones:

```python
from src.evaluation.langsmith_evaluator import LangSmithEvaluator
from src.evaluation.evaluation_config import (
    CASINO_REVIEW_DATASET,
    AFFILIATE_CONTENT_DATASET,
    GENERAL_QUERY_DATASET
)

# Initialize evaluator
evaluator = LangSmithEvaluator()

# Create casino review dataset
evaluator.create_dataset(
    dataset_name=CASINO_REVIEW_DATASET.name,
    examples=CASINO_REVIEW_DATASET.examples,
    description=CASINO_REVIEW_DATASET.description
)
```

### Running Evaluations

Evaluate your chain on a dataset:

```python
from src.evaluation.langsmith_evaluator import evaluate_chain_on_dataset, EvaluationConfig
from src.chains.universal_rag_lcel import create_universal_rag_chain

# Create chain
chain = create_universal_rag_chain()

# Configure evaluation
config = EvaluationConfig(
    dataset_name="casino-review-evaluation",
    evaluators=["qa", "embedding_distance", "criteria"],
    tags=["production", "casino-reviews"]
)

# Run evaluation
results = evaluate_chain_on_dataset(
    chain=chain.ainvoke,
    config=config
)

print(f"Evaluation completed: {results}")
```

### Available Evaluators

1. **QA Evaluator** (`"qa"`): Question answering correctness
   - Requires LLM
   - Compares responses to reference answers

2. **Embedding Distance** (`"embedding_distance"`): Semantic similarity
   - No LLM required
   - Measures semantic similarity to reference

3. **Cosine Similarity** (`"cosine_similarity"`): Vector similarity
   - No LLM required
   - Uses cosine distance between embeddings

4. **Criteria Evaluator** (`"criteria"`): Custom criteria
   - Requires LLM
   - Evaluates against custom criteria

### Evaluation Results

View results in LangSmith UI or programmatically:

```python
# Get evaluation results
results = evaluator.get_evaluation_results(
    dataset_name="casino-review-evaluation",
    limit=100
)

for result in results:
    print(f"Run ID: {result['run_id']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    print(f"Success: {result.get('error') is None}")
```

---

## Prompt Testing

### Testing Prompt Variations

Test different prompt versions with LangSmith:

```python
from src.chains.advanced_prompt_system import OptimizedPromptManager
from src.prompts.langsmith_prompt_tester import PromptTestConfig

# Initialize prompt manager with LangSmith testing
manager = OptimizedPromptManager(enable_langsmith_testing=True)

# Test current prompts
test_queries = [
    "Betway Casino Review",
    "Compare TrustDice vs Stake",
    "Best crypto casino bonuses"
]

results = manager.test_prompt_variation(
    test_queries=test_queries,
    prompt_version="v1.0",
    tags=["casino-reviews"]
)

print(f"Success Rate: {results['success_rate']:.2%}")
```

### Comparing Prompt Versions

A/B test different prompt versions:

```python
from src.prompts.langsmith_prompt_tester import LangSmithPromptTester

tester = LangSmithPromptTester()

# Define prompt variations
def prompt_v1(query: str) -> str:
    return f"Answer this question: {query}"

def prompt_v2(query: str) -> str:
    return f"Provide a comprehensive answer to: {query}"

# Compare versions
comparison = tester.compare_prompt_versions(
    prompt_funcs={"v1": prompt_v1, "v2": prompt_v2},
    test_queries=test_queries,
    prompt_name="casino_review_prompt"
)

print(f"Best Version: {comparison['best_version']}")
```

### Creating Prompt Test Suites

Create comprehensive test suites:

```python
from src.prompts.langsmith_prompt_tester import create_prompt_test_suite, PromptTestConfig

# Define test configurations
configs = [
    PromptTestConfig(
        prompt_name="casino_review_prompt",
        prompt_version="v1.0",
        test_queries=[
            "Betway Casino Review",
            "TrustDice comprehensive analysis"
        ],
        expected_behaviors={
            "min_length": 1000,
            "should_include_ratings": True
        },
        tags=["casino", "review"]
    )
]

# Create test suite
suite_id = create_prompt_test_suite(
    suite_name="casino-review-prompts",
    prompt_configs=configs,
    description="Test suite for casino review prompts"
)
```

---

## API Middleware

### Automatic API Tracing

All API endpoints are automatically traced when LangSmith is enabled:

```python
# Start API server
python -m src.api.main

# All requests are automatically traced:
# GET /api/v1/config/prompt-optimization
# POST /api/v1/contextual/query
# GET /retrieval/api/v1/config
```

**Traced Information:**
- Request path and method
- Query parameters
- Response status codes
- Request duration
- Error details (if any)

### Accessing Traces

Traces are available in LangSmith UI:

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Select project: `rag-cms-api`
3. View traces for each API request

---

## Best Practices

### 1. Project Organization

Use different projects for different environments:

```bash
# Development
LANGSMITH_PROJECT=universal-rag-cms-dev

# Staging
LANGSMITH_PROJECT=universal-rag-cms-staging

# Production
LANGSMITH_PROJECT=universal-rag-cms-prod
```

### 2. Evaluation Strategy

- **Regular Evaluations**: Run evaluations weekly or after major changes
- **Multiple Evaluators**: Use 2-3 evaluators for comprehensive assessment
- **Baseline Metrics**: Establish baseline metrics before optimizations
- **Regression Detection**: Monitor for quality degradation over time

### 3. Prompt Testing Workflow

1. **Create Test Suite**: Define test queries and expected behaviors
2. **Test Variations**: Test multiple prompt versions
3. **Compare Results**: Use comparison tools to find best version
4. **Deploy Best**: Deploy winning prompt version
5. **Monitor Performance**: Track performance in production

### 4. Production Monitoring

- **Enable Tracing**: Set `LANGSMITH_TRACING=true` in production
- **Monitor Traces**: Check LangSmith UI regularly for issues
- **Set Alerts**: Configure alerts for error rates or latency spikes
- **Review Metrics**: Analyze performance trends weekly

### 5. Performance Considerations

- **Sampling**: LangSmith tracing has minimal overhead (<5ms)
- **Async Operations**: Tracing doesn't block chain execution
- **Graceful Degradation**: System works without LangSmith if unavailable

---

## Troubleshooting

### LangSmith Not Tracing

**Symptoms**: No traces appearing in LangSmith UI

**Solutions**:
1. Check environment variables:
   ```bash
   echo $LANGSMITH_TRACING  # Should be "true"
   echo $LANGSMITH_API_KEY   # Should be set
   ```

2. Verify API key is valid:
   ```python
   from langsmith import Client
   client = Client()
   print(client.validate_api_key())  # Should print True
   ```

3. Check logs for errors:
   ```
   ⚠️ Failed to initialize LangSmith tracer: ...
   ```

### Evaluation Failures

**Symptoms**: Evaluations failing with errors

**Solutions**:
1. Ensure dataset exists:
   ```python
   evaluator = LangSmithEvaluator()
   datasets = list(evaluator.client.list_datasets())
   print([d.name for d in datasets])
   ```

2. Check evaluator requirements:
   - QA and Criteria evaluators require LLM
   - Embedding distance requires embeddings

3. Verify chain signature:
   - Chain must accept dict input: `{"query": "..."}`
   - Chain must return string or dict with "answer" key

### Prompt Testing Not Working

**Symptoms**: Prompt testing methods raising errors

**Solutions**:
1. Verify LangSmith is enabled:
   ```python
   from src.utils.langsmith_utils import is_langsmith_enabled
   print(is_langsmith_enabled())  # Should be True
   ```

2. Check prompt tester initialization:
   ```python
   from src.prompts.langsmith_prompt_tester import LangSmithPromptTester
   tester = LangSmithPromptTester()  # Should not raise error
   ```

### Import Errors

**Symptoms**: `ImportError: LangSmith not available`

**Solutions**:
1. Install LangSmith:
   ```bash
   pip install langsmith>=0.1.0
   ```

2. Verify installation:
   ```python
   import langsmith
   print(langsmith.__version__)
   ```

---

## Integration Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│         LangSmith Integration           │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   Observability (Tracing)        │  │
│  │   - Universal RAG Chain           │  │
│  │   - Agent Orchestrators          │  │
│  │   - API Middleware                │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   Evaluation Framework           │  │
│  │   - QA Evaluator                 │  │
│  │   - Embedding Distance           │  │
│  │   - Criteria Evaluator            │  │
│  │   - Dataset Management           │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   Prompt Testing                 │  │
│  │   - Prompt Tester                │  │
│  │   - Version Comparison           │  │
│  │   - Performance Tracking         │  │
│  └──────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

### File Structure

```
src/
├── utils/
│   └── langsmith_utils.py          # Shared utilities
├── evaluation/
│   ├── __init__.py
│   ├── langsmith_evaluator.py      # Evaluation framework
│   └── evaluation_config.py        # Dataset configurations
├── prompts/
│   └── langsmith_prompt_tester.py  # Prompt testing
├── chains/
│   ├── universal_rag_lcel.py      # Tracing integration
│   └── advanced_prompt_system.py   # Prompt testing integration
├── agents/
│   ├── orchestrator.py             # Tracing integration
│   └── lcel_orchestrator.py        # Tracing integration
└── api/
    └── main.py                     # API middleware
```

---

## Examples

### Complete Example: Chain with Tracing and Evaluation

```python
import asyncio
from src.chains.universal_rag_lcel import create_universal_rag_chain
from src.evaluation.langsmith_evaluator import run_evaluation

async def main():
    # Create chain (tracing enabled automatically)
    chain = create_universal_rag_chain()
    
    # Generate content (automatically traced)
    response = await chain.ainvoke({
        "query": "Betway Casino Review 2025"
    })
    
    print(f"Response: {response.answer[:200]}...")
    
    # Run evaluation on dataset
    results = run_evaluation(
        chain=chain.ainvoke,
        dataset_name="casino-review-evaluation",
        evaluator_types=["qa", "embedding_distance"]
    )
    
    print(f"Evaluation Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example: Prompt Testing Workflow

```python
from src.chains.advanced_prompt_system import OptimizedPromptManager
from src.prompts.langsmith_prompt_tester import LangSmithPromptTester, PromptTestConfig

# Initialize components
manager = OptimizedPromptManager(enable_langsmith_testing=True)
tester = LangSmithPromptTester()

# Create test suite
configs = [
    PromptTestConfig(
        prompt_name="casino_review",
        prompt_version="v1.0",
        test_queries=["Betway Review", "TrustDice Analysis"],
        expected_behaviors={"min_length": 1000}
    )
]

suite_id = tester.create_prompt_test_suite(
    suite_name="casino-prompts",
    prompt_configs=configs
)

# Test prompts
results = manager.test_prompt_variation(
    test_queries=["Betway Review"],
    prompt_version="v1.0"
)

print(f"Success Rate: {results['success_rate']:.2%}")
```

---

## Migration from Custom Monitoring

### Parallel Implementation (Recommended)

Run LangSmith alongside existing monitoring:

1. Both systems collect metrics
2. Compare outputs for consistency
3. Gradually migrate to LangSmith
4. Keep custom analytics for business metrics

### Full Migration

When ready to fully migrate:

1. Verify LangSmith captures all needed metrics
2. Remove custom monitoring code
3. Use LangSmith as single source of truth
4. Keep custom business analytics separate

---

## Resources

- [LangSmith Documentation](https://docs.langchain.com/langsmith/home)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [LangChain Evaluation Guide](https://docs.langchain.com/langsmith/evaluation)
- [LangSmith Prompt Testing](https://docs.langchain.com/langsmith/prompt-engineering)

---

## Support

For issues or questions:

1. Check LangSmith UI for trace details
2. Review logs for error messages
3. Verify environment configuration
4. Consult LangSmith documentation

---

**Last Updated**: 2025-01-27  
**Integration Version**: 1.0.0  
**LangSmith Version**: >=0.1.0


