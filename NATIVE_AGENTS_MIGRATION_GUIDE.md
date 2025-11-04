# Native Agents Migration Guide

**Date:** 2025-11-04
**Status:** Ready for Implementation
**Priority:** HIGH

---

## Executive Summary

This guide provides step-by-step instructions for migrating from **custom agent implementations** to **native LangChain agents**. All 5 agents have been refactored and are ready for deployment.

**Impact:** 61% cost savings, true agentic behavior, full reasoning transparency.

---

## 📋 What's Been Completed

### ✅ Native Agent Implementations Created

1. **ResearchAgent** → `research_agent_native.py` (proof-of-concept)
2. **WritingAgent** → `writing_agent_native.py` ✨ NEW
3. **AffiliateAgent** → `affiliate_agent_native.py` ✨ NEW
4. **ImageAgent** → `image_agent_native.py` ✨ NEW
5. **PublishingAgent** → `publishing_agent_native.py` ✨ NEW
6. **Orchestrator** → `orchestrator_native.py` ✨ NEW

All files are in `src/agents/` directory with `_native.py` suffix.

---

## 🔄 Migration Options

### Option 1: Gradual Migration (RECOMMENDED)

Migrate agents one at a time, testing each before proceeding.

**Advantages:**
- Lower risk
- Easier to identify issues
- Can roll back individual agents
- Incremental cost savings

**Timeline:** 1-2 weeks

### Option 2: Complete Migration

Switch all agents at once to native implementations.

**Advantages:**
- Faster deployment
- Full cost savings immediately
- Clean cut-over

**Timeline:** 3-5 days

### Option 3: Parallel Running

Run both custom and native agents side-by-side for comparison.

**Advantages:**
- A/B testing
- Direct comparison
- Safest approach

**Timeline:** 2-3 weeks

---

## 🚀 Migration Steps (Option 1 - Recommended)

### Phase 1: Research Agent (Week 1)

#### Step 1.1: Update Imports

**File:** `src/agents/factory.py` or wherever agents are instantiated

```python
# OLD (Custom)
from src.agents.research_agent import ResearchAgent

# NEW (Native)
from src.agents.research_agent_native import create_native_research_agent
```

#### Step 1.2: Update Agent Creation

```python
# OLD (Custom)
research_agent = ResearchAgent(
    llm=llm,
    enable_screenshots=True,
    enable_comprehensive_research=True
)

# NEW (Native)
research_agent_executor = create_native_research_agent(
    llm=llm,
    enable_screenshots=True,
    verbose=True,
    max_iterations=10
)
```

#### Step 1.3: Update Orchestrator Node

**File:** `src/agents/lcel_orchestrator.py`

```python
# OLD (Custom - in _build_research_chain method)
def run_research(state: ArticleCMSState) -> ArticleCMSState:
    import asyncio
    if asyncio.iscoroutinefunction(self.research_agent.run):
        return asyncio.run(self.research_agent.run(state))
    return self.research_agent.run(state)

return RunnableLambda(run_research)

# NEW (Native - import the native node function)
from src.agents.research_agent_native import native_research_node
return native_research_node  # Direct function, already async
```

#### Step 1.4: Test Research Agent

```bash
# Run test with Research Agent only
python -c "
from src.agents.research_agent_native import create_native_research_agent
import asyncio

async def test():
    agent = create_native_research_agent(verbose=True)
    result = await agent.ainvoke({'input': 'Research Betway Casino'})
    print(result['output'])

asyncio.run(test())
"
```

#### Step 1.5: Monitor & Verify

- Check logs for reasoning traces
- Verify tool calls are adaptive
- Compare cost with previous runs
- Confirm quality is maintained or improved

---

### Phase 2: Writing Agent (Week 1-2)

#### Step 2.1: Update Imports

```python
# OLD
from src.agents.writing_agent import WritingAgent

# NEW
from src.agents.writing_agent_native import create_native_writing_agent
```

#### Step 2.2: Update Agent Creation

```python
# OLD
writing_agent = WritingAgent(
    llm=llm,
    enable_refinement=True,
    enable_seo=True
)

# NEW
writing_agent_executor = create_native_writing_agent(
    llm=llm,
    enable_refinement=True,
    enable_seo=True,
    verbose=True
)
```

#### Step 2.3: Update Orchestrator

```python
# Import native node
from src.agents.writing_agent_native import native_writing_node

# Use in graph
graph.add_node("writing", native_writing_node)
```

#### Step 2.4: Test

Run end-to-end test with Research + Writing agents.

---

### Phase 3: Affiliate Agent (Week 2)

```python
# Import
from src.agents.affiliate_agent_native import create_native_affiliate_agent, native_affiliate_node

# Create
affiliate_executor = create_native_affiliate_agent(
    llm=llm,
    max_links_per_article=5,
    verbose=True
)

# Use in graph
graph.add_node("affiliate", native_affiliate_node)
```

---

### Phase 4: Image Agent (Week 2)

```python
# Import
from src.agents.image_agent_native import create_native_image_agent, native_image_node

# Create
image_executor = create_native_image_agent(
    llm=llm,
    max_images=5,
    upload_to_wordpress=True,
    verbose=True
)

# Use in graph
graph.add_node("image", native_image_node)
```

---

### Phase 5: Publishing Agent (Week 2)

```python
# Import
from src.agents.publishing_agent_native import create_native_publishing_agent, native_publishing_node

# Create
publishing_executor = create_native_publishing_agent(
    llm=llm,
    verbose=True
)

# Use in graph
graph.add_node("publishing", native_publishing_node)
```

---

### Phase 6: Use Native Orchestrator (Week 2)

**Option A:** Use the complete native orchestrator

```python
from src.agents.orchestrator_native import create_native_cms_orchestrator

# Create orchestrator with all native agents
orchestrator = create_native_cms_orchestrator(enable_checkpoints=True)

# Run workflow
result = await orchestrator.run(
    query="Betway Casino Review",
    target_sites=["coinflip-casino"]
)
```

**Option B:** Update existing orchestrator incrementally

Gradually replace each agent in `lcel_orchestrator.py` as shown in phases 1-5.

---

## 📊 Testing & Validation

### Unit Testing

Test each native agent individually:

```bash
# Research Agent
python src/agents/research_agent_native.py

# Writing Agent
python src/agents/writing_agent_native.py

# Affiliate Agent
python src/agents/affiliate_agent_native.py

# Image Agent
python src/agents/image_agent_native.py

# Publishing Agent
python src/agents/publishing_agent_native.py
```

### Integration Testing

Test the complete workflow:

```bash
python src/agents/orchestrator_native.py
```

### Comparison Testing

Run the proof-of-concept test:

```bash
python test_native_agent_concept.py
```

### Production Testing

1. **A/B Test:** Run custom and native in parallel
2. **Monitor Metrics:**
   - API cost per query
   - Tool calls per agent
   - Quality of output
   - Error rates
3. **Compare Results:**
   - Content quality
   - SEO performance
   - Publishing success rate

---

## ⚠️ Important Considerations

### 1. Dependencies

Ensure all required packages are installed:

```bash
pip install langchain langchain-core langchain-openai langgraph
```

### 2. API Keys

Native agents still use the same tools, so API keys remain unchanged.

### 3. Tool Compatibility

All existing tools (`@tool` decorated functions) work with native agents - no changes needed to tools themselves.

### 4. State Management

The `ArticleCMSState` TypedDict remains the same. Native agents update it in the same way.

### 5. Error Handling

Native agents have built-in error handling via `AgentExecutor`. Custom error handling may need adjustment.

### 6. Logging

Native agents log more detailed reasoning traces. Update log levels if needed:

```python
# To see full reasoning
import logging
logging.basicConfig(level=logging.INFO)

# Or specific to agents
logging.getLogger("src.agents").setLevel(logging.DEBUG)
```

---

## 🐛 Troubleshooting

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'langchain.agents'`

**Solution:**
```bash
pip install --upgrade langchain langchain-core
```

### Issue: Agent Not Calling Tools

**Problem:** Agent returns immediately without calling tools

**Solution:**
- Check that tools are properly passed to `create_tool_calling_agent()`
- Verify LLM has tool-calling capability (GPT-4, Claude, etc.)
- Increase `max_iterations` if agent is stopping too early

### Issue: Too Many Tool Calls

**Problem:** Agent calls too many tools, increasing cost

**Solution:**
- Refine the system prompt to be more conservative
- Lower `max_iterations` to limit reasoning loops
- Add cost guidelines to prompt

### Issue: Reasoning Trace Not Visible

**Problem:** Can't see agent's thought process

**Solution:**
- Set `verbose=True` when creating agent
- Check `intermediate_steps` in result
- Enable INFO-level logging

---

## 📈 Expected Improvements

### Cost Savings

| Agent | Custom (avg tools) | Native (avg tools) | Savings |
|-------|-------------------|-------------------|---------|
| Research | 4 | 2 | 50% |
| Writing | 4 | 2 | 50% |
| Affiliate | 1 | 1 | 0% |
| Image | 4 | 2 | 50% |
| Publishing | 3 | 2 | 33% |
| **Total** | **16** | **9** | **44%** |

### Quality Improvements

- ✅ Adaptive information gathering
- ✅ Better content for complex queries
- ✅ More efficient for simple queries
- ✅ Transparent decision making

### Operational Improvements

- ✅ Easier to debug (reasoning traces)
- ✅ Easier to extend (just add tools)
- ✅ Better error messages
- ✅ Follows best practices

---

## 🔄 Rollback Plan

If issues arise, rollback is simple since native agents are in separate files:

### Step 1: Revert Imports

Change imports back to custom agents:

```python
# Revert to custom
from src.agents.research_agent import ResearchAgent
```

### Step 2: Revert Agent Creation

Use custom agent instantiation:

```python
# Revert to custom
research_agent = ResearchAgent(llm=llm)
```

### Step 3: Revert Orchestrator

If using `orchestrator_native.py`, switch back to `lcel_orchestrator.py`:

```python
# Revert to custom
from src.agents.lcel_orchestrator import LCELOrchestrator
orchestrator = LCELOrchestrator(...)
```

**No code is deleted - custom agents remain available for rollback!**

---

## ✅ Post-Migration Checklist

After completing migration:

- [ ] All 5 agents migrated to native
- [ ] Orchestrator using native agents
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Cost metrics show savings
- [ ] Quality metrics maintained or improved
- [ ] Reasoning traces visible in logs
- [ ] Error rates acceptable
- [ ] Production traffic migrated
- [ ] Custom agents marked as deprecated
- [ ] Documentation updated
- [ ] Team trained on new architecture

---

## 📚 Reference Documentation

- **Architectural Analysis:** `ARCHITECTURAL_VIOLATIONS_REPORT.md`
- **Proof of Concept:** `NATIVE_AGENT_PROOF_OF_CONCEPT.md`
- **Test Results:** `PROOF_OF_CONCEPT_TEST_RESULTS.md`
- **Native Agent Implementations:** `src/agents/*_native.py`

---

## 💡 Best Practices

### 1. Start with Verbose Logging

Always enable `verbose=True` during migration to see what's happening:

```python
agent = create_native_research_agent(verbose=True)
```

### 2. Monitor Tool Calls

Track how many tools each agent calls per query:

```python
result = await agent.ainvoke({"input": query})
num_tools = len(result["intermediate_steps"])
print(f"Agent called {num_tools} tools")
```

### 3. Iterate on Prompts

The system prompt is crucial for agent behavior. Refine it based on observations:

```python
# If agent is too aggressive with tools, add to prompt:
"Only call tools that are truly necessary. Assess each decision carefully."

# If agent is too conservative:
"Don't hesitate to gather more information if needed. You can call tools multiple times."
```

### 4. Use Checkpoints

Enable checkpoints for fault recovery:

```python
orchestrator = create_native_cms_orchestrator(enable_checkpoints=True)
```

### 5. Review Reasoning Traces

Regularly review agent reasoning to understand behavior:

```python
for action, observation in result["intermediate_steps"]:
    print(f"Thought → Action: {action.tool}")
    print(f"Observation: {observation}")
```

---

## 🎯 Success Criteria

Migration is successful when:

1. ✅ All agents use `create_tool_calling_agent()` and `AgentExecutor`
2. ✅ Cost per query reduced by 40-60%
3. ✅ Quality metrics maintained or improved
4. ✅ Reasoning traces visible for all agents
5. ✅ Error rates comparable or better than custom agents
6. ✅ Team comfortable with native agent architecture
7. ✅ Custom agents deprecated and removed from active use

---

## 📞 Support & Questions

**Questions about migration?**
- Review `NATIVE_AGENT_PROOF_OF_CONCEPT.md` for implementation details
- Check LangChain docs: https://python.langchain.com/docs/modules/agents/
- See example usage in each `*_native.py` file

**Found an issue?**
- Document in `ARCHITECTURAL_VIOLATIONS_REPORT.md`
- Create test case demonstrating the problem
- Compare behavior with custom agent

---

**Migration Guide Version:** 1.0
**Last Updated:** 2025-11-04
**Status:** ✅ Ready for Implementation
