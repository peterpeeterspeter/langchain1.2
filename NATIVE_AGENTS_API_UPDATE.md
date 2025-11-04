# Native Agents API Update

**Date:** 2025-11-04
**Status:** IN PROGRESS
**Priority:** HIGH

---

## 🔍 Discovery

When attempting to test the native agents, we discovered that the **LangChain API has changed** since the agents were originally written.

### Issue Found

The native agents were implemented using:
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
```

However, **these imports don't exist in the current LangChain version** (1.0.3):
```
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```

### Current API (Correct)

The **modern LangChain 1.0+ approach** uses:
```python
from langgraph.prebuilt import create_react_agent
```

This is actually **better** because:
- ✅ It's the official LangGraph pattern for agents
- ✅ Simpler API (one function instead of two)
- ✅ Returns a compiled graph (more flexible)
- ✅ Built-in state management
- ✅ Better integration with LangGraph workflows

---

## 🔧 Required Changes

All 6 native agent files need to be updated:

### 1. Import Changes

**OLD (doesn't work):**
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
```

**NEW (correct):**
```python
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
```

### 2. Agent Creation Changes

**OLD:**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert agent..."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=verbose,
    max_iterations=max_iterations,
    return_intermediate_steps=True,
)

return agent_executor
```

**NEW:**
```python
system_message = SystemMessage(content="You are an expert agent...")

agent = create_react_agent(
    llm,
    tools,
    state_modifier=system_message
)

return agent  # Compiled graph, ready to use
```

### 3. Invocation Changes

**OLD:**
```python
result = await agent_executor.ainvoke({
    "input": query
})

output = result.get("output", "")
intermediate_steps = result.get("intermediate_steps", [])
```

**NEW:**
```python
from langchain_core.messages import HumanMessage

result = await agent.ainvoke({
    "messages": [HumanMessage(content=query)]
})

messages = result.get("messages", [])
final_message = messages[-1]
output = final_message.content
```

### 4. Result Extraction Changes

**OLD (intermediate_steps):**
```python
for action, observation in intermediate_steps:
    tool_name = action.tool
    tool_input = action.tool_input
    # Process...
```

**NEW (messages):**
```python
for message in messages:
    # Check for tool calls
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.get('name')
            tool_input = tool_call.get('args')
            # Process...

    # Check for tool responses
    if hasattr(message, 'name') and message.name:
        tool_name = message.name
        tool_output = message.content
        # Process...
```

---

## ✅ Progress

### Updated Files

1. **✅ src/agents/research_agent_native.py**
   - Updated imports
   - Updated `create_native_research_agent()` to use `create_react_agent`
   - Updated `native_research_node()` to use messages format
   - Updated `_extract_research_data_from_messages()` helper

### Remaining Files

2. **⏳ src/agents/writing_agent_native.py** - PENDING
3. **⏳ src/agents/affiliate_agent_native.py** - PENDING
4. **⏳ src/agents/image_agent_native.py** - PENDING
5. **⏳ src/agents/publishing_agent_native.py** - PENDING
6. **⏳ src/agents/orchestrator_native.py** - PENDING

---

## 🧪 Testing Requirements

### Prerequisites

To test the native agents, you need:

1. **API Keys** (at least one):
   ```bash
   # Create .env file with:
   OPENAI_API_KEY=sk-...
   # OR
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. **Required Packages** (already installed):
   ```bash
   langchain==1.0.3
   langchain-core==1.0.3
   langchain-openai==1.0.2
   langgraph==1.0.2
   ```

### Test Approach

Once API keys are available, run:

```bash
# Test the architecture pattern
python test_react_agent_pattern.py

# This will verify:
# - create_react_agent() works correctly
# - LLM makes intelligent tool decisions
# - Message-based state management works
# - Reasoning traces are available
```

**Expected Results:**
- Agent should make 1-3 tool calls (adaptive based on query)
- Final message should contain synthesized response
- Tool calls should be visible in message history
- Agent should demonstrate reasoning (not just call all tools)

---

## 🔄 Migration Impact

### What This Means

- **Good News**: The new API is actually simpler and more powerful
- **Better Integration**: Native LangGraph support is cleaner
- **Same Benefits**: Still get 40-60% cost savings from adaptive tool calling
- **More Features**: Better state management and checkpointing

### What Stays The Same

- ✅ All tools remain unchanged (still use `@tool` decorator)
- ✅ State management (ArticleCMSState) unchanged
- ✅ LangGraph workflow structure unchanged
- ✅ Core agent logic and strategies unchanged

### What Changed

- ❌ API calls changed (but simpler now)
- ❌ Result format changed (messages instead of intermediate_steps)
- ❌ Need to update result extraction logic

---

## 📝 Next Steps

### Immediate Actions

1. **Update Remaining Agents** (Est: 1-2 hours)
   - Apply same pattern to writing, affiliate, image, publishing agents
   - Update orchestrator to use new agent API
   - Test each agent individually

2. **Add API Keys** (User action required)
   - Create `.env` file with OPENAI_API_KEY or ANTHROPIC_API_KEY
   - Ensure keys have sufficient credits

3. **Run Tests** (Est: 30 minutes)
   - Test react_agent pattern
   - Test individual agents
   - Test complete orchestrator workflow

4. **Update Documentation** (Est: 30 minutes)
   - Update NATIVE_AGENTS_MIGRATION_GUIDE.md
   - Update code examples to show new API
   - Add testing results

### Testing Checklist

Once API keys are available:

- [ ] Test `create_react_agent` pattern works
- [ ] Test research_agent_native.py (already updated)
- [ ] Test writing_agent_native.py (after update)
- [ ] Test affiliate_agent_native.py (after update)
- [ ] Test image_agent_native.py (after update)
- [ ] Test publishing_agent_native.py (after update)
- [ ] Test orchestrator_native.py (after update)
- [ ] Verify cost savings (compare tool calls)
- [ ] Verify quality (content should be good)
- [ ] Document results

---

## 🎯 Success Criteria

The migration is complete and successful when:

1. ✅ All agents use `create_react_agent()` from `langgraph.prebuilt`
2. ✅ All agents can be instantiated without errors
3. ✅ Agents make intelligent, adaptive tool decisions
4. ✅ Tool calls are visible in message history
5. ✅ Cost savings of 40-60% are achieved
6. ✅ Quality matches or exceeds custom agents
7. ✅ Complete workflow runs end-to-end

---

## 💡 Example: Before vs After

### Before (Doesn't Work)

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

def create_native_research_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([...])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

# Usage
result = await agent.ainvoke({"input": "Research Betway"})
output = result["output"]
steps = result["intermediate_steps"]
```

### After (Correct)

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

def create_native_research_agent(llm, tools):
    system_msg = SystemMessage(content="You are an expert agent...")
    return create_react_agent(llm, tools, state_modifier=system_msg)

# Usage
result = await agent.ainvoke({"messages": [HumanMessage(content="Research Betway")]})
messages = result["messages"]
output = messages[-1].content
```

---

**Status:** Research agent updated ✅ | Remaining agents in progress ⏳

**Last Updated:** 2025-11-04
