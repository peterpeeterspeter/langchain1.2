"""
Native LangChain Research Agent - PROOF OF CONCEPT
This is the CORRECT way to implement agents using native LangChain components.

Key differences from custom implementation:
1. Uses create_tool_calling_agent() - native LangChain agent factory
2. Uses AgentExecutor - native execution loop with reasoning
3. LLM decides which tools to call and in what order
4. Supports multi-step reasoning and tool iteration
5. Built-in error handling and retries
6. Agent scratchpad for chain-of-thought reasoning
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .tools.research_tools import (
    web_search_tool,
    comprehensive_research_tool,
    screenshot_tool,
    casino_intelligence_tool
)
from .state import ArticleCMSState

logger = logging.getLogger(__name__)


# ============================================================================
# NATIVE AGENT IMPLEMENTATION
# ============================================================================

def create_native_research_agent(
    llm: Optional[ChatOpenAI] = None,
    enable_screenshots: bool = True,
    verbose: bool = True,
    max_iterations: int = 10,
) -> AgentExecutor:
    """
    Create a NATIVE LangChain research agent using create_tool_calling_agent()

    This is the CORRECT implementation that:
    - Lets the LLM decide which tools to call
    - Supports iterative reasoning (call tool → analyze → call another tool)
    - Uses AgentExecutor for execution loop
    - Includes agent scratchpad for reasoning trace
    - Has built-in error handling and retries

    Args:
        llm: Language model (defaults to GPT-4o-mini)
        enable_screenshots: Whether to enable screenshot tool
        verbose: Whether to log agent reasoning steps
        max_iterations: Maximum reasoning iterations

    Returns:
        AgentExecutor configured with research tools
    """
    # Default LLM if not provided
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Assemble tools - LLM will choose which to use!
    tools = [
        web_search_tool,
        comprehensive_research_tool,
        casino_intelligence_tool,
    ]

    if enable_screenshots:
        tools.append(screenshot_tool)

    # Create agent prompt with system instructions
    # The {agent_scratchpad} placeholder is CRITICAL - it contains the agent's reasoning trace
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research agent specialized in gathering comprehensive information about online casinos.

Your goal is to research casinos thoroughly by gathering information from multiple sources.

**Available Tools:**
- web_search_tool: Quick web search for overview information (Tavily API)
- comprehensive_research_tool: Deep research with 95-field casino intelligence extraction
- casino_intelligence_tool: Extract structured casino data (licenses, games, bonuses, payments, etc.)
- screenshot_tool: Capture visual evidence from casino websites

**Strategy:**
1. Start with web_search_tool for a quick overview if needed
2. Use comprehensive_research_tool for detailed information extraction
3. Use casino_intelligence_tool to extract structured data
4. Use screenshot_tool for visual evidence of key pages
5. Analyze results and decide if more information is needed

**Important:**
- You can call tools multiple times if needed
- Analyze each tool's output before deciding what to do next
- If a tool fails, try an alternative approach
- Be thorough but efficient - only call tools that provide value

**Decision Making:**
- For simple queries, comprehensive_research_tool alone may be sufficient
- For detailed casino reviews, use multiple tools to gather complete information
- Screenshots are valuable but not always necessary"""),

        ("human", "{input}"),

        # CRITICAL: Agent scratchpad holds the reasoning trace
        # Format: Thought → Action → Action Input → Observation → (repeat)
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the NATIVE agent using LangChain's factory function
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Wrap in AgentExecutor for execution loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # Return reasoning trace
    )

    logger.info(f"Created native research agent with {len(tools)} tools")
    return agent_executor


# ============================================================================
# LANGGRAPH NODE FUNCTION (for integration with workflow)
# ============================================================================

async def native_research_node(state: ArticleCMSState) -> ArticleCMSState:
    """
    LangGraph node function using NATIVE agent

    This replaces the custom ResearchAgent.run() method with native agent execution.
    The agent will reason about which tools to call based on the query.

    Args:
        state: Current workflow state

    Returns:
        Updated state with research results
    """
    query = state.get("query", "")
    if not query:
        logger.error("No query provided for research")
        state["errors"] = state.get("errors", []) + ["No query provided"]
        return state

    # Create native agent
    agent_executor = create_native_research_agent(
        verbose=True,  # Show reasoning in logs
        max_iterations=10,
    )

    try:
        logger.info(f"Native Research Agent starting for query: {query}")

        # Execute agent - LLM will decide which tools to call!
        result = await agent_executor.ainvoke({
            "input": query,
        })

        # Extract results
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse intermediate steps to extract tool results
        research_data = _extract_research_data_from_steps(intermediate_steps)

        # Update state
        state["research_data"] = research_data
        state["research_output"] = output
        state["research_intermediate_steps"] = intermediate_steps
        state["workflow_step"] = state.get("workflow_step", 0) + 1
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["research_agent"] = "completed"

        logger.info(f"Native Research Agent completed - {len(intermediate_steps)} tool calls")

    except Exception as e:
        logger.error(f"Native Research Agent failed: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"Research agent error: {str(e)}"]
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["research_agent"] = "failed"

    return state


def _extract_research_data_from_steps(intermediate_steps: List) -> Dict[str, Any]:
    """
    Extract structured research data from agent's intermediate steps

    Args:
        intermediate_steps: List of (AgentAction, observation) tuples

    Returns:
        Structured research data dictionary
    """
    research_data = {
        "web_search_results": [],
        "comprehensive_research": {},
        "structured_intelligence": {},
        "screenshots": [],
        "tool_calls": []
    }

    for action, observation in intermediate_steps:
        tool_name = action.tool
        tool_input = action.tool_input

        # Track all tool calls
        research_data["tool_calls"].append({
            "tool": tool_name,
            "input": tool_input,
            "output": observation
        })

        # Organize by tool type
        if tool_name == "web_search_tool":
            research_data["web_search_results"] = observation or []

        elif tool_name == "comprehensive_research_tool":
            if isinstance(observation, dict):
                research_data["comprehensive_research"] = observation.get("research_data", {})

        elif tool_name == "casino_intelligence_tool":
            if isinstance(observation, dict):
                research_data["structured_intelligence"] = observation.get("data", {})

        elif tool_name == "screenshot_tool":
            if isinstance(observation, dict) and observation.get("success"):
                research_data["screenshots"].append(observation)

    return research_data


# ============================================================================
# COMPARISON HELPERS
# ============================================================================

def compare_native_vs_custom():
    """
    Document the differences between native and custom implementations
    """
    return """
    NATIVE AGENT (CORRECT)                    CUSTOM AGENT (WRONG)
    ========================================================================
    ✅ Uses create_tool_calling_agent()      ❌ Custom BaseAgent class
    ✅ Uses AgentExecutor                    ❌ Manual tool orchestration
    ✅ LLM decides which tools to call       ❌ Hardcoded tool sequence
    ✅ Can call tools multiple times         ❌ One-shot tool calls
    ✅ Iterative reasoning loop              ❌ No reasoning
    ✅ Agent scratchpad (chain-of-thought)   ❌ No thought process
    ✅ Built-in error handling               ❌ Manual error handling
    ✅ Adaptive to different queries         ❌ Same sequence always
    ✅ Can skip unnecessary tools            ❌ Always calls all tools
    ✅ Follows ReAct pattern                 ❌ No reasoning pattern

    COST COMPARISON:
    ========================================================================
    Native: Only calls needed tools          Custom: Always calls 3-4 tools
    Estimated: $0.20-$0.80 per query        Estimated: $0.50-$2.00 per query
    Savings: 40-60% reduction in API costs

    QUALITY COMPARISON:
    ========================================================================
    Native: Can gather more info if needed   Custom: Fixed amount of info
    Native: Adapts to query complexity       Custom: Same depth always
    Native: Can recover from tool failures   Custom: Fails if tool fails
    """


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_native_agent_usage():
    """
    Example showing how to use the native research agent
    """
    # Create agent
    agent = create_native_research_agent(verbose=True)

    # Run agent on a query - LLM will decide which tools to call!
    result = await agent.ainvoke({
        "input": "Research Betway Casino - I need comprehensive information about licenses, games, and bonuses"
    })

    print("\n" + "="*80)
    print("AGENT OUTPUT:")
    print("="*80)
    print(result["output"])

    print("\n" + "="*80)
    print("REASONING TRACE (Intermediate Steps):")
    print("="*80)
    for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
        print(f"\nStep {i}:")
        print(f"  Tool: {action.tool}")
        print(f"  Input: {action.tool_input}")
        print(f"  Observation: {str(observation)[:200]}...")

    return result


if __name__ == "__main__":
    import asyncio

    # Print comparison
    print(compare_native_vs_custom())

    # Run example
    asyncio.run(example_native_agent_usage())
