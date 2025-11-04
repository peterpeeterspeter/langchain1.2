"""
Native LangChain Writing Agent
Uses create_tool_calling_agent() for LLM-driven content generation

This is the CORRECT way to implement the Writing Agent using native LangChain components.
"""

import logging
from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .tools.writing_tools import (
    content_generation_tool,
    template_selection_tool,
    content_refinement_tool,
    seo_optimization_tool
)
from .state import ArticleCMSState

logger = logging.getLogger(__name__)


def create_native_writing_agent(
    llm: Optional[ChatOpenAI] = None,
    enable_refinement: bool = True,
    enable_seo: bool = True,
    verbose: bool = True,
    max_iterations: int = 10,
) -> AgentExecutor:
    """
    Create a NATIVE LangChain writing agent using create_tool_calling_agent()

    The agent will:
    - Analyze the query and research data
    - Decide which writing tools to use
    - Generate content adaptively
    - Refine and optimize based on quality needs

    Args:
        llm: Language model (defaults to GPT-4o-mini)
        enable_refinement: Whether to enable content refinement tool
        enable_seo: Whether to enable SEO optimization tool
        verbose: Whether to log agent reasoning steps
        max_iterations: Maximum reasoning iterations

    Returns:
        AgentExecutor configured with writing tools
    """
    # Default LLM
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Higher temp for creative writing

    # Assemble tools - LLM will choose which to use!
    tools = [
        template_selection_tool,
        content_generation_tool,
    ]

    if enable_refinement:
        tools.append(content_refinement_tool)

    if enable_seo:
        tools.append(seo_optimization_tool)

    # Create agent prompt with writing instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content writing agent specialized in creating high-quality casino reviews and articles.

**Available Tools:**
- template_selection_tool: Choose the appropriate content template based on query type
- content_generation_tool: Generate content using Universal RAG Chain
- content_refinement_tool: Refine and improve content quality (structure, readability, engagement)
- seo_optimization_tool: Optimize content for SEO with metadata

**Writing Strategy:**
1. **Analyze**: Understand the query type and target audience
2. **Template**: Select appropriate template using template_selection_tool
3. **Generate**: Create initial content using content_generation_tool with research data
4. **Evaluate**: Assess content quality and decide if refinement is needed
5. **Refine** (optional): If quality can be improved, use content_refinement_tool
6. **Optimize** (optional): If SEO is important, use seo_optimization_tool

**Decision Guidelines:**
- For quick overviews: template + generation may be sufficient
- For comprehensive reviews: use all tools for highest quality
- For SEO-focused content: always use seo_optimization_tool
- Assess each output and decide next steps dynamically

**Quality Standards:**
- Content should be informative, accurate, and engaging
- Maintain consistent tone and voice
- Include relevant information from research data
- Optimize for readability and user experience

You can iterate and call tools multiple times if needed to achieve high quality."""),

        ("human", "{input}"),

        # CRITICAL: Agent scratchpad for reasoning trace
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create NATIVE agent using LangChain factory
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Wrap in AgentExecutor for execution loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info(f"Created native writing agent with {len(tools)} tools")
    return agent_executor


async def native_writing_node(state: ArticleCMSState) -> ArticleCMSState:
    """
    LangGraph node function using NATIVE writing agent

    This replaces the custom WritingAgent.run() method with native agent execution.
    The agent will reason about which writing tools to call based on the query and research data.

    Args:
        state: Current workflow state

    Returns:
        Updated state with generated content
    """
    query = state.get("query", "")
    research_data = state.get("research_data", {})

    if not query:
        logger.error("No query provided for writing")
        state["errors"] = state.get("errors", []) + ["No query provided"]
        return state

    # Create native agent
    agent = create_native_writing_agent(verbose=True)

    try:
        logger.info(f"Native Writing Agent starting for query: {query}")

        # Prepare input for agent - include research data context
        research_summary = _summarize_research(research_data)
        agent_input = f"""
Generate high-quality content for: {query}

Research Data Available:
{research_summary}

Please create engaging, accurate content that incorporates the research findings.
"""

        # Execute agent - LLM will decide which tools to call!
        result = await agent.ainvoke({
            "input": agent_input,
        })

        # Extract results
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse intermediate steps to extract generated content and metadata
        content_data = _extract_writing_data_from_steps(intermediate_steps)

        # Update state
        state["draft_content"] = content_data.get("draft_content", output)
        state["final_content"] = content_data.get("refined_content", state["draft_content"])
        state["seo_metadata"] = content_data.get("seo_metadata", {})
        state["template_used"] = content_data.get("template_id", "default")
        state["writing_output"] = output
        state["writing_intermediate_steps"] = intermediate_steps
        state["workflow_step"] = state.get("workflow_step", 0) + 1
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["writing_agent"] = "completed"

        logger.info(f"Native Writing Agent completed - {len(intermediate_steps)} tool calls")

    except Exception as e:
        logger.error(f"Native Writing Agent failed: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"Writing agent error: {str(e)}"]
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["writing_agent"] = "failed"

    return state


def _summarize_research(research_data: Dict) -> str:
    """Create a concise summary of research data for agent input"""
    summary_parts = []

    web_results = research_data.get("web_search_results", [])
    if web_results:
        summary_parts.append(f"- Web search found {len(web_results)} relevant sources")

    comprehensive = research_data.get("comprehensive_research", {})
    if comprehensive:
        summary_parts.append(f"- Comprehensive research with {len(comprehensive)} data points")

    structured = research_data.get("structured_intelligence", {})
    if structured:
        summary_parts.append(f"- Structured casino data extracted")

    if not summary_parts:
        return "No research data available"

    return "\n".join(summary_parts)


def _extract_writing_data_from_steps(intermediate_steps: list) -> Dict[str, Any]:
    """
    Extract content and metadata from agent's intermediate steps

    Args:
        intermediate_steps: List of (AgentAction, observation) tuples

    Returns:
        Dictionary with extracted writing data
    """
    writing_data = {
        "template_id": "default",
        "draft_content": "",
        "refined_content": "",
        "seo_metadata": {},
        "tool_calls": []
    }

    for action, observation in intermediate_steps:
        tool_name = action.tool
        tool_input = action.tool_input

        # Track all tool calls
        writing_data["tool_calls"].append({
            "tool": tool_name,
            "input": tool_input,
            "output": observation
        })

        # Extract by tool type
        if tool_name == "template_selection_tool":
            if isinstance(observation, dict):
                writing_data["template_id"] = observation.get("template_id", "default")

        elif tool_name == "content_generation_tool":
            if isinstance(observation, dict):
                writing_data["draft_content"] = observation.get("content", "")

        elif tool_name == "content_refinement_tool":
            if isinstance(observation, dict):
                writing_data["refined_content"] = observation.get("refined_content", "")

        elif tool_name == "seo_optimization_tool":
            if isinstance(observation, dict):
                writing_data["seo_metadata"] = observation.get("seo_metadata", {})

    # If no refined content, use draft
    if not writing_data["refined_content"] and writing_data["draft_content"]:
        writing_data["refined_content"] = writing_data["draft_content"]

    return writing_data


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_native_writing_agent():
        """Test the native writing agent"""
        agent = create_native_writing_agent(verbose=True)

        result = await agent.ainvoke({
            "input": "Generate a casino review for Betway Casino based on available research"
        })

        print("\n" + "="*80)
        print("AGENT OUTPUT:")
        print("="*80)
        print(result["output"])

        print("\n" + "="*80)
        print("REASONING TRACE:")
        print("="*80)
        for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
            print(f"\nStep {i}:")
            print(f"  Tool: {action.tool}")
            print(f"  Input: {action.tool_input}")
            print(f"  Output: {str(observation)[:200]}...")

    asyncio.run(test_native_writing_agent())
