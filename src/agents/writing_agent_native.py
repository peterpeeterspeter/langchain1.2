"""
Native LangChain Writing Agent
Uses create_tool_calling_agent() for LLM-driven content generation

This is the CORRECT way to implement the Writing Agent using native LangChain components.
"""

import logging
from typing import Any, Dict, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
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
):
    """
    Create a NATIVE LangGraph writing agent using create_react_agent()

    The agent will:
    - Analyze the query and research data
    - Decide which writing tools to use
    - Generate content adaptively
    - Refine and optimize based on quality needs

    Args:
        llm: Language model (defaults to GPT-4o-mini)
        enable_refinement: Whether to enable content refinement tool
        enable_seo: Whether to enable SEO optimization tool
        verbose: Whether to log agent reasoning steps (not used)
        max_iterations: Maximum reasoning iterations (not used)

    Returns:
        Compiled LangGraph agent
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

    # Create system message with writing instructions
    system_message = SystemMessage(content="""You are an expert content writing agent specialized in creating high-quality casino reviews and articles.

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

You can iterate and call tools multiple times if needed to achieve high quality.""")

    # Create NATIVE agent using LangGraph factory
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_message
    )

    logger.info(f"Created native writing agent with {len(tools)} tools")
    return agent


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
        from langchain_core.messages import HumanMessage
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=agent_input)]
        })

        # Extract results from messages
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        output = final_message.content if final_message and hasattr(final_message, 'content') else ""

        # Parse messages to extract generated content and metadata
        content_data = _extract_writing_data_from_messages(messages)

        # Update state
        state["draft_content"] = content_data.get("draft_content", output)
        state["final_content"] = content_data.get("refined_content", state["draft_content"])
        state["seo_metadata"] = content_data.get("seo_metadata", {})
        state["template_used"] = content_data.get("template_id", "default")
        state["writing_output"] = output
        state["writing_messages"] = messages
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


def _extract_writing_data_from_messages(messages: list) -> Dict[str, Any]:
    """
    Extract content and metadata from agent's message history

    Args:
        messages: List of messages from agent execution

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

    for message in messages:
        # Check for tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', '')
                tool_input = tool_call.get('args', {})
                writing_data["tool_calls"].append({
                    "tool": tool_name,
                    "input": tool_input
                })

        # Check for tool responses
        if hasattr(message, 'name') and message.name:
            tool_name = message.name
            tool_output = message.content

            # Extract by tool type
            if tool_name == "template_selection_tool":
                if isinstance(tool_output, dict):
                    writing_data["template_id"] = tool_output.get("template_id", "default")

            elif tool_name == "content_generation_tool":
                if isinstance(tool_output, dict):
                    writing_data["draft_content"] = tool_output.get("content", "")

            elif tool_name == "content_refinement_tool":
                if isinstance(tool_output, dict):
                    writing_data["refined_content"] = tool_output.get("refined_content", "")

            elif tool_name == "seo_optimization_tool":
                if isinstance(tool_output, dict):
                    writing_data["seo_metadata"] = tool_output.get("seo_metadata", {})

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
