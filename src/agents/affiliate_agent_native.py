"""
Native LangChain Affiliate Agent
Uses create_tool_calling_agent() for intelligent affiliate link insertion

This is the CORRECT way to implement the Affiliate Agent using native LangChain components.
"""

import logging
from typing import Any, Dict, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from .tools.affiliate_tools import (
    affiliate_link_database_tool,
    link_insertion_tool,
    link_validation_tool,
    tracking_parameter_tool
)
from .state import ArticleCMSState

logger = logging.getLogger(__name__)


def create_native_affiliate_agent(
    llm: Optional[ChatOpenAI] = None,
    max_links_per_article: int = 5,
    verbose: bool = True,
    max_iterations: int = 10,
) -> AgentExecutor:
    """
    Create a NATIVE LangChain affiliate agent using create_tool_calling_agent()

    The agent will:
    - Analyze content for affiliate link opportunities
    - Query affiliate database for relevant links
    - Insert links contextually and naturally
    - Validate and add tracking parameters

    Args:
        llm: Language model (defaults to GPT-4o-mini)
        max_links_per_article: Maximum affiliate links per article
        verbose: Whether to log agent reasoning steps
        max_iterations: Maximum reasoning iterations

    Returns:
        AgentExecutor configured with affiliate tools
    """
    # Default LLM
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Assemble tools - LLM will choose which to use!
    tools = [
        affiliate_link_database_tool,
        link_insertion_tool,
        link_validation_tool,
        tracking_parameter_tool
    ]

    # Create agent prompt with affiliate strategy
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert affiliate marketing agent specialized in inserting affiliate links contextually and naturally into content.

**Available Tools:**
- affiliate_link_database_tool: Query the affiliate link database by category, casino name, or keywords
- link_insertion_tool: Insert affiliate links into content at appropriate positions
- link_validation_tool: Verify that affiliate links are valid and working
- tracking_parameter_tool: Add UTM tracking parameters to links for analytics

**Affiliate Strategy:**
1. **Analyze**: Review the content to identify natural affiliate link opportunities
2. **Query**: Use affiliate_link_database_tool to find relevant affiliate links
3. **Select**: Choose the most appropriate links (max {max_links_per_article} per article)
4. **Insert**: Use link_insertion_tool to add links contextually
5. **Validate** (optional): Verify important links with link_validation_tool
6. **Track** (optional): Add tracking parameters for analytics if needed

**Guidelines:**
- Insert links naturally where they add value to the reader
- Avoid over-saturating content with links (max {max_links_per_article})
- Prefer contextual placement over forced insertion
- Use anchor text that flows naturally with the content
- Maintain content quality and readability

**Link Placement Best Practices:**
- Place links where readers naturally look for related information
- Use descriptive anchor text that indicates destination
- Distribute links throughout content (not all at top)
- Ensure links are relevant to surrounding context
- Never compromise content quality for link placement

**Decision Making:**
- For casino reviews: Query database for that specific casino's affiliate program
- For comparison articles: May need links for multiple casinos
- For informational content: Be selective, only add highly relevant links
- Always assess if a link adds value before inserting

You can iterate and refine link placement for optimal results."""),

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

    logger.info(f"Created native affiliate agent with {len(tools)} tools (max {max_links_per_article} links/article)")
    return agent_executor


async def native_affiliate_node(state: ArticleCMSState) -> ArticleCMSState:
    """
    LangGraph node function using NATIVE affiliate agent

    This replaces the custom AffiliateAgent.run() method with native agent execution.
    The agent will reason about which affiliate links to insert and where.

    Args:
        state: Current workflow state

    Returns:
        Updated state with affiliate links inserted
    """
    content = state.get("final_content", "") or state.get("draft_content", "")
    query = state.get("query", "")

    if not content:
        logger.error("No content available for affiliate link insertion")
        state["errors"] = state.get("errors", []) + ["No content for affiliate links"]
        return state

    # Create native agent
    agent = create_native_affiliate_agent(verbose=True, max_links_per_article=5)

    try:
        logger.info(f"Native Affiliate Agent starting for query: {query}")

        # Prepare input for agent
        agent_input = f"""
Analyze this content and insert appropriate affiliate links:

Query/Topic: {query}

Content Preview: {content[:500]}...

Please:
1. Identify natural affiliate link opportunities
2. Query the database for relevant affiliate links
3. Insert links contextually (max 5 links)
4. Ensure links enhance rather than detract from content
"""

        # Execute agent - LLM will decide which tools to call!
        result = await agent.ainvoke({
            "input": agent_input,
        })

        # Extract results
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse intermediate steps to extract enhanced content and links
        affiliate_data = _extract_affiliate_data_from_steps(intermediate_steps, content)

        # Update state
        state["final_content"] = affiliate_data.get("enhanced_content", content)
        state["affiliate_links"] = affiliate_data.get("affiliate_links", [])
        state["tracking_codes"] = affiliate_data.get("tracking_codes", {})
        state["affiliate_output"] = output
        state["affiliate_intermediate_steps"] = intermediate_steps
        state["workflow_step"] = state.get("workflow_step", 0) + 1
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["affiliate_agent"] = "completed"

        links_inserted = len(affiliate_data.get("affiliate_links", []))
        logger.info(f"Native Affiliate Agent completed - inserted {links_inserted} links")

    except Exception as e:
        logger.error(f"Native Affiliate Agent failed: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"Affiliate agent error: {str(e)}"]
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["affiliate_agent"] = "failed"

    return state


def _extract_affiliate_data_from_steps(intermediate_steps: list, original_content: str) -> Dict[str, Any]:
    """
    Extract affiliate links and enhanced content from agent's intermediate steps

    Args:
        intermediate_steps: List of (AgentAction, observation) tuples
        original_content: Original content before link insertion

    Returns:
        Dictionary with affiliate data
    """
    affiliate_data = {
        "enhanced_content": original_content,
        "affiliate_links": [],
        "tracking_codes": {},
        "tool_calls": []
    }

    for action, observation in intermediate_steps:
        tool_name = action.tool
        tool_input = action.tool_input

        # Track all tool calls
        affiliate_data["tool_calls"].append({
            "tool": tool_name,
            "input": tool_input,
            "output": observation
        })

        # Extract by tool type
        if tool_name == "affiliate_link_database_tool":
            # Database query - links available for insertion
            if isinstance(observation, dict):
                available_links = observation.get("links", [])
                logger.debug(f"Found {len(available_links)} affiliate links in database")

        elif tool_name == "link_insertion_tool":
            # Link insertion - get enhanced content and insertions
            if isinstance(observation, dict):
                affiliate_data["enhanced_content"] = observation.get("enhanced_content", original_content)
                insertions = observation.get("insertions", [])

                # Extract affiliate link details
                for insertion in insertions:
                    affiliate_data["affiliate_links"].append({
                        "link_id": insertion.get("link_id"),
                        "url": insertion.get("final_url"),
                        "anchor_text": insertion.get("anchor_text"),
                        "position": insertion.get("position", 0)
                    })

                    # Store tracking codes
                    link_id = insertion.get("link_id")
                    final_url = insertion.get("final_url")
                    if link_id and final_url:
                        affiliate_data["tracking_codes"][link_id] = final_url

        elif tool_name == "tracking_parameter_tool":
            # Tracking parameters added
            if isinstance(observation, dict):
                tracked_urls = observation.get("tracked_urls", {})
                affiliate_data["tracking_codes"].update(tracked_urls)

    return affiliate_data


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_native_affiliate_agent():
        """Test the native affiliate agent"""
        agent = create_native_affiliate_agent(verbose=True)

        sample_content = """
        Betway Casino is one of the leading online casinos in the industry.
        They offer a wide range of games and excellent customer support.
        New players can claim a generous welcome bonus.
        """

        result = await agent.ainvoke({
            "input": f"Insert affiliate links into this Betway Casino content:\n\n{sample_content}"
        })

        print("\n" + "="*80)
        print("AGENT OUTPUT:")
        print("="*80)
        print(result["output"])

    asyncio.run(test_native_affiliate_agent())
