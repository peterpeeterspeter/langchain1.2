"""
Native LangChain Publishing Agent
Uses create_tool_calling_agent() for intelligent multi-site WordPress publishing

This is the CORRECT way to implement the Publishing Agent using native LangChain components.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from .tools.publishing_tools import (
    wordpress_publish_tool,
    site_registry_tool,
    content_adaptation_tool
)
from .state import ArticleCMSState

logger = logging.getLogger(__name__)


def create_native_publishing_agent(
    llm: Optional[ChatOpenAI] = None,
    verbose: bool = True,
    max_iterations: int = 15,  # Higher for multi-site publishing
) -> AgentExecutor:
    """
    Create a NATIVE LangChain publishing agent using create_tool_calling_agent()

    The agent will:
    - Query site registry for target sites
    - Adapt content for each site if needed
    - Publish to WordPress sites
    - Handle errors and retry logic

    Args:
        llm: Language model (defaults to GPT-4o-mini)
        verbose: Whether to log agent reasoning steps
        max_iterations: Maximum reasoning iterations (higher for multi-site)

    Returns:
        AgentExecutor configured with publishing tools
    """
    # Default LLM
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)  # Low temp for consistency

    # Assemble tools - LLM will choose which to use!
    tools = [
        site_registry_tool,
        content_adaptation_tool,
        wordpress_publish_tool,
    ]

    # Create agent prompt with publishing strategy
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert WordPress publishing agent specialized in multi-site content distribution.

**Available Tools:**
- site_registry_tool: Query the WordPress site registry to get site configurations
- content_adaptation_tool: Adapt content for specific site requirements (style, tone, formatting)
- wordpress_publish_tool: Publish content to a WordPress site via REST API

**Publishing Strategy:**
1. **Identify**: Use site_registry_tool to get configurations for target sites
2. **Plan**: Determine which sites need content adaptation
3. **Adapt** (if needed): Use content_adaptation_tool to customize content per site
4. **Publish**: Use wordpress_publish_tool to publish to each target site
5. **Verify**: Check publishing results and retry if needed

**Multi-Site Publishing:**
- Each site may have different requirements (categories, tags, formatting)
- Some sites may need content adaptation (tone, style, length)
- Handle each site independently - one failure shouldn't stop others
- Track success/failure for each site

**Content Adaptation Scenarios:**
- Different brand voices (formal vs casual)
- Different content lengths (full vs summary)
- Different formatting styles (HTML vs plain text)
- Different SEO requirements (keywords, meta descriptions)

**Publishing Best Practices:**
- Always query site registry first to get current configurations
- Adapt content only when site requires it (saves time)
- Publish to sites sequentially for reliability
- Handle errors gracefully and continue with remaining sites
- Log detailed results for each site

**Error Handling:**
- If a site fails, note the error and continue with others
- Don't let one failure block all publishing
- Provide clear error messages for debugging

**Decision Making:**
- For sites with "content_adaptation: true": Use content_adaptation_tool first
- For sites with "content_adaptation: false": Publish original content directly
- Assess each site's needs independently

You should iterate through target sites and publish to each one."""),

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

    logger.info(f"Created native publishing agent with {len(tools)} tools")
    return agent_executor


async def native_publishing_node(state: ArticleCMSState) -> ArticleCMSState:
    """
    LangGraph node function using NATIVE publishing agent

    This replaces the custom PublishingAgent.run() method with native agent execution.
    The agent will reason about how to publish to each target site.

    Args:
        state: Current workflow state

    Returns:
        Updated state with publishing results
    """
    content = state.get("final_content", "")
    title = state.get("seo_metadata", {}).get("title", state.get("query", "Untitled"))
    target_sites = state.get("target_sites", [])
    images = state.get("images", [])
    wordpress_media_ids = state.get("wordpress_media_ids", [])

    if not content:
        logger.error("No content available for publishing")
        state["errors"] = state.get("errors", []) + ["No content to publish"]
        return state

    if not target_sites:
        logger.warning("No target sites specified - skipping publishing")
        state["published_posts"] = []
        state["site_statuses"] = {}
        state["workflow_step"] = state.get("workflow_step", 0) + 1
        return state

    # Create native agent
    agent = create_native_publishing_agent(verbose=True)

    try:
        logger.info(f"Native Publishing Agent starting for {len(target_sites)} sites")

        # Get featured image (first WordPress media ID)
        featured_image_id = wordpress_media_ids[0] if wordpress_media_ids else None

        # Prepare input for agent
        agent_input = f"""
Publish this content to the target WordPress sites:

Title: {title}
Target Sites: {', '.join(target_sites)}
Featured Image ID: {featured_image_id or 'None'}

Content Preview:
{content[:500]}...

Tasks:
1. Query site registry to get configurations for these sites: {target_sites}
2. For each site:
   - Check if content adaptation is needed
   - Adapt content if required
   - Publish to WordPress
   - Track success/failure
3. Provide detailed results for each site

Please publish to all target sites, handling each independently.
"""

        # Execute agent - LLM will decide which tools to call!
        result = await agent.ainvoke({
            "input": agent_input,
        })

        # Extract results
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse intermediate steps to extract publishing results
        publishing_data = _extract_publishing_data_from_steps(intermediate_steps, target_sites)

        # Update state
        state["published_posts"] = publishing_data.get("published_posts", [])
        state["site_statuses"] = publishing_data.get("site_statuses", {})
        state["post_urls"] = publishing_data.get("post_urls", {})
        state["publishing_output"] = output
        state["publishing_intermediate_steps"] = intermediate_steps
        state["workflow_step"] = state.get("workflow_step", 0) + 1
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["publishing_agent"] = "completed"

        sites_published = len([s for s in publishing_data.get("site_statuses", {}).values() if s == "success"])
        logger.info(f"Native Publishing Agent completed - published to {sites_published}/{len(target_sites)} sites")

    except Exception as e:
        logger.error(f"Native Publishing Agent failed: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"Publishing agent error: {str(e)}"]
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["publishing_agent"] = "failed"

    return state


def _extract_publishing_data_from_steps(intermediate_steps: list, target_sites: List[str]) -> Dict[str, Any]:
    """
    Extract publishing results from agent's intermediate steps

    Args:
        intermediate_steps: List of (AgentAction, observation) tuples
        target_sites: List of target site IDs

    Returns:
        Dictionary with publishing data
    """
    publishing_data = {
        "published_posts": [],
        "site_statuses": {},
        "post_urls": {},
        "site_configs": {},
        "tool_calls": []
    }

    # Initialize all target sites as pending
    for site_id in target_sites:
        publishing_data["site_statuses"][site_id] = "pending"

    for action, observation in intermediate_steps:
        tool_name = action.tool
        tool_input = action.tool_input

        # Track all tool calls
        publishing_data["tool_calls"].append({
            "tool": tool_name,
            "input": tool_input,
            "output": observation
        })

        # Extract by tool type
        if tool_name == "site_registry_tool":
            # Site configurations
            if isinstance(observation, dict):
                sites = observation.get("sites", [])
                for site in sites:
                    site_id = site.get("site_id")
                    if site_id:
                        publishing_data["site_configs"][site_id] = site

        elif tool_name == "content_adaptation_tool":
            # Content adapted for specific site
            if isinstance(observation, dict):
                site_id = tool_input.get("site_config", {}).get("site_id")
                if site_id:
                    logger.debug(f"Content adapted for site: {site_id}")

        elif tool_name == "wordpress_publish_tool":
            # Publishing result
            if isinstance(observation, dict):
                site_id = tool_input.get("site_id")
                success = observation.get("success", False)
                post_id = observation.get("post_id")
                post_url = observation.get("post_url")
                error = observation.get("error")

                if site_id:
                    # Update site status
                    publishing_data["site_statuses"][site_id] = "success" if success else "failed"

                    if success and post_id:
                        # Add to published posts
                        publishing_data["published_posts"].append({
                            "site_id": site_id,
                            "post_id": post_id,
                            "post_url": post_url
                        })

                        # Store post URL
                        if post_url:
                            publishing_data["post_urls"][site_id] = post_url

                    elif error:
                        logger.error(f"Publishing failed for {site_id}: {error}")

    return publishing_data


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_native_publishing_agent():
        """Test the native publishing agent"""
        agent = create_native_publishing_agent(verbose=True)

        result = await agent.ainvoke({
            "input": "Publish content to sites: coinflip-casino, bitcoin-casino. Title: 'Betway Casino Review'. Content: 'Betway is a great casino...'"
        })

        print("\n" + "="*80)
        print("AGENT OUTPUT:")
        print("="*80)
        print(result["output"])

    asyncio.run(test_native_publishing_agent())
