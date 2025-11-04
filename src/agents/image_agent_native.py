"""
Native LangChain Image Agent
Uses create_tool_calling_agent() for intelligent image acquisition and upload

This is the CORRECT way to implement the Image Agent using native LangChain components.
"""

import logging
from typing import Any, Dict, List, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from .tools.image_tools import (
    image_search_tool,
    image_selection_tool,
    alt_text_generation_tool,
    wordpress_image_upload_tool
)
from .state import ArticleCMSState

logger = logging.getLogger(__name__)


def create_native_image_agent(
    llm: Optional[ChatOpenAI] = None,
    max_images: int = 5,
    upload_to_wordpress: bool = True,
    verbose: bool = True,
    max_iterations: int = 10,
):
    """
    Create a NATIVE LangChain image agent using create_react_agent()

    The agent will:
    - Analyze content to determine image needs
    - Search for relevant images
    - Select best images for context
    - Generate SEO-optimized alt text
    - Upload to WordPress if needed

    Args:
        llm: Language model (defaults to GPT-4o-mini)
        max_images: Maximum images to acquire
        upload_to_wordpress: Whether to enable WordPress upload tool
        verbose: Whether to log agent reasoning steps
        max_iterations: Maximum reasoning iterations

    Returns:
        Compiled LangGraph agent image tools
    """
    # Default LLM
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Assemble tools - LLM will choose which to use!
    tools = [
        image_search_tool,
        image_selection_tool,
        alt_text_generation_tool,
    ]

    if upload_to_wordpress:
        tools.append(wordpress_image_upload_tool)

    # Create agent prompt with image strategy
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert image acquisition agent specialized in finding, selecting, and preparing images for casino content.

**Available Tools:**
- image_search_tool: Search for images using DataForSEO (casino screenshots, logos, game images)
- image_selection_tool: Intelligently select the best images from search results
- alt_text_generation_tool: Generate SEO-optimized alt text for images
- wordpress_image_upload_tool: Upload images to WordPress media library

**Image Strategy:**
1. **Analyze**: Understand what types of images would enhance the content
2. **Search**: Use image_search_tool to find relevant images
3. **Select**: Use image_selection_tool to choose best images (max {max_images})
4. **Alt Text**: Generate descriptive alt text with alt_text_generation_tool
5. **Upload** (optional): Upload to WordPress with wordpress_image_upload_tool

**Image Types for Casino Content:**
- Casino homepage screenshots (show layout, design, features)
- Game thumbnails (popular slots, table games)
- Casino logos (branding, recognition)
- Bonus/promotion graphics (welcome offers, special deals)
- Payment method icons (deposit/withdrawal options)
- License/certificate badges (trust signals)

**Selection Criteria:**
- Relevance to content topic
- Image quality and clarity
- SEO value (good for search rankings)
- User value (helps readers understand)
- Visual appeal (engaging, professional)
- Appropriate for target audience

**Decision Guidelines:**
- For casino reviews: Homepage screenshot + games + logo (3-5 images)
- For game reviews: Game screenshots + gameplay (2-3 images)
- For bonus guides: Bonus/promotion graphics (1-2 images)
- For payment guides: Payment method icons (3-4 images)
- Always assess content needs before searching

**Quality Standards:**
- Only select high-quality, clear images
- Ensure images are relevant and add value
- Generate descriptive, keyword-rich alt text
- Avoid generic or low-quality stock photos

You can iterate to find the best images for the content."""),

        ("human", "{input}"),

        # CRITICAL: Agent scratchpad for reasoning trace
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create NATIVE agent using LangChain factory
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Wrap in AgentExecutor for execution loop
    # REMOVED - using create_react_agent instead
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info(f"Created native image agent with {len(tools)} tools (max {max_images} images)")
    return agent


async def native_image_node(state: ArticleCMSState) -> ArticleCMSState:
    """
    LangGraph node function using NATIVE image agent

    This replaces the custom ImageAgent.run() method with native agent execution.
    The agent will reason about which images to find and how to acquire them.

    Args:
        state: Current workflow state

    Returns:
        Updated state with images selected and uploaded
    """
    content = state.get("final_content", "") or state.get("draft_content", "")
    query = state.get("query", "")
    target_sites = state.get("target_sites", [])

    if not content:
        logger.error("No content available for image processing")
        state["errors"] = state.get("errors", []) + ["No content for images"]
        return state

    # Determine if WordPress upload is needed
    upload_to_wordpress = len(target_sites) > 0

    # Create native agent
    agent = create_native_image_agent(
        verbose=True,
        max_images=5,
        upload_to_wordpress=upload_to_wordpress
    )

    try:
        logger.info(f"Native Image Agent starting for query: {query}")

        # Prepare input for agent
        content_preview = content[:800] if len(content) > 800 else content
        agent_input = f"""
Find and prepare images for this content:

Topic: {query}

Content Preview:
{content_preview}

Tasks:
1. Determine what types of images would enhance this content
2. Search for relevant, high-quality images
3. Select the best images (max 5)
4. Generate SEO-optimized alt text for each image
{"5. Upload images to WordPress media library" if upload_to_wordpress else ""}

Please acquire images that add value and enhance reader understanding.
"""

        # Execute agent - LLM will decide which tools to call!
        result = await agent.ainvoke({
            "input": agent_input,
        })

        # Extract results
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse intermediate steps to extract images
        image_data = _extract_image_data_from_steps(intermediate_steps)

        # Update state
        state["images"] = image_data.get("images", [])
        state["wordpress_media_ids"] = image_data.get("wordpress_media_ids", [])
        state["image_alt_texts"] = image_data.get("alt_texts", {})
        state["image_output"] = output
        state["image_intermediate_steps"] = intermediate_steps
        state["workflow_step"] = state.get("workflow_step", 0) + 1
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["image_agent"] = "completed"

        images_found = len(image_data.get("images", []))
        logger.info(f"Native Image Agent completed - acquired {images_found} images")

    except Exception as e:
        logger.error(f"Native Image Agent failed: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"Image agent error: {str(e)}"]
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["image_agent"] = "failed"

    return state


def _extract_image_data_from_steps(intermediate_steps: list) -> Dict[str, Any]:
    """
    Extract images and metadata from agent's intermediate steps

    Args:
        intermediate_steps: List of (AgentAction, observation) tuples

    Returns:
        Dictionary with image data
    """
    image_data = {
        "images": [],
        "wordpress_media_ids": [],
        "alt_texts": {},
        "tool_calls": []
    }

    for action, observation in intermediate_steps:
        tool_name = action.tool
        tool_input = action.tool_input

        # Track all tool calls
        image_data["tool_calls"].append({
            "tool": tool_name,
            "input": tool_input,
            "output": observation
        })

        # Extract by tool type
        if tool_name == "image_search_tool":
            # Image search results
            if isinstance(observation, dict):
                search_results = observation.get("images", [])
                logger.debug(f"Found {len(search_results)} images from search")

        elif tool_name == "image_selection_tool":
            # Selected images
            if isinstance(observation, dict):
                selected = observation.get("selected_images", [])
                image_data["images"].extend(selected)

        elif tool_name == "alt_text_generation_tool":
            # Alt text for images
            if isinstance(observation, dict):
                alt_texts = observation.get("alt_texts", {})
                image_data["alt_texts"].update(alt_texts)

        elif tool_name == "wordpress_image_upload_tool":
            # WordPress media IDs
            if isinstance(observation, dict):
                media_id = observation.get("media_id")
                if media_id:
                    image_data["wordpress_media_ids"].append(media_id)

    return image_data


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_native_image_agent():
        """Test the native image agent"""
        agent = create_native_image_agent(verbose=True, upload_to_wordpress=False)

        result = await agent.ainvoke({
            "input": "Find images for a Betway Casino review article about their slot games and welcome bonus"
        })

        print("\n" + "="*80)
        print("AGENT OUTPUT:")
        print("="*80)
        print(result["output"])

    asyncio.run(test_native_image_agent())
