#!/usr/bin/env python3
"""
Publish article using Claude (Anthropic) instead of OpenAI
"""

import asyncio
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run the CMS workflow using Claude"""
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import StateGraph, END
    from src.agents.state import create_initial_state
    from src.agents.research_agent_native import create_native_research_agent
    from src.agents.writing_agent_native import create_native_writing_agent
    from src.agents.affiliate_agent_native import create_native_affiliate_agent
    from src.agents.image_agent_native import create_native_image_agent
    from src.agents.publishing_agent_native import create_native_publishing_agent

    logger.info("="*80)
    logger.info("RUNNING ARTICLE CMS WORKFLOW WITH CLAUDE")
    logger.info("="*80)

    # Check Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not found. Please configure .env file.")
        return

    logger.info("✅ Anthropic API key found")

    # Create Claude LLM
    try:
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",  # Use Claude 3.5 Haiku
            temperature=0.7,
            max_tokens=4096
        )
        logger.info("✅ Created Claude 3.5 Haiku LLM")
    except Exception as e:
        logger.error(f"Failed to create Claude LLM: {e}")
        return

    # Create agents with Claude LLM
    logger.info("\n🤖 Creating agents with Claude...")

    async def research_node(state):
        from src.agents.research_agent_native import native_research_node
        from langchain_core.messages import HumanMessage

        agent = create_native_research_agent(llm=llm, verbose=True)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Research this topic: {state['query']}")]
        })
        state["comprehensive_research"] = "Research completed with Claude"
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["research_agent"] = "completed"
        return state

    async def writing_node(state):
        from langchain_core.messages import HumanMessage

        agent = create_native_writing_agent(llm=llm, verbose=True)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"""Write a comprehensive article about: {state['query']}

Please create high-quality content that is:
- Well-structured and informative
- SEO-optimized
- 1500-2000 words
- Engaging and professional

Topic: {state['query']}
""")]
        })

        # Extract content from result
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None

        if final_message and hasattr(final_message, 'content'):
            state["final_content"] = final_message.content
            state["seo_metadata"] = {"title": f"{state['query']} - Complete Guide"}
        else:
            state["final_content"] = f"Article about {state['query']}"
            state["seo_metadata"] = {"title": state['query']}

        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["writing_agent"] = "completed"
        return state

    async def affiliate_node(state):
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["affiliate_agent"] = "skipped"
        state["inserted_links"] = []
        return state

    async def image_node(state):
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["image_agent"] = "skipped"
        state["images"] = []
        state["wordpress_media_ids"] = []
        return state

    async def publishing_node(state):
        from langchain_core.messages import HumanMessage

        if not state.get("final_content"):
            logger.error("No content to publish")
            state["errors"] = state.get("errors", []) + ["No content to publish"]
            return state

        agent = create_native_publishing_agent(llm=llm, verbose=True)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"""Publish this content to WordPress site: {state['target_sites']}

Title: {state.get('seo_metadata', {}).get('title', state['query'])}

Content:
{state['final_content']}

Please:
1. Query the site registry for site configuration
2. Publish to WordPress
3. Return the post URL
""")]
        })

        # Extract publishing results
        messages = result.get("messages", [])
        published_posts = []
        post_urls = {}

        for msg in messages:
            if hasattr(msg, 'name') and msg.name == 'wordpress_publish_tool':
                content = msg.content
                if isinstance(content, str):
                    # Parse the response
                    import json
                    try:
                        data = json.loads(content)
                        if data.get("success"):
                            site_id = data.get("site_id", state['target_sites'][0])
                            post_url = data.get("post_url")
                            published_posts.append({
                                "site_id": site_id,
                                "post_url": post_url,
                                "post_id": data.get("post_id")
                            })
                            if post_url:
                                post_urls[site_id] = post_url
                    except:
                        pass

        state["published_posts"] = published_posts
        state["post_urls"] = post_urls
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["publishing_agent"] = "completed" if published_posts else "failed"

        return state

    # Build workflow graph
    from src.agents.state import ArticleCMSState

    workflow = StateGraph(ArticleCMSState)
    workflow.add_node("research", research_node)
    workflow.add_node("writing", writing_node)
    workflow.add_node("affiliate", affiliate_node)
    workflow.add_node("image", image_node)
    workflow.add_node("publishing", publishing_node)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "writing")
    workflow.add_edge("writing", "affiliate")
    workflow.add_edge("affiliate", "image")
    workflow.add_edge("image", "publishing")
    workflow.add_edge("publishing", END)

    app = workflow.compile()

    logger.info("✅ Workflow graph created")

    # Define query
    query = "Betway Casino Review - Complete 2024 Guide"
    target_sites = ["crashcasino-io"]

    logger.info(f"\n📝 Input:")
    logger.info(f"   Query: {query}")
    logger.info(f"   Target Sites: {target_sites}")

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        target_sites=target_sites
    )

    # Run workflow
    logger.info("\n🚀 Starting Claude-powered workflow...\n")
    start_time = datetime.now()

    try:
        result = await app.ainvoke(initial_state)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print results
        logger.info("\n" + "="*80)
        logger.info("📊 WORKFLOW COMPLETE")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f} seconds")

        # Check for errors
        if result.get("errors"):
            logger.error(f"\n❌ Errors occurred:")
            for error in result["errors"]:
                logger.error(f"   - {error}")

        # Check published posts
        published_posts = result.get("published_posts", [])
        post_urls = result.get("post_urls", {})

        if published_posts:
            logger.info(f"\n✅ Successfully published to {len(published_posts)} sites:")
            for post in published_posts:
                site_id = post.get("site_id")
                post_id = post.get("post_id")
                post_url = post.get("post_url")
                logger.info(f"\n   📄 {site_id}:")
                logger.info(f"      Post ID: {post_id}")
                logger.info(f"      URL: {post_url}")
        else:
            logger.warning("\n⚠️  No posts were published")

        # Print article details
        logger.info(f"\n📖 Article Details:")
        logger.info(f"   Title: {result.get('seo_metadata', {}).get('title', 'N/A')}")
        logger.info(f"   Content length: {len(result.get('final_content', ''))} characters")
        logger.info(f"   Word count: {len(result.get('final_content', '').split())} words")

        # Agent statuses
        logger.info(f"\n🤖 Agent Statuses:")
        for agent, status in result.get("agent_statuses", {}).items():
            status_icon = "✅" if status == "completed" else "⚠️" if status == "skipped" else "❌"
            logger.info(f"   {status_icon} {agent}: {status}")

        logger.info("\n" + "="*80)

        if published_posts:
            print("\n\n🎉 ARTICLE PUBLISHED SUCCESSFULLY! 🎉\n")
            for site_id, url in post_urls.items():
                print(f"📄 View your article at: {url}\n")
        else:
            print("\n⚠️  Article was written but not published. Check logs for details.\n")

    except Exception as e:
        logger.error(f"\n❌ Workflow failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
