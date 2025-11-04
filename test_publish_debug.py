#!/usr/bin/env python3
"""
Debug publishing and save the article content
"""

import asyncio
import logging
import os
import json
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
    """Run workflow and debug publishing"""
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage
    from src.agents.state import create_initial_state, ArticleCMSState
    from src.agents.research_agent_native import create_native_research_agent
    from src.agents.writing_agent_native import create_native_writing_agent
    from src.agents.publishing_agent_native import create_native_publishing_agent

    logger.info("="*80)
    logger.info("PUBLISHING DEBUG TEST")
    logger.info("="*80)

    # Create Claude LLM
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=4096
    )

    # Quick workflow - just writing and publishing
    async def writing_node(state):
        agent = create_native_writing_agent(llm=llm, verbose=True)
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"""Write a SHORT test article about: {state['query']}

Keep it brief - about 300 words for testing.
Include proper HTML formatting.
""")]
        })

        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None

        if final_message and hasattr(final_message, 'content'):
            state["final_content"] = final_message.content
            state["seo_metadata"] = {"title": f"{state['query']}"}
            logger.info(f"✅ Article written: {len(final_message.content)} chars")

        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["writing_agent"] = "completed"
        return state

    async def publishing_node(state):
        logger.info(f"\n{'='*80}")
        logger.info("PUBLISHING NODE - DEBUG INFO")
        logger.info(f"{'='*80}")
        logger.info(f"Target sites: {state.get('target_sites')}")
        logger.info(f"Content length: {len(state.get('final_content', ''))} chars")
        logger.info(f"Title: {state.get('seo_metadata', {}).get('title')}")

        agent = create_native_publishing_agent(llm=llm, verbose=True)

        content = state.get('final_content', '')
        title = state.get('seo_metadata', {}).get('title', state['query'])

        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"""Publish this article to WordPress.

Target site: crashcasino-io

Title: {title}

Content:
{content}

Steps:
1. Query site_registry_tool to get the site configuration for crashcasino-io
2. Use wordpress_publish_tool to publish the article
3. Return the post URL
""")]
        })

        logger.info(f"\n{'='*80}")
        logger.info("PUBLISHING RESULT MESSAGES")
        logger.info(f"{'='*80}")

        messages = result.get("messages", [])
        for i, msg in enumerate(messages):
            logger.info(f"\nMessage {i+1}:")
            logger.info(f"  Type: {type(msg).__name__}")
            if hasattr(msg, 'content'):
                logger.info(f"  Content: {str(msg.content)[:200]}...")
            if hasattr(msg, 'name'):
                logger.info(f"  Name: {msg.name}")
            if hasattr(msg, 'tool_calls'):
                logger.info(f"  Tool calls: {msg.tool_calls}")

        # Try to extract publishing result
        published_posts = []
        for msg in messages:
            if hasattr(msg, 'name') and msg.name == 'wordpress_publish_tool':
                logger.info(f"\n📝 Found WordPress publish tool response:")
                logger.info(f"   Content: {msg.content}")

                # Try to parse JSON response
                try:
                    if isinstance(msg.content, str):
                        data = json.loads(msg.content)
                        logger.info(f"   Parsed data: {data}")

                        if data.get("success"):
                            published_posts.append({
                                "site_id": data.get("site_id", "crashcasino-io"),
                                "post_url": data.get("post_url"),
                                "post_id": data.get("post_id")
                            })
                            logger.info(f"   ✅ Success! URL: {data.get('post_url')}")
                        else:
                            logger.error(f"   ❌ Failed: {data.get('error')}")
                except Exception as e:
                    logger.error(f"   Error parsing response: {e}")

        state["published_posts"] = published_posts
        state["agent_statuses"] = state.get("agent_statuses", {})
        state["agent_statuses"]["publishing_agent"] = "completed" if published_posts else "failed"

        return state

    # Build minimal workflow
    workflow = StateGraph(ArticleCMSState)
    workflow.add_node("writing", writing_node)
    workflow.add_node("publishing", publishing_node)
    workflow.set_entry_point("writing")
    workflow.add_edge("writing", "publishing")
    workflow.add_edge("publishing", END)
    app = workflow.compile()

    # Run
    initial_state = create_initial_state(
        query="Betway Casino Review - Quick Test",
        target_sites=["crashcasino-io"]
    )

    logger.info("\n🚀 Starting debug workflow...\n")
    start_time = datetime.now()

    result = await app.ainvoke(initial_state)

    duration = (datetime.now() - start_time).total_seconds()

    # Results
    logger.info(f"\n{'='*80}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Content: {len(result.get('final_content', ''))} chars")

    published_posts = result.get("published_posts", [])
    if published_posts:
        logger.info(f"\n✅ PUBLISHED!")
        for post in published_posts:
            logger.info(f"   URL: {post.get('post_url')}")
            logger.info(f"   ID: {post.get('post_id')}")
            print(f"\n🎉 Article published: {post.get('post_url')}\n")
    else:
        logger.warning(f"\n⚠️  Publishing failed")

        # Save article to file
        if result.get('final_content'):
            filename = f"betway_article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(filename, 'w') as f:
                f.write(f"<h1>{result.get('seo_metadata', {}).get('title')}</h1>\n\n")
                f.write(result.get('final_content'))
            logger.info(f"\n💾 Article saved to: {filename}")
            print(f"\n💾 Article saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())
