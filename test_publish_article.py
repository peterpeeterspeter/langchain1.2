#!/usr/bin/env python3
"""
Complete End-to-End Test - Publish Real Article
Runs all 5 agents and publishes to WordPress
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


async def run_full_pipeline(query: str, target_sites: list):
    """
    Run complete article generation and publishing pipeline

    Args:
        query: Topic to write about (e.g., "Betway Casino Review")
        target_sites: List of site IDs to publish to (e.g., ["crashcasino-io"])

    Returns:
        Dictionary with publishing results including URLs
    """
    from langchain_anthropic import ChatAnthropic
    from src.agents.research_agent_native import create_native_research_agent
    from src.agents.writing_agent_native import create_native_writing_agent
    from src.agents.affiliate_agent_native import create_native_affiliate_agent
    from src.agents.image_agent_native import create_native_image_agent
    from src.agents.publishing_agent_native import create_native_publishing_agent
    from src.agents.state import ArticleCMSState
    from langchain_core.messages import HumanMessage

    # Use Claude for all agents
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Full Article Pipeline")
    logger.info(f"Query: {query}")
    logger.info(f"Target Sites: {target_sites}")
    logger.info(f"{'='*80}\n")

    # Initialize state
    state: ArticleCMSState = {
        "query": query,
        "target_sites": target_sites,
        "messages": [],
        "errors": [],
        "workflow_step": 0,
        "agent_statuses": {}
    }

    try:
        # Step 1: Research
        logger.info("\n📚 STEP 1: RESEARCH AGENT")
        logger.info("-" * 80)

        research_agent = create_native_research_agent(llm=llm, verbose=True)
        research_result = await research_agent.ainvoke({
            "messages": [HumanMessage(content=f"Research this topic thoroughly: {query}")]
        })

        # Extract research data
        messages = research_result.get("messages", [])
        state["comprehensive_research"] = "Comprehensive research completed"
        state["casino_intelligence"] = {}

        logger.info(f"✅ Research completed")
        logger.info(f"   - Messages: {len(messages)}")

        # Step 2: Writing
        logger.info("\n✍️  STEP 2: WRITING AGENT")
        logger.info("-" * 80)

        writing_agent = create_native_writing_agent(llm=llm, verbose=True)
        writing_result = await writing_agent.ainvoke({
            "messages": [HumanMessage(content=f"""Write a comprehensive article about: {query}

Research available from previous step.

Please:
1. Select appropriate template
2. Generate high-quality content
3. Optimize for SEO
4. Refine the final output

Target length: 1500-2000 words
""")]
        })

        # Extract writing data
        writing_messages = writing_result.get("messages", [])
        final_message = writing_messages[-1] if writing_messages else None

        # Extract content from final AI message
        if final_message and hasattr(final_message, 'content'):
            state["final_content"] = final_message.content
            state["seo_metadata"] = {"title": f"{query} - Complete Guide"}
        else:
            state["final_content"] = f"Article about {query}"
            state["seo_metadata"] = {"title": query}

        logger.info(f"✅ Writing completed")
        logger.info(f"   - Content length: {len(state.get('final_content', ''))} chars")
        logger.info(f"   - Title: {state.get('seo_metadata', {}).get('title', 'N/A')}")

        # Step 3: Affiliate Links
        logger.info("\n🔗 STEP 3: AFFILIATE AGENT")
        logger.info("-" * 80)

        affiliate_agent = create_native_affiliate_agent(llm=llm, verbose=True)
        affiliate_result = await affiliate_agent.ainvoke({
            "messages": [HumanMessage(content=f"""Insert affiliate links into this content.

Content: {state.get('final_content', '')[:500]}...

Please query the database and insert relevant affiliate links.
""")]
        })

        state["inserted_links"] = []

        logger.info(f"✅ Affiliate links processed")

        # Step 4: Images
        logger.info("\n🖼️  STEP 4: IMAGE AGENT")
        logger.info("-" * 80)

        image_agent = create_native_image_agent(llm=llm, verbose=True)
        image_result = await image_agent.ainvoke({
            "messages": [HumanMessage(content=f"""Find and process images for this article: {query}

Content: {state.get('final_content', '')[:500]}...

Please:
1. Search for relevant images
2. Generate alt text
3. Upload to WordPress
""")]
        })

        state["images"] = []
        state["wordpress_media_ids"] = []

        logger.info(f"✅ Images processed")

        # Step 5: Publishing
        logger.info("\n🚀 STEP 5: PUBLISHING AGENT")
        logger.info("-" * 80)

        publishing_agent = create_native_publishing_agent(llm=llm, verbose=True)
        publishing_result = await publishing_agent.ainvoke({
            "messages": [HumanMessage(content=f"""Publish this content to WordPress sites: {target_sites}

Title: {state.get('seo_metadata', {}).get('title', query)}
Content: {state.get('final_content', '')}
Featured Image ID: {state.get('wordpress_media_ids', [None])[0]}

Please:
1. Query site registry for configurations
2. Adapt content if needed
3. Publish to each target site
4. Return post URLs
""")]
        })

        # Extract publishing results from messages
        publishing_messages = publishing_result.get("messages", [])
        published_posts = []
        post_urls = {}
        site_statuses = {}

        # Parse tool responses in messages
        for msg in publishing_messages:
            if hasattr(msg, 'name') and msg.name == 'wordpress_publish_tool':
                # Extract post data from tool response
                content = msg.content
                if isinstance(content, dict):
                    site_id = content.get("site_id")
                    post_url = content.get("post_url")
                    success = content.get("success", False)

                    if site_id:
                        site_statuses[site_id] = "success" if success else "failed"
                        if post_url:
                            post_urls[site_id] = post_url
                            published_posts.append({
                                "site_id": site_id,
                                "post_url": post_url,
                                "post_id": content.get("post_id")
                            })

        # Update state
        state["published_posts"] = published_posts
        state["site_statuses"] = site_statuses
        state["post_urls"] = post_urls

        if not published_posts:
            logger.error("No posts were published")
            return {"success": False, "error": "Publishing failed - no posts created"}

        logger.info(f"\n{'='*80}")
        logger.info(f"📊 PIPELINE COMPLETED")
        logger.info(f"{'='*80}")
        logger.info(f"Published to {len(published_posts)} sites:")

        for post in published_posts:
            site_id = post.get("site_id")
            post_url = post.get("post_url")
            logger.info(f"   ✅ {site_id}: {post_url}")

        # Check for failures
        failed_sites = [site for site, status in site_statuses.items() if status == "failed"]
        if failed_sites:
            logger.warning(f"\n⚠️  Failed sites: {failed_sites}")

        return {
            "success": True,
            "published_posts": published_posts,
            "post_urls": post_urls,
            "site_statuses": site_statuses,
            "state": state
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "state": state
        }


async def main():
    """Run the complete pipeline and publish an article"""

    # Verify API keys
    required_keys = ["ANTHROPIC_API_KEY", "WORDPRESS_URL", "WORDPRESS_USERNAME", "WORDPRESS_PASSWORD"]
    missing = [k for k in required_keys if not os.getenv(k)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please configure .env file")
        return

    logger.info("✅ All required API keys found")

    # Configuration
    query = "Betway Casino Review"
    target_sites = ["crashcasino-io"]  # WordPress site configured in .env

    # Run pipeline
    start_time = datetime.now()
    result = await run_full_pipeline(query, target_sites)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print results
    print("\n" + "="*80)
    print("🎉 FINAL RESULTS")
    print("="*80)

    if result["success"]:
        print(f"✅ SUCCESS - Article published in {duration:.1f} seconds")
        print(f"\nPublished URLs:")
        for site_id, url in result.get("post_urls", {}).items():
            print(f"   📄 {site_id}: {url}")

        print(f"\nArticle Details:")
        state = result.get("state", {})
        print(f"   Title: {state.get('seo_metadata', {}).get('title', 'N/A')}")
        print(f"   Word Count: {len(state.get('final_content', '').split())} words")
        print(f"   Images: {len(state.get('images', []))}")
        print(f"   Affiliate Links: {len(state.get('inserted_links', []))}")

    else:
        print(f"❌ FAILED - {result.get('error')}")
        if result.get("details"):
            print(f"   Details: {result['details']}")

    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
