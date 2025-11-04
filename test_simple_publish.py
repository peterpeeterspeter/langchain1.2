#!/usr/bin/env python3
"""
Simple end-to-end test using the standard CMS workflow
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
    """Run the CMS workflow to publish an article"""
    from src.agents.orchestrator_native import create_native_cms_orchestrator

    logger.info("="*80)
    logger.info("RUNNING FULL ARTICLE CMS WORKFLOW")
    logger.info("="*80)

    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.error("No API keys found. Please configure .env file.")
        return

    # Create the native orchestrator
    logger.info("\n📊 Creating Native Article CMS Orchestrator...")
    orchestrator = create_native_cms_orchestrator(enable_checkpoints=True)

    # Show orchestrator status
    status = orchestrator.get_workflow_status()
    logger.info(f"\n🤖 Orchestrator: {status['orchestrator']}")
    logger.info(f"   Workflow Engine: {status['workflow_engine']}")
    logger.info(f"   Checkpoints: {status['checkpoints']}")

    # Define query
    query = "Betway Casino Review - Complete 2024 Guide"
    target_sites = ["crashcasino-io"]  # Publish to crashcasino.io

    logger.info(f"\n📝 Input:")
    logger.info(f"   Query: {query}")
    logger.info(f"   Target Sites: {target_sites}")

    # Run the workflow
    logger.info("\n🚀 Starting native agent workflow...\n")
    start_time = datetime.now()

    try:
        result = await orchestrator.run(
            query=query,
            target_sites=target_sites
        )

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
        logger.info(f"   Images: {len(result.get('images', []))}")
        logger.info(f"   Affiliate links: {len(result.get('inserted_links', []))}")

        # Agent statuses
        logger.info(f"\n🤖 Agent Statuses:")
        for agent, status in result.get("agent_statuses", {}).items():
            status_icon = "✅" if status == "completed" else "❌"
            logger.info(f"   {status_icon} {agent}: {status}")

        logger.info("\n" + "="*80)

        if published_posts:
            print("\n\n🎉 ARTICLE PUBLISHED SUCCESSFULLY! 🎉\n")
            for site_id, url in post_urls.items():
                print(f"View your article at: {url}\n")

    except Exception as e:
        logger.error(f"\n❌ Workflow failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
