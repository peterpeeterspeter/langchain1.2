#!/usr/bin/env python3
"""
Direct WordPress publishing - bypass site registry
Uses WordPress credentials from .env directly
"""

import asyncio
import logging
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def publish_to_wordpress(title: str, content: str, status: str = "publish") -> dict:
    """
    Publish directly to WordPress using REST API

    Args:
        title: Post title
        content: Post content (HTML)
        status: Post status (draft or publish)

    Returns:
        Dictionary with result
    """
    wordpress_url = os.getenv("WORDPRESS_URL")
    wordpress_user = os.getenv("WORDPRESS_USERNAME")
    wordpress_pass = os.getenv("WORDPRESS_PASSWORD")

    if not all([wordpress_url, wordpress_user, wordpress_pass]):
        return {
            "success": False,
            "error": "Missing WordPress credentials in .env"
        }

    # WordPress REST API endpoint
    api_url = f"{wordpress_url}/wp-json/wp/v2/posts"

    # Post data
    post_data = {
        "title": title,
        "content": content,
        "status": status,
        "author": 1  # Default author ID
    }

    logger.info(f"Publishing to {wordpress_url}...")
    logger.info(f"Title: {title}")
    logger.info(f"Content length: {len(content)} chars")
    logger.info(f"Status: {status}")

    try:
        # Make POST request
        response = requests.post(
            api_url,
            json=post_data,
            auth=(wordpress_user, wordpress_pass),
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        logger.info(f"Response status: {response.status_code}")

        if response.status_code in [200, 201]:
            data = response.json()
            post_id = data.get("id")
            post_url = data.get("link")

            logger.info(f"✅ Success!")
            logger.info(f"   Post ID: {post_id}")
            logger.info(f"   URL: {post_url}")

            return {
                "success": True,
                "post_id": post_id,
                "post_url": post_url,
                "status_code": response.status_code
            }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error(f"❌ Failed: {error_msg}")

            return {
                "success": False,
                "error": error_msg,
                "status_code": response.status_code
            }

    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def main():
    """Generate article and publish directly"""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    from src.agents.writing_agent_native import create_native_writing_agent

    logger.info("="*80)
    logger.info("DIRECT WORDPRESS PUBLISHING TEST")
    logger.info("="*80)

    # Check WordPress credentials
    if not all([os.getenv("WORDPRESS_URL"), os.getenv("WORDPRESS_USERNAME"), os.getenv("WORDPRESS_PASSWORD")]):
        logger.error("Missing WordPress credentials in .env")
        return

    logger.info("✅ WordPress credentials found")

    # Create Claude LLM
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=4096
    )

    # Generate article
    logger.info("\n📝 Generating article with Claude...\n")

    writing_agent = create_native_writing_agent(llm=llm, verbose=True)
    result = await writing_agent.ainvoke({
        "messages": [HumanMessage(content="""Write a comprehensive Betway Casino review article.

Requirements:
- Professional and informative tone
- 1500-2000 words
- Include sections on: games, bonuses, payment methods, user experience
- SEO-optimized
- Proper HTML formatting with headings and paragraphs
- Engaging introduction and conclusion

Topic: Betway Casino Review - Complete 2024 Guide
""")]
    })

    # Extract content
    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None

    if not final_message or not hasattr(final_message, 'content'):
        logger.error("Failed to generate article")
        return

    article_content = final_message.content
    article_title = "Betway Casino Review - Complete 2024 Guide"

    logger.info(f"\n✅ Article generated!")
    logger.info(f"   Title: {article_title}")
    logger.info(f"   Length: {len(article_content)} characters")
    logger.info(f"   Words: {len(article_content.split())} words")

    # Save article to file
    filename = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, 'w') as f:
        f.write(f"<h1>{article_title}</h1>\n\n")
        f.write(article_content)
    logger.info(f"\n💾 Article saved to: {filename}")

    # Publish to WordPress
    logger.info(f"\n{'='*80}")
    logger.info("🚀 PUBLISHING TO WORDPRESS")
    logger.info(f"{'='*80}\n")

    publish_result = publish_to_wordpress(
        title=article_title,
        content=article_content,
        status="publish"  # Change to "draft" if you want to review first
    )

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info("📊 FINAL RESULTS")
    logger.info(f"{'='*80}")

    if publish_result.get("success"):
        logger.info("✅ ARTICLE PUBLISHED SUCCESSFULLY!")
        logger.info(f"\n   📄 Post ID: {publish_result.get('post_id')}")
        logger.info(f"   🔗 URL: {publish_result.get('post_url')}")

        print(f"\n\n🎉 SUCCESS! Article published! 🎉\n")
        print(f"📄 View your article at:\n   {publish_result.get('post_url')}\n")
    else:
        logger.error(f"❌ PUBLISHING FAILED")
        logger.error(f"   Error: {publish_result.get('error')}")
        logger.info(f"\n   💾 Article saved locally: {filename}")

        print(f"\n\n⚠️  Publishing failed, but article was saved to {filename}\n")


if __name__ == "__main__":
    asyncio.run(main())
