"""
Publishing Agent for Agent-Based CMS
Publishes content to multiple WordPress sites
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent, AgentResult
from .state import ArticleCMSState
from .tools.publishing_tools import (
    wordpress_publish_tool,
    site_registry_tool,
    content_adaptation_tool
)

logger = logging.getLogger(__name__)


class PublishingAgent(BaseAgent):
    """
    Publishing Agent - Publishes content to WordPress sites
    
    Uses tools:
    - wordpress_publish_tool: Publish to WordPress
    - site_registry_tool: Manage site registry
    - content_adaptation_tool: Adapt content for sites
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        **kwargs
    ):
        """
        Initialize Publishing Agent
        
        Args:
            llm: Language model for agent reasoning
        """
        tools = [
            wordpress_publish_tool,
            site_registry_tool,
            content_adaptation_tool
        ]
        
        super().__init__(
            name="publishing_agent",
            llm=llm,
            tools=tools,
            **kwargs
        )
    
    async def execute(self, state: ArticleCMSState) -> AgentResult:
        """
        Execute publishing agent logic
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResult with publishing results
        """
        content = state.get("final_content", "")
        title = state.get("seo_metadata", {}).get("title", state.get("query", "Untitled"))
        target_sites = state.get("target_sites", [])
        images = state.get("images", [])
        wordpress_media_ids = state.get("wordpress_media_ids", [])
        
        if not content:
            return AgentResult(
                success=False,
                error="No content available for publishing"
            )
        
        if not target_sites:
            logger.warning("Publishing Agent: No target sites specified")
            return AgentResult(
                success=True,
                state_updates={
                    "published_posts": [],
                    "site_statuses": {},
                    "post_urls": {}
                },
                metadata={"sites_targeted": 0, "sites_published": 0}
            )
        
        try:
            # Step 1: Get site configurations
            logger.info(f"Publishing Agent: Getting configurations for {len(target_sites)} sites")
            sites_result = await site_registry_tool.ainvoke({
                "action": "list"
            })
            
            all_sites = sites_result.get("sites", [])
            target_site_configs = [s for s in all_sites if s.get("site_id") in target_sites]
            
            if not target_site_configs:
                return AgentResult(
                    success=False,
                    error=f"None of the target sites {target_sites} found in registry"
                )
            
            # Step 2: Publish to each site
            published_posts = []
            site_statuses = {}
            post_urls = {}
            sites_published = 0
            sites_failed = 0
            
            # Get featured image (first WordPress media ID or first image URL)
            featured_image_media_id = None
            if wordpress_media_ids:
                featured_image_media_id = wordpress_media_ids[0]
            elif images:
                # Could upload first image here if needed
                pass
            
            # Get meta description
            meta_description = state.get("seo_metadata", {}).get("description", "")
            
            for site_config in target_site_configs:
                site_id = site_config.get("site_id")
                site_name = site_config.get("site_name", site_id)
                
                try:
                    # Adapt content if needed
                    adapted_content = content
                    if site_config.get("content_adaptation", False):
                        logger.info(f"Publishing Agent: Adapting content for {site_name}")
                        adaptation_result = await content_adaptation_tool.ainvoke({
                            "content": content,
                            "site_config": site_config
                        })
                        adapted_content = adaptation_result.get("adapted_content", content)
                    
                    # Publish to WordPress
                    logger.info(f"Publishing Agent: Publishing to {site_name} ({site_id})")
                    publish_result = await wordpress_publish_tool.ainvoke({
                        "site_id": site_id,
                        "title": title,
                        "content": adapted_content,
                        "featured_image_media_id": featured_image_media_id,
                        "status": site_config.get("default_status", "publish"),
                        "categories": site_config.get("default_category_ids", []),
                        "tags": site_config.get("default_tags", []),
                        "meta_description": meta_description
                    })
                    
                    if publish_result.get("success"):
                        post_id = publish_result.get("post_id")
                        post_url = publish_result.get("post_url", "")
                        
                        published_posts.append({
                            "site_id": site_id,
                            "site_name": site_name,
                            "post_id": post_id,
                            "post_url": post_url,
                            "status": "published"
                        })
                        
                        site_statuses[site_id] = "success"
                        post_urls[site_id] = post_url
                        sites_published += 1
                        
                        logger.info(f"Publishing Agent: Successfully published to {site_name} - Post ID: {post_id}")
                    else:
                        error = publish_result.get("error", "Unknown error")
                        site_statuses[site_id] = f"failed: {error}"
                        sites_failed += 1
                        logger.error(f"Publishing Agent: Failed to publish to {site_name}: {error}")
                        
                except Exception as e:
                    site_statuses[site_id] = f"error: {str(e)}"
                    sites_failed += 1
                    logger.error(f"Publishing Agent: Exception publishing to {site_id}: {e}")
            
            # Prepare state updates
            state_updates = {
                "published_posts": published_posts,
                "site_statuses": site_statuses,
                "post_urls": post_urls,
                "workflow_step": state.get("workflow_step", 0) + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "sites_targeted": len(target_sites),
                    "sites_published": sites_published,
                    "sites_failed": sites_failed
                }
            }
            
            success = sites_published > 0
            
            logger.info(f"Publishing Agent: Published to {sites_published}/{len(target_sites)} sites")
            
            return AgentResult(
                success=success,
                state_updates=state_updates,
                metadata={
                    "sites_targeted": len(target_sites),
                    "sites_published": sites_published,
                    "sites_failed": sites_failed
                }
            )
            
        except Exception as e:
            logger.error(f"Publishing Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )

