"""
Publishing Tools for Publishing Agent
Tools for WordPress publishing and multi-site management
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.integrations.wordpress_publisher import WordPressConfig, WordPressIntegration, WordPressRESTPublisher
    from src.integrations.wordpress_site_registry import WordPressSiteRegistry, WordPressSiteConfig

try:
    from src.integrations.wordpress_publisher import WordPressConfig, WordPressIntegration, WordPressRESTPublisher
    from src.integrations.wordpress_site_registry import WordPressSiteRegistry, WordPressSiteConfig
    PUBLISHING_AVAILABLE = True
except ImportError as e:
    PUBLISHING_AVAILABLE = False
    WordPressConfig = None  # type: ignore
    WordPressIntegration = None  # type: ignore
    WordPressRESTPublisher = None  # type: ignore
    WordPressSiteRegistry = None  # type: ignore
    WordPressSiteConfig = None  # type: ignore
    logging.warning(f"Publishing components not available: {e}")

logger = logging.getLogger(__name__)

# Global registry instance
_site_registry: Optional[Any] = None


def _get_site_registry() -> Optional[WordPressSiteRegistry]:
    """Get or create WordPress Site Registry"""
    global _site_registry
    if _site_registry is None and PUBLISHING_AVAILABLE:
        try:
            _site_registry = WordPressSiteRegistry()
        except Exception as e:
            logger.error(f"Failed to create site registry: {e}")
    return _site_registry


@tool
async def wordpress_publish_tool(
    site_id: str,
    title: str,
    content: str,
    featured_image_media_id: Optional[int] = None,
    status: Optional[str] = None,
    categories: Optional[List[int]] = None,
    tags: Optional[List[str]] = None,
    meta_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Publish content to a WordPress site
    
    Args:
        site_id: WordPress site ID from registry
        title: Post title
        content: Post content (HTML)
        featured_image_media_id: WordPress media ID for featured image
        status: Post status (draft, publish, private)
        categories: Category IDs
        tags: Tag names
        meta_description: SEO meta description
        
    Returns:
        Dictionary with publishing result:
        - success: Whether publishing succeeded
        - post_id: WordPress post ID
        - post_url: Published post URL
        - site_id: Site ID
    """
    if not PUBLISHING_AVAILABLE:
        return {
            "success": False,
            "error": "Publishing components not available"
        }
    
    try:
        registry = _get_site_registry()
        if not registry:
            return {
                "success": False,
                "error": "Failed to initialize site registry"
            }
        
        # Get site configuration
        site_config = await registry.get_site(site_id)
        if not site_config:
            return {
                "success": False,
                "error": f"Site {site_id} not found in registry"
            }
        
        # Create WordPress config
        wp_config = WordPressConfig(
            site_url=str(site_config.site_url),
            username=site_config.username,
            application_password=site_config.application_password,
            default_status=status or site_config.default_status,
            default_author_id=site_config.default_author_id,
            default_category_ids=categories or site_config.default_category_ids,
            default_tags=tags or site_config.default_tags
        )
        
        # Publish post using WordPressRESTPublisher directly
        logger.info(f"Publishing to WordPress site: {site_id}")
        
        # Use async context manager for publisher
        async with WordPressRESTPublisher(wp_config) as publisher:
            result = await publisher.publish_post(
                title=title,
                content=content,
                status=status or site_config.default_status,
                featured_image_url=None,  # Can be added later if needed
                categories=categories or site_config.default_category_ids,
                tags=tags or site_config.default_tags,
                meta_description=meta_description or ""
            )
        
        if result and result.get("id"):
            # WordPress API returns post with 'id' field
            post_id = result.get("id")
            post_url = result.get("link", "")
            
            return {
                "success": True,
                "post_id": post_id,
                "post_url": post_url,
                "site_id": site_id,
                "site_name": site_config.site_name
            }
        else:
            return {
                "success": False,
                "error": "Publishing failed - no post ID returned",
                "site_id": site_id,
                "result": result
            }
            
    except Exception as e:
        logger.error(f"WordPress publishing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "site_id": site_id
        }


@tool
async def site_registry_tool(
    action: str,
    site_id: Optional[str] = None,
    site_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Manage WordPress site registry
    
    Args:
        action: Action to perform (list, get, register, validate)
        site_id: Site ID (for get/validate actions)
        site_config: Site configuration dict (for register action)
        
    Returns:
        Dictionary with result based on action
    """
    if not PUBLISHING_AVAILABLE:
        return {
            "success": False,
            "error": "Publishing components not available"
        }
    
    try:
        registry = _get_site_registry()
        if not registry:
            return {
                "success": False,
                "error": "Failed to initialize site registry"
            }
        
        if action == "list":
            sites = await registry.get_sites(active_only=True)
            return {
                "success": True,
                "sites": [site.model_dump() for site in sites],
                "count": len(sites)
            }
        
        elif action == "get":
            if not site_id:
                return {"success": False, "error": "site_id required for get action"}
            
            site = await registry.get_site(site_id)
            if site:
                return {
                    "success": True,
                    "site": site.model_dump()
                }
            else:
                return {
                    "success": False,
                    "error": f"Site {site_id} not found"
                }
        
        elif action == "register":
            if not site_config:
                return {"success": False, "error": "site_config required for register action"}
            
            try:
                config = WordPressSiteConfig(**site_config)
                success = await registry.register_site(config)
                return {
                    "success": success,
                    "site_id": config.site_id
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        elif action == "validate":
            if not site_id:
                return {"success": False, "error": "site_id required for validate action"}
            
            result = await registry.validate_site(site_id)
            return result
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Valid actions: list, get, register, validate"
            }
            
    except Exception as e:
        logger.error(f"Site registry operation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool
async def content_adaptation_tool(
    content: str,
    site_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adapt content for specific WordPress site
    
    Args:
        content: Original content
        site_config: Target site configuration
        
    Returns:
        Dictionary with adapted content:
        - adapted_content: Content adapted for site
        - adaptations_made: List of adaptations applied
    """
    try:
        adapted_content = content
        adaptations = []
        
        # Check max content length
        max_length = site_config.get("max_content_length")
        if max_length and len(content) > max_length:
            adapted_content = content[:max_length - 3] + "..."
            adaptations.append(f"Truncated to {max_length} characters")
        
        # Site-specific adaptations could be added here
        # (e.g., remove certain sections, adjust formatting, etc.)
        
        return {
            "adapted_content": adapted_content,
            "adaptations_made": adaptations
        }
        
    except Exception as e:
        logger.error(f"Content adaptation failed: {e}")
        return {
            "adapted_content": content,
            "adaptations_made": [],
            "error": str(e)
        }

