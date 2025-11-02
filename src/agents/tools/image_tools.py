"""
Image Tools for Image Agent
Tools for image search, optimization, and WordPress upload
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.integrations.dataforseo_image_search import DataForSEOImageSearch, ImageSearchRequest
    from src.integrations.wordpress_publisher import WordPressConfig, WordPressIntegration

try:
    from src.integrations.dataforseo_image_search import DataForSEOImageSearch, ImageSearchRequest
    from src.integrations.wordpress_publisher import WordPressConfig, WordPressIntegration
    IMAGE_AVAILABLE = True
except ImportError as e:
    IMAGE_AVAILABLE = False
    DataForSEOImageSearch = None  # type: ignore
    ImageSearchRequest = None  # type: ignore
    WordPressConfig = None  # type: ignore
    WordPressIntegration = None  # type: ignore
    logging.warning(f"Image components not available: {e}")

logger = logging.getLogger(__name__)

# Global instances
_dataforseo_client: Optional[Any] = None


def _get_dataforseo_client() -> Optional[DataForSEOImageSearch]:
    """Get or create DataForSEO client"""
    global _dataforseo_client
    if _dataforseo_client is None and IMAGE_AVAILABLE:
        try:
            _dataforseo_client = DataForSEOImageSearch()
        except Exception as e:
            logger.error(f"Failed to create DataForSEO client: {e}")
    return _dataforseo_client


@tool
async def image_search_tool(
    query: str,
    max_results: int = 10,
    image_size: Optional[str] = None,
    image_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for images using DataForSEO
    
    Args:
        query: Image search query
        max_results: Maximum number of results (1-100)
        image_size: Filter by size (small, medium, large, extra_large)
        image_type: Filter by type (photo, clipart, line_drawing, animated)
        
    Returns:
        Dictionary with image search results:
        - images: List of image metadata
        - total_found: Total images found
        - quality_scores: Quality assessment
    """
    if not IMAGE_AVAILABLE:
        return {
            "images": [],
            "total_found": 0,
            "error": "Image search components not available"
        }
    
    try:
        client = _get_dataforseo_client()
        if not client:
            return {
                "images": [],
                "total_found": 0,
                "error": "Failed to initialize DataForSEO client"
            }
        
        # Create search request
        search_request = ImageSearchRequest(
            keyword=query,
            max_results=min(max_results, 100),
            image_size=image_size if image_size else None,
            image_type=image_type if image_type else None
        )
        
        # Perform search
        logger.info(f"Image search tool: Searching for '{query}'")
        results = await client.search_images_async(search_request)
        
        # Format results
        images = []
        for result in results[:max_results]:
            images.append({
                "url": result.url,
                "title": result.title,
                "alt_text": result.alt_text,
                "width": result.width,
                "height": result.height,
                "file_size": result.file_size,
                "quality_score": result.quality_score if hasattr(result, 'quality_score') else 0.0
            })
        
        return {
            "images": images,
            "total_found": len(images),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return {
            "images": [],
            "total_found": 0,
            "error": str(e)
        }


@tool
async def image_selection_tool(
    images: List[Dict[str, Any]],
    content: str,
    max_select: int = 5
) -> Dict[str, Any]:
    """
    Select best images for content based on relevance
    
    Args:
        images: List of candidate images
        content: Content to match images against
        max_select: Maximum images to select
        
    Returns:
        Dictionary with selected images:
        - selected_images: List of selected image metadata
        - selection_reason: Reason for each selection
    """
    try:
        # Simple relevance scoring (full implementation would use embeddings/LLM)
        content_lower = content.lower()
        content_keywords = set(content_lower.split()[:20])  # Top 20 words
        
        scored_images = []
        for img in images:
            score = 0.0
            title = (img.get("title", "") or "").lower()
            alt_text = (img.get("alt_text", "") or "").lower()
            
            # Score based on keyword matches
            for keyword in content_keywords:
                if len(keyword) > 3:  # Only meaningful keywords
                    if keyword in title:
                        score += 0.1
                    if keyword in alt_text:
                        score += 0.05
            
            # Quality score boost
            quality = img.get("quality_score", 0.0)
            score += quality * 0.3
            
            # Size preference (medium-large preferred)
            width = img.get("width", 0)
            height = img.get("height", 0)
            if 800 <= width <= 1920 and 600 <= height <= 1080:
                score += 0.1
            
            scored_images.append({
                **img,
                "relevance_score": score
            })
        
        # Sort by score and select top images
        scored_images.sort(key=lambda x: x["relevance_score"], reverse=True)
        selected = scored_images[:max_select]
        
        return {
            "selected_images": selected,
            "selection_reason": [f"Relevance score: {img['relevance_score']:.2f}" for img in selected]
        }
        
    except Exception as e:
        logger.error(f"Image selection failed: {e}")
        return {
            "selected_images": images[:max_select] if images else [],
            "selection_reason": [],
            "error": str(e)
        }


@tool
async def alt_text_generation_tool(
    image_url: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate SEO-optimized alt text for an image
    
    Args:
        image_url: Image URL
        context: Surrounding content context
        
    Returns:
        Dictionary with generated alt text:
        - alt_text: Generated alt text
        - seo_score: SEO quality score
    """
    try:
        # Simple alt text generation (full implementation would use LLM)
        filename = image_url.split("/")[-1].split(".")[0]
        
        # Clean filename
        alt_text = filename.replace("-", " ").replace("_", " ").title()
        
        # Enhance with context if provided
        if context:
            # Extract key terms from context
            context_words = context.lower().split()[:10]
            relevant_terms = [w for w in context_words if len(w) > 4][:3]
            if relevant_terms:
                alt_text = f"{alt_text} - {', '.join(relevant_terms)}"
        
        # Ensure reasonable length
        if len(alt_text) > 125:
            alt_text = alt_text[:122] + "..."
        
        return {
            "alt_text": alt_text,
            "seo_score": 0.8 if len(alt_text) > 20 else 0.5
        }
        
    except Exception as e:
        logger.error(f"Alt text generation failed: {e}")
        return {
            "alt_text": "Image",
            "seo_score": 0.0,
            "error": str(e)
        }


@tool
async def wordpress_image_upload_tool(
    image_url: str,
    site_config: Dict[str, Any],
    alt_text: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Upload image to WordPress media library
    
    Args:
        image_url: Image URL to upload
        site_config: WordPress site configuration
        alt_text: Image alt text
        title: Image title
        
    Returns:
        Dictionary with upload result:
        - success: Whether upload succeeded
        - media_id: WordPress media ID
        - wordpress_url: WordPress-hosted image URL
    """
    if not IMAGE_AVAILABLE:
        return {
            "success": False,
            "error": "WordPress components not available"
        }
    
    try:
        # Create WordPress config from site_config
        wp_config = WordPressConfig(
            site_url=site_config.get("site_url", ""),
            username=site_config.get("username", ""),
            application_password=site_config.get("application_password", "")
        )
        
        # Create WordPress integration
        wp_integration = WordPressIntegration(
            wordpress_config=wp_config,
            supabase_client=None
        )
        
        # Upload image
        logger.info(f"Uploading image to WordPress: {image_url}")
        result = await wp_integration.upload_image_to_wordpress(
            image_url=image_url,
            alt_text=alt_text or "",
            title=title or ""
        )
        
        if result and result.get("success"):
            return {
                "success": True,
                "media_id": result.get("media_id"),
                "wordpress_url": result.get("wordpress_url", image_url)
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Upload failed") if result else "Unknown error"
            }
            
    except Exception as e:
        logger.error(f"WordPress image upload failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

