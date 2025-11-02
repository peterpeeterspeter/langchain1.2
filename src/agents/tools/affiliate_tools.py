"""
Affiliate Link Tools for Affiliate Agent
Tools for managing and inserting affiliate links
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

try:
    from src.integrations.affiliate_link_manager import AffiliateLinkManager
    from src.schemas.affiliate_link_schema import AffiliateLinkCategory
    AFFILIATE_AVAILABLE = True
except ImportError as e:
    AFFILIATE_AVAILABLE = False
    logging.warning(f"Affiliate components not available: {e}")

logger = logging.getLogger(__name__)

# Global manager instance
_affiliate_manager: Optional[AffiliateLinkManager] = None


def _get_affiliate_manager() -> Optional[AffiliateLinkManager]:
    """Get or create AffiliateLinkManager instance"""
    global _affiliate_manager
    if _affiliate_manager is None and AFFILIATE_AVAILABLE:
        try:
            _affiliate_manager = AffiliateLinkManager()
        except Exception as e:
            logger.error(f"Failed to create AffiliateLinkManager: {e}")
    return _affiliate_manager


@tool
async def affiliate_link_database_tool(
    category: Optional[str] = None,
    active_only: bool = True
) -> Dict[str, Any]:
    """
    Query affiliate link database
    
    Args:
        category: Filter by category (casino, sportsbook, crypto_casino, etc.)
        active_only: Only return active links
        
    Returns:
        Dictionary with affiliate links:
        - links: List of affiliate link data
        - total_count: Total number of links
    """
    if not AFFILIATE_AVAILABLE:
        return {
            "links": [],
            "total_count": 0,
            "error": "Affiliate components not available"
        }
    
    try:
        manager = _get_affiliate_manager()
        if not manager:
            return {
                "links": [],
                "total_count": 0,
                "error": "Failed to initialize affiliate manager"
            }
        
        category_enum = None
        if category:
            try:
                category_enum = AffiliateLinkCategory[category.upper()]
            except KeyError:
                logger.warning(f"Invalid category: {category}")
        
        links = await manager.get_affiliate_links(
            category=category_enum,
            active_only=active_only
        )
        
        # Convert to dict format
        links_data = [link.model_dump() for link in links]
        
        return {
            "links": links_data,
            "total_count": len(links_data)
        }
        
    except Exception as e:
        logger.error(f"Affiliate link database query failed: {e}")
        return {
            "links": [],
            "total_count": 0,
            "error": str(e)
        }


@tool
async def link_insertion_tool(
    content: str,
    category: Optional[str] = None,
    max_links: int = 5
) -> Dict[str, Any]:
    """
    Insert affiliate links contextually into content
    
    Args:
        content: Content to insert links into
        category: Filter links by category
        max_links: Maximum number of links to insert
        
    Returns:
        Dictionary with results:
        - enhanced_content: Content with links inserted
        - insertions: List of insertion records
        - links_inserted: Number of links inserted
    """
    if not AFFILIATE_AVAILABLE:
        return {
            "enhanced_content": content,
            "insertions": [],
            "links_inserted": 0,
            "error": "Affiliate components not available"
        }
    
    try:
        manager = _get_affiliate_manager()
        if not manager:
            return {
                "enhanced_content": content,
                "insertions": [],
                "links_inserted": 0,
                "error": "Failed to initialize affiliate manager"
            }
        
        # Find opportunities
        category_enum = None
        if category:
            try:
                category_enum = AffiliateLinkCategory[category.upper()]
            except KeyError:
                pass
        
        opportunities = await manager.find_matching_links(
            content=content,
            category=category_enum,
            max_links=max_links * 2  # Get more opportunities to filter
        )
        
        if not opportunities:
            return {
                "enhanced_content": content,
                "insertions": [],
                "links_inserted": 0,
                "message": "No affiliate link opportunities found"
            }
        
        # Insert links
        enhanced_content, insertions = await manager.insert_affiliate_links(
            content=content,
            opportunities=opportunities,
            max_insertions=max_links
        )
        
        # Convert insertions to dict
        insertions_data = [ins.model_dump() for ins in insertions]
        
        return {
            "enhanced_content": enhanced_content,
            "insertions": insertions_data,
            "links_inserted": len(insertions_data)
        }
        
    except Exception as e:
        logger.error(f"Link insertion failed: {e}")
        return {
            "enhanced_content": content,
            "insertions": [],
            "links_inserted": 0,
            "error": str(e)
        }


@tool
async def link_validation_tool(link_id: str) -> Dict[str, Any]:
    """
    Validate affiliate link
    
    Args:
        link_id: Affiliate link ID to validate
        
    Returns:
        Validation result dictionary
    """
    if not AFFILIATE_AVAILABLE:
        return {
            "valid": False,
            "error": "Affiliate components not available"
        }
    
    try:
        manager = _get_affiliate_manager()
        if not manager:
            return {
                "valid": False,
                "error": "Failed to initialize affiliate manager"
            }
        
        result = await manager.validate_link(link_id)
        return result
        
    except Exception as e:
        logger.error(f"Link validation failed: {e}")
        return {
            "valid": False,
            "error": str(e)
        }


@tool
async def tracking_parameter_tool(
    base_url: str,
    source: str,
    campaign: Optional[str] = None,
    term: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate tracking parameters (UTM) for affiliate links
    
    Args:
        base_url: Base affiliate URL
        source: Traffic source
        campaign: Campaign name
        term: Keyword/term
        
    Returns:
        Dictionary with tracking URL and parameters
    """
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        parsed = urlparse(base_url)
        query_params = parse_qs(parsed.query)
        
        # Add UTM parameters
        query_params["utm_source"] = [source]
        query_params["utm_medium"] = ["affiliate"]
        if campaign:
            query_params["utm_campaign"] = [campaign]
        if term:
            query_params["utm_term"] = [term]
        
        # Reconstruct URL
        new_query = urlencode(query_params, doseq=True)
        tracking_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return {
            "tracking_url": tracking_url,
            "utm_source": source,
            "utm_medium": "affiliate",
            "utm_campaign": campaign,
            "utm_term": term
        }
        
    except Exception as e:
        logger.error(f"Tracking parameter generation failed: {e}")
        return {
            "tracking_url": base_url,
            "error": str(e)
        }

