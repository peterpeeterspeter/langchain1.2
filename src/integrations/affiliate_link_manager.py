"""
Affiliate Link Manager for Agent-Based CMS
Manages affiliate link registry, insertion, and tracking
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
from datetime import datetime

from supabase import create_client, Client

from ..schemas.affiliate_link_schema import (
    AffiliateLink,
    AffiliateLinkInsertion,
    AffiliateLinkOpportunity,
    UTMParameters,
    AffiliateLinkStatus,
    AffiliateLinkCategory
)

logger = logging.getLogger(__name__)


class AffiliateLinkManager:
    """
    Manages affiliate links database and operations
    
    Features:
    - Link registry (Supabase)
    - Context-aware link detection
    - Link insertion with tracking
    - UTM parameter generation
    - Link validation
    """
    
    def __init__(self, supabase_client: Optional[Client] = None):
        """
        Initialize Affiliate Link Manager
        
        Args:
            supabase_client: Supabase client instance
        """
        if supabase_client:
            self.supabase = supabase_client
        else:
            # Auto-initialize from environment
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
            else:
                self.supabase = None
                logger.warning("Supabase not configured - affiliate links will use in-memory storage")
        
        # In-memory cache (fallback if no Supabase)
        self._link_cache: Dict[str, AffiliateLink] = {}
    
    async def get_affiliate_links(
        self,
        category: Optional[AffiliateLinkCategory] = None,
        active_only: bool = True,
        limit: int = 100
    ) -> List[AffiliateLink]:
        """
        Get affiliate links from database
        
        Args:
            category: Filter by category
            active_only: Only return active links
            limit: Maximum number of links to return
            
        Returns:
            List of AffiliateLink objects
        """
        try:
            if self.supabase:
                query = self.supabase.table("affiliate_links").select("*")
                
                if active_only:
                    query = query.eq("active", True)
                
                if category:
                    query = query.eq("category", category.value)
                
                query = query.limit(limit)
                response = query.execute()
                
                links = [AffiliateLink(**link) for link in response.data]
                return links
            else:
                # Fallback to cache
                links = list(self._link_cache.values())
                if active_only:
                    links = [l for l in links if l.active]
                if category:
                    links = [l for l in links if l.category == category]
                return links[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get affiliate links: {e}")
            return []
    
    async def find_matching_links(
        self,
        content: str,
        category: Optional[AffiliateLinkCategory] = None,
        max_links: int = 10
    ) -> List[AffiliateLinkOpportunity]:
        """
        Find affiliate link opportunities in content
        
        Args:
            content: Content to analyze
            category: Filter by category
            max_links: Maximum opportunities to return
            
        Returns:
            List of AffiliateLinkOpportunity objects
        """
        try:
            # Get all relevant links
            links = await self.get_affiliate_links(category=category, active_only=True)
            
            opportunities = []
            content_lower = content.lower()
            
            for link in links:
                # Check if any keywords match
                for keyword in link.keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in content_lower:
                        # Find positions where keyword appears
                        positions = [m.start() for m in re.finditer(re.escape(keyword_lower), content_lower)]
                        
                        for pos in positions[:3]:  # Limit to first 3 occurrences
                            # Extract context around keyword
                            start = max(0, pos - 50)
                            end = min(len(content), pos + len(keyword) + 50)
                            context = content[start:end]
                            
                            # Calculate confidence score
                            confidence = self._calculate_match_confidence(keyword, content, pos)
                            
                            opportunity = AffiliateLinkOpportunity(
                                link_id=link.id,
                                confidence=confidence,
                                suggested_anchor_text=keyword.title(),
                                suggested_position=pos,
                                context_match=context,
                                keyword_match=keyword
                            )
                            opportunities.append(opportunity)
            
            # Sort by confidence and limit
            opportunities.sort(key=lambda x: x.confidence, reverse=True)
            return opportunities[:max_links]
            
        except Exception as e:
            logger.error(f"Failed to find matching links: {e}")
            return []
    
    async def insert_affiliate_links(
        self,
        content: str,
        opportunities: List[AffiliateLinkOpportunity],
        max_insertions: int = 5
    ) -> Tuple[str, List[AffiliateLinkInsertion]]:
        """
        Insert affiliate links into content
        
        Args:
            content: Original content
            opportunities: List of link opportunities
            max_insertions: Maximum number of links to insert
            
        Returns:
            Tuple of (enhanced_content, insertions)
        """
        try:
            enhanced_content = content
            insertions = []
            inserted_count = 0
            
            # Sort opportunities by position (descending) to insert from end to start
            sorted_opportunities = sorted(opportunities, key=lambda x: x.suggested_position, reverse=True)
            
            # Get link details
            link_ids = {opp.link_id for opp in sorted_opportunities}
            links_dict = {}
            for link_id in link_ids:
                link = await self.get_link_by_id(link_id)
                if link:
                    links_dict[link_id] = link
            
            for opportunity in sorted_opportunities[:max_insertions]:
                if inserted_count >= max_insertions:
                    break
                
                link = links_dict.get(opportunity.link_id)
                if not link or not link.active:
                    continue
                
                # Check usage limits
                if link.usage_count >= link.max_uses_per_article * 100:  # Approximate limit
                    continue
                
                # Generate tracking URL
                tracking_url = self._generate_tracking_url(link, opportunity)
                
                # Create HTML link
                anchor_text = opportunity.suggested_anchor_text
                html_link = f'<a href="{tracking_url}" target="_blank" rel="nofollow sponsored" title="{link.product_name}">{anchor_text}</a>'
                
                # Insert link at position
                pos = opportunity.suggested_position
                keyword = opportunity.keyword_match
                
                # Find exact keyword position (case-insensitive)
                content_lower = enhanced_content.lower()
                keyword_lower = keyword.lower()
                
                # Find first occurrence near suggested position
                search_start = max(0, pos - 20)
                search_end = min(len(enhanced_content), pos + len(keyword) + 20)
                search_area = enhanced_content[search_start:search_end]
                
                # Replace first occurrence in search area
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                match = pattern.search(search_area)
                
                if match:
                    actual_pos = search_start + match.start()
                    # Replace keyword with link
                    enhanced_content = (
                        enhanced_content[:actual_pos] +
                        html_link +
                        enhanced_content[actual_pos + len(keyword):]
                    )
                    
                    # Record insertion
                    insertion = AffiliateLinkInsertion(
                        link_id=link.id,
                        inserted_at=datetime.now(),
                        position=actual_pos,
                        anchor_text=anchor_text,
                        final_url=tracking_url,
                        context=opportunity.context_match
                    )
                    insertions.append(insertion)
                    
                    # Update link usage
                    await self._increment_link_usage(link.id)
                    
                    inserted_count += 1
            
            return enhanced_content, insertions
            
        except Exception as e:
            logger.error(f"Failed to insert affiliate links: {e}")
            return content, []
    
    async def get_link_by_id(self, link_id: str) -> Optional[AffiliateLink]:
        """Get affiliate link by ID"""
        try:
            if self.supabase:
                response = self.supabase.table("affiliate_links").select("*").eq("id", link_id).execute()
                if response.data:
                    return AffiliateLink(**response.data[0])
            else:
                return self._link_cache.get(link_id)
        except Exception as e:
            logger.error(f"Failed to get link {link_id}: {e}")
            return None
    
    def _generate_tracking_url(self, link: AffiliateLink, opportunity: AffiliateLinkOpportunity) -> str:
        """Generate tracking URL with UTM parameters"""
        base_url = str(link.affiliate_url)
        
        # Generate UTM parameters
        utm_params = UTMParameters(
            utm_source="affiliate_cms",
            utm_medium="affiliate",
            utm_campaign=link.merchant.lower().replace(" ", "_"),
            utm_term=opportunity.keyword_match,
            utm_content=link.id[:8]
        )
        
        # Add tracking ID to template
        tracking_url = link.tracking_template.format(
            url=base_url,
            tracking_id=link.tracking_id
        )
        
        # Add UTM parameters
        parsed = urlparse(tracking_url)
        query_params = parse_qs(parsed.query)
        query_params.update({
            "utm_source": [utm_params.utm_source],
            "utm_medium": [utm_params.utm_medium]
        })
        if utm_params.utm_campaign:
            query_params["utm_campaign"] = [utm_params.utm_campaign]
        if utm_params.utm_term:
            query_params["utm_term"] = [utm_params.utm_term]
        if utm_params.utm_content:
            query_params["utm_content"] = [utm_params.utm_content]
        
        # Reconstruct URL
        new_query = urlencode(query_params, doseq=True)
        final_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return final_url
    
    def _calculate_match_confidence(self, keyword: str, content: str, position: int) -> float:
        """Calculate confidence score for keyword match"""
        # Base confidence
        confidence = 0.7
        
        # Boost if keyword appears in title/heading
        if position < 200:  # Near beginning (likely heading)
            confidence += 0.2
        
        # Boost if keyword is longer (more specific)
        if len(keyword) > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _increment_link_usage(self, link_id: str):
        """Increment usage count for a link"""
        try:
            if self.supabase:
                # Get current usage
                response = self.supabase.table("affiliate_links").select("usage_count").eq("id", link_id).execute()
                if response.data:
                    current_count = response.data[0].get("usage_count", 0)
                    # Update
                    self.supabase.table("affiliate_links").update({
                        "usage_count": current_count + 1,
                        "last_used": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }).eq("id", link_id).execute()
        except Exception as e:
            logger.warning(f"Failed to increment link usage: {e}")
    
    async def validate_link(self, link_id: str) -> Dict[str, Any]:
        """
        Validate affiliate link (check if still active, etc.)
        
        Args:
            link_id: Link ID to validate
            
        Returns:
            Validation result dictionary
        """
        link = await self.get_link_by_id(link_id)
        if not link:
            return {
                "valid": False,
                "error": "Link not found"
            }
        
        # Basic validation
        is_valid = (
            link.active and
            link.status == AffiliateLinkStatus.ACTIVE and
            len(link.affiliate_url) > 0
        )
        
        return {
            "valid": is_valid,
            "link_id": link_id,
            "status": link.status.value,
            "active": link.active
        }

