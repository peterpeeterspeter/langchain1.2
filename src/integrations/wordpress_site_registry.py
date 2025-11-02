"""
WordPress Site Registry for Multi-Site Publishing
Manages configuration for multiple WordPress sites
"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl

from supabase import create_client, Client

logger = logging.getLogger(__name__)


class WordPressSiteConfig(BaseModel):
    """Configuration for a WordPress site"""
    site_id: str = Field(description="Unique site identifier")
    site_name: str = Field(description="Display name for the site")
    site_url: HttpUrl = Field(description="WordPress site URL")
    username: str = Field(description="WordPress username")
    application_password: str = Field(description="WordPress application password")
    
    # Publishing defaults
    default_status: str = Field(default="publish", description="Default post status")
    default_author_id: int = Field(default=1, description="Default author ID")
    default_category_ids: List[int] = Field(default_factory=list, description="Default category IDs")
    default_tags: List[str] = Field(default_factory=list, description="Default tags")
    
    # Site-specific settings
    content_adaptation: bool = Field(default=False, description="Whether to adapt content for this site")
    featured_image_required: bool = Field(default=False, description="Whether featured image is required")
    max_content_length: Optional[int] = Field(default=None, description="Maximum content length")
    
    # Metadata
    active: bool = Field(default=True, description="Whether site is active")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class WordPressSiteRegistry:
    """
    Registry for managing multiple WordPress sites
    
    Features:
    - Site configuration storage (Supabase)
    - Site validation
    - Site listing and filtering
    """
    
    def __init__(self, supabase_client: Optional[Client] = None):
        """
        Initialize WordPress Site Registry
        
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
                logger.warning("Supabase not configured - site registry will use in-memory storage")
        
        # In-memory cache (fallback if no Supabase)
        self._site_cache: Dict[str, WordPressSiteConfig] = {}
    
    async def register_site(self, config: WordPressSiteConfig) -> bool:
        """
        Register a new WordPress site
        
        Args:
            config: Site configuration
            
        Returns:
            True if registration succeeded
        """
        try:
            if self.supabase:
                # Store in Supabase
                site_data = config.model_dump(mode='json')  # Use mode='json' to serialize HttpUrl properly
                site_data["updated_at"] = datetime.now().isoformat()
                site_data["created_at"] = config.created_at.isoformat()
                # Ensure site_url is a string
                if "site_url" in site_data and not isinstance(site_data["site_url"], str):
                    site_data["site_url"] = str(site_data["site_url"])
                
                response = self.supabase.table("wordpress_sites").upsert(site_data).execute()
                logger.info(f"Registered WordPress site: {config.site_id}")
                return True
            else:
                # Store in cache
                self._site_cache[config.site_id] = config
                logger.info(f"Registered WordPress site (cache): {config.site_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register site {config.site_id}: {e}")
            return False
    
    async def get_site(self, site_id: str) -> Optional[WordPressSiteConfig]:
        """Get site configuration by ID"""
        try:
            if self.supabase:
                response = self.supabase.table("wordpress_sites").select("*").eq("site_id", site_id).execute()
                if response.data:
                    return WordPressSiteConfig(**response.data[0])
            else:
                return self._site_cache.get(site_id)
        except Exception as e:
            logger.error(f"Failed to get site {site_id}: {e}")
            return None
    
    async def get_sites(
        self,
        active_only: bool = True,
        site_ids: Optional[List[str]] = None
    ) -> List[WordPressSiteConfig]:
        """
        Get multiple site configurations
        
        Args:
            active_only: Only return active sites
            site_ids: Filter by specific site IDs
            
        Returns:
            List of WordPressSiteConfig objects
        """
        try:
            if self.supabase:
                query = self.supabase.table("wordpress_sites").select("*")
                
                if active_only:
                    query = query.eq("active", True)
                
                if site_ids:
                    query = query.in_("site_id", site_ids)
                
                response = query.execute()
                return [WordPressSiteConfig(**site) for site in response.data]
            else:
                # Fallback to cache
                sites = list(self._site_cache.values())
                if active_only:
                    sites = [s for s in sites if s.active]
                if site_ids:
                    sites = [s for s in sites if s.site_id in site_ids]
                return sites
                
        except Exception as e:
            logger.error(f"Failed to get sites: {e}")
            return []
    
    async def validate_site(self, site_id: str) -> Dict[str, Any]:
        """
        Validate site configuration and connectivity
        
        Args:
            site_id: Site ID to validate
            
        Returns:
            Validation result dictionary
        """
        site = await self.get_site(site_id)
        if not site:
            return {
                "valid": False,
                "error": "Site not found"
            }
        
        # Basic validation
        is_valid = (
            site.active and
            len(site.site_url) > 0 and
            len(site.username) > 0 and
            len(site.application_password) > 0
        )
        
        # TODO: Add connectivity test (ping WordPress REST API)
        
        return {
            "valid": is_valid,
            "site_id": site_id,
            "site_name": site.site_name,
            "site_url": str(site.site_url),
            "active": site.active
        }
    
    async def update_site(self, site_id: str, updates: Dict[str, Any]) -> bool:
        """Update site configuration"""
        try:
            site = await self.get_site(site_id)
            if not site:
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(site, key):
                    setattr(site, key, value)
            
            site.updated_at = datetime.now()
            
            # Save
            return await self.register_site(site)
            
        except Exception as e:
            logger.error(f"Failed to update site {site_id}: {e}")
            return False

