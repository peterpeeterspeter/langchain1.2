"""
Affiliate Link Schema for Agent-Based CMS
Defines the structure for affiliate link management
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl


class AffiliateLinkStatus(str, Enum):
    """Status of an affiliate link"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    EXPIRED = "expired"


class AffiliateLinkCategory(str, Enum):
    """Categories for affiliate links"""
    CASINO = "casino"
    SPORTSBOOK = "sportsbook"
    CRYPTO_CASINO = "crypto_casino"
    BONUS = "bonus"
    PAYMENT = "payment"
    SOFTWARE = "software"
    OTHER = "other"


class AffiliateLink(BaseModel):
    """
    Affiliate link model for database storage
    """
    id: str = Field(description="Unique identifier for the affiliate link")
    merchant: str = Field(description="Merchant/affiliate program name")
    product_name: str = Field(description="Product or service name")
    affiliate_url: HttpUrl = Field(description="Base affiliate URL")
    commission_rate: float = Field(default=0.0, ge=0.0, le=100.0, description="Commission rate percentage")
    keywords: List[str] = Field(default_factory=list, description="Keywords that trigger this link")
    category: AffiliateLinkCategory = Field(default=AffiliateLinkCategory.OTHER)
    status: AffiliateLinkStatus = Field(default=AffiliateLinkStatus.ACTIVE)
    tracking_template: str = Field(default="{url}?ref={tracking_id}", description="URL template with tracking parameters")
    tracking_id: str = Field(description="Affiliate tracking ID")
    
    # Optional fields
    description: Optional[str] = Field(default=None, description="Link description")
    image_url: Optional[HttpUrl] = Field(default=None, description="Product image URL")
    priority: int = Field(default=0, description="Priority for link insertion (higher = more priority)")
    max_uses_per_article: int = Field(default=3, description="Maximum times to use this link per article")
    min_content_length: int = Field(default=500, description="Minimum content length to use this link")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = Field(default=None)
    usage_count: int = Field(default=0, description="Total number of times used")
    
    # Validation
    active: bool = Field(default=True, description="Whether link is currently active")
    
    class Config:
        use_enum_values = True


class AffiliateLinkInsertion(BaseModel):
    """Record of affiliate link insertion in content"""
    link_id: str = Field(description="Affiliate link ID")
    article_id: Optional[str] = Field(default=None, description="Article identifier")
    inserted_at: datetime = Field(default_factory=datetime.now)
    position: int = Field(description="Character position in content")
    anchor_text: str = Field(description="Anchor text used")
    final_url: HttpUrl = Field(description="Final URL with tracking parameters")
    context: str = Field(description="Surrounding content context")


class AffiliateLinkOpportunity(BaseModel):
    """Detected opportunity for affiliate link insertion"""
    link_id: str = Field(description="Matching affiliate link ID")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for insertion")
    suggested_anchor_text: str = Field(description="Suggested anchor text")
    suggested_position: int = Field(description="Suggested character position")
    context_match: str = Field(description="Matching context snippet")
    keyword_match: str = Field(description="Matching keyword")


class UTMParameters(BaseModel):
    """UTM tracking parameters"""
    utm_source: str = Field(description="Traffic source")
    utm_medium: str = Field(default="affiliate", description="Traffic medium")
    utm_campaign: Optional[str] = Field(default=None, description="Campaign name")
    utm_term: Optional[str] = Field(default=None, description="Term/keyword")
    utm_content: Optional[str] = Field(default=None, description="Content identifier")
    
    def to_query_string(self) -> str:
        """Convert to URL query string"""
        params = {
            "utm_source": self.utm_source,
            "utm_medium": self.utm_medium
        }
        if self.utm_campaign:
            params["utm_campaign"] = self.utm_campaign
        if self.utm_term:
            params["utm_term"] = self.utm_term
        if self.utm_content:
            params["utm_content"] = self.utm_content
        
        return "&".join([f"{k}={v}" for k, v in params.items()])

