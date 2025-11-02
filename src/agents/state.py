"""
State Schema Definitions for Agent-Based CMS
Defines the state structure used throughout the LangGraph workflow
"""

from typing import TypedDict, Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class AgentStatus(str, Enum):
    """Status of an agent execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArticleCMSState(TypedDict):
    """
    Main state schema for the Article CMS workflow
    
    This state is passed between agents in the LangGraph workflow
    """
    # Input
    query: str
    target_sites: List[str]  # WordPress site IDs to publish to
    
    # Research Phase
    research_data: Dict[str, Any]  # Comprehensive research findings
    research_urls: List[str]  # URLs used in research
    screenshots: List[Dict[str, Any]]  # Screenshot metadata
    structured_data: Dict[str, Any]  # 95-field casino intelligence
    
    # Writing Phase
    draft_content: str  # Initial draft
    final_content: str  # Final polished content
    content_revisions: List[str]  # Revision history
    seo_metadata: Dict[str, Any]  # SEO title, description, keywords
    
    # Affiliate Links Phase
    affiliate_links: List[Dict[str, Any]]  # Inserted affiliate links
    affiliate_opportunities: List[Dict[str, Any]]  # Detected opportunities
    tracking_codes: Dict[str, str]  # UTM parameters per link
    
    # Image Phase
    images: List[Dict[str, Any]]  # Selected images with metadata
    wordpress_media_ids: List[int]  # WordPress media library IDs
    image_alt_texts: Dict[str, str]  # Alt text per image
    
    # Publishing Phase
    published_posts: List[Dict[str, Any]]  # Published post metadata
    site_statuses: Dict[str, str]  # Status per site (success/failed)
    post_urls: Dict[str, str]  # Published post URLs
    
    # Metadata & Control
    errors: List[str]  # Error messages
    warnings: List[str]  # Warning messages
    metadata: Dict[str, Any]  # Additional metadata
    agent_statuses: Dict[str, str]  # Status of each agent
    
    # Workflow Control
    current_agent: Optional[str]  # Currently executing agent
    workflow_step: int  # Current step in workflow
    requires_human_review: bool  # Flag for human-in-the-loop


class AgentState(BaseModel):
    """
    State information for individual agent execution
    """
    agent_name: str = Field(description="Name of the agent")
    status: AgentStatus = Field(default=AgentStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ResearchAgentState(BaseModel):
    """Specific state for Research Agent"""
    research_completed: bool = False
    sources_found: int = 0
    screenshots_captured: int = 0
    structured_data_extracted: bool = False
    quality_score: Optional[float] = None


class WritingAgentState(BaseModel):
    """Specific state for Writing Agent"""
    draft_created: bool = False
    content_length: int = 0
    revisions_made: int = 0
    quality_score: Optional[float] = None
    seo_optimized: bool = False


class AffiliateAgentState(BaseModel):
    """Specific state for Affiliate Link Agent"""
    opportunities_detected: int = 0
    links_inserted: int = 0
    links_validated: int = 0
    tracking_added: bool = False


class ImageAgentState(BaseModel):
    """Specific state for Image Agent"""
    images_searched: int = 0
    images_selected: int = 0
    images_uploaded: int = 0
    alt_texts_generated: int = 0


class PublishingAgentState(BaseModel):
    """Specific state for Publishing Agent"""
    sites_targeted: int = 0
    sites_published: int = 0
    sites_failed: int = 0
    content_adapted: bool = False


def create_initial_state(query: str, target_sites: List[str] = None) -> ArticleCMSState:
    """
    Create initial state for the CMS workflow
    
    Args:
        query: The content query/topic
        target_sites: List of WordPress site IDs to publish to
        
    Returns:
        Initialized ArticleCMSState
    """
    return {
        "query": query,
        "target_sites": target_sites or [],
        "research_data": {},
        "research_urls": [],
        "screenshots": [],
        "structured_data": {},
        "draft_content": "",
        "final_content": "",
        "content_revisions": [],
        "seo_metadata": {},
        "affiliate_links": [],
        "affiliate_opportunities": [],
        "tracking_codes": {},
        "images": [],
        "wordpress_media_ids": [],
        "image_alt_texts": {},
        "published_posts": [],
        "site_statuses": {},
        "post_urls": {},
        "errors": [],
        "warnings": [],
        "metadata": {},
        "agent_statuses": {},
        "current_agent": None,
        "workflow_step": 0,
        "requires_human_review": False,
    }

