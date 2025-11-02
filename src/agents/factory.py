"""
Factory Functions for Agent-Based CMS
Easy setup and configuration of the complete CMS system
"""

import logging
from typing import Optional, List

from langchain_openai import ChatOpenAI

from .research_agent import ResearchAgent
from .writing_agent import WritingAgent
from .affiliate_agent import AffiliateAgent
from .image_agent import ImageAgent
from .publishing_agent import PublishingAgent
from .orchestrator import ArticleCMSOrchestrator

logger = logging.getLogger(__name__)


def create_agent_based_cms(
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    enable_research: bool = True,
    enable_writing: bool = True,
    enable_affiliate: bool = True,
    enable_images: bool = True,
    enable_publishing: bool = True,
    max_affiliate_links: int = 5,
    max_images: int = 5,
    enable_checkpoints: bool = True,
    llm: Optional[ChatOpenAI] = None,
    **kwargs
) -> ArticleCMSOrchestrator:
    """
    Create a complete Agent-Based CMS orchestrator with all agents
    
    Args:
        llm_model: LLM model name for agents
        temperature: LLM temperature
        enable_research: Enable research agent
        enable_writing: Enable writing agent
        enable_affiliate: Enable affiliate link agent
        enable_images: Enable image agent
        enable_publishing: Enable publishing agent
        max_affiliate_links: Maximum affiliate links per article
        max_images: Maximum images per article
        enable_checkpoints: Enable state checkpointing
        llm: Optional pre-configured LLM instance
        **kwargs: Additional arguments passed to agents
        
    Returns:
        Configured ArticleCMSOrchestrator
    """
    import os
    
    # Create LLM if not provided
    if llm is None:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it or provide a pre-configured llm instance."
            )
        llm = ChatOpenAI(model=llm_model, temperature=temperature)
    
    # Create agents
    research_agent = None
    if enable_research:
        research_agent = ResearchAgent(llm=llm, **kwargs)
        logger.info("✅ Research Agent initialized")
    
    writing_agent = None
    if enable_writing:
        writing_agent = WritingAgent(llm=llm, **kwargs)
        logger.info("✅ Writing Agent initialized")
    
    affiliate_agent = None
    if enable_affiliate:
        affiliate_agent = AffiliateAgent(
            llm=llm,
            max_links_per_article=max_affiliate_links,
            **kwargs
        )
        logger.info("✅ Affiliate Agent initialized")
    
    image_agent = None
    if enable_images:
        # Initialize enhanced image system components
        playwright_engine = None
        dataforseo_client = None
        gemini_api_key = None
        bulletproof_uploader = None
        
        try:
            # Try to initialize Playwright engine
            try:
                from src.integrations.playwright_screenshot_engine import (
                    ScreenshotService,
                    BrowserPoolManager,
                    ScreenshotConfig
                )
                # Initialize browser pool and service
                browser_pool = BrowserPoolManager()
                config = ScreenshotConfig()
                playwright_engine = ScreenshotService(
                    browser_pool=browser_pool,
                    config=config
                )
            except Exception as e:
                logger.warning(f"Playwright engine not available: {e}")
            
            # Try to initialize DataForSEO client
            try:
                from src.integrations.dataforseo_image_search import DataForSEOImageSearch
                dataforseo_client = DataForSEOImageSearch()
            except Exception as e:
                logger.warning(f"DataForSEO client not available: {e}")
            
            # Get Gemini API key for image generation
            import os
            gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                logger.info("✅ Gemini API key found for image generation")
            else:
                logger.warning("⚠️ Gemini API key not found (GOOGLE_API_KEY or GEMINI_API_KEY)")
            
            # Try to initialize bulletproof uploader
            try:
                from src.integrations.bulletproof_image_integrator import BulletproofImageUploader
                supabase_client = kwargs.get("supabase_client")
                if not supabase_client:
                    # Try to initialize Supabase client from environment
                    import os
                    from supabase import create_client
                    supabase_url = os.getenv("SUPABASE_URL")
                    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
                    if supabase_url and supabase_key:
                        supabase_client = create_client(supabase_url, supabase_key)
                
                if supabase_client:
                    bulletproof_uploader = BulletproofImageUploader(supabase_client)
            except Exception as e:
                logger.warning(f"Bulletproof uploader not available: {e}")
            
        except Exception as e:
            logger.warning(f"Some enhanced image system components unavailable: {e}")
        
        image_agent = ImageAgent(
            llm=llm,
            max_images=max_images,
            upload_to_wordpress=True,
            playwright_engine=playwright_engine,
            dataforseo_client=dataforseo_client,
            gemini_api_key=gemini_api_key,
            bulletproof_uploader=bulletproof_uploader,
            **kwargs
        )
        logger.info("✅ Image Agent initialized with enhanced image system")
    
    publishing_agent = None
    if enable_publishing:
        publishing_agent = PublishingAgent(llm=llm, **kwargs)
        logger.info("✅ Publishing Agent initialized")
    
    # Create orchestrator
    orchestrator = ArticleCMSOrchestrator(
        research_agent=research_agent,
        writing_agent=writing_agent,
        affiliate_agent=affiliate_agent,
        image_agent=image_agent,
        publishing_agent=publishing_agent,
        enable_checkpoints=enable_checkpoints
    )
    
    logger.info("✅ Agent-Based CMS Orchestrator created successfully")
    
    return orchestrator

