"""
Image Agent for Agent-Based CMS
Searches, selects, and uploads images for content using enhanced multi-strategy system
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent, AgentResult
from .state import ArticleCMSState
from .tools.image_tools import (
    image_search_tool,
    image_selection_tool,
    alt_text_generation_tool,
    wordpress_image_upload_tool
)

# Import enhanced image system
try:
    from src.integrations.enhanced_image_system import (
        EnhancedImageSystem,
        ContentTypeDetector,
        ContentType
    )
    ENHANCED_IMAGE_SYSTEM_AVAILABLE = True
except ImportError:
    ENHANCED_IMAGE_SYSTEM_AVAILABLE = False
    EnhancedImageSystem = None
    ContentTypeDetector = None
    ContentType = None

logger = logging.getLogger(__name__)


class ImageAgent(BaseAgent):
    """
    Image Agent - Handles image search, selection, and WordPress upload
    
    Uses tools:
    - image_search_tool: Search for images using DataForSEO
    - image_selection_tool: Select best images for content
    - alt_text_generation_tool: Generate SEO alt text
    - wordpress_image_upload_tool: Upload to WordPress media library
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_images: int = 5,
        upload_to_wordpress: bool = True,
        playwright_engine=None,
        dataforseo_client=None,
        gemini_api_key=None,
        bulletproof_uploader=None,
        **kwargs
    ):
        """
        Initialize Image Agent with enhanced multi-strategy system
        
        Args:
            llm: Language model for agent reasoning
            max_images: Maximum images to select and upload
            upload_to_wordpress: Whether to upload images to WordPress
            playwright_engine: Playwright engine for casino screenshots
            dataforseo_client: DataForSEO client for image search
            gemini_api_key: Gemini API key for image generation
            bulletproof_uploader: Bulletproof image uploader
        """
        tools = [
            image_search_tool,
            image_selection_tool,
            alt_text_generation_tool,
        ]
        
        if upload_to_wordpress:
            tools.append(wordpress_image_upload_tool)
        
        super().__init__(
            name="image_agent",
            llm=llm,
            tools=tools,
            **kwargs
        )
        
        self.max_images = max_images
        self.upload_to_wordpress = upload_to_wordpress
        
        # Initialize enhanced image system if available
        if ENHANCED_IMAGE_SYSTEM_AVAILABLE:
            self.enhanced_image_system = EnhancedImageSystem(
                playwright_engine=playwright_engine,
                dataforseo_client=dataforseo_client,
                gemini_api_key=gemini_api_key,
                bulletproof_uploader=bulletproof_uploader
            )
        else:
            self.enhanced_image_system = None
    
    async def execute(self, state: ArticleCMSState) -> AgentResult:
        """
        Execute image agent logic
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResult with images selected and uploaded
        """
        content = state.get("final_content", "") or state.get("draft_content", "")
        query = state.get("query", "")
        target_sites = state.get("target_sites", [])
        
        if not content:
            return AgentResult(
                success=False,
                error="No content available for image processing"
            )
        
        try:
            # Use enhanced image system if available
            if self.enhanced_image_system:
                logger.info("Image Agent: Using enhanced multi-strategy image system")
                images, strategy = await self.enhanced_image_system.acquire_images(
                    query=query,
                    content=content,
                    max_images=self.max_images
                )
                
                # Detect content type for contextual embedding
                content_type = ContentTypeDetector.detect(query, content)
                
                # Embed images contextually into content
                enhanced_content = self.enhanced_image_system.embed_images_contextually(
                    content=content,
                    images=images,
                    content_type=content_type
                )
                
                selected_images = images
            else:
                # Fallback to original logic
                logger.info("Image Agent: Using fallback image search")
                
                # Step 1: Generate image search queries from content
                search_queries = self._generate_image_queries(query, content)
                
                # Step 2: Search for images
                all_images = []
                for search_query in search_queries[:3]:  # Limit to 3 searches
                    logger.info(f"Image Agent: Searching for images: {search_query}")
                    search_result = await image_search_tool.ainvoke({
                        "query": search_query,
                        "max_results": 20
                    })
                    
                    images = search_result.get("images", [])
                    all_images.extend(images)
                
                if not all_images:
                    logger.warning("Image Agent: No images found")
                    return AgentResult(
                        success=True,
                        state_updates={
                            "images": [],
                            "wordpress_media_ids": [],
                            "image_alt_texts": {},
                            "final_content": content  # Return original content
                        },
                        metadata={"images_found": 0}
                    )
                
                # Step 3: Select best images
                logger.info(f"Image Agent: Selecting best images from {len(all_images)} candidates")
                selection_result = await image_selection_tool.ainvoke({
                    "images": all_images,
                    "content": content,
                    "max_select": self.max_images
                })
                
                selected_images = selection_result.get("selected_images", [])
                enhanced_content = content  # No embedding in fallback mode
            
            # Step 4: Generate alt text for selected images
            images_with_alt = []
            alt_texts = {}
            
            for img in selected_images:
                image_url = img.get("url", "")
                if not image_url:
                    continue
                
                logger.info(f"Image Agent: Generating alt text for image")
                alt_result = await alt_text_generation_tool.ainvoke({
                    "image_url": image_url,
                    "context": content[:500]  # First 500 chars for context
                })
                
                alt_text = alt_result.get("alt_text", img.get("title", "Image"))
                alt_texts[image_url] = alt_text
                
                images_with_alt.append({
                    **img,
                    "alt_text": alt_text
                })
            
            # Step 5: Upload to WordPress if enabled and sites configured
            wordpress_media_ids = []
            image_metadata = []
            
            if self.upload_to_wordpress and target_sites:
                # Get first site config (in full implementation, would get from registry)
                # For now, use environment variables
                site_config = {
                    "site_url": "",  # Will be populated from target_sites registry
                    "username": "",
                    "application_password": ""
                }
                
                # Upload each selected image
                for img in images_with_alt[:self.max_images]:
                    image_url = img.get("url", "")
                    alt_text = img.get("alt_text", "")
                    
                    if not image_url:
                        continue
                    
                    try:
                        logger.info(f"Image Agent: Uploading image to WordPress: {image_url[:50]}...")
                        upload_result = await wordpress_image_upload_tool.ainvoke({
                            "image_url": image_url,
                            "site_config": site_config,
                            "alt_text": alt_text,
                            "title": img.get("title", "")
                        })
                        
                        if upload_result.get("success"):
                            media_id = upload_result.get("media_id")
                            wordpress_url = upload_result.get("wordpress_url", image_url)
                            
                            wordpress_media_ids.append(media_id)
                            image_metadata.append({
                                **img,
                                "wordpress_media_id": media_id,
                                "wordpress_url": wordpress_url
                            })
                        else:
                            # Fallback to original URL
                            image_metadata.append(img)
                            
                    except Exception as e:
                        logger.warning(f"Image upload failed for {image_url}: {e}")
                        image_metadata.append(img)
            else:
                # No WordPress upload, just use original images
                image_metadata = images_with_alt
            
            # Prepare state updates
            state_updates = {
                "images": image_metadata,
                "wordpress_media_ids": wordpress_media_ids,
                "image_alt_texts": alt_texts,
                "final_content": enhanced_content if self.enhanced_image_system else state.get("final_content", content),
                "workflow_step": state.get("workflow_step", 0) + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "images_searched": len(selected_images) if self.enhanced_image_system else len(all_images) if 'all_images' in locals() else 0,
                    "images_selected": len(selected_images),
                    "images_uploaded": len(wordpress_media_ids),
                    "content_type": content_type.value if self.enhanced_image_system and 'content_type' in locals() else "unknown"
                }
            }
            
            logger.info(f"Image Agent: Processed {len(selected_images)} images, uploaded {len(wordpress_media_ids)} to WordPress")
            
            return AgentResult(
                success=True,
                state_updates=state_updates,
                metadata={
                    "images_selected": len(selected_images),
                    "images_uploaded": len(wordpress_media_ids),
                    "alt_texts_generated": len(alt_texts)
                }
            )
            
        except Exception as e:
            logger.error(f"Image Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def _generate_image_queries(self, query: str, content: str) -> List[str]:
        """Generate image search queries from query and content"""
        queries = []
        
        # Extract key terms from query
        query_terms = query.lower().split()
        if len(query_terms) >= 2:
            queries.append(" ".join(query_terms[:3]))  # First 3 terms
        
        # Extract key terms from content (first 200 chars)
        content_preview = content[:200].lower()
        content_words = [w for w in content_preview.split() if len(w) > 4][:5]
        if content_words:
            queries.append(" ".join(content_words[:3]))
        
        # Add generic casino-related queries if applicable
        if "casino" in query.lower() or "casino" in content_preview:
            casino_name = query.split()[0] if query.split() else "casino"
            queries.append(f"{casino_name} casino")
            queries.append(f"{casino_name} games")
        
        # Remove duplicates and limit
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:5]  # Max 5 queries

