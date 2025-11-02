"""
Affiliate Link Agent for Agent-Based CMS
Inserts and manages affiliate links intelligently
"""

import logging
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent, AgentResult
from .state import ArticleCMSState
from .tools.affiliate_tools import (
    affiliate_link_database_tool,
    link_insertion_tool,
    link_validation_tool,
    tracking_parameter_tool
)

logger = logging.getLogger(__name__)


class AffiliateAgent(BaseAgent):
    """
    Affiliate Link Agent - Inserts affiliate links contextually
    
    Uses tools:
    - affiliate_link_database_tool: Query affiliate link registry
    - link_insertion_tool: Insert links into content
    - link_validation_tool: Verify link validity
    - tracking_parameter_tool: Add UTM parameters
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        max_links_per_article: int = 5,
        **kwargs
    ):
        """
        Initialize Affiliate Agent
        
        Args:
            llm: Language model for agent reasoning
            max_links_per_article: Maximum links to insert per article
        """
        tools = [
            affiliate_link_database_tool,
            link_insertion_tool,
            link_validation_tool,
            tracking_parameter_tool
        ]
        
        super().__init__(
            name="affiliate_agent",
            llm=llm,
            tools=tools,
            **kwargs
        )
        
        self.max_links_per_article = max_links_per_article
    
    async def execute(self, state: ArticleCMSState) -> AgentResult:
        """
        Execute affiliate agent logic
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResult with affiliate links inserted
        """
        content = state.get("final_content", "") or state.get("draft_content", "")
        query = state.get("query", "")
        
        if not content:
            return AgentResult(
                success=False,
                error="No content available for affiliate link insertion"
            )
        
        try:
            # Step 1: Detect affiliate link opportunities
            logger.info("Affiliate Agent: Detecting link opportunities")
            
            # Determine category from query/content
            category = self._detect_category(query, content)
            
            # Insert affiliate links
            logger.info(f"Affiliate Agent: Inserting links (category: {category}, max: {self.max_links_per_article})")
            insertion_result = await link_insertion_tool.ainvoke({
                "content": content,
                "category": category,
                "max_links": self.max_links_per_article
            })
            
            enhanced_content = insertion_result.get("enhanced_content", content)
            insertions = insertion_result.get("insertions", [])
            links_inserted = insertion_result.get("links_inserted", 0)
            
            # Extract affiliate links and tracking codes
            affiliate_links = []
            tracking_codes = {}
            
            for insertion in insertions:
                link_id = insertion.get("link_id")
                final_url = insertion.get("final_url")
                anchor_text = insertion.get("anchor_text")
                
                affiliate_links.append({
                    "link_id": link_id,
                    "url": final_url,
                    "anchor_text": anchor_text,
                    "position": insertion.get("position", 0)
                })
                
                # Extract tracking codes from URL
                if final_url:
                    tracking_codes[link_id] = final_url
            
            # Prepare state updates
            state_updates = {
                "final_content": enhanced_content,  # Update final content with links
                "affiliate_links": affiliate_links,
                "tracking_codes": tracking_codes,
                "workflow_step": state.get("workflow_step", 0) + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "affiliate_category": category,
                    "affiliate_links_inserted": links_inserted
                }
            }
            
            logger.info(f"Affiliate Agent: Inserted {links_inserted} affiliate links")
            
            return AgentResult(
                success=True,
                state_updates=state_updates,
                metadata={
                    "links_inserted": links_inserted,
                    "category": category,
                    "opportunities_found": len(insertions)
                }
            )
            
        except Exception as e:
            logger.error(f"Affiliate Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def _detect_category(self, query: str, content: str) -> Optional[str]:
        """Detect affiliate link category from query/content"""
        query_lower = query.lower()
        content_lower = content.lower()[:500]  # Check first 500 chars
        
        combined = f"{query_lower} {content_lower}"
        
        if "casino" in combined or "gambling" in combined:
            if "crypto" in combined or "bitcoin" in combined:
                return "crypto_casino"
            return "casino"
        elif "sportsbook" in combined or "betting" in combined:
            return "sportsbook"
        elif "bonus" in combined or "promotion" in combined:
            return "bonus"
        elif "payment" in combined or "deposit" in combined:
            return "payment"
        elif "software" in combined or "provider" in combined:
            return "software"
        
        return None  # Let the tool handle default

