"""
Writing Agent for Agent-Based CMS
Generates high-quality content using Universal RAG Chain and templates
"""

import logging
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent, AgentResult
from .state import ArticleCMSState
from .tools.writing_tools import (
    content_generation_tool,
    template_selection_tool,
    content_refinement_tool,
    seo_optimization_tool
)

logger = logging.getLogger(__name__)


class WritingAgent(BaseAgent):
    """
    Writing Agent - Generates high-quality content from research
    
    Uses tools:
    - content_generation_tool: Uses Universal RAG Chain
    - template_selection_tool: Chooses appropriate template
    - content_refinement_tool: Improves content quality
    - seo_optimization_tool: Optimizes for SEO
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_refinement: bool = True,
        enable_seo: bool = True,
        **kwargs
    ):
        """
        Initialize Writing Agent
        
        Args:
            llm: Language model for agent reasoning
            enable_refinement: Whether to refine content
            enable_seo: Whether to optimize for SEO
        """
        tools = [
            content_generation_tool,
            template_selection_tool,
        ]
        
        if enable_refinement:
            tools.append(content_refinement_tool)
        
        if enable_seo:
            tools.append(seo_optimization_tool)
        
        super().__init__(
            name="writing_agent",
            llm=llm,
            tools=tools,
            **kwargs
        )
        
        self.enable_refinement = enable_refinement
        self.enable_seo = enable_seo
    
    async def execute(self, state: ArticleCMSState) -> AgentResult:
        """
        Execute writing agent logic
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResult with generated content
        """
        query = state.get("query", "")
        research_data = state.get("research_data", {})
        
        if not query:
            return AgentResult(
                success=False,
                error="No query provided for content generation"
            )
        
        try:
            # Step 1: Select appropriate template
            logger.info(f"Writing Agent: Selecting template for query")
            template_result = await template_selection_tool.ainvoke({
                "query": query
            })
            
            template_id = template_result.get("template_id", "default")
            query_type = template_result.get("query_type", "general_info")
            expertise_level = template_result.get("expertise_level", "intermediate")
            
            # Step 2: Generate initial content
            logger.info(f"Writing Agent: Generating content with template {template_id}")
            generation_result = await content_generation_tool.ainvoke({
                "query": query,
                "research_data": research_data,
                "context": f"Template: {template_id}, Type: {query_type}, Level: {expertise_level}"
            })
            
            draft_content = generation_result.get("content", "")
            confidence_score = generation_result.get("confidence_score", 0.0)
            
            if not draft_content:
                return AgentResult(
                    success=False,
                    error="Content generation returned empty content"
                )
            
            # Step 3: Refine content if enabled
            refined_content = draft_content
            improvements_made = []
            if self.enable_refinement:
                logger.info("Writing Agent: Refining content")
                refinement_result = await content_refinement_tool.ainvoke({
                    "content": draft_content,
                    "improvement_areas": ["structure", "readability", "engagement"]
                })
                refined_content = refinement_result.get("refined_content", draft_content)
                improvements_made = refinement_result.get("improvements_made", [])
            
            # Step 4: SEO optimization if enabled
            seo_metadata = {}
            if self.enable_seo:
                logger.info("Writing Agent: Optimizing for SEO")
                seo_result = await seo_optimization_tool.ainvoke({
                    "content": refined_content
                })
                seo_metadata = seo_result.get("seo_metadata", {})
            
            # Step 5: Add authoritative links (non-competitor organizations)
            content_with_links = refined_content
            try:
                from src.chains.authoritative_hyperlink_engine import (
                    AuthoritativeHyperlinkEngine,
                    LinkGenerationConfig
                )
                # ✅ FIXED: Create config first, then engine
                config = LinkGenerationConfig(
                    max_links_per_content=8,
                    max_links_per_category=3,
                    min_confidence_score=0.7
                )
                hyperlink_engine = AuthoritativeHyperlinkEngine(config=config)
                link_result = await hyperlink_engine.generate_hyperlinks(
                    content=refined_content,
                    query=query
                )
                if link_result and link_result.get("enhanced_content"):
                    content_with_links = link_result.get("enhanced_content", refined_content)
                    links_added = link_result.get("links_added", 0)
                    logger.info(f"Writing Agent: Added {links_added} authoritative links")
            except Exception as e:
                logger.warning(f"Writing Agent: Failed to add authoritative links: {e}")
                # Continue without links
            
            # Step 6: Ensure content is HTML formatted (if not already)
            final_html_content = self._ensure_html_format(content_with_links)

            # ✅ FIXED: Force fallback - if final_content is empty, use draft_content
            if not final_html_content or final_html_content.strip() == "":
                logger.warning(f"Writing Agent: Final content empty, falling back to draft content")
                final_html_content = self._ensure_html_format(draft_content)

            # Prepare state updates
            state_updates = {
                "draft_content": draft_content,
                "final_content": final_html_content,  # Return HTML-formatted content
                "content_revisions": [draft_content, refined_content] if refined_content != draft_content else [draft_content],
                "seo_metadata": seo_metadata,
                "workflow_step": state.get("workflow_step", 0) + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "template_used": template_id,
                    "query_type": query_type,
                    "expertise_level": expertise_level,
                    "improvements_made": improvements_made,
                    "generation_confidence": confidence_score,
                    "html_formatted": True
                }
            }
            
            logger.info(f"Writing Agent: Generated {len(final_html_content)} characters of HTML content (confidence: {confidence_score:.2f})")
            
            return AgentResult(
                success=True,
                state_updates=state_updates,
                metadata={
                    "content_length": len(final_html_content),
                    "confidence_score": confidence_score,
                    "template_used": template_id,
                    "improvements_count": len(improvements_made),
                    "html_formatted": True
                }
            )
            
        except Exception as e:
            logger.error(f"Writing Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def _ensure_html_format(self, content: str) -> str:
        """Ensure content is properly formatted as HTML"""
        try:
            from bs4 import BeautifulSoup
            
            # Check if already HTML
            if content.strip().startswith('<') and '<html' not in content.lower():
                # Already HTML but not full document - just return as is
                return content
            
            # Convert markdown/plain text to HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # If no HTML structure, convert from markdown/plain text
            if not soup.find_all(['p', 'div', 'h1', 'h2', 'h3']):
                lines = content.split('\n')
                html_parts = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Headers
                    if line.startswith('# '):
                        html_parts.append(f'<h1>{line[2:]}</h1>')
                    elif line.startswith('## '):
                        html_parts.append(f'<h2>{line[3:]}</h2>')
                    elif line.startswith('### '):
                        html_parts.append(f'<h3>{line[4:]}</h3>')
                    # Lists
                    elif line.startswith('- ') or line.startswith('* '):
                        if not html_parts or not html_parts[-1].startswith('<ul'):
                            html_parts.append('<ul>')
                        html_parts.append(f'<li>{line[2:]}</li>')
                    # Tables (basic markdown table support)
                    elif '|' in line and line.count('|') >= 2:
                        if not html_parts or not html_parts[-1].startswith('<table'):
                            html_parts.append('<table class="wp-table"><tbody>')
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        html_parts.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
                    # Paragraphs
                    else:
                        html_parts.append(f'<p>{line}</p>')
                
                # Close open tags
                if html_parts and html_parts[-1].startswith('<ul'):
                    html_parts.append('</ul>')
                if html_parts and html_parts[-1].startswith('<table'):
                    html_parts.append('</tbody></table>')
                
                return '\n'.join(html_parts)
            
            return str(soup)
            
        except Exception as e:
            logger.warning(f"HTML formatting failed: {e}, returning original content")
            return content

