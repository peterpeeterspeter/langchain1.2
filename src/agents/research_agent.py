"""
Research Agent for Agent-Based CMS
Gathers comprehensive information using web search, comprehensive research, and screenshots
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent, AgentResult
from .state import ArticleCMSState
from .tools.research_tools import (
    web_search_tool,
    comprehensive_research_tool,
    screenshot_tool,
    casino_intelligence_tool
)

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Research Agent - Gathers comprehensive information about topics
    
    Uses tools:
    - web_search_tool: Quick web search via Tavily
    - comprehensive_research_tool: Deep research with WebBaseLoader
    - screenshot_tool: Visual evidence capture
    - casino_intelligence_tool: Structured casino data extraction
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        enable_screenshots: bool = True,
        enable_comprehensive_research: bool = True,
        **kwargs
    ):
        """
        Initialize Research Agent
        
        Args:
            llm: Language model for agent reasoning
            enable_screenshots: Whether to capture screenshots
            enable_comprehensive_research: Whether to perform deep research
        """
        tools = [
            web_search_tool,
            comprehensive_research_tool,
            casino_intelligence_tool
        ]
        
        if enable_screenshots:
            tools.append(screenshot_tool)
        
        super().__init__(
            name="research_agent",
            llm=llm,
            tools=tools,
            **kwargs
        )
        
        self.enable_screenshots = enable_screenshots
        self.enable_comprehensive_research = enable_comprehensive_research
        
        # Research prompt template
        self.research_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research agent specialized in gathering comprehensive information.

Your goal is to:
1. Use web_search_tool for quick overview information
2. Use comprehensive_research_tool for deep, structured research
3. Use casino_intelligence_tool for casino-specific structured data
4. Use screenshot_tool to capture visual evidence of important pages

Decide which tools to use based on the query. Be thorough but efficient."""),
            HumanMessage(content="{query}")
        ])
    
    async def execute(self, state: ArticleCMSState) -> AgentResult:
        """
        Execute research agent logic
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResult with research findings
        """
        query = state.get("query", "")
        if not query:
            return AgentResult(
                success=False,
                error="No query provided for research"
            )
        
        try:
            # Extract casino name from query (simple extraction)
            casino_name = self._extract_casino_name(query)
            
            # Step 1: Quick web search for overview (optional - requires Tavily)
            web_results = []
            try:
                logger.info(f"Research Agent: Performing web search for: {query}")
                web_results = await web_search_tool.ainvoke({"query": query})
            except Exception as e:
                logger.warning(f"Web search unavailable (Tavily not configured): {e}")
                # Continue without web search - comprehensive research will still work
            
            # Step 2: Comprehensive research if enabled
            research_data = {}
            urls_used = []
            if self.enable_comprehensive_research and casino_name:
                logger.info(f"Research Agent: Performing comprehensive research for: {casino_name}")
                comprehensive_result = await comprehensive_research_tool.ainvoke({
                    "query": query,
                    "base_domain": None,  # Auto-detect
                    "categories": None  # All categories
                })
                research_data = comprehensive_result.get("research_data", {})
                urls_used = comprehensive_result.get("urls_used", [])
            
            # Step 3: Casino intelligence extraction
            structured_data = {}
            if casino_name:
                logger.info(f"Research Agent: Extracting casino intelligence for: {casino_name}")
                intelligence_result = await casino_intelligence_tool.ainvoke({
                    "casino_name": casino_name,
                    "extract_all_fields": True
                })
                structured_data = intelligence_result.get("data", {})
            
            # Step 4: Capture screenshots of key URLs
            screenshots = []
            if self.enable_screenshots and urls_used:
                # Capture screenshots of top 3 URLs
                for url in urls_used[:3]:
                    try:
                        logger.info(f"Research Agent: Capturing screenshot of: {url}")
                        screenshot_result = await screenshot_tool.ainvoke({
                            "url": url,
                            "screenshot_type": "full_page"
                        })
                        if screenshot_result.get("success"):
                            screenshots.append(screenshot_result)
                    except Exception as e:
                        logger.warning(f"Screenshot capture failed for {url}: {e}")
            
            # Prepare state updates
            state_updates = {
                "research_data": {
                    "web_search_results": web_results,
                    "comprehensive_research": research_data,
                    "structured_intelligence": structured_data,
                    "research_quality": self._calculate_quality_score(web_results, research_data, structured_data)
                },
                "research_urls": urls_used,
                "screenshots": screenshots,
                "structured_data": structured_data,
                "workflow_step": state.get("workflow_step", 0) + 1
            }
            
            logger.info(f"Research Agent: Completed research - {len(web_results)} web results, {len(urls_used)} URLs, {len(screenshots)} screenshots")
            
            return AgentResult(
                success=True,
                state_updates=state_updates,
                metadata={
                    "web_results_count": len(web_results),
                    "urls_researched": len(urls_used),
                    "screenshots_captured": len(screenshots),
                    "structured_fields": len(structured_data) if isinstance(structured_data, dict) else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Research Agent execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def _extract_casino_name(self, query: str) -> Optional[str]:
        """Extract casino name from query"""
        # Simple extraction - can be enhanced with LLM
        query_lower = query.lower()
        
        # Common casino name patterns
        casino_keywords = ["casino", "betway", "betsson", "trustdice", "stake", "roobet", 
                          "bc.game", "crashino", "napoleon", "ladbrokes", "bet365"]
        
        for keyword in casino_keywords:
            if keyword in query_lower:
                # Extract the word before "casino" or use the keyword itself
                if keyword == "casino":
                    # Try to find casino name before "casino"
                    parts = query_lower.split("casino")
                    if parts[0]:
                        return parts[0].strip().split()[-1]  # Last word before casino
                else:
                    return keyword.replace(".", "").title()
        
        return None
    
    def _calculate_quality_score(
        self,
        web_results: List[Dict[str, Any]],
        research_data: Dict[str, Any],
        structured_data: Dict[str, Any]
    ) -> float:
        """Calculate research quality score"""
        score = 0.0
        
        # Web search quality (0-0.3)
        if web_results:
            score += min(len(web_results) / 10.0, 0.3)
        
        # Comprehensive research quality (0-0.4)
        if research_data:
            docs_count = research_data.get("total_documents", 0)
            score += min(docs_count / 20.0, 0.4)
        
        # Structured data quality (0-0.3)
        if structured_data:
            fields_populated = structured_data.get("extracted_fields", 0)
            score += min(fields_populated / 95.0, 0.3)
        
        return min(score, 1.0)

