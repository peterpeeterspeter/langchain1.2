"""
LangGraph Orchestrator for Agent-Based CMS
Main workflow orchestrator coordinating all agents using LangChain LCEL + LangGraph
"""

import logging
import uuid
from typing import Dict, Any, List, Optional

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ArticleCMSState, create_initial_state
from .base_agent import BaseAgent
from src.utils.langsmith_utils import get_langsmith_callbacks, get_langsmith_config

logger = logging.getLogger(__name__)


class ArticleCMSOrchestrator:
    """
    Main orchestrator for the Agent-Based CMS workflow
    
    Uses LangGraph StateGraph to coordinate:
    - Research Agent
    - Writing Agent
    - Affiliate Link Agent
    - Image Agent
    - Publishing Agent
    """
    
    def __init__(
        self,
        research_agent: Optional[BaseAgent] = None,
        writing_agent: Optional[BaseAgent] = None,
        affiliate_agent: Optional[BaseAgent] = None,
        image_agent: Optional[BaseAgent] = None,
        publishing_agent: Optional[BaseAgent] = None,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the orchestrator using LangChain LCEL + LangGraph
        
        Args:
            research_agent: Research agent instance
            writing_agent: Writing agent instance
            affiliate_agent: Affiliate link agent instance
            image_agent: Image agent instance
            publishing_agent: Publishing agent instance
            enable_checkpoints: Enable state checkpointing
        """
        self.research_agent = research_agent
        self.writing_agent = writing_agent
        self.affiliate_agent = affiliate_agent
        self.image_agent = image_agent
        self.publishing_agent = publishing_agent
        self.enable_checkpoints = enable_checkpoints
        
        # Build LCEL chains for each agent phase
        self.research_chain = self._build_research_lcel_chain()
        self.writing_chain = self._build_writing_lcel_chain()
        self.affiliate_chain = self._build_affiliate_lcel_chain()
        self.image_chain = self._build_image_lcel_chain()
        self.publishing_chain = self._build_publishing_lcel_chain()
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        # Add checkpointing if enabled
        if enable_checkpoints:
            checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.graph.compile()
        
        logger.info("ArticleCMSOrchestrator initialized with LangChain LCEL + LangGraph")
    
    def _build_research_lcel_chain(self):
        """Build LCEL chain for research phase"""
        if not self.research_agent:
            return RunnablePassthrough()
        
        async def run_research(state: ArticleCMSState) -> ArticleCMSState:
            """LCEL compatible async research runner"""
            try:
                logger.info(f"Research LCEL chain: Starting for query: {state.get('query', '')[:50]}")
                result = await self.research_agent.run(state)
                logger.info(f"Research LCEL chain: Completed successfully")
                return result
            except Exception as e:
                logger.error(f"Research LCEL chain error: {e}", exc_info=True)
                state["errors"].append(f"Research failed: {str(e)}")
                return state
        
        return RunnableLambda(run_research)
    
    def _build_writing_lcel_chain(self):
        """Build LCEL chain for writing phase"""
        if not self.writing_agent:
            return RunnablePassthrough()
        
        async def run_writing(state: ArticleCMSState) -> ArticleCMSState:
            """LCEL compatible async writing runner"""
            try:
                logger.info(f"Writing LCEL chain: Starting with research data available: {bool(state.get('research_data'))}")
                result = await self.writing_agent.run(state)
                content = result.get('final_content', '') or result.get('draft_content', '')
                logger.info(f"Writing LCEL chain: Generated {len(content)} characters")
                return result
            except Exception as e:
                logger.error(f"Writing LCEL chain error: {e}", exc_info=True)
                state["errors"].append(f"Writing failed: {str(e)}")
                return state
        
        return RunnableLambda(run_writing)
    
    def _build_affiliate_lcel_chain(self):
        """Build LCEL chain for affiliate link phase"""
        if not self.affiliate_agent:
            return RunnablePassthrough()

        async def run_affiliate(state: ArticleCMSState) -> Dict[str, Any]:
            """LCEL compatible async affiliate runner - returns updates only"""
            updated_state = await self.affiliate_agent.run(state)
            # Return only fields that affiliate agent modifies
            updates = {}
            affiliate_fields = {
                'affiliate_links', 'affiliate_opportunities',
                'tracking_codes',             }
            for key in affiliate_fields:
                if key in updated_state:
                    updates[key] = updated_state[key]
            return updates

        return RunnableLambda(run_affiliate)
    
    def _build_image_lcel_chain(self):
        """Build LCEL chain for image phase"""
        if not self.image_agent:
            return RunnablePassthrough()
        
        async def run_image(state: ArticleCMSState) -> Dict[str, Any]:
            """LCEL compatible async image runner - returns updates only"""
            updated_state = await self.image_agent.run(state)
            # Return only fields that image agent modifies
            updates = {}
            image_fields = {
                'images', 'wordpress_media_ids', 'image_alt_texts',
                'final_content',             }
            for key in image_fields:
                if key in updated_state:
                    updates[key] = updated_state[key]
            return updates

        return RunnableLambda(run_image)
    
    def _build_publishing_lcel_chain(self):
        """Build LCEL chain for publishing phase"""
        if not self.publishing_agent:
            return RunnablePassthrough()
        
        async def run_publishing(state: ArticleCMSState) -> ArticleCMSState:
            """LCEL compatible async publishing runner"""
            return await self.publishing_agent.run(state)
        
        return RunnableLambda(run_publishing)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph"""
        
        workflow = StateGraph(ArticleCMSState)
        
        # Add nodes for each agent
        if self.research_agent:
            workflow.add_node("research", self._run_research_agent)
        
        if self.writing_agent:
            workflow.add_node("writing", self._run_writing_agent)
        
        if self.affiliate_agent:
            workflow.add_node("affiliate", self._run_affiliate_agent)
        
        if self.image_agent:
            workflow.add_node("image", self._run_image_agent)
        
        if self.publishing_agent:
            workflow.add_node("publishing", self._run_publishing_agent)
        
        # Determine entry point (first available agent)
        entry_point = None
        if self.research_agent:
            entry_point = "research"
        elif self.writing_agent:
            entry_point = "writing"
        elif self.affiliate_agent:
            entry_point = "affiliate"
        elif self.image_agent:
            entry_point = "image"
        elif self.publishing_agent:
            entry_point = "publishing"
        
        if not entry_point:
            # No agents enabled - create a passthrough node
            def passthrough(state: ArticleCMSState) -> ArticleCMSState:
                return state
            workflow.add_node("passthrough", passthrough)
            workflow.set_entry_point("passthrough")
            workflow.add_edge("passthrough", END)
            return workflow
        
        # Set entry point
        workflow.set_entry_point(entry_point)
        current_node = entry_point
        
        # Research -> Writing
        if self.research_agent and self.writing_agent:
            workflow.add_edge("research", "writing")
            current_node = "writing"
        
        # Writing -> Affiliate & Image (parallel)
        if self.writing_agent:
            if self.affiliate_agent and self.image_agent:
                workflow.add_edge("writing", "affiliate")
                workflow.add_edge("writing", "image")
                # Both converge to publishing
                if self.publishing_agent:
                    workflow.add_edge("affiliate", "publishing")
                    workflow.add_edge("image", "publishing")
                    current_node = "publishing"
                else:
                    # No publishing - both end
                    workflow.add_edge("affiliate", END)
                    workflow.add_edge("image", END)
                    return workflow
            elif self.affiliate_agent:
                workflow.add_edge("writing", "affiliate")
                if self.publishing_agent:
                    workflow.add_edge("affiliate", "publishing")
                    current_node = "publishing"
                else:
                    workflow.add_edge("affiliate", END)
                    return workflow
            elif self.image_agent:
                workflow.add_edge("writing", "image")
                if self.publishing_agent:
                    workflow.add_edge("image", "publishing")
                    current_node = "publishing"
                else:
                    workflow.add_edge("image", END)
                    return workflow
            elif self.publishing_agent:
                workflow.add_edge("writing", "publishing")
                current_node = "publishing"
            else:
                # Writing is last node
                workflow.add_edge("writing", END)
                return workflow
        
        # Handle cases where writing is not enabled
        if not self.writing_agent:
            if self.research_agent:
                # Research -> next available
                if self.affiliate_agent:
                    workflow.add_edge("research", "affiliate")
                    current_node = "affiliate"
                elif self.image_agent:
                    workflow.add_edge("research", "image")
                    current_node = "image"
                elif self.publishing_agent:
                    workflow.add_edge("research", "publishing")
                    current_node = "publishing"
                else:
                    workflow.add_edge("research", END)
                    return workflow
            elif self.affiliate_agent:
                if self.image_agent:
                    workflow.add_edge("affiliate", "image")
                    current_node = "image"
                elif self.publishing_agent:
                    workflow.add_edge("affiliate", "publishing")
                    current_node = "publishing"
                else:
                    workflow.add_edge("affiliate", END)
                    return workflow
            elif self.image_agent:
                if self.publishing_agent:
                    workflow.add_edge("image", "publishing")
                    current_node = "publishing"
                else:
                    workflow.add_edge("image", END)
                    return workflow
        
        # End
        if self.publishing_agent:
            workflow.add_edge("publishing", END)
        elif current_node:
            workflow.add_edge(current_node, END)
        
        return workflow
    
    async def _run_research_agent(self, state: ArticleCMSState) -> ArticleCMSState:
        """Run research agent node using LCEL chain"""
        logger.info("🔍 Research node: Executing LCEL chain")
        try:
            result = await self.research_chain.ainvoke(state)
            logger.info(f"🔍 Research node: Completed - {len(result.get('research_urls', []))} URLs, {len(result.get('screenshots', []))} screenshots")
            return result
        except Exception as e:
            logger.error(f"Research LCEL chain failed: {e}", exc_info=True)
            state["errors"].append(f"Research failed: {str(e)}")
            return state
    
    async def _run_writing_agent(self, state: ArticleCMSState) -> ArticleCMSState:
        """Run writing agent node using LCEL chain"""
        logger.info("✍️  Writing node: Executing LCEL chain")
        try:
            result = await self.writing_chain.ainvoke(state)
            content = result.get('final_content', '') or result.get('draft_content', '')
            logger.info(f"✍️  Writing node: Completed - {len(content)} characters generated")
            return result
        except Exception as e:
            logger.error(f"Writing LCEL chain failed: {e}", exc_info=True)
            state["errors"].append(f"Writing failed: {str(e)}")
            return state
    
    async def _run_affiliate_agent(self, state: ArticleCMSState) -> Dict[str, Any]:
        """Run affiliate link agent node using LCEL chain - returns only updates"""
        try:
            result = await self.affiliate_chain.ainvoke(state)
            # Return only fields that affiliate agent modifies
            updates = {}
            affiliate_fields = {
                'affiliate_links', 'affiliate_opportunities',
                'tracking_codes',             }
            for key in affiliate_fields:
                if key in result:
                    updates[key] = result[key]
            logger.info(f"🔗 Affiliate node: Completed - {len(result.get('affiliate_links', []))} links found")
            return updates
        except Exception as e:
            logger.error(f"Affiliate LCEL chain failed: {e}", exc_info=True)
            return {"errors": [f"Affiliate failed: {str(e)}"]}

    async def _run_image_agent(self, state: ArticleCMSState) -> Dict[str, Any]:
        """Run image agent node using LCEL chain - returns only updates"""
        try:
            result = await self.image_chain.ainvoke(state)
            # Return only fields that image agent modifies
            updates = {}
            image_fields = {
                'images', 'wordpress_media_ids', 'image_alt_texts',
                'final_content',             }
            for key in image_fields:
                if key in result:
                    updates[key] = result[key]
            logger.info(f"🖼️ Image node: Completed - {len(result.get('images', []))} images found")
            return updates
        except Exception as e:
            logger.error(f"Image LCEL chain failed: {e}", exc_info=True)
            return {"errors": [f"Image failed: {str(e)}"]}
    
    async def _run_publishing_agent(self, state: ArticleCMSState) -> ArticleCMSState:
        """Run publishing agent node using LCEL chain"""
        try:
            return await self.publishing_chain.ainvoke(state)
        except Exception as e:
            logger.error(f"Publishing LCEL chain failed: {e}", exc_info=True)
            state["errors"].append(f"Publishing failed: {str(e)}")
            return state
    
    async def run(
        self,
        query: str,
        target_sites: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ArticleCMSState:
        """
        Run the complete CMS workflow
        
        Args:
            query: Content query/topic
            target_sites: WordPress site IDs to publish to
            config: Additional configuration
            
        Returns:
            Final state with all results
        """
        initial_state = create_initial_state(query, target_sites or [])
        
        # Handle checkpointing configuration
        workflow_config = config or {}
        if self.enable_checkpoints:
            # Checkpointer requires thread_id, checkpoint_ns, or checkpoint_id
            if "thread_id" not in workflow_config and "checkpoint_ns" not in workflow_config and "checkpoint_id" not in workflow_config:
                workflow_config["thread_id"] = str(uuid.uuid4())
        
        logger.info(f"Starting LCEL+LangGraph workflow for query: {query[:100]}...")
        
        try:
            # Add LangSmith tracing to workflow config
            langsmith_config = get_langsmith_config(metadata={
                "query": query,
                "target_sites": target_sites or [],
                "workflow_type": "cms_orchestrator",
                "checkpoints_enabled": self.enable_checkpoints,
            })
            
            # Merge LangSmith config with workflow config
            if langsmith_config.get("callbacks"):
                if "callbacks" not in workflow_config:
                    workflow_config["callbacks"] = []
                workflow_config["callbacks"].extend(langsmith_config["callbacks"])
            
            if langsmith_config.get("metadata"):
                if "metadata" not in workflow_config:
                    workflow_config["metadata"] = {}
                workflow_config["metadata"].update(langsmith_config["metadata"])
            
            # Run the LangGraph workflow with LCEL chains and LangSmith tracing
            final_state = await self.app.ainvoke(initial_state, config=workflow_config)
            
            logger.info("LCEL+LangGraph workflow completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"CMS workflow failed: {e}", exc_info=True)
            initial_state["errors"].append(f"Workflow failed: {str(e)}")
            return initial_state
    
    def get_graph_visualization(self) -> str:
        """Get visual representation of the graph (for debugging)"""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")
            return "Graph visualization unavailable"


def create_cms_orchestrator(
    research_agent: Optional[BaseAgent] = None,
    writing_agent: Optional[BaseAgent] = None,
    affiliate_agent: Optional[BaseAgent] = None,
    image_agent: Optional[BaseAgent] = None,
    publishing_agent: Optional[BaseAgent] = None,
    enable_checkpoints: bool = True,
) -> ArticleCMSOrchestrator:
    """
    Factory function to create CMS orchestrator
    
    Args:
        research_agent: Research agent instance
        writing_agent: Writing agent instance
        affiliate_agent: Affiliate link agent instance
        image_agent: Image agent instance
        publishing_agent: Publishing agent instance
        enable_checkpoints: Enable state checkpointing
        
    Returns:
        Configured ArticleCMSOrchestrator
    """
    return ArticleCMSOrchestrator(
        research_agent=research_agent,
        writing_agent=writing_agent,
        affiliate_agent=affiliate_agent,
        image_agent=image_agent,
        publishing_agent=publishing_agent,
        enable_checkpoints=enable_checkpoints,
    )

