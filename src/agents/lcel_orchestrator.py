"""
LangChain LCEL + LangGraph Orchestrator
Uses LCEL chains for agent logic, LangGraph for workflow orchestration
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ArticleCMSState, create_initial_state
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LCELOrchestrator:
    """
    Enhanced orchestrator using LangChain LCEL + LangGraph
    
    Uses:
    - LangChain LCEL chains for agent logic (composable, type-safe)
    - LangGraph StateGraph for workflow orchestration
    - RunnableParallel for concurrent operations
    - RunnableBranch for conditional logic
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
        """Initialize LCEL orchestrator"""
        self.research_agent = research_agent
        self.writing_agent = writing_agent
        self.affiliate_agent = affiliate_agent
        self.image_agent = image_agent
        self.publishing_agent = publishing_agent
        self.enable_checkpoints = enable_checkpoints
        
        # Build LCEL chains for each phase
        self.research_chain = self._build_research_chain()
        self.writing_chain = self._build_writing_chain()
        self.affiliate_chain = self._build_affiliate_chain()
        self.image_chain = self._build_image_chain()
        self.publishing_chain = self._build_publishing_chain()
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        
        # Compile with checkpoints if enabled
        if enable_checkpoints:
            checkpointer = MemorySaver()
            self.app = self.graph.compile(checkpointer=checkpointer)
        else:
            self.app = self.graph.compile()
        
        logger.info("LCELOrchestrator initialized with LangChain LCEL + LangGraph")
    
    def _build_research_chain(self):
        """Build LCEL chain for research phase"""
        if not self.research_agent:
            return RunnablePassthrough()
        
        def run_research(state: ArticleCMSState) -> ArticleCMSState:
            """LCEL compatible research runner"""
            import asyncio
            if asyncio.iscoroutinefunction(self.research_agent.run):
                return asyncio.run(self.research_agent.run(state))
            return self.research_agent.run(state)
        
        return RunnableLambda(run_research)
    
    def _build_writing_chain(self):
        """Build LCEL chain for writing phase"""
        if not self.writing_agent:
            return RunnablePassthrough()
        
        def run_writing(state: ArticleCMSState) -> ArticleCMSState:
            """LCEL compatible writing runner"""
            import asyncio
            if asyncio.iscoroutinefunction(self.writing_agent.run):
                return asyncio.run(self.writing_agent.run(state))
            return self.writing_agent.run(state)
        
        return RunnableLambda(run_writing)
    
    def _build_affiliate_chain(self):
        """Build LCEL chain for affiliate link phase"""
        if not self.affiliate_agent:
            return RunnablePassthrough()
        
        def run_affiliate(state: ArticleCMSState) -> Dict[str, Any]:
            """LCEL compatible affiliate runner - returns updates only"""
            import asyncio
            if asyncio.iscoroutinefunction(self.affiliate_agent.run):
                updated_state = asyncio.run(self.affiliate_agent.run(state))
            else:
                updated_state = self.affiliate_agent.run(state)
            
            # Return only fields that affiliate agent modifies
            updates = {}
            affiliate_fields = {
                'final_content', 'affiliate_links', 'affiliate_opportunities',
                'tracking_codes', 'agent_statuses', 'errors', 'warnings', 'metadata'
            }
            for key in affiliate_fields:
                if key in updated_state:
                    if key not in state or updated_state[key] != state.get(key):
                        updates[key] = updated_state[key]
            return updates
        
        return RunnableLambda(run_affiliate)
    
    def _build_image_chain(self):
        """Build LCEL chain for image phase"""
        if not self.image_agent:
            return RunnablePassthrough()
        
        def run_image(state: ArticleCMSState) -> Dict[str, Any]:
            """LCEL compatible image runner - returns updates only"""
            import asyncio
            if asyncio.iscoroutinefunction(self.image_agent.run):
                updated_state = asyncio.run(self.image_agent.run(state))
            else:
                updated_state = self.image_agent.run(state)
            
            # Return only fields that image agent modifies
            updates = {}
            image_fields = {
                'images', 'wordpress_media_ids', 'image_alt_texts',
                'final_content', 'agent_statuses', 'errors', 'warnings', 'metadata'
            }
            for key in image_fields:
                if key in updated_state:
                    if key not in state or updated_state[key] != state.get(key):
                        updates[key] = updated_state[key]
            return updates
        
        return RunnableLambda(run_image)
    
    def _build_publishing_chain(self):
        """Build LCEL chain for publishing phase"""
        if not self.publishing_agent:
            return RunnablePassthrough()
        
        def run_publishing(state: ArticleCMSState) -> ArticleCMSState:
            """LCEL compatible publishing runner"""
            import asyncio
            if asyncio.iscoroutinefunction(self.publishing_agent.run):
                return asyncio.run(self.publishing_agent.run(state))
            return self.publishing_agent.run(state)
        
        return RunnableLambda(run_publishing)
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph StateGraph with LCEL chains"""
        
        workflow = StateGraph(ArticleCMSState)
        
        # Add nodes using LCEL chains
        if self.research_agent:
            workflow.add_node("research", self._run_research_lcel)
        
        if self.writing_agent:
            workflow.add_node("writing", self._run_writing_lcel)
        
        # Parallel nodes for affiliate and image (both run after writing)
        if self.affiliate_agent:
            workflow.add_node("affiliate", self._run_affiliate_lcel)
        
        if self.image_agent:
            workflow.add_node("image", self._run_image_lcel)
        
        # Merge parallel results
        if self.affiliate_agent or self.image_agent:
            workflow.add_node("merge_parallel", self._merge_parallel_results)
        
        if self.publishing_agent:
            workflow.add_node("publishing", self._run_publishing_lcel)
        
        # Determine entry point
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
            def passthrough(state: ArticleCMSState) -> ArticleCMSState:
                return state
            workflow.add_node("passthrough", passthrough)
            workflow.set_entry_point("passthrough")
            workflow.add_edge("passthrough", END)
            return workflow
        
        workflow.set_entry_point(entry_point)
        
        # Research -> Writing
        if self.research_agent and self.writing_agent:
            workflow.add_edge("research", "writing")
        
        # Writing -> Parallel (Affiliate + Image)
        if self.writing_agent:
            if self.affiliate_agent and self.image_agent:
                # Both run in parallel
                workflow.add_edge("writing", "affiliate")
                workflow.add_edge("writing", "image")
                # Both converge to merge
                workflow.add_edge("affiliate", "merge_parallel")
                workflow.add_edge("image", "merge_parallel")
                # Merge -> Publishing
                if self.publishing_agent:
                    workflow.add_edge("merge_parallel", "publishing")
                else:
                    workflow.add_edge("merge_parallel", END)
            elif self.affiliate_agent:
                workflow.add_edge("writing", "affiliate")
                if self.publishing_agent:
                    workflow.add_edge("affiliate", "publishing")
                else:
                    workflow.add_edge("affiliate", END)
            elif self.image_agent:
                workflow.add_edge("writing", "image")
                if self.publishing_agent:
                    workflow.add_edge("image", "publishing")
                else:
                    workflow.add_edge("image", END)
            elif self.publishing_agent:
                workflow.add_edge("writing", "publishing")
            else:
                workflow.add_edge("writing", END)
        
        # Handle cases without writing agent
        if not self.writing_agent:
            if self.research_agent:
                if self.affiliate_agent and self.image_agent:
                    workflow.add_edge("research", "affiliate")
                    workflow.add_edge("research", "image")
                    workflow.add_edge("affiliate", "merge_parallel")
                    workflow.add_edge("image", "merge_parallel")
                    if self.publishing_agent:
                        workflow.add_edge("merge_parallel", "publishing")
                    else:
                        workflow.add_edge("merge_parallel", END)
                elif self.affiliate_agent:
                    workflow.add_edge("research", "affiliate")
                    if self.publishing_agent:
                        workflow.add_edge("affiliate", "publishing")
                    else:
                        workflow.add_edge("affiliate", END)
                elif self.image_agent:
                    workflow.add_edge("research", "image")
                    if self.publishing_agent:
                        workflow.add_edge("image", "publishing")
                    else:
                        workflow.add_edge("image", END)
                elif self.publishing_agent:
                    workflow.add_edge("research", "publishing")
                else:
                    workflow.add_edge("research", END)
        
        # Publishing -> END
        if self.publishing_agent:
            workflow.add_edge("publishing", END)
        
        return workflow
    
    async def _run_research_lcel(self, state: ArticleCMSState) -> ArticleCMSState:
        """Run research using LCEL chain"""
        try:
            result = await self.research_chain.ainvoke(state)
            return result
        except Exception as e:
            logger.error(f"Research chain failed: {e}", exc_info=True)
            state["errors"].append(f"Research failed: {str(e)}")
            return state
    
    async def _run_writing_lcel(self, state: ArticleCMSState) -> ArticleCMSState:
        """Run writing using LCEL chain"""
        try:
            result = await self.writing_chain.ainvoke(state)
            return result
        except Exception as e:
            logger.error(f"Writing chain failed: {e}", exc_info=True)
            state["errors"].append(f"Writing failed: {str(e)}")
            return state
    
    async def _run_affiliate_lcel(self, state: ArticleCMSState) -> Dict[str, Any]:
        """Run affiliate using LCEL chain - returns updates only"""
        try:
            updates = await self.affiliate_chain.ainvoke(state)
            return updates
        except Exception as e:
            logger.error(f"Affiliate chain failed: {e}", exc_info=True)
            return {"errors": [f"Affiliate failed: {str(e)}"]}
    
    async def _run_image_lcel(self, state: ArticleCMSState) -> Dict[str, Any]:
        """Run image using LCEL chain - returns updates only"""
        try:
            updates = await self.image_chain.ainvoke(state)
            return updates
        except Exception as e:
            logger.error(f"Image chain failed: {e}", exc_info=True)
            return {"errors": [f"Image failed: {str(e)}"]}
    
    async def _merge_parallel_results(self, state: ArticleCMSState) -> ArticleCMSState:
        """Merge results from parallel affiliate and image operations"""
        # This node receives state updates from both parallel branches
        # LangGraph automatically merges the updates
        return state
    
    async def _run_publishing_lcel(self, state: ArticleCMSState) -> ArticleCMSState:
        """Run publishing using LCEL chain"""
        try:
            result = await self.publishing_chain.ainvoke(state)
            return result
        except Exception as e:
            logger.error(f"Publishing chain failed: {e}", exc_info=True)
            state["errors"].append(f"Publishing failed: {str(e)}")
            return state
    
    async def run(
        self,
        query: str,
        target_sites: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ArticleCMSState:
        """
        Run the complete CMS workflow using LCEL + LangGraph
        
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
            if "thread_id" not in workflow_config and "checkpoint_ns" not in workflow_config and "checkpoint_id" not in workflow_config:
                workflow_config["thread_id"] = str(uuid.uuid4())
        
        logger.info(f"Starting LCEL+LangGraph workflow for query: {query[:100]}...")
        
        try:
            # Run the graph with LCEL chains
            final_state = await self.app.ainvoke(initial_state, config=workflow_config)
            
            logger.info("LCEL+LangGraph workflow completed successfully")
            return final_state
            
        except Exception as e:
            logger.error(f"LCEL+LangGraph workflow failed: {e}", exc_info=True)
            initial_state["errors"].append(f"Workflow failed: {str(e)}")
            return initial_state
    
    def get_graph_visualization(self) -> str:
        """Get visual representation of the graph"""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            logger.warning(f"Could not generate graph visualization: {e}")
            return "Graph visualization unavailable"


# Update orchestrator import to use LCEL version
from .orchestrator import ArticleCMSOrchestrator as BaseOrchestrator

# Create hybrid orchestrator that uses LCEL internally
class ArticleCMSOrchestrator(LCELOrchestrator):
    """
    Production orchestrator using LCEL + LangGraph
    
    Maintains backward compatibility with existing API
    while using enhanced LCEL patterns internally
    """
    pass

