"""
Base Agent Class for Agent-Based CMS
Provides common functionality for all specialized agents
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .state import ArticleCMSState, AgentState, AgentStatus

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    """Result from agent execution"""
    success: bool
    state_updates: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all CMS agents
    
    Provides common functionality:
    - Tool management
    - State updates
    - Error handling
    - Logging
    - Retry logic
    """
    
    def __init__(
        self,
        name: str,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[List[BaseTool]] = None,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize base agent
        
        Args:
            name: Agent name for identification
            llm: Language model for agent reasoning
            tools: List of tools available to the agent
            max_retries: Maximum retry attempts on failure
        """
        self.name = name
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.tools = tools or []
        self.max_retries = max_retries
        self.agent_state = AgentState(agent_name=name)
        
        logger.info(f"Initialized {self.name} agent with {len(self.tools)} tools")
    
    @abstractmethod
    async def execute(self, state: ArticleCMSState) -> AgentResult:
        """
        Execute the agent's main logic
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResult with state updates
        """
        pass
    
    async def run(self, state: ArticleCMSState) -> ArticleCMSState:
        """
        Run the agent with retry logic and state management
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        self.agent_state.status = AgentStatus.IN_PROGRESS
        self.agent_state.started_at = datetime.now()
        # Don't update current_agent in parallel nodes - only update agent_statuses
        # state["current_agent"] = self.name  # Commented out to avoid parallel update conflicts
        state["agent_statuses"][self.name] = AgentStatus.IN_PROGRESS.value
        
        start_time = time.time()
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                logger.info(f"{self.name} executing (attempt {retry_count + 1}/{self.max_retries + 1})")
                
                result = await self.execute(state)
                
                duration = time.time() - start_time
                self.agent_state.duration_seconds = duration
                
                if result.success:
                    self.agent_state.status = AgentStatus.COMPLETED
                    self.agent_state.completed_at = datetime.now()
                    state["agent_statuses"][self.name] = AgentStatus.COMPLETED.value
                    
                    # Apply state updates (skip immutable fields like 'query')
                    immutable_fields = {'query', 'target_sites'}  # Fields that shouldn't be updated
                    for key, value in result.state_updates.items():
                        if key in immutable_fields:
                            continue  # Skip immutable fields to avoid parallel update conflicts
                        if key in state:
                            if isinstance(state[key], list) and isinstance(value, list):
                                state[key].extend(value)
                            elif isinstance(state[key], dict) and isinstance(value, dict):
                                state[key].update(value)
                            else:
                                state[key] = value
                    
                    logger.info(f"{self.name} completed successfully in {duration:.2f}s")
                    # Return full state - caller can filter if needed for parallel execution
                    return state
                else:
                    raise Exception(result.error or "Agent execution failed")
                    
            except Exception as e:
                retry_count += 1
                self.agent_state.retry_count = retry_count
                error_msg = str(e)
                
                logger.warning(f"{self.name} attempt {retry_count} failed: {error_msg}")
                
                if retry_count > self.max_retries:
                    self.agent_state.status = AgentStatus.FAILED
                    self.agent_state.error_message = error_msg
                    state["agent_statuses"][self.name] = AgentStatus.FAILED.value
                    state["errors"].append(f"{self.name}: {error_msg}")
                    
                    logger.error(f"{self.name} failed after {retry_count} attempts")
                    return state
                
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** retry_count)
        
        return state
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a tool by its name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        logger.debug(f"Added tool {tool.name} to {self.name}")
    
    def add_tools(self, tools: List[BaseTool]):
        """Add multiple tools to the agent"""
        self.tools.extend(tools)
        logger.debug(f"Added {len(tools)} tools to {self.name}")
    
    def update_state_metadata(self, state: ArticleCMSState, key: str, value: Any):
        """Helper to update state metadata"""
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"][key] = value

