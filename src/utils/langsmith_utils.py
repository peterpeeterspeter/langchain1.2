"""
LangSmith Integration Utilities
Shared utilities for LangSmith tracing and evaluation across the codebase
"""

import os
import logging
from typing import List, Optional, Dict, Any
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Import settings with dotenv loading
try:
    from src.config.production_settings import get_settings
except ImportError:
    # Fallback if import fails
    def get_settings():
        class MockSettings:
            langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
            langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
            langsmith_project = os.getenv("LANGSMITH_PROJECT", "universal-rag-cms")
            langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
            
            def is_langsmith_enabled(self) -> bool:
                return self.langsmith_tracing and bool(self.langsmith_api_key)
        return MockSettings()

# Try to import LangSmith
try:
    from langsmith import traceable, Client
    from langchain_core.tracers.langchain import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not available. Install with: pip install langsmith")


def get_langsmith_callbacks(project_name: Optional[str] = None) -> List[BaseCallbackHandler]:
    """
    Get LangSmith callbacks for tracing if enabled.
    
    Args:
        project_name: Optional project name for organizing traces
        
    Returns:
        List of callback handlers including LangSmith tracer if available
    """
    callbacks = []
    
    if not LANGSMITH_AVAILABLE:
        return callbacks
    
    # Use settings to get configuration (loads from .env via dotenv)
    settings = get_settings()
    
    if settings.is_langsmith_enabled():
        try:
            # Use project name from parameter or settings
            project = project_name or settings.langsmith_project
            
            # Create LangSmith Client with API key
            client_kwargs = {"api_key": settings.langsmith_api_key}
            if settings.langsmith_endpoint:
                client_kwargs["api_url"] = settings.langsmith_endpoint
            
            langsmith_client = Client(**client_kwargs)
            
            # Create LangChainTracer with Client instance
            tracer = LangChainTracer(
                project_name=project,
                client=langsmith_client
            )
            callbacks.append(tracer)
            logger.info(f"✅ LangSmith tracing enabled for project: {project}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize LangSmith tracer: {e}")
    else:
        logger.debug("LangSmith tracing not enabled (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
    
    return callbacks


def get_langsmith_config(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get LangSmith configuration for chain/agent invocations.
    
    Args:
        metadata: Optional metadata to include in traces
        
    Returns:
        Configuration dictionary with callbacks and metadata
    """
    config = {}
    
    # Get LangSmith callbacks
    callbacks = get_langsmith_callbacks()
    if callbacks:
        config["callbacks"] = callbacks
        
        # Add metadata if provided
        if metadata:
            config["metadata"] = metadata
    
    return config


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled and configured."""
    if not LANGSMITH_AVAILABLE:
        return False
    
    # Use settings to check (loads from .env via dotenv)
    settings = get_settings()
    return settings.is_langsmith_enabled()

