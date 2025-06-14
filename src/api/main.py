"""
Main FastAPI application for the Universal RAG CMS API.

This module sets up the FastAPI application with all routers and middleware.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

from .config_management import router as config_router
from .retrieval_config_api import router as retrieval_config_router
from .contextual_retrieval_api import router as contextual_retrieval_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Universal RAG CMS API",
    description="REST API for configuration management, monitoring, A/B testing, and contextual retrieval",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Include routers
app.include_router(config_router)
app.include_router(retrieval_config_router, prefix="/retrieval")
app.include_router(contextual_retrieval_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Universal RAG CMS API",
        "version": "2.1.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "config": "/api/v1/config",
            "retrieval_config": "/retrieval/api/v1/config",
            "contextual_retrieval": "/api/v1/contextual",
            "health": "/health"
        },
        "features": [
            "Configuration Management",
            "Real-time Monitoring",
            "Performance Profiling", 
            "Feature Flags & A/B Testing",
            "Contextual Retrieval Configuration",
            "Contextual Document Querying",
            "Document Ingestion & Migration",
            "WebSocket Metrics Streaming"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health():
    """General health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "config_management": "healthy",
            "monitoring": "healthy",
            "retrieval_config": "healthy",
            "contextual_retrieval": "healthy",
            "feature_flags": "healthy"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 