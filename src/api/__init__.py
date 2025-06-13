"""
API package for the Universal RAG CMS system.

This package contains REST API endpoints for:
- Configuration management
- Monitoring and analytics
- Performance profiling
- Feature flags and A/B testing
"""

from .config_management import router as config_router

__all__ = ["config_router"] 