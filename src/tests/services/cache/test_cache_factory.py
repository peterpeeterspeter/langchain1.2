"""
Test module for CacheFactory.

This module contains unit tests for the CacheFactory class,
which provides factory methods for creating different types of
cache instances with content-type specific configurations.

Test coverage includes:
- Cache instance creation for different content types
- Configuration parameter validation
- Factory method functionality
- Error handling for invalid configurations
- Integration with RedisSemanticCache instances
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the module under test (will be available once class is implemented)
# from src.services.cache.cache_factory import CacheFactory 