"""
Test module for ContentTypeConfig.

This module contains unit tests for the ContentTypeConfig class,
which defines configuration systems for content-type specific cache settings
including distance thresholds, TTL values, and embedding parameters.

Test coverage includes:
- Configuration loading for different content types (news, reviews, regulatory)
- Distance threshold validation and optimization
- TTL configuration in seconds
- Embedding model parameter configuration
- Configuration validation using Pydantic models
- Factory method functionality
"""

import pytest
from unittest.mock import Mock
from typing import Dict, Any
from pydantic import ValidationError

# Import the module under test (will be available once class is implemented)
# from src.services.cache.content_type_config import ContentTypeConfig 