"""
Pytest configuration and global fixtures for comprehensive testing framework.
Supports Task 10: Setup Comprehensive Testing Framework
"""

import pytest
import asyncio
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch
import uuid
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import testing dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

try:
    from src.chains.enhanced_confidence_scoring_system import (
        EnhancedRAGResponse, ConfidenceFactors, SourceQualityAnalyzer,
        IntelligentCache, ResponseValidator, UniversalRAGEnhancementSystem
    )
    ENHANCED_CONFIDENCE_AVAILABLE = True
except ImportError:
    ENHANCED_CONFIDENCE_AVAILABLE = False


# ============================================================================
# SESSION FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "database": {
            "test_db_prefix": "test_",
            "isolation_enabled": True,
            "cleanup_after_test": True
        },
        "performance": {
            "max_execution_time": 30.0,
            "memory_limit_mb": 500,
            "benchmark_iterations": {
                "small": 10,
                "medium": 100,
                "large": 1000
            }
        },
        "mock_services": {
            "openai_api": True,
            "anthropic_api": True,
            "supabase_storage": True,
            "external_apis": True
        },
        "cache": {
            "test_cache_size": 100,
            "clear_between_tests": True
        }
    }


@pytest.fixture(scope="session")
def performance_tracker():
    """Performance tracking for tests."""
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_timer(self, test_name: str):
            self.start_times[test_name] = datetime.now()
        
        def end_timer(self, test_name: str):
            if test_name in self.start_times:
                duration = datetime.now() - self.start_times[test_name]
                self.metrics[test_name] = {
                    'duration_ms': duration.total_seconds() * 1000,
                    'timestamp': datetime.now().isoformat()
                }
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    self.metrics[test_name].update({
                        'memory_mb': process.memory_info().rss / 1024 / 1024,
                        'cpu_percent': process.cpu_percent()
                    })
        
        def get_metrics(self) -> Dict[str, Any]:
            return self.metrics.copy()
    
    return PerformanceTracker()


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
async def supabase_test_client():
    """Create isolated Supabase test client."""
    if not SUPABASE_AVAILABLE:
        pytest.skip("Supabase not available")
    
    # Get test credentials from environment
    test_url = os.getenv("SUPABASE_TEST_URL")
    test_key = os.getenv("SUPABASE_TEST_ANON_KEY")
    
    if not test_url or not test_key:
        pytest.skip("Supabase test credentials not configured")
    
    client = create_client(test_url, test_key)
    
    # Create test isolation
    test_session_id = str(uuid.uuid4())[:8]
    
    yield client, test_session_id
    
    # Cleanup test data
    await cleanup_test_data(client, test_session_id)


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for unit tests."""
    mock_client = Mock(spec=Client)
    
    # Mock database operations
    mock_client.table.return_value.select.return_value.execute.return_value = Mock(
        data=[], count=0
    )
    mock_client.table.return_value.insert.return_value.execute.return_value = Mock(
        data=[{"id": 1}], count=1
    )
    mock_client.table.return_value.update.return_value.execute.return_value = Mock(
        data=[{"id": 1}], count=1
    )
    mock_client.table.return_value.delete.return_value.execute.return_value = Mock(
        data=[], count=1
    )
    
    # Mock auth operations
    mock_client.auth.sign_up.return_value = Mock(
        user=Mock(id="test-user-id", email="test@example.com")
    )
    mock_client.auth.sign_in_with_password.return_value = Mock(
        user=Mock(id="test-user-id", email="test@example.com"),
        session=Mock(access_token="test-token")
    )
    
    # Mock storage operations
    mock_client.storage.from_.return_value.upload.return_value = Mock(
        path="test-file.txt"
    )
    mock_client.storage.from_.return_value.download.return_value = b"test content"
    
    return mock_client


@pytest.fixture
async def test_database_schema():
    """Create test database schema for isolation."""
    return {
        "documents": {
            "id": "uuid",
            "content": "text",
            "metadata": "jsonb",
            "embedding": "vector(1536)",
            "created_at": "timestamp",
            "updated_at": "timestamp"
        },
        "queries": {
            "id": "uuid", 
            "query_text": "text",
            "query_type": "text",
            "metadata": "jsonb",
            "created_at": "timestamp"
        },
        "responses": {
            "id": "uuid",
            "query_id": "uuid",
            "content": "text",
            "confidence_score": "float",
            "sources": "jsonb",
            "metadata": "jsonb",
            "created_at": "timestamp"
        }
    }


# ============================================================================
# MOCK SERVICE FIXTURES  
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = Mock()
    
    # Mock chat completions
    mock_client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(
                message=Mock(content="This is a test response from OpenAI"),
                finish_reason="stop"
            )
        ],
        usage=Mock(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70
        )
    )
    
    # Mock embeddings
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    
    return mock_client


@pytest.fixture  
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock_client = Mock()
    
    mock_client.messages.create.return_value = Mock(
        content=[Mock(text="This is a test response from Claude")],
        usage=Mock(
            input_tokens=45,
            output_tokens=25
        )
    )
    
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()
    
    # Mock similarity search
    from langchain_core.documents import Document
    mock_store.similarity_search.return_value = [
        Document(
            page_content="Test document content",
            metadata={"source": "test.pdf", "page": 1}
        )
    ]
    
    mock_store.similarity_search_with_score.return_value = [
        (Document(
            page_content="Test document content",
            metadata={"source": "test.pdf", "page": 1}
        ), 0.85)
    ]
    
    # Mock add/delete operations
    mock_store.add_documents.return_value = ["doc-id-1", "doc-id-2"]
    mock_store.delete.return_value = True
    
    return mock_store


# ============================================================================
# ENHANCED CONFIDENCE SYSTEM FIXTURES
# ============================================================================

@pytest.fixture
def sample_enhanced_rag_response():
    """Sample EnhancedRAGResponse for testing."""
    if not ENHANCED_CONFIDENCE_AVAILABLE:
        pytest.skip("Enhanced confidence system not available")
    
    return EnhancedRAGResponse(
        content="This is a comprehensive test response about casino safety...",
        sources=[
            {
                "content": "Licensed casinos are regulated by gaming authorities...",
                "metadata": {"source": "gaming-commission.gov", "authority": "high"},
                "quality_score": 0.85
            },
            {
                "content": "Look for SSL certificates and secure payment methods...", 
                "metadata": {"source": "casino-safety.com", "authority": "medium"},
                "quality_score": 0.72
            }
        ],
        confidence_score=0.8,
        response_time=1.5,
        cached=False,
        metadata={}
    )


@pytest.fixture
def mock_confidence_factors():
    """Mock confidence factors for testing."""
    if not ENHANCED_CONFIDENCE_AVAILABLE:
        pytest.skip("Enhanced confidence system not available")
    
    return ConfidenceFactors(
        completeness=0.85,
        relevance=0.78,
        accuracy_indicators=0.82,
        source_reliability=0.75,
        source_coverage=0.80,
        source_consistency=0.77,
        intent_alignment=0.83,
        expertise_match=0.79,
        format_appropriateness=0.81,
        retrieval_quality=0.76,
        generation_stability=0.84,
        optimization_effectiveness=0.73
    )


@pytest.fixture
def test_cache_system():
    """Test cache system with small size for testing."""
    if not ENHANCED_CONFIDENCE_AVAILABLE:
        pytest.skip("Enhanced confidence system not available")
    
    from src.chains.enhanced_confidence_scoring_system import IntelligentCache, CacheStrategy
    return IntelligentCache(strategy=CacheStrategy.BALANCED, max_size=10)


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_test_queries():
    """Sample queries for testing different query types."""
    return {
        "factual": "What is the house edge in blackjack?",
        "comparison": "Which is better: Betway or 888 Casino?",
        "tutorial": "How do I play poker step by step?",
        "review": "Is Betfair Casino trustworthy?",
        "promotional": "What are the best casino bonuses in 2024?",
        "news": "Latest gambling regulation updates",
        "troubleshooting": "Why can't I withdraw from my casino account?"
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="Blackjack has one of the lowest house edges in casino games, typically around 0.5% with basic strategy.",
            metadata={
                "source": "gambling-math.edu",
                "title": "Mathematical Analysis of Casino Games",
                "author": "Dr. John Smith",
                "published_date": "2024-01-15",
                "authority_score": 0.9
            }
        ),
        Document(
            page_content="Basic strategy in blackjack involves hitting, standing, doubling, and splitting based on mathematical probabilities.",
            metadata={
                "source": "blackjack-guide.com", 
                "title": "Complete Blackjack Strategy Guide",
                "author": "Casino Expert",
                "published_date": "2024-02-10",
                "authority_score": 0.7
            }
        ),
        Document(
            page_content="Licensed online casinos use random number generators to ensure fair play in digital blackjack games.",
            metadata={
                "source": "gaming-commission.gov",
                "title": "Online Casino Regulation Guidelines",
                "published_date": "2024-01-01",
                "authority_score": 0.95
            }
        )
    ]


@pytest.fixture
def temp_test_directory():
    """Temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def cleanup_test_data(client, session_id: str):
    """Clean up test data from Supabase."""
    try:
        # Clean up test tables with session_id
        tables_to_clean = ["documents", "queries", "responses", "users"]
        
        for table in tables_to_clean:
            try:
                await client.table(table).delete().like("id", f"%{session_id}%").execute()
            except Exception as e:
                print(f"Warning: Could not clean table {table}: {e}")
                
    except Exception as e:
        print(f"Warning: Test cleanup failed: {e}")


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment(performance_tracker, request):
    """Setup test environment before each test."""
    test_name = request.node.name
    performance_tracker.start_timer(test_name)
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup and performance tracking
    performance_tracker.end_timer(test_name)
    
    # Reset environment
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


def pytest_configure(config):
    """Configure pytest with comprehensive markers."""
    markers = [
        "unit: mark test as a unit test",
        "integration: mark test as an integration test", 
        "performance: mark test as a performance test",
        "slow: mark test as slow running",
        "smoke: mark test as smoke test for basic functionality",
        "supabase: mark test as requiring Supabase connection",
        "confidence: mark test as testing confidence scoring system",
        "cache: mark test as testing caching functionality",
        "validation: mark test as testing response validation",
        "security: mark test as testing security features",
        "api: mark test as testing API endpoints",
        "end_to_end: mark test as end-to-end workflow test"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Automatically mark tests based on their location
        test_path = str(item.fspath)
        
        if "unit" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "performance" in test_path:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "security" in test_path:
            item.add_marker(pytest.mark.security)
        
        # Mark tests requiring external services
        if "supabase" in test_path or "supabase" in item.name.lower():
            item.add_marker(pytest.mark.supabase)
        
        if "confidence" in test_path or "confidence" in item.name.lower():
            item.add_marker(pytest.mark.confidence)
        
        if "cache" in test_path or "cache" in item.name.lower():
            item.add_marker(pytest.mark.cache)


def pytest_runtest_setup(item):
    """Setup individual test runs."""
    # Skip tests if dependencies not available
    if item.get_closest_marker("supabase") and not SUPABASE_AVAILABLE:
        pytest.skip("Supabase not available")
    
    if item.get_closest_marker("confidence") and not ENHANCED_CONFIDENCE_AVAILABLE:
        pytest.skip("Enhanced confidence system not available")


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Session-wide cleanup."""
    yield
    
    # Clean up any remaining test artifacts
    test_files = Path.cwd().glob("test_*")
    for test_file in test_files:
        try:
            if test_file.is_file():
                test_file.unlink()
            elif test_file.is_dir():
                shutil.rmtree(test_file)
        except Exception as e:
            print(f"Warning: Could not clean up {test_file}: {e}")


# Performance test configuration
@pytest.fixture(scope="session")  
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_execution_time": 30.0,
        "memory_limit_mb": 500,
        "iteration_counts": {
            "small": 100,
            "medium": 1000, 
            "large": 10000
        },
        "benchmarks": {
            "response_time_target_ms": 2000,
            "confidence_calculation_target_ms": 500,
            "cache_hit_rate_target": 0.7,
            "memory_usage_target_mb": 100
        }
    } 