"""
Test fixtures and configuration data for config/monitoring testing.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from src.config.prompt_config import (
    PromptOptimizationConfig,
    QueryClassificationConfig,
    ContextFormattingConfig,
    CacheConfig,
    PerformanceConfig,
    FeatureFlags,
    QueryType
)

class TestConfigFixtures:
    """Test fixtures for configuration testing."""
    
    @staticmethod
    def get_default_config() -> PromptOptimizationConfig:
        """Get default configuration for testing."""
        return PromptOptimizationConfig()
    
    @staticmethod
    def get_custom_config() -> PromptOptimizationConfig:
        """Get custom configuration for testing edge cases."""
        return PromptOptimizationConfig(
            query_classification=QueryClassificationConfig(
                confidence_threshold=0.8,
                fallback_type=QueryType.GUIDE,
                enable_multi_classification=True,
                max_classification_attempts=3
            ),
            context_formatting=ContextFormattingConfig(
                max_context_length=4000,
                quality_threshold=0.85,
                freshness_weight=0.4,
                relevance_weight=0.6,
                include_metadata=False,
                max_chunks_per_source=5,
                chunk_overlap_ratio=0.2
            ),
            cache_config=CacheConfig(
                casino_review_ttl=48,
                news_ttl=1,
                product_review_ttl=24,
                technical_doc_ttl=336,  # 2 weeks
                general_ttl=12,
                guide_ttl=96,
                faq_ttl=144
            ),
            performance=PerformanceConfig(
                enable_monitoring=True,
                enable_profiling=True,
                response_time_warning_ms=1500,
                response_time_critical_ms=4000,
                error_rate_warning_percent=3.0,
                error_rate_critical_percent=8.0,
                min_samples_for_alerts=50,
                alert_cooldown_minutes=10
            ),
            feature_flags=FeatureFlags(
                enable_contextual_retrieval=True,
                enable_hybrid_search=False,
                enable_query_expansion=True,
                enable_response_caching=True,
                enable_semantic_cache=False,
                enable_auto_retry=False,
                enable_cost_optimization=True,
                ab_test_percentage=75.5
            ),
            version="2.0.0",
            updated_by="test_user"
        )
    
    @staticmethod
    def get_invalid_config_data() -> List[Dict[str, Any]]:
        """Get invalid configuration data for validation testing."""
        return [
            {
                "name": "invalid_confidence_threshold_low",
                "data": {
                    "query_classification": {
                        "confidence_threshold": 0.3  # Below 0.5 minimum
                    }
                },
                "expected_error": "Confidence threshold should be between 0.5 and 0.95"
            },
            {
                "name": "invalid_confidence_threshold_high", 
                "data": {
                    "query_classification": {
                        "confidence_threshold": 0.99  # Above 0.95 maximum
                    }
                },
                "expected_error": "Confidence threshold should be between 0.5 and 0.95"
            },
            {
                "name": "invalid_weight_sum",
                "data": {
                    "context_formatting": {
                        "freshness_weight": 0.3,
                        "relevance_weight": 0.8  # Sum > 1.0
                    }
                },
                "expected_error": "Freshness and relevance weights must sum to 1.0"
            },
            {
                "name": "invalid_max_context_length_low",
                "data": {
                    "context_formatting": {
                        "max_context_length": 400  # Below 500 minimum
                    }
                },
                "expected_error": "ensure this value is greater than or equal to 500"
            },
            {
                "name": "invalid_max_context_length_high",
                "data": {
                    "context_formatting": {
                        "max_context_length": 12000  # Above 10000 maximum
                    }
                },
                "expected_error": "ensure this value is less than or equal to 10000"
            }
        ]
    
    @staticmethod
    def get_edge_case_configs() -> List[Dict[str, Any]]:
        """Get edge case configurations for stress testing."""
        return [
            {
                "name": "minimum_values",
                "config": PromptOptimizationConfig(
                    query_classification=QueryClassificationConfig(
                        confidence_threshold=0.5,
                        max_classification_attempts=1
                    ),
                    context_formatting=ContextFormattingConfig(
                        max_context_length=500,
                        quality_threshold=0.0,
                        freshness_weight=0.0,
                        relevance_weight=1.0,
                        max_chunks_per_source=1,
                        chunk_overlap_ratio=0.0
                    ),
                    performance=PerformanceConfig(
                        response_time_warning_ms=100,
                        response_time_critical_ms=1000,
                        error_rate_warning_percent=0.1,
                        error_rate_critical_percent=1.0,
                        min_samples_for_alerts=10,
                        alert_cooldown_minutes=5
                    ),
                    feature_flags=FeatureFlags(ab_test_percentage=0.0)
                )
            },
            {
                "name": "maximum_values",
                "config": PromptOptimizationConfig(
                    query_classification=QueryClassificationConfig(
                        confidence_threshold=0.95,
                        max_classification_attempts=5
                    ),
                    context_formatting=ContextFormattingConfig(
                        max_context_length=10000,
                        quality_threshold=1.0,
                        freshness_weight=1.0,
                        relevance_weight=0.0,
                        max_chunks_per_source=10,
                        chunk_overlap_ratio=0.5
                    ),
                    performance=PerformanceConfig(
                        response_time_warning_ms=10000,
                        response_time_critical_ms=30000,
                        error_rate_warning_percent=20.0,
                        error_rate_critical_percent=30.0,
                        min_samples_for_alerts=1000,
                        alert_cooldown_minutes=60
                    ),
                    feature_flags=FeatureFlags(ab_test_percentage=100.0)
                )
            }
        ]

class MockSupabaseClient:
    """Mock Supabase client for testing database operations"""
    
    def __init__(self, fail_mode: str = None):
        self.fail_mode = fail_mode
        self.data_store = {
            "prompt_configurations": [
                {
                    "id": 1,
                    "name": "test_config",
                    "config_data": {"cache": {"default_ttl": 300}},
                    "is_active": False,  # Make inactive so test records take precedence
                    "created_at": "2024-01-20T10:00:00Z"
                }
            ],
            "performance_metrics": [],
            "feature_flags": []
        }
    
    def table(self, table_name: str):
        """Get table instance"""
        data = self.data_store.get(table_name, [])
        return MockTable(table_name, self.fail_mode, data)
    
    @property 
    def tables(self):
        """For backward compatibility with test code that accesses tables directly"""
        class TablesProxy:
            def __init__(self, client):
                self.client = client
                
            def __getitem__(self, table_name):
                return self.client.table(table_name)
        
        return TablesProxy(self)
    
    def from_(self, table_name: str):
        """Alternative table access method"""
        return self.table(table_name)
    
    def rpc(self, function_name: str, params: dict = None):
        """Mock RPC function calls"""
        if function_name == "validate_configuration":
            if self.fail_mode == "validation_error":
                return MockResult(data=[{"is_valid": False, "errors": ["Validation failed"]}])
            
            # Mock validation logic
            config = params.get("config_data", {}) if params else {}
            is_valid = True
            errors = []
            
            # Simple validation checks
            if not config:
                is_valid = False
                errors.append("Configuration cannot be empty")
            
            return MockResult(data=[{
                "is_valid": is_valid,
                "errors": errors,
                "validation_timestamp": "2024-01-20T10:00:00Z"
            }])
        
        return MockResult(data=[])

class MockTable:
    """Mock Supabase table with full CRUD operations"""
    
    def __init__(self, table_name: str, fail_mode: str = None, data: list = None):
        self.table_name = table_name
        self.fail_mode = fail_mode
        self._data = data or []
        
    def select(self, columns="*"):
        """Mock select operation - returns new query instance"""
        if self.fail_mode == "database_error":
            return MockResult(error="Database connection failed")
        return MockQuery(self._data, self.fail_mode, columns)
    
    def insert(self, data):
        """Mock insert operation"""
        if self.fail_mode == "insert_error":
            return MockResult(error="Insert failed")
        
        # Simulate auto-incrementing ID
        new_record = {**data, "id": len(self._data) + 1, "created_at": "2024-01-20T10:00:00Z"}
        self._data.append(new_record)
        return MockResult(data=[new_record])
    
    def update(self, data):
        """Mock update operation - returns new query instance for chaining"""
        if self.fail_mode == "update_error":
            return MockResult(error="Update failed")
        return MockQuery(self._data, self.fail_mode, update_data=data)
    
    def delete(self):
        """Mock delete operation"""
        if self.fail_mode == "delete_error":
            return MockResult(error="Delete failed")
        return MockResult(data=[])

class MockQuery:
    """Mock Supabase query for method chaining"""
    
    def __init__(self, data: list, fail_mode: str = None, columns: str = "*", update_data: dict = None):
        self._data = data
        self.fail_mode = fail_mode
        self._columns = columns
        self._filters = {}
        self._update_data = update_data
        
    def eq(self, column, value):
        """Mock eq method for filtering"""
        self._filters[column] = value
        return self
        
    def execute(self):
        """Mock execute method"""
        if self.fail_mode == "database_error":
            return MockResult(error="Database connection failed")
        
        if self._update_data:
            # Handle update operation
            updated_records = []
            for record in self._data:
                matches = all(record.get(k) == v for k, v in self._filters.items())
                if matches:
                    record.update(self._update_data)
                    updated_records.append(record)
            return MockResult(data=updated_records)
        else:
            # Handle select operation
            filtered_data = []
            for record in self._data:
                matches = all(record.get(k) == v for k, v in self._filters.items())
                if matches:
                    filtered_data.append(record)
            return MockResult(data=filtered_data)
    
    def single(self):
        """Mock single method - executes and returns single record"""
        if self.fail_mode == "database_error":
            return MockResult(error="Database connection failed")
        
        # Apply filters to find matching records
        for record in self._data:
            matches = all(record.get(k) == v for k, v in self._filters.items())
            if matches:
                return MockResult(data=record)  # Return the record directly for single()
        
        return MockResult(data=None)

class MockResult:
    """Mock Supabase query result"""
    
    def __init__(self, data=None, error=None):
        self.data = data or []
        self.error = error
        self._count = len(self.data) if isinstance(self.data, list) else (1 if self.data else 0)
        
    def eq(self, column, value):
        """Mock eq method for Supabase queries"""
        return self
    
    def execute(self):
        """Mock execute method"""
        return self
        
    def single(self):
        """Mock single method"""
        # If data is already a single record (not a list), return it directly
        if not isinstance(self.data, list):
            return self.data
        # If data is a list, return the first item or None
        if self.data:
            return self.data[0]
        return None

class PerformanceTestData:
    """Test data for performance testing."""
    
    @staticmethod
    def get_query_metrics_sample() -> List[Dict[str, Any]]:
        """Get sample query metrics for performance testing."""
        base_time = datetime.utcnow()
        return [
            {
                "id": f"metric_{i}",
                "query_text": f"Test query {i}",
                "query_type": QueryType.CASINO_REVIEW.value,
                "total_latency_ms": 1000 + (i * 50),
                "retrieval_latency_ms": 300 + (i * 10),
                "generation_latency_ms": 700 + (i * 40),
                "relevance_score": max(0.1, 0.9 - (i * 0.1)),
                "confidence_score": max(0.1, 0.85 - (i * 0.05)),
                "cache_hit": i % 3 == 0,
                "error_occurred": i % 10 == 9,
                "created_at": base_time - timedelta(hours=i)
            }
            for i in range(100)
        ]
    
    @staticmethod
    def get_performance_profiles_sample() -> List[Dict[str, Any]]:
        """Get sample performance profiles."""
        return [
            {
                "id": f"profile_{i}",
                "query_id": f"metric_{i}",
                "embedding_generation_ms": 100 + (i * 5),
                "vector_search_ms": 150 + (i * 3),
                "context_preparation_ms": 50 + (i * 2),
                "llm_inference_ms": 600 + (i * 30),
                "peak_memory_mb": 256 + (i * 10),
                "input_tokens": 500 + (i * 20),
                "output_tokens": 150 + (i * 5),
                "total_cost": 0.001 + (i * 0.0001)
            }
            for i in range(50)
        ] 