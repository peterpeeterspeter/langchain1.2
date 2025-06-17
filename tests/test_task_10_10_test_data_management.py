"""
Test Suite for Task 10.10: Test Data Management & Fixtures

This test suite validates:
- Realistic test dataset generation
- Mock external API responses
- Database state management
- Test data cleanup
- Environment-specific configurations
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

# Import the test data management system
from tests.fixtures.test_data_manager import (
    TestDataManager, TestDataConfig, TestDataCategory, TestDataComplexity,
    DocumentDataGenerator, create_test_data_manager, generate_casino_test_data,
    create_mock_api_responses
)
from tests.fixtures.advanced_test_fixtures import (
    performance_test_config, mock_external_apis, security_test_scenarios,
    edge_case_test_data, test_metrics_collector
)


@pytest.mark.unit
@pytest.mark.task_10_10
class TestDataManagerCore:
    """Test core TestDataManager functionality."""
    
    def test_data_manager_initialization(self):
        """Test TestDataManager initialization with default config."""
        manager = TestDataManager()
        
        assert manager.config is not None
        assert manager.config.seed == 42
        assert manager.config.locale == "en_US"
        assert len(manager.config.categories) > 0
        assert manager.config.complexity == TestDataComplexity.MEDIUM
        assert manager.config.count == 100
    
    def test_data_manager_custom_config(self):
        """Test TestDataManager with custom configuration."""
        config = TestDataConfig(
            seed=12345,
            locale="en_GB",
            categories=[TestDataCategory.CASINO_REVIEW, TestDataCategory.GAME_GUIDE],
            complexity=TestDataComplexity.COMPLEX,
            count=50,
            include_edge_cases=False,
            generate_embeddings=True,
            embedding_dimension=768
        )
        
        manager = TestDataManager(config)
        
        assert manager.config.seed == 12345
        assert manager.config.locale == "en_GB"
        assert len(manager.config.categories) == 2
        assert manager.config.complexity == TestDataComplexity.COMPLEX
        assert manager.config.count == 50
        assert manager.config.include_edge_cases is False
        assert manager.config.generate_embeddings is True
        assert manager.config.embedding_dimension == 768
    
    def test_data_manager_context_manager(self):
        """Test TestDataManager as context manager with cleanup."""
        temp_dirs_created = []
        
        with TestDataManager() as manager:
            # Create some temporary resources
            temp_dir = manager.create_temp_directory()
            temp_dirs_created.append(temp_dir)
            
            assert temp_dir.exists()
            assert len(manager._temp_dirs) == 1
        
        # After context exit, temporary directories should be cleaned up
        for temp_dir in temp_dirs_created:
            assert not temp_dir.exists()
    
    def test_cleanup_callbacks(self):
        """Test cleanup callback registration and execution."""
        callback_executed = {"value": False}
        
        def cleanup_callback():
            callback_executed["value"] = True
        
        with TestDataManager() as manager:
            manager.register_cleanup(cleanup_callback)
            assert len(manager._cleanup_callbacks) == 1
        
        # Callback should be executed during cleanup
        assert callback_executed["value"] is True


@pytest.mark.unit
@pytest.mark.task_10_10
class TestDocumentDataGenerator:
    """Test DocumentDataGenerator functionality."""
    
    def test_document_generation_basic(self):
        """Test basic document generation."""
        with TestDataManager() as manager:
            generator = DocumentDataGenerator(manager)
            documents = generator.generate_documents(10)
            
            assert len(documents) == 10
            
            for doc in documents:
                assert "id" in doc
                assert "content" in doc
                assert "metadata" in doc
                assert "category" in doc
                assert "complexity" in doc
                assert "created_at" in doc
                assert "updated_at" in doc
                
                # Validate content is not empty
                assert len(doc["content"]) > 0
                
                # Validate metadata structure
                metadata = doc["metadata"]
                assert "document_id" in metadata
                assert "category" in metadata
                assert "quality_score" in metadata
                assert "word_count" in metadata
                assert "tags" in metadata
                assert isinstance(metadata["tags"], list)
    
    def test_document_generation_with_embeddings(self):
        """Test document generation with embeddings."""
        config = TestDataConfig(
            generate_embeddings=True,
            embedding_dimension=100,
            count=5
        )
        
        with TestDataManager(config) as manager:
            generator = DocumentDataGenerator(manager)
            documents = generator.generate_documents(5)
            
            for doc in documents:
                assert "embedding" in doc
                assert doc["embedding"] is not None
                assert len(doc["embedding"]) == 100
                assert all(isinstance(x, float) for x in doc["embedding"])
    
    def test_document_complexity_distribution(self):
        """Test that document complexity is distributed correctly."""
        config = TestDataConfig(
            include_edge_cases=True,
            count=100
        )
        
        with TestDataManager(config) as manager:
            generator = DocumentDataGenerator(manager)
            documents = generator.generate_documents(100)
            
            complexities = [doc["complexity"] for doc in documents]
            
            # Check that we have different complexity levels
            unique_complexities = set(complexities)
            assert len(unique_complexities) > 1
            
            # Check distribution roughly matches expected ratios
            simple_count = complexities.count(TestDataComplexity.SIMPLE.value)
            medium_count = complexities.count(TestDataComplexity.MEDIUM.value)
            complex_count = complexities.count(TestDataComplexity.COMPLEX.value)
            edge_case_count = complexities.count(TestDataComplexity.EDGE_CASE.value)
            
            # Simple should be most common (60%)
            assert simple_count > medium_count
            # Edge cases should be least common (5%)
            assert edge_case_count < simple_count
    
    def test_document_categories(self):
        """Test document generation with specific categories."""
        config = TestDataConfig(
            categories=[TestDataCategory.CASINO_REVIEW, TestDataCategory.GAME_GUIDE],
            count=20
        )
        
        with TestDataManager(config) as manager:
            generator = DocumentDataGenerator(manager)
            documents = generator.generate_documents(20)
            
            categories = [doc["category"] for doc in documents]
            unique_categories = set(categories)
            
            assert len(unique_categories) <= 2
            assert TestDataCategory.CASINO_REVIEW.value in unique_categories
            assert TestDataCategory.GAME_GUIDE.value in unique_categories
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        config1 = TestDataConfig(seed=12345, count=5)
        config2 = TestDataConfig(seed=12345, count=5)
        
        with TestDataManager(config1) as manager1:
            generator1 = DocumentDataGenerator(manager1)
            documents1 = generator1.generate_documents(5)
        
        with TestDataManager(config2) as manager2:
            generator2 = DocumentDataGenerator(manager2)
            documents2 = generator2.generate_documents(5)
        
        # Documents should be identical with same seed
        for doc1, doc2 in zip(documents1, documents2):
            assert doc1["id"] == doc2["id"]
            assert doc1["content"] == doc2["content"]
            assert doc1["category"] == doc2["category"]


@pytest.mark.unit
@pytest.mark.task_10_10
class TestMockAPIResponses:
    """Test mock API response generation."""
    
    def test_create_mock_api_responses(self):
        """Test creation of mock API responses."""
        responses = create_mock_api_responses()
        
        assert "openai" in responses
        assert "anthropic" in responses
        assert "dataforseo" in responses
        
        # Check OpenAI responses
        openai_responses = responses["openai"]
        assert len(openai_responses) == 20
        
        for response in openai_responses[:3]:  # Check first few
            assert "id" in response
            assert "object" in response
            assert response["object"] == "chat.completion"
            assert "choices" in response
            assert "usage" in response
            assert len(response["choices"]) > 0
            assert "message" in response["choices"][0]
        
        # Check Anthropic responses
        anthropic_responses = responses["anthropic"]
        assert len(anthropic_responses) == 20
        
        for response in anthropic_responses[:3]:  # Check first few
            assert "id" in response
            assert "type" in response
            assert response["type"] == "message"
            assert "content" in response
            assert "usage" in response
        
        # Check DataForSEO responses
        dataforseo_responses = responses["dataforseo"]
        assert len(dataforseo_responses) == 10
        
        for response in dataforseo_responses[:3]:  # Check first few
            assert "status_code" in response
            assert response["status_code"] == 20000
            assert "tasks" in response
            assert len(response["tasks"]) > 0
    
    def test_mock_external_apis_fixture(self, mock_external_apis):
        """Test the mock external APIs fixture."""
        # Test normal operation
        openai_response = mock_external_apis.mock_openai_response("test prompt")
        
        assert "id" in openai_response
        assert "choices" in openai_response
        assert mock_external_apis.call_counts["openai"] == 1
        
        # Test failure mode
        mock_external_apis.set_failure_mode("openai", "rate_limit")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            mock_external_apis.mock_openai_response("another prompt")
        
        # Call count should still increment even on failure
        assert mock_external_apis.call_counts["openai"] == 2
        
        # Test API stats
        stats = mock_external_apis.get_api_stats()
        assert stats["total_calls"] == 2
        assert stats["failure_modes"]["openai"] == "rate_limit"


@pytest.mark.integration
@pytest.mark.task_10_10
class TestDatabaseSeeding:
    """Test database seeding functionality."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        mock_client = Mock()
        
        # Mock table operations
        mock_table = Mock()
        mock_client.table.return_value = mock_table
        
        # Mock insert operation
        async def mock_insert(data):
            return Mock(data=data)
        
        mock_table.insert.return_value.execute = AsyncMock(side_effect=mock_insert)
        
        return mock_client
    
    @pytest.mark.asyncio
    async def test_database_seeding_basic(self, mock_supabase_client):
        """Test basic database seeding functionality."""
        from tests.fixtures.test_data_manager import seed_test_database
        
        # Define tables to seed
        table_counts = {
            "documents": 10,
            "queries": 5,
            "responses": 8
        }
        
        # Seed the database
        seeded_data = await seed_test_database(mock_supabase_client, table_counts)
        
        # Verify seeded data
        assert "documents" in seeded_data
        assert "queries" in seeded_data
        assert "responses" in seeded_data
        
        assert len(seeded_data["documents"]) == 10
        assert len(seeded_data["queries"]) == 5
        assert len(seeded_data["responses"]) == 8
        
        # Verify client was called correctly
        assert mock_supabase_client.table.call_count == 3
    
    @pytest.mark.asyncio
    async def test_database_seeding_with_mock_client(self):
        """Test database seeding with mock client."""
        from tests.fixtures.test_data_manager import seed_test_database
        
        # Create a simple mock that doesn't have table method
        mock_client = Mock()
        mock_client.table = None
        
        # Should handle gracefully and return the generated data
        seeded_data = await seed_test_database(mock_client, {"documents": 5})
        
        assert "documents" in seeded_data
        assert len(seeded_data["documents"]) == 5


@pytest.mark.unit
@pytest.mark.task_10_10
class TestConvenienceFunctions:
    """Test convenience functions for common scenarios."""
    
    def test_create_test_data_manager(self):
        """Test create_test_data_manager convenience function."""
        manager = create_test_data_manager(seed=999, count=50)
        
        assert manager.config.seed == 999
        assert manager.config.count == 50
        assert isinstance(manager, TestDataManager)
    
    def test_generate_casino_test_data(self):
        """Test generate_casino_test_data convenience function."""
        test_data = generate_casino_test_data(count=20)
        
        assert "documents" in test_data
        assert len(test_data["documents"]) == 20
        
        # Verify all documents are casino-related categories
        categories = [doc["category"] for doc in test_data["documents"]]
        valid_categories = {
            TestDataCategory.CASINO_REVIEW.value,
            TestDataCategory.GAME_GUIDE.value,
            TestDataCategory.PROMOTION.value
        }
        
        for category in categories:
            assert category in valid_categories


@pytest.mark.integration
@pytest.mark.task_10_10
class TestAdvancedFixtures:
    """Test advanced fixtures and scenarios."""
    
    def test_performance_test_config(self, performance_test_config):
        """Test performance test configuration fixture."""
        assert "load_testing" in performance_test_config
        assert "stress_testing" in performance_test_config
        assert "benchmark_testing" in performance_test_config
        
        load_config = performance_test_config["load_testing"]
        assert "concurrent_users" in load_config
        assert "request_rates" in load_config
        assert isinstance(load_config["concurrent_users"], list)
        assert len(load_config["concurrent_users"]) > 0
    
    def test_security_test_scenarios(self, security_test_scenarios):
        """Test security test scenarios fixture."""
        assert "injection_attacks" in security_test_scenarios
        assert "authentication_bypass" in security_test_scenarios
        assert "rate_limiting" in security_test_scenarios
        
        injection_attacks = security_test_scenarios["injection_attacks"]
        assert "sql_injection" in injection_attacks
        assert "prompt_injection" in injection_attacks
        assert isinstance(injection_attacks["sql_injection"], list)
        assert len(injection_attacks["sql_injection"]) > 0
    
    def test_edge_case_test_data(self, edge_case_test_data):
        """Test edge case test data fixture."""
        assert "extreme_inputs" in edge_case_test_data
        assert "boundary_values" in edge_case_test_data
        
        extreme_inputs = edge_case_test_data["extreme_inputs"]
        assert "empty_strings" in extreme_inputs
        assert "very_long_strings" in extreme_inputs
        assert "special_characters" in extreme_inputs
        
        # Verify edge cases are actually extreme
        long_strings = extreme_inputs["very_long_strings"]
        assert any(len(s) > 1000 for s in long_strings)
    
    def test_test_metrics_collector(self, test_metrics_collector):
        """Test test metrics collector fixture."""
        # Record some test metrics
        test_metrics_collector.record_execution_time("test_operation", 150.5)
        test_metrics_collector.record_cache_hit()
        test_metrics_collector.record_cache_miss()
        test_metrics_collector.record_cache_hit()
        
        # Get summary
        summary = test_metrics_collector.get_summary()
        
        assert summary["total_operations"] == 1
        assert summary["avg_execution_time_ms"] == 150.5
        assert summary["cache_hit_rate"] == 2/3  # 2 hits, 1 miss
        assert summary["total_errors"] == 0


@pytest.mark.performance
@pytest.mark.task_10_10
class TestPerformanceScenarios:
    """Test performance-related test data scenarios."""
    
    def test_large_dataset_generation(self):
        """Test generation of large datasets for performance testing."""
        config = TestDataConfig(
            count=1000,
            generate_embeddings=False  # Skip embeddings for speed
        )
        
        start_time = datetime.now()
        
        with TestDataManager(config) as manager:
            generator = DocumentDataGenerator(manager)
            documents = generator.generate_documents(1000)
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Verify large dataset
        assert len(documents) == 1000
        
        # Performance assertion (should generate 1000 docs in reasonable time)
        assert generation_time < 10.0  # Less than 10 seconds
        
        # Verify data quality maintained
        for doc in documents[:10]:  # Sample first 10
            assert len(doc["content"]) > 0
            assert "metadata" in doc
            assert len(doc["metadata"]["tags"]) > 0
    
    def test_memory_efficient_generation(self):
        """Test memory-efficient generation for large datasets."""
        # Generate documents in batches to test memory efficiency
        total_docs = 500
        batch_size = 100
        all_documents = []
        
        with TestDataManager() as manager:
            generator = DocumentDataGenerator(manager)
            
            for i in range(0, total_docs, batch_size):
                batch_count = min(batch_size, total_docs - i)
                batch_docs = generator.generate_documents(batch_count)
                all_documents.extend(batch_docs)
                
                # Verify batch quality
                assert len(batch_docs) == batch_count
        
        # Verify total results
        assert len(all_documents) == total_docs
        
        # Verify unique IDs (no duplicates across batches)
        doc_ids = [doc["id"] for doc in all_documents]
        assert len(set(doc_ids)) == len(doc_ids)


@pytest.mark.integration
@pytest.mark.task_10_10
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows using test data management."""
    
    @pytest.mark.asyncio
    async def test_complete_test_data_workflow(self, mock_external_apis, test_metrics_collector):
        """Test complete workflow from data generation to API mocking."""
        # Step 1: Generate test data
        test_metrics_collector.record_execution_time("data_generation_start", 0)
        
        casino_data = generate_casino_test_data(count=10)
        documents = casino_data["documents"]
        
        test_metrics_collector.record_execution_time("data_generation_complete", 100)
        
        # Step 2: Mock API responses
        api_responses = create_mock_api_responses()
        
        # Step 3: Simulate processing workflow
        processed_results = []
        
        for i, doc in enumerate(documents[:3]):  # Process first 3 documents
            # Simulate AI processing
            prompt = f"Analyze this document: {doc['content'][:100]}..."
            
            try:
                ai_response = mock_external_apis.mock_openai_response(prompt)
                test_metrics_collector.record_cache_miss()  # First time processing
                
                processed_result = {
                    "document_id": doc["id"],
                    "ai_response": ai_response,
                    "processing_time_ms": 150 + (i * 10),
                    "success": True
                }
                
                test_metrics_collector.record_execution_time(f"process_doc_{i}", 150 + (i * 10))
                
            except Exception as e:
                processed_result = {
                    "document_id": doc["id"],
                    "error": str(e),
                    "success": False
                }
            
            processed_results.append(processed_result)
        
        # Step 4: Verify workflow results
        assert len(processed_results) == 3
        assert all(result["success"] for result in processed_results)
        
        # Step 5: Check metrics
        api_stats = mock_external_apis.get_api_stats()
        assert api_stats["total_calls"] == 3
        assert api_stats["call_counts"]["openai"] == 3
        
        metrics_summary = test_metrics_collector.get_summary()
        assert metrics_summary["total_operations"] >= 4  # data_gen + 3 doc processing
        assert metrics_summary["cache_hit_rate"] == 0.0  # All cache misses in this test
    
    def test_test_environment_isolation(self):
        """Test that test environments are properly isolated."""
        # Create two separate test environments
        env1_data = []
        env2_data = []
        
        # Environment 1
        with TestDataManager(TestDataConfig(seed=111, count=5)) as manager1:
            generator1 = DocumentDataGenerator(manager1)
            env1_data = generator1.generate_documents(5)
        
        # Environment 2  
        with TestDataManager(TestDataConfig(seed=222, count=5)) as manager2:
            generator2 = DocumentDataGenerator(manager2)
            env2_data = generator2.generate_documents(5)
        
        # Verify environments are isolated (different data)
        assert len(env1_data) == len(env2_data) == 5
        
        # Data should be different due to different seeds
        env1_content = [doc["content"] for doc in env1_data]
        env2_content = [doc["content"] for doc in env2_data]
        
        # At least some content should be different
        assert env1_content != env2_content
        
        # But structure should be the same
        for doc1, doc2 in zip(env1_data, env2_data):
            assert set(doc1.keys()) == set(doc2.keys())


@pytest.mark.unit
@pytest.mark.task_10_10
class TestErrorHandling:
    """Test error handling in test data management."""
    
    def test_missing_faker_fallback(self):
        """Test fallback behavior when Faker is not available."""
        # Temporarily disable faker
        with patch('tests.fixtures.test_data_manager.FAKER_AVAILABLE', False):
            with TestDataManager() as manager:
                assert manager.faker is None
                
                generator = DocumentDataGenerator(manager)
                documents = generator.generate_documents(3)
                
                # Should still generate documents, just with simpler content
                assert len(documents) == 3
                for doc in documents:
                    assert "content" in doc
                    assert len(doc["content"]) > 0
    
    def test_missing_numpy_fallback(self):
        """Test fallback behavior when NumPy is not available."""
        config = TestDataConfig(generate_embeddings=True, embedding_dimension=10)
        
        with patch('tests.fixtures.test_data_manager.NUMPY_AVAILABLE', False):
            with TestDataManager(config) as manager:
                generator = DocumentDataGenerator(manager)
                documents = generator.generate_documents(2)
                
                # Should still generate embeddings using fallback method
                for doc in documents:
                    assert "embedding" in doc
                    assert len(doc["embedding"]) == 10
                    assert all(isinstance(x, float) for x in doc["embedding"])
    
    def test_cleanup_error_handling(self):
        """Test that cleanup errors don't crash the system."""
        def failing_callback():
            raise Exception("Cleanup failed")
        
        # Should not raise exception even with failing callback
        with TestDataManager() as manager:
            manager.register_cleanup(failing_callback)
            # Context exit should handle the exception gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 