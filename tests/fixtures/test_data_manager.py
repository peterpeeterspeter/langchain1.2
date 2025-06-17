"""
Comprehensive Test Data Management & Fixtures System
Task 10.10: Test Data Management & Fixtures

This module provides:
- Realistic test dataset generation
- Mock external API responses
- Database state management
- Test data cleanup
- Environment-specific configurations
"""

import json
import uuid
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import tempfile
import shutil

# Core imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

# Project imports
from src.chains.enhanced_confidence_scoring_system import (
    EnhancedRAGResponse, ConfidenceFactors, QueryType
)


class TestDataCategory(Enum):
    """Categories of test data for different testing scenarios."""
    CASINO_REVIEW = "casino_review"
    GAME_GUIDE = "game_guide"
    PROMOTION = "promotion"
    STRATEGY = "strategy"
    NEWS = "news"
    TECHNICAL_DOC = "technical_doc"
    FAQ = "faq"
    COMPARISON = "comparison"


class TestDataComplexity(Enum):
    """Complexity levels for test data generation."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EDGE_CASE = "edge_case"


@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    seed: int = 42
    locale: str = "en_US"
    categories: List[TestDataCategory] = None
    complexity: TestDataComplexity = TestDataComplexity.MEDIUM
    count: int = 100
    include_edge_cases: bool = True
    generate_embeddings: bool = False
    embedding_dimension: int = 1536
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = list(TestDataCategory)


class TestDataManager:
    """
    Comprehensive test data management system.
    
    Features:
    - Realistic dataset generation
    - Mock API response creation
    - Database state management
    - Test environment isolation
    - Cleanup automation
    """
    
    def __init__(self, config: TestDataConfig = None):
        self.config = config or TestDataConfig()
        self.faker = None
        self._temp_dirs = []
        self._cleanup_callbacks = []
        
        # Initialize faker if available
        if FAKER_AVAILABLE:
            self.faker = Faker(self.config.locale)
            self.faker.seed_instance(self.config.seed)
        
        # Set numpy seed if available
        if NUMPY_AVAILABLE:
            np.random.seed(self.config.seed)
        
        # Set Python random seed
        random.seed(self.config.seed)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up all temporary resources."""
        # Clean up temporary directories
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        # Execute cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Warning: Cleanup callback failed: {e}")
        
        self._temp_dirs.clear()
        self._cleanup_callbacks.clear()
    
    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback."""
        self._cleanup_callbacks.append(callback)
    
    def create_temp_directory(self) -> Path:
        """Create a temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_data_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir


class DocumentDataGenerator:
    """Generate realistic document test data."""
    
    def __init__(self, manager: TestDataManager):
        self.manager = manager
        self.faker = manager.faker
        self._doc_counter = 0  # Add counter for unique IDs
    
    def generate_documents(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic document data."""
        documents = []
        
        for i in range(count):
            category = random.choice(self.manager.config.categories)
            complexity = self._determine_complexity(i, count)
            
            doc = self._create_document(self._doc_counter, category, complexity)
            documents.append(doc)
            self._doc_counter += 1  # Increment counter for unique IDs
        
        return documents
    
    def _determine_complexity(self, index: int, total: int) -> TestDataComplexity:
        """Determine complexity based on index and configuration."""
        if not self.manager.config.include_edge_cases:
            return self.manager.config.complexity
        
        # Distribute complexities
        ratio = index / total
        if ratio < 0.6:
            return TestDataComplexity.SIMPLE
        elif ratio < 0.8:
            return TestDataComplexity.MEDIUM
        elif ratio < 0.95:
            return TestDataComplexity.COMPLEX
        else:
            return TestDataComplexity.EDGE_CASE
    
    def _create_document(self, index: int, category: TestDataCategory, 
                        complexity: TestDataComplexity) -> Dict[str, Any]:
        """Create a single document with realistic content."""
        
        # Generate content based on category and complexity
        content = self._generate_content(category, complexity, index)
        
        # Generate metadata
        metadata = self._generate_metadata(category, complexity, index)
        
        # Generate embedding if requested
        embedding = None
        if self.manager.config.generate_embeddings:
            embedding = self._generate_embedding(content)
        
        return {
            "id": f"doc_{index:04d}",
            "content": content,
            "metadata": metadata,
            "embedding": embedding,
            "category": category.value,
            "complexity": complexity.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _generate_content(self, category: TestDataCategory, 
                         complexity: TestDataComplexity, index: int) -> str:
        """Generate realistic content based on category and complexity."""
        
        if not self.faker:
            return f"Test document {index} for {category.value} with {complexity.value} complexity."
        
        # Base content templates by category
        templates = {
            TestDataCategory.CASINO_REVIEW: [
                "comprehensive review of {casino_name} casino",
                "detailed analysis of {casino_name} gaming platform",
                "in-depth evaluation of {casino_name} online casino"
            ],
            TestDataCategory.GAME_GUIDE: [
                "complete guide to playing {game_name}",
                "strategy guide for {game_name} players",
                "beginner's tutorial for {game_name}"
            ],
            TestDataCategory.PROMOTION: [
                "exclusive {promotion_type} promotion details",
                "limited time {promotion_type} offer analysis",
                "comprehensive {promotion_type} bonus review"
            ]
        }
        
        # Get template for category
        category_templates = templates.get(category, [
            f"detailed information about {category.value}",
            f"comprehensive guide to {category.value}",
            f"expert analysis of {category.value}"
        ])
        
        template = random.choice(category_templates)
        
        # Fill template with fake data
        content_parts = [
            template.format(
                casino_name=self.faker.company(),
                game_name=random.choice(["Blackjack", "Poker", "Roulette", "Slots"]),
                promotion_type=random.choice(["Welcome Bonus", "Free Spins", "Cashback"])
            )
        ]
        
        # Add complexity-based content
        if complexity in [TestDataComplexity.MEDIUM, TestDataComplexity.COMPLEX]:
            content_parts.extend([
                self.faker.paragraph(nb_sentences=5),
                self.faker.paragraph(nb_sentences=3)
            ])
        
        if complexity == TestDataComplexity.COMPLEX:
            content_parts.extend([
                self.faker.paragraph(nb_sentences=7),
                self.faker.paragraph(nb_sentences=4)
            ])
        
        if complexity == TestDataComplexity.EDGE_CASE:
            # Add edge case content (very long, special characters, etc.)
            content_parts.extend([
                "Special characters: àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ",
                "Numbers and symbols: 1234567890 !@#$%^&*()_+-=[]{}|;:,.<>?",
                self.faker.paragraph(nb_sentences=15)  # Very long paragraph
            ])
        
        return " ".join(content_parts)
    
    def _generate_metadata(self, category: TestDataCategory, 
                          complexity: TestDataComplexity, index: int) -> Dict[str, Any]:
        """Generate realistic metadata."""
        
        base_metadata = {
            "document_id": f"doc_{index:04d}",
            "category": category.value,
            "complexity": complexity.value,
            "source": "test_generator",
            "quality_score": round(random.uniform(0.3, 0.95), 2),
            "word_count": random.randint(100, 2000),
            "reading_level": random.choice(["beginner", "intermediate", "advanced"]),
            "language": "en",
            "tags": self._generate_tags(category)
        }
        
        if self.faker:
            base_metadata.update({
                "title": self.faker.sentence(nb_words=6).rstrip('.'),
                "author": self.faker.name(),
                "published_date": self.faker.date_between(
                    start_date='-2y', end_date='today'
                ).isoformat(),
                "url": self.faker.url(),
                "domain": self.faker.domain_name()
            })
        
        # Add complexity-specific metadata
        if complexity == TestDataComplexity.COMPLEX:
            base_metadata.update({
                "references": [self.faker.url() for _ in range(3)] if self.faker else [],
                "citations": random.randint(5, 15),
                "expert_reviewed": True
            })
        
        if complexity == TestDataComplexity.EDGE_CASE:
            base_metadata.update({
                "special_encoding": "utf-8",
                "contains_tables": True,
                "contains_images": True,
                "file_size_kb": random.randint(1000, 5000)
            })
        
        return base_metadata
    
    def _generate_tags(self, category: TestDataCategory) -> List[str]:
        """Generate relevant tags for the category."""
        tag_sets = {
            TestDataCategory.CASINO_REVIEW: [
                "casino", "review", "gambling", "online", "bonus", "games"
            ],
            TestDataCategory.GAME_GUIDE: [
                "guide", "strategy", "tutorial", "gaming", "tips", "rules"
            ],
            TestDataCategory.PROMOTION: [
                "promotion", "bonus", "offer", "deal", "discount", "special"
            ]
        }
        
        available_tags = tag_sets.get(category, ["general", "content", "test"])
        return random.sample(available_tags, k=min(3, len(available_tags)))
    
    def _generate_embedding(self, content: str) -> List[float]:
        """Generate deterministic embedding for content."""
        if not NUMPY_AVAILABLE:
            # Fallback to simple hash-based embedding
            content_hash = hashlib.md5(content.encode()).hexdigest()
            seed = int(content_hash[:8], 16)
            random.seed(seed)
            return [random.uniform(-1, 1) for _ in range(self.manager.config.embedding_dimension)]
        
        # Use numpy for more realistic embeddings
        content_hash = hashlib.md5(content.encode()).hexdigest()
        seed = int(content_hash[:8], 16)
        np.random.seed(seed)
        
        # Generate embedding with some structure
        embedding = np.random.normal(0, 0.1, self.manager.config.embedding_dimension)
        
        # Add some category-specific patterns
        if "casino" in content.lower():
            embedding[:10] += 0.5
        if "game" in content.lower():
            embedding[10:20] += 0.5
        if "bonus" in content.lower():
            embedding[20:30] += 0.5
        
        return embedding.tolist()


class QueryDataGenerator:
    """Generate realistic query test data."""
    
    def __init__(self, manager: TestDataManager):
        self.manager = manager
        self.faker = manager.faker
    
    def generate_queries(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic query data."""
        queries = []
        
        for i in range(count):
            query = self._create_query(i)
            queries.append(query)
        
        return queries
    
    def _create_query(self, index: int) -> Dict[str, Any]:
        """Create a single query with realistic content."""
        
        # Query templates by type
        query_templates = {
            QueryType.CASINO_REVIEW: [
                "best online casino for {game}",
                "review of {casino} casino",
                "is {casino} casino safe and reliable",
                "{casino} casino bonus codes"
            ],
            QueryType.GAME_GUIDE: [
                "how to play {game}",
                "{game} strategy guide",
                "best {game} tips for beginners",
                "{game} rules and regulations"
            ],
            QueryType.PROMOTION_ANALYSIS: [
                "best casino bonuses {month}",
                "no deposit bonus codes",
                "free spins promotions",
                "cashback offers comparison"
            ]
        }
        
        # Select random query type and template
        query_type = random.choice(list(QueryType))
        templates = query_templates.get(query_type, ["general query about gambling"])
        template = random.choice(templates)
        
        # Fill template with realistic data
        query_text = template.format(
            game=random.choice(["blackjack", "poker", "roulette", "slots"]),
            casino=self.faker.company().lower().replace(' ', '') if self.faker else "testcasino",
            month=random.choice(["2024", "this month", "January", "February"])
        )
        
        return {
            "id": f"query_{index:04d}",
            "query_text": query_text,
            "query_type": query_type.value,
            "expected_results": random.randint(5, 20),
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "metadata": {
                "user_intent": self._determine_intent(query_type),
                "expected_categories": self._get_expected_categories(query_type),
                "complexity_score": random.uniform(0.1, 0.9)
            },
            "created_at": datetime.now().isoformat()
        }
    
    def _determine_intent(self, query_type: QueryType) -> str:
        """Determine user intent based on query type."""
        intent_mapping = {
            QueryType.CASINO_REVIEW: "research",
            QueryType.GAME_GUIDE: "learning",
            QueryType.PROMOTION_ANALYSIS: "comparison",
            QueryType.STRATEGY_GUIDE: "improvement",
            QueryType.NEWS_SUMMARY: "information",
            QueryType.FAQ_RESPONSE: "support",
            QueryType.COMPARISON_ANALYSIS: "decision_making",
            QueryType.TECHNICAL_EXPLANATION: "understanding"
        }
        return intent_mapping.get(query_type, "general")
    
    def _get_expected_categories(self, query_type: QueryType) -> List[str]:
        """Get expected document categories for query type."""
        category_mapping = {
            QueryType.CASINO_REVIEW: ["casino_review", "comparison"],
            QueryType.GAME_GUIDE: ["game_guide", "strategy"],
            QueryType.PROMOTION_ANALYSIS: ["promotion", "casino_review"],
            QueryType.STRATEGY_GUIDE: ["strategy", "game_guide"],
            QueryType.NEWS_SUMMARY: ["news"],
            QueryType.FAQ_RESPONSE: ["faq"],
            QueryType.COMPARISON_ANALYSIS: ["comparison", "casino_review"],
            QueryType.TECHNICAL_EXPLANATION: ["technical_doc"]
        }
        return category_mapping.get(query_type, ["general"])


class MockAPIResponseGenerator:
    """Generate mock API responses for external services."""
    
    def __init__(self, manager: TestDataManager):
        self.manager = manager
        self.faker = manager.faker
    
    def generate_openai_responses(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock OpenAI API responses."""
        responses = []
        
        for i in range(count):
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self._generate_ai_response(i)
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": random.randint(100, 500),
                    "completion_tokens": random.randint(50, 300),
                    "total_tokens": random.randint(150, 800)
                }
            }
            responses.append(response)
        
        return responses
    
    def generate_anthropic_responses(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate mock Anthropic API responses."""
        responses = []
        
        for i in range(count):
            response = {
                "id": f"msg_{uuid.uuid4().hex[:20]}",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": self._generate_ai_response(i)
                }],
                "model": "claude-3-sonnet-20240229",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": random.randint(100, 500),
                    "output_tokens": random.randint(50, 300)
                }
            }
            responses.append(response)
        
        return responses
    
    def generate_dataforseo_responses(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate mock DataForSEO API responses."""
        responses = []
        
        for i in range(count):
            response = {
                "version": "0.1.20240801",
                "status_code": 20000,
                "status_message": "Ok.",
                "time": f"0.{random.randint(100, 999)} sec.",
                "cost": round(random.uniform(0.001, 0.01), 4),
                "tasks_count": 1,
                "tasks_error": 0,
                "tasks": [{
                    "id": f"task_{uuid.uuid4().hex[:8]}",
                    "status_code": 20000,
                    "status_message": "Ok.",
                    "time": f"0.{random.randint(100, 999)} sec.",
                    "cost": round(random.uniform(0.001, 0.01), 4),
                    "result_count": random.randint(5, 25),
                    "path": ["v3", "serp", "google", "images", "live", "advanced"],
                    "data": {
                        "api": "serp",
                        "function": "images",
                        "se": "google"
                    },
                    "result": [{
                        "keyword": f"test query {i}",
                        "type": "images",
                        "se_domain": "google.com",
                        "location_code": 2840,
                        "language_code": "en",
                        "check_url": f"https://www.google.com/search?q=test+query+{i}",
                        "datetime": datetime.now().isoformat(),
                        "items_count": random.randint(10, 100),
                        "items": self._generate_image_items(random.randint(5, 15))
                    }]
                }]
            }
            responses.append(response)
        
        return responses
    
    def _generate_ai_response(self, index: int) -> str:
        """Generate realistic AI response content."""
        if not self.faker:
            return f"This is a test AI response number {index}."
        
        responses = [
            self.faker.paragraph(nb_sentences=3),
            self.faker.paragraph(nb_sentences=5),
            f"Based on the information provided, {self.faker.sentence()}",
            f"Here's a comprehensive analysis: {self.faker.paragraph(nb_sentences=4)}"
        ]
        
        return random.choice(responses)
    
    def _generate_image_items(self, count: int) -> List[Dict[str, Any]]:
        """Generate mock image search results."""
        items = []
        
        for i in range(count):
            item = {
                "type": "images_element",
                "rank_group": i + 1,
                "rank_absolute": i + 1,
                "position": "left",
                "xpath": f"/html[1]/body[1]/div[7]/div[1]/div[10]/div[1]/div[2]/div[2]/div[1]/div[1]/div[{i+1}]",
                "title": self.faker.sentence(nb_words=4).rstrip('.') if self.faker else f"Test Image {i}",
                "subtitle": self.faker.company() if self.faker else f"Test Source {i}",
                "alt": self.faker.sentence(nb_words=6).rstrip('.') if self.faker else f"Alt text {i}",
                "url": f"https://example.com/image_{i}.jpg",
                "source_url": f"https://example.com/page_{i}.html",
                "encoded_url": f"https://encrypted-tbn0.gstatic.com/images?q=tbn:test_{i}",
                "width": random.choice([800, 1024, 1200, 1600]),
                "height": random.choice([600, 768, 900, 1200])
            }
            items.append(item)
        
        return items


class DatabaseSeeder:
    """Seed test databases with realistic data."""
    
    def __init__(self, manager: TestDataManager):
        self.manager = manager
        self.document_generator = DocumentDataGenerator(manager)
        self.query_generator = QueryDataGenerator(manager)
    
    async def seed_database(self, client, tables: Dict[str, int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Seed database with test data.
        
        Args:
            client: Database client (Supabase or mock)
            tables: Dict of table names and record counts to create
        
        Returns:
            Dict of created data by table name
        """
        if tables is None:
            tables = {
                "documents": 100,
                "queries": 50,
                "responses": 75,
                "configurations": 10
            }
        
        seeded_data = {}
        
        for table_name, count in tables.items():
            if table_name == "documents":
                data = self.document_generator.generate_documents(count)
            elif table_name == "queries":
                data = self.query_generator.generate_queries(count)
            elif table_name == "responses":
                data = self._generate_responses(count)
            elif table_name == "configurations":
                data = self._generate_configurations(count)
            else:
                data = self._generate_generic_data(table_name, count)
            
            # Insert data into database
            try:
                if hasattr(client, 'table'):
                    # Supabase client
                    result = await client.table(table_name).insert(data).execute()
                    seeded_data[table_name] = result.data
                else:
                    # Mock client
                    seeded_data[table_name] = data
            except Exception as e:
                print(f"Warning: Failed to seed {table_name}: {e}")
                seeded_data[table_name] = data
        
        return seeded_data
    
    def _generate_responses(self, count: int) -> List[Dict[str, Any]]:
        """Generate test response data."""
        responses = []
        
        for i in range(count):
            response = {
                "id": f"response_{i:04d}",
                "query_id": f"query_{random.randint(0, 49):04d}",
                "content": self.manager.faker.paragraph(nb_sentences=5) if self.manager.faker else f"Test response {i}",
                "confidence_score": round(random.uniform(0.3, 0.95), 2),
                "sources": [
                    {
                        "document_id": f"doc_{random.randint(0, 99):04d}",
                        "relevance_score": round(random.uniform(0.5, 1.0), 2),
                        "chunk_index": random.randint(0, 10)
                    }
                    for _ in range(random.randint(1, 5))
                ],
                "metadata": {
                    "generation_time_ms": random.randint(500, 3000),
                    "token_count": random.randint(100, 500),
                    "model_used": random.choice(["gpt-4", "claude-3-sonnet"]),
                    "cache_hit": random.choice([True, False])
                },
                "created_at": datetime.now().isoformat()
            }
            responses.append(response)
        
        return responses
    
    def _generate_configurations(self, count: int) -> List[Dict[str, Any]]:
        """Generate test configuration data."""
        configurations = []
        
        for i in range(count):
            config = {
                "id": f"config_{i:04d}",
                "name": f"test_config_{i}",
                "config_data": {
                    "cache": {
                        "default_ttl": random.randint(300, 3600),
                        "max_size": random.randint(100, 1000)
                    },
                    "retrieval": {
                        "max_results": random.randint(5, 20),
                        "similarity_threshold": round(random.uniform(0.5, 0.9), 2)
                    },
                    "generation": {
                        "max_tokens": random.randint(500, 2000),
                        "temperature": round(random.uniform(0.1, 0.9), 2)
                    }
                },
                "is_active": i == 0,  # Make first config active
                "version": f"1.{i}.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            configurations.append(config)
        
        return configurations
    
    def _generate_generic_data(self, table_name: str, count: int) -> List[Dict[str, Any]]:
        """Generate generic test data for unknown tables."""
        data = []
        
        for i in range(count):
            record = {
                "id": f"{table_name}_{i:04d}",
                "name": f"Test {table_name} {i}",
                "data": {"test": True, "index": i},
                "created_at": datetime.now().isoformat()
            }
            data.append(record)
        
        return data


class TestEnvironmentManager:
    """Manage test environment isolation and cleanup."""
    
    def __init__(self, manager: TestDataManager):
        self.manager = manager
        self.active_environments = {}
        self.isolation_enabled = True
    
    def create_isolated_environment(self, env_name: str) -> Dict[str, Any]:
        """Create an isolated test environment."""
        env_id = f"{env_name}_{uuid.uuid4().hex[:8]}"
        
        environment = {
            "id": env_id,
            "name": env_name,
            "created_at": datetime.now(),
            "temp_dir": self.manager.create_temp_directory(),
            "database_prefix": f"test_{env_id}_",
            "cleanup_callbacks": [],
            "resources": {}
        }
        
        self.active_environments[env_id] = environment
        
        # Register cleanup
        self.manager.register_cleanup(lambda: self.cleanup_environment(env_id))
        
        return environment
    
    def cleanup_environment(self, env_id: str):
        """Clean up a specific test environment."""
        if env_id not in self.active_environments:
            return
        
        environment = self.active_environments[env_id]
        
        # Execute environment-specific cleanup callbacks
        for callback in environment["cleanup_callbacks"]:
            try:
                callback()
            except Exception as e:
                print(f"Warning: Environment cleanup callback failed: {e}")
        
        # Clean up resources
        for resource_name, resource in environment["resources"].items():
            try:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            except Exception as e:
                print(f"Warning: Failed to cleanup resource {resource_name}: {e}")
        
        # Remove from active environments
        del self.active_environments[env_id]
    
    def get_environment_config(self, env_id: str) -> Dict[str, Any]:
        """Get configuration for a specific environment."""
        if env_id not in self.active_environments:
            raise ValueError(f"Environment {env_id} not found")
        
        environment = self.active_environments[env_id]
        
        return {
            "database_url": f"test_db_{env_id}",
            "temp_directory": str(environment["temp_dir"]),
            "isolation_prefix": environment["database_prefix"],
            "environment_id": env_id
        }


# Convenience functions for common test data scenarios
def create_test_data_manager(seed: int = 42, **kwargs) -> TestDataManager:
    """Create a test data manager with common configuration."""
    config = TestDataConfig(seed=seed, **kwargs)
    return TestDataManager(config)


def generate_casino_test_data(count: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """Generate casino-specific test data."""
    config = TestDataConfig(
        categories=[TestDataCategory.CASINO_REVIEW, TestDataCategory.GAME_GUIDE, 
                   TestDataCategory.PROMOTION],
        count=count
    )
    
    with TestDataManager(config) as manager:
        doc_generator = DocumentDataGenerator(manager)
        
        return {
            "documents": doc_generator.generate_documents(count)
        }


def create_mock_api_responses() -> Dict[str, List[Dict[str, Any]]]:
    """Create mock API responses for testing external integrations."""
    responses = {
        "openai": [],
        "anthropic": [],
        "dataforseo": []
    }
    
    # Generate OpenAI mock responses
    for i in range(20):
        responses["openai"].append({
            "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock OpenAI response {i+1} for casino content analysis."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": random.randint(50, 200),
                "completion_tokens": random.randint(100, 500),
                "total_tokens": random.randint(150, 700)
            }
        })
    
    # Generate Anthropic mock responses
    for i in range(20):
        responses["anthropic"].append({
            "id": f"msg_{uuid.uuid4().hex[:20]}",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": f"Mock Anthropic response {i+1} for gaming content analysis."
            }],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": random.randint(50, 200),
                "output_tokens": random.randint(100, 500)
            }
        })
    
    # Generate DataForSEO mock responses
    for i in range(10):
        responses["dataforseo"].append({
            "status_code": 20000,
            "status_message": "Ok.",
            "time": f"{random.uniform(0.1, 2.0):.3f} sec.",
            "cost": round(random.uniform(0.001, 0.01), 4),
            "tasks": [{
                "id": f"task_{uuid.uuid4().hex[:8]}",
                "status_code": 20000,
                "result": [{
                    "keyword": f"casino query {i+1}",
                    "items": [
                        {
                            "type": "images_element",
                            "title": f"Casino image {j+1}",
                            "url": f"https://example.com/casino_image_{i}_{j}.jpg",
                            "width": random.choice([800, 1200, 1600]),
                            "height": random.choice([600, 900, 1200])
                        }
                        for j in range(5)
                    ]
                }]
            }]
        })
    
    return responses


async def seed_test_database(client, table_counts: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Seed test database with generated data.
    
    Args:
        client: Database client (Supabase or mock)
        table_counts: Dictionary mapping table names to document counts
    
    Returns:
        Dictionary of seeded data by table
    """
    seeded_data = {}
    
    for table_name, count in table_counts.items():
        if table_name == "documents":
            # Generate document data
            config = TestDataConfig(count=count, generate_embeddings=True)
            with TestDataManager(config) as manager:
                generator = DocumentDataGenerator(manager)
                documents = generator.generate_documents(count)
                seeded_data[table_name] = documents
                
                # Insert into database if client supports it
                if hasattr(client, 'table') and client.table is not None:
                    try:
                        await client.table(table_name).insert(documents).execute()
                    except Exception as e:
                        print(f"Warning: Could not insert {table_name} data: {e}")
        
        elif table_name == "queries":
            # Generate query data
            queries = []
            for i in range(count):
                queries.append({
                    "id": f"query_{i:04d}",
                    "query_text": f"test query {i+1} about casino games",
                    "query_type": random.choice(list(TestDataCategory)).value,
                    "user_id": f"user_{random.randint(1, 100)}",
                    "created_at": datetime.now().isoformat(),
                    "metadata": {
                        "source": "test_suite",
                        "complexity": random.choice(list(TestDataComplexity)).value
                    }
                })
            seeded_data[table_name] = queries
            
            # Insert into database if client supports it
            if hasattr(client, 'table') and client.table is not None:
                try:
                    await client.table(table_name).insert(queries).execute()
                except Exception as e:
                    print(f"Warning: Could not insert {table_name} data: {e}")
        
        elif table_name == "responses":
            # Generate response data
            responses = []
            for i in range(count):
                responses.append({
                    "id": f"response_{i:04d}",
                    "query_id": f"query_{i:04d}",
                    "response_text": f"Mock response {i+1} for casino query",
                    "confidence_score": round(random.uniform(0.5, 0.95), 2),
                    "response_time_ms": random.randint(200, 2000),
                    "created_at": datetime.now().isoformat(),
                    "metadata": {
                        "model": random.choice(["gpt-4", "claude-3-sonnet"]),
                        "tokens_used": random.randint(100, 1000)
                    }
                })
            seeded_data[table_name] = responses
            
            # Insert into database if client supports it
            if hasattr(client, 'table') and client.table is not None:
                try:
                    await client.table(table_name).insert(responses).execute()
                except Exception as e:
                    print(f"Warning: Could not insert {table_name} data: {e}")
    
    return seeded_data 