"""
LangSmith Prompt Testing Integration
Provides prompt testing and versioning capabilities using LangSmith
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client
    from langsmith.schemas import Example, Dataset
    LANGSMITH_PROMPT_TESTING_AVAILABLE = True
except ImportError:
    LANGSMITH_PROMPT_TESTING_AVAILABLE = False
    logger.warning("LangSmith prompt testing not available. Install with: pip install langsmith")


@dataclass
class PromptTestConfig:
    """Configuration for prompt testing"""
    prompt_name: str
    prompt_version: str
    test_queries: List[str]
    expected_behaviors: Dict[str, Any]
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class LangSmithPromptTester:
    """
    LangSmith prompt testing framework for prompt optimization
    
    Supports:
    - Creating prompt test suites
    - Versioning prompts
    - A/B testing prompt variations
    - Tracking performance changes
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LangSmith prompt tester
        
        Args:
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
        """
        if not LANGSMITH_PROMPT_TESTING_AVAILABLE:
            raise ImportError(
                "LangSmith prompt testing not available. Install with: pip install langsmith"
            )
        
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("LangSmith API key required. Set LANGSMITH_API_KEY environment variable.")
        
        self.client = Client(api_key=self.api_key)
    
    def create_prompt_test_suite(
        self,
        suite_name: str,
        prompt_configs: List[PromptTestConfig],
        description: Optional[str] = None
    ) -> str:
        """
        Create a prompt test suite with multiple prompt configurations
        
        Args:
            suite_name: Name of the test suite
            prompt_configs: List of prompt test configurations
            description: Optional suite description
            
        Returns:
            Suite identifier
        """
        try:
            # Create dataset for the test suite
            dataset_name = f"{suite_name}-prompt-tests"
            
            # Check if dataset exists
            existing_datasets = list(self.client.list_datasets(dataset_name=dataset_name))
            if existing_datasets:
                logger.info(f"Test suite '{suite_name}' already exists")
                return dataset_name
            
            # Create new dataset
            self.client.create_dataset(
                dataset_name=dataset_name,
                description=description or f"Prompt test suite: {suite_name}",
            )
            
            # Add test examples for each prompt config
            for config in prompt_configs:
                for query in config.test_queries:
                    self.client.create_example(
                        inputs={"query": query, "prompt_name": config.prompt_name},
                        outputs={
                            "expected_behaviors": config.expected_behaviors,
                            "prompt_version": config.prompt_version,
                        },
                        dataset_name=dataset_name,
                        metadata={
                            "prompt_name": config.prompt_name,
                            "prompt_version": config.prompt_version,
                            "tags": config.tags or [],
                            **(config.metadata or {}),
                        },
                    )
            
            logger.info(f"✅ Created prompt test suite '{suite_name}' with {len(prompt_configs)} prompt configs")
            return dataset_name
            
        except Exception as e:
            logger.error(f"Failed to create prompt test suite '{suite_name}': {e}")
            raise
    
    def test_prompt_variation(
        self,
        prompt_func: Callable,
        test_queries: List[str],
        prompt_name: str,
        prompt_version: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Test a prompt variation against test queries
        
        Args:
            prompt_func: Function that takes a query and returns a prompt
            test_queries: List of test queries
            prompt_name: Name of the prompt
            prompt_version: Version identifier
            tags: Optional tags for tracking
            
        Returns:
            Test results with performance metrics
        """
        try:
            results = []
            
            for query in test_queries:
                try:
                    # Generate prompt
                    prompt = prompt_func(query)
                    
                    # Record test run
                    run = self.client.create_run(
                        name=f"{prompt_name}-{prompt_version}",
                        run_type="chain",
                        inputs={"query": query},
                        outputs={"prompt": prompt},
                        tags=(tags or []) + [prompt_name, prompt_version],
                        metadata={
                            "prompt_name": prompt_name,
                            "prompt_version": prompt_version,
                        },
                    )
                    
                    results.append({
                        "query": query,
                        "prompt": prompt,
                        "run_id": str(run.id),
                        "success": True,
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to test prompt for query '{query}': {e}")
                    results.append({
                        "query": query,
                        "error": str(e),
                        "success": False,
                    })
            
            success_count = sum(1 for r in results if r.get("success"))
            
            return {
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "total_tests": len(test_queries),
                "successful_tests": success_count,
                "success_rate": success_count / len(test_queries) if test_queries else 0,
                "results": results,
            }
            
        except Exception as e:
            logger.error(f"Failed to test prompt variation: {e}")
            raise
    
    def compare_prompt_versions(
        self,
        prompt_funcs: Dict[str, Callable],
        test_queries: List[str],
        prompt_name: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple prompt versions side-by-side
        
        Args:
            prompt_funcs: Dictionary mapping version names to prompt functions
            test_queries: List of test queries
            prompt_name: Name of the prompt
            tags: Optional tags for tracking
            
        Returns:
            Comparison results with performance metrics for each version
        """
        try:
            comparison_results = {}
            
            for version, prompt_func in prompt_funcs.items():
                results = self.test_prompt_variation(
                    prompt_func=prompt_func,
                    test_queries=test_queries,
                    prompt_name=prompt_name,
                    prompt_version=version,
                    tags=(tags or []) + ["comparison"],
                )
                comparison_results[version] = results
            
            # Determine best version
            best_version = max(
                comparison_results.items(),
                key=lambda x: x[1]["success_rate"]
            )[0]
            
            return {
                "prompt_name": prompt_name,
                "comparison_results": comparison_results,
                "best_version": best_version,
                "total_queries": len(test_queries),
            }
            
        except Exception as e:
            logger.error(f"Failed to compare prompt versions: {e}")
            raise
    
    def track_prompt_performance(
        self,
        prompt_name: str,
        prompt_version: str,
        query: str,
        response: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Track prompt performance metrics for analysis
        
        Args:
            prompt_name: Name of the prompt
            prompt_version: Version identifier
            query: Input query
            response: Generated response
            metrics: Optional performance metrics
        """
        try:
            self.client.create_run(
                name=f"{prompt_name}-{prompt_version}",
                run_type="chain",
                inputs={"query": query},
                outputs={"response": response},
                tags=[prompt_name, prompt_version, "performance"],
                metadata={
                    "prompt_name": prompt_name,
                    "prompt_version": prompt_version,
                    **(metrics or {}),
                },
            )
            
        except Exception as e:
            logger.warning(f"Failed to track prompt performance: {e}")


def create_prompt_test_suite(
    suite_name: str,
    prompt_configs: List[PromptTestConfig],
    description: Optional[str] = None
) -> str:
    """
    Convenience function to create a prompt test suite
    
    Args:
        suite_name: Name of the test suite
        prompt_configs: List of prompt test configurations
        description: Optional suite description
        
    Returns:
        Suite identifier
    """
    tester = LangSmithPromptTester()
    return tester.create_prompt_test_suite(suite_name, prompt_configs, description)


