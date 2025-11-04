"""
LangSmith Evaluation Framework for Universal RAG CMS
Provides systematic evaluation using LangSmith evaluation tools
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import LangSmith evaluation components
try:
    from langsmith import Client, RunEvaluator
    from langchain.evaluation import load_evaluator
    from langchain.evaluation.schema import EvaluatorType
    from langchain_core.language_models import BaseLanguageModel
    LANGSMITH_EVAL_AVAILABLE = True
except ImportError:
    LANGSMITH_EVAL_AVAILABLE = False
    logger.warning("LangSmith evaluation not available. Install with: pip install langsmith langchain")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    dataset_name: str
    evaluators: List[str]  # List of evaluator types: "qa", "embedding_distance", "criteria", etc.
    llm: Optional[BaseLanguageModel] = None
    max_examples: Optional[int] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class LangSmithEvaluator:
    """
    LangSmith evaluation framework for systematic RAG chain evaluation
    
    Supports multiple evaluator types:
    - QA: Question answering correctness
    - Embedding Distance: Semantic similarity
    - Criteria: Custom criteria evaluation
    - Cosine Similarity: Vector similarity
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LangSmith evaluator
        
        Args:
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
        """
        if not LANGSMITH_EVAL_AVAILABLE:
            raise ImportError(
                "LangSmith evaluation not available. Install with: pip install langsmith langchain"
            )
        
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        if not self.api_key:
            raise ValueError("LangSmith API key required. Set LANGSMITH_API_KEY environment variable.")
        
        self.client = Client(api_key=self.api_key)
        self.evaluators = {}
    
    def load_evaluator(self, evaluator_type: str, llm: Optional[BaseLanguageModel] = None) -> RunEvaluator:
        """
        Load a LangSmith evaluator
        
        Args:
            evaluator_type: Type of evaluator ("qa", "embedding_distance", "criteria", etc.)
            llm: Optional LLM for evaluators that need it
            
        Returns:
            Configured evaluator
        """
        if evaluator_type in self.evaluators:
            return self.evaluators[evaluator_type]
        
        try:
            if evaluator_type == "qa":
                evaluator = load_evaluator("qa", llm=llm)
            elif evaluator_type == "embedding_distance":
                evaluator = load_evaluator("embedding_distance")
            elif evaluator_type == "cosine_similarity":
                evaluator = load_evaluator("embedding_distance", distance_metric="cosine")
            elif evaluator_type == "criteria":
                if not llm:
                    raise ValueError("LLM required for criteria evaluator")
                evaluator = load_evaluator("criteria", llm=llm)
            else:
                raise ValueError(f"Unknown evaluator type: {evaluator_type}")
            
            self.evaluators[evaluator_type] = evaluator
            logger.info(f"✅ Loaded evaluator: {evaluator_type}")
            return evaluator
            
        except Exception as e:
            logger.error(f"Failed to load evaluator {evaluator_type}: {e}")
            raise
    
    def create_dataset(
        self,
        dataset_name: str,
        examples: List[Dict[str, Any]],
        description: Optional[str] = None
    ) -> str:
        """
        Create a LangSmith dataset
        
        Args:
            dataset_name: Name of the dataset
            examples: List of examples with "inputs" and "outputs" keys
            description: Optional dataset description
            
        Returns:
            Dataset name/ID
        """
        try:
            # Check if dataset already exists
            existing_datasets = list(self.client.list_datasets(dataset_name=dataset_name))
            if existing_datasets:
                logger.info(f"Dataset '{dataset_name}' already exists, using existing dataset")
                return dataset_name
            
            # Create new dataset
            self.client.create_dataset(
                dataset_name=dataset_name,
                description=description or f"Evaluation dataset for {dataset_name}",
            )
            
            # Add examples
            for example in examples:
                self.client.create_example(
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                    dataset_name=dataset_name,
                )
            
            logger.info(f"✅ Created dataset '{dataset_name}' with {len(examples)} examples")
            return dataset_name
            
        except Exception as e:
            logger.error(f"Failed to create dataset '{dataset_name}': {e}")
            raise
    
    def evaluate_chain(
        self,
        chain: Callable,
        dataset_name: str,
        evaluators: List[str],
        llm: Optional[BaseLanguageModel] = None,
        max_examples: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a chain on a LangSmith dataset
        
        Args:
            chain: Chain/function to evaluate
            dataset_name: Name of the dataset
            evaluators: List of evaluator types to use
            llm: Optional LLM for evaluators
            max_examples: Maximum number of examples to evaluate
            tags: Optional tags for the evaluation run
            metadata: Optional metadata for the evaluation run
            
        Returns:
            Evaluation results
        """
        try:
            from langchain.smith import RunEvalConfig, run_on_dataset
            
            # Load evaluators
            eval_config = RunEvalConfig(
                evaluators=[self.load_evaluator(eval_type, llm=llm) for eval_type in evaluators],
                custom_evaluators=[],
            )
            
            # Run evaluation
            results = run_on_dataset(
                client=self.client,
                dataset_name=dataset_name,
                llm_or_chain_factory=lambda: chain,
                evaluation=eval_config,
                max_examples=max_examples,
                tags=tags or [],
                metadata=metadata or {},
            )
            
            logger.info(f"✅ Evaluation completed for dataset '{dataset_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate chain on dataset '{dataset_name}': {e}")
            raise
    
    def get_evaluation_results(
        self,
        dataset_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation results for a dataset
        
        Args:
            dataset_name: Name of the dataset
            limit: Maximum number of results to return
            
        Returns:
            List of evaluation results
        """
        try:
            # Get runs for the dataset
            runs = list(self.client.list_runs(
                dataset_name=dataset_name,
                limit=limit
            ))
            
            results = []
            for run in runs:
                results.append({
                    "run_id": str(run.id),
                    "inputs": run.inputs,
                    "outputs": run.outputs,
                    "error": run.error,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "latency_ms": run.total_time.total_seconds() * 1000 if run.total_time else None,
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get evaluation results for '{dataset_name}': {e}")
            raise


def create_evaluation_suite(
    dataset_name: str,
    examples: List[Dict[str, Any]],
    evaluators: List[str],
    llm: Optional[BaseLanguageModel] = None,
    description: Optional[str] = None
) -> LangSmithEvaluator:
    """
    Create a complete evaluation suite with dataset and evaluators
    
    Args:
        dataset_name: Name of the dataset
        examples: List of examples with "inputs" and "outputs"
        evaluators: List of evaluator types
        llm: Optional LLM for evaluators
        description: Optional dataset description
        
    Returns:
        Configured LangSmithEvaluator instance
    """
    evaluator = LangSmithEvaluator()
    
    # Create dataset
    evaluator.create_dataset(dataset_name, examples, description)
    
    # Pre-load evaluators
    for eval_type in evaluators:
        evaluator.load_evaluator(eval_type, llm)
    
    return evaluator


def evaluate_chain_on_dataset(
    chain: Callable,
    config: EvaluationConfig,
    llm: Optional[BaseLanguageModel] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a chain on a dataset
    
    Args:
        chain: Chain/function to evaluate
        config: Evaluation configuration
        llm: Optional LLM for evaluators
        
    Returns:
        Evaluation results
    """
    evaluator = LangSmithEvaluator()
    
    return evaluator.evaluate_chain(
        chain=chain,
        dataset_name=config.dataset_name,
        evaluators=config.evaluators,
        llm=llm or config.llm,
        max_examples=config.max_examples,
        tags=config.tags,
        metadata=config.metadata,
    )


def run_evaluation(
    chain: Callable,
    dataset_name: str,
    evaluator_types: List[str] = ["qa", "embedding_distance"],
    llm: Optional[BaseLanguageModel] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a quick evaluation on a chain
    
    Args:
        chain: Chain/function to evaluate
        dataset_name: Name of the dataset
        evaluator_types: List of evaluator types to use
        llm: Optional LLM for evaluators
        **kwargs: Additional arguments passed to evaluate_chain
        
    Returns:
        Evaluation results
    """
    evaluator = LangSmithEvaluator()
    
    return evaluator.evaluate_chain(
        chain=chain,
        dataset_name=dataset_name,
        evaluators=evaluator_types,
        llm=llm,
        **kwargs
    )


