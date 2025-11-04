"""
Evaluation Configuration and Dataset Definitions
Pre-configured evaluation datasets for casino reviews, affiliate content, and general queries
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EvaluationDataset:
    """Evaluation dataset configuration"""
    name: str
    description: str
    examples: List[Dict[str, Any]]
    evaluators: List[str] = field(default_factory=lambda: ["qa", "embedding_distance"])
    tags: List[str] = field(default_factory=list)


# Casino Review Evaluation Dataset
CASINO_REVIEW_DATASET = EvaluationDataset(
    name="casino-review-evaluation",
    description="Evaluation dataset for casino review content generation",
    evaluators=["qa", "embedding_distance", "criteria"],
    tags=["casino", "review", "content-generation"],
    examples=[
        {
            "inputs": {
                "query": "Betway Casino Review 2025"
            },
            "outputs": {
                "expected_topics": [
                    "licensing and regulation",
                    "game selection",
                    "bonus offers",
                    "payment methods",
                    "user experience"
                ],
                "min_length": 1000,
                "should_include_ratings": True
            }
        },
        {
            "inputs": {
                "query": "TrustDice Casino comprehensive analysis"
            },
            "outputs": {
                "expected_topics": [
                    "crypto payments",
                    "provably fair gaming",
                    "license information",
                    "customer support"
                ],
                "min_length": 1500,
                "should_include_ratings": True
            }
        },
        {
            "inputs": {
                "query": "Compare Napoleon Sports and Betway casinos"
            },
            "outputs": {
                "expected_topics": [
                    "comparison table",
                    "pros and cons",
                    "recommendations"
                ],
                "min_length": 2000,
                "should_include_ratings": True
            }
        },
    ]
)

# Affiliate Content Evaluation Dataset
AFFILIATE_CONTENT_DATASET = EvaluationDataset(
    name="affiliate-content-evaluation",
    description="Evaluation dataset for affiliate link integration and content",
    evaluators=["qa", "criteria"],
    tags=["affiliate", "content", "links"],
    examples=[
        {
            "inputs": {
                "query": "Best crypto casino bonuses with affiliate links"
            },
            "outputs": {
                "expected_features": [
                    "affiliate links embedded",
                    "bonus information",
                    "crypto payment methods"
                ],
                "min_affiliate_links": 2,
                "max_affiliate_links": 5
            }
        },
        {
            "inputs": {
                "query": "Top 5 online casinos for slots"
            },
            "outputs": {
                "expected_features": [
                    "numbered list",
                    "affiliate links",
                    "slot game focus"
                ],
                "min_affiliate_links": 3,
                "max_affiliate_links": 5
            }
        },
    ]
)

# General Query Evaluation Dataset
GENERAL_QUERY_DATASET = EvaluationDataset(
    name="general-query-evaluation",
    description="Evaluation dataset for general RAG queries",
    evaluators=["qa", "embedding_distance"],
    tags=["general", "rag", "qa"],
    examples=[
        {
            "inputs": {
                "query": "What are the latest trends in online gambling?"
            },
            "outputs": {
                "expected_topics": [
                    "trends",
                    "technology",
                    "regulations"
                ],
                "min_length": 500
            }
        },
        {
            "inputs": {
                "query": "How do provably fair games work?"
            },
            "outputs": {
                "expected_topics": [
                    "blockchain",
                    "verification",
                    "transparency"
                ],
                "min_length": 800
            }
        },
        {
            "inputs": {
                "query": "What is responsible gambling?"
            },
            "outputs": {
                "expected_topics": [
                    "self-exclusion",
                    "limits",
                    "resources"
                ],
                "min_length": 600
            }
        },
    ]
)


def get_all_datasets() -> List[EvaluationDataset]:
    """Get all evaluation datasets"""
    return [
        CASINO_REVIEW_DATASET,
        AFFILIATE_CONTENT_DATASET,
        GENERAL_QUERY_DATASET,
    ]


def get_dataset_by_name(name: str) -> EvaluationDataset:
    """Get a specific dataset by name"""
    datasets = {
        "casino-review": CASINO_REVIEW_DATASET,
        "affiliate-content": AFFILIATE_CONTENT_DATASET,
        "general-query": GENERAL_QUERY_DATASET,
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name]


