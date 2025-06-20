#!/usr/bin/env python3
"""
Quick Demo Test for Universal RAG CMS
Demonstrates core functionality without requiring full database setup.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append('.')

async def test_universal_rag_components():
    """Test key Universal RAG CMS components."""
    
    print("🧪 Universal RAG CMS Quick Test Demo")
    print("=" * 50)
    
    # Test 1: Query Classification
    print("\n1. Testing Query Classification...")
    try:
        from src.chains.advanced_prompt_system import QueryClassifier
        classifier = QueryClassifier()
        
        test_queries = [
            "Tell me about Betway Casino bonuses",
            "How to install Python?",
            "What are the latest news in crypto?",
            "Review of iPhone 15",
        ]
        
        for query in test_queries:
            query_type = classifier.classify_query(query)
            print(f"   ✅ '{query[:30]}...' → {query_type}")
            
    except Exception as e:
        print(f"   ❌ Query Classification failed: {e}")
    
    # Test 2: Enhanced Confidence System
    print("\n2. Testing Enhanced Confidence System...")
    try:
        from src.chains.enhanced_confidence_scoring_system import EnhancedConfidenceCalculator
        confidence_calc = EnhancedConfidenceCalculator()
        
        # Create a mock response for testing
        mock_response = {
            'content': 'This is a detailed casino review with comprehensive information.',
            'sources': [
                {'url': 'https://example.com', 'authority_score': 0.9},
                {'url': 'https://another.com', 'authority_score': 0.7}
            ],
            'query_match_score': 0.85
        }
        
        confidence_score = confidence_calc.calculate_base_confidence(
            response_content=mock_response['content'],
            sources=mock_response['sources'],
            query_match_score=mock_response['query_match_score']
        )
        
        print(f"   ✅ Base confidence score: {confidence_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Enhanced Confidence System failed: {e}")
    
    # Test 3: Template System
    print("\n3. Testing Template System...")
    try:
        from src.templates.improved_template_manager import ImprovedTemplateManager
        template_mgr = ImprovedTemplateManager()
        
        # Test template selection
        casino_template = template_mgr.get_optimized_template(
            query_type="casino_review",
            expertise_level="intermediate"
        )
        
        print(f"   ✅ Casino review template loaded: {len(casino_template)} characters")
        
    except Exception as e:
        print(f"   ❌ Template System failed: {e}")
    
    # Test 4: Configuration System
    print("\n4. Testing Configuration System...")
    try:
        from src.config.prompt_config import PromptOptimizationConfig, QueryType
        
        config = PromptOptimizationConfig()
        cache_ttl = config.cache_config.get_ttl(QueryType.CASINO_REVIEW)
        
        print(f"   ✅ Configuration loaded: Cache TTL for casino reviews = {cache_ttl} hours")
        
    except Exception as e:
        print(f"   ❌ Configuration System failed: {e}")
    
    # Test 5: Universal RAG Chain Creation
    print("\n5. Testing Universal RAG Chain Creation...")
    try:
        from src.chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain without database dependencies (mock mode)
        rag_chain = create_universal_rag_chain(
            enable_supabase=False,  # Disable database for quick test
            enable_vector_store=False
        )
        
        print(f"   ✅ Universal RAG Chain created successfully!")
        print(f"   📊 Chain type: {type(rag_chain).__name__}")
        
    except Exception as e:
        print(f"   ❌ Universal RAG Chain creation failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick test completed!")
    print("💡 For full testing, run: python tests/run_tests.py --type unit")

if __name__ == "__main__":
    asyncio.run(test_universal_rag_components()) 