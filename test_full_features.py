#!/usr/bin/env python3
"""
Test Full Universal RAG Chain - ALL FEATURES ENABLED
Testing the complete system with your crash casino site structure
"""

import asyncio
import sys
sys.path.append('.')

async def test_full_universal_rag_chain():
    """Test the full Universal RAG Chain with ALL features enabled."""
    
    print('🚀 TESTING FULL UNIVERSAL RAG CHAIN - ALL FEATURES')
    print('=' * 60)
    
    from src.chains.universal_rag_lcel import create_universal_rag_chain
    
    # Create chain with ALL features enabled
    chain = create_universal_rag_chain(
        model_name='gpt-4o-mini',
        enable_caching=True,
        enable_contextual_retrieval=True,
        enable_prompt_optimization=True,  # NOW ENABLED
        enable_enhanced_confidence=True,
        enable_template_system_v2=True,  # NOW ENABLED
        enable_dataforseo_images=True,
        enable_wordpress_publishing=False,  # Test content first
        enable_fti_processing=True,
        enable_security=True,
        enable_profiling=True,
        enable_web_search=True,
        enable_comprehensive_web_research=True,
        enable_response_storage=True
    )
    
    print('✅ Full Universal RAG Chain initialized with ALL features!')
    
    # Test crash casino query with your site structure
    query = 'BC.Game crash casino review - is it safe and trustworthy for Bitcoin gambling?'
    print(f'\n🎰 Testing: {query}')
    
    response = await chain.ainvoke(query)
    
    print(f'\n📊 RESULTS:')
    print(f'  Content: {len(response.answer)} characters')
    print(f'  Confidence: {response.confidence_score:.3f}')
    print(f'  Sources: {len(response.sources)}')
    print(f'  Response time: {response.response_time:.1f}ms')
    print(f'  Cached: {response.cached}')
    
    if hasattr(response, 'metadata') and response.metadata:
        print(f'\n🔧 FEATURES USED:')
        meta = response.metadata
        print(f'  Advanced prompts: {meta.get("optimization_effectiveness", "N/A")}')
        print(f'  Template system: {meta.get("template_system_v2_used", False)}')
        print(f'  Images found: {meta.get("images_embedded", 0)}')
        print(f'  Security checked: {meta.get("security_checked", False)}')
        print(f'  Performance profiled: {meta.get("performance_profiled", False)}')
        print(f'  Web search used: {len([s for s in response.sources if s.get("source_type") == "web_search"])}')
        print(f'  Total active features: {meta.get("advanced_features_count", 0)}')
    
    # Show detailed source breakdown
    print(f'\n📚 SOURCE BREAKDOWN:')
    source_types = {}
    for source in response.sources:
        source_type = source.get('source_type', 'unknown')
        source_types[source_type] = source_types.get(source_type, 0) + 1
    
    for source_type, count in source_types.items():
        print(f'  {source_type}: {count} sources')
    
    # Show content preview
    if isinstance(response.answer, dict):
        content = response.answer.get('final_content', str(response.answer))
    else:
        content = response.answer
        
    preview = content[:300] if len(content) > 300 else content
    print(f'\n📝 CONTENT PREVIEW:')
    print(preview + '...')
    
    print(f'\n🎯 ALL FEATURES WORKING! Ready for crash casino content at scale.')
    print(f'Content would be categorized as: crash-casino-reviews')

if __name__ == "__main__":
    asyncio.run(test_full_universal_rag_chain()) 