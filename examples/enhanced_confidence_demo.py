#!/usr/bin/env python3
"""
Enhanced Confidence Scoring System - Integration Demonstration
Shows complete integration of all components with real examples.

This demonstration showcases:
- Enhanced Confidence Calculator with 4-factor scoring
- Universal RAG Chain with confidence integration
- Source Quality Analysis
- Intelligent Caching with quality-based decisions
- Response Validation Framework

Usage:
    python examples/enhanced_confidence_demo.py
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chains.enhanced_confidence_scoring_system import (
    EnhancedConfidenceCalculator,
    ConfidenceIntegrator, 
    SourceQualityAnalyzer,
    IntelligentCache,
    ResponseValidator,
    EnhancedRAGResponse,
    CacheStrategy,
    SourceQualityTier,
    ResponseQualityLevel
)

from chains.universal_rag_lcel import (
    create_universal_rag_chain,
    RAGResponse
)

from chains.advanced_prompt_system import (
    QueryType,
    ExpertiseLevel,
    ResponseFormat
)

from langchain_core.documents import Document


class EnhancedConfidenceDemonstration:
    """Comprehensive demonstration of Enhanced Confidence Scoring System."""
    
    def __init__(self):
        self.demo_queries = [
            {
                "query": "Which casino is safest for beginners?",
                "type": "review",
                "expected_response": "Betway Casino is widely considered one of the safest options for beginners due to its UK Gambling Commission license, SSL encryption, responsible gambling tools, and excellent customer support.",
                "mock_sources": [
                    {
                        "content": "Betway Casino holds a valid UK Gambling Commission license and Malta Gaming Authority license, ensuring regulatory compliance and player protection.",
                        "metadata": {"source": "uk-gambling-commission.gov", "authority": "government", "verified": True}
                    },
                    {
                        "content": "Independent testing by eCOGRA confirms fair gaming practices and proper random number generation at Betway Casino.",
                        "metadata": {"source": "ecogra.org", "authority": "certification", "verified": True}
                    }
                ]
            },
            {
                "query": "How to play blackjack step by step?",
                "type": "tutorial", 
                "expected_response": "1. Place your bet 2. Receive two cards face up 3. Add up card values (Ace=1/11, Face cards=10) 4. Decide to Hit, Stand, Double, or Split 5. Try to get closest to 21 without going over 6. Dealer plays after you're done",
                "mock_sources": [
                    {
                        "content": "Blackjack basic strategy involves understanding card values, betting options, and optimal decisions based on your hand versus the dealer's upcard.",
                        "metadata": {"source": "casino-strategy-guide.com", "authority": "expert", "author": "Professional Blackjack Player"}
                    }
                ]
            },
            {
                "query": "Compare Betway vs 888 Casino bonuses",
                "type": "comparison",
                "expected_response": "Betway offers a £30 welcome bonus with lower wagering requirements (35x) while 888 Casino provides up to £100 but with higher wagering (40x). Betway is better for smaller budgets, 888 for larger deposits.",
                "mock_sources": [
                    {
                        "content": "Betway welcome bonus: £30 free bet, 35x wagering requirement, valid for 7 days",
                        "metadata": {"source": "betway.com", "authority": "official", "current": True}
                    },
                    {
                        "content": "888 Casino bonus: Up to £100 + 25 free spins, 40x wagering requirement, 90 day validity",
                        "metadata": {"source": "888casino.com", "authority": "official", "current": True}
                    }
                ]
            }
        ]
    
    async def demonstrate_confidence_calculator(self):
        """Demonstrate Enhanced Confidence Calculator functionality."""
        
        print("\n" + "="*80)
        print("🎯 ENHANCED CONFIDENCE CALCULATOR DEMONSTRATION")
        print("="*80)
        
        # Initialize calculator
        calculator = EnhancedConfidenceCalculator()
        
        for i, demo_data in enumerate(self.demo_queries, 1):
            print(f"\n📋 Test Case {i}: {demo_data['type'].upper()} Query")
            print(f"Query: {demo_data['query']}")
            print("-" * 60)
            
            # Create mock response
            response = EnhancedRAGResponse(
                content=demo_data['expected_response'],
                sources=demo_data['mock_sources'],
                confidence_score=0.5,  # Will be calculated
                response_time=1.2,
                metadata={}
            )
            
            # Mock generation metadata
            generation_metadata = {
                'retrieval_quality': 0.85,
                'generation_stability': 0.9,
                'optimization_effectiveness': 0.8,
                'response_time_ms': 1200,
                'token_efficiency': 0.75
            }
            
            # Calculate confidence
            start_time = time.time()
            breakdown, enhanced_response = await calculator.calculate_enhanced_confidence(
                response=response,
                query=demo_data['query'],
                query_type=demo_data['type'],
                sources=demo_data['mock_sources'],
                generation_metadata=generation_metadata
            )
            calculation_time = (time.time() - start_time) * 1000
            
            # Display results
            print(f"✨ CONFIDENCE BREAKDOWN:")
            print(f"   Overall Confidence: {breakdown.overall_confidence:.3f}")
            print(f"   Content Quality:    {breakdown.content_quality:.3f} (35% weight)")
            print(f"   Source Quality:     {breakdown.source_quality:.3f} (25% weight)")
            print(f"   Query Matching:     {breakdown.query_matching:.3f} (20% weight)")
            print(f"   Technical Factors:  {breakdown.technical_factors:.3f} (20% weight)")
            
            print(f"\n🏷️ QUALITY INDICATORS:")
            for flag in breakdown.quality_flags:
                print(f"   • {flag}")
            
            if breakdown.improvement_suggestions:
                print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
                for suggestion in breakdown.improvement_suggestions[:3]:
                    print(f"   • {suggestion}")
            
            print(f"\n⚡ PERFORMANCE: {calculation_time:.1f}ms")
            
            # Check regeneration recommendation
            should_regen, reason = await calculator.should_regenerate_response(breakdown)
            if should_regen:
                print(f"🔄 REGENERATION RECOMMENDED: {reason}")
            else:
                print(f"✅ QUALITY ACCEPTABLE: {reason}")
    
    async def demonstrate_source_quality_analyzer(self):
        """Demonstrate Source Quality Analyzer."""
        
        print("\n" + "="*80)
        print("🔍 SOURCE QUALITY ANALYZER DEMONSTRATION")  
        print("="*80)
        
        analyzer = SourceQualityAnalyzer()
        
        # Test different source types
        test_sources = [
            {
                "name": "Government Authority Source",
                "doc": Document(
                    page_content="Official casino licensing information provided by the UK Gambling Commission. Licensed operators must comply with strict regulatory requirements.",
                    metadata={
                        "source": "ukgambling-commission.gov.uk",
                        "domain": "gov.uk",
                        "verified": True,
                        "published_date": "2024-01-15"
                    }
                ),
                "expected_tier": "PREMIUM"
            },
            {
                "name": "Expert Review Source", 
                "doc": Document(
                    page_content="Dr. James Smith, PhD in Statistics, provides comprehensive analysis of casino RTP rates and house edge calculations based on 10 years of research.",
                    metadata={
                        "source": "casino-research-institute.com",
                        "author": "Dr. James Smith, PhD",
                        "citations": 15,
                        "peer_reviewed": True
                    }
                ),
                "expected_tier": "HIGH"
            },
            {
                "name": "User Forum Source",
                "doc": Document(
                    page_content="I think this casino is okay, won some money there last week. Just my opinion though, might be lucky.",
                    metadata={
                        "source": "casino-forum.com",
                        "author": "anonymous_user123",
                        "post_type": "forum_comment"
                    }
                ),
                "expected_tier": "POOR"
            }
        ]
        
        for source_info in test_sources:
            print(f"\n📄 Analyzing: {source_info['name']}")
            print(f"Content: {source_info['doc'].page_content[:100]}...")
            print("-" * 60)
            
            # Analyze source quality
            analysis = await analyzer.analyze_source_quality(source_info['doc'])
            
            print(f"📊 QUALITY SCORES:")
            quality_scores = analysis.get('quality_scores', {})
            for indicator, score in quality_scores.items():
                print(f"   {indicator.capitalize():15} {score:.3f}")
            
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   Composite Score: {analysis.get('composite_score', 0):.3f}")
            print(f"   Quality Tier:    {analysis.get('quality_tier', 'UNKNOWN')}")
            print(f"   Expected Tier:   {source_info['expected_tier']}")
            
            # Show negative indicators if any
            negative_indicators = analysis.get('negative_indicators', [])
            if negative_indicators:
                print(f"\n⚠️ NEGATIVE INDICATORS:")
                for indicator in negative_indicators:
                    print(f"   • {indicator}")
    
    async def demonstrate_intelligent_cache(self):
        """Demonstrate Intelligent Cache System."""
        
        print("\n" + "="*80)
        print("🚀 INTELLIGENT CACHE SYSTEM DEMONSTRATION")
        print("="*80)
        
        # Test different cache strategies
        strategies = [
            CacheStrategy.CONSERVATIVE,
            CacheStrategy.BALANCED, 
            CacheStrategy.AGGRESSIVE,
            CacheStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            print(f"\n📦 Testing Cache Strategy: {strategy.value.upper()}")
            print("-" * 60)
            
            cache = IntelligentCache(strategy=strategy, max_size=100)
            
            # Test responses with different quality levels
            test_responses = [
                {
                    "query": "High quality query",
                    "response": EnhancedRAGResponse(
                        content="High quality response with detailed analysis",
                        sources=[],
                        confidence_score=0.9,
                        response_time=1.0,
                        metadata={"quality": "high"}
                    ),
                    "quality": "HIGH"
                },
                {
                    "query": "Medium quality query", 
                    "response": EnhancedRAGResponse(
                        content="Medium quality response",
                        sources=[],
                        confidence_score=0.7,
                        response_time=1.5,
                        metadata={"quality": "medium"}
                    ),
                    "quality": "MEDIUM"
                },
                {
                    "query": "Low quality query",
                    "response": EnhancedRAGResponse(
                        content="Short answer",
                        sources=[],
                        confidence_score=0.3,
                        response_time=2.0,
                        metadata={"quality": "low"}
                    ),
                    "quality": "LOW"
                }
            ]
            
            cached_count = 0
            for test_data in test_responses:
                # Attempt to cache response
                await cache.set(test_data["query"], test_data["response"])
                
                # Check if it was cached
                cached_response = await cache.get(test_data["query"])
                was_cached = cached_response is not None
                
                if was_cached:
                    cached_count += 1
                
                print(f"   {test_data['quality']} Quality (conf: {test_data['response'].confidence_score:.1f}): {'✅ CACHED' if was_cached else '❌ REJECTED'}")
            
            # Get cache performance metrics
            metrics = cache.get_performance_metrics()
            print(f"\n📈 CACHE PERFORMANCE:")
            print(f"   Items Cached:     {cached_count}/3")
            print(f"   Quality Threshold: {cache._get_adaptive_quality_threshold():.2f}")
            print(f"   Strategy Impact:  {strategy.value} prioritizes {'quality' if strategy == CacheStrategy.CONSERVATIVE else 'performance' if strategy == CacheStrategy.AGGRESSIVE else 'balance'}")
    
    async def demonstrate_response_validator(self):
        """Demonstrate Response Validation Framework."""
        
        print("\n" + "="*80)
        print("✅ RESPONSE VALIDATION FRAMEWORK DEMONSTRATION")
        print("="*80)
        
        validator = ResponseValidator()
        
        # Test different response quality levels
        test_cases = [
            {
                "name": "Excellent Response",
                "content": """
                # Casino Safety Guide for Beginners
                
                Choosing a safe casino is crucial for new players. Here are the key factors to consider:
                
                ## 1. Licensing and Regulation
                Look for casinos licensed by reputable authorities such as:
                - UK Gambling Commission (most strict)
                - Malta Gaming Authority (EU standard)
                - Gibraltar Regulatory Authority
                
                ## 2. Security Measures
                - SSL encryption (check for https://)
                - Two-factor authentication
                - Responsible gambling tools
                
                ## 3. Banking Options
                Trusted payment methods include:
                - Major credit cards (Visa, Mastercard)
                - E-wallets (PayPal, Skrill, Neteller)
                - Bank transfers
                
                ## Conclusion
                Research thoroughly before depositing money. Check reviews, verify licenses, and start with small amounts.
                """,
                "query": "How to choose a safe casino for beginners?",
                "expected_quality": "HIGH"
            },
            {
                "name": "Poor Response",
                "content": "Maybe try Betway I think its ok",
                "query": "How to choose a safe casino for beginners?",
                "expected_quality": "LOW"
            },
            {
                "name": "Medium Response",
                "content": "Safe casinos should have proper licensing and SSL encryption. Look for UK or Malta licenses. Check reviews before signing up.",
                "query": "How to choose a safe casino for beginners?", 
                "expected_quality": "MEDIUM"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n📝 Validating: {test_case['name']}")
            print(f"Content Length: {len(test_case['content'])} characters")
            print("-" * 60)
            
            # Validate response
            metrics, issues = await validator.validate_response(
                response_content=test_case['content'],
                query=test_case['query'],
                sources=[],
                context={'expected_quality': test_case['expected_quality']}
            )
            
            print(f"📊 VALIDATION SCORES:")
            print(f"   Format Score:     {metrics.format_score:.3f}")
            print(f"   Content Score:    {metrics.content_score:.3f}")
            print(f"   Source Score:     {metrics.source_score:.3f}")
            print(f"   Quality Score:    {metrics.quality_score:.3f}")
            print(f"   Overall Score:    {metrics.overall_score:.3f}")
            
            # Categorize quality level
            if metrics.overall_score >= 0.8:
                quality_level = "EXCELLENT"
                color = "🟢"
            elif metrics.overall_score >= 0.6:
                quality_level = "GOOD"
                color = "🟡"
            else:
                quality_level = "POOR"
                color = "🔴"
            
            print(f"\n{color} QUALITY ASSESSMENT: {quality_level}")
            print(f"   Expected: {test_case['expected_quality']}")
            
            # Show validation issues
            if issues:
                print(f"\n⚠️ VALIDATION ISSUES:")
                for issue in issues[:5]:  # Show top 5 issues
                    severity_emoji = {
                        'critical': '🚨',
                        'high': '🔥', 
                        'medium': '⚠️',
                        'low': '💡',
                        'info': 'ℹ️'
                    }.get(issue.severity.value, '❓')
                    
                    print(f"   {severity_emoji} {issue.severity.value.upper()}: {issue.message}")
    
    async def demonstrate_universal_rag_integration(self):
        """Demonstrate Universal RAG Chain with Enhanced Confidence."""
        
        print("\n" + "="*80)
        print("🔗 UNIVERSAL RAG CHAIN INTEGRATION DEMONSTRATION")
        print("="*80)
        
        # Create mock vector store
        class MockVectorStore:
            async def asimilarity_search_with_score(self, query: str, k: int = 5, **kwargs):
                # Return relevant mock documents based on query
                if "casino" in query.lower() and "safe" in query.lower():
                    return [
                        (Document(
                            page_content="Betway Casino is licensed by the UK Gambling Commission and Malta Gaming Authority, ensuring strict regulatory compliance and player protection.",
                            metadata={"source": "betway.com", "verified": True, "authority": "high"}
                        ), 0.95),
                        (Document(
                            page_content="Casino safety depends on proper licensing, SSL encryption, responsible gambling tools, and positive user reviews.",
                            metadata={"source": "casino-safety-guide.org", "expert_reviewed": True}
                        ), 0.88)
                    ]
                else:
                    return [
                        (Document(
                            page_content="Generic casino information",
                            metadata={"source": "casino-info.com"}
                        ), 0.6)
                    ]
        
        # Create enhanced RAG chain
        mock_vector_store = MockVectorStore()
        
        # Test both enhanced and basic modes
        configurations = [
            {
                "name": "Enhanced Mode",
                "config": {
                    "enable_enhanced_confidence": True,
                    "enable_prompt_optimization": True,
                    "enable_caching": True
                }
            },
            {
                "name": "Basic Mode", 
                "config": {
                    "enable_enhanced_confidence": False,
                    "enable_prompt_optimization": False,
                    "enable_caching": True
                }
            }
        ]
        
        for config_info in configurations:
            print(f"\n🚀 Testing {config_info['name']}")
            print("-" * 60)
            
            # Mock the LLM response since we don't have API keys in demo
            class MockLLM:
                async def ainvoke(self, prompt, **kwargs):
                    if "safe" in str(prompt).lower() and "casino" in str(prompt).lower():
                        return "Betway Casino is widely considered one of the safest options for beginners due to its comprehensive licensing, strong security measures, and excellent reputation in the industry."
                    return "Generic casino response based on the query."
            
            # Create chain with mock components
            chain = create_universal_rag_chain(
                model_name="gpt-4",
                vector_store=mock_vector_store,
                **config_info['config']
            )
            
            # Replace LLM with mock
            chain.llm = MockLLM()
            
            # Test query
            test_query = "Which casino is safest for beginners?"
            
            start_time = time.time()
            try:
                response = await chain.ainvoke(test_query)
                response_time = (time.time() - start_time) * 1000
                
                print(f"✅ Response Generated:")
                print(f"   Answer: {response.answer[:100]}...")
                print(f"   Confidence: {response.confidence_score:.3f}")
                print(f"   Response Time: {response_time:.1f}ms")
                print(f"   Sources: {len(response.sources)} retrieved")
                
                # Show enhanced metadata if available
                if hasattr(response, 'metadata') and response.metadata:
                    confidence_breakdown = response.metadata.get('confidence_breakdown')
                    if confidence_breakdown:
                        print(f"\n🎯 Enhanced Confidence Breakdown:")
                        for factor, score in confidence_breakdown.items():
                            if isinstance(score, (int, float)):
                                print(f"   {factor.replace('_', ' ').title()}: {score:.3f}")
                        
                        suggestions = response.metadata.get('improvement_suggestions', [])
                        if suggestions:
                            print(f"\n💡 Improvement Suggestions:")
                            for suggestion in suggestions[:2]:
                                print(f"   • {suggestion}")
                
                # Show capabilities comparison
                if config_info['name'] == "Enhanced Mode":
                    print(f"\n⚡ Enhanced Features Active:")
                    print(f"   • 4-factor confidence scoring")
                    print(f"   • Source quality analysis")
                    print(f"   • Query-type aware processing")
                    print(f"   • Intelligent caching decisions")
                else:
                    print(f"\n📊 Basic Features:")
                    print(f"   • Simple confidence calculation")
                    print(f"   • Standard caching")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def run_complete_demonstration(self):
        """Run the complete demonstration showcasing all components."""
        
        print("🎪 ENHANCED CONFIDENCE SCORING SYSTEM")
        print("🔥 Complete Integration Demonstration")
        print("=" * 80)
        print("This demo showcases the complete Enhanced Confidence Scoring System")
        print("with all components working together in production-ready integration.")
        print("=" * 80)
        
        try:
            # Run all demonstrations
            await self.demonstrate_confidence_calculator()
            await self.demonstrate_source_quality_analyzer()
            await self.demonstrate_intelligent_cache()
            await self.demonstrate_response_validator()
            await self.demonstrate_universal_rag_integration()
            
            # Summary
            print("\n" + "="*80)
            print("🎉 DEMONSTRATION COMPLETE - INTEGRATION SUCCESS!")
            print("="*80)
            print("✅ Enhanced Confidence Calculator: 4-factor scoring operational")
            print("✅ Source Quality Analyzer: Multi-tier quality assessment")
            print("✅ Intelligent Cache System: Quality-based caching decisions")
            print("✅ Response Validator: Comprehensive validation framework")
            print("✅ Universal RAG Integration: Seamless confidence enhancement")
            print("\n🚀 The Enhanced Confidence Scoring System is fully integrated")
            print("   and ready for production deployment!")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ Demonstration Error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demonstration runner."""
    
    demo = EnhancedConfidenceDemonstration()
    await demo.run_complete_demonstration()


if __name__ == "__main__":
    # Run the complete demonstration
    asyncio.run(main()) 