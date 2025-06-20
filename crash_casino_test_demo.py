#!/usr/bin/env python3
"""
Crash Casino Site Demo Test
Demonstrates Universal RAG CMS with your actual WordPress category structure
"""

import sys
import asyncio
from pathlib import Path
import os

# Add project root to path
sys.path.append('.')

async def test_crash_casino_rag_system():
    """Test the Universal RAG system with crash casino content mapping to your actual categories."""
    
    print("🎰 Crash Casino RAG System Demo")
    print("=" * 50)
    
    # Test 1: Crash Casino Review (maps to "Crash Casino Reviews")
    print("\n1. Testing Crash Casino Review...")
    try:
        from src.chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain with WordPress publishing enabled
        chain = create_universal_rag_chain(
            model_name="gpt-4o-mini",
            enable_wordpress_publishing=True,
            enable_dataforseo_images=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True
        )
        
        # Test query that should map to "Crash Casino Reviews" category
        test_query = "Review BC.Game crash casino - is it safe and trustworthy?"
        
        print(f"Query: {test_query}")
        
        # Test the content generation (without actual publishing)
        response = await chain.ainvoke({
            "question": test_query,
            "publish_to_wordpress": False  # Set to True when ready to publish
        })
        
        print(f"✅ Generated crash casino review: {len(response.answer)} characters")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Sources: {len(response.sources)}")
        
    except Exception as e:
        print(f"❌ Crash casino review test failed: {e}")
    
    # Test 2: Crash Game Strategy (maps to "Crash Strategies" -> "General Tactics")
    print("\n2. Testing Crash Game Strategy...")
    try:
        strategy_query = "Best crash game strategy for beginners using Bitcoin"
        
        print(f"Query: {strategy_query}")
        
        response = await chain.ainvoke({
            "question": strategy_query,
            "publish_to_wordpress": False
        })
        
        print(f"✅ Generated strategy guide: {len(response.answer)} characters")
        print(f"Confidence: {response.confidence_score:.2f}")
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
    
    # Test 3: Aviator Game Review (maps to "Crash Games" -> "Aviator")
    print("\n3. Testing Aviator Game Review...")
    try:
        aviator_query = "Complete Aviator crash game review and how to play guide"
        
        print(f"Query: {aviator_query}")
        
        response = await chain.ainvoke({
            "question": aviator_query,
            "publish_to_wordpress": False
        })
        
        print(f"✅ Generated Aviator review: {len(response.answer)} characters")
        print(f"Confidence: {response.confidence_score:.2f}")
        
    except Exception as e:
        print(f"❌ Aviator test failed: {e}")

def create_wordpress_category_mapping():
    """
    Create the WordPress category mapping for your crash casino site.
    This updates the Universal RAG Chain to use your actual category structure.
    """
    
    category_mapping = {
        # Main categories with actual IDs (you'll need to update these)
        "crash_casino_review": {
            "primary": "crash-casino-reviews",  # Crash Casino Reviews
            "parent": "casino-reviews",         # Casino Reviews (parent)
            "description": "Reviews of crash gambling casinos"
        },
        
        "individual_casino_review": {
            "primary": "individual-casino-reviews",  # Individual Casino Reviews
            "parent": "casino-reviews",
            "description": "Detailed individual casino reviews"
        },
        
        "crypto_casino_review": {
            "primary": "licensed-crypto-casinos",  # Licensed Crypto Casinos
            "parent": "casino-reviews", 
            "description": "Reviews of licensed cryptocurrency casinos"
        },
        
        "mobile_casino": {
            "primary": "mobile-casino",  # Mobile Casino
            "parent": "casino-reviews",
            "description": "Mobile-optimized casino reviews"
        },
        
        "new_casinos": {
            "primary": "new-casinos-2025",  # New Casinos 2025
            "parent": "casino-reviews",
            "description": "Latest casino launches for 2025"
        },
        
        "top_crash_casinos": {
            "primary": "top-crash-casinos",  # Top Crash Casinos
            "parent": "casino-reviews",
            "description": "Best crash gambling casinos ranked"
        },
        
        # Crash Games category
        "aviator_review": {
            "primary": "aviator",  # Aviator
            "parent": "crash-games",
            "description": "Aviator crash game reviews and guides"
        },
        
        "jetx_review": {
            "primary": "jetx",  # JetX
            "parent": "crash-games", 
            "description": "JetX crash game reviews and strategies"
        },
        
        "spaceman_review": {
            "primary": "spaceman",  # Spaceman
            "parent": "crash-games",
            "description": "Spaceman crash game analysis"
        },
        
        "best_crash_games": {
            "primary": "best-crash-games-2025",  # Best Crash Games 2025
            "parent": "crash-games",
            "description": "Top crash games for 2025"
        },
        
        "crash_game_reviews": {
            "primary": "crash-game-reviews",  # Crash Game Reviews
            "parent": "crash-games",
            "description": "Individual crash game reviews"
        },
        
        # Strategy categories
        "general_strategy": {
            "primary": "general-tactics",  # General Tactics
            "parent": "strategy",
            "description": "General crash gambling strategies"
        },
        
        "multiplier_strategy": {
            "primary": "multiplier-tactics",  # Multiplier Tactics
            "parent": "strategy", 
            "description": "Multiplier-focused crash strategies"
        },
        
        "crypto_strategy": {
            "primary": "coin-specific",  # Coin-Specific (BTC, ETH, SOL)
            "parent": "strategy",
            "description": "Cryptocurrency-specific strategies"
        },
        
        # Other categories
        "site_specific": {
            "primary": "crashcasino-io",  # CrashCasino.io
            "parent": None,
            "description": "CrashCasino.io specific content"
        },
        
        "guides": {
            "primary": "casino-guides",  # Guides
            "parent": None,
            "description": "Casino and crash game guides"
        },
        
        "general": {
            "primary": "general",  # General
            "parent": None,
            "description": "General casino content"
        }
    }
    
    return category_mapping

def show_query_to_category_mapping():
    """Show how different queries map to your WordPress categories."""
    
    print("\n📂 Query to WordPress Category Mapping")
    print("=" * 50)
    
    mappings = [
        {
            "query": "BC.Game crash casino review",
            "category": "crash-casino-reviews", 
            "parent": "casino-reviews",
            "reason": "Crash casino review content"
        },
        {
            "query": "Best Aviator strategy for beginners",
            "category": "aviator",
            "parent": "crash-games", 
            "reason": "Aviator-specific content"
        },
        {
            "query": "Bitcoin crash gambling tactics",
            "category": "coin-specific",
            "parent": "strategy",
            "reason": "Cryptocurrency-specific strategy"
        },
        {
            "query": "Top 10 crash casinos 2025",
            "category": "top-crash-casinos",
            "parent": "casino-reviews",
            "reason": "Ranking/comparison content"
        },
        {
            "query": "JetX game review and tips",
            "category": "jetx", 
            "parent": "crash-games",
            "reason": "JetX-specific content"
        },
        {
            "query": "How to use multiplier strategy in crash games",
            "category": "multiplier-tactics",
            "parent": "strategy", 
            "reason": "Multiplier strategy content"
        },
        {
            "query": "Mobile crash casino apps review",
            "category": "mobile-casino",
            "parent": "casino-reviews",
            "reason": "Mobile-focused casino content"
        }
    ]
    
    for i, mapping in enumerate(mappings, 1):
        print(f"\n{i}. Query: '{mapping['query']}'")
        print(f"   → Category: {mapping['category']}")
        print(f"   → Parent: {mapping['parent']}")
        print(f"   → Reason: {mapping['reason']}")

def create_content_templates():
    """Create content templates optimized for crash casino content."""
    
    templates = {
        "crash_casino_review": """
# {casino_name} Crash Casino Review 2025: Complete Analysis

## Executive Summary
- **Overall Rating**: {rating}/10
- **License**: {license_info}
- **Crash Games**: {crash_games_count}+ games
- **Crypto Support**: {crypto_currencies}
- **Min Deposit**: {min_deposit}

## About {casino_name}
{casino_description}

## Crash Games Selection
### Available Crash Games:
{crash_games_list}

### Game Providers:
{game_providers}

## Bonuses & Promotions
### Welcome Bonus
- **Amount**: {welcome_bonus}
- **Wagering**: {wagering_requirements}
- **Crash Games Eligible**: {crash_bonus_eligible}

## Payment Methods
### Cryptocurrency Support:
{crypto_payment_methods}

### Traditional Methods:
{traditional_payment_methods}

## Safety & Licensing
- **License**: {license_details}
- **SSL Encryption**: ✅
- **Fair Gaming**: {provably_fair}

## Mobile Experience
{mobile_review}

## Pros & Cons
### ✅ Pros:
{pros_list}

### ❌ Cons: 
{cons_list}

## Final Verdict
{final_verdict}

**Rating: {final_rating}/10**
        """,
        
        "crash_strategy_guide": """
# {strategy_title}: Complete Guide for 2025

## Strategy Overview
{strategy_overview}

## How It Works
{strategy_mechanics}

## Step-by-Step Implementation
{step_by_step_guide}

## Risk Management
{risk_management_tips}

## Recommended Casinos
{recommended_casinos}

## Advanced Tips
{advanced_tips}

## Common Mistakes to Avoid
{common_mistakes}

## Expected Results
{expected_results}
        """,
        
        "crash_game_review": """
# {game_name} Review: Complete 2025 Analysis

## Game Overview
- **Provider**: {game_provider}
- **RTP**: {rtp_percentage}%
- **Max Multiplier**: {max_multiplier}x
- **Min Bet**: {min_bet}
- **Max Bet**: {max_bet}

## How to Play {game_name}
{gameplay_guide}

## Key Features
{key_features}

## Strategies & Tips
{strategy_tips}

## Where to Play
{available_casinos}

## Mobile Compatibility
{mobile_compatibility}

## Our Verdict
{final_assessment}

**Rating: {game_rating}/10**
        """
    }
    
    return templates

async def main():
    """Main demo function."""
    
    print("🎰 CRASH CASINO RAG SYSTEM DEMO")
    print("Testing with your actual WordPress category structure")
    print("=" * 70)
    
    # Show category mapping
    show_query_to_category_mapping()
    
    # Test the RAG system
    await test_crash_casino_rag_system()
    
    # Show available templates
    print("\n📝 Available Content Templates:")
    templates = create_content_templates()
    for template_name in templates.keys():
        print(f"  • {template_name}")
    
    print("\n✅ Demo completed! Your Universal RAG system is ready for crash casino content.")
    print("\n🚀 Next steps:")
    print("1. Update WordPress credentials in environment variables")
    print("2. Set publish_to_wordpress=True to enable actual publishing")
    print("3. Customize category mappings as needed")
    print("4. Test with your specific crash casino queries")

if __name__ == "__main__":
    asyncio.run(main()) 