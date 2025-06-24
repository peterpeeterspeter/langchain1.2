#!/usr/bin/env python3
"""
🎰 COINFLIP AUTO-CATEGORIZATION EXAMPLE
====================================

Shows how casino content is automatically categorized as MT Casinos
with complete metadata mapping from 95-field intelligence

Author: AI Assistant
Created: 2025-01-21
"""

import json
from typing import Dict, Any

# Example: Automatic categorization process
def demonstrate_auto_categorization():
    """
    Demonstrates automatic categorization of casino content as MT Casinos
    """
    
    print("🎯 AUTOMATIC MT CASINOS CATEGORIZATION")
    print("="*50)
    
    # Step 1: Input - 95-field casino intelligence (from Universal RAG Chain)
    sample_intelligence = {
        "casino_name": "Betway Casino",
        "casino_url": "https://betway.com",
        "overall_rating": 4.2,
        "license_info": "UK Gambling Commission, Malta Gaming Authority",
        "total_games": 850,
        "slot_games": 600,
        "table_games": 150,
        "live_games": 100,
        "welcome_bonus": "$1000 + 200 Free Spins",
        "bonus_percentage": 100,
        "wagering_requirements": "35x",
        "deposit_methods": ["Visa", "Mastercard", "PayPal", "Skrill", "Neteller"],
        "withdrawal_methods": ["Bank Transfer", "PayPal", "Skrill"],
        "min_deposit": "$10",
        "withdrawal_time": "24-48 hours",
        "game_providers": ["NetEnt", "Microgaming", "Evolution Gaming"],
        "support_languages": ["English", "German", "Spanish"],
        "mobile_compatibility": "Full mobile site + iOS/Android apps"
    }
    
    # Step 2: Automatic MT Casinos metadata mapping
    mt_casinos_metadata = auto_map_to_mt_casinos(sample_intelligence)
    
    # Step 3: Display results
    display_categorization_results(sample_intelligence, mt_casinos_metadata)
    
    return mt_casinos_metadata

def auto_map_to_mt_casinos(intelligence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically maps 95-field intelligence to MT Casinos metadata
    """
    
    # Automatic mapping logic
    metadata = {
        # Core MT Casinos fields
        'post_type': 'mt_casinos',  # AUTOMATICALLY ASSIGNED
        'meta_fields': {
            # Basic Info (6 fields)
            '_casino_name': intelligence.get('casino_name', ''),
            '_casino_url': intelligence.get('casino_url', ''),
            '_casino_affiliate_url': f"{intelligence.get('casino_url', '')}?ref=affiliate",
            '_casino_rating': float(intelligence.get('overall_rating', 0)),
            '_casino_logo': f"logo-{intelligence.get('casino_name', '').lower().replace(' ', '-')}.png",
            '_casino_banner': f"banner-{intelligence.get('casino_name', '').lower().replace(' ', '-')}.jpg",
            
            # License & Security (4 fields)
            '_casino_license_info': intelligence.get('license_info', ''),
            '_casino_license_jurisdictions': json.dumps(intelligence.get('license_info', '').split(', ')),
            '_casino_safety_rating': calculate_safety_rating(intelligence),
            '_casino_security_features': json.dumps(['SSL Encryption', '2FA Available', 'RNG Certified']),
            
            # Games (5 fields) 
            '_casino_total_games': int(intelligence.get('total_games', 0)),
            '_casino_slot_games': int(intelligence.get('slot_games', 0)),
            '_casino_table_games': int(intelligence.get('table_games', 0)),
            '_casino_live_games': int(intelligence.get('live_games', 0)),
            '_casino_game_providers': json.dumps(intelligence.get('game_providers', [])),
            
            # Bonuses (5 fields)
            '_casino_welcome_bonus': intelligence.get('welcome_bonus', ''),
            '_casino_bonus_amount': extract_bonus_amount(intelligence.get('welcome_bonus', '')),
            '_casino_bonus_percentage': int(intelligence.get('bonus_percentage', 0)),
            '_casino_wagering_requirements': intelligence.get('wagering_requirements', ''),
            '_casino_bonus_codes': json.dumps(['WELCOME100', 'FREESPINS']),
            
            # Payments (5 fields)
            '_casino_deposit_methods': json.dumps(intelligence.get('deposit_methods', [])),
            '_casino_withdrawal_methods': json.dumps(intelligence.get('withdrawal_methods', [])),
            '_casino_min_deposit': intelligence.get('min_deposit', ''),
            '_casino_max_withdrawal': '$50,000/month',
            '_casino_withdrawal_time': intelligence.get('withdrawal_time', ''),
            
            # Support & UX (4 fields)
            '_casino_support_languages': json.dumps(intelligence.get('support_languages', [])),
            '_casino_support_methods': json.dumps(['Live Chat', 'Email', 'Phone']),
            '_casino_mobile_compatibility': intelligence.get('mobile_compatibility', ''),
            '_casino_app_available': 'Yes' if 'app' in intelligence.get('mobile_compatibility', '').lower() else 'No',
            
            # Coinflip Review Criteria (6 fields) - AUTO-CALCULATED
            '_casino_criteria_games': calculate_games_score(intelligence),
            '_casino_criteria_bonuses': calculate_bonuses_score(intelligence),
            '_casino_criteria_payments': calculate_payments_score(intelligence),
            '_casino_criteria_support': calculate_support_score(intelligence),
            '_casino_criteria_mobile': calculate_mobile_score(intelligence),
            '_casino_criteria_security': calculate_security_score(intelligence),
        },
        
        # WordPress taxonomy (AUTO-ASSIGNED)
        'categories': auto_assign_categories(intelligence),
        'tags': auto_generate_tags(intelligence),
        
        # Coinflip shortcodes (AUTO-GENERATED)
        'shortcodes': auto_generate_shortcodes(intelligence)
    }
    
    return metadata

def calculate_safety_rating(intelligence: Dict[str, Any]) -> float:
    """Calculate safety rating from intelligence data"""
    base_score = 3.0
    
    # Boost for known licenses
    license_info = intelligence.get('license_info', '').lower()
    if 'uk gambling commission' in license_info:
        base_score += 1.0
    if 'malta gaming authority' in license_info:
        base_score += 0.5
    
    return min(5.0, base_score)

def extract_bonus_amount(bonus_text: str) -> str:
    """Extract bonus amount from bonus description"""
    import re
    amounts = re.findall(r'\$[\d,]+', bonus_text)
    return amounts[0] if amounts else '$0'

def calculate_games_score(intelligence: Dict[str, Any]) -> float:
    """Calculate games score for Coinflip criteria"""
    total_games = int(intelligence.get('total_games', 0))
    providers_count = len(intelligence.get('game_providers', []))
    
    score = 2.0  # Base score
    
    if total_games > 500:
        score += 1.0
    if total_games > 1000:
        score += 0.5
    if providers_count > 10:
        score += 1.0
    if providers_count > 20:
        score += 0.5
    
    return min(5.0, score)

def calculate_bonuses_score(intelligence: Dict[str, Any]) -> float:
    """Calculate bonuses score for Coinflip criteria"""
    bonus_percentage = int(intelligence.get('bonus_percentage', 0))
    wagering = intelligence.get('wagering_requirements', '')
    
    score = 2.0
    
    if bonus_percentage >= 100:
        score += 1.5
    if bonus_percentage >= 200:
        score += 0.5
    
    # Lower wagering = better score
    if '35x' in wagering or '30x' in wagering:
        score += 1.0
    elif '25x' in wagering or '20x' in wagering:
        score += 1.5
    
    return min(5.0, score)

def calculate_payments_score(intelligence: Dict[str, Any]) -> float:
    """Calculate payments score for Coinflip criteria"""
    deposit_methods = len(intelligence.get('deposit_methods', []))
    withdrawal_methods = len(intelligence.get('withdrawal_methods', []))
    
    score = 2.0
    
    if deposit_methods >= 5:
        score += 1.0
    if deposit_methods >= 8:
        score += 0.5
    if withdrawal_methods >= 3:
        score += 1.0
    if withdrawal_methods >= 5:
        score += 0.5
    
    return min(5.0, score)

def calculate_support_score(intelligence: Dict[str, Any]) -> float:
    """Calculate support score for Coinflip criteria"""
    languages = len(intelligence.get('support_languages', []))
    
    score = 3.0  # Base score
    
    if languages >= 3:
        score += 1.0
    if languages >= 5:
        score += 1.0
    
    return min(5.0, score)

def calculate_mobile_score(intelligence: Dict[str, Any]) -> float:
    """Calculate mobile score for Coinflip criteria"""
    mobile_info = intelligence.get('mobile_compatibility', '').lower()
    
    score = 2.0
    
    if 'mobile site' in mobile_info:
        score += 1.0
    if 'app' in mobile_info:
        score += 1.5
    if 'ios' in mobile_info and 'android' in mobile_info:
        score += 0.5
    
    return min(5.0, score)

def calculate_security_score(intelligence: Dict[str, Any]) -> float:
    """Calculate security score for Coinflip criteria"""
    license_info = intelligence.get('license_info', '').lower()
    
    score = 2.0
    
    if 'uk gambling commission' in license_info:
        score += 1.5
    if 'malta' in license_info:
        score += 1.0
    if 'curacao' in license_info:
        score += 0.5
    
    return min(5.0, score)

def auto_assign_categories(intelligence: Dict[str, Any]) -> list:
    """Automatically assign WordPress categories"""
    categories = ['MT Casinos']  # Always MT Casinos
    
    # Add based on games
    total_games = int(intelligence.get('total_games', 0))
    if total_games > 500:
        categories.append('Premium Casinos')
    
    # Add based on bonus
    bonus_text = intelligence.get('welcome_bonus', '').lower()
    if 'free spins' in bonus_text:
        categories.append('Free Spins Casinos')
    
    # Add based on providers
    providers = intelligence.get('game_providers', [])
    if 'NetEnt' in providers:
        categories.append('NetEnt Casinos')
    if 'Microgaming' in providers:
        categories.append('Microgaming Casinos')
    
    return categories

def auto_generate_tags(intelligence: Dict[str, Any]) -> list:
    """Automatically generate WordPress tags"""
    tags = []
    
    # Casino name tag
    casino_name = intelligence.get('casino_name', '')
    if casino_name:
        tags.append(casino_name)
        tags.append(casino_name.replace(' ', '').lower())
    
    # License tags
    license_info = intelligence.get('license_info', '')
    if 'UK Gambling Commission' in license_info:
        tags.extend(['UKGC', 'UK Licensed'])
    if 'Malta' in license_info:
        tags.extend(['MGA', 'Malta Licensed'])
    
    # Game provider tags
    providers = intelligence.get('game_providers', [])
    tags.extend(providers)
    
    # Payment method tags
    payment_methods = intelligence.get('deposit_methods', [])
    tags.extend(payment_methods)
    
    # Bonus tags
    bonus_text = intelligence.get('welcome_bonus', '').lower()
    if 'free spins' in bonus_text:
        tags.append('Free Spins')
    if '$' in bonus_text:
        tags.append('Welcome Bonus')
    
    return list(set(tags))  # Remove duplicates

def auto_generate_shortcodes(intelligence: Dict[str, Any]) -> list:
    """Automatically generate Coinflip shortcodes"""
    shortcodes = []
    
    # Casino rating shortcode
    rating = intelligence.get('overall_rating', 0)
    shortcodes.append(f'[mt_casino_rating rating="{rating}" max="5"]')
    
    # Games count shortcode
    total_games = intelligence.get('total_games', 0)
    shortcodes.append(f'[mt_games_count total="{total_games}"]')
    
    # Bonus shortcode
    bonus = intelligence.get('welcome_bonus', '')
    if bonus:
        shortcodes.append(f'[mt_bonus_highlight text="{bonus}"]')
    
    # Payment methods shortcode
    deposit_methods = intelligence.get('deposit_methods', [])
    if deposit_methods:
        methods_str = ','.join(deposit_methods)
        shortcodes.append(f'[mt_payment_methods methods="{methods_str}"]')
    
    # Affiliate button shortcode
    casino_url = intelligence.get('casino_url', '')
    casino_name = intelligence.get('casino_name', '')
    if casino_url and casino_name:
        shortcodes.append(f'[mt_affiliate_button url="{casino_url}" text="Play at {casino_name}"]')
    
    return shortcodes

def display_categorization_results(original_intelligence: Dict, mt_metadata: Dict):
    """Display the automatic categorization results"""
    
    print("\n📊 INPUT: 95-Field Casino Intelligence")
    print("-" * 40)
    print(f"Casino Name: {original_intelligence['casino_name']}")
    print(f"Total Games: {original_intelligence['total_games']}")
    print(f"Welcome Bonus: {original_intelligence['welcome_bonus']}")
    print(f"License: {original_intelligence['license_info']}")
    
    print("\n✅ OUTPUT: MT Casinos Metadata (AUTOMATIC)")
    print("-" * 40)
    print(f"Post Type: {mt_metadata['post_type']} ✅")
    print(f"Metadata Fields: {len(mt_metadata['meta_fields'])} ✅")
    print(f"Categories: {len(mt_metadata['categories'])} ✅")
    print(f"Tags: {len(mt_metadata['tags'])} ✅")
    print(f"Shortcodes: {len(mt_metadata['shortcodes'])} ✅")
    
    print("\n🎯 KEY METADATA FIELDS:")
    print("-" * 40)
    key_fields = [
        '_casino_name', '_casino_rating', '_casino_total_games',
        '_casino_welcome_bonus', '_casino_criteria_games',
        '_casino_criteria_bonuses', '_casino_criteria_payments'
    ]
    
    for field in key_fields:
        value = mt_metadata['meta_fields'][field]
        print(f"{field}: {value}")
    
    print("\n🏷️ AUTOMATIC CATEGORIES:")
    print("-" * 40)
    for category in mt_metadata['categories']:
        print(f"• {category}")
    
    print("\n🔧 AUTOMATIC SHORTCODES:")
    print("-" * 40)
    for shortcode in mt_metadata['shortcodes'][:3]:  # Show first 3
        print(f"• {shortcode}")
    
    print("\n🎉 RESULT: 100% Automatic MT Casinos Categorization!")


if __name__ == "__main__":
    demonstrate_auto_categorization() 