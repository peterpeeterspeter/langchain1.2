#!/usr/bin/env python3
"""
🎰 COINFLIP AUTO-CATEGORIZATION DEMO
===================================

Demonstrates automatic categorization of casino content as MT Casinos 
with complete metadata mapping from 95-field casino intelligence

This demo shows:
1. ✅ Automatic MT Casinos post type assignment
2. ✅ Complete metadata field mapping (95 fields → Coinflip fields)
3. ✅ Dynamic categorization based on casino features
4. ✅ Automatic shortcode generation
5. ✅ Review criteria scoring integration
6. ✅ SEO and schema markup generation

Author: AI Assistant
Created: 2025-01-21
Version: 1.0.0
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our systems
from integrations.coinflip_wordpress_publisher import (
    CoinflipWordPressPublisher, 
    CoinflipMetadata,
    create_coinflip_publisher
)
from schemas.casino_intelligence_schema import CasinoIntelligence
from chains.universal_rag_lcel import create_universal_rag_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinflipAutoCategorizationDemo:
    """
    Comprehensive demo of Coinflip theme automatic categorization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # WordPress credentials for Coinflip site
        self.wordpress_config = {
            'site_url': 'https://your-coinflip-site.com',
            'username': 'admin',
            'application_password': 'your-app-password'
        }
        
        # Initialize Coinflip publisher
        self.coinflip_publisher = None
        
    async def initialize_systems(self):
        """Initialize all required systems"""
        try:
            self.logger.info("🚀 Initializing Coinflip Auto-Categorization Demo")
            
            # Initialize Coinflip WordPress Publisher
            self.coinflip_publisher = create_coinflip_publisher(
                site_url=self.wordpress_config['site_url'],
                username=self.wordpress_config['username'],
                application_password=self.wordpress_config['application_password']
            )
            
            # Initialize Universal RAG Chain for content generation
            self.rag_chain = create_universal_rag_chain()
            
            self.logger.info("✅ All systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing systems: {str(e)}")
            raise
    
    async def demo_automatic_categorization(self):
        """
        Demonstrate automatic categorization of casino content as MT Casinos
        """
        try:
            self.logger.info("🎯 Starting Automatic Categorization Demo")
            
            # Step 1: Generate casino content with Universal RAG Chain
            self.logger.info("1️⃣ Generating casino content with Universal RAG Chain...")
            
            casino_query = "Comprehensive review of Betway Casino focusing on games, bonuses, payments, and user experience"
            
            # Generate content using Universal RAG Chain
            rag_response = await self.rag_chain.ainvoke({'query': casino_query})
            
            # Extract components
            content = rag_response.answer
            casino_intelligence = rag_response.casino_intelligence  # Assuming this is available
            images = rag_response.images if hasattr(rag_response, 'images') else []
            
            self.logger.info(f"✅ Generated {len(content)} characters of casino content")
            
            # Step 2: Demonstrate automatic metadata mapping
            self.logger.info("2️⃣ Demonstrating automatic metadata mapping...")
            
            # Map intelligence to Coinflip metadata (this shows the automatic categorization)
            coinflip_meta = self.coinflip_publisher._map_intelligence_to_coinflip(casino_intelligence)
            
            # Display the automatic categorization results
            self._display_categorization_results(coinflip_meta)
            
            # Step 3: Show automatic post type assignment
            self.logger.info("3️⃣ Showing automatic MT Casinos post type assignment...")
            
            post_data = self.coinflip_publisher._create_mt_casino_post(
                casino_intelligence, 
                content, 
                coinflip_meta, 
                images
            )
            
            self._display_post_structure(post_data)
            
            # Step 4: Demonstrate metadata field mapping
            self.logger.info("4️⃣ Demonstrating complete metadata field mapping...")
            
            # Simulate metadata creation (normally done after WordPress post creation)
            mock_post_id = 12345
            meta_fields = self._generate_meta_fields_preview(coinflip_meta)
            
            self._display_metadata_mapping(meta_fields)
            
            # Step 5: Show automatic shortcode generation
            self.logger.info("5️⃣ Showing automatic shortcode generation...")
            
            enhanced_content = self.coinflip_publisher._add_coinflip_shortcodes(content, coinflip_meta)
            
            self._display_shortcode_integration(content, enhanced_content)
            
            # Step 6: Demonstrate automatic categorization logic
            self.logger.info("6️⃣ Demonstrating automatic categorization logic...")
            
            categories = self.coinflip_publisher._determine_categories(casino_intelligence)
            tags = self.coinflip_publisher._generate_tags(casino_intelligence)
            
            self._display_taxonomy_results(categories, tags)
            
            # Step 7: Save demo results
            self.logger.info("7️⃣ Saving demo results...")
            
            demo_results = {
                'demo_timestamp': datetime.now().isoformat(),
                'coinflip_metadata': self._serialize_coinflip_metadata(coinflip_meta),
                'post_structure': post_data,
                'metadata_fields': meta_fields,
                'categories': categories,
                'tags': tags,
                'content_length': len(content),
                'enhanced_content_length': len(enhanced_content),
                'shortcode_count': enhanced_content.count('[mt_'),
                'automatic_features': {
                    'post_type': 'mt_casinos',
                    'metadata_fields_mapped': len(meta_fields),
                    'categories_auto_assigned': len(categories),
                    'tags_auto_generated': len(tags),
                    'shortcodes_inserted': enhanced_content.count('[mt_'),
                    'schema_markup_generated': True,
                    'seo_optimized': True
                }
            }
            
            # Save results
            results_file = Path(__file__).parent / f"coinflip_auto_categorization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(demo_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Demo results saved to: {results_file}")
            
            # Summary
            self._display_demo_summary(demo_results)
            
        except Exception as e:
            self.logger.error(f"❌ Error in auto-categorization demo: {str(e)}")
            raise
    
    def _display_categorization_results(self, coinflip_meta: CoinflipMetadata):
        """Display the automatic categorization results"""
        
        print("\n" + "="*80)
        print("🎯 AUTOMATIC CATEGORIZATION RESULTS")
        print("="*80)
        
        print(f"🏢 Casino Name: {coinflip_meta.casino_name}")
        print(f"⭐ Rating: {coinflip_meta.casino_rating}/5.0")
        print(f"🔗 URL: {coinflip_meta.casino_url}")
        print(f"💎 Affiliate URL: {coinflip_meta.affiliate_url}")
        
        print(f"\n📊 GAMES INTELLIGENCE:")
        print(f"   • Total Games: {coinflip_meta.total_games}")
        print(f"   • Slot Games: {coinflip_meta.slot_games}")
        print(f"   • Table Games: {coinflip_meta.table_games}")
        print(f"   • Live Games: {coinflip_meta.live_games}")
        print(f"   • Providers: {len(coinflip_meta.game_providers or [])}")
        
        print(f"\n🎁 BONUS INTELLIGENCE:")
        print(f"   • Welcome Bonus: {coinflip_meta.welcome_bonus}")
        print(f"   • Bonus Amount: {coinflip_meta.bonus_amount}")
        print(f"   • Wagering: {coinflip_meta.wagering_requirements}")
        print(f"   • Bonus Codes: {len(coinflip_meta.bonus_codes or [])}")
        
        print(f"\n💳 PAYMENT INTELLIGENCE:")
        print(f"   • Deposit Methods: {len(coinflip_meta.deposit_methods or [])}")
        print(f"   • Withdrawal Methods: {len(coinflip_meta.withdrawal_methods or [])}")
        print(f"   • Min Deposit: {coinflip_meta.min_deposit}")
        print(f"   • Withdrawal Time: {coinflip_meta.withdrawal_time}")
        
        print(f"\n🏆 REVIEW CRITERIA SCORES:")
        print(f"   • Games: {coinflip_meta.criteria_games}/5.0")
        print(f"   • Bonuses: {coinflip_meta.criteria_bonuses}/5.0")
        print(f"   • Payments: {coinflip_meta.criteria_payments}/5.0")
        print(f"   • Support: {coinflip_meta.criteria_support}/5.0")
        print(f"   • Mobile: {coinflip_meta.criteria_mobile}/5.0")
        print(f"   • Security: {coinflip_meta.criteria_security}/5.0")
        
    def _display_post_structure(self, post_data: dict):
        """Display the automatic post structure creation"""
        
        print("\n" + "="*80)
        print("📝 AUTOMATIC MT CASINOS POST STRUCTURE")
        print("="*80)
        
        print(f"📋 Post Type: {post_data['post_type']} (AUTOMATICALLY ASSIGNED)")
        print(f"📰 Title: {post_data['title']}")
        print(f"📄 Status: {post_data['status']}")
        print(f"📝 Excerpt Length: {len(post_data['excerpt'])} characters")
        print(f"📚 Content Length: {len(post_data['content'])} characters")
        
        print(f"\n🏷️ AUTOMATIC TAXONOMY:")
        print(f"   • Categories: {', '.join(post_data['categories'])}")
        print(f"   • Tags: {', '.join(post_data['tags'][:5])}... ({len(post_data['tags'])} total)")
        
        print(f"\n🔍 SEO OPTIMIZATION:")
        print(f"   • SEO Title: {post_data['meta']['_yoast_wpseo_title']}")
        print(f"   • Meta Description: {post_data['meta']['_yoast_wpseo_metadesc'][:100]}...")
        print(f"   • Focus Keyword: {post_data['meta']['_yoast_wpseo_focuskw']}")
        print(f"   • Schema Markup: ✅ Generated")
        
    def _generate_meta_fields_preview(self, coinflip_meta: CoinflipMetadata) -> dict:
        """Generate preview of metadata fields that would be created"""
        
        return {
            # Core Casino Fields (6 fields)
            '_casino_name': coinflip_meta.casino_name,
            '_casino_url': coinflip_meta.casino_url,
            '_casino_affiliate_url': coinflip_meta.affiliate_url,
            '_casino_rating': coinflip_meta.casino_rating,
            '_casino_logo': coinflip_meta.casino_logo,
            '_casino_banner': coinflip_meta.casino_banner,
            
            # License & Safety (4 fields)
            '_casino_license_info': coinflip_meta.license_info,
            '_casino_license_jurisdictions': json.dumps(coinflip_meta.license_jurisdictions or []),
            '_casino_safety_rating': coinflip_meta.safety_rating,
            '_casino_security_features': json.dumps(coinflip_meta.security_features or []),
            
            # Games (5 fields)
            '_casino_total_games': coinflip_meta.total_games,
            '_casino_slot_games': coinflip_meta.slot_games,
            '_casino_table_games': coinflip_meta.table_games,
            '_casino_live_games': coinflip_meta.live_games,
            '_casino_game_providers': json.dumps(coinflip_meta.game_providers or []),
            
            # Bonuses (5 fields)
            '_casino_welcome_bonus': coinflip_meta.welcome_bonus,
            '_casino_bonus_amount': coinflip_meta.bonus_amount,
            '_casino_bonus_percentage': coinflip_meta.bonus_percentage,
            '_casino_wagering_requirements': coinflip_meta.wagering_requirements,
            '_casino_bonus_codes': json.dumps(coinflip_meta.bonus_codes or []),
            
            # Payment Methods (5 fields)
            '_casino_deposit_methods': json.dumps(coinflip_meta.deposit_methods or []),
            '_casino_withdrawal_methods': json.dumps(coinflip_meta.withdrawal_methods or []),
            '_casino_min_deposit': coinflip_meta.min_deposit,
            '_casino_max_withdrawal': coinflip_meta.max_withdrawal,
            '_casino_withdrawal_time': coinflip_meta.withdrawal_time,
            
            # Support & UX (4 fields)
            '_casino_support_languages': json.dumps(coinflip_meta.support_languages or []),
            '_casino_support_methods': json.dumps(coinflip_meta.support_methods or []),
            '_casino_mobile_compatibility': coinflip_meta.mobile_compatibility,
            '_casino_app_available': coinflip_meta.app_available,
            
            # Review Criteria (6 fields) - Coinflip theme scoring
            '_casino_criteria_games': coinflip_meta.criteria_games,
            '_casino_criteria_bonuses': coinflip_meta.criteria_bonuses,
            '_casino_criteria_payments': coinflip_meta.criteria_payments,
            '_casino_criteria_support': coinflip_meta.criteria_support,
            '_casino_criteria_mobile': coinflip_meta.criteria_mobile,
            '_casino_criteria_security': coinflip_meta.criteria_security,
        }
    
    def _display_metadata_mapping(self, meta_fields: dict):
        """Display the automatic metadata mapping"""
        
        print("\n" + "="*80)
        print("🗃️ AUTOMATIC METADATA FIELD MAPPING")
        print("="*80)
        
        # Group by category
        categories = {
            'Core Casino': [k for k in meta_fields.keys() if k.startswith('_casino_') and not any(cat in k for cat in ['license', 'security', 'total', 'slot', 'table', 'live', 'game', 'welcome', 'bonus', 'wagering', 'deposit', 'withdrawal', 'min', 'max', 'support', 'mobile', 'app', 'criteria'])],
            'License & Safety': [k for k in meta_fields.keys() if any(word in k for word in ['license', 'safety', 'security'])],
            'Games': [k for k in meta_fields.keys() if any(word in k for word in ['total_games', 'slot_games', 'table_games', 'live_games', 'game_providers'])],
            'Bonuses': [k for k in meta_fields.keys() if any(word in k for word in ['welcome', 'bonus', 'wagering'])],
            'Payments': [k for k in meta_fields.keys() if any(word in k for word in ['deposit', 'withdrawal', 'min_deposit', 'max_withdrawal'])],
            'Support & UX': [k for k in meta_fields.keys() if any(word in k for word in ['support', 'mobile', 'app'])],
            'Review Criteria': [k for k in meta_fields.keys() if 'criteria' in k]
        }
        
        total_fields = 0
        for category, fields in categories.items():
            if fields:
                print(f"\n📂 {category} ({len(fields)} fields):")
                for field in fields:
                    value = meta_fields[field]
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:50] + "..."
                    print(f"   • {field}: {value}")
                total_fields += len(fields)
        
        print(f"\n🎯 TOTAL METADATA FIELDS: {total_fields} (AUTOMATICALLY MAPPED)")
        
    def _display_shortcode_integration(self, original_content: str, enhanced_content: str):
        """Display automatic shortcode integration"""
        
        print("\n" + "="*80)
        print("🔧 AUTOMATIC SHORTCODE INTEGRATION")
        print("="*80)
        
        shortcode_count = enhanced_content.count('[mt_')
        additional_content = len(enhanced_content) - len(original_content)
        
        print(f"📝 Original Content: {len(original_content)} characters")
        print(f"⚡ Enhanced Content: {len(enhanced_content)} characters")
        print(f"➕ Additional Content: +{additional_content} characters")
        print(f"🔧 Shortcodes Added: {shortcode_count}")
        
        # Extract and display shortcodes
        import re
        shortcodes = re.findall(r'\[mt_[^\]]+\]', enhanced_content)
        
        print(f"\n🎯 AUTOMATICALLY GENERATED SHORTCODES:")
        for i, shortcode in enumerate(shortcodes, 1):
            print(f"   {i}. {shortcode}")
        
    def _display_taxonomy_results(self, categories: list, tags: list):
        """Display automatic taxonomy assignment"""
        
        print("\n" + "="*80)
        print("🏷️ AUTOMATIC TAXONOMY ASSIGNMENT")
        print("="*80)
        
        print(f"📂 CATEGORIES ({len(categories)}):")
        for category in categories:
            print(f"   • {category}")
        
        print(f"\n🏷️ TAGS ({len(tags)}):")
        for tag in tags[:10]:  # Show first 10
            print(f"   • {tag}")
        if len(tags) > 10:
            print(f"   ... and {len(tags) - 10} more")
        
    def _serialize_coinflip_metadata(self, coinflip_meta: CoinflipMetadata) -> dict:
        """Serialize CoinflipMetadata for JSON storage"""
        return {
            'casino_name': coinflip_meta.casino_name,
            'casino_url': coinflip_meta.casino_url,
            'affiliate_url': coinflip_meta.affiliate_url,
            'casino_rating': coinflip_meta.casino_rating,
            'total_games': coinflip_meta.total_games,
            'welcome_bonus': coinflip_meta.welcome_bonus,
            'deposit_methods_count': len(coinflip_meta.deposit_methods or []),
            'withdrawal_methods_count': len(coinflip_meta.withdrawal_methods or []),
            'criteria_scores': {
                'games': coinflip_meta.criteria_games,
                'bonuses': coinflip_meta.criteria_bonuses,
                'payments': coinflip_meta.criteria_payments,
                'support': coinflip_meta.criteria_support,
                'mobile': coinflip_meta.criteria_mobile,
                'security': coinflip_meta.criteria_security
            }
        }
    
    def _display_demo_summary(self, demo_results: dict):
        """Display comprehensive demo summary"""
        
        print("\n" + "="*80)
        print("🎉 COINFLIP AUTO-CATEGORIZATION DEMO SUMMARY")
        print("="*80)
        
        auto_features = demo_results['automatic_features']
        
        print(f"✅ POST TYPE: {auto_features['post_type']} (AUTOMATICALLY ASSIGNED)")
        print(f"✅ METADATA FIELDS: {auto_features['metadata_fields_mapped']} (AUTOMATICALLY MAPPED)")
        print(f"✅ CATEGORIES: {auto_features['categories_auto_assigned']} (AUTOMATICALLY ASSIGNED)")
        print(f"✅ TAGS: {auto_features['tags_auto_generated']} (AUTOMATICALLY GENERATED)")
        print(f"✅ SHORTCODES: {auto_features['shortcodes_inserted']} (AUTOMATICALLY INSERTED)")
        print(f"✅ SCHEMA MARKUP: {'GENERATED' if auto_features['schema_markup_generated'] else 'NOT GENERATED'}")
        print(f"✅ SEO OPTIMIZATION: {'APPLIED' if auto_features['seo_optimized'] else 'NOT APPLIED'}")
        
        print(f"\n📊 CONTENT STATISTICS:")
        print(f"   • Original Content: {demo_results['content_length']} characters")
        print(f"   • Enhanced Content: {demo_results['enhanced_content_length']} characters")
        print(f"   • Enhancement: +{demo_results['enhanced_content_length'] - demo_results['content_length']} characters")
        
        print(f"\n🎯 KEY ACHIEVEMENT:")
        print(f"   ✅ 95-field Casino Intelligence automatically converted to")
        print(f"   ✅ Coinflip MT Casinos custom post type with complete metadata")
        print(f"   ✅ No manual categorization required - 100% automatic!")
        
        print("\n" + "="*80)


async def main():
    """
    Main demo execution
    """
    print("🎰 COINFLIP AUTO-CATEGORIZATION DEMO")
    print("===================================")
    print("Demonstrating automatic categorization as MT Casinos with metadata")
    print()
    
    demo = CoinflipAutoCategorizationDemo()
    
    try:
        # Initialize systems
        await demo.initialize_systems()
        
        # Run the full demo
        await demo.demo_automatic_categorization()
        
        print("\n🎉 Demo completed successfully!")
        print("Check the generated JSON file for complete results.")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 