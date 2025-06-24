#!/usr/bin/env python3
"""
🎰 COINFLIP MT CASINO PUBLISHER - Enhanced WordPress Integration
=============================================================

Advanced publisher for Coinflip theme MT Casino custom post types.
Maps 95-field casino intelligence to proper MT Casino structure.

Author: AI Assistant
Created: 2025-01-23
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass

from .wordpress_publisher import WordPressRESTPublisher, WordPressConfig, WordPressAuthManager

# Handle relative imports for standalone execution
try:
    from ..schemas.casino_intelligence_schema import CasinoIntelligence, GameProvider, PaymentMethodType, LicenseAuthority
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from schemas.casino_intelligence_schema import CasinoIntelligence, GameProvider, PaymentMethodType, LicenseAuthority

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# MT CASINO POST TYPES & TAXONOMIES
# ============================================================================

@dataclass
class MTCasinoPostType:
    """MT Casino post type definitions"""
    name: str
    endpoint: str
    description: str
    fields: List[str]


MT_CASINO_POST_TYPES = {
    'mt_listing': MTCasinoPostType(
        name='mt_listing',
        endpoint='/wp-json/wp/v2/mt_listing',
        description='Main casino listings',
        fields=['mt_rating', 'mt_bonus_amount', 'mt_license', 'mt_established', 'mt_payment_methods']
    ),
    'mt_bonus': MTCasinoPostType(
        name='mt_bonus',
        endpoint='/wp-json/wp/v2/mt_bonus',
        description='Casino bonus posts',
        fields=['mt_bonus_type', 'mt_bonus_percentage', 'mt_wagering_requirements', 'mt_bonus_value']
    ),
    'mt_bookmaker': MTCasinoPostType(
        name='mt_bookmaker',
        endpoint='/wp-json/wp/v2/mt_bookmaker',
        description='Bookmaker listings',
        fields=['mt_odds_types', 'mt_sports_coverage', 'mt_live_betting']
    ),
    'mt_reviews': MTCasinoPostType(
        name='mt_reviews',
        endpoint='/wp-json/wp/v2/mt_reviews',
        description='Casino review posts',
        fields=['mt_review_rating', 'mt_pros', 'mt_cons', 'mt_verdict']
    ),
    'mt_slots': MTCasinoPostType(
        name='mt_slots',
        endpoint='/wp-json/wp/v2/mt_slots',
        description='Slot game listings',
        fields=['mt_game_provider', 'mt_rtp', 'mt_volatility', 'mt_max_win']
    )
}

MT_CASINO_TAXONOMIES = {
    'mt_listing_category': ['crypto-casino', 'live-casino', 'mobile-casino', 'new-casino'],
    'mt_bonus_category': ['welcome-bonus', 'no-deposit', 'free-spins', 'cashback'],
    'mt_software': ['netent', 'microgaming', 'playtech', 'pragmatic-play', 'evolution-gaming'],
    'mt_payment': ['bitcoin', 'ethereum', 'visa', 'mastercard', 'paypal', 'skrill'],
    'mt_country': ['united-kingdom', 'canada', 'australia', 'germany', 'sweden'],
    'mt_license': ['mga', 'ukgc', 'curacao', 'gibraltar', 'kahnawake']
}


# ============================================================================
# CONTENT TYPE ANALYZER
# ============================================================================

class ContentTypeAnalyzer:
    """Analyzes content to determine appropriate MT Casino post type"""
    
    @staticmethod
    def analyze_content_type(content: str, casino_data: Optional[CasinoIntelligence] = None) -> str:
        """
        Analyze content to determine the best MT Casino post type
        
        Returns:
            str: MT Casino post type ('mt_listing', 'mt_bonus', 'mt_slots', 'mt_reviews')
        """
        content_lower = content.lower()
        
        # Bonus-focused content
        bonus_keywords = ['bonus', 'free spins', 'welcome offer', 'promotion', 'cashback', 'no deposit']
        if any(keyword in content_lower for keyword in bonus_keywords):
            if casino_data and casino_data.bonuses.welcome_bonus.bonus_amount:
                return 'mt_bonus'
        
        # Slot/game-focused content
        slot_keywords = ['slot', 'game', 'jackpot', 'rtp', 'volatility', 'provider']
        if any(keyword in content_lower for keyword in slot_keywords):
            if casino_data and casino_data.games.game_portfolio.slot_games_count:
                return 'mt_slots'
        
        # Review-focused content
        review_keywords = ['review', 'pros', 'cons', 'verdict', 'rating', 'experience']
        if any(keyword in content_lower for keyword in review_keywords):
            return 'mt_reviews'
        
        # Default to main casino listing
        return 'mt_listing'


# ============================================================================
# METADATA MAPPER
# ============================================================================

class MTCasinoMetadataMapper:
    """Maps 95-field casino intelligence to MT Casino custom fields"""
    
    @staticmethod
    def map_to_mt_listing(casino_data: CasinoIntelligence) -> Dict[str, Any]:
        """Map casino intelligence to mt_listing custom fields"""
        return {
            'mt_rating': casino_data.overall_rating or 0,
            'mt_bonus_amount': casino_data.bonuses.welcome_bonus.bonus_amount or 'N/A',
            'mt_license': MTCasinoMetadataMapper._format_license(casino_data.trustworthiness.license_info.primary_license),
            'mt_established': casino_data.trustworthiness.reputation_metrics.years_in_operation or 'Unknown',
            'mt_payment_methods': MTCasinoMetadataMapper._format_payment_methods(casino_data.payments.payment_methods),
            'mt_software_providers': MTCasinoMetadataMapper._format_software_providers(casino_data.games.software_providers.primary_providers),
            'mt_min_deposit': casino_data.payments.minimum_deposit_amount or 'N/A',
            'mt_withdrawal_time': casino_data.payments.withdrawal_processing_time or 'N/A',
            'mt_mobile_compatible': casino_data.user_experience.mobile_compatibility,
            'mt_live_chat': casino_data.user_experience.customer_support.live_chat_available,
            'mt_total_games': casino_data.games.game_portfolio.total_games or 0,
            'mt_safety_score': casino_data.safety_score or 0,
            'mt_player_experience_score': casino_data.player_experience_score or 0,
            'mt_value_score': casino_data.value_score or 0
        }
    
    @staticmethod
    def map_to_mt_bonus(casino_data: CasinoIntelligence) -> Dict[str, Any]:
        """Map casino intelligence to mt_bonus custom fields"""
        welcome_bonus = casino_data.bonuses.welcome_bonus
        return {
            'mt_bonus_type': welcome_bonus.bonus_type or 'Welcome Bonus',
            'mt_bonus_percentage': welcome_bonus.bonus_percentage or 0,
            'mt_bonus_value': welcome_bonus.bonus_amount or 'N/A',
            'mt_wagering_requirements': welcome_bonus.wagering_requirements or 'N/A',
            'mt_free_spins': welcome_bonus.free_spins_count or 0,
            'mt_min_deposit': welcome_bonus.minimum_deposit or 'N/A',
            'mt_time_limit': welcome_bonus.time_limit or 'N/A',
            'mt_game_restrictions': ', '.join(welcome_bonus.game_restrictions) if welcome_bonus.game_restrictions else 'N/A',
            'mt_bonus_rating': casino_data.value_score or 0
        }
    
    @staticmethod
    def _format_license(license_authority: Optional[LicenseAuthority]) -> str:
        """Format license authority for display"""
        if not license_authority:
            return 'Unlicensed'
        return license_authority.value if hasattr(license_authority, 'value') else str(license_authority)
    
    @staticmethod
    def _format_payment_methods(payment_methods: List[Any]) -> str:
        """Format payment methods for display"""
        if not payment_methods:
            return 'N/A'
        method_names = [method.name for method in payment_methods[:5]]  # Limit to 5
        return ', '.join(method_names)
    
    @staticmethod
    def _format_software_providers(providers: List[GameProvider]) -> str:
        """Format software providers for display"""
        if not providers:
            return 'N/A'
        provider_names = [provider.value if hasattr(provider, 'value') else str(provider) for provider in providers[:5]]
        return ', '.join(provider_names)


# ============================================================================
# ENHANCED COINFLIP MT CASINO PUBLISHER
# ============================================================================

class CoinflipMTCasinoPublisher(WordPressRESTPublisher):
    """Enhanced publisher for Coinflip theme MT Casino integration"""
    
    def __init__(self, config: WordPressConfig):
        """Initialize with WordPress configuration"""
        super().__init__(config)
        self.content_analyzer = ContentTypeAnalyzer()
        self.metadata_mapper = MTCasinoMetadataMapper()
    
    async def publish_mt_casino_content(
        self,
        content: str,
        title: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Publish content using MT Casino post types with intelligent mapping
        """
        start_time = datetime.now()
        results = {
            'success': False,
            'main_post': None,
            'errors': [],
            'processing_time': 0,
            'mt_casino_features_used': []
        }
        
        try:
            logger.info(f"🎰 Starting MT Casino publishing for: {title}")
            
            # Determine primary post type
            primary_post_type = self.content_analyzer.analyze_content_type(content, casino_data)
            logger.info(f"📊 Detected content type: {primary_post_type}")
            
            # Try MT Casino post type first
            mt_casino_result = await self._publish_mt_casino_post(
                content, title, primary_post_type, casino_data, images
            )
            
            if mt_casino_result.get('success'):
                results['main_post'] = mt_casino_result
                results['success'] = True
                results['mt_casino_features_used'].append(f"MT Casino: {primary_post_type}")
            else:
                # Fallback to enhanced regular post
                fallback_result = await self._publish_enhanced_fallback(
                    content, title, primary_post_type, casino_data, images
                )
                results['main_post'] = fallback_result
                results['success'] = fallback_result.get('success', False)
                results['mt_casino_features_used'].append("Fallback with MT Casino styling")
            
        except Exception as e:
            error_msg = f"MT Casino publishing failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        finally:
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    async def _publish_mt_casino_post(
        self,
        content: str,
        title: str,
        post_type: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Attempt to publish to MT Casino post type"""
        
        try:
            # Prepare post data
            post_data = {
                'title': title,
                'content': content,
                'status': 'draft'
            }
            
            # Add MT Casino custom fields
            if casino_data:
                if post_type == 'mt_listing':
                    custom_fields = self.metadata_mapper.map_to_mt_listing(casino_data)
                elif post_type == 'mt_bonus':
                    custom_fields = self.metadata_mapper.map_to_mt_bonus(casino_data)
                else:
                    custom_fields = {}
                
                if custom_fields:
                    post_data['meta'] = custom_fields
            
            # Get MT Casino endpoint
            mt_post_type = MT_CASINO_POST_TYPES.get(post_type)
            if not mt_post_type:
                return {'success': False, 'error': f'Unknown post type: {post_type}'}
            
            endpoint = mt_post_type.endpoint
            
            # Attempt to publish to MT Casino endpoint
            from urllib.parse import urljoin
            endpoint_url = urljoin(self.config.site_url, endpoint)
            
            if not self.session:
                raise Exception("Session not initialized. Use 'async with' context manager.")
            
            async with self.session.post(endpoint_url, 
                                       headers=self.auth_manager.headers, 
                                       json=post_data) as response:
                
                if response.status in [200, 201]:
                    result = await response.json()
                    post_id = result['id']
                    logger.info(f"✅ Published to {post_type}: Post ID {post_id}")
                    return {
                        'success': True,
                        'post_id': post_id,
                        'post_type': post_type,
                        'url': result.get('link'),
                        'method': 'mt_casino_native'
                    }
                else:
                    error_text = await response.text()
                    return {'success': False, 'error': f'HTTP {response.status}: {error_text}'}
                
        except Exception as e:
            logger.error(f"❌ MT Casino post failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _publish_enhanced_fallback(
        self,
        content: str,
        title: str,
        intended_post_type: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Publish as enhanced regular post with MT Casino styling"""
        
        # Create MT Casino styled content
        enhanced_content = self._create_mt_casino_styled_content(content, intended_post_type, casino_data)
        
        # Publish as regular post
        post_data = {
            'title': f"🎰 {title}",
            'content': enhanced_content,
            'status': 'draft',
            'categories': [1]  # Default category
        }
        
        # Add casino metadata as custom fields
        if casino_data:
            post_data['meta'] = {
                'casino_name': casino_data.casino_name,
                'overall_rating': casino_data.overall_rating or 0,
                'intended_post_type': intended_post_type,
                'mt_casino_fallback': True
            }
        
        try:
            # Publish to regular posts endpoint
            from urllib.parse import urljoin
            posts_url = urljoin(self.config.site_url, '/wp-json/wp/v2/posts')
            
            if not self.session:
                raise Exception("Session not initialized. Use 'async with' context manager.")
            
            async with self.session.post(posts_url, 
                                       headers=self.auth_manager.headers, 
                                       json=post_data) as response:
                
                if response.status in [200, 201]:
                    result = await response.json()
                    post_id = result['id']
                    logger.info(f"✅ Published enhanced fallback: Post ID {post_id}")
                    return {
                        'success': True,
                        'post_id': post_id,
                        'post_type': 'post',
                        'intended_type': intended_post_type,
                        'url': result.get('link'),
                        'method': 'enhanced_fallback'
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Enhanced fallback failed: HTTP {response.status}: {error_text}")
                    return {'success': False, 'error': f'HTTP {response.status}: {error_text}'}
        
        except Exception as e:
            logger.error(f"❌ Enhanced fallback failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_mt_casino_styled_content(self, content: str, post_type: str, casino_data: Optional[CasinoIntelligence] = None) -> str:
        """Create MT Casino styled content for fallback posts"""
        
        styled_content = f'<div class="mt-casino-{post_type} mt-casino-fallback">\n'
        
        # Add post type badge
        type_names = {
            'mt_listing': '🏆 Casino Review',
            'mt_bonus': '💰 Casino Bonus',
            'mt_slots': '🎰 Slot Games',
            'mt_reviews': '⭐ Expert Review'
        }
        
        type_name = type_names.get(post_type, '🎲 Casino Content')
        styled_content += f'<div class="mt-casino-badge">{type_name}</div>\n\n'
        
        # Add casino quick facts if available
        if casino_data:
            styled_content += self._create_quick_facts_box(casino_data, post_type)
        
        # Add main content
        styled_content += content
        
        # Add MT Casino compatibility notice
        styled_content += '\n\n<div class="mt-casino-notice">'
        styled_content += '<p><strong>💡 MT Casino Ready:</strong> This content is optimized for the Coinflip theme MT Casino features.</p>'
        styled_content += '</div>'
        
        styled_content += '</div>'
        
        return styled_content
    
    def _create_quick_facts_box(self, casino_data: CasinoIntelligence, post_type: str) -> str:
        """Create a quick facts summary box"""
        
        facts = f'<div class="mt-casino-quick-facts">\n'
        facts += f'<h3>🎯 Quick Facts - {casino_data.casino_name}</h3>\n'
        facts += '<div class="facts-grid">\n'
        
        if post_type == 'mt_listing':
            if casino_data.overall_rating:
                facts += f'<div class="fact-item"><strong>Rating:</strong> {casino_data.overall_rating}/10 ⭐</div>\n'
            if casino_data.bonuses.welcome_bonus.bonus_amount:
                facts += f'<div class="fact-item"><strong>Welcome Bonus:</strong> {casino_data.bonuses.welcome_bonus.bonus_amount} 💰</div>\n'
            if casino_data.games.game_portfolio.total_games:
                facts += f'<div class="fact-item"><strong>Total Games:</strong> {casino_data.games.game_portfolio.total_games} 🎮</div>\n'
        
        elif post_type == 'mt_bonus':
            bonus = casino_data.bonuses.welcome_bonus
            if bonus.bonus_amount:
                facts += f'<div class="fact-item"><strong>Bonus:</strong> {bonus.bonus_amount}</div>\n'
            if bonus.wagering_requirements:
                facts += f'<div class="fact-item"><strong>Wagering:</strong> {bonus.wagering_requirements}</div>\n'
        
        facts += '</div>\n</div>\n\n'
        
        return facts


# ============================================================================
# INTEGRATION CLASS & FACTORY
# ============================================================================

class CoinflipMTCasinoIntegration:
    """Main integration class for Coinflip MT Casino publishing"""
    
    def __init__(self, wp_config: WordPressConfig):
        self.publisher = CoinflipMTCasinoPublisher(wp_config)
    
    async def publish_casino_content(
        self,
        content: str,
        title: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Main publishing method"""
        return await self.publisher.publish_mt_casino_content(
            content=content,
            title=title,
            casino_data=casino_data,
            images=images
        )


def create_coinflip_mt_casino_publisher(
    site_url: str,
    username: str,
    application_password: str
) -> CoinflipMTCasinoIntegration:
    """Factory function to create Coinflip MT Casino publisher"""
    wp_config = WordPressConfig(
        site_url=site_url,
        username=username,
        application_password=application_password
    )
    return CoinflipMTCasinoIntegration(wp_config) 