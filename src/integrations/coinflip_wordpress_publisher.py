#!/usr/bin/env python3
"""
🎰 COINFLIP MT CASINO PUBLISHER - Enhanced WordPress Integration
=============================================================

Advanced publisher for Coinflip theme MT Casino custom post types.
Maps 95-field casino intelligence to proper MT Casino structure.

FEATURES:
- Intelligent content type detection (casino, bonus, slot, review)
- Multi-post strategy (main + related posts)
- Rich MT Casino metadata mapping
- Taxonomy management (categories, software, payments)
- Graceful fallback to regular posts
- Integration with existing WordPress REST API

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
from ..schemas.casino_intelligence_schema import CasinoIntelligence, GameProvider, PaymentMethodType, LicenseAuthority

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
    def map_to_mt_slots(casino_data: CasinoIntelligence) -> Dict[str, Any]:
        """Map casino intelligence to mt_slots custom fields"""
        games = casino_data.games
        return {
            'mt_total_slots': games.game_portfolio.slot_games_count or 0,
            'mt_game_providers': MTCasinoMetadataMapper._format_software_providers(games.software_providers.primary_providers),
            'mt_progressive_jackpots': games.game_portfolio.progressive_jackpot_count or 0,
            'mt_popular_slots': ', '.join(games.game_portfolio.popular_slot_titles) if games.game_portfolio.popular_slot_titles else 'N/A',
            'mt_demo_mode': games.demo_mode_available,
            'mt_mobile_optimized': games.mobile_game_optimization,
            'mt_game_quality_rating': games.game_quality_rating or 0
        }
    
    @staticmethod
    def map_to_mt_reviews(casino_data: CasinoIntelligence) -> Dict[str, Any]:
        """Map casino intelligence to mt_reviews custom fields"""
        return {
            'mt_review_rating': casino_data.overall_rating or 0,
            'mt_safety_rating': casino_data.safety_score or 0,
            'mt_games_rating': casino_data.games.game_quality_rating or 0,
            'mt_bonuses_rating': casino_data.value_score or 0,
            'mt_support_rating': casino_data.user_experience.customer_support.support_quality_rating or 0,
            'mt_pros': MTCasinoMetadataMapper._generate_pros(casino_data),
            'mt_cons': MTCasinoMetadataMapper._generate_cons(casino_data),
            'mt_verdict': MTCasinoMetadataMapper._generate_verdict(casino_data)
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
    
    @staticmethod
    def _generate_pros(casino_data: CasinoIntelligence) -> str:
        """Generate pros list from casino data"""
        pros = []
        
        # Safety & licensing
        if casino_data.trustworthiness.license_info.primary_license:
            pros.append(f"Licensed by {MTCasinoMetadataMapper._format_license(casino_data.trustworthiness.license_info.primary_license)}")
        
        # Games
        if casino_data.games.game_portfolio.total_games and casino_data.games.game_portfolio.total_games > 1000:
            pros.append(f"Large game selection ({casino_data.games.game_portfolio.total_games}+ games)")
        
        # Bonuses
        if casino_data.bonuses.welcome_bonus.bonus_amount:
            pros.append(f"Generous welcome bonus ({casino_data.bonuses.welcome_bonus.bonus_amount})")
        
        # Support
        if casino_data.user_experience.customer_support.live_chat_available:
            pros.append("24/7 live chat support")
        
        # Mobile
        if casino_data.user_experience.mobile_compatibility:
            pros.append("Mobile-friendly platform")
        
        return '\n'.join([f"• {pro}" for pro in pros[:5]])  # Limit to 5 pros
    
    @staticmethod
    def _generate_cons(casino_data: CasinoIntelligence) -> str:
        """Generate cons list from casino data"""
        cons = []
        
        # Licensing concerns
        if not casino_data.trustworthiness.license_info.primary_license:
            cons.append("No clear licensing information")
        
        # Payment limitations
        if not casino_data.payments.payment_methods:
            cons.append("Limited payment options")
        
        # Support limitations
        if not casino_data.user_experience.customer_support.live_chat_available:
            cons.append("No live chat support")
        
        # Mobile limitations
        if not casino_data.user_experience.mobile_compatibility:
            cons.append("Limited mobile optimization")
        
        # Withdrawal times
        if casino_data.payments.withdrawal_processing_time and '24' not in casino_data.payments.withdrawal_processing_time.lower():
            cons.append("Slow withdrawal processing")
        
        return '\n'.join([f"• {con}" for con in cons[:5]])  # Limit to 5 cons
    
    @staticmethod
    def _generate_verdict(casino_data: CasinoIntelligence) -> str:
        """Generate overall verdict from casino data"""
        rating = casino_data.overall_rating or 0
        casino_name = casino_data.casino_name
        
        if rating >= 8:
            return f"{casino_name} is an excellent choice for online casino enthusiasts, offering top-tier gaming experience and reliable service."
        elif rating >= 6:
            return f"{casino_name} provides a solid gaming experience with good features and acceptable service quality."
        elif rating >= 4:
            return f"{casino_name} is an average casino with some positive aspects but also areas for improvement."
        else:
            return f"{casino_name} has significant limitations and may not be suitable for most players."


# ============================================================================
# TAXONOMY MANAGER
# ============================================================================

class MTCasinoTaxonomyManager:
    """Manages MT Casino taxonomies and categories"""
    
    @staticmethod
    def determine_categories(casino_data: CasinoIntelligence, post_type: str) -> List[str]:
        """Determine appropriate categories based on casino data and post type"""
        categories = []
        
        if post_type == 'mt_listing':
            # Crypto support
            if casino_data.payments.cryptocurrency_support:
                categories.append('crypto-casino')
            
            # Live dealer games
            if casino_data.games.game_portfolio.live_dealer_games_count and casino_data.games.game_portfolio.live_dealer_games_count > 0:
                categories.append('live-casino')
            
            # Mobile optimization
            if casino_data.user_experience.mobile_compatibility:
                categories.append('mobile-casino')
            
            # New casino (less than 3 years)
            if casino_data.trustworthiness.reputation_metrics.years_in_operation and casino_data.trustworthiness.reputation_metrics.years_in_operation < 3:
                categories.append('new-casino')
        
        elif post_type == 'mt_bonus':
            bonus = casino_data.bonuses.welcome_bonus
            if bonus.bonus_type:
                bonus_type_lower = bonus.bonus_type.lower()
                if 'welcome' in bonus_type_lower:
                    categories.append('welcome-bonus')
                elif 'deposit' in bonus_type_lower and 'no' in bonus_type_lower:
                    categories.append('no-deposit')
                elif 'spin' in bonus_type_lower:
                    categories.append('free-spins')
                elif 'cashback' in bonus_type_lower:
                    categories.append('cashback')
        
        return categories[:3]  # Limit to 3 categories
    
    @staticmethod
    def determine_software_tags(casino_data: CasinoIntelligence) -> List[str]:
        """Determine software provider tags"""
        providers = casino_data.games.software_providers.primary_providers
        if not providers:
            return []
        
        # Map to taxonomy terms
        provider_map = {
            'NetEnt': 'netent',
            'Microgaming': 'microgaming',
            'Playtech': 'playtech',
            'Pragmatic Play': 'pragmatic-play',
            'Evolution Gaming': 'evolution-gaming'
        }
        
        tags = []
        for provider in providers[:5]:  # Limit to 5
            provider_str = provider.value if hasattr(provider, 'value') else str(provider)
            if provider_str in provider_map:
                tags.append(provider_map[provider_str])
        
        return tags
    
    @staticmethod
    def determine_payment_tags(casino_data: CasinoIntelligence) -> List[str]:
        """Determine payment method tags"""
        if not casino_data.payments.payment_methods:
            return []
        
        payment_map = {
            'Bitcoin': 'bitcoin',
            'Ethereum': 'ethereum',
            'Visa': 'visa',
            'Mastercard': 'mastercard',
            'PayPal': 'paypal',
            'Skrill': 'skrill'
        }
        
        tags = []
        for method in casino_data.payments.payment_methods[:5]:  # Limit to 5
            if method.name in payment_map:
                tags.append(payment_map[method.name])
        
        return tags


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
        self.taxonomy_manager = MTCasinoTaxonomyManager()
    
    async def publish_mt_casino_content(
        self,
        content: str,
        title: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None,
        generate_related_posts: bool = True
    ) -> Dict[str, Any]:
        """
        Publish content using MT Casino post types with intelligent mapping
        
        Args:
            content: The main content to publish
            title: Post title
            casino_data: 95-field casino intelligence data
            images: List of image URLs
            generate_related_posts: Whether to generate related posts (bonus, slots, etc.)
        
        Returns:
            Dict with publishing results
        """
        start_time = datetime.now()
        results = {
            'success': False,
            'main_post': None,
            'related_posts': [],
            'errors': [],
            'processing_time': 0,
            'mt_casino_features_used': []
        }
        
        try:
            logger.info(f"🎰 Starting MT Casino publishing for: {title}")
            
            # Determine primary post type
            primary_post_type = self.content_analyzer.analyze_content_type(content, casino_data)
            logger.info(f"📊 Detected content type: {primary_post_type}")
            
            # Publish main post
            main_post_result = await self._publish_single_mt_casino_post(
                content=content,
                title=title,
                post_type=primary_post_type,
                casino_data=casino_data,
                images=images
            )
            
            results['main_post'] = main_post_result
            results['mt_casino_features_used'].append(f"Main post: {primary_post_type}")
            
            # Generate related posts if requested and casino data available
            if generate_related_posts and casino_data and main_post_result.get('success'):
                related_posts = await self._generate_related_posts(
                    casino_data=casino_data,
                    main_post_id=main_post_result.get('post_id'),
                    primary_post_type=primary_post_type
                )
                results['related_posts'] = related_posts
                results['mt_casino_features_used'].extend([f"Related: {post['post_type']}" for post in related_posts if post.get('success')])
            
            # Calculate overall success
            results['success'] = main_post_result.get('success', False)
            
            logger.info(f"✅ MT Casino publishing completed: {len(results['related_posts'])} related posts created")
            
        except Exception as e:
            error_msg = f"MT Casino publishing failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        finally:
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    async def _publish_single_mt_casino_post(
        self,
        content: str,
        title: str,
        post_type: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Publish a single MT Casino post with proper metadata and taxonomies"""
        
        try:
            # Prepare post data
            post_data = {
                'title': title,
                'content': content,
                'status': 'draft',  # Start as draft for review
                'type': post_type
            }
            
            # Add MT Casino custom fields if casino data available
            if casino_data:
                custom_fields = self._get_custom_fields_for_post_type(post_type, casino_data)
                if custom_fields:
                    post_data['meta'] = custom_fields
            
            # Add featured image if available
            if images and len(images) > 0:
                # Use first image as featured image
                post_data['featured_media'] = images[0]  # This would need proper media upload
            
            # Attempt to publish to MT Casino post type
            mt_casino_post_type = MT_CASINO_POST_TYPES.get(post_type)
            if mt_casino_post_type:
                endpoint = mt_casino_post_type.endpoint
                
                # Try MT Casino post type first
                response = await self._make_wp_request('POST', endpoint, json=post_data)
                
                if response and response.get('id'):
                    post_id = response['id']
                    
                    # Set taxonomies if casino data available
                    if casino_data:
                        await self._set_mt_casino_taxonomies(post_id, post_type, casino_data)
                    
                    logger.info(f"✅ Published to {post_type}: Post ID {post_id}")
                    return {
                        'success': True,
                        'post_id': post_id,
                        'post_type': post_type,
                        'url': response.get('link'),
                        'method': 'mt_casino_post_type'
                    }
            
            # Fallback to regular post with MT Casino styling
            logger.warning(f"⚠️ MT Casino post type {post_type} not available, falling back to regular post")
            return await self._publish_fallback_post(content, title, post_type, casino_data, images)
            
        except Exception as e:
            logger.error(f"❌ Failed to publish {post_type}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'post_type': post_type
            }
    
    async def _publish_fallback_post(
        self,
        content: str,
        title: str,
        intended_post_type: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Publish as regular post with MT Casino styling and metadata"""
        
        # Enhance content with MT Casino styling
        enhanced_content = self._add_mt_casino_styling(content, intended_post_type, casino_data)
        
        # Create regular post
        post_data = {
            'title': f"[{intended_post_type.upper()}] {title}",
            'content': enhanced_content,
            'status': 'draft',
            'categories': [1],  # Default category
            'tags': []
        }
        
        # Add custom fields to simulate MT Casino data
        if casino_data:
            custom_fields = self._get_custom_fields_for_post_type(intended_post_type, casino_data)
            if custom_fields:
                post_data['meta'] = custom_fields
        
        try:
            # Publish regular post
            response = await self._make_wp_request('POST', '/wp-json/wp/v2/posts', json=post_data)
            
            if response and response.get('id'):
                post_id = response['id']
                logger.info(f"✅ Published fallback post: Post ID {post_id}")
                return {
                    'success': True,
                    'post_id': post_id,
                    'post_type': 'post',
                    'intended_type': intended_post_type,
                    'url': response.get('link'),
                    'method': 'fallback_with_styling'
                }
        
        except Exception as e:
            logger.error(f"❌ Fallback publishing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'method': 'fallback_failed'
            }
    
    def _get_custom_fields_for_post_type(self, post_type: str, casino_data: CasinoIntelligence) -> Dict[str, Any]:
        """Get appropriate custom fields based on post type"""
        
        mapper_methods = {
            'mt_listing': self.metadata_mapper.map_to_mt_listing,
            'mt_bonus': self.metadata_mapper.map_to_mt_bonus,
            'mt_slots': self.metadata_mapper.map_to_mt_slots,
            'mt_reviews': self.metadata_mapper.map_to_mt_reviews
        }
        
        mapper_method = mapper_methods.get(post_type)
        if mapper_method:
            return mapper_method(casino_data)
        
        return {}
    
    async def _set_mt_casino_taxonomies(self, post_id: int, post_type: str, casino_data: CasinoIntelligence):
        """Set MT Casino taxonomies for the post"""
        
        try:
            # Determine categories
            categories = self.taxonomy_manager.determine_categories(casino_data, post_type)
            if categories:
                # This would require checking if taxonomies exist and creating terms
                logger.info(f"📁 Would set categories for post {post_id}: {categories}")
            
            # Determine software tags
            software_tags = self.taxonomy_manager.determine_software_tags(casino_data)
            if software_tags:
                logger.info(f"🏷️ Would set software tags for post {post_id}: {software_tags}")
            
            # Determine payment tags
            payment_tags = self.taxonomy_manager.determine_payment_tags(casino_data)
            if payment_tags:
                logger.info(f"💳 Would set payment tags for post {post_id}: {payment_tags}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to set taxonomies for post {post_id}: {str(e)}")
    
    def _add_mt_casino_styling(self, content: str, post_type: str, casino_data: Optional[CasinoIntelligence] = None) -> str:
        """Add MT Casino-style formatting to content for fallback posts"""
        
        styled_content = f'<div class="mt-casino-{post_type} mt-casino-styled">\n\n'
        
        # Add type indicator
        type_names = {
            'mt_listing': 'Casino Review',
            'mt_bonus': 'Casino Bonus',
            'mt_slots': 'Slot Games',
            'mt_reviews': 'Expert Review'
        }
        
        type_name = type_names.get(post_type, 'Casino Content')
        styled_content += f'<div class="mt-casino-header">\n'
        styled_content += f'<span class="mt-casino-type-badge">{type_name}</span>\n'
        styled_content += f'</div>\n\n'
        
        # Add casino data summary if available
        if casino_data:
            styled_content += self._create_casino_data_summary(casino_data, post_type)
        
        # Add main content
        styled_content += content
        
        # Add disclaimer
        styled_content += '\n\n<div class="mt-casino-disclaimer">\n'
        styled_content += '<p><small>This content is styled for MT Casino compatibility. For full MT Casino features, ensure the MT Casino plugin is installed and activated.</small></p>\n'
        styled_content += '</div>\n\n'
        
        styled_content += '</div>'
        
        return styled_content
    
    def _create_casino_data_summary(self, casino_data: CasinoIntelligence, post_type: str) -> str:
        """Create a summary box with key casino data"""
        
        summary = '<div class="mt-casino-summary">\n'
        summary += f'<h3>Quick Facts - {casino_data.casino_name}</h3>\n'
        summary += '<ul class="mt-casino-facts">\n'
        
        if post_type == 'mt_listing':
            if casino_data.overall_rating:
                summary += f'<li><strong>Rating:</strong> {casino_data.overall_rating}/10</li>\n'
            if casino_data.trustworthiness.license_info.primary_license:
                license_name = self.metadata_mapper._format_license(casino_data.trustworthiness.license_info.primary_license)
                summary += f'<li><strong>License:</strong> {license_name}</li>\n'
            if casino_data.bonuses.welcome_bonus.bonus_amount:
                summary += f'<li><strong>Welcome Bonus:</strong> {casino_data.bonuses.welcome_bonus.bonus_amount}</li>\n'
            if casino_data.games.game_portfolio.total_games:
                summary += f'<li><strong>Total Games:</strong> {casino_data.games.game_portfolio.total_games}</li>\n'
        
        elif post_type == 'mt_bonus':
            bonus = casino_data.bonuses.welcome_bonus
            if bonus.bonus_amount:
                summary += f'<li><strong>Bonus Amount:</strong> {bonus.bonus_amount}</li>\n'
            if bonus.wagering_requirements:
                summary += f'<li><strong>Wagering:</strong> {bonus.wagering_requirements}</li>\n'
            if bonus.free_spins_count:
                summary += f'<li><strong>Free Spins:</strong> {bonus.free_spins_count}</li>\n'
        
        elif post_type == 'mt_slots':
            games = casino_data.games
            if games.game_portfolio.slot_games_count:
                summary += f'<li><strong>Slot Games:</strong> {games.game_portfolio.slot_games_count}</li>\n'
            if games.software_providers.primary_providers:
                providers = self.metadata_mapper._format_software_providers(games.software_providers.primary_providers)
                summary += f'<li><strong>Providers:</strong> {providers}</li>\n'
        
        summary += '</ul>\n'
        summary += '</div>\n\n'
        
        return summary
    
    async def _generate_related_posts(
        self,
        casino_data: CasinoIntelligence,
        main_post_id: int,
        primary_post_type: str
    ) -> List[Dict[str, Any]]:
        """Generate related posts (bonus, slots, etc.) based on casino data"""
        
        related_posts = []
        
        # Generate bonus post if main post is not bonus and we have bonus data
        if primary_post_type != 'mt_bonus' and casino_data.bonuses.welcome_bonus.bonus_amount:
            bonus_content = self._generate_bonus_content(casino_data)
            bonus_result = await self._publish_single_mt_casino_post(
                content=bonus_content,
                title=f"{casino_data.casino_name} Welcome Bonus Review",
                post_type='mt_bonus',
                casino_data=casino_data
            )
            related_posts.append(bonus_result)
        
        # Generate slots post if main post is not slots and we have game data
        if primary_post_type != 'mt_slots' and casino_data.games.game_portfolio.slot_games_count:
            slots_content = self._generate_slots_content(casino_data)
            slots_result = await self._publish_single_mt_casino_post(
                content=slots_content,
                title=f"{casino_data.casino_name} Slot Games Collection",
                post_type='mt_slots',
                casino_data=casino_data
            )
            related_posts.append(slots_result)
        
        return related_posts
    
    def _generate_bonus_content(self, casino_data: CasinoIntelligence) -> str:
        """Generate content focused on bonuses"""
        bonus = casino_data.bonuses.welcome_bonus
        content = f"## {casino_data.casino_name} Welcome Bonus\n\n"
        
        if bonus.bonus_amount:
            content += f"The welcome bonus at {casino_data.casino_name} offers up to {bonus.bonus_amount}.\n\n"
        
        if bonus.bonus_type:
            content += f"**Bonus Type:** {bonus.bonus_type}\n\n"
        
        if bonus.wagering_requirements:
            content += f"**Wagering Requirements:** {bonus.wagering_requirements}\n\n"
        
        if bonus.free_spins_count:
            content += f"**Free Spins:** {bonus.free_spins_count} free spins included\n\n"
        
        content += "### How to Claim\n\n"
        content += f"1. Register at {casino_data.casino_name}\n"
        content += "2. Make your first deposit\n"
        content += "3. Bonus is automatically credited\n\n"
        
        content += "### Terms and Conditions\n\n"
        if bonus.minimum_deposit:
            content += f"- Minimum deposit: {bonus.minimum_deposit}\n"
        if bonus.time_limit:
            content += f"- Time limit: {bonus.time_limit}\n"
        if bonus.game_restrictions:
            content += f"- Game restrictions: {', '.join(bonus.game_restrictions[:3])}\n"
        
        return content
    
    def _generate_slots_content(self, casino_data: CasinoIntelligence) -> str:
        """Generate content focused on slot games"""
        games = casino_data.games
        content = f"## {casino_data.casino_name} Slot Games\n\n"
        
        if games.game_portfolio.slot_games_count:
            content += f"{casino_data.casino_name} features over {games.game_portfolio.slot_games_count} slot games.\n\n"
        
        content += "### Software Providers\n\n"
        if games.software_providers.primary_providers:
            providers = [provider.value if hasattr(provider, 'value') else str(provider) for provider in games.software_providers.primary_providers]
            content += f"Games are powered by top providers including {', '.join(providers)}.\n\n"
        
        if games.game_portfolio.popular_slot_titles:
            content += "### Popular Slots\n\n"
            for title in games.game_portfolio.popular_slot_titles[:5]:
                content += f"- {title}\n"
            content += "\n"
        
        if games.game_portfolio.progressive_jackpot_count:
            content += f"### Progressive Jackpots\n\n"
            content += f"The casino offers {games.game_portfolio.progressive_jackpot_count} progressive jackpot slots.\n\n"
        
        content += "### Features\n\n"
        if games.demo_mode_available:
            content += "- Demo mode available for all slots\n"
        if games.mobile_game_optimization:
            content += "- Mobile-optimized gameplay\n"
        if games.search_and_filter_functionality:
            content += "- Advanced search and filter options\n"
        
        return content


# ============================================================================
# INTEGRATION CLASS
# ============================================================================

class CoinflipMTCasinoIntegration:
    """Main integration class for Coinflip MT Casino publishing"""
    
    def __init__(self, wp_config: WordPressConfig):
        """Initialize with WordPress configuration"""
        self.publisher = CoinflipMTCasinoPublisher(wp_config)
    
    async def publish_casino_content(
        self,
        content: str,
        title: str,
        casino_data: Optional[CasinoIntelligence] = None,
        images: Optional[List[str]] = None,
        multi_post_strategy: bool = True
    ) -> Dict[str, Any]:
        """
        Main method to publish casino content with full MT Casino integration
        
        Args:
            content: Generated content from Universal RAG Chain
            title: Content title
            casino_data: 95-field casino intelligence data
            images: List of image URLs
            multi_post_strategy: Generate multiple related posts
        
        Returns:
            Comprehensive publishing results
        """
        return await self.publisher.publish_mt_casino_content(
            content=content,
            title=title,
            casino_data=casino_data,
            images=images,
            generate_related_posts=multi_post_strategy
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_coinflip_mt_casino_publisher(
    site_url: str,
    username: str,
    application_password: str
) -> CoinflipMTCasinoIntegration:
    """
    Factory function to create Coinflip MT Casino publisher
    
    Args:
        site_url: WordPress site URL
        username: WordPress username
        application_password: WordPress application password
    
    Returns:
        CoinflipMTCasinoIntegration instance ready for publishing
    """
    wp_config = WordPressConfig(
        site_url=site_url,
        username=username,
        application_password=application_password
    )
    
    return CoinflipMTCasinoIntegration(wp_config)

# Example usage
async def example_coinflip_publishing():
    """
    Example of how to use the Coinflip WordPress publisher
    """
    # Initialize publisher
    publisher = create_coinflip_mt_casino_publisher(
        site_url="https://your-coinflip-site.com",
        username="admin",
        application_password="your-app-password"
    )
    
    # Assume we have casino intelligence data
    # casino_intelligence = CasinoIntelligence(...)
    # content = "Generated casino review content..."
    # images = [{'id': 123, 'url': 'image.jpg'}]
    
    # Publish as MT Casino
    # result = await publisher.publish_casino_content(
    #     content=content,
    #     title=title,
    #     casino_data=casino_intelligence,
    #     images=images
    # )
    
    # print(f"Published: {result}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_coinflip_publishing()) 