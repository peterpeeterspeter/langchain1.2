#!/usr/bin/env python3
"""
🎰 ENHANCED CASINO WORDPRESS PUBLISHER
=====================================

Enterprise-grade WordPress publishing system specifically designed for casino reviews
using the comprehensive 95-field casino intelligence data structure.

This module implements TASK 18: Enhanced WordPress Casino Publishing System
Integrates with:
- Task 17.1: 95-Field Casino Intelligence Schema  
- Existing WordPress REST API Publisher
- LangChain components for content generation

Features:
- Converts 95-field casino intelligence to WordPress-compatible metadata
- Generates SEO-optimized casino review content using LangChain
- Manages crash casino site categorization and taxonomy
- Enhanced HTML formatting with casino-specific structured content
- LCEL workflows for complete publishing pipeline
- Media asset management for casino images and branding

Author: AI Assistant  
Created: 2025-01-20
Version: 1.0.0
"""

import asyncio
import logging
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

# LangChain imports for content generation
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Local imports
from ..schemas.casino_intelligence_schema import (
    CasinoIntelligence, 
    LicenseAuthority, 
    CurrencyCode,
    PaymentMethodType,
    GameProvider
)
from .wordpress_publisher import (
    WordPressRESTPublisher, 
    WordPressConfig, 
    RichHTMLFormatter
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SUBTASK 18.1: CASINO INTELLIGENCE DATA CONNECTOR
# ============================================================================

@dataclass
class CasinoReviewMetadata:
    """WordPress-compatible metadata structure for casino reviews"""
    
    # Core casino information
    casino_name: str
    casino_url: Optional[str] = None
    overall_rating: Optional[float] = None
    extraction_timestamp: Optional[datetime] = None
    
    # Trustworthiness metadata
    primary_license: Optional[str] = None
    license_authorities: List[str] = field(default_factory=list)
    safety_score: Optional[float] = None
    years_in_operation: Optional[int] = None
    
    # Games metadata  
    total_games: Optional[int] = None
    software_providers: List[str] = field(default_factory=list)
    live_dealer_available: bool = False
    mobile_optimized: bool = False
    
    # Bonuses metadata
    welcome_bonus_amount: Optional[str] = None
    welcome_bonus_percentage: Optional[int] = None
    free_spins_count: Optional[int] = None
    wagering_requirements: Optional[str] = None
    
    # Payments metadata
    withdrawal_time: Optional[str] = None
    min_deposit: Optional[str] = None
    max_withdrawal: Optional[str] = None
    payment_methods_count: Optional[int] = None
    crypto_supported: bool = False
    
    # User experience metadata
    live_chat_available: bool = False
    support_24_7: bool = False
    website_rating: Optional[float] = None
    
    # Innovation metadata
    vr_gaming: bool = False
    blockchain_integration: bool = False
    provably_fair: bool = False
    
    # SEO metadata
    meta_description: Optional[str] = None
    focus_keywords: List[str] = field(default_factory=list)
    category_ids: List[int] = field(default_factory=list)
    tag_names: List[str] = field(default_factory=list)
    
    # Custom fields for WordPress
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_wordpress_custom_fields(self) -> Dict[str, Any]:
        """Convert metadata to WordPress custom fields format"""
        fields = {}
        
        # Basic info fields
        if self.casino_url:
            fields['casino_url'] = self.casino_url
        if self.overall_rating is not None:
            fields['overall_rating'] = str(self.overall_rating)
        if self.extraction_timestamp:
            fields['last_updated'] = self.extraction_timestamp.isoformat()
            
        # Trustworthiness fields
        if self.primary_license:
            fields['primary_license'] = self.primary_license
        if self.license_authorities:
            fields['license_authorities'] = json.dumps(self.license_authorities)
        if self.safety_score is not None:
            fields['safety_score'] = str(self.safety_score)
        if self.years_in_operation is not None:
            fields['years_in_operation'] = str(self.years_in_operation)
            
        # Games fields
        if self.total_games is not None:
            fields['total_games'] = str(self.total_games)
        if self.software_providers:
            fields['software_providers'] = json.dumps(self.software_providers)
        fields['live_dealer_available'] = 'yes' if self.live_dealer_available else 'no'
        fields['mobile_optimized'] = 'yes' if self.mobile_optimized else 'no'
        
        # Bonuses fields
        if self.welcome_bonus_amount:
            fields['welcome_bonus_amount'] = self.welcome_bonus_amount
        if self.welcome_bonus_percentage is not None:
            fields['welcome_bonus_percentage'] = str(self.welcome_bonus_percentage)
        if self.free_spins_count is not None:
            fields['free_spins_count'] = str(self.free_spins_count)
        if self.wagering_requirements:
            fields['wagering_requirements'] = self.wagering_requirements
            
        # Payments fields
        if self.withdrawal_time:
            fields['withdrawal_time'] = self.withdrawal_time
        if self.min_deposit:
            fields['min_deposit'] = self.min_deposit
        if self.max_withdrawal:
            fields['max_withdrawal'] = self.max_withdrawal
        if self.payment_methods_count is not None:
            fields['payment_methods_count'] = str(self.payment_methods_count)
        fields['crypto_supported'] = 'yes' if self.crypto_supported else 'no'
        
        # Support fields
        fields['live_chat_available'] = 'yes' if self.live_chat_available else 'no'
        fields['support_24_7'] = 'yes' if self.support_24_7 else 'no'
        if self.website_rating is not None:
            fields['website_rating'] = str(self.website_rating)
            
        # Innovation fields
        fields['vr_gaming'] = 'yes' if self.vr_gaming else 'no'
        fields['blockchain_integration'] = 'yes' if self.blockchain_integration else 'no'
        fields['provably_fair'] = 'yes' if self.provably_fair else 'no'
        
        # Add any additional custom fields
        fields.update(self.custom_fields)
        
        return fields


class CasinoIntelligenceDataConnector:
    """
    SUBTASK 18.1: Casino Intelligence Data Connector
    
    Connects to, validates, and transforms the 95-field casino intelligence data 
    from Task 17 for WordPress publishing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".CasinoIntelligenceDataConnector")
        self._validation_errors = []
        self._transformation_stats = {}
        
    def validate_casino_intelligence(self, casino_data: CasinoIntelligence) -> Tuple[bool, List[str]]:
        """
        Validate casino intelligence data before processing
        
        Args:
            casino_data: CasinoIntelligence object to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required field validation
        if not casino_data.casino_name or casino_data.casino_name.strip() == "":
            errors.append("Casino name is required and cannot be empty")
            
        # URL validation
        if casino_data.casino_url:
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(casino_data.casino_url):
                errors.append(f"Invalid casino URL format: {casino_data.casino_url}")
        
        # Rating validation
        if casino_data.overall_rating is not None:
            if not (0 <= casino_data.overall_rating <= 10):
                errors.append(f"Overall rating must be between 0-10, got: {casino_data.overall_rating}")
                
        # Safety score validation
        if casino_data.safety_score is not None:
            if not (0 <= casino_data.safety_score <= 10):
                errors.append(f"Safety score must be between 0-10, got: {casino_data.safety_score}")
        
        # License validation
        if (casino_data.trustworthiness.license_info.primary_license and 
            casino_data.trustworthiness.license_info.primary_license not in LicenseAuthority):
            errors.append(f"Invalid primary license authority: {casino_data.trustworthiness.license_info.primary_license}")
            
        # Game count validation
        if casino_data.games.game_portfolio.total_games is not None:
            if casino_data.games.game_portfolio.total_games < 0:
                errors.append("Total games count cannot be negative")
                
        # Bonus percentage validation
        if casino_data.bonuses.welcome_bonus.bonus_percentage is not None:
            if casino_data.bonuses.welcome_bonus.bonus_percentage < 0:
                errors.append("Bonus percentage cannot be negative")
        
        self._validation_errors = errors
        return len(errors) == 0, errors
    
    def sanitize_casino_data(self, casino_data: CasinoIntelligence) -> CasinoIntelligence:
        """
        Sanitize casino data to prevent injection attacks and clean formatting
        
        Args:
            casino_data: Raw casino intelligence data
            
        Returns:
            Sanitized casino intelligence data
        """
        
        def sanitize_string(value: Optional[str]) -> Optional[str]:
            """Sanitize individual string values"""
            if not value:
                return value
                
            # Remove potential HTML/script tags
            value = re.sub(r'<[^>]*>', '', str(value))
            
            # Remove potential SQL injection characters
            value = re.sub(r'[;\'\"\\]', '', value)
            
            # Clean excessive whitespace
            value = ' '.join(value.split())
            
            return value.strip() if value else None
        
        def sanitize_list(value_list: List[str]) -> List[str]:
            """Sanitize list of strings"""
            if not value_list:
                return []
            return [sanitize_string(item) for item in value_list if sanitize_string(item)]
        
        # Create a copy to avoid modifying original data
        sanitized_data = casino_data.copy(deep=True)
        
        # Sanitize core fields
        sanitized_data.casino_name = sanitize_string(casino_data.casino_name)
        sanitized_data.casino_url = sanitize_string(casino_data.casino_url)
        
        # Sanitize trustworthiness fields
        if casino_data.trustworthiness.reputation_metrics.parent_company:
            sanitized_data.trustworthiness.reputation_metrics.parent_company = sanitize_string(
                casino_data.trustworthiness.reputation_metrics.parent_company
            )
            
        # Sanitize list fields
        sanitized_data.trustworthiness.reputation_metrics.awards_and_certifications = sanitize_list(
            casino_data.trustworthiness.reputation_metrics.awards_and_certifications
        )
        
        sanitized_data.games.game_portfolio.popular_slot_titles = sanitize_list(
            casino_data.games.game_portfolio.popular_slot_titles
        )
        
        sanitized_data.games.software_providers.all_providers = sanitize_list(
            casino_data.games.software_providers.all_providers
        )
        
        self.logger.info(f"Sanitized casino data for: {sanitized_data.casino_name}")
        return sanitized_data
    
    def transform_to_wordpress_metadata(self, casino_data: CasinoIntelligence) -> CasinoReviewMetadata:
        """
        Transform 95-field casino intelligence to WordPress-compatible metadata
        
        Args:
            casino_data: Validated and sanitized casino intelligence
            
        Returns:
            CasinoReviewMetadata object ready for WordPress publishing
        """
        
        # Extract software providers
        software_providers = []
        if casino_data.games.software_providers.primary_providers:
            software_providers.extend([provider.value for provider in casino_data.games.software_providers.primary_providers])
        if casino_data.games.software_providers.all_providers:
            software_providers.extend(casino_data.games.software_providers.all_providers)
        # Remove duplicates while preserving order
        software_providers = list(dict.fromkeys(software_providers))
        
        # Extract license authorities
        license_authorities = []
        if casino_data.trustworthiness.license_info.primary_license:
            license_authorities.append(casino_data.trustworthiness.license_info.primary_license.value)
        if casino_data.trustworthiness.license_info.additional_licenses:
            license_authorities.extend([license.value for license in casino_data.trustworthiness.license_info.additional_licenses])
        license_authorities = list(dict.fromkeys(license_authorities))
        
        # Generate focus keywords for SEO
        focus_keywords = [casino_data.casino_name.lower()]
        if casino_data.trustworthiness.license_info.primary_license:
            if "malta" in casino_data.trustworthiness.license_info.primary_license.value.lower():
                focus_keywords.extend(["malta casino", "mga licensed"])
            elif "uk" in casino_data.trustworthiness.license_info.primary_license.value.lower():
                focus_keywords.extend(["uk casino", "ukgc licensed"])
        
        # Add game-related keywords
        if casino_data.games.game_portfolio.total_games and casino_data.games.game_portfolio.total_games > 1000:
            focus_keywords.append("large game selection")
        if casino_data.games.game_portfolio.live_dealer_games_count and casino_data.games.game_portfolio.live_dealer_games_count > 0:
            focus_keywords.append("live dealer casino")
        if casino_data.payments.cryptocurrency_support:
            focus_keywords.append("crypto casino")
            
        # Generate meta description
        meta_description = self._generate_meta_description(casino_data)
        
        # Determine default tag names for crash casino categorization
        tag_names = ["casino review", "online casino"]
        if casino_data.games.game_portfolio.specialty_games_count and casino_data.games.game_portfolio.specialty_games_count > 0:
            tag_names.append("crash casino")
        if casino_data.bonuses.welcome_bonus.free_spins_count and casino_data.bonuses.welcome_bonus.free_spins_count > 0:
            tag_names.append("free spins casino")
        if casino_data.payments.cryptocurrency_support:
            tag_names.append("cryptocurrency casino")
            
        # Create the metadata object
        metadata = CasinoReviewMetadata(
            casino_name=casino_data.casino_name,
            casino_url=casino_data.casino_url,
            overall_rating=casino_data.overall_rating,
            extraction_timestamp=casino_data.extraction_timestamp,
            
            # Trustworthiness
            primary_license=casino_data.trustworthiness.license_info.primary_license.value if casino_data.trustworthiness.license_info.primary_license else None,
            license_authorities=license_authorities,
            safety_score=casino_data.safety_score,
            years_in_operation=casino_data.trustworthiness.reputation_metrics.years_in_operation,
            
            # Games
            total_games=casino_data.games.game_portfolio.total_games,
            software_providers=software_providers,
            live_dealer_available=bool(casino_data.games.game_portfolio.live_dealer_games_count and casino_data.games.game_portfolio.live_dealer_games_count > 0),
            mobile_optimized=casino_data.games.mobile_game_optimization,
            
            # Bonuses
            welcome_bonus_amount=casino_data.bonuses.welcome_bonus.bonus_amount,
            welcome_bonus_percentage=casino_data.bonuses.welcome_bonus.bonus_percentage,
            free_spins_count=casino_data.bonuses.welcome_bonus.free_spins_count,
            wagering_requirements=casino_data.bonuses.welcome_bonus.wagering_requirements,
            
            # Payments
            withdrawal_time=casino_data.payments.withdrawal_processing_time,
            min_deposit=casino_data.payments.minimum_deposit_amount,
            max_withdrawal=casino_data.payments.withdrawal_limits_daily,
            payment_methods_count=casino_data.payments.payment_method_count,
            crypto_supported=casino_data.payments.cryptocurrency_support,
            
            # User Experience
            live_chat_available=casino_data.user_experience.customer_support.live_chat_available,
            support_24_7=casino_data.user_experience.customer_support.support_24_7,
            website_rating=casino_data.user_experience.website_design_rating,
            
            # Innovation
            vr_gaming=casino_data.innovations.vr_gaming_support,
            blockchain_integration=casino_data.innovations.blockchain_integration,
            provably_fair=casino_data.innovations.provably_fair_games,
            
            # SEO
            meta_description=meta_description,
            focus_keywords=focus_keywords,
            tag_names=tag_names
        )
        
        # Track transformation statistics
        self._transformation_stats = {
            'fields_transformed': len(metadata.to_wordpress_custom_fields()),
            'software_providers_count': len(software_providers),
            'license_authorities_count': len(license_authorities),
            'focus_keywords_count': len(focus_keywords),
            'transformation_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Transformed casino data for '{casino_data.casino_name}' - {self._transformation_stats['fields_transformed']} custom fields created")
        
        return metadata
    
    def _generate_meta_description(self, casino_data: CasinoIntelligence) -> str:
        """Generate SEO-optimized meta description"""
        description_parts = []
        
        # Start with casino name and basic info
        description_parts.append(f"{casino_data.casino_name}")
        
        # Add licensing info if available
        if casino_data.trustworthiness.license_info.primary_license:
            license_short = casino_data.trustworthiness.license_info.primary_license.value.split('(')[1].replace(')', '') if '(' in casino_data.trustworthiness.license_info.primary_license.value else ""
            if license_short:
                description_parts.append(f"{license_short} licensed casino")
            else:
                description_parts.append("licensed casino")
        
        # Add game count if significant
        if casino_data.games.game_portfolio.total_games and casino_data.games.game_portfolio.total_games >= 500:
            description_parts.append(f"with {casino_data.games.game_portfolio.total_games}+ games")
        
        # Add welcome bonus if available
        if casino_data.bonuses.welcome_bonus.bonus_amount:
            description_parts.append(f"Welcome bonus: {casino_data.bonuses.welcome_bonus.bonus_amount}")
            
        # Add key features
        features = []
        if casino_data.games.game_portfolio.live_dealer_games_count and casino_data.games.game_portfolio.live_dealer_games_count > 0:
            features.append("live dealer games")
        if casino_data.payments.cryptocurrency_support:
            features.append("crypto payments")
        if casino_data.user_experience.customer_support.support_24_7:
            features.append("24/7 support")
            
        if features:
            description_parts.append("Features: " + ", ".join(features))
        
        # Combine and truncate to recommended length
        description = " | ".join(description_parts)
        if len(description) > 155:  # SEO recommended max length
            description = description[:152] + "..."
            
        return description
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about the last transformation operation"""
        return self._transformation_stats.copy()
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors from the last validation operation"""
        return self._validation_errors.copy() 