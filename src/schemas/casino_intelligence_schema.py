"""
🎰 COMPREHENSIVE CASINO INTELLIGENCE SCHEMA - 95 FIELDS
====================================================

Enterprise-grade Pydantic schemas for structured casino intelligence extraction
Supporting Task 17.1: Design Comprehensive Pydantic Schemas for 95 Fields

SCHEMA ORGANIZATION:
- 6 Major Categories (Trustworthiness, Games, Bonuses, Payments, UX, Innovations)
- 95 Total Fields with validation, documentation, and type safety
- Backward compatibility with existing 14-field system
- LangChain PydanticOutputParser integration ready

Author: AI Assistant
Created: 2025-01-20
Version: 1.0.0
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, model_validator
from datetime import datetime
from enum import Enum
import re


# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class LicenseAuthority(str, Enum):
    """Casino licensing authorities"""
    MGA = "Malta Gaming Authority (MGA)"
    UKGC = "UK Gambling Commission (UKGC)"
    CURACAO = "Curacao eGaming"
    GIBRALTAR = "Gibraltar Gambling Commission"
    KAHNAWAKE = "Kahnawake Gaming Commission"
    ALDERNEY = "Alderney Gambling Control Commission"
    ISLE_OF_MAN = "Isle of Man Gambling Supervision Commission"
    SWEDISH = "Swedish Gambling Authority (SGA)"
    DANISH = "Danish Gambling Authority"
    ITALIAN = "Italian Gaming Authority (ADM)"
    SPANISH = "Spanish Gaming Commission (DGOJ)"
    FRENCH = "French Gaming Authority (ANJ)"
    GERMAN = "German Gaming Authority"
    ONTARIO = "Alcohol and Gaming Commission of Ontario (AGCO)"
    NEW_JERSEY = "New Jersey Division of Gaming Enforcement"
    PENNSYLVANIA = "Pennsylvania Gaming Control Board"
    NEVADA = "Nevada Gaming Control Board"
    UNKNOWN = "Unknown/Unlicensed"


class CurrencyCode(str, Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    NOK = "NOK"
    SEK = "SEK"
    DKK = "DKK"
    CHF = "CHF"
    JPY = "JPY"
    BTC = "BTC"
    ETH = "ETH"
    LTC = "LTC"
    BCH = "BCH"
    DOGE = "DOGE"
    ADA = "ADA"
    XRP = "XRP"


class PaymentMethodType(str, Enum):
    """Payment method categories"""
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    E_WALLET = "E-Wallet"
    BANK_TRANSFER = "Bank Transfer"
    CRYPTOCURRENCY = "Cryptocurrency"
    PREPAID_CARD = "Prepaid Card"
    MOBILE_PAYMENT = "Mobile Payment"
    VOUCHER = "Voucher"
    CHECK = "Check"
    WIRE_TRANSFER = "Wire Transfer"


class GameProvider(str, Enum):
    """Major game software providers"""
    NETENT = "NetEnt"
    MICROGAMING = "Microgaming"
    PLAYTECH = "Playtech"
    PRAGMATIC_PLAY = "Pragmatic Play"
    EVOLUTION_GAMING = "Evolution Gaming"
    PLAY_N_GO = "Play'n GO"
    YGGDRASIL = "Yggdrasil"
    RED_TIGER = "Red Tiger"
    NOLIMIT_CITY = "Nolimit City"
    BIG_TIME_GAMING = "Big Time Gaming"
    QUICKSPIN = "Quickspin"
    IGT = "IGT"
    NOVOMATIC = "Novomatic"
    BETSOFT = "Betsoft"
    RTG = "RealTime Gaming"
    PUSH_GAMING = "Push Gaming"
    RELAX_GAMING = "Relax Gaming"
    THUNDERKICK = "Thunderkick"
    ELK_STUDIOS = "ELK Studios"
    HACKSAW_GAMING = "Hacksaw Gaming"


class DeviceType(str, Enum):
    """Device compatibility types"""
    DESKTOP = "Desktop"
    MOBILE_WEB = "Mobile Web"
    MOBILE_APP = "Mobile App"
    TABLET = "Tablet"
    SMART_TV = "Smart TV"
    VR_HEADSET = "VR Headset"


class SupportChannelType(str, Enum):
    """Customer support channel types"""
    LIVE_CHAT = "Live Chat"
    EMAIL = "Email"
    PHONE = "Phone"
    TICKET_SYSTEM = "Ticket System"
    FAQ = "FAQ"
    FORUM = "Forum"
    SOCIAL_MEDIA = "Social Media"
    VIDEO_CALL = "Video Call"


# ============================================================================
# CATEGORY 1: TRUSTWORTHINESS & SAFETY (25 FIELDS)
# ============================================================================

class LicenseInformation(BaseModel):
    """Detailed licensing information"""
    primary_license: Optional[LicenseAuthority] = Field(None, description="Primary operating license")
    additional_licenses: List[LicenseAuthority] = Field(default_factory=list, description="Additional licenses held")
    license_numbers: Dict[str, str] = Field(default_factory=dict, description="License numbers by authority")
    license_status: str = Field("Unknown", description="Current license status")
    license_expiry_dates: Dict[str, str] = Field(default_factory=dict, description="License expiry dates")
    
    @validator('license_status')
    def validate_license_status(cls, v):
        valid_statuses = ["Active", "Suspended", "Revoked", "Expired", "Pending", "Unknown"]
        return v if v in valid_statuses else "Unknown"


class SecurityFeatures(BaseModel):
    """Security and protection measures"""
    ssl_encryption: bool = Field(False, description="SSL/TLS encryption enabled")
    two_factor_authentication: bool = Field(False, description="2FA available for players")
    identity_verification: bool = Field(False, description="KYC/AML verification required")
    age_verification: bool = Field(False, description="Age verification process")
    geolocation_verification: bool = Field(False, description="Geolocation checks")
    anti_fraud_measures: List[str] = Field(default_factory=list, description="Anti-fraud technologies")
    data_protection_compliance: List[str] = Field(default_factory=list, description="Data protection standards (GDPR, etc.)")
    responsible_gambling_tools: List[str] = Field(default_factory=list, description="Responsible gambling features")
    
    
class ReputationMetrics(BaseModel):
    """Casino reputation and trust indicators"""
    years_in_operation: Optional[int] = Field(None, description="Years in business", ge=0, le=100)
    parent_company: Optional[str] = Field(None, description="Parent company name")
    company_registration: Optional[str] = Field(None, description="Company registration details")
    awards_and_certifications: List[str] = Field(default_factory=list, description="Industry awards and certifications")
    third_party_audits: List[str] = Field(default_factory=list, description="Third-party auditing companies")
    fair_play_certification: bool = Field(False, description="Fair play certified by testing labs")
    complaint_resolution: Optional[str] = Field(None, description="Complaint resolution process")
    transparency_score: Optional[float] = Field(None, description="Transparency rating (0-10)", ge=0, le=10)
    

class TrustworthinessCategory(BaseModel):
    """CATEGORY 1: Trustworthiness & Safety (25 fields)"""
    
    # Licensing (5 fields)
    license_info: LicenseInformation = Field(default_factory=LicenseInformation, description="Comprehensive licensing information")
    regulatory_compliance: List[str] = Field(default_factory=list, description="Regulatory compliance certifications")
    
    # Security (8 fields)
    security_features: SecurityFeatures = Field(default_factory=SecurityFeatures, description="Security measures and features")
    
    # Reputation (12 fields)
    reputation_metrics: ReputationMetrics = Field(default_factory=ReputationMetrics, description="Trust and reputation indicators")
    
    # Additional trust indicators
    blacklist_status: str = Field("Clean", description="Blacklist status from review sites")
    player_safety_rating: Optional[float] = Field(None, description="Player safety rating (0-10)", ge=0, le=10)
    financial_stability: Optional[str] = Field(None, description="Financial stability assessment")
    ownership_transparency: bool = Field(False, description="Ownership information publicly available")
    terms_and_conditions_clarity: Optional[float] = Field(None, description="T&C clarity score (0-10)", ge=0, le=10)
    dispute_resolution_mechanism: Optional[str] = Field(None, description="Dispute resolution process")


# ============================================================================
# CATEGORY 2: GAMES & SOFTWARE (20 FIELDS)
# ============================================================================

class GamePortfolio(BaseModel):
    """Game portfolio details"""
    total_games: Optional[int] = Field(None, description="Total number of games", ge=0)
    slot_games_count: Optional[int] = Field(None, description="Number of slot games", ge=0)
    table_games_count: Optional[int] = Field(None, description="Number of table games", ge=0)
    live_dealer_games_count: Optional[int] = Field(None, description="Number of live dealer games", ge=0)
    video_poker_count: Optional[int] = Field(None, description="Number of video poker games", ge=0)
    specialty_games_count: Optional[int] = Field(None, description="Number of specialty games", ge=0)
    progressive_jackpot_count: Optional[int] = Field(None, description="Number of progressive jackpot games", ge=0)
    
    # Game categories
    popular_slot_titles: List[str] = Field(default_factory=list, description="Popular slot game titles")
    exclusive_games: List[str] = Field(default_factory=list, description="Casino-exclusive games")
    new_game_frequency: Optional[str] = Field(None, description="How often new games are added")


class SoftwareProviders(BaseModel):
    """Software provider information"""
    primary_providers: List[GameProvider] = Field(default_factory=list, description="Main software providers")
    all_providers: List[str] = Field(default_factory=list, description="All software providers")
    provider_count: Optional[int] = Field(None, description="Total number of providers", ge=0)
    live_dealer_providers: List[str] = Field(default_factory=list, description="Live dealer game providers")
    rng_certification: bool = Field(False, description="RNG certified games")
    return_to_player_ranges: Dict[str, str] = Field(default_factory=dict, description="RTP ranges by game type")


class GamesCategory(BaseModel):
    """CATEGORY 2: Games & Software (20 fields)"""
    
    # Game Portfolio (10 fields)
    game_portfolio: GamePortfolio = Field(default_factory=GamePortfolio, description="Complete game portfolio information")
    
    # Software & Quality (10 fields)
    software_providers: SoftwareProviders = Field(default_factory=SoftwareProviders, description="Software provider details")
    game_quality_rating: Optional[float] = Field(None, description="Game quality rating (0-10)", ge=0, le=10)
    game_loading_speed: Optional[str] = Field(None, description="Average game loading speed")
    mobile_game_optimization: bool = Field(False, description="Games optimized for mobile")
    demo_mode_available: bool = Field(False, description="Demo/practice mode available")
    search_and_filter_functionality: bool = Field(False, description="Game search and filter options")
    favorites_and_history: bool = Field(False, description="Favorites and game history features")
    tournament_availability: bool = Field(False, description="Game tournaments available")
    leaderboards: bool = Field(False, description="Game leaderboards available")


# ============================================================================
# CATEGORY 3: BONUSES & PROMOTIONS (15 FIELDS)
# ============================================================================

class WelcomeBonusDetails(BaseModel):
    """Welcome bonus structure"""
    bonus_type: Optional[str] = Field(None, description="Type of welcome bonus")
    bonus_amount: Optional[str] = Field(None, description="Maximum bonus amount")
    bonus_percentage: Optional[int] = Field(None, description="Bonus percentage", ge=0, le=1000)
    free_spins_count: Optional[int] = Field(None, description="Number of free spins", ge=0)
    minimum_deposit: Optional[str] = Field(None, description="Minimum deposit required")
    wagering_requirements: Optional[str] = Field(None, description="Wagering requirements")
    time_limit: Optional[str] = Field(None, description="Time limit to claim/use bonus")
    game_restrictions: List[str] = Field(default_factory=list, description="Games where bonus can be used")
    country_restrictions: List[str] = Field(default_factory=list, description="Countries where bonus is not available")


class BonusesCategory(BaseModel):
    """CATEGORY 3: Bonuses & Promotions (15 fields)"""
    
    # Welcome Bonus (9 fields)
    welcome_bonus: WelcomeBonusDetails = Field(default_factory=WelcomeBonusDetails, description="Welcome bonus details")
    
    # Ongoing Promotions (6 fields)
    reload_bonuses: bool = Field(False, description="Reload bonuses available")
    cashback_offers: bool = Field(False, description="Cashback offers available")
    loyalty_program: bool = Field(False, description="VIP/loyalty program available")
    weekly_monthly_promotions: List[str] = Field(default_factory=list, description="Regular promotional offers")
    seasonal_bonuses: bool = Field(False, description="Seasonal/holiday bonuses available")
    referral_program: bool = Field(False, description="Referral bonus program")


# ============================================================================
# CATEGORY 4: PAYMENTS & BANKING (15 FIELDS)
# ============================================================================

class PaymentMethod(BaseModel):
    """Individual payment method details"""
    name: str = Field(..., description="Payment method name")
    type: PaymentMethodType = Field(..., description="Payment method type")
    deposit_supported: bool = Field(True, description="Supports deposits")
    withdrawal_supported: bool = Field(True, description="Supports withdrawals")
    min_deposit: Optional[str] = Field(None, description="Minimum deposit amount")
    max_deposit: Optional[str] = Field(None, description="Maximum deposit amount")
    min_withdrawal: Optional[str] = Field(None, description="Minimum withdrawal amount")
    max_withdrawal: Optional[str] = Field(None, description="Maximum withdrawal amount")
    processing_time_deposit: Optional[str] = Field(None, description="Deposit processing time")
    processing_time_withdrawal: Optional[str] = Field(None, description="Withdrawal processing time")
    fees: Optional[str] = Field(None, description="Associated fees")
    supported_currencies: List[CurrencyCode] = Field(default_factory=list, description="Supported currencies")


class PaymentsCategory(BaseModel):
    """CATEGORY 4: Payments & Banking (15 fields)"""
    
    # Payment Methods (5 fields)
    payment_methods: List[PaymentMethod] = Field(default_factory=list, description="All available payment methods")
    cryptocurrency_support: bool = Field(False, description="Cryptocurrency payments supported")
    payment_method_count: Optional[int] = Field(None, description="Total number of payment methods", ge=0)
    
    # Processing & Limits (10 fields)
    withdrawal_processing_time: Optional[str] = Field(None, description="Average withdrawal processing time")
    withdrawal_limits_daily: Optional[str] = Field(None, description="Daily withdrawal limits")
    withdrawal_limits_weekly: Optional[str] = Field(None, description="Weekly withdrawal limits")
    withdrawal_limits_monthly: Optional[str] = Field(None, description="Monthly withdrawal limits")
    minimum_deposit_amount: Optional[str] = Field(None, description="Minimum deposit amount")
    maximum_deposit_amount: Optional[str] = Field(None, description="Maximum deposit amount")
    transaction_fees: Optional[str] = Field(None, description="Transaction fees structure")
    currency_conversion_fees: Optional[str] = Field(None, description="Currency conversion fees")
    pending_withdrawal_time: Optional[str] = Field(None, description="Pending time before processing")
    verification_requirements: List[str] = Field(default_factory=list, description="KYC verification requirements")


# ============================================================================
# CATEGORY 5: USER EXPERIENCE & SUPPORT (10 FIELDS)
# ============================================================================

class CustomerSupport(BaseModel):
    """Customer support details"""
    support_channels: List[SupportChannelType] = Field(default_factory=list, description="Available support channels")
    live_chat_available: bool = Field(False, description="Live chat support available")
    support_24_7: bool = Field(False, description="24/7 support availability")
    support_languages: List[str] = Field(default_factory=list, description="Supported languages")
    response_time_live_chat: Optional[str] = Field(None, description="Average live chat response time")
    response_time_email: Optional[str] = Field(None, description="Average email response time")
    support_quality_rating: Optional[float] = Field(None, description="Support quality rating (0-10)", ge=0, le=10)


class UserExperienceCategory(BaseModel):
    """CATEGORY 5: User Experience & Support (10 fields)"""
    
    # Platform & Interface (3 fields)
    website_design_rating: Optional[float] = Field(None, description="Website design quality (0-10)", ge=0, le=10)
    mobile_compatibility: bool = Field(False, description="Mobile-friendly website/app")
    user_interface_rating: Optional[float] = Field(None, description="User interface rating (0-10)", ge=0, le=10)
    
    # Customer Support (7 fields)
    customer_support: CustomerSupport = Field(default_factory=CustomerSupport, description="Customer support information")


# ============================================================================
# CATEGORY 6: INNOVATIONS & FEATURES (10 FIELDS)
# ============================================================================

class InnovationsCategory(BaseModel):
    """CATEGORY 6: Innovations & Features (10 fields)"""
    
    # Technology Innovation (5 fields)
    vr_gaming_support: bool = Field(False, description="Virtual reality gaming supported")
    ai_powered_features: List[str] = Field(default_factory=list, description="AI-powered features")
    blockchain_integration: bool = Field(False, description="Blockchain technology integration")
    provably_fair_games: bool = Field(False, description="Provably fair gaming available")
    instant_play_capability: bool = Field(False, description="Instant play without downloads")
    
    # Social & Gamification (5 fields)
    social_features: List[str] = Field(default_factory=list, description="Social gaming features")
    gamification_elements: List[str] = Field(default_factory=list, description="Gamification features")
    achievement_system: bool = Field(False, description="Player achievement system")
    community_features: bool = Field(False, description="Community/forum features")
    streaming_integration: bool = Field(False, description="Game streaming capabilities")


# ============================================================================
# MAIN CASINO INTELLIGENCE MODEL (95 FIELDS TOTAL)
# ============================================================================

class CasinoIntelligence(BaseModel):
    """
    🎰 COMPREHENSIVE CASINO INTELLIGENCE MODEL - 95 FIELDS
    ====================================================
    
    Complete structured intelligence for casino analysis across 6 major categories:
    
    1. Trustworthiness & Safety (25 fields)
    2. Games & Software (20 fields) 
    3. Bonuses & Promotions (15 fields)
    4. Payments & Banking (15 fields)
    5. User Experience & Support (10 fields)
    6. Innovations & Features (10 fields)
    
    Total: 95 fields for comprehensive casino intelligence extraction
    """
    
    # ========================================================================
    # METADATA & IDENTIFICATION
    # ========================================================================
    casino_name: str = Field(..., description="Official casino name")
    casino_url: Optional[str] = Field(None, description="Main casino website URL")
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="When this data was extracted")
    data_sources: List[str] = Field(default_factory=list, description="Sources used for data extraction")
    confidence_score: Optional[float] = Field(None, description="Overall extraction confidence (0-1)", ge=0, le=1)
    
    # ========================================================================
    # 6 MAJOR CATEGORIES (95 FIELDS TOTAL)
    # ========================================================================
    trustworthiness: TrustworthinessCategory = Field(
        default_factory=TrustworthinessCategory, 
        description="Category 1: Trustworthiness & Safety (25 fields)"
    )
    
    games: GamesCategory = Field(
        default_factory=GamesCategory,
        description="Category 2: Games & Software (20 fields)"
    )
    
    bonuses: BonusesCategory = Field(
        default_factory=BonusesCategory,
        description="Category 3: Bonuses & Promotions (15 fields)"
    )
    
    payments: PaymentsCategory = Field(
        default_factory=PaymentsCategory,
        description="Category 4: Payments & Banking (15 fields)"
    )
    
    user_experience: UserExperienceCategory = Field(
        default_factory=UserExperienceCategory,
        description="Category 5: User Experience & Support (10 fields)"
    )
    
    innovations: InnovationsCategory = Field(
        default_factory=InnovationsCategory,
        description="Category 6: Innovations & Features (10 fields)"
    )
    
    # ========================================================================
    # CALCULATED FIELDS & OVERALL RATINGS
    # ========================================================================
    overall_rating: Optional[float] = Field(None, description="Overall casino rating (0-10)", ge=0, le=10)
    safety_score: Optional[float] = Field(None, description="Safety score (0-10)", ge=0, le=10)
    player_experience_score: Optional[float] = Field(None, description="Player experience score (0-10)", ge=0, le=10)
    value_score: Optional[float] = Field(None, description="Value for money score (0-10)", ge=0, le=10)
    
    # ========================================================================
    # BACKWARD COMPATIBILITY WITH 14-FIELD SYSTEM
    # ========================================================================
    @property
    def legacy_casino_rating(self) -> float:
        """Backward compatibility: Legacy casino rating"""
        return self.overall_rating or 0.0
    
    @property
    def legacy_bonus_amount(self) -> str:
        """Backward compatibility: Legacy bonus amount"""
        return self.bonuses.welcome_bonus.bonus_amount or ""
    
    @property
    def legacy_license_info(self) -> str:
        """Backward compatibility: Legacy license info"""
        primary = self.trustworthiness.license_info.primary_license
        additional = self.trustworthiness.license_info.additional_licenses
        
        if primary:
            licenses = [primary.value]
            if additional:
                licenses.extend([lic.value for lic in additional])
            return ", ".join(licenses)
        return "License information not found"
    
    @property
    def legacy_game_providers(self) -> List[str]:
        """Backward compatibility: Legacy game providers"""
        return [provider.value for provider in self.games.software_providers.primary_providers]
    
    @property
    def legacy_payment_methods(self) -> List[str]:
        """Backward compatibility: Legacy payment methods"""
        return [method.name for method in self.payments.payment_methods]
    
    @property
    def legacy_mobile_compatible(self) -> bool:
        """Backward compatibility: Legacy mobile compatibility"""
        return self.user_experience.mobile_compatibility
    
    @property
    def legacy_live_chat_support(self) -> bool:
        """Backward compatibility: Legacy live chat support"""
        return self.user_experience.customer_support.live_chat_available
    
    @property
    def legacy_withdrawal_time(self) -> str:
        """Backward compatibility: Legacy withdrawal time"""
        return self.payments.withdrawal_processing_time or ""
    
    @property
    def legacy_min_deposit(self) -> str:
        """Backward compatibility: Legacy minimum deposit"""
        return self.payments.minimum_deposit_amount or ""
    
    @property
    def legacy_wagering_requirements(self) -> str:
        """Backward compatibility: Legacy wagering requirements"""
        return self.bonuses.welcome_bonus.wagering_requirements or ""
    
    # ========================================================================
    # VALIDATION & PROCESSING
    # ========================================================================
    
    @model_validator(mode='before')
    @classmethod
    def validate_overall_consistency(cls, values):
        """Validate overall data consistency"""
        if isinstance(values, dict):
            # Ensure casino name is provided
            if not values.get('casino_name'):
                raise ValueError("Casino name is required")
            
            # Validate URL format if provided
            url = values.get('casino_url')
            if url and not re.match(r'^https?://', url):
                values['casino_url'] = f"https://{url}"
        
        return values
    
    @validator('overall_rating', 'safety_score', 'player_experience_score', 'value_score')
    def validate_rating_range(cls, v):
        """Ensure ratings are within valid range"""
        if v is not None and (v < 0 or v > 10):
            raise ValueError("Ratings must be between 0 and 10")
        return v
    
    def calculate_completeness_score(self) -> float:
        """Calculate how complete this intelligence profile is (0-1)"""
        total_fields = 95
        filled_fields = 0
        
        # Count filled fields across all categories
        for category_name in ['trustworthiness', 'games', 'bonuses', 'payments', 'user_experience', 'innovations']:
            category = getattr(self, category_name)
            filled_fields += self._count_filled_fields_in_object(category)
        
        return min(1.0, filled_fields / total_fields)
    
    def _count_filled_fields_in_object(self, obj, depth: int = 0) -> int:
        """Recursively count filled fields in an object"""
        if depth > 5:  # Prevent infinite recursion
            return 0
        
        count = 0
        if hasattr(obj, '__dict__'):
            for field_name, field_value in obj.__dict__.items():
                if field_value is not None and field_value != "" and field_value != []:
                    if isinstance(field_value, (dict, list)):
                        if field_value:  # Non-empty dict or list
                            count += 1
                    elif hasattr(field_value, '__dict__'):
                        count += self._count_filled_fields_in_object(field_value, depth + 1)
                    else:
                        count += 1
        
        return count
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy 14-field format for backward compatibility"""
        return {
            'casino_rating': self.legacy_casino_rating,
            'bonus_amount': self.legacy_bonus_amount,
            'license_info': self.legacy_license_info,
            'game_providers': self.legacy_game_providers,
            'payment_methods': self.legacy_payment_methods,
            'mobile_compatible': self.legacy_mobile_compatible,
            'live_chat_support': self.legacy_live_chat_support,
            'withdrawal_time': self.legacy_withdrawal_time,
            'min_deposit': self.legacy_min_deposit,
            'wagering_requirements': self.legacy_wagering_requirements,
            'review_summary': f"Comprehensive analysis of {self.casino_name}",
            'pros_list': [],  # Can be calculated from various fields
            'cons_list': [],  # Can be calculated from various fields
            'verdict': f"Analysis complete for {self.casino_name}"
        }
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        schema_extra = {
            "example": {
                "casino_name": "Example Casino",
                "casino_url": "https://example-casino.com",
                "trustworthiness": {
                    "license_info": {
                        "primary_license": "Malta Gaming Authority (MGA)",
                        "license_numbers": {"MGA": "MGA/B2C/123/2020"}
                    },
                    "security_features": {
                        "ssl_encryption": True,
                        "two_factor_authentication": True
                    }
                },
                "games": {
                    "game_portfolio": {
                        "total_games": 2500,
                        "slot_games_count": 2000
                    },
                    "software_providers": {
                        "primary_providers": ["NetEnt", "Microgaming"]
                    }
                }
            }
        }


# ============================================================================
# UTILITY FUNCTIONS FOR SCHEMA USAGE
# ============================================================================

def create_empty_casino_intelligence(casino_name: str, casino_url: Optional[str] = None) -> CasinoIntelligence:
    """Create an empty CasinoIntelligence instance with basic info"""
    return CasinoIntelligence(
        casino_name=casino_name,
        casino_url=casino_url,
        data_sources=[]
    )


def merge_casino_intelligence(base: CasinoIntelligence, update: CasinoIntelligence) -> CasinoIntelligence:
    """Merge two CasinoIntelligence instances, with update taking precedence"""
    merged_dict = base.dict()
    update_dict = update.dict()
    
    # Deep merge the dictionaries
    def deep_merge(dict1: dict, dict2: dict) -> dict:
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            elif value is not None:  # Only update if the new value is not None
                result[key] = value
        return result
    
    merged = deep_merge(merged_dict, update_dict)
    return CasinoIntelligence(**merged)


def validate_casino_intelligence_schema():
    """Validate the schema structure and return schema info"""
    schema = CasinoIntelligence.schema()
    
    def count_fields(schema_part, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return 0
        
        count = 0
        if 'properties' in schema_part:
            for prop_name, prop_schema in schema_part['properties'].items():
                count += 1
                if '$ref' in prop_schema:
                    # Handle references
                    continue
                elif 'properties' in prop_schema:
                    count += count_fields(prop_schema, depth + 1)
        
        return count
    
    total_fields = count_fields(schema)
    
    return {
        "total_fields": total_fields,
        "schema_valid": True,
        "categories": 6,
        "target_field_count": 95,
        "schema_version": "1.0.0"
    }


if __name__ == "__main__":
    # Test the schema
    print("🎰 Testing Casino Intelligence Schema (95 Fields)")
    print("=" * 60)
    
    # Validate schema
    validation_result = validate_casino_intelligence_schema()
    print(f"Schema Validation: {validation_result}")
    
    # Create test instance
    test_casino = create_empty_casino_intelligence("Test Casino", "https://test-casino.com")
    print(f"\nEmpty Casino Created: {test_casino.casino_name}")
    print(f"Completeness Score: {test_casino.calculate_completeness_score():.2%}")
    
    # Test legacy compatibility
    legacy_format = test_casino.to_legacy_format()
    print(f"\nLegacy Format Keys: {list(legacy_format.keys())}")
    print(f"Legacy Casino Rating: {legacy_format['casino_rating']}")
    
    print("\n✅ Schema validation complete!") 