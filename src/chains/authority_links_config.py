"""
Configuration for Authority Links by Region and Presets
Provides region-specific authoritative links and configuration presets
"""

from typing import Dict, List, Any
from .authoritative_hyperlink_engine import AuthorityLink, LinkCategory, LinkGenerationConfig

class AuthorityLinkPresets:
    """Preset configurations for different use cases"""
    
    @staticmethod
    def seo_optimized() -> Dict[str, Any]:
        """SEO-optimized configuration"""
        return {
            "max_links_per_content": 6,
            "max_links_per_category": 2,
            "min_confidence_score": 0.8,
            "prefer_first_occurrence": True,
            "enable_semantic_matching": True,
            "enable_compliance_section": True
        }
    
    @staticmethod
    def content_rich() -> Dict[str, Any]:
        """Content-rich configuration with more links"""
        return {
            "max_links_per_content": 10,
            "max_links_per_category": 3,
            "min_confidence_score": 0.7,
            "prefer_first_occurrence": True,
            "enable_semantic_matching": True,
            "enable_compliance_section": True
        }
    
    @staticmethod
    def minimal() -> Dict[str, Any]:
        """Minimal configuration for conservative linking"""
        return {
            "max_links_per_content": 4,
            "max_links_per_category": 1,
            "min_confidence_score": 0.9,
            "prefer_first_occurrence": True,
            "enable_semantic_matching": True,
            "enable_compliance_section": True
        }

def get_authority_links_for_region(region: str = "uk") -> List[AuthorityLink]:
    """Get authority links specific to a region"""
    
    # Base international links
    base_links = [
        # International Game Providers
        AuthorityLink(
            url="https://www.microgaming.com/",
            name="Microgaming",
            category=LinkCategory.GAME_PROVIDERS,
            keywords=["Microgaming", "slots", "progressive jackpots", "Mega Moolah", "online slots"],
            anchor_variations=["Microgaming", "Microgaming slots", "progressive slots"],
            context_requirements=["games", "slots", "software"],
            authority_score=0.88,
            description="Leading casino game provider with progressive jackpots"
        ),
        AuthorityLink(
            url="https://www.netent.com/",
            name="NetEnt",
            category=LinkCategory.GAME_PROVIDERS,
            keywords=["NetEnt", "Starburst", "Gonzo's Quest", "slot games", "premium slots"],
            anchor_variations=["NetEnt", "NetEnt games", "premium slots"],
            context_requirements=["games", "slots", "software"],
            authority_score=0.89,
            description="Premium casino game developer known for high-quality slots"
        ),
        AuthorityLink(
            url="https://www.evolution.com/",
            name="Evolution Gaming",
            category=LinkCategory.GAME_PROVIDERS,
            keywords=["Evolution Gaming", "live casino", "live dealer", "blackjack", "roulette"],
            anchor_variations=["Evolution Gaming", "live casino provider", "live dealer games"],
            context_requirements=["live", "dealer", "casino"],
            authority_score=0.92,
            description="Leading live casino game provider"
        ),
        
        # International Payment Security
        AuthorityLink(
            url="https://www.pcisecuritystandards.org/",
            name="PCI Security Standards",
            category=LinkCategory.PAYMENT_SECURITY,
            keywords=["PCI DSS", "payment security", "card security", "financial security"],
            anchor_variations=["PCI DSS", "payment security standards", "card protection"],
            context_requirements=["payment", "security", "banking"],
            authority_score=0.96,
            description="Payment card industry security standards"
        )
    ]
    
    # Region-specific links
    region_links = {
        "uk": [
            # UK Responsible Gambling
            AuthorityLink(
                url="https://www.gambleaware.org/",
                name="GambleAware",
                category=LinkCategory.RESPONSIBLE_GAMBLING,
                keywords=["responsible gambling", "problem gambling", "gambling addiction", "help", "support"],
                anchor_variations=["GambleAware", "responsible gambling resources", "gambling support"],
                context_requirements=["gambling", "betting", "casino"],
                authority_score=0.95,
                description="UK's leading charity for responsible gambling"
            ),
            AuthorityLink(
                url="https://www.gamstop.co.uk/",
                name="GAMSTOP",
                category=LinkCategory.RESPONSIBLE_GAMBLING,
                keywords=["self-exclusion", "block gambling", "gambling ban", "stop gambling"],
                anchor_variations=["GAMSTOP", "self-exclusion service", "gambling block"],
                context_requirements=["UK", "self-exclusion", "responsible"],
                authority_score=0.98,
                description="UK's national self-exclusion scheme"
            ),
            
            # UK Regulatory
            AuthorityLink(
                url="https://www.gamblingcommission.gov.uk/",
                name="UK Gambling Commission",
                category=LinkCategory.REGULATORY,
                keywords=["gambling license", "UKGC", "regulated", "licensing", "gambling commission"],
                anchor_variations=["UK Gambling Commission", "UKGC", "gambling regulator"],
                context_requirements=["license", "regulation", "UK"],
                authority_score=1.0,
                description="UK's official gambling regulator"
            )
        ],
        
        "malta": [
            # Malta Regulatory
            AuthorityLink(
                url="https://www.mga.org.mt/",
                name="Malta Gaming Authority",
                category=LinkCategory.REGULATORY,
                keywords=["MGA", "Malta license", "gaming authority", "Malta gambling"],
                anchor_variations=["Malta Gaming Authority", "MGA", "Malta regulator"],
                context_requirements=["Malta", "license", "MGA"],
                authority_score=0.92,
                description="Malta's gaming regulator"
            ),
            
            # Malta Responsible Gambling
            AuthorityLink(
                url="https://www.responsiblegaming.org.mt/",
                name="Responsible Gaming Foundation",
                category=LinkCategory.RESPONSIBLE_GAMBLING,
                keywords=["responsible gaming", "Malta gambling support", "problem gambling"],
                anchor_variations=["Responsible Gaming Foundation", "gambling support Malta"],
                context_requirements=["Malta", "responsible", "gambling"],
                authority_score=0.88,
                description="Malta's responsible gaming foundation"
            )
        ],
        
        "curacao": [
            # Curacao Regulatory
            AuthorityLink(
                url="https://www.gaming-curacao.com/",
                name="Curacao Gaming Control Board",
                category=LinkCategory.REGULATORY,
                keywords=["Curacao license", "gaming control board", "Curacao gambling"],
                anchor_variations=["Curacao Gaming Control Board", "Curacao regulator"],
                context_requirements=["Curacao", "license"],
                authority_score=0.75,
                description="Curacao's gaming regulator"
            )
        ],
        
        "us": [
            # US Responsible Gambling
            AuthorityLink(
                url="https://www.ncpgambling.org/",
                name="National Council on Problem Gambling",
                category=LinkCategory.RESPONSIBLE_GAMBLING,
                keywords=["problem gambling", "gambling addiction", "NCPG", "gambling help"],
                anchor_variations=["National Council on Problem Gambling", "NCPG", "gambling support"],
                context_requirements=["gambling", "problem", "addiction"],
                authority_score=0.93,
                description="US national organization for problem gambling resources"
            )
        ]
    }
    
    # Combine base links with region-specific links
    all_links = base_links.copy()
    if region.lower() in region_links:
        all_links.extend(region_links[region.lower()])
    
    return all_links

def get_casino_specific_links(casino_name: str = "") -> List[AuthorityLink]:
    """Get links specific to casino types or features"""
    
    casino_specific = []
    
    # Crypto casino specific links
    if any(term in casino_name.lower() for term in ["crypto", "bitcoin", "ethereum"]):
        casino_specific.extend([
            AuthorityLink(
                url="https://bitcoin.org/",
                name="Bitcoin",
                category=LinkCategory.PAYMENT_SECURITY,
                keywords=["Bitcoin", "cryptocurrency", "crypto payments", "BTC"],
                anchor_variations=["Bitcoin", "cryptocurrency", "crypto payments"],
                context_requirements=["crypto", "bitcoin", "cryptocurrency"],
                authority_score=0.90,
                description="Official Bitcoin information"
            )
        ])
    
    # Live casino specific links
    if "live" in casino_name.lower():
        casino_specific.extend([
            AuthorityLink(
                url="https://www.evolution.com/",
                name="Evolution Gaming",
                category=LinkCategory.GAME_PROVIDERS,
                keywords=["Evolution Gaming", "live casino", "live dealer", "blackjack", "roulette"],
                anchor_variations=["Evolution Gaming", "live casino provider", "live dealer games"],
                context_requirements=["live", "dealer", "casino"],
                authority_score=0.92,
                description="Leading live casino game provider"
            )
        ])
    
    return casino_specific

def get_compliance_links() -> List[AuthorityLink]:
    """Get compliance and security related links"""
    return [
        AuthorityLink(
            url="https://www.ssl.com/",
            name="SSL Certificates",
            category=LinkCategory.PAYMENT_SECURITY,
            keywords=["SSL", "encryption", "secure connection", "HTTPS", "security certificate"],
            anchor_variations=["SSL encryption", "secure connection", "SSL certificates"],
            context_requirements=["security", "encryption", "secure"],
            authority_score=0.85,
            description="SSL encryption and security certificates"
        ),
        AuthorityLink(
            url="https://www.ecogra.org/",
            name="eCOGRA",
            category=LinkCategory.INDUSTRY_STANDARDS,
            keywords=["eCOGRA", "fair gaming", "testing", "certification", "random number"],
            anchor_variations=["eCOGRA", "fair gaming certification", "game testing"],
            context_requirements=["fair", "testing", "certification"],
            authority_score=0.89,
            description="Independent testing and certification for online gaming"
        ),
        AuthorityLink(
            url="https://www.itechlab.com/",
            name="iTech Labs",
            category=LinkCategory.INDUSTRY_STANDARDS,
            keywords=["iTech Labs", "game testing", "RNG testing", "certification"],
            anchor_variations=["iTech Labs", "game testing", "RNG certification"],
            context_requirements=["testing", "RNG", "random"],
            authority_score=0.87,
            description="Gaming testing laboratory and certification"
        )
    ]

def create_custom_link_set(
    region: str = "uk",
    casino_name: str = "",
    include_compliance: bool = True,
    include_providers: bool = True
) -> List[AuthorityLink]:
    """Create a custom set of authority links based on parameters"""
    
    links = get_authority_links_for_region(region)
    
    if casino_name:
        links.extend(get_casino_specific_links(casino_name))
    
    if include_compliance:
        links.extend(get_compliance_links())
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_links = []
    for link in links:
        if link.url not in seen_urls:
            seen_urls.add(link.url)
            unique_links.append(link)
    
    return unique_links 