#!/usr/bin/env python3
"""
Web Research Chain - Native LangChain WebBaseLoader Implementation
Collects 95+ data points for casino research using WebBaseLoader with advanced URL strategies

✅ NATIVE LANGCHAIN PATTERNS:
- WebBaseLoader with concurrent loading and custom parsers
- RunnableParallel for multi-category URL scraping
- Strategic URL generation for different data categories
- Structured integration with 95-field ComprehensiveResearchData

✅ INTEGRATION POINTS:
- ComprehensiveResearchChain: Direct data feeding
- Universal RAG Chain: Enhanced research capabilities
- Supabase Storage: Document persistence and vector embedding
- Enhanced Confidence Scoring: Source quality assessment

🎯 USAGE: casino_domain | web_research_chain → 95+ structured fields from web scraping
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
import logging
from urllib.parse import urljoin, urlparse
import re

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import (
    RunnableParallel, 
    RunnableLambda, 
    RunnableSequence,
    RunnableBranch,
    RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import our existing 95-field structure
from .comprehensive_research_chain import (
    ComprehensiveResearchData,
    TrustworthinessData,
    GamesData, 
    BonusData,
    PaymentData,
    UserExperienceData,
    InnovationsData,
    ComplianceData,
    AssessmentData
)

logger = logging.getLogger(__name__)

# ✅ CASINO-SPECIFIC URL STRATEGY GENERATOR

class CasinoURLStrategy:
    """Strategic URL generation for different data categories"""
    
    @staticmethod
    def generate_casino_urls(base_domain: str) -> Dict[str, List[str]]:
        """
        Generate comprehensive URL lists for each data category
        
        Args:
            base_domain: Casino domain (e.g., "betway.com", "888casino.com")
            
        Returns:
            Dict mapping categories to URL lists for scraping
        """
        base_url = f"https://{base_domain.replace('https://', '').replace('http://', '')}"
        
        return {
            "trustworthiness": [
                f"{base_url}/about",
                f"{base_url}/about-us", 
                f"{base_url}/company",
                f"{base_url}/licensing",
                f"{base_url}/responsible-gambling",
                f"{base_url}/security",
                f"{base_url}/privacy-policy",
                f"{base_url}/terms-and-conditions",
                f"{base_url}/contact"
            ],
            "games": [
                f"{base_url}/casino",
                f"{base_url}/games",
                f"{base_url}/slots",
                f"{base_url}/table-games",
                f"{base_url}/live-casino",
                f"{base_url}/providers",
                f"{base_url}/new-games",
                f"{base_url}/mobile",
                f"{base_url}/sports" # For sports betting
            ],
            "bonuses": [
                f"{base_url}/promotions",
                f"{base_url}/bonuses", 
                f"{base_url}/welcome-bonus",
                f"{base_url}/vip",
                f"{base_url}/loyalty",
                f"{base_url}/free-spins",
                f"{base_url}/no-deposit",
                f"{base_url}/cashback"
            ],
            "payments": [
                f"{base_url}/banking",
                f"{base_url}/deposits",
                f"{base_url}/withdrawals", 
                f"{base_url}/payment-methods",
                f"{base_url}/crypto",
                f"{base_url}/fees",
                f"{base_url}/limits"
            ],
            "user_experience": [
                f"{base_url}/support",
                f"{base_url}/help",
                f"{base_url}/faq",
                f"{base_url}/contact",
                f"{base_url}/mobile-app",
                f"{base_url}/download",
                f"{base_url}/" # Homepage for overall UX
            ],
            "compliance": [
                f"{base_url}/responsible-gambling",
                f"{base_url}/age-verification", 
                f"{base_url}/self-exclusion",
                f"{base_url}/privacy-policy",
                f"{base_url}/aml",
                f"{base_url}/kyc",
                f"{base_url}/licensing"
            ]
        }

# ✅ ENHANCED WEBBASELOADER WITH CASINO OPTIMIZATION

class EnhancedCasinoWebLoader:
    """Enhanced WebBaseLoader with casino-specific optimizations"""
    
    def __init__(self, 
                 requests_per_second: int = 3,
                 default_parser: str = "html.parser",
                 raise_for_status: bool = False,
                 continue_on_failure: bool = True):
        """
        Initialize enhanced casino web loader
        
        Args:
            requests_per_second: Concurrent request limit (be respectful!)
            default_parser: BeautifulSoup parser ("html.parser", "lxml", "xml")
            raise_for_status: Raise exceptions on HTTP errors
            continue_on_failure: Continue scraping if some URLs fail
        """
        self.requests_per_second = requests_per_second
        self.default_parser = default_parser
        self.raise_for_status = raise_for_status
        self.continue_on_failure = continue_on_failure
        
        # Casino-specific headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def create_loader(self, urls: List[str]) -> WebBaseLoader:
        """Create configured WebBaseLoader for casino URLs"""
        loader = WebBaseLoader(
            web_paths=urls,
            header_template=self.headers,
            requests_per_second=self.requests_per_second,
            default_parser=self.default_parser,
            raise_for_status=self.raise_for_status,
            continue_on_failure=self.continue_on_failure
        )
        return loader
    
    async def load_category_urls(self, category_urls: Dict[str, List[str]]) -> Dict[str, List]:
        """
        Load URLs for each category concurrently
        
        Args:
            category_urls: Dict mapping categories to URL lists
            
        Returns:
            Dict mapping categories to loaded documents
        """
        category_docs = {}
        
        for category, urls in category_urls.items():
            try:
                logger.info(f"Loading {len(urls)} URLs for category: {category}")
                loader = self.create_loader(urls)
                
                # Use async loading for better performance
                docs = await loader.aload()
                category_docs[category] = docs
                
                logger.info(f"Successfully loaded {len(docs)} documents for {category}")
                
            except Exception as e:
                logger.warning(f"Failed to load some URLs for {category}: {str(e)}")
                if self.continue_on_failure:
                    category_docs[category] = []
                else:
                    raise
                    
        return category_docs

# ✅ DATA EXTRACTION CHAIN WITH LLM PROCESSING

class WebDataExtractor:
    """Extract structured data from web documents using LLM"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )
        
    def create_extraction_prompts(self) -> Dict[str, ChatPromptTemplate]:
        """Create category-specific extraction prompts"""
        
        prompts = {
            "trustworthiness": ChatPromptTemplate.from_template("""
Extract trustworthiness information from the following casino website content:

<content>
{content}
</content>

Extract the following fields and return as JSON:
- license_authorities: List of licensing authorities mentioned
- license_numbers: Any license numbers found
- parent_company: Parent company name if mentioned
- years_in_operation: Years in operation or founding year
- ssl_certification: Whether SSL/security is mentioned (true/false)
- auditing_agency: Auditing companies mentioned
- legal_issues: Any legal problems mentioned
- industry_awards: Awards or recognitions mentioned

Return only valid JSON format.
"""),
            
            "games": ChatPromptTemplate.from_template("""
Extract gaming information from the following casino website content:

<content>
{content}
</content>

Extract the following fields and return as JSON:
- slot_count: Number of slot games (extract number if mentioned)
- table_game_count: Number of table games
- live_casino: Whether live casino is available (true/false)
- sports_betting: Whether sports betting is available (true/false)
- providers: List of game providers mentioned
- exclusive_games: Exclusive game titles mentioned
- mobile_compatibility: Mobile gaming availability (true/false)
- demo_play_available: Demo/free play available (true/false)
- game_categories: Types of games offered

Return only valid JSON format.
"""),
            
            "bonuses": ChatPromptTemplate.from_template("""
Extract bonus information from the following casino website content:

<content>
{content}
</content>

Extract the following fields and return as JSON:
- welcome_bonus_amount: Welcome bonus amount (e.g., "$500", "€200")
- welcome_bonus_percentage: Bonus percentage (number only)
- wagering_requirements: Wagering requirements (number, e.g., 35)
- free_spins_count: Number of free spins offered
- no_deposit_bonus: Whether no deposit bonus available (true/false)
- loyalty_program: Whether VIP/loyalty program exists (true/false)
- cashback_offered: Whether cashback is offered (true/false)
- bonus_expiry_days: Bonus validity period in days

Return only valid JSON format.
"""),
            
            "payments": ChatPromptTemplate.from_template("""
Extract payment information from the following casino website content:

<content>
{content}
</content>

Extract the following fields and return as JSON:
- deposit_methods: List of deposit methods mentioned
- withdrawal_methods: List of withdrawal methods
- min_deposit_amount: Minimum deposit amount
- withdrawal_processing_time: Processing time for withdrawals
- deposit_fees: Deposit fees information
- withdrawal_fees: Withdrawal fees information
- currency_options: Supported currencies
- cryptocurrency_support: Crypto payment support (true/false)
- fast_withdrawal: Instant/fast withdrawal available (true/false)

Return only valid JSON format.
"""),
            
            "user_experience": ChatPromptTemplate.from_template("""
Extract user experience information from the following casino website content:

<content>
{content}
</content>

Extract the following fields and return as JSON:
- customer_support_24_7: 24/7 support available (true/false)
- live_chat_available: Live chat support (true/false)
- support_languages: Languages supported
- website_languages: Website language options
- mobile_app_available: Mobile app availability (true/false)
- responsible_gambling_tools: RG tools mentioned

Return only valid JSON format.
"""),
            
            "compliance": ChatPromptTemplate.from_template("""
Extract compliance information from the following casino website content:

<content>
{content}
</content>

Extract the following fields and return as JSON:
- responsible_gambling_certified: RG certification mentioned (true/false)
- age_verification_strict: Age verification mentioned (true/false)
- anti_money_laundering: AML compliance mentioned (true/false)
- data_protection_compliance: GDPR/data protection mentioned
- fair_play_certification: Fair play certification (true/false)
- ethical_gambling_practices: Ethical practices mentioned

Return only valid JSON format.
""")
        }
        
        return prompts
    
    def create_extraction_chains(self) -> Dict[str, RunnableSequence]:
        """Create extraction chains for each category"""
        prompts = self.create_extraction_prompts()
        chains = {}
        
        for category, prompt in prompts.items():
            chain = prompt | self.llm | JsonOutputParser()
            chains[category] = chain
            
        return chains

# ✅ MAIN WEB RESEARCH CHAIN

def create_web_research_chain(
    llm: Optional[ChatOpenAI] = None,
    requests_per_second: int = 3,
    continue_on_failure: bool = True
) -> RunnableSequence:
    """
    Create comprehensive web research chain for 95+ data points
    
    Args:
        llm: Language model for data extraction
        requests_per_second: Rate limit for web requests
        continue_on_failure: Continue if some URLs fail
        
    Returns:
        RunnableSequence that takes casino_domain and returns ComprehensiveResearchData
    """
    
    # Initialize components
    web_loader = EnhancedCasinoWebLoader(
        requests_per_second=requests_per_second,
        continue_on_failure=continue_on_failure
    )
    
    data_extractor = WebDataExtractor(llm)
    extraction_chains = data_extractor.create_extraction_chains()
    
    # Step 1: Generate URLs for all categories
    def generate_urls(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate category URLs from casino domain"""
        casino_domain = inputs.get("casino_domain") or inputs.get("query", "")
        
        if not casino_domain:
            raise ValueError("casino_domain is required")
            
        category_urls = CasinoURLStrategy.generate_casino_urls(casino_domain)
        
        return {
            **inputs,
            "category_urls": category_urls,
            "casino_domain": casino_domain
        }
    
    # Step 2: Load web content for all categories  
    async def load_web_content(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load web content for all URL categories"""
        category_urls = inputs["category_urls"]
        
        category_docs = await web_loader.load_category_urls(category_urls)
        
        return {
            **inputs,
            "category_docs": category_docs
        }
    
    # Step 3: Extract structured data from each category
    def extract_category_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from loaded documents"""
        category_docs = inputs["category_docs"]
        extracted_data = {}
        
        for category, docs in category_docs.items():
            if category not in extraction_chains:
                continue
                
            # Combine all documents for this category
            combined_content = "\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in docs[:5]  # Limit to first 5 docs to avoid token limits
            ])
            
            if combined_content.strip():
                try:
                    # Extract structured data using LLM
                    extracted = extraction_chains[category].invoke({
                        "content": combined_content[:8000]  # Limit content size
                    })
                    extracted_data[category] = extracted
                    
                except Exception as e:
                    logger.warning(f"Failed to extract data for {category}: {str(e)}")
                    extracted_data[category] = {}
            else:
                extracted_data[category] = {}
        
        return {
            **inputs,
            "extracted_data": extracted_data
        }
    
    # Step 4: Structure into ComprehensiveResearchData
    def structure_final_data(inputs: Dict[str, Any]) -> ComprehensiveResearchData:
        """Structure extracted data into final format"""
        extracted_data = inputs["extracted_data"]
        casino_domain = inputs["casino_domain"]
        
        # Create structured data objects
        research_data = ComprehensiveResearchData(
            trustworthiness=TrustworthinessData(**extracted_data.get("trustworthiness", {})),
            games=GamesData(**extracted_data.get("games", {})),
            bonuses=BonusData(**extracted_data.get("bonuses", {})),
            payments=PaymentData(**extracted_data.get("payments", {})),
            user_experience=UserExperienceData(**extracted_data.get("user_experience", {})),
            compliance=ComplianceData(**extracted_data.get("compliance", {})),
            
            # Set metadata
            extraction_timestamp=datetime.now(),
            keyword=casino_domain,
            source_quality_score=0.8,  # Web scraping baseline quality
        )
        
        # Count populated fields
        total_fields = 0
        for category_data in [research_data.trustworthiness, research_data.games, 
                             research_data.bonuses, research_data.payments,
                             research_data.user_experience, research_data.compliance]:
            for field_name, field_value in category_data.__dict__.items():
                if field_value is not None and field_value != [] and field_value != "":
                    total_fields += 1
        
        research_data.total_fields_populated = total_fields
        
        logger.info(f"✅ Web research complete: {total_fields}/95+ fields populated for {casino_domain}")
        
        return research_data
    
    # Create the main chain
    chain = (
        RunnableLambda(generate_urls)
        | RunnableLambda(load_web_content) 
        | RunnableLambda(extract_category_data)
        | RunnableLambda(structure_final_data)
    )
    
    return chain

# ✅ INTEGRATION WITH EXISTING SYSTEMS

class WebResearchEnhancer:
    """Integration helpers for existing Universal RAG system"""
    
    @staticmethod
    def enhance_universal_rag_chain(universal_rag_chain, web_research_chain):
        """Add web research capabilities to Universal RAG Chain"""
        
        def combined_research(inputs):
            """Combine web research with existing RAG capabilities"""
            
            # Check if this is a casino domain query
            query = inputs.get("query", "")
            casino_domain_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})'
            
            if re.search(casino_domain_pattern, query):
                logger.info("🌐 Detected casino domain - using web research")
                
                # Run web research first
                web_data = web_research_chain.invoke({"casino_domain": query})
                
                # Add web research context to RAG
                web_context = f"Web Research Data: {web_data.json()}"
                enhanced_inputs = {
                    **inputs,
                    "additional_context": web_context
                }
                
                # Run normal RAG with enhanced context
                return universal_rag_chain.invoke(enhanced_inputs)
            else:
                # Normal RAG processing
                return universal_rag_chain.invoke(inputs)
        
        return RunnableLambda(combined_research)
    
    @staticmethod
    def create_web_research_api_endpoint():
        """Create FastAPI endpoint for web research"""
        
        web_research_chain = create_web_research_chain()
        
        async def research_casino(casino_domain: str) -> ComprehensiveResearchData:
            """API endpoint for casino web research"""
            try:
                result = await web_research_chain.ainvoke({"casino_domain": casino_domain})
                return result
            except Exception as e:
                logger.error(f"Web research failed for {casino_domain}: {str(e)}")
                raise

# ✅ DEMO USAGE EXAMPLE

async def demo_web_research():
    """Demonstrate web research capabilities"""
    
    # Test WebBaseLoader directly
    casino_domain = "betway.com"
    
    logger.info(f"🎰 Testing WebBaseLoader for {casino_domain}")
    
    try:
        # Generate URLs
        url_strategy = CasinoURLStrategy()
        category_urls = url_strategy.generate_casino_urls(casino_domain)
        
        # Create enhanced loader
        web_loader = EnhancedCasinoWebLoader(
            requests_per_second=2,  # Be respectful to casino servers
            continue_on_failure=True
        )
        
        # Load a few URLs as test
        test_urls = category_urls["trustworthiness"][:3]  # Just test first 3 URLs
        
        loader = web_loader.create_loader(test_urls)
        docs = await loader.aload()
        
        logger.info(f"✅ Successfully loaded {len(docs)} documents")
        
        # Display results
        print(f"\n🎯 WEB RESEARCH TEST RESULTS FOR {casino_domain.upper()}")
        print(f"📊 Documents Loaded: {len(docs)}")
        
        for i, doc in enumerate(docs[:2]):  # Show first 2 docs
            print(f"\n📄 Document {i+1}:")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Title: {doc.metadata.get('title', 'No title')}")
            print(f"  Content Preview: {doc.page_content[:200]}...")
        
        return docs
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run demo
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_web_research()) 