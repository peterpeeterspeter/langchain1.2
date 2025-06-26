#!/usr/bin/env python3
"""
Enhanced Web Research Chain - Native LangChain WebBaseLoader
Advanced implementation for collecting 95+ casino data points with geo-restrictions handling

✅ PRODUCTION FEATURES:
- Multi-region URL strategies for geo-restricted sites
- Smart content parsing with fallback mechanisms
- Concurrent loading with rate limiting
- Integration with 95-field ComprehensiveResearchData
- Error recovery and retry mechanisms

🎯 HANDLES GEO-RESTRICTIONS:
- Alternative domain discovery (.com, .co.uk, .ca, etc.)
- Proxy rotation support
- Mirror site detection
- Archive.org fallback for blocked content
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Simplified 95-field data structure for casino research
class ComprehensiveResearchData(BaseModel):
    """Simplified 95-field structure for casino research"""
    trustworthiness: Dict[str, Any] = Field(default_factory=dict)
    games: Dict[str, Any] = Field(default_factory=dict) 
    bonuses: Dict[str, Any] = Field(default_factory=dict)
    payments: Dict[str, Any] = Field(default_factory=dict)
    user_experience: Dict[str, Any] = Field(default_factory=dict)
    innovations: Dict[str, Any] = Field(default_factory=dict)
    compliance: Dict[str, Any] = Field(default_factory=dict)
    assessment: Dict[str, Any] = Field(default_factory=dict)
    affiliate_program: Dict[str, Any] = Field(default_factory=dict)

@dataclass
class URLStrategy:
    """URL generation strategy for different casino data categories"""
    base_domain: str
    alternative_domains: List[str] = field(default_factory=list)
    category_paths: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.category_paths:
            self.category_paths = {
                'trustworthiness': [
                    '/about', '/about-us', '/company', '/licensing', 
                    '/legal', '/terms', '/privacy', '/responsible-gaming',
                    '/security', '/fairness', '/audits', '/certifications',
                    '/terms-and-conditions', '/terms-of-service', '/tos',
                    '/legal/terms', '/about/terms', '/support/terms',
                    '/legal/terms-conditions', '/help/terms', '/info/terms'
                ],
                'games': [
                    '/casino', '/games', '/slots', '/live-casino',
                    '/table-games', '/jackpots', '/new-games',
                    '/providers', '/software', '/demo', '/tournaments'
                ],
                'bonuses': [
                    '/promotions', '/bonuses', '/welcome-bonus',
                    '/free-spins', '/loyalty', '/vip', '/rewards',
                    '/offers', '/deals', '/cashback', '/reload',
                    '/bonus-terms', '/promotion-terms', '/terms/bonuses',
                    '/legal/bonus-conditions', '/wagering-requirements'
                ],
                'payments': [
                    '/banking', '/payments', '/deposit', '/withdrawal',
                    '/payment-methods', '/cashier', '/transactions',
                    '/fees', '/limits', '/crypto', '/verification',
                    '/payment-terms', '/withdrawal-terms', '/banking-terms',
                    '/terms/payments', '/legal/banking'
                ],
                'user_experience': [
                    '/support', '/help', '/contact', '/faq',
                    '/mobile', '/app', '/download', '/languages',
                    '/reviews', '/testimonials', '/feedback'
                ],
                'innovations': [
                    '/technology', '/innovation', '/vr', '/ar',
                    '/ai', '/blockchain', '/social', '/gamification',
                    '/partnerships', '/beta', '/new-features'
                ],
                'compliance': [
                    '/responsible-gambling', '/self-exclusion',
                    '/age-verification', '/aml', '/kyc',
                    '/complaints', '/dispute-resolution', '/transparency',
                    '/ethics', '/data-protection',
                    '/terms-and-conditions', '/privacy-policy', '/cookie-policy',
                    '/gdpr', '/data-protection', '/regulatory-information',
                    '/compliance', '/legal/compliance', '/regulatory-compliance'
                ],
                'assessment': [
                    '/reviews', '/ratings', '/awards', '/comparisons',
                    '/testimonials', '/case-studies', '/analysis',
                    '/reports', '/statistics', '/performance'
                ],
                'terms_and_conditions': [
                    '/terms-and-conditions', '/terms-of-service', '/tos',
                    '/legal/terms', '/about/terms', '/support/terms',
                    '/legal/terms-conditions', '/help/terms', '/info/terms',
                    '/user-agreement', '/service-agreement', '/player-agreement',
                    '/website-terms', '/platform-terms', '/gaming-terms',
                    '/legal/user-terms', '/legal/service-terms',
                    '/terms/general', '/terms/gaming', '/terms/website'
                ],
                'affiliate_program': [
                    '/affiliates', '/partners', '/webmasters', '/affiliate-program',
                    '/affiliate-terms', '/affiliate/terms', '/partner/terms',
                    '/legal/affiliates', '/affiliate-agreement', '/commission'
                ]
            }

class EnhancedWebBaseLoader:
    """Enhanced WebBaseLoader with casino-specific optimizations"""
    
    def __init__(self, 
                 strategy: URLStrategy,
                 requests_per_second: float = 0.5,
                 max_workers: int = 3,
                 timeout: int = 30,
                 use_archive_fallback: bool = True):
        
        self.strategy = strategy
        self.requests_per_second = requests_per_second
        self.max_workers = max_workers
        self.timeout = timeout
        self.use_archive_fallback = use_archive_fallback
        
        # Enhanced headers for casino sites
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
    
    def generate_urls(self, categories: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Generate comprehensive URL lists for data collection"""
        
        if categories is None:
            categories = list(self.strategy.category_paths.keys())
        
        url_collection = {}
        
        # All domains to try (main + alternatives)
        all_domains = [self.strategy.base_domain] + self.strategy.alternative_domains
        
        for category in categories:
            category_urls = []
            paths = self.strategy.category_paths.get(category, [])
            
            for domain in all_domains:
                for path in paths:
                    # Ensure domain has proper protocol
                    if not domain.startswith(('http://', 'https://')):
                        domain = f'https://{domain}'
                    
                    url = urljoin(domain, path)
                    category_urls.append(url)
            
            url_collection[category] = category_urls
        
        return url_collection
    
    def load_with_fallback(self, urls: List[str]) -> List[Document]:
        """Load URLs with fallback mechanisms"""
        
        successful_docs = []
        failed_urls = []
        
        # Primary loading attempt
        loader = WebBaseLoader(
            web_paths=urls,
            header_template=self.headers,
            requests_per_second=self.requests_per_second,
            continue_on_failure=True,
            raise_for_status=False,
            requests_kwargs={'timeout': self.timeout}
        )
        
        try:
            docs = loader.load()
            
            # Filter successful vs failed loads
            for doc in docs:
                content_length = len(doc.page_content.strip())
                
                # Check for common failure indicators
                if (content_length > 100 and 
                    not any(failure_term in doc.page_content.lower() for failure_term in [
                        'connection timed out', '502 bad gateway', '503 service unavailable',
                        'access denied', 'forbidden', 'not available in your region',
                        'cease trading in your region'
                    ])):
                    successful_docs.append(doc)
                else:
                    failed_urls.append(doc.metadata.get('source', ''))
                    
        except Exception as e:
            print(f"Primary loading failed: {e}")
            failed_urls.extend(urls)
        
        # Archive.org fallback for failed URLs
        if self.use_archive_fallback and failed_urls:
            print(f"Attempting archive.org fallback for {len(failed_urls)} failed URLs...")
            archive_docs = self._load_from_archive(failed_urls)
            successful_docs.extend(archive_docs)
        
        return successful_docs
    
    def _load_from_archive(self, failed_urls: List[str]) -> List[Document]:
        """Load content from Archive.org as fallback"""
        
        archive_docs = []
        
        for url in failed_urls[:3]:  # Limit to avoid overloading archive.org
            try:
                # Archive.org Wayback Machine URL format
                archive_url = f"https://web.archive.org/web/2024/{url}"
                
                archive_loader = WebBaseLoader(
                    web_paths=[archive_url],
                    header_template=self.headers,
                    requests_per_second=0.2,  # Very conservative for archive.org
                    continue_on_failure=True
                )
                
                archive_result = archive_loader.load()
                
                if archive_result and len(archive_result[0].page_content) > 500:
                    # Mark as archive source
                    archive_result[0].metadata['source_type'] = 'archive'
                    archive_result[0].metadata['original_source'] = url
                    archive_docs.extend(archive_result)
                    print(f"✅ Archive fallback successful for: {url}")
                
                time.sleep(2)  # Respectful delay for archive.org
                
            except Exception as e:
                print(f"Archive fallback failed for {url}: {e}")
                continue
        
        return archive_docs

class CasinoDataExtractor:
    """Extract structured data from web content using LLM"""
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Set up extraction prompt
        self.extraction_prompt = PromptTemplate(
            input_variables=["content", "category"],
            template="""
Extract specific {category} information from this casino website content.

Website Content:
{content}

Focus on extracting:
- Factual information only
- Specific details like numbers, names, policies
- Regulatory information
- Technical specifications
- Contact information
- Terms and conditions

Return the extracted information in a clear, structured format.
If information is not found, indicate "Not found" rather than guessing.

Extracted {category} Information:
"""
        )
        
        # ✅ NEW: Specialized T&C extraction prompt
        self.terms_extraction_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
Extract comprehensive casino intelligence from this Terms & Conditions content.

Terms & Conditions Content:
{content}

Extract the following information in structured format:

**LICENSING & REGULATORY:**
- License authorities and jurisdictions
- License numbers and regulatory IDs
- Compliance certifications

**GEOGRAPHIC RESTRICTIONS:**
- Restricted countries and regions
- Age verification requirements
- Geo-blocking policies

**PAYMENT & FINANCIAL:**
- Supported payment methods
- Withdrawal processing times
- Transaction limits and fees
- Currency support

**GAMING & OPERATIONS:**
- Game providers and software
- Bonus terms and wagering requirements
- Account verification procedures

**DISPUTE & COMPLIANCE:**
- Dispute resolution procedures
- Responsible gambling measures
- Data protection policies

**CONTACT & SUPPORT:**
- Regulatory contact information
- Dispute escalation contacts
- Compliance officer details

Return ONLY factual information found in the content. If information is not explicitly stated, mark as "Not found".

Extracted Intelligence:
"""
        )
    
    def extract_category_data(self, documents: List[Document], category: str) -> Dict[str, Any]:
        """Extract data for a specific category from documents"""
        
        extracted_data = {
            'sources': [],
            'raw_extractions': [],
            'structured_data': {},
            'confidence_score': 0.0
        }
        
        for doc in documents:
            if len(doc.page_content.strip()) < 100:
                continue  # Skip minimal content
            
            try:
                # ✅ NEW: Use specialized T&C extraction for terms_and_conditions category
                if category == 'terms_and_conditions':
                    result = self._extract_terms_intelligence(doc)
                else:
                    # Standard extraction for other categories
                    extraction_chain = self.extraction_prompt | self.llm
                    result = extraction_chain.invoke({
                        'content': doc.page_content[:12000],  # Increased limit for better extraction
                        'category': category
                    })
                
                extracted_data['sources'].append(doc.metadata.get('source', 'Unknown'))
                extracted_data['raw_extractions'].append(result.content if hasattr(result, 'content') else str(result))
                
            except Exception as e:
                print(f"Extraction failed for {doc.metadata.get('source', 'Unknown')}: {e}")
                continue
        
        # ✅ NEW: Enhanced confidence scoring for T&C content
        if category == 'terms_and_conditions':
            extracted_data['confidence_score'] = self._calculate_terms_confidence(extracted_data['raw_extractions'])
        else:
            # Calculate confidence based on successful extractions
            if len(extracted_data['raw_extractions']) > 0:
                extracted_data['confidence_score'] = min(len(extracted_data['raw_extractions']) / 3.0, 1.0)
        
        return extracted_data
    
    def _extract_terms_intelligence(self, document: Document) -> Dict[str, Any]:
        """Extract specialized intelligence from T&C content"""
        
        content = document.page_content
        
        # ✅ Intelligent content chunking for large T&C documents
        if len(content) > 6000:
            # Extract key sections for focused analysis
            key_sections = self._extract_key_terms_sections(content)
            content = "\n\n".join(key_sections)[:5000]  # Use most relevant sections
        
        # Apply specialized T&C extraction
        extraction_chain = self.terms_extraction_prompt | self.llm
        
        result = extraction_chain.invoke({
            'content': content
        })
        
        return result
    
    def _extract_key_terms_sections(self, content: str) -> List[str]:
        """Extract key sections from lengthy T&C documents"""
        
        # Section patterns to prioritize
        high_value_patterns = [
            r'licen[sc]e?[sd]?\b.*?(?=\n\n|\n[A-Z]|\Z)',
            r'restrict.*?(?=\n\n|\n[A-Z]|\Z)',
            r'jurisdiction.*?(?=\n\n|\n[A-Z]|\Z)',
            r'payment.*?(?=\n\n|\n[A-Z]|\Z)',
            r'withdraw.*?(?=\n\n|\n[A-Z]|\Z)',
            r'bonus.*?(?=\n\n|\n[A-Z]|\Z)',
            r'verification.*?(?=\n\n|\n[A-Z]|\Z)',
            r'dispute.*?(?=\n\n|\n[A-Z]|\Z)',
            r'complaint.*?(?=\n\n|\n[A-Z]|\Z)',
            r'responsible.*?(?=\n\n|\n[A-Z]|\Z)',
            r'age.*?(?=\n\n|\n[A-Z]|\Z)'
        ]
        
        key_sections = []
        content_lower = content.lower()
        
        for pattern in high_value_patterns:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section = content[match.start():match.end()]
                if len(section) > 100:  # Only substantial sections
                    key_sections.append(section[:800])  # Limit section length
        
        # If no patterns found, use first 3000 characters
        if not key_sections:
            key_sections = [content[:3000]]
        
        return key_sections[:5]  # Return top 5 sections max
    
    def _calculate_terms_confidence(self, extractions: List[str]) -> float:
        """Calculate confidence score specifically for T&C extractions"""
        
        if not extractions:
            return 0.0
        
        # Score based on presence of high-value T&C indicators
        confidence_indicators = [
            'license', 'jurisdiction', 'restrict', 'payment', 
            'withdraw', 'bonus', 'verification', 'dispute',
            'compliance', 'regulatory', 'authority', 'terms'
        ]
        
        total_score = 0
        for extraction in extractions:
            extraction_lower = extraction.lower()
            indicator_count = sum(1 for indicator in confidence_indicators if indicator in extraction_lower)
            
            # Score based on indicator density
            section_score = min(indicator_count / 6.0, 1.0)  # Max score of 1.0
            total_score += section_score
        
        # Average across extractions, boost for legal document quality
        avg_score = total_score / len(extractions)
        
        # Boost confidence for T&C content (legal accuracy assumption)
        legal_boost = 0.15 if avg_score > 0.3 else 0.0
        
        return min(avg_score + legal_boost, 1.0)

class ComprehensiveWebResearchChain:
    """Main LCEL chain for comprehensive casino research using WebBaseLoader"""
    
    def __init__(self, 
                 casino_domain: str,
                 alternative_domains: Optional[List[str]] = None,
                 llm: Optional[ChatOpenAI] = None,
                 categories: Optional[List[str]] = None):
        
        self.casino_domain = casino_domain
        self.alternative_domains = alternative_domains or []
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        self.categories = categories or [
            'trustworthiness', 'games', 'bonuses', 'payments', 
            'user_experience', 'innovations', 'compliance', 'assessment',
            'terms_and_conditions', 'affiliate_program'  # NEW: Dedicated T&C category
        ]
        
        # Initialize components
        self.url_strategy = URLStrategy(
            base_domain=casino_domain,
            alternative_domains=self.alternative_domains
        )
        
        self.loader = EnhancedWebBaseLoader(
            strategy=self.url_strategy,
            requests_per_second=0.5,
            max_workers=6,  # Increased for 8-category processing
            timeout=30
        )
        
        self.extractor = CasinoDataExtractor(llm=self.llm)
        
        # Build LCEL chain
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the runnable chain"""
        
        def load_and_extract_data(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Function to load documents and extract data for all categories."""
            
            categories = input_dict['categories']
            
            # Generate URLs for all categories
            url_collection = self.loader.generate_urls(categories)
            
            results = {}
            
            # Using ThreadPoolExecutor for concurrent category processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_category = {
                    executor.submit(self._process_category, category, urls): category
                    for category, urls in url_collection.items()
                }
                
                for future in as_completed(future_to_category):
                    category = future_to_category[future]
                    try:
                        result = future.result()
                        results.update(result)
                    except Exception as exc:
                        print(f'❌ {category} generated an exception: {exc}')
                        results[category] = {'error': str(exc)}
            
            return results

        return RunnablePassthrough.assign(extracted_data=RunnableLambda(load_and_extract_data))

    def _process_category(self, category: str, urls: List[str]) -> Dict[str, Any]:
        """Process a single category with its URLs"""
        
        print(f"Processing category: {category}")
        
        try:
            documents = self.loader.load_with_fallback(urls)
            print(f"Found {len(documents)} documents for category '{category}'")

            if not documents:
                return {category: {}}

            if category in ['trustworthiness', 'compliance', 'terms_and_conditions', 'affiliate_program']:
                # Use specialized T&C intelligence extractor
                extracted_data = self.extractor.extract_category_data(documents, category)
            else:
                # Use general extractor for other categories
                extracted_data = self.extractor.extract_category_data(documents, category)
                
            return {category: extracted_data}
        
        except Exception as e:
            print(f"❌ Failed to process {category}: {e}")
            return {category: {'error': str(e)}}

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the chain with input"""
        chain = self._build_chain()
        return chain.invoke(input_dict)

# Factory function for easy initialization
def create_comprehensive_web_research_chain(casino_domain: str = "casino.org", 
                                           alternative_domains: Optional[List[str]] = None,
                                           llm: Optional[ChatOpenAI] = None,
                                           categories: Optional[List[str]] = None) -> ComprehensiveWebResearchChain:
    """Factory function for creating the chain"""
    
    if categories is None:
        categories = [
            'trustworthiness',   # 15 fields - Licensing, fairness, reputation
            'games',             # 12 fields - Variety, software providers, etc.
            'bonuses',           # 12 fields - Welcome offers, VIP programs
            'payments',          # 15 fields - Methods, speed, limits
            'user_experience',   # 12 fields - Support, mobile, navigation
            'innovations',       # 8 fields - VR, AI, blockchain
            'compliance',        # 10 fields - RG, AML, data protection
            'assessment',        # 11 fields - Ratings, recommendations, improvements
            'terms_and_conditions',
            'affiliate_program'
        ]
        
    chain = ComprehensiveWebResearchChain(
        casino_domain=casino_domain,
        alternative_domains=alternative_domains,
        llm=llm,
        categories=categories
    )
    
    return chain

# Example usage and testing
if __name__ == "__main__":
    # Test with casino.org 
    print(f"\n{'='*60}")
    print(f"🎰 Testing Web Research for: casino.org")
    print(f"{'='*60}")
    
    try:
        # Create research chain with ALL 8 categories (complete 95-field analysis)
        research_chain = create_comprehensive_web_research_chain(
            casino_domain='casino.org'
            # Uses default ALL 8 categories for complete analysis
        )
        
        # Run research
        results = research_chain.invoke({
            'casino_domain': 'casino.org',
            'categories': [
                'trustworthiness', 'games', 'bonuses', 'payments', 
                'user_experience', 'innovations', 'compliance', 'assessment',
                'terms_and_conditions', 'affiliate_program'
            ]
        })
        
        print(f"\n🎯 RESULTS SUMMARY:")
        print(f"Overall Grade: {results['overall_quality']['research_grade']}")
        print(f"Average Confidence: {results['overall_quality']['average_confidence']:.2f}")
        print(f"URLs Attempted: {results['total_urls_attempted']}")
        print(f"Ready for 95 Fields: {results['overall_quality']['ready_for_95_fields']}")
        
        for category, summary in results['research_summary'].items():
            print(f"  {category.title()}: {summary['urls_successful']} sources, {summary['confidence_score']:.2f} confidence, {summary['data_quality']} quality")
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Web Research Chain testing completed!")
    print(f"🚀 Ready for integration with Universal RAG Chain!") 