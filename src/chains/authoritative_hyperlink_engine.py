"""
Authoritative Hyperlink Generation Engine using LangChain LCEL
Implements semantic-based contextual linking for casino content
"""

from langchain_core.runnables import (
    RunnableSequence, 
    RunnableLambda, 
    RunnablePassthrough,
    RunnableParallel
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import logging
import re
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

# ============= MODELS =============

class LinkCategory(Enum):
    """Categories of authoritative links"""
    RESPONSIBLE_GAMBLING = "responsible_gambling"
    REGULATORY = "regulatory"
    GAME_PROVIDERS = "game_providers"
    PAYMENT_SECURITY = "payment_security"
    INDUSTRY_STANDARDS = "industry_standards"

class AuthorityLink(BaseModel):
    """Model for an authoritative link"""
    url: str
    name: str
    category: LinkCategory
    keywords: List[str] = Field(description="Keywords that trigger this link")
    anchor_variations: List[str] = Field(description="Different anchor text variations")
    context_requirements: List[str] = Field(default_factory=list, description="Required context for linking")
    authority_score: float = Field(default=1.0, description="Authority score 0-1")
    description: str = Field(default="")

class LinkPlacement(BaseModel):
    """Model for where to place a link in content"""
    original_text: str
    linked_text: str
    start_position: int
    end_position: int
    link: AuthorityLink
    confidence_score: float

class LinkGenerationConfig(BaseModel):
    """Configuration for link generation"""
    max_links_per_content: int = Field(default=8)
    max_links_per_category: int = Field(default=3)
    min_confidence_score: float = Field(default=0.7)
    prefer_first_occurrence: bool = Field(default=True)
    enable_semantic_matching: bool = Field(default=True)
    enable_compliance_section: bool = Field(default=True)

# ============= AUTHORITY LINK DATABASE =============

class AuthorityLinkDatabase:
    """Database of authoritative links with semantic search"""
    
    def __init__(self, embeddings_model: Optional[OpenAIEmbeddings] = None):
        self.embeddings = embeddings_model or OpenAIEmbeddings()
        self.links = self._initialize_links()
        self.vector_store = self._create_vector_store()
    
    def _initialize_links(self) -> List[AuthorityLink]:
        """Initialize the authority link database"""
        return [
            # Responsible Gambling Resources
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
            
            # Regulatory Authorities
            AuthorityLink(
                url="https://www.gamblingcommission.gov.uk/",
                name="UK Gambling Commission",
                category=LinkCategory.REGULATORY,
                keywords=["gambling license", "UKGC", "regulated", "licensing", "gambling commission"],
                anchor_variations=["UK Gambling Commission", "UKGC", "gambling regulator"],
                context_requirements=["license", "regulation", "UK"],
                authority_score=1.0,
                description="UK's gambling regulator"
            ),
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
            
            # Game Providers
            AuthorityLink(
                url="https://www.microgaming.com/",
                name="Microgaming",
                category=LinkCategory.GAME_PROVIDERS,
                keywords=["Microgaming", "slots", "progressive jackpots", "Mega Moolah"],
                anchor_variations=["Microgaming", "Microgaming slots", "progressive slots"],
                context_requirements=["games", "slots", "software"],
                authority_score=0.88,
                description="Leading casino game provider"
            ),
            AuthorityLink(
                url="https://www.netent.com/",
                name="NetEnt",
                category=LinkCategory.GAME_PROVIDERS,
                keywords=["NetEnt", "Starburst", "Gonzo's Quest", "slot games"],
                anchor_variations=["NetEnt", "NetEnt games", "premium slots"],
                context_requirements=["games", "slots", "software"],
                authority_score=0.89,
                description="Premium casino game developer"
            ),
            
            # Payment Security
            AuthorityLink(
                url="https://www.pcisecuritystandards.org/",
                name="PCI Security Standards",
                category=LinkCategory.PAYMENT_SECURITY,
                keywords=["PCI DSS", "payment security", "card security", "financial security"],
                anchor_variations=["PCI DSS", "payment security standards", "card protection"],
                context_requirements=["payment", "security", "banking"],
                authority_score=0.96,
                description="Payment card industry security standards"
            ),
            AuthorityLink(
                url="https://www.ssl.com/",
                name="SSL Certificates",
                category=LinkCategory.PAYMENT_SECURITY,
                keywords=["SSL", "encryption", "secure connection", "HTTPS"],
                anchor_variations=["SSL encryption", "secure connection", "SSL certificates"],
                context_requirements=["security", "encryption", "secure"],
                authority_score=0.85,
                description="SSL encryption and security"
            )
        ]
    
    def _create_vector_store(self) -> FAISS:
        """Create vector store for semantic matching"""
        try:
            documents = []
            for link in self.links:
                # Create searchable content from link metadata
                content = f"{link.name} {link.description} {' '.join(link.keywords)} {' '.join(link.anchor_variations)}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "link_id": f"{link.category.value}_{link.name.lower().replace(' ', '_')}",
                        "url": link.url,
                        "name": link.name,
                        "category": link.category.value,
                        "authority_score": link.authority_score,
                        "keywords": link.keywords,
                        "anchor_variations": link.anchor_variations,
                        "context_requirements": link.context_requirements
                    }
                )
                documents.append(doc)
            
            if documents:
                return FAISS.from_documents(documents, self.embeddings)
            else:
                # Create empty vector store
                return FAISS.from_texts(["empty"], self.embeddings)
        except Exception as e:
            logger.warning(f"Failed to create vector store: {e}")
            # Fallback to empty vector store
            return FAISS.from_texts(["empty"], self.embeddings)
    
    async def find_relevant_links(self, content: str, query: str = "", max_links: int = 10) -> List[AuthorityLink]:
        """Find relevant links using semantic search"""
        try:
            search_text = f"{content} {query}"
            similar_docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                search_text,
                k=max_links
            )
            
            relevant_links = []
            for doc in similar_docs:
                metadata = doc.metadata
                if metadata.get("name") != "empty":  # Skip empty placeholder
                    # Find the corresponding link
                    for link in self.links:
                        if (link.name == metadata.get("name") and 
                            link.category.value == metadata.get("category")):
                            relevant_links.append(link)
                            break
            
            return relevant_links
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

# ============= MAIN HYPERLINK ENGINE =============

class AuthoritativeHyperlinkEngine:
    """Main engine for generating authoritative hyperlinks"""
    
    def __init__(self, config: LinkGenerationConfig, llm: Optional[ChatOpenAI] = None):
        self.config = config
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.link_db = AuthorityLinkDatabase()
    
    async def generate_hyperlinks(
        self, 
        content: str, 
        structured_data: Optional[Dict[str, Any]] = None,
        query: str = ""
    ) -> Dict[str, Any]:
        """Generate hyperlinks for content using LCEL pipeline"""
        
        # Create LCEL chain for hyperlink generation
        hyperlink_chain = (
            RunnablePassthrough.assign(
                relevant_links=RunnableLambda(self._find_relevant_links)
            )
            | RunnablePassthrough.assign(
                enhanced_content=RunnableLambda(self._apply_basic_hyperlinks)
            )
            | RunnableLambda(self._format_result)
        )
        
        # Execute chain
        try:
            result = await hyperlink_chain.ainvoke({
                "content": content,
                "structured_data": structured_data or {},
                "query": query,
                "config": self.config
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Hyperlink generation failed: {e}")
            return {
                "enhanced_content": content,
                "links_added": 0,
                "placements": [],
                "error": str(e)
            }
    
    async def _find_relevant_links(self, inputs: Dict[str, Any]) -> List[AuthorityLink]:
        """Find relevant links for content"""
        content = inputs["content"]
        query = inputs.get("query", "")
        structured_data = inputs.get("structured_data", {})
        
        # Enhance search with structured data context
        search_context = content
        if structured_data:
            # Add relevant structured data to search context
            if "trustworthiness" in structured_data:
                search_context += f" {structured_data['trustworthiness']}"
            if "games" in structured_data:
                search_context += f" {structured_data['games']}"
        
        relevant_links = await self.link_db.find_relevant_links(
            content=search_context,
            query=query,
            max_links=self.config.max_links_per_content * 2  # Get more for filtering
        )
        
        return relevant_links
    
    async def _apply_basic_hyperlinks(self, inputs: Dict[str, Any]) -> str:
        """Apply basic hyperlinks to content using keyword matching"""
        content = inputs["content"]
        relevant_links = inputs["relevant_links"]
        
        if not relevant_links:
            return content
        
        enhanced_content = content
        links_applied = 0
        category_counts = {}
        
        for link in relevant_links:
            if links_applied >= self.config.max_links_per_content:
                break
                
            # Check category limits
            category = link.category.value
            if category_counts.get(category, 0) >= self.config.max_links_per_category:
                continue
            
            # Try to find keywords in content
            for keyword in link.keywords:
                if keyword.lower() in enhanced_content.lower():
                    # Check context requirements
                    context_match = True
                    if link.context_requirements:
                        context_match = any(
                            req.lower() in enhanced_content.lower() 
                            for req in link.context_requirements
                        )
                    
                    if context_match:
                        # Use first anchor variation as anchor text
                        anchor_text = link.anchor_variations[0] if link.anchor_variations else link.name
                        
                        # Create HTML link
                        html_link = f'<a href="{link.url}" target="_blank" rel="noopener noreferrer" title="{link.description}">{anchor_text}</a>'
                        
                        # Replace first occurrence of keyword with link
                        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                        if pattern.search(enhanced_content):
                            enhanced_content = pattern.sub(html_link, enhanced_content, count=1)
                            links_applied += 1
                            category_counts[category] = category_counts.get(category, 0) + 1
                            break
        
        return enhanced_content
    
    def _format_result(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format final result"""
        return {
            "enhanced_content": inputs["enhanced_content"],
            "links_added": len([l for l in inputs["relevant_links"]]),
            "relevant_links_found": len(inputs["relevant_links"]),
            "config": self.config.dict()
        }

# ============= FACTORY FUNCTION =============

def create_authoritative_hyperlink_engine(
    max_links: int = 8,
    max_per_category: int = 3,
    min_confidence: float = 0.7,
    llm: Optional[ChatOpenAI] = None
) -> AuthoritativeHyperlinkEngine:
    """Factory function to create hyperlink engine"""
    config = LinkGenerationConfig(
        max_links_per_content=max_links,
        max_links_per_category=max_per_category,
        min_confidence_score=min_confidence
    )
    
    return AuthoritativeHyperlinkEngine(config=config, llm=llm) 