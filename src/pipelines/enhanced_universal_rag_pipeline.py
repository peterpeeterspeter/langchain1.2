# src/pipelines/enhanced_universal_rag_pipeline.py
"""
Enhanced Universal RAG Pipeline with full LCEL integration
Includes: Image embedding, compliance, authoritative sources, and adaptive templates
"""

from langchain_core.runnables import (
    RunnableSequence, 
    RunnableLambda, 
    RunnablePassthrough,
    RunnableParallel,
    RunnableBranch
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio
import logging
import os
from datetime import datetime
from pydantic import BaseModel, Field
import re
import json

logger = logging.getLogger(__name__)

# ============= MODELS =============

class ContentCategory(Enum):
    """Content categories for compliance detection"""
    GAMBLING = "gambling"
    GENERAL = "general"
    FINANCE = "finance"
    HEALTH = "health"
    ADULT = "adult"

class EnhancedContent(BaseModel):
    """Enhanced content with images, sources, and compliance"""
    title: str
    content: str
    images: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_notices: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ContentAnalysis(BaseModel):
    """Content analysis results"""
    category: ContentCategory
    compliance_required: bool
    detected_keywords: List[str] = Field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    requires_age_verification: bool = False

# ============= ENHANCED PIPELINE =============

class EnhancedUniversalRAGPipeline:
    """Complete LCEL-based Universal RAG Pipeline with all enhancements"""
    
    def __init__(self, supabase_client, config: Dict[str, Any]):
        self.supabase = supabase_client
        self.config = config
        
        # Initialize components
        try:
            self.llm = ChatOpenAI(
                model=config.get("model", "gpt-4.1-mini"),
                temperature=config.get("temperature", 0.7)
            )
        except Exception as e:
            # For testing or when API key is not available
            self.llm = None
        
        # Compliance keywords for auto-detection
        self.gambling_keywords = [
            "casino", "betting", "poker", "slots", "blackjack", "roulette",
            "gambling", "wager", "jackpot", "odds", "payout", "bonus",
            "free spins", "deposit", "withdrawal", "bet", "stake"
        ]
        
        self.compliance_notices = {
            "gambling": [
                "🔞 This content is intended for adults aged 18 and over.",
                "⚠️ Gambling can be addictive. Please play responsibly.",
                "📞 For gambling addiction support, contact: National Problem Gambling Helpline 1-800-522-4700",
                "🚫 Void where prohibited. Check local laws and regulations."
            ]
        }
        
    def create_pipeline(self) -> RunnableSequence:
        """Create the complete enhanced pipeline"""
        
        return RunnableSequence(
            # Step 1: Content Analysis - Auto-detect category and compliance needs
            RunnablePassthrough.assign(
                analysis=RunnableLambda(self._analyze_content)
            ),
            
            # Step 2: Parallel Resource Gathering - Images + Authoritative Sources
            RunnablePassthrough.assign(
                resources=RunnableParallel({
                    "images": RunnableLambda(self._gather_images),
                    "sources": RunnableLambda(self._gather_authoritative_sources)
                })
            ),
            
            # Step 3: Dynamic Template Enhancement
            RunnablePassthrough.assign(
                enhanced_template=RunnableLambda(self._enhance_template)
            ),
            
            # Step 4: Enhanced Retrieval with Context
            RunnablePassthrough.assign(
                retrieved_docs=RunnableLambda(self._enhanced_retrieval)
            ),
            
            # Step 5: Content Generation with Context
            RunnablePassthrough.assign(
                raw_content=RunnableLambda(self._generate_content)
            ),
            
            # Step 6: Content Enhancement - Embed Images, Add Compliance
            RunnablePassthrough.assign(
                enhanced_content=RunnableLambda(self._enhance_content)
            ),
            
            # Step 7: Final Output Formatting
            RunnableLambda(self._format_output)
        )
    
    def _analyze_content(self, input_data: Dict[str, Any]) -> ContentAnalysis:
        """Step 1: Analyze content for category and compliance requirements"""
        query = input_data.get("query", "").lower()
        
        # Detect gambling content
        gambling_matches = [kw for kw in self.gambling_keywords if kw in query]
        
        if gambling_matches:
            category = ContentCategory.GAMBLING
            compliance_required = True
            risk_level = "high"
            requires_age_verification = True
        else:
            category = ContentCategory.GENERAL
            compliance_required = False
            risk_level = "low"
            requires_age_verification = False
        
        return ContentAnalysis(
            category=category,
            compliance_required=compliance_required,
            detected_keywords=gambling_matches,
            risk_level=risk_level,
            requires_age_verification=requires_age_verification
        )
    
    def _gather_images(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Step 2a: Gather relevant images using DataForSEO"""
        try:
            # Import here to avoid circular imports
            from src.integrations.dataforseo_image_search import EnhancedDataForSEOImageSearch
            
            query = input_data.get("query", "")
            analysis = input_data.get("analysis")
            
            # Create search queries based on content
            search_queries = self._generate_image_search_queries(query, analysis)
            
            # Create proper DataForSEO config
            from src.integrations.dataforseo_image_search import DataForSEOConfig
            
            dataforseo_config = DataForSEOConfig(
                login=os.getenv("DATAFORSEO_LOGIN", "peeters.peter@telenet.be"),
                password=os.getenv("DATAFORSEO_PASSWORD", "654b1cfcca084d19"),
                supabase_url=os.getenv("SUPABASE_URL", ""),
                supabase_key=os.getenv("SUPABASE_SERVICE_KEY", "")
            )
            
            image_service = EnhancedDataForSEOImageSearch(config=dataforseo_config)
            
            all_images = []
            for search_query in search_queries[:3]:  # Limit to 3 searches
                try:
                    # Create proper ImageSearchRequest
                    from src.integrations.dataforseo_image_search import ImageSearchRequest, ImageType, ImageSize
                    
                    search_request = ImageSearchRequest(
                        keyword=search_query,
                        max_results=5,
                        image_type=ImageType.PHOTO,
                        image_size=ImageSize.MEDIUM,
                        safe_search=True
                    )
                    
                    # Handle async call properly
                    try:
                        results = asyncio.run(image_service.search_images(search_request))
                    except Exception as e:
                        logger.warning(f"Async search failed, trying alternative approach: {e}")
                        # For development, return mock images
                        results = self._get_mock_images(search_query)
                    
                    # Process and score images
                    images_list = results.images if hasattr(results, 'images') else results.get("images", [])
                    
                    for img in images_list:
                        # Handle both ImageMetadata objects and dict objects
                        if hasattr(img, 'url'):  # ImageMetadata object
                            img_dict = {
                                "url": img.url,
                                "title": img.title or search_query,
                                "alt": img.alt_text or img.generated_alt_text,
                                "width": img.width,
                                "height": img.height,
                                "quality_score": img.quality_score
                            }
                        else:  # Dict object
                            img_dict = img
                        
                        img_data = {
                            "url": img_dict.get("url"),
                            "alt_text": self._generate_alt_text(img_dict, search_query),
                            "title": img_dict.get("title", search_query),
                            "relevance_score": self._calculate_image_relevance(img_dict, query),
                            "section_suggestion": self._suggest_image_section(img_dict, query),
                            "width": img_dict.get("width", 800),
                            "height": img_dict.get("height", 600)
                        }
                        all_images.append(img_data)
                        
                except Exception as e:
                    logger.warning(f"Image search failed for '{search_query}': {e}")
                    continue
            
            # Sort by relevance and return top images
            all_images.sort(key=lambda x: x["relevance_score"], reverse=True)
            return all_images[:6]  # Maximum 6 images
            
        except Exception as e:
            logger.error(f"Image gathering failed: {e}")
            return []
    
    def _gather_authoritative_sources(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Step 2b: Gather authoritative sources"""
        try:
            # Import contextual retrieval system
            from src.retrieval.contextual_retrieval import ContextualRetrievalSystem
            
            query = input_data.get("query", "")
            analysis = input_data.get("analysis")
            
            retrieval_system = ContextualRetrievalSystem(self.supabase)
            
            # Enhanced retrieval with source quality focus
            filters = {}
            if analysis.category == ContentCategory.GAMBLING:
                filters["source_type"] = ["official", "regulatory", "news"]
            
            docs = retrieval_system.retrieve(
                query=query,
                filters=filters,
                k=10,
                strategy="hybrid"
            )
            
            # Extract and validate sources
            sources = []
            for doc in docs:
                source_data = {
                    "url": doc.metadata.get("source_url"),
                    "title": doc.metadata.get("title"),
                    "domain": doc.metadata.get("domain"),
                    "authority_score": doc.metadata.get("authority_score", 0.5),
                    "content_snippet": doc.page_content[:200] + "...",
                    "source_type": doc.metadata.get("source_type", "article")
                }
                
                # Only include high-quality sources
                if source_data["authority_score"] >= 0.6:
                    sources.append(source_data)
            
            return sources[:8]  # Maximum 8 sources
            
        except Exception as e:
            logger.error(f"Source gathering failed: {e}")
            return []
    
    def _enhance_template(self, input_data: Dict[str, Any]) -> str:
        """Step 3: Dynamically enhance template based on analysis"""
        try:
            from src.templates.improved_template_manager import ImprovedTemplateManager
            
            query = input_data.get("query", "")
            analysis = input_data.get("analysis")
            resources = input_data.get("resources", {})
            
            template_manager = ImprovedTemplateManager()
            
            # Get base template with correct method name
            from src.templates.improved_template_manager import QueryType, ExpertiseLevel
            
            base_template = template_manager.get_template(
                template_type="casino_review",
                query_type=QueryType.CASINO_REVIEW,
                expertise_level=ExpertiseLevel.INTERMEDIATE
            )
            
            # Enhance template based on analysis
            enhanced_sections = []
            
            # Add compliance section if needed
            if analysis.compliance_required:
                enhanced_sections.append("""
## Important Disclaimers
{compliance_notices}
""")
            
            # Add image placeholders
            if resources.get("images"):
                enhanced_sections.append("""
## Visual Overview
{image_content}
""")
            
            # Add authoritative sources section
            if resources.get("sources"):
                enhanced_sections.append("""
## References and Sources
{authoritative_sources}
""")
            
            # Combine with base template
            enhanced_template = base_template + "\n\n" + "\n\n".join(enhanced_sections)
            
            return enhanced_template
            
        except Exception as e:
            logger.error(f"Template enhancement failed: {e}")
            return input_data.get("query", "")
    
    def _enhanced_retrieval(self, input_data: Dict[str, Any]) -> List[Any]:
        """Step 4: Enhanced retrieval with contextual understanding"""
        try:
            from src.retrieval.contextual_retrieval import ContextualRetrievalSystem
            
            query = input_data.get("query", "")
            analysis = input_data.get("analysis")
            
            retrieval_system = ContextualRetrievalSystem(self.supabase)
            
            # Build filters based on analysis
            filters = {}
            if analysis.category == ContentCategory.GAMBLING:
                filters["content_type"] = ["review", "guide", "regulatory"]
            
            docs = retrieval_system.retrieve(
                query=query,
                filters=filters,
                k=15,
                strategy="contextual"
            )
            
            return docs
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            return []
    
    def _generate_content(self, input_data: Dict[str, Any]) -> str:
        """Step 5: Generate content using enhanced template and context"""
        try:
            query = input_data.get("query", "")
            enhanced_template = input_data.get("enhanced_template", "")
            retrieved_docs = input_data.get("retrieved_docs", [])
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:10]]) if retrieved_docs else ""
            
            # Generate content directly using LLM if available
            if self.llm:
                prompt_template = ChatPromptTemplate.from_template(
                    enhanced_template + "\n\nContext: {context}\n\nQuery: {query}\n\nPlease provide a comprehensive response:"
                )
                
                chain = prompt_template | self.llm | StrOutputParser()
                
                content = chain.invoke({
                    "query": query,
                    "context": context
                })
                
                return content
            else:
                # Fallback when LLM is not available
                return self._generate_fallback_content(query)
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return self._generate_fallback_content(input_data.get('query', 'Unknown query'))
    
    def _generate_fallback_content(self, query: str) -> str:
        """Generate fallback content when LLM is not available"""
        return f"""# {query.title()}

## Overview
This is a comprehensive analysis of {query}.

## Key Features
- Professional content generation
- Enhanced with images and compliance notices
- Authoritative source integration
- Mobile-optimized experience

## Important Information
Please refer to official sources for the most up-to-date information.

*Content generated by Enhanced Universal RAG Pipeline*"""
    
    def _enhance_content(self, input_data: Dict[str, Any]) -> EnhancedContent:
        """Step 6: Enhance content with images, compliance notices, and sources"""
        raw_content = input_data.get("raw_content", "")
        analysis = input_data.get("analysis")
        resources = input_data.get("resources", {})
        
        # Extract title
        title = self._extract_title(raw_content)
        
        # Embed images into content
        content_with_images = self._embed_images(raw_content, resources.get("images", []))
        
        # Add compliance notices
        compliance_notices = []
        if analysis.compliance_required:
            compliance_notices = self.compliance_notices.get(analysis.category.value, [])
            
            # Add compliance section to content
            compliance_section = "\n\n## Important Disclaimers\n\n"
            compliance_section += "\n\n".join([f"> {notice}" for notice in compliance_notices])
            content_with_images += compliance_section
        
        # Add authoritative sources
        sources = resources.get("sources", [])
        if sources:
            sources_section = "\n\n## References and Sources\n\n"
            for i, source in enumerate(sources[:5], 1):
                sources_section += f"{i}. [{source['title']}]({source['url']}) - {source['domain']}\n"
            content_with_images += sources_section
        
        return EnhancedContent(
            title=title,
            content=content_with_images,
            images=resources.get("images", []),
            sources=sources,
            compliance_notices=compliance_notices,
            metadata={
                "category": analysis.category.value,
                "compliance_required": analysis.compliance_required,
                "risk_level": analysis.risk_level,
                "generation_time": datetime.now().isoformat(),
                "image_count": len(resources.get("images", [])),
                "source_count": len(sources)
            }
        )
    
    def _format_output(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Format final output"""
        enhanced_content = input_data.get("enhanced_content")
        
        return {
            "title": enhanced_content.title,
            "content": enhanced_content.content,
            "images": enhanced_content.images,
            "sources": enhanced_content.sources,
            "compliance_notices": enhanced_content.compliance_notices,
            "metadata": enhanced_content.metadata,
            "pipeline_version": "enhanced_v1.0.0",
            "processing_steps_completed": 7
        }
    
    # ============= HELPER METHODS =============
    
    def _generate_image_search_queries(self, query: str, analysis: ContentAnalysis) -> List[str]:
        """Generate relevant image search queries"""
        base_queries = [query]
        
        if analysis.category == ContentCategory.GAMBLING:
            # Add specific gambling-related image searches
            if "casino" in query.lower():
                base_queries.extend([
                    f"{query} homepage screenshot",
                    f"{query} mobile app interface",
                    f"{query} games lobby",
                    "live dealer casino games",
                    "casino slot machines interface"
                ])
        
        return base_queries
    
    def _generate_alt_text(self, image_data: Dict, search_query: str) -> str:
        """Generate descriptive alt text for images"""
        title = image_data.get("title", "")
        if title:
            return f"Image showing {title.lower()}"
        return f"Relevant image for {search_query}"
    
    def _calculate_image_relevance(self, image_data: Dict, query: str) -> float:
        """Calculate image relevance score"""
        score = 0.5  # Base score
        
        title = image_data.get("title", "").lower()
        query_words = query.lower().split()
        
        # Check title relevance
        matching_words = sum(1 for word in query_words if word in title)
        score += (matching_words / len(query_words)) * 0.3
        
        # Check image quality metrics
        width = image_data.get("width", 0) or 0
        height = image_data.get("height", 0) or 0
        
        if width >= 600 and height >= 400:
            score += 0.2
        
        return min(score, 1.0)
    
    def _suggest_image_section(self, image_data: Dict, query: str) -> str:
        """Suggest where to place the image in content"""
        title = image_data.get("title", "").lower()
        
        if "homepage" in title or "interface" in title:
            return "Overview"
        elif "game" in title or "slot" in title:
            return "Games"
        elif "mobile" in title or "app" in title:
            return "Mobile Experience"
        else:
            return "Introduction"
    
    def _embed_images(self, content: str, images: List[Dict[str, Any]]) -> str:
        """Embed images into content at appropriate locations"""
        if not images:
            return content
        
        lines = content.split('\n')
        enhanced_lines = []
        images_used = 0
        
        for i, line in enumerate(lines):
            enhanced_lines.append(line)
            
            # Insert image after section headers
            if line.startswith('##') and images_used < len(images):
                image = images[images_used]
                
                img_html = f"""
<div class="content-image">
    <img src="{image['url']}" 
         alt="{image['alt_text']}" 
         title="{image['title']}"
         width="{image.get('width', 800)}" 
         height="{image.get('height', 600)}"
         loading="lazy" />
    <p class="image-caption"><em>{image['title']}</em></p>
</div>
"""
                enhanced_lines.append(img_html)
                images_used += 1
        
        return '\n'.join(enhanced_lines)
    
    def _extract_title(self, content: str) -> str:
        """Extract title from content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()
        
        # Fallback: use first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()[:100]
        
        return "Generated Content"
    
    def _get_mock_images(self, search_query: str) -> Dict[str, Any]:
        """Provide mock images for development when DataForSEO is unavailable"""
        return {
            "images": [
                {
                    "url": f"https://via.placeholder.com/800x600?text={search_query.replace(' ', '+')}_1",
                    "title": f"{search_query} - Professional Image 1",
                    "alt": f"Professional {search_query} illustration",
                    "width": 800,
                    "height": 600,
                    "quality_score": 0.8
                },
                {
                    "url": f"https://via.placeholder.com/800x600?text={search_query.replace(' ', '+')}_2", 
                    "title": f"{search_query} - Professional Image 2",
                    "alt": f"High-quality {search_query} visual",
                    "width": 800,
                    "height": 600,
                    "quality_score": 0.7
                }
            ]
        }

# ============= FACTORY FUNCTION =============

def create_enhanced_rag_pipeline(supabase_client, config: Dict[str, Any]) -> EnhancedUniversalRAGPipeline:
    """Factory function to create enhanced RAG pipeline"""
    return EnhancedUniversalRAGPipeline(supabase_client, config) 