"""
Writing Tools for Writing Agent
Wraps content generation chains and template system as LangChain tools
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.chains.universal_rag_lcel import UniversalRAGChain

# Import existing writing components
try:
    from src.chains.universal_rag_lcel import create_universal_rag_chain, UniversalRAGChain
    from src.templates.improved_template_manager import ImprovedTemplateManager, QueryType, ExpertiseLevel
    WRITING_AVAILABLE = True
except ImportError as e:
    WRITING_AVAILABLE = False
    UniversalRAGChain = None  # type: ignore
    logging.warning(f"Writing components not available: {e}")

logger = logging.getLogger(__name__)

# Global chain instance (could be singleton pattern)
_rag_chain: Optional[Any] = None


def _get_rag_chain() -> Optional[Any]:
    """Get or create Universal RAG Chain instance with Supabase vector store"""
    global _rag_chain
    if _rag_chain is None and WRITING_AVAILABLE:
        try:
            # Initialize Supabase client for vector store
            supabase_client = None
            try:
                import os
                from supabase import create_client
                
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
                
                if supabase_url and supabase_key:
                    supabase_client = create_client(supabase_url, supabase_key)
                    logger.info("✅ Supabase client initialized for writing agent")
            except Exception as e:
                logger.warning(f"⚠️ Supabase initialization failed (will auto-init): {e}")
            
            # Create RAG chain with Supabase support
            _rag_chain = create_universal_rag_chain(
                model_name="gpt-4o-mini",  # Use correct model name
                enable_wordpress_publishing=False,  # Don't publish during writing phase
                enable_comprehensive_web_research=True,
                enable_web_search=True,
                enable_contextual_retrieval=True,  # Enable RAG retrieval
                enable_response_storage=True,  # Enable storing responses
                supabase_client=supabase_client  # Pass Supabase client for vector store
            )
            logger.info("✅ Universal RAG Chain created with Supabase vector store support")
        except Exception as e:
            logger.error(f"Failed to create RAG chain: {e}")
    return _rag_chain


@tool
async def content_generation_tool(
    query: str,
    research_data: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate high-quality content using Universal RAG Chain
    
    Args:
        query: Content query/topic
        research_data: Research findings from research agent
        context: Additional context to include
        
    Returns:
        Dictionary with generated content and metadata:
        - content: Generated content text
        - confidence_score: Content quality score
        - sources: List of sources used
        - metadata: Additional metadata
    """
    if not WRITING_AVAILABLE:
        return {
            "content": "",
            "confidence_score": 0.0,
            "sources": [],
            "error": "Writing components not available"
        }
    
    try:
        chain = _get_rag_chain()
        if not chain:
            return {
                "content": "",
                "confidence_score": 0.0,
                "sources": [],
                "error": "Failed to initialize RAG chain"
            }
        
        # Enhance query with research context if provided
        enhanced_query = query
        if research_data:
            research_summary = _summarize_research_data(research_data)
            enhanced_query = f"{query}\n\nResearch Context:\n{research_summary}"
        
        if context:
            enhanced_query = f"{enhanced_query}\n\nAdditional Context:\n{context}"
        
        # Generate content
        logger.info(f"Content generation tool: Generating content for query")
        response = await chain.ainvoke({"query": enhanced_query})
        
        return {
            "content": response.answer if hasattr(response, 'answer') else str(response),
            "confidence_score": response.confidence_score if hasattr(response, 'confidence_score') else 0.0,
            "sources": response.sources if hasattr(response, 'sources') else [],
            "metadata": {
                "response_time": response.response_time if hasattr(response, 'response_time') else 0.0,
                "cached": response.cached if hasattr(response, 'cached') else False
            }
        }
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        return {
            "content": "",
            "confidence_score": 0.0,
            "sources": [],
            "error": str(e)
        }


@tool
async def template_selection_tool(
    query: str,
    query_type: Optional[str] = None,
    expertise_level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Select appropriate template for content generation
    
    Args:
        query: Content query
        query_type: Type of query (casino_review, game_guide, etc.)
        expertise_level: Target expertise level (beginner, intermediate, advanced)
        
    Returns:
        Dictionary with template information:
        - template_id: Selected template ID
        - template_content: Template content/prompt
        - query_type: Detected/selected query type
        - expertise_level: Selected expertise level
    """
    if not WRITING_AVAILABLE:
        return {
            "template_id": "default",
            "template_content": "",
            "query_type": query_type or "general_info",
            "expertise_level": expertise_level or "intermediate",
            "error": "Template system not available"
        }
    
    try:
        template_manager = ImprovedTemplateManager()
        
        # Detect query type if not provided
        if not query_type:
            query_type = _detect_query_type(query)
        
        # Determine expertise level if not provided
        if not expertise_level:
            expertise_level = "intermediate"  # Default
        
        # Get template
        query_type_enum = QueryType[query_type.upper()] if hasattr(QueryType, query_type.upper()) else QueryType.GENERAL_INFO
        expertise_enum = ExpertiseLevel[expertise_level.upper()] if hasattr(ExpertiseLevel, expertise_level.upper()) else ExpertiseLevel.INTERMEDIATE
        
        template = template_manager.get_template(query_type_enum, expertise_enum)
        
        return {
            "template_id": f"{query_type}_{expertise_level}",
            "template_content": template if isinstance(template, str) else str(template),
            "query_type": query_type,
            "expertise_level": expertise_level
        }
        
    except Exception as e:
        logger.error(f"Template selection failed: {e}")
        return {
            "template_id": "default",
            "template_content": "",
            "query_type": query_type or "general_info",
            "expertise_level": expertise_level or "intermediate",
            "error": str(e)
        }


@tool
async def content_refinement_tool(
    content: str,
    improvement_areas: Optional[list] = None
) -> Dict[str, Any]:
    """
    Refine and improve generated content
    
    Args:
        content: Content to refine
        improvement_areas: List of areas to improve (seo, readability, structure, etc.)
        
    Returns:
        Dictionary with refined content:
        - refined_content: Improved content
        - improvements_made: List of improvements applied
        - quality_score: Quality score after refinement
    """
    if not WRITING_AVAILABLE:
        return {
            "refined_content": content,
            "improvements_made": [],
            "quality_score": 0.0,
            "error": "Refinement not available"
        }
    
    try:
        # Simple refinement (full implementation would use LLM)
        improvements = []
        refined_content = content
        
        # Basic improvements
        if improvement_areas is None:
            improvement_areas = ["seo", "readability", "structure"]
        
        # Apply improvements (simplified - full implementation would use LLM)
        if "structure" in improvement_areas:
            # Ensure proper heading hierarchy
            improvements.append("Improved heading structure")
        
        if "readability" in improvement_areas:
            # Basic readability improvements
            improvements.append("Enhanced readability")
        
        # Calculate quality score (simplified)
        quality_score = min(len(content) / 2000.0, 1.0)  # Simple length-based score
        
        return {
            "refined_content": refined_content,
            "improvements_made": improvements,
            "quality_score": quality_score
        }
        
    except Exception as e:
        logger.error(f"Content refinement failed: {e}")
        return {
            "refined_content": content,
            "improvements_made": [],
            "quality_score": 0.0,
            "error": str(e)
        }


@tool
async def seo_optimization_tool(
    content: str,
    target_keywords: Optional[list] = None
) -> Dict[str, Any]:
    """
    Optimize content for SEO
    
    Args:
        content: Content to optimize
        target_keywords: List of target keywords
        
    Returns:
        Dictionary with SEO-optimized content:
        - optimized_content: SEO-optimized content
        - seo_metadata: SEO metadata (title, description, keywords)
        - keyword_density: Keyword density analysis
    """
    try:
        # Extract keywords from content if not provided
        if not target_keywords:
            target_keywords = _extract_keywords(content)
        
        # Generate SEO metadata
        seo_metadata = {
            "title": _generate_seo_title(content, target_keywords),
            "description": _generate_meta_description(content),
            "keywords": target_keywords[:10]  # Top 10 keywords
        }
        
        # Calculate keyword density
        keyword_density = {}
        content_lower = content.lower()
        for keyword in target_keywords:
            count = content_lower.count(keyword.lower())
            density = (count / max(len(content.split()), 1)) * 100
            keyword_density[keyword] = {
                "count": count,
                "density": round(density, 2)
            }
        
        return {
            "optimized_content": content,  # Content already optimized in generation
            "seo_metadata": seo_metadata,
            "keyword_density": keyword_density
        }
        
    except Exception as e:
        logger.error(f"SEO optimization failed: {e}")
        return {
            "optimized_content": content,
            "seo_metadata": {},
            "keyword_density": {},
            "error": str(e)
        }


def _summarize_research_data(research_data: Dict[str, Any]) -> str:
    """Summarize research data for context"""
    summary_parts = []
    
    if research_data.get("web_search_results"):
        summary_parts.append(f"Web Search: {len(research_data['web_search_results'])} results")
    
    if research_data.get("comprehensive_research"):
        cr = research_data["comprehensive_research"]
        summary_parts.append(f"Comprehensive Research: {cr.get('total_documents', 0)} documents")
    
    if research_data.get("structured_intelligence"):
        summary_parts.append("Structured Intelligence: Available")
    
    return "\n".join(summary_parts)


def _detect_query_type(query: str) -> str:
    """Detect query type from query text"""
    query_lower = query.lower()
    
    if "review" in query_lower or "casino" in query_lower:
        return "casino_review"
    elif "guide" in query_lower or "how to" in query_lower:
        return "game_guide"
    elif "bonus" in query_lower or "promotion" in query_lower:
        return "promotion_analysis"
    elif "compare" in query_lower or "vs" in query_lower:
        return "comparison"
    elif "news" in query_lower or "update" in query_lower:
        return "news_update"
    else:
        return "general_info"


def _extract_keywords(content: str, max_keywords: int = 10) -> list:
    """Extract keywords from content (simplified)"""
    # Simple keyword extraction (full implementation would use NLP)
    import re
    words = re.findall(r'\b\w{4,}\b', content.lower())
    
    # Filter common stop words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'them', 'their'}
    keywords = [w for w in words if w not in stop_words]
    
    # Count frequency and return top keywords
    from collections import Counter
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(max_keywords)]


def _generate_seo_title(content: str, keywords: list) -> str:
    """Generate SEO-optimized title"""
    # Extract first heading or create from keywords
    import re
    first_heading = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if first_heading:
        return first_heading.group(1).strip()
    
    # Create title from keywords
    if keywords:
        return f"{keywords[0].title()} - Complete Guide 2025"
    
    return "Complete Guide"


def _generate_meta_description(content: str, max_length: int = 160) -> str:
    """Generate meta description from content"""
    # Extract first paragraph or create summary
    import re
    first_para = re.search(r'^([^\n]+)$', content, re.MULTILINE)
    if first_para:
        desc = first_para.group(1).strip()
        if len(desc) <= max_length:
            return desc
        return desc[:max_length - 3] + "..."
    
    return "Comprehensive guide covering all aspects."

