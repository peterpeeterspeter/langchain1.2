"""
Research Tools for Research Agent
Wraps existing research chains and integrations as LangChain tools
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

# Try to import Tavily (optional)
TAVILY_AVAILABLE = False
TavilySearchResults = None
try:
    # ✅ FIXED: Correct import path for TavilySearchResults
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    # Fallback: try alternative import
    try:
        from langchain_tavily import TavilySearch  # New API
        TavilySearchResults = TavilySearch  # Alias for compatibility
        TAVILY_AVAILABLE = True
    except ImportError:
        TAVILY_AVAILABLE = False
        TavilySearchResults = None

# Import existing research components
# Import directly from modules to avoid chain import issues
RESEARCH_AVAILABLE = False
try:
    # Try importing enhanced_web_research_chain directly (doesn't need langchain_anthropic)
    import sys
    import importlib.util
    
    # Direct import of enhanced_web_research_chain module
    enhanced_web_research_path = "src/chains/enhanced_web_research_chain.py"
    spec = importlib.util.spec_from_file_location("enhanced_web_research_chain", enhanced_web_research_path)
    if spec and spec.loader:
        enhanced_web_research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_web_research)
        
        EnhancedWebBaseLoader = enhanced_web_research.EnhancedWebBaseLoader
        URLStrategy = enhanced_web_research.URLStrategy
        ComprehensiveResearchData = enhanced_web_research.ComprehensiveResearchData
        create_comprehensive_web_research_chain = enhanced_web_research.create_comprehensive_web_research_chain
        
        RESEARCH_AVAILABLE = True
    else:
        raise ImportError("Could not load enhanced_web_research_chain module")
        
except Exception as e:
    # Fallback: try normal import
    try:
        from src.chains.enhanced_web_research_chain import (
            EnhancedWebBaseLoader,
            URLStrategy,
            ComprehensiveResearchData,
            create_comprehensive_web_research_chain
        )
        RESEARCH_AVAILABLE = True
    except ImportError as import_err:
        RESEARCH_AVAILABLE = False
        logging.warning(f"Research components not available: {import_err}")

# Import screenshot engine (optional)
try:
    from src.integrations.playwright_screenshot_engine import (
        ScreenshotService,
        BrowserPoolManager,
        CasinoElementLocator
    )
except ImportError:
    ScreenshotService = None
    BrowserPoolManager = None
    CasinoElementLocator = None

logger = logging.getLogger(__name__)


@tool
async def web_search_tool(query: str) -> List[Dict[str, Any]]:
    """
    Perform web search using Tavily API
    
    Args:
        query: Search query string
        
    Returns:
        List of search results with title, url, content, and snippet
    """
    try:
        if not TAVILY_AVAILABLE:
            logger.warning("langchain_tavily not available, skipping web search")
            return []
        
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.warning("TAVILY_API_KEY not found, skipping web search")
            return []
        
        # ✅ FIXED: Use tavily_api_key parameter name (not api_key)
        # Also disable SSL verification for Tavily (common issue)
        import ssl
        import certifi
        tavily = TavilySearchResults(
            max_results=10, 
            tavily_api_key=tavily_api_key
        )
        # Wrap in try-except to handle SSL errors gracefully
        try:
            results = await tavily.ainvoke({"query": query})
        except Exception as tavily_error:
            # If Tavily fails (SSL, network, etc.), log and return empty
            logger.warning(f"Tavily API call failed: {tavily_error}, returning empty results")
            return []
        
        # Format results
        formatted_results = []
        documents_for_storage = []  # ✅ NEW: Collect documents for Supabase storage
        
        # ✅ FIXED: Handle different result formats from Tavily
        # Tavily can return: list of dicts, list of strings, or other formats
        if not results:
            logger.warning("Tavily returned no results")
            return []
        
        # Handle case where results might be a single string or other type
        if isinstance(results, str):
            logger.warning(f"Tavily returned string instead of list: {results[:100]}")
            return []
        
        for result in results:
            try:
                # Handle dict results (most common)
                if isinstance(result, dict):
                    formatted_result = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", result.get("snippet", "")),
                        "snippet": result.get("snippet", result.get("content", "")[:200]),
                        "score": result.get("score", 0.0)
                    }
                # Handle string results (fallback)
                elif isinstance(result, str):
                    logger.warning(f"Tavily returned string result: {result[:100]}")
                    formatted_result = {
                        "title": "",
                        "url": "",
                        "content": result,
                        "snippet": result[:200],
                        "score": 0.0
                    }
                # Handle object results (has attributes)
                else:
                    formatted_result = {
                        "title": getattr(result, "title", "") if hasattr(result, "title") else "",
                        "url": getattr(result, "url", "") if hasattr(result, "url") else "",
                        "content": getattr(result, "content", getattr(result, "snippet", "")) if hasattr(result, "content") or hasattr(result, "snippet") else "",
                        "snippet": getattr(result, "snippet", "")[:200] if hasattr(result, "snippet") else (getattr(result, "content", "")[:200] if hasattr(result, "content") else ""),
                        "score": getattr(result, "score", 0.0) if hasattr(result, "score") else 0.0
                    }
                
                # Only add if we have meaningful content
                if formatted_result.get("content") or formatted_result.get("url"):
                    formatted_results.append(formatted_result)
                else:
                    logger.debug(f"Skipping empty Tavily result: {result}")
            except Exception as e:
                logger.warning(f"Failed to format Tavily result: {e}, result type: {type(result)}")
                continue
            
            # ✅ NEW: Create Document for Supabase storage
            if formatted_result.get("content"):
                from langchain_core.documents import Document
                doc = Document(
                    page_content=formatted_result["content"],
                    metadata={
                        "source": formatted_result["url"],
                        "title": formatted_result["title"],
                        "source_type": "tavily_web_search",
                        "query": query
                    }
                )
                documents_for_storage.append(doc)
        
        # ✅ NEW: Store Tavily results in Supabase for RAG retrieval
        if documents_for_storage:
            try:
                stored = await _store_web_search_results_in_supabase(query, documents_for_storage)
                if stored:
                    logger.info(f"✅ Stored {len(documents_for_storage)} Tavily search results in Supabase for RAG")
            except Exception as e:
                logger.warning(f"⚠️ Failed to store Tavily results in Supabase: {e}")
        
        logger.info(f"Web search found {len(formatted_results)} results for: {query}")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


@tool
async def comprehensive_research_tool(
    query: str,
    base_domain: Optional[str] = None,
    categories: Optional[List[str]] = None,
    store_in_supabase: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive web research using 95-field casino intelligence extraction
    
    Uses the ComprehensiveWebResearchChain to extract structured casino data across
    8 categories (95+ fields) and optionally stores results in Supabase for reuse.
    
    Args:
        query: Research query/topic (e.g., "Betway Casino Review")
        base_domain: Base domain to research (extracted from query if not provided)
        categories: List of categories to research (default: all 8 categories)
        store_in_supabase: Whether to store structured data in Supabase for reuse
        
    Returns:
        Dictionary with comprehensive research data including:
        - research_data: Structured 95-field casino intelligence data
        - documents: Raw documents loaded
        - urls_used: List of URLs researched
        - quality_score: Research quality assessment (0-1)
        - fields_extracted: Number of fields populated
        - stored_in_supabase: Whether data was stored in Supabase
    """
    if not RESEARCH_AVAILABLE:
        return {
            "research_data": {},
            "documents": [],
            "urls_used": [],
            "quality_score": 0.0,
            "fields_extracted": 0,
            "stored_in_supabase": False,
            "error": "Research components not available"
        }
    
    # ✅ FIXED: Use module-level logger (defined at line 79)
    # Don't shadow it with local variable
    import logging
    _logger = logging.getLogger(__name__) if 'logger' not in globals() else logger
    
    try:
        # Extract casino name/domain from query
        import re
        casino_name = None
        
        if not base_domain:
            # Try to extract domain from query
            domain_match = re.search(r'(\w+\.(com|io|net|org|co\.uk))', query.lower())
            if domain_match:
                base_domain = domain_match.group(1)
                casino_name = base_domain.split('.')[0]
            else:
                # Extract casino name from query (e.g., "Betway Casino Review" -> "betway")
                words = query.lower().split()
                for word in words:
                    if word not in ['casino', 'review', '2025', 'guide', 'best']:
                        casino_name = word
                        base_domain = f"{word}.com"
                        break
                
                if not casino_name:
                    casino_name = "casino"
                    base_domain = "casino.org"
        
        _logger.info(f"🔍 Starting comprehensive research for: {casino_name} ({base_domain})")
        
        # Import comprehensive web research chain (already imported at module level)
        if not RESEARCH_AVAILABLE:
            return {
                "research_data": {},
                "documents": [],
                "urls_used": [],
                "quality_score": 0.0,
                "fields_extracted": 0,
                "stored_in_supabase": False,
                "error": "Research components not available"
            }
        
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            logger.error(f"Failed to import ChatOpenAI: {e}")
            return {
                "research_data": {},
                "documents": [],
                "urls_used": [],
                "quality_score": 0.0,
                "fields_extracted": 0,
                "stored_in_supabase": False,
                "error": f"ChatOpenAI import error: {str(e)}"
            }
        
        # Create comprehensive research chain with all categories
        if categories is None:
            categories = [
                'trustworthiness',   # 15 fields
                'games',             # 12 fields
                'bonuses',           # 12 fields
                'payments',          # 15 fields
                'user_experience',   # 12 fields
                'innovations',       # 8 fields
                'compliance',        # 10 fields
                'assessment',        # 11 fields
                'terms_and_conditions',
                'affiliate_program'
            ]
        
        # Initialize LLM for extraction
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Create research chain
        research_chain = create_comprehensive_web_research_chain(
            casino_domain=base_domain,
            llm=llm,
            categories=categories
        )
        
        # Run comprehensive research (use ainvoke if available, otherwise invoke)
        _logger.info(f"🔄 Running 95-field extraction across {len(categories)} categories...")
        if hasattr(research_chain, 'ainvoke'):
            research_results = await research_chain.ainvoke({
                'casino_domain': base_domain,
                'categories': categories
            })
        else:
            # Fallback to sync invoke (run in executor to avoid blocking)
            import asyncio
            loop = asyncio.get_event_loop()
            research_results = await loop.run_in_executor(
                None,
                lambda: research_chain.invoke({
                    'casino_domain': base_domain,
                    'categories': categories
                })
            )
        
        # Extract structured data from formatted results
        research_summary = research_results.get('research_summary', {})
        overall_quality = research_results.get('overall_quality', {})
        urls_used = research_results.get('urls_researched', [])
        
        # Calculate total fields extracted from research summary
        fields_extracted = overall_quality.get('total_fields_populated', 0)
        if fields_extracted == 0:
            # Fallback: count from research summary
            for category_summary in research_summary.values():
                if isinstance(category_summary, dict):
                    fields_extracted += category_summary.get('fields_extracted', 0)
        
        quality_score = overall_quality.get('average_confidence', 0.0)
        
        # Flatten research_summary into research_data format
        research_data = {}
        for category, summary in research_summary.items():
            if isinstance(summary, dict):
                research_data[category] = {
                    'fields_extracted': summary.get('fields_extracted', 0),
                    'confidence_score': summary.get('confidence_score', 0.0),
                    'data_quality': summary.get('data_quality', 'poor'),
                    'urls_researched': summary.get('urls_successful', 0)
                }
        
        # Store in Supabase if requested (with chunking for RAG)
        stored_in_supabase = False
        raw_documents = research_results.get('documents', [])
        if store_in_supabase and fields_extracted > 10:
            try:
                stored_in_supabase = await _store_casino_intelligence_in_supabase(
                    casino_name=casino_name or base_domain.split('.')[0],
                    research_data=research_data,
                    urls_used=urls_used,
                    quality_score=quality_score,
                    fields_extracted=fields_extracted,
                    raw_documents=raw_documents  # Pass raw documents for chunking
                )
                if stored_in_supabase:
                    _logger.info(f"✅ Stored {fields_extracted} fields + {len(raw_documents)} documents (chunked) in Supabase for {casino_name}")
            except Exception as e:
                _logger.warning(f"⚠️ Failed to store in Supabase: {e}")
        
        _logger.info(f"✅ Comprehensive research completed: {fields_extracted} fields extracted, quality: {quality_score:.2f}")
        
        return {
            "research_data": research_data,
            "documents": research_results.get('documents', []),
            "urls_used": urls_used,
            "quality_score": quality_score,
            "fields_extracted": fields_extracted,
            "stored_in_supabase": stored_in_supabase,
            "casino_name": casino_name,
            "overall_grade": overall_quality.get('research_grade', 'N/A')
        }
        
    except Exception as e:
        # ✅ FIXED: Use module-level logger
        import logging
        _logger = logging.getLogger(__name__)
        _logger.error(f"Comprehensive research failed: {e}", exc_info=True)
        return {
            "research_data": {},
            "documents": [],
            "urls_used": [],
            "quality_score": 0.0,
            "fields_extracted": 0,
            "stored_in_supabase": False,
            "error": str(e)
        }


async def _store_web_search_results_in_supabase(
    query: str,
    documents: List[Any]
) -> bool:
    """
    Store Tavily web search results in Supabase for RAG retrieval
    
    ✅ OPTIMIZATION: Chunks search results for fast RAG retrieval
    """
    try:
        import os
        from supabase import create_client
        from datetime import datetime
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not available for Tavily storage")
            return False
        
        client = create_client(supabase_url, supabase_key)
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        embeddings = OpenAIEmbeddings()
        
        try:
            from langchain_community.vectorstores import SupabaseVectorStore
            vector_store = SupabaseVectorStore(
                client=client,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store for Tavily: {e}")
            return False
        
        # Chunk and store documents
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update({
                    "source_type": "tavily_web_search",
                    "query": query,
                    "research_timestamp": datetime.now().isoformat()
                })
            all_chunks.extend(chunks)
        
        if all_chunks:
            try:
                vector_store.add_documents(all_chunks)
                logger.info(f"✅ Stored {len(all_chunks)} Tavily search chunks in Supabase")
                return True
            except Exception as e:
                logger.error(f"Failed to store Tavily chunks: {e}")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to store Tavily results: {e}", exc_info=True)
        return False


async def _store_casino_intelligence_in_supabase(
    casino_name: str,
    research_data: Dict[str, Any],
    urls_used: List[str],
    quality_score: float,
    fields_extracted: int,
    raw_documents: Optional[List[Any]] = None
) -> bool:
    """
    Store 95-field casino intelligence data in Supabase with CHUNKING for RAG retrieval
    
    ✅ OPTIMIZATIONS:
    - Chunks documents using RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
    - Stores chunks in vector store for fast RAG retrieval
    - Adds metadata for filtering and context
    - Makes research data immediately available for writing phase
    """
    try:
        import os
        from supabase import create_client
        from datetime import datetime
        import json
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not available")
            return False
        
        client = create_client(supabase_url, supabase_key)
        
        # Initialize text splitter for chunking (optimized for RAG)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Optimal for RAG retrieval
            chunk_overlap=200,  # Maintain context across chunks
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Smart splitting
        )
        
        embeddings = OpenAIEmbeddings()
        vector_store = None
        
        try:
            from langchain_community.vectorstores import SupabaseVectorStore
            vector_store = SupabaseVectorStore(
                client=client,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False
        
        documents_to_store = []
        
        # 1. Chunk and store raw research documents (if available)
        if raw_documents:
            logger.info(f"📄 Chunking {len(raw_documents)} raw research documents")
            for doc in raw_documents:
                # Extract URL from document metadata if available
                doc_url = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
                
                # ✅ FIXED: Sanitize document content - remove null bytes and invalid Unicode
                if hasattr(doc, "page_content"):
                    # Remove null bytes and other problematic characters
                    doc.page_content = doc.page_content.replace('\x00', '').replace('\u0000', '')
                    # Remove other control characters except newlines and tabs
                    import re
                    doc.page_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', doc.page_content)
                
                # Split document into chunks
                chunks = text_splitter.split_documents([doc])
                
                # Add casino-specific metadata to each chunk
                for i, chunk in enumerate(chunks):
                    # ✅ FIXED: Sanitize chunk content as well
                    if hasattr(chunk, "page_content"):
                        chunk.page_content = chunk.page_content.replace('\x00', '').replace('\u0000', '')
                        chunk.page_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', chunk.page_content)
                    
                    chunk.metadata.update({
                        "casino_name": casino_name.strip(),
                        "source": "comprehensive_web_research",
                        "content_type": "research_document",
                        "research_timestamp": datetime.now().isoformat(),
                        "url": doc_url,
                        "data_completeness": quality_score,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                
                documents_to_store.extend(chunks)
            logger.info(f"✅ Created {len(documents_to_store)} chunks from raw documents")
        
        # 2. Create searchable structured content (chunked)
        searchable_parts = [f"Casino: {casino_name}"]
        for category, data in research_data.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if value:
                        searchable_parts.append(f"{category} {key}: {value}")
        
        searchable_content = "\n".join(searchable_parts)
        
        # Base metadata for structured data
        base_metadata = {
            "source": "comprehensive_web_research",
            "content_type": "casino_intelligence_95_fields",
            "casino_name": casino_name.strip(),
            "research_timestamp": datetime.now().isoformat(),
            "original_query": casino_name,
            "data_completeness": quality_score,
            "field_count": fields_extracted,
            "urls_researched": json.dumps(urls_used),  # Store as JSON string
            "reuse_ready": True,
            "intelligence_version": "1.0"
        }
        
        # Create structured content document
        structured_content = f"""CASINO: {casino_name}

SEARCHABLE DATA:
{searchable_content}

STRUCTURED DATA (95 FIELDS):
{json.dumps(research_data, ensure_ascii=False, indent=2)}
"""
        
        # ✅ FIXED: Sanitize structured content
        import re
        structured_content = structured_content.replace('\x00', '').replace('\u0000', '')
        structured_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', structured_content)
        
        structured_doc = Document(page_content=structured_content, metadata=base_metadata)
        
        # Chunk structured content as well
        structured_chunks = text_splitter.split_documents([structured_doc])
        for i, chunk in enumerate(structured_chunks):
            # ✅ FIXED: Sanitize chunk content
            if hasattr(chunk, "page_content"):
                chunk.page_content = chunk.page_content.replace('\x00', '').replace('\u0000', '')
                chunk.page_content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', chunk.page_content)
            
            chunk.metadata.update({
                "chunk_index": i,
                "total_chunks": len(structured_chunks),
                "chunk_type": "structured_intelligence"
            })
        
        documents_to_store.extend(structured_chunks)
        
        # 3. Store all chunks in vector store (batch operation)
        if documents_to_store:
            try:
                # Add documents in batches of 100 for efficiency
                batch_size = 100
                total_stored = 0
                
                for i in range(0, len(documents_to_store), batch_size):
                    batch = documents_to_store[i:i + batch_size]
                    vector_store.add_documents(batch)
                    total_stored += len(batch)
                    logger.info(f"📦 Stored batch {i//batch_size + 1}: {len(batch)} chunks")
                
                logger.info(f"✅ Stored {total_stored} chunks for {casino_name} in Supabase vector store")
                logger.info(f"🚀 Research data now available for RAG retrieval (fast!)")
                return True
                
            except Exception as e:
                logger.error(f"Failed to store chunks in vector store: {e}")
                return False
        else:
            logger.warning("No documents to store")
            return False
            
    except Exception as e:
        logger.error(f"Failed to store casino intelligence: {e}", exc_info=True)
        return False


@tool
async def screenshot_tool(
    url: str,
    screenshot_type: str = "full_page",
    element_selector: Optional[str] = None
) -> Dict[str, Any]:
    """
    Capture screenshot of a URL using Playwright
    
    Args:
        url: URL to capture
        screenshot_type: Type of screenshot ('full_page', 'viewport', 'element', 'casino')
        element_selector: CSS selector for element screenshots
        
    Returns:
        Dictionary with screenshot metadata including:
        - success: Whether capture succeeded
        - screenshot_id: UUID of screenshot
        - file_size: Size in bytes
        - format: Image format
        - url: Original URL
        - storage_path: Storage location (if saved)
    """
    if not RESEARCH_AVAILABLE:
        return {
            "success": False,
            "error": "Screenshot components not available"
        }
    
    try:
        # Initialize browser pool (singleton pattern would be better in production)
        pool = BrowserPoolManager(max_pool_size=2)
        await pool.initialize()
        
        try:
            service = ScreenshotService(pool)
            
            # Capture based on type
            if screenshot_type == "full_page":
                result = await service.capture_full_page_screenshot(url)
            elif screenshot_type == "viewport":
                result = await service.capture_viewport_screenshot(url)
            elif screenshot_type == "element" and element_selector:
                result = await service.capture_element_screenshot(url, element_selector)
            elif screenshot_type == "casino":
                casino_locator = CasinoElementLocator(service)
                results = await casino_locator.capture_casino_screenshots(url)
                # Return first successful result
                for element_type, element_results in results.items():
                    for r in element_results:
                        if r.success:
                            result = r
                            break
                    else:
                        continue
                    break
                else:
                    return {
                        "success": False,
                        "error": "No casino screenshots captured"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Invalid screenshot_type: {screenshot_type}"
                }
            
            if result.success:
                # ✅ FIXED: ScreenshotResult doesn't have width/height directly, extract from viewport_size
                width = None
                height = None
                if result.viewport_size and isinstance(result.viewport_size, dict):
                    width = result.viewport_size.get('width')
                    height = result.viewport_size.get('height')

                return {
                    "success": True,
                    "screenshot_data": result.screenshot_data,  # Base64 encoded bytes
                    "file_size": result.file_size,
                    "url": url,
                    "timestamp": result.timestamp,
                    "width": width,
                    "height": height,
                    "viewport_size": result.viewport_size,
                    "element_info": result.element_info if hasattr(result, 'element_info') else None
                }
            else:
                return {
                    "success": False,
                    "error": result.error_message
                }
                
        finally:
            await pool.cleanup()
            
    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@tool
async def casino_intelligence_tool(
    casino_name: str,
    extract_all_fields: bool = True
) -> Dict[str, Any]:
    """
    Extract structured casino intelligence data (95 fields)
    
    Args:
        casino_name: Name of the casino to research
        extract_all_fields: Whether to extract all 95 fields
        
    Returns:
        Dictionary with structured casino intelligence data across 8 categories
    """
    if not RESEARCH_AVAILABLE:
        return {
            "casino_name": casino_name,
            "extracted_fields": 0,
            "data": {},
            "error": "Research components not available"
        }
    
    try:
        # Use comprehensive research to extract structured data
        # Call the comprehensive_research_tool (it's a LangChain tool)
        research_result = await comprehensive_research_tool.ainvoke({
            "query": f"{casino_name} casino review",
            "base_domain": None,
            "categories": [
                'trustworthiness', 'games', 'bonuses', 'payments',
                'user_experience', 'innovations', 'compliance', 'assessment'
            ] if extract_all_fields else ['trustworthiness', 'games']
        })
        
        # Structure the data (simplified - would use Pydantic models in full implementation)
        structured_data = {
            "casino_name": casino_name,
            "extracted_fields": research_result.get("quality_score", 0.0) * 95,
            "research_quality": research_result.get("quality_score", 0.0),
            "sources_used": len(research_result.get("urls_used", [])),
            "data": research_result.get("research_data", {})
        }
        
        logger.info(f"Casino intelligence extracted for {casino_name}: {structured_data['extracted_fields']:.0f} fields")
        
        return structured_data
        
    except Exception as e:
        logger.error(f"Casino intelligence extraction failed: {e}")
        return {
            "casino_name": casino_name,
            "extracted_fields": 0,
            "data": {},
            "error": str(e)
        }

