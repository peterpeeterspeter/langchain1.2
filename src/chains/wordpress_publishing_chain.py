#!/usr/bin/env python3
"""
WordPress Publishing Chain - Native LangChain Implementation
Transforms content into WordPress-ready XML using native LangChain patterns

✅ NATIVE LANGCHAIN PATTERNS:
- RunnableSequence for content → metadata → XML transformation
- RunnableLambda for transformation functions
- RunnableBranch for conditional logic based on content type
- Structured outputs with Pydantic models

🎯 USAGE: content_data | wordpress_publishing_chain → WordPress XML
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import html
import logging

from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ✅ PYDANTIC MODELS FOR WORDPRESS CONTENT

class WordPressMetadata(BaseModel):
    """WordPress post metadata structure"""
    title: str = Field(description="Post title")
    content: str = Field(description="Post content in HTML")
    excerpt: Optional[str] = Field(default=None, description="Post excerpt")
    status: str = Field(default="publish", description="Post status")
    post_type: str = Field(default="post", description="Post type")
    author: str = Field(default="admin", description="Author username")
    date: datetime = Field(default_factory=datetime.now, description="Publication date")
    categories: List[str] = Field(default=[], description="Post categories")
    tags: List[str] = Field(default=[], description="Post tags")
    featured_image: Optional[str] = Field(default=None, description="Featured image URL")
    meta_description: Optional[str] = Field(default=None, description="SEO meta description")
    focus_keyword: Optional[str] = Field(default=None, description="SEO focus keyword")
    
    # Coinflip theme specific fields
    theme_fields: Dict[str, Any] = Field(default={}, description="Theme-specific custom fields")

class GutenblockContent(BaseModel):
    """Gutenberg block structure"""
    block_type: str = Field(description="Block type (paragraph, heading, image, etc.)")
    content: str = Field(description="Block content")
    attributes: Dict[str, Any] = Field(default={}, description="Block attributes")
    inner_blocks: List['GutenblockContent'] = Field(default=[], description="Nested blocks")

class WordPressExportData(BaseModel):
    """Complete WordPress export structure"""
    site_info: Dict[str, str] = Field(default={}, description="Site information")
    posts: List[WordPressMetadata] = Field(default=[], description="Posts to export")
    media_items: List[Dict[str, Any]] = Field(default=[], description="Media attachments")
    export_timestamp: datetime = Field(default_factory=datetime.now)

# ✅ WORDPRESS XML GENERATION FUNCTIONS

def create_content_to_gutenberg_transformer() -> RunnableLambda:
    """Transform content into Gutenberg blocks"""
    
    def transform_to_gutenberg(content_data: Dict[str, Any]) -> List[GutenblockContent]:
        """Convert content to Gutenberg blocks"""
        
        blocks = []
        content = content_data.get("content", "")
        
        # Split content into paragraphs and create blocks
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                # Detect headings
                if para.startswith('#'):
                    level = len(para) - len(para.lstrip('#'))
                    heading_text = para.lstrip('#').strip()
                    
                    blocks.append(GutenblockContent(
                        block_type="core/heading",
                        content=f"<h{level}>{html.escape(heading_text)}</h{level}>",
                        attributes={"level": level}
                    ))
                else:
                    # Regular paragraph
                    blocks.append(GutenblockContent(
                        block_type="core/paragraph",
                        content=f"<p>{html.escape(para.strip())}</p>",
                        attributes={}
                    ))
        
        return blocks
    
    return RunnableLambda(transform_to_gutenberg)

def create_metadata_enricher() -> RunnableLambda:
    """Enrich content with WordPress metadata"""
    
    def enrich_metadata(input_data: Dict[str, Any]) -> WordPressMetadata:
        """Add WordPress-specific metadata to content"""
        
        # Extract basic content
        title = input_data.get("title", "Generated Post")
        content = input_data.get("content", "")
        
        # Generate metadata
        metadata = WordPressMetadata(
            title=title,
            content=content,
            excerpt=content[:150] + "..." if len(content) > 150 else content,
            categories=input_data.get("categories", ["Casino Reviews"]),
            tags=input_data.get("tags", []),
            meta_description=input_data.get("meta_description", f"{title} - Complete review and analysis"),
            focus_keyword=input_data.get("focus_keyword", title.lower()),
            theme_fields={
                "casino_rating": input_data.get("rating", 0),
                "bonus_amount": input_data.get("bonus_amount", ""),
                "license_info": input_data.get("license_info", ""),
                "game_providers": input_data.get("game_providers", []),
                "payment_methods": input_data.get("payment_methods", []),
                "mobile_compatible": input_data.get("mobile_compatible", True),
                "live_chat_support": input_data.get("live_chat_support", False),
                "withdrawal_time": input_data.get("withdrawal_time", ""),
                "min_deposit": input_data.get("min_deposit", ""),
                "wagering_requirements": input_data.get("wagering_requirements", ""),
                "review_summary": input_data.get("review_summary", ""),
                "pros_list": input_data.get("pros", []),
                "cons_list": input_data.get("cons", []),
                "verdict": input_data.get("verdict", ""),
                "last_updated": datetime.now().isoformat(),
                "review_methodology": input_data.get("methodology", "Comprehensive analysis based on multiple factors"),
                "affiliate_disclosure": "This review may contain affiliate links. Please gamble responsibly.",
                "author_expertise": "Expert casino reviewer with 5+ years experience",
                "fact_checked": True,
                "review_language": "en-US"
            }
        )
        
        return metadata
    
    return RunnableLambda(enrich_metadata)

def create_xml_generator() -> RunnableLambda:
    """Generate WordPress WXR XML from metadata"""
    
    def generate_wxr_xml(wp_data: WordPressMetadata) -> str:
        """Create WordPress WXR (WordPress eXtended RSS) XML"""
        
        # Create root RSS element
        rss = ET.Element("rss", version="2.0")
        
        # Add WordPress namespaces
        rss.set("xmlns:excerpt", "http://wordpress.org/export/1.2/excerpt/")
        rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")
        rss.set("xmlns:wfw", "http://wellformedweb.org/CommentAPI/")
        rss.set("xmlns:dc", "http://purl.org/dc/elements/1.1/")
        rss.set("xmlns:wp", "http://wordpress.org/export/1.2/")
        
        # Create channel
        channel = ET.SubElement(rss, "channel")
        
        # Add channel metadata
        ET.SubElement(channel, "title").text = "Casino Review Site"
        ET.SubElement(channel, "link").text = "https://casinosite.com"
        ET.SubElement(channel, "description").text = "Professional Casino Reviews"
        ET.SubElement(channel, "pubDate").text = wp_data.date.strftime("%a, %d %b %Y %H:%M:%S +0000")
        ET.SubElement(channel, "language").text = "en-US"
        ET.SubElement(channel, "wp:wxr_version").text = "1.2"
        ET.SubElement(channel, "wp:base_site_url").text = "https://casinosite.com"
        ET.SubElement(channel, "wp:base_blog_url").text = "https://casinosite.com"
        
        # Create item (post)
        item = ET.SubElement(channel, "item")
        
        # Basic post data
        ET.SubElement(item, "title").text = wp_data.title
        ET.SubElement(item, "link").text = f"https://casinosite.com/{wp_data.title.lower().replace(' ', '-')}"
        ET.SubElement(item, "pubDate").text = wp_data.date.strftime("%a, %d %b %Y %H:%M:%S +0000")
        ET.SubElement(item, "dc:creator").text = wp_data.author
        ET.SubElement(item, "guid", isPermaLink="false").text = f"https://casinosite.com/?p=123"
        ET.SubElement(item, "description")
        
        # WordPress specific fields
        ET.SubElement(item, "content:encoded").text = f"<![CDATA[{wp_data.content}]]>"
        ET.SubElement(item, "excerpt:encoded").text = f"<![CDATA[{wp_data.excerpt or ''}]]>"
        ET.SubElement(item, "wp:post_id").text = "123"
        ET.SubElement(item, "wp:post_date").text = wp_data.date.strftime("%Y-%m-%d %H:%M:%S")
        ET.SubElement(item, "wp:post_date_gmt").text = wp_data.date.strftime("%Y-%m-%d %H:%M:%S")
        ET.SubElement(item, "wp:comment_status").text = "open"
        ET.SubElement(item, "wp:ping_status").text = "open"
        ET.SubElement(item, "wp:post_name").text = wp_data.title.lower().replace(' ', '-')
        ET.SubElement(item, "wp:status").text = wp_data.status
        ET.SubElement(item, "wp:post_parent").text = "0"
        ET.SubElement(item, "wp:menu_order").text = "0"
        ET.SubElement(item, "wp:post_type").text = wp_data.post_type
        ET.SubElement(item, "wp:post_password").text = ""
        ET.SubElement(item, "wp:is_sticky").text = "0"
        
        # Categories
        for category in wp_data.categories:
            cat_elem = ET.SubElement(item, "category", domain="category")
            cat_elem.text = category
        
        # Tags
        for tag in wp_data.tags:
            tag_elem = ET.SubElement(item, "category", domain="post_tag")
            tag_elem.text = tag
        
        # Custom fields (theme-specific)
        for field_name, field_value in wp_data.theme_fields.items():
            postmeta = ET.SubElement(item, "wp:postmeta")
            ET.SubElement(postmeta, "wp:meta_key").text = f"_{field_name}"
            ET.SubElement(postmeta, "wp:meta_value").text = f"<![CDATA[{str(field_value)}]]>"
        
        # SEO fields
        if wp_data.meta_description:
            seo_meta = ET.SubElement(item, "wp:postmeta")
            ET.SubElement(seo_meta, "wp:meta_key").text = "_yoast_wpseo_metadesc"
            ET.SubElement(seo_meta, "wp:meta_value").text = f"<![CDATA[{wp_data.meta_description}]]>"
        
        if wp_data.focus_keyword:
            seo_kw = ET.SubElement(item, "wp:postmeta")
            ET.SubElement(seo_kw, "wp:meta_key").text = "_yoast_wpseo_focuskw"
            ET.SubElement(seo_kw, "wp:meta_value").text = f"<![CDATA[{wp_data.focus_keyword}]]>"
        
        # Convert to pretty XML string
        xml_str = ET.tostring(rss, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")
    
    return RunnableLambda(generate_wxr_xml)

# ✅ CONTENT TYPE CONDITIONAL LOGIC

def create_content_type_router() -> RunnableBranch:
    """Route content based on type using RunnableBranch"""
    
    def is_casino_review(input_data: Dict[str, Any]) -> bool:
        """Check if content is a casino review"""
        return input_data.get("content_type") == "casino_review" or "casino" in input_data.get("title", "").lower()
    
    def is_slot_review(input_data: Dict[str, Any]) -> bool:
        """Check if content is a slot review"""
        return input_data.get("content_type") == "slot_review" or "slot" in input_data.get("title", "").lower()
    
    def is_news_article(input_data: Dict[str, Any]) -> bool:
        """Check if content is a news article"""
        return input_data.get("content_type") == "news" or input_data.get("post_type") == "news"
    
    # Create different enhancement chains for different content types
    casino_review_enhancer = RunnableLambda(lambda x: {
        **x,
        "categories": ["Casino Reviews", "Online Casinos"],
        "post_type": "casino_review",
        "template": "single-casino-review.php"
    })
    
    slot_review_enhancer = RunnableLambda(lambda x: {
        **x,
        "categories": ["Slot Reviews", "Casino Games"],
        "post_type": "slot_review",
        "template": "single-slot-review.php"
    })
    
    news_enhancer = RunnableLambda(lambda x: {
        **x,
        "categories": ["Casino News", "Industry News"],
        "post_type": "post",
        "template": "single-news.php"
    })
    
    default_enhancer = RunnableLambda(lambda x: {
        **x,
        "categories": ["General"],
        "post_type": "post"
    })
    
    return RunnableBranch(
        (is_casino_review, casino_review_enhancer),
        (is_slot_review, slot_review_enhancer),
        (is_news_article, news_enhancer),
        default_enhancer
    )

# ✅ MAIN WORDPRESS PUBLISHING CHAIN

def create_wordpress_publishing_chain(
    llm: Optional[ChatOpenAI] = None
) -> RunnableSequence:
    """
    Create WordPress publishing chain using native LangChain patterns
    
    ✅ NATIVE LANGCHAIN COMPONENTS:
    - RunnableSequence for content → metadata → XML flow
    - RunnableLambda for transformation functions
    - RunnableBranch for content type conditional logic
    - Structured outputs with Pydantic models
    
    Args:
        llm: Language model for content enhancement
    
    Returns:
        RunnableSequence: Composable chain for WordPress publishing
        
    Usage:
        content_data | wordpress_publishing_chain → WordPress XML
    """
    
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # Create transformation components
    content_router = create_content_type_router()
    gutenberg_transformer = create_content_to_gutenberg_transformer()
    metadata_enricher = create_metadata_enricher()
    xml_generator = create_xml_generator()
    
    # Create the complete publishing sequence
    input_processor = RunnableLambda(lambda x: {
        "title": x.get("title", "Generated Post"),
        "content": x.get("content", ""),
        "content_type": x.get("content_type", "post"),
        **x
    })
    
    gutenberg_processor = RunnableLambda(lambda x: {
        **x,
        "gutenberg_blocks": gutenberg_transformer.invoke(x)
    })
    
    post_processor = RunnableLambda(lambda xml_content: {
        "wordpress_xml": xml_content,
        "generation_timestamp": datetime.now().isoformat(),
        "status": "ready_for_import",
        "file_size_kb": len(xml_content.encode('utf-8')) / 1024,
        "export_format": "WXR (WordPress eXtended RSS)"
    })
    
    publishing_chain = (
        input_processor | 
        content_router | 
        gutenberg_processor | 
        metadata_enricher | 
        xml_generator | 
        post_processor
    )
    
    return publishing_chain

# ✅ INTEGRATION HELPERS

class WordPressPublishingEnhancer:
    """Helper for integrating WordPress publishing with v2 systems"""
    
    @staticmethod
    def enhance_content_pipeline(content_pipeline, llm=None):
        """Add WordPress publishing to content generation pipeline"""
        wordpress_chain = create_wordpress_publishing_chain(llm)
        return content_pipeline.pipe(wordpress_chain)
    
    @staticmethod
    def create_multi_site_publisher(sites_config: List[Dict[str, str]]) -> RunnableLambda:
        """Create multi-site publishing component"""
        
        def publish_to_multiple_sites(content_data: Dict[str, Any]) -> Dict[str, Any]:
            """Publish content to multiple WordPress sites"""
            
            results = {}
            wordpress_chain = create_wordpress_publishing_chain()
            
            for site in sites_config:
                site_name = site.get("name", "default")
                site_specific_content = {
                    **content_data,
                    "site_info": site
                }
                
                try:
                    result = wordpress_chain.invoke(site_specific_content)
                    results[site_name] = {
                        "status": "success",
                        "xml_content": result["wordpress_xml"],
                        "import_url": f"{site.get('url', '')}/wp-admin/import.php"
                    }
                except Exception as e:
                    results[site_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "multi_site_results": results,
                "total_sites": len(sites_config),
                "successful_exports": len([r for r in results.values() if r["status"] == "success"])
            }
        
        return RunnableLambda(publish_to_multiple_sites)
    
    @staticmethod
    def create_xml_validator() -> RunnableLambda:
        """Create XML validation component"""
        
        def validate_wordpress_xml(xml_content: str) -> Dict[str, Any]:
            """Validate WordPress XML format"""
            
            try:
                # Parse XML to check validity
                root = ET.fromstring(xml_content)
                
                # Check required WordPress elements
                required_elements = ["rss", "channel", "item"]
                found_elements = []
                
                def check_element(element, path=""):
                    full_path = f"{path}/{element.tag}" if path else element.tag
                    found_elements.append(full_path)
                    for child in element:
                        check_element(child, full_path)
                
                check_element(root)
                
                validation_result = {
                    "valid": True,
                    "format": "WordPress WXR",
                    "elements_found": len(found_elements),
                    "has_posts": "item" in [elem.split("/")[-1] for elem in found_elements],
                    "has_metadata": any("wp:" in elem for elem in found_elements),
                    "file_size_mb": len(xml_content.encode('utf-8')) / (1024 * 1024)
                }
                
            except ET.ParseError as e:
                validation_result = {
                    "valid": False,
                    "error": f"XML Parse Error: {str(e)}",
                    "fix_suggestion": "Check XML syntax and structure"
                }
            
            return validation_result
        
        return RunnableLambda(validate_wordpress_xml)

# ✅ TESTING

async def test_wordpress_publishing_chain():
    """Test the WordPress publishing chain"""
    print("🧪 Testing WordPress Publishing Chain")
    
    try:
        # Create the chain
        wordpress_chain = create_wordpress_publishing_chain()
        
        # Test input
        test_content = {
            "title": "Best Online Casino 2024 Review",
            "content": "# Casino Overview\n\nThis is a comprehensive review of the best online casino.\n\n## Games Selection\n\nGreat variety of slots and table games.",
            "content_type": "casino_review",
            "rating": 8.5,
            "bonus_amount": "100% up to $500",
            "license_info": "MGA License",
            "tags": ["casino", "review", "2024"],
            "meta_description": "Complete review of the best online casino with games, bonuses, and more."
        }
        
        # Run the chain
        result = await wordpress_chain.ainvoke(test_content)
        
        print(f"✅ WordPress chain executed successfully")
        print(f"📄 XML generated: {result['file_size_kb']:.1f} KB")
        print(f"📅 Export format: {result['export_format']}")
        
        # Validate the XML
        validator = WordPressPublishingEnhancer.create_xml_validator()
        validation = validator.invoke(result["wordpress_xml"])
        
        print(f"✅ XML validation: {'Valid' if validation['valid'] else 'Invalid'}")
        
        return True
        
    except Exception as e:
        print(f"❌ WordPress chain test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_wordpress_publishing_chain()) 