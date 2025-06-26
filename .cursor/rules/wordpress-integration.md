---
description: WordPress publishing using LangChain best practices
globs: "**/*wordpress*.py"
alwaysApply: true
---

# WordPress Integration Rules

## Use LangChain Output Parsers for WordPress
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class WordPressPost(BaseModel):
    title: str = Field(description="SEO-optimized post title")
    content: str = Field(description="Full HTML content with proper formatting")
    excerpt: str = Field(description="Brief post excerpt for preview")
    tags: List[str] = Field(description="Relevant tags for categorization")
    categories: List[str] = Field(description="WordPress categories")
    meta_description: str = Field(description="SEO meta description")
    featured_image_url: Optional[str] = Field(description="URL to featured image")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
parser = PydanticOutputParser(pydantic_object=WordPressPost)
```

## LangChain Chain Integration Patterns

### 1. Use RunnableLambda for WordPress Steps
```python
from langchain_core.runnables import RunnableLambda

async def wordpress_publishing_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Native LangChain step for WordPress publishing"""
    # Extract content from chain inputs
    content = inputs.get("content", "")
    
    # Use parser to structure WordPress data
    wp_data = parser.parse(content)
    
    # Publish using WordPress service
    result = await wordpress_service.publish(wp_data)
    
    return {**inputs, "wordpress_result": result}

# Integrate into LCEL chain
wordpress_step = RunnableLambda(wordpress_publishing_step)
```

### 2. Output Parser Integration
```python
from langchain_core.prompts import PromptTemplate

# Create WordPress-specific prompt with parser instructions
wordpress_prompt = PromptTemplate.from_template("""
Create a WordPress post based on the following content:

Content: {content}
Query: {query}

{format_instructions}

Response:""")

# Always include format instructions
prompt_with_parser = wordpress_prompt.partial(
    format_instructions=parser.get_format_instructions()
)
```

### 3. Chain Composition with WordPress
```python
from langchain_core.runnables import RunnablePassthrough

# Native LangChain chain with WordPress integration
wordpress_chain = (
    {"content": RunnablePassthrough(), "query": RunnablePassthrough()} 
    | prompt_with_parser
    | llm
    | parser  # Parse to structured WordPress data
    | RunnableLambda(wordpress_publishing_step)  # Publish to WordPress
)
```

## Error Handling Best Practices

### 1. Graceful Parser Failures
```python
from langchain_core.output_parsers import OutputFixingParser

# Use OutputFixingParser for robust parsing
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=llm
)

# Fallback parser for critical failures
def safe_wordpress_parse(content: str) -> WordPressPost:
    try:
        return fixing_parser.parse(content)
    except Exception as e:
        logging.warning(f"Parser failed, using fallback: {e}")
        return WordPressPost(
            title="Generated Content",
            content=content,
            excerpt=content[:150] + "...",
            tags=["automated"],
            categories=["General"],
            meta_description=content[:160]
        )
```

### 2. Chain Resilience
```python
from langchain_core.runnables import RunnableBranch

# Conditional WordPress publishing based on content validation
wordpress_conditional = RunnableBranch(
    (lambda x: validate_wordpress_content(x), wordpress_publishing_step),
    (lambda x: True, RunnableLambda(lambda x: {**x, "wordpress_skipped": "Validation failed"}))
)
```

## WordPress Service Integration

### 1. Native Service Pattern
```python
class WordPressService:
    """WordPress service following LangChain service patterns"""
    
    def __init__(self, config: WordPressConfig):
        self.config = config
        
    async def publish_post(self, post_data: WordPressPost) -> Dict[str, Any]:
        """Publish structured post data to WordPress"""
        try:
            # Convert Pydantic model to WordPress API format
            wp_payload = {
                "title": post_data.title,
                "content": post_data.content,
                "excerpt": post_data.excerpt,
                "tags": post_data.tags,
                "categories": self._resolve_categories(post_data.categories),
                "meta": post_data.custom_fields
            }
            
            # Use WordPress REST API
            response = await self._make_wp_request("POST", "/wp-json/wp/v2/posts", json=wp_payload)
            
            return {
                "success": True,
                "post_id": response.get("id"),
                "url": response.get("link"),
                "wordpress_data": post_data.dict()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "wordpress_data": post_data.dict()
            }
```

### 2. Chain Integration
```python
# WordPress service as RunnableLambda
def create_wordpress_runnable(wp_service: WordPressService):
    async def publish_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        wp_data = inputs.get("wordpress_data")
        if not wp_data:
            return {**inputs, "wordpress_error": "No WordPress data provided"}
            
        result = await wp_service.publish_post(wp_data)
        return {**inputs, "wordpress_result": result}
    
    return RunnableLambda(publish_step)
```

## Template System Integration

### 1. WordPress Template Selection
```python
from langchain_core.runnables import RunnableLambda

def select_wordpress_template(inputs: Dict[str, Any]) -> str:
    """Select appropriate WordPress template based on content type"""
    content_type = inputs.get("content_type", "general")
    
    templates = {
        "casino_review": """Create a comprehensive casino review post:
        
        Title: {title}
        Content: {content}
        
        {format_instructions}""",
        
        "general": """Create a WordPress post:
        
        Content: {content}
        
        {format_instructions}"""
    }
    
    template = templates.get(content_type, templates["general"])
    return PromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions()
    )

wordpress_template_selector = RunnableLambda(select_wordpress_template)
```

## Validation and Quality Control

### 1. Content Validation
```python
def validate_wordpress_content(inputs: Dict[str, Any]) -> bool:
    """Validate content before WordPress publishing"""
    content = inputs.get("content", "")
    
    # LangChain-style validation checks
    checks = [
        len(content) > 100,  # Minimum content length
        "<h2>" in content or "##" in content,  # Has headings
        len(content.split()) > 50,  # Word count check
    ]
    
    return all(checks)

# Use in conditional chain
validation_step = RunnableLambda(
    lambda x: {**x, "valid_for_wordpress": validate_wordpress_content(x)}
)
```

### 2. SEO Optimization
```python
class SEOWordPressPost(WordPressPost):
    """Extended WordPress post with SEO optimization"""
    focus_keyword: str = Field(description="Primary SEO keyword")
    readability_score: float = Field(ge=0, le=100, description="Content readability score")
    word_count: int = Field(description="Total word count")
    internal_links: List[str] = Field(description="Internal link suggestions")
    
seo_parser = PydanticOutputParser(pydantic_object=SEOWordPressPost)
```

## Anti-Patterns to Avoid

### ❌ Don't Use Custom LLM Calls
```python
# BAD: Custom LLM implementation
def custom_wordpress_generation(content):
    prompt = f"Create WordPress post: {content}"
    response = openai.chat.completions.create(...)  # Direct API call
    return response

# GOOD: Use LangChain patterns
wordpress_chain = prompt | llm | parser
```

### ❌ Don't Create Complex Custom Classes
```python
# BAD: Complex custom class
class WordPressIntegration:
    def __init__(self, 20_parameters):
        # 500 lines of custom code
        self.complex_state = {}
        self.custom_llm_handler = CustomLLMHandler()
        self.custom_parser = CustomParser()
        # ... more custom implementations

# GOOD: Simple chain composition
wordpress_chain = content_chain | wordpress_formatter | publisher
```

### ❌ Don't Bypass Output Parsers
```python
# BAD: Manual string parsing
def manual_parse_wordpress(response):
    lines = response.split('\n')
    title = lines[0].replace('Title:', '')
    # ... manual parsing logic

# GOOD: Use PydanticOutputParser
result = parser.parse(response)
```

### ❌ Don't Create Monolithic Functions
```python
# BAD: Single large function
def generate_and_publish_wordpress(query):
    # Generate content
    # Parse content  
    # Validate content
    # Publish to WordPress
    # ... 200 lines of code

# GOOD: Composable chain steps
chain = (
    content_generation_step
    | parsing_step  
    | validation_step
    | publishing_step
)
```

## ✅ Complete Good vs Bad Examples

### The LangChain Way: Simple & Composable
```python
# ✅ GOOD: Simple chain composition
wordpress_chain = content_chain | wordpress_formatter | publisher

# ✅ GOOD: Individual step functions
content_chain = prompt | llm | parser
wordpress_formatter = RunnableLambda(format_for_wordpress)
publisher = RunnableLambda(publish_to_wp)

# ✅ GOOD: Clean pipeline
result = await wordpress_chain.ainvoke({"query": "Betsson casino review"})
```

### The Anti-Pattern: Complex Custom Classes
```python
# ❌ BAD: Over-engineered custom class
class WordPressIntegration:
    def __init__(self, 
                 llm_model, 
                 wordpress_url, 
                 username, 
                 password,
                 custom_parser,
                 validation_engine,
                 seo_optimizer,
                 image_processor,
                 content_enhancer,
                 metadata_generator,
                 category_mapper,
                 tag_extractor,
                 formatting_engine,
                 publishing_queue,
                 error_handler,
                 retry_mechanism,
                 logging_system,
                 metrics_collector,
                 config_manager,
                 cache_layer):
        # 500+ lines of initialization
        self.setup_complex_state()
        self.initialize_custom_components()
        # ... endless complexity

    def generate_and_publish(self, query):
        # 200+ lines of tightly coupled code
        pass

# ❌ BAD: Usage requires understanding entire class
integration = WordPressIntegration(param1, param2, ..., param20)
result = integration.generate_and_publish(query)
```

### Key Difference: Composition vs Inheritance
```python
# ✅ GOOD: Function composition (LangChain way)
def create_wordpress_pipeline():
    return (
        {"content": content_generator, "metadata": metadata_extractor}
        | wordpress_formatter
        | content_validator  
        | publisher
    )

# ❌ BAD: Class inheritance and complex state
class WordPressPublisher(BasePublisher, ContentGenerator, SEOOptimizer):
    def __init__(self):
        super().__init__()
        self.state = ComplexStateManager()
        self.dependencies = DependencyContainer()
```

## Configuration Best Practices

### 1. Environment-Based Config
```python
from pydantic import BaseSettings

class WordPressConfig(BaseSettings):
    """WordPress configuration following LangChain patterns"""
    wordpress_url: str
    wordpress_username: str
    wordpress_password: str
    default_categories: List[str] = ["General"]
    auto_publish: bool = False
    
    class Config:
        env_prefix = "WORDPRESS_"
        case_sensitive = False
```

### 2. Chain Configuration
```python
def create_wordpress_chain(config: WordPressConfig) -> Runnable:
    """Factory function for WordPress chains"""
    wp_service = WordPressService(config)
    
    return (
        content_prep_step
        | wordpress_template_selector
        | llm
        | parser
        | create_wordpress_runnable(wp_service)
    )
```

## Testing Patterns

### 1. Mock WordPress Service
```python
class MockWordPressService(WordPressService):
    """Mock service for testing WordPress integration"""
    
    async def publish_post(self, post_data: WordPressPost) -> Dict[str, Any]:
        return {
            "success": True,
            "post_id": 12345,
            "url": "https://example.com/test-post",
            "wordpress_data": post_data.dict()
        }
```

### 2. Chain Testing
```python
import pytest
from langchain_core.runnables import RunnableLambda

@pytest.mark.asyncio
async def test_wordpress_chain():
    """Test WordPress chain with mock service"""
    mock_service = MockWordPressService(test_config)
    test_chain = create_wordpress_chain(test_config)
    
    result = await test_chain.ainvoke({
        "content": "Test content",
        "content_type": "general"
    })
    
    assert result["wordpress_result"]["success"]
    assert "post_id" in result["wordpress_result"]
```

---

**Key Principles:**
1. **Simple chain composition over complex classes**: `content_chain | wordpress_formatter | publisher`
2. **Use LangChain Hub instead of custom templates**: `hub.pull("rlm/rag-prompt")`
3. Always use PydanticOutputParser for structured WordPress data
4. Integrate WordPress steps as RunnableLambda in LCEL chains  
5. Handle errors gracefully with fallback parsers
6. Validate content before publishing
7. Use configuration objects for WordPress settings
8. Test with mock services
9. Follow composable chain patterns over monolithic functions
10. **Prefer native LangChain primitives over custom implementations** 