# Agent-Based CMS Documentation

## Overview

The Agent-Based CMS is a LangGraph-powered content management system that orchestrates multiple specialized agents to research, write, optimize, and publish content to WordPress sites.

## Architecture

### Components

1. **Research Agent** - Gathers comprehensive information using web search, comprehensive research, and screenshots
2. **Writing Agent** - Generates high-quality content using Universal RAG Chain and templates
3. **Affiliate Link Agent** - Intelligently inserts affiliate links with tracking
4. **Image Agent** - Searches, selects, and uploads images to WordPress
5. **Publishing Agent** - Publishes content to multiple WordPress sites

### Workflow

```
Query → Research → Writing → Affiliate + Images (parallel) → Publishing
```

## Quick Start

### Basic Usage

```python
from src.agents.factory import create_agent_based_cms

# Create CMS orchestrator
cms = create_agent_based_cms(
    llm_model="gpt-4o-mini",
    enable_research=True,
    enable_writing=True,
    enable_affiliate=True,
    enable_images=True,
    enable_publishing=True
)

# Run workflow
final_state = await cms.run(
    query="Betway Casino Review 2025",
    target_sites=["crashcasino"]
)
```

### Configuration

#### Environment Variables

Required:
- `OPENAI_API_KEY` - For LLM operations
- `TAVILY_API_KEY` - For web search
- `SUPABASE_URL` - For database storage
- `SUPABASE_SERVICE_KEY` - For database access
- `DATAFORSEO_LOGIN` - For image search
- `DATAFORSEO_PASSWORD` - For image search

WordPress (per site):
- `WORDPRESS_URL` - WordPress site URL
- `WORDPRESS_USERNAME` - WordPress username
- `WORDPRESS_PASSWORD` - WordPress application password

## WordPress Site Registry

### Register a Site

```python
from src.integrations.wordpress_site_registry import WordPressSiteRegistry, WordPressSiteConfig

registry = WordPressSiteRegistry()

site_config = WordPressSiteConfig(
    site_id="crashcasino",
    site_name="Crash Casino",
    site_url="https://www.crashcasino.io",
    username="nmlwh",
    application_password="your-app-password",
    default_status="publish",
    default_category_ids=[1],
    default_tags=["casino", "review"]
)

await registry.register_site(site_config)
```

## Affiliate Link Management

### Register Affiliate Links

Affiliate links are stored in Supabase. The system automatically detects opportunities and inserts links contextually.

See `database/migrations/006_affiliate_links.sql` for schema.

## Database Migrations

Run migrations to set up required tables:

1. `006_affiliate_links.sql` - Affiliate link registry
2. `007_wordpress_sites.sql` - WordPress site registry

## State Management

The workflow uses `ArticleCMSState` to pass data between agents:

```python
{
    "query": str,
    "target_sites": List[str],
    "research_data": Dict,
    "final_content": str,
    "affiliate_links": List[Dict],
    "images": List[Dict],
    "published_posts": List[Dict],
    ...
}
```

## Advanced Usage

### Custom Agent Configuration

```python
from src.agents.research_agent import ResearchAgent
from src.agents.writing_agent import WritingAgent
from src.agents.orchestrator import ArticleCMSOrchestrator

research_agent = ResearchAgent(
    llm=ChatOpenAI(model="gpt-4"),
    enable_screenshots=True,
    enable_comprehensive_research=True
)

writing_agent = WritingAgent(
    llm=ChatOpenAI(model="gpt-4"),
    enable_refinement=True,
    enable_seo=True
)

orchestrator = ArticleCMSOrchestrator(
    research_agent=research_agent,
    writing_agent=writing_agent,
    # ... other agents
)
```

### State Checkpointing

Enable checkpointing for state persistence:

```python
cms = create_agent_based_cms(enable_checkpoints=True)
```

## Tools Reference

### Research Tools
- `web_search_tool` - Tavily web search
- `comprehensive_research_tool` - Deep research with WebBaseLoader
- `screenshot_tool` - Playwright screenshot capture
- `casino_intelligence_tool` - Structured casino data extraction

### Writing Tools
- `content_generation_tool` - Universal RAG Chain content generation
- `template_selection_tool` - Template selection
- `content_refinement_tool` - Content improvement
- `seo_optimization_tool` - SEO optimization

### Affiliate Tools
- `affiliate_link_database_tool` - Query affiliate links
- `link_insertion_tool` - Insert links into content
- `link_validation_tool` - Validate links
- `tracking_parameter_tool` - Generate UTM parameters

### Image Tools
- `image_search_tool` - DataForSEO image search
- `image_selection_tool` - Select best images
- `alt_text_generation_tool` - Generate alt text
- `wordpress_image_upload_tool` - Upload to WordPress

### Publishing Tools
- `wordpress_publish_tool` - Publish to WordPress
- `site_registry_tool` - Manage site registry
- `content_adaptation_tool` - Adapt content for sites

## Error Handling

All agents include retry logic with exponential backoff. Errors are collected in `state["errors"]`.

## Monitoring

Agent statuses are tracked in `state["agent_statuses"]`:
- `pending` - Not started
- `in_progress` - Currently executing
- `completed` - Successfully completed
- `failed` - Execution failed

## Examples

See `examples/agent_based_cms_example.py` for a complete example.

## Integration with Existing Systems

The Agent-Based CMS integrates with:
- Universal RAG Chain (content generation)
- Template System v2.0 (content templates)
- DataForSEO (image search)
- WordPress Publisher (publishing)
- Supabase (storage and registry)

## Performance

Typical workflow execution time:
- Research: 30-60 seconds
- Writing: 20-40 seconds
- Affiliate: 5-10 seconds
- Images: 20-40 seconds
- Publishing: 10-20 seconds per site

Total: ~2-3 minutes for single site, +10-20 seconds per additional site.

