# Agent-Based CMS System

## Overview

Complete LangGraph-based agent orchestration system for automated content creation and publishing. This system uses 5 specialized agents working together to research, write, optimize, and publish content to WordPress sites.

## Quick Start

```python
from src.agents.factory import create_agent_based_cms

# Create CMS
cms = create_agent_based_cms()

# Run workflow
state = await cms.run(
    query="Betway Casino Review 2025",
    target_sites=["crashcasino"]
)
```

## Architecture

### Agents

1. **ResearchAgent** - Gathers information
   - Web search (Tavily)
   - Comprehensive research (WebBaseLoader)
   - Screenshot capture (Playwright)
   - Casino intelligence extraction

2. **WritingAgent** - Generates content
   - Universal RAG Chain integration
   - Template system v2.0
   - Content refinement
   - SEO optimization

3. **AffiliateAgent** - Inserts affiliate links
   - Context-aware link detection
   - Automatic insertion
   - UTM tracking parameters
   - Link validation

4. **ImageAgent** - Handles images
   - DataForSEO image search
   - Intelligent selection
   - Alt text generation
   - WordPress upload

5. **PublishingAgent** - Publishes content
   - Multi-site publishing
   - Content adaptation
   - Site registry management

### Workflow

```
Query Input
    ↓
Research Agent (web search, deep research, screenshots)
    ↓
Writing Agent (content generation, templates, SEO)
    ↓
    ├─→ Affiliate Agent (link insertion)
    └─→ Image Agent (image search & upload)
    ↓
Publishing Agent (multi-site publishing)
    ↓
Final State (published posts, URLs, metadata)
```

## State Schema

The `ArticleCMSState` TypedDict manages workflow state:

```python
{
    "query": str,                    # Input query
    "target_sites": List[str],       # WordPress site IDs
    "research_data": Dict,           # Research findings
    "final_content": str,            # Generated content
    "affiliate_links": List[Dict],   # Inserted links
    "images": List[Dict],            # Selected images
    "published_posts": List[Dict],   # Published posts
    "errors": List[str],            # Error messages
    "agent_statuses": Dict[str, str] # Agent execution status
}
```

## Tools

Each agent has access to specialized LangChain tools:

- **Research Tools**: `web_search_tool`, `comprehensive_research_tool`, `screenshot_tool`, `casino_intelligence_tool`
- **Writing Tools**: `content_generation_tool`, `template_selection_tool`, `content_refinement_tool`, `seo_optimization_tool`
- **Affiliate Tools**: `affiliate_link_database_tool`, `link_insertion_tool`, `link_validation_tool`, `tracking_parameter_tool`
- **Image Tools**: `image_search_tool`, `image_selection_tool`, `alt_text_generation_tool`, `wordpress_image_upload_tool`
- **Publishing Tools**: `wordpress_publish_tool`, `site_registry_tool`, `content_adaptation_tool`

## Configuration

### WordPress Site Registry

Register sites before publishing:

```python
from src.integrations.wordpress_site_registry import WordPressSiteRegistry, WordPressSiteConfig

registry = WordPressSiteRegistry()

await registry.register_site(WordPressSiteConfig(
    site_id="crashcasino",
    site_name="Crash Casino",
    site_url="https://www.crashcasino.io",
    username="nmlwh",
    application_password="your-password",
    default_status="publish"
))
```

### Affiliate Links

Affiliate links are stored in Supabase. The system automatically detects opportunities and inserts links contextually based on keywords.

## Database Setup

Run migrations:

```sql
-- Affiliate links
\i database/migrations/006_affiliate_links.sql

-- WordPress sites
\i database/migrations/007_wordpress_sites.sql
```

## Example

See `examples/agent_based_cms_example.py` for a complete example.

## Integration

The Agent-Based CMS integrates with existing Universal RAG CMS components:

- Universal RAG Chain (content generation)
- Template System v2.0 (34 specialized templates)
- DataForSEO (image search)
- WordPress Publisher (multi-site publishing)
- Supabase (storage and registries)

## Error Handling

All agents include:
- Retry logic with exponential backoff
- Error collection in state
- Graceful degradation
- Status tracking

## Performance

Typical execution times:
- Research: 30-60s
- Writing: 20-40s
- Affiliate: 5-10s
- Images: 20-40s
- Publishing: 10-20s per site

Total: ~2-3 minutes for single site.

## Documentation

See `docs/AGENT_BASED_CMS.md` for complete documentation.

