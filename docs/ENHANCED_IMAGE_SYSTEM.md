# Enhanced Image System for Agent-Based CMS

## Overview

The Enhanced Image System provides intelligent, multi-strategy image acquisition and contextual placement for rich HTML content generation. It automatically detects content type and uses the most appropriate image acquisition strategy.

## Content Type Detection

The system automatically detects content type and applies appropriate strategies:

### Casino Reviews
- **Strategy**: Playwright screenshots (logos, lobbies) + DataForSEO fallback
- **Use Case**: Casino site reviews where anti-scraping measures prevent direct image access
- **Images**: Casino logos, lobby screenshots, platform interface

### Game Reviews
- **Strategy**: DataForSEO image search
- **Use Case**: Reviews of specific games (Aviator, Crash, Slots, Live Roulette)
- **Images**: Game screenshots, provider logos, gameplay images

### Bonus Articles
- **Strategy**: Gemini-optimized prompts + DataForSEO
- **Use Case**: Bonus offers, promotions, welcome bonuses
- **Images**: Illustrations, promotional graphics, bonus-related imagery

### General Articles
- **Strategy**: DataForSEO + Gemini prompts
- **Use Case**: General gambling/casino content
- **Images**: Relevant stock photos, illustrations

## Architecture

### Components

1. **ContentTypeDetector**: Detects content type from query and content
2. **EnhancedImageSystem**: Main orchestrator for multi-strategy image acquisition
3. **ImageStrategy**: Configuration for each content type
4. **Contextual Placement**: Smart HTML embedding (hero, inline, gallery)

### Image Acquisition Strategies

#### 1. Playwright Screenshots (Casino Reviews)
- Uses Playwright screenshot engine
- Captures casino logos and lobby screenshots
- Handles anti-scraping measures
- Returns structured image metadata

#### 2. DataForSEO Search (Games, General)
- Uses DataForSEO image search API
- Generates content-specific search queries
- Filters by image type, size, quality
- Returns high-quality game and general images

#### 3. Gemini Optimization (Bonus Articles)
- Uses Gemini 2.5 Flash to generate optimized search prompts
- Enhances image search with AI-generated queries
- Falls back to DataForSEO with optimized queries

## Contextual Image Placement

Images are embedded into HTML content with intelligent placement:

### Hero Image
- Placed after title/first paragraph
- High-quality, landscape orientation preferred
- Eager loading for immediate display

### Inline Images
- Placed after relevant section headers (H2, H3)
- Maximum 3 inline images
- Lazy loading for performance

### Gallery Section
- Remaining images displayed in responsive grid
- Gallery at end of content
- Responsive design (auto-fit columns)

## HTML Formatting

The Writing Agent now returns **rich HTML content** with:

- ✅ Proper HTML structure (h1, h2, h3, p, ul, ol, table)
- ✅ Responsive images with lazy loading
- ✅ Proper alt text and captions
- ✅ CSS classes for WordPress styling
- ✅ Authoritative links (non-competitor organizations)
- ✅ Tables with proper HTML structure

## Integration Points

### Writing Agent
- Returns HTML-formatted content
- Adds authoritative links automatically
- Ensures proper HTML structure

### Image Agent
- Uses Enhanced Image System when available
- Falls back to basic search if enhanced system unavailable
- Embeds images contextually into content

### Publishing Agent
- WordPress publisher handles embedded images
- Uses bulletproof uploader for reliable WordPress uploads
- RichHTMLFormatter enhances final HTML

## Bulletproof Image Uploader

The system uses V1 bulletproof patterns for reliable WordPress uploads:

- ✅ Retry logic with exponential backoff
- ✅ Image optimization (resize, format conversion)
- ✅ WordPress media library integration
- ✅ Proper error handling and recovery

## Authoritative Links

The Writing Agent automatically adds links to high-authority, non-competitor organizations:

- **Responsible Gambling**: GambleAware, GAMSTOP, BeGambleAware
- **Regulatory**: UK Gambling Commission, MGA, Curacao eGaming
- **Game Providers**: NetEnt, Evolution Gaming, Microgaming
- **Payment Security**: PCI DSS, SSL certificates
- **Industry Standards**: eCOGRA, iTech Labs

## Usage Example

```python
from src.agents.factory import create_agent_based_cms

# Create CMS with enhanced image system
cms = create_agent_based_cms(
    enable_images=True,
    max_images=5,
    upload_to_wordpress=True
)

# Run workflow
result = await cms.run(
    query="Betway Casino Review 2025",
    target_sites=["crashcasino.io"]
)

# Result contains:
# - final_content: Rich HTML with embedded images
# - images: List of image metadata
# - wordpress_media_ids: WordPress media IDs
# - content_type: Detected content type
```

## Configuration

### Environment Variables

- `GOOGLE_API_KEY`: For Gemini image prompt optimization
- `DATAFORSEO_LOGIN`: DataForSEO credentials
- `DATAFORSEO_PASSWORD`: DataForSEO credentials
- `WORDPRESS_URL`: WordPress site URL
- `WORDPRESS_USERNAME`: WordPress username
- `WORDPRESS_PASSWORD`: WordPress application password

### Content Type Detection Rules

**Casino Review**: Contains "casino review", "casino rating", casino name patterns
**Game Review**: Contains game names (aviator, crash, roulette, slots)
**Bonus Article**: Contains "bonus", "promotion", "welcome bonus"
**General Article**: Default fallback

## Benefits

1. **Intelligent Strategy Selection**: Automatically chooses best image source
2. **Contextual Placement**: Images placed where they make sense
3. **Rich HTML Output**: Properly formatted, WordPress-ready HTML
4. **Authoritative Links**: Adds credibility with non-competitor links
5. **Bulletproof Uploads**: Reliable WordPress integration
6. **Anti-Scraping Handling**: Playwright handles casino site protections

## Future Enhancements

- [ ] Gemini image generation integration (when available)
- [ ] Image quality scoring and filtering
- [ ] Automatic image cropping and optimization
- [ ] Image copyright detection
- [ ] Multi-language image search support

