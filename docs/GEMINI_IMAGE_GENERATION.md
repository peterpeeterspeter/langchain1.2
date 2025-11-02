# Gemini Image Generation Integration

## Overview

The Enhanced Image System now uses **Gemini 2.5 Flash native image generation** for creating high-quality images directly from text prompts. This is especially useful for bonus articles and general content where custom illustrations are needed.

## Reference Documentation

- **Official Docs**: https://ai.google.dev/gemini-api/docs/image-generation
- **Model**: `gemini-2.5-flash-image`
- **API**: Google Generative AI SDK (`google.genai`)

## Features

### Direct Image Generation
- **Text-to-Image**: Generate images from descriptive prompts
- **High Quality**: Professional illustrations suitable for articles
- **Aspect Ratios**: Supports multiple aspect ratios (16:9 default for articles)
- **SynthID Watermark**: All generated images include SynthID watermark for authenticity

### Integration Points

1. **Bonus Articles**: Generates promotional graphics and bonus illustrations
2. **General Content**: Creates custom illustrations when stock photos aren't suitable
3. **Fallback Strategy**: Uses DataForSEO if Gemini generation fails

## Usage

### Setup

Set your Gemini API key in environment variables:

```bash
export GOOGLE_API_KEY="your-api-key-here"
# OR
export GEMINI_API_KEY="your-api-key-here"
```

### Content Type Detection

The system automatically uses Gemini generation for:
- **Bonus Articles**: Welcome bonuses, promotions, free spins
- **General Articles**: When appropriate prompts can be generated

### Image Generation Process

1. **Prompt Generation**: System generates optimized prompts from content
2. **Image Creation**: Gemini generates images (16:9 aspect ratio for articles)
3. **Base64 Encoding**: Images are converted to base64 for storage/upload
4. **WordPress Upload**: Images are uploaded to WordPress media library

## Code Example

```python
from src.integrations.enhanced_image_system import EnhancedImageSystem

# Initialize with Gemini API key
image_system = EnhancedImageSystem(
    gemini_api_key="your-api-key"
)

# Generate images for bonus article
images, strategy = await image_system.acquire_images(
    query="Welcome Bonus Guide 2025",
    content="Comprehensive guide to casino welcome bonuses...",
    max_images=3
)

# Images will be generated with prompts like:
# - "Professional illustration of casino welcome bonus offer"
# - "Casino bonus graphic with gift box and coins"
# - "Free spins promotion illustration"
```

## Pricing

According to the [official documentation](https://ai.google.dev/gemini-api/docs/image-generation):
- **Token-based pricing**: $30 per 1 million tokens
- **Image output**: 1290 tokens per image (flat rate, up to 1024x1024px)
- **Cost per image**: ~$0.039 per image

## Aspect Ratios Available

| Aspect Ratio | Resolution | Use Case |
|-------------|------------|----------|
| 16:9 | 1344x768 | **Default** - Articles, banners |
| 1:1 | 1024x1024 | Social media, thumbnails |
| 4:3 | 1184x864 | Standard content |
| 21:9 | 1536x672 | Wide banners |

## Benefits Over Search-Based Approaches

1. **Custom Illustrations**: Generate exactly what you need
2. **No Copyright Issues**: Generated images are original
3. **Consistent Style**: All images match your brand/content style
4. **No Stock Photo Limits**: Generate unlimited variations
5. **Contextual Relevance**: Images match your content perfectly

## Limitations

1. **Latency**: Higher than search-based (generation takes time)
2. **Cost**: ~$0.039 per image vs free search results
3. **Watermark**: All images include SynthID watermark
4. **Preview Status**: Currently in preview (production usage allowed)

## Integration with Enhanced Image System

The Gemini image generation is seamlessly integrated:

1. **Content Type Detection**: Automatically selects Gemini for bonus/general articles
2. **Multi-Strategy**: Falls back to DataForSEO if Gemini fails
3. **Contextual Placement**: Generated images placed intelligently in HTML
4. **WordPress Upload**: Automatic upload to WordPress media library

## Error Handling

- Falls back to DataForSEO if Gemini API key not available
- Falls back to DataForSEO if generation fails
- Logs warnings but continues workflow
- Graceful degradation ensures content generation continues

## Best Practices

1. **Use for Bonus Articles**: Perfect for promotional graphics
2. **Combine with Search**: Use Gemini for custom + search for game screenshots
3. **Prompt Quality**: System generates optimized prompts automatically
4. **Aspect Ratio**: 16:9 works best for article layouts
5. **Cost Management**: Limit max_images for Gemini generation

## Example Output

Generated images include:
- **URL**: Base64 data URL (`data:image/png;base64,...`)
- **Title**: Generated from prompt
- **Alt Text**: SEO-optimized alt text
- **Dimensions**: Width/height metadata
- **Source**: "gemini"
- **Type**: "generated"
- **Quality Score**: 0.9 (high quality)

## Future Enhancements

- [ ] Image editing capabilities (text + image to image)
- [ ] Style transfer from reference images
- [ ] Iterative refinement through conversation
- [ ] Custom aspect ratios per content type
- [ ] Batch generation optimization

