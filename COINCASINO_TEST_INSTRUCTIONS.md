# Coincasino Complete Test Instructions

## Quick Start

The test is currently waiting for the **OPENAI_API_KEY** to be set. Here's how to run it:

### Option 1: Using .env file (Recommended)

1. Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=AIzaSyAtLmUjequWO4pwVTDzEl3Z66P12YPtZtE
WORDPRESS_URL=https://www.crashcasino.io
WORDPRESS_USERNAME=nmlwh
WORDPRESS_PASSWORD=your-wordpress-app-password
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_KEY=your-supabase-service-key
```

2. Run the test:

```bash
./run_coincasino_test.sh
```

### Option 2: Export environment variables

```bash
export OPENAI_API_KEY='your-openai-key'
export GOOGLE_API_KEY='AIzaSyAtLmUjequWO4pwVTDzEl3Z66P12YPtZtE'
export WORDPRESS_URL='https://www.crashcasino.io'
export WORDPRESS_USERNAME='nmlwh'
export WORDPRESS_PASSWORD='your-app-password'

python3 test_coincasino_simple.py
```

### Option 3: Direct Python with inline env

```bash
OPENAI_API_KEY='your-key' GOOGLE_API_KEY='AIzaSyAtLmUjequWO4pwVTDzEl3Z66P12YPtZtE' python3 test_coincasino_simple.py
```

## What the Test Does

The complete test runs the full workflow:

1. **🔍 Research Agent** (1-2 minutes)
   - Performs comprehensive 95-field casino intelligence extraction
   - Researches coincasino.com across 10 categories
   - Stores results in Supabase for reuse

2. **✍️ Writing Agent** (~30 seconds)
   - Generates rich HTML content
   - Adds authoritative links (non-competitor organizations)
   - Formats with tables, lists, proper structure

3. **🖼️ Image Agent** (~30 seconds)
   - Detects content type (casino review)
   - Uses Playwright for casino screenshots (logo, lobby)
   - Falls back to DataForSEO or Gemini if needed
   - Embeds images contextually in HTML

4. **🔗 Affiliate Agent** (~10 seconds)
   - Inserts affiliate links into content
   - Validates and tracks links

5. **📮 Publishing Agent** (~10 seconds)
   - Publishes to WordPress (crashcasino.io)
   - Uploads images to WordPress media library
   - Creates post with featured image

## Expected Output

- ✅ Content: ~10,000+ characters of rich HTML
- ✅ Images: 2-5 images (screenshots + generated)
- ✅ Affiliate Links: 2-3 contextual links
- ✅ Published Post: WordPress post ID and URL

## Troubleshooting

### Test Hangs on Research

The comprehensive research can take 1-2 minutes. This is normal as it:
- Loads multiple URLs in parallel
- Extracts 95+ fields of casino intelligence
- Processes data across 10 categories

If it takes longer than 5 minutes, check:
- Internet connection
- Supabase connectivity
- OpenAI API rate limits

### Missing WordPress Publishing

If publishing is skipped:
- Check WordPress credentials are set
- Verify WordPress REST API is enabled
- Check site URL is accessible

### No Images Generated

If no images are found:
- Playwright may fail on casino anti-scraping (normal)
- DataForSEO requires credentials
- Gemini generation requires GOOGLE_API_KEY

## Test Files

- `test_coincasino_simple.py` - Main test script
- `run_coincasino_test.sh` - Shell wrapper with setup
- `test_coincasino_output.html` - Generated HTML output
- `test_coincasino_output.log` - Full execution log

## Next Steps

Once you set your OPENAI_API_KEY, run:

```bash
python3 test_coincasino_simple.py
```

Or use the shell wrapper:

```bash
./run_coincasino_test.sh
```

