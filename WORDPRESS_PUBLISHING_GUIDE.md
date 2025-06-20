# 🎰 WordPress Casino Review Publishing Guide

## Overview
Your RAG system now includes advanced WordPress publishing with automatic category assignment, comprehensive metadata generation, and SEO optimization specifically designed for casino review content.

## 🚀 Quick Setup

### 1. Discover Your WordPress Structure
First, run the discovery tool to understand your site's categories and custom fields:

```bash
python discover_wordpress_structure.py
```

This will:
- ✅ Scan your WordPress categories and find casino-related ones
- ✅ Discover existing custom fields used for casino reviews
- ✅ Generate a configuration file customized for your site
- ✅ Create a JSON backup of your site structure

### 2. Configure Your Categories

Update the category mapping in `src/chains/universal_rag_lcel.py` (line ~2387):

```python
# ✅ CUSTOMIZE: Your WordPress category mapping
category_mapping = {
    "casino_review": [5, 12],      # UPDATE: Your "Casino Reviews" + "Featured Reviews" IDs
    "game_guide": [8],             # UPDATE: Your "Game Guides" category ID
    "bonus_analysis": [15],        # UPDATE: Your "Bonuses & Promotions" category ID
    "comparison": [18, 5],         # UPDATE: Your "Comparisons" category ID
    "news": [22],                  # UPDATE: Your "Industry News" category ID
    "tutorial": [25],              # UPDATE: Your "How-To Guides" category ID
    "regulatory": [28],            # UPDATE: Your "Regulations & Legal" category ID
    "general": [1],                # UPDATE: Your default category ID
}
```

### 3. Configure Your Custom Fields

Update the custom fields mapping in `src/chains/universal_rag_lcel.py` (line ~2425):

```python
# ✅ CUSTOMIZE: Your WordPress custom field names
custom_fields = {
    # Casino-specific metadata
    "casino_rating": structured_metadata.get("casino_rating", 0),
    "bonus_amount": structured_metadata.get("bonus_amount", ""),
    "license_info": structured_metadata.get("license_info", ""),
    "min_deposit": structured_metadata.get("min_deposit", ""),
    "withdrawal_time": structured_metadata.get("withdrawal_time", ""),
    "wagering_requirements": structured_metadata.get("wagering_requirements", ""),
    "mobile_compatible": structured_metadata.get("mobile_compatible", True),
    "live_chat_support": structured_metadata.get("live_chat_support", False),
    
    # Game and software providers
    "game_providers": ",".join(structured_metadata.get("game_providers", [])),
    "payment_methods": ",".join(structured_metadata.get("payment_methods", [])),
    
    # Update field names to match your WordPress custom fields
}
```

### 4. Set Environment Variables

Add to your `.env` file:

```bash
# WordPress Publishing Configuration
WORDPRESS_SITE_URL=https://yoursite.com
WORDPRESS_USERNAME=your_username
WORDPRESS_APP_PASSWORD=your_application_password

# Optional: Default publishing settings
WORDPRESS_DEFAULT_STATUS=draft
WORDPRESS_DEFAULT_AUTHOR_ID=1
```

## 📝 How to Publish Casino Reviews

### Method 1: Enable Publishing in Query
```python
# When calling the RAG chain, add the publish flag
response = await rag_chain.ainvoke({
    "question": "Review Betway Casino safety and bonuses",
    "publish_to_wordpress": True  # ✅ This enables WordPress publishing
})

# Check publishing results
if response.metadata.get("wordpress_published"):
    print(f"✅ Published to WordPress!")
    print(f"Post ID: {response.metadata['wordpress_post_id']}")
    print(f"Edit URL: {response.metadata['wordpress_edit_url']}")
    print(f"Category: {response.metadata['wordpress_category']}")
    print(f"Custom Fields: {response.metadata['wordpress_custom_fields_count']}")
```

### Method 2: Enable in Configuration
```python
# Create chain with auto-publishing for casino reviews
chain = create_universal_rag_chain(
    enable_wordpress_publishing=True,
    # ... other settings
)

# Publishing will be triggered based on content type
```

## 🎯 What Gets Published Automatically

### 1. Smart Category Assignment
The system automatically determines content type and assigns appropriate categories:

- **Casino Review Queries**: "Is Betway Casino safe?" → `casino_review` category
- **Bonus Questions**: "What's the welcome bonus at 888 Casino?" → `bonus_analysis` category  
- **Game Guides**: "How to play blackjack strategy" → `game_guide` category
- **Comparisons**: "Betway vs 888 Casino" → `comparison` category

### 2. SEO-Optimized Titles
- **Casino Reviews**: "Betway Casino Review 2024: 8.5/10 Rating & Detailed Analysis"
- **Bonus Analysis**: "888 Casino Bonus Review: $200 Welcome Offer Analysis"  
- **Comparisons**: "Casino Comparison 2024: Betway vs 888 Casino - Expert Analysis"

### 3. Comprehensive Metadata (Custom Fields)
```json
{
  "casino_rating": 8.5,
  "bonus_amount": "$200 + 100 Free Spins",
  "license_info": "Malta Gaming Authority (MGA), UK Gambling Commission",
  "min_deposit": "$10",
  "withdrawal_time": "24-48 hours",
  "wagering_requirements": "35x",
  "mobile_compatible": true,
  "live_chat_support": true,
  "game_providers": "NetEnt,Microgaming,Evolution Gaming",
  "payment_methods": "Visa,Mastercard,PayPal,Skrill",
  "confidence_score": 0.89,
  "sources_count": 7,
  "fact_checked": true,
  "pros_list": "Licensed and regulated|Excellent game selection|24/7 support",
  "cons_list": "High wagering requirements|Limited crypto options",
  "verdict": "Highly recommended casino with excellent features and strong regulation."
}
```

### 4. Smart Tag Generation
- **Content-based**: "casino review", "online casino", "2024"
- **License-based**: "MGA licensed", "UKGC licensed" 
- **Provider-based**: "NetEnt games", "Evolution Gaming"
- **Payment-based**: "PayPal casino", "crypto casino"
- **Casino-specific**: "Betway casino", "Betway review"

### 5. Featured Images
Automatically selects the best image from DataForSEO search results based on:
- ✅ High relevance score
- ✅ Appropriate dimensions (800x400+ for featured images)
- ✅ Casino-related content in alt text

## 🔧 Advanced Configuration

### Custom Category Detection
You can add custom logic for category detection:

```python
async def _determine_wordpress_category(self, query: str, query_analysis, structured_metadata):
    # Add your custom logic here
    if "crypto" in query.lower():
        return "crypto_casino", [30]  # Your crypto casino category
    
    if "mobile" in query.lower():
        return "mobile_casino", [31]  # Your mobile casino category
    
    # ... existing logic
```

### Custom Field Mapping
Map your existing WordPress custom fields:

```python
# If your site uses different field names
field_mapping = {
    "casino_rating": "rating_score",           # Your actual field name
    "bonus_amount": "welcome_bonus",           # Your actual field name
    "license_info": "licensing_details",       # Your actual field name
    # ... add all your custom field mappings
}
```

## 📊 Monitoring Publishing Results

### Check Publishing Status
```python
response = await rag_chain.ainvoke({
    "question": "Review 888 Casino",
    "publish_to_wordpress": True
})

# Publishing metadata available in response
publishing_info = {
    "published": response.metadata.get("wordpress_published", False),
    "post_id": response.metadata.get("wordpress_post_id"),
    "post_url": response.metadata.get("wordpress_url"),
    "edit_url": response.metadata.get("wordpress_edit_url"),
    "category": response.metadata.get("wordpress_category"),
    "custom_fields_count": response.metadata.get("wordpress_custom_fields_count", 0),
    "tags_count": response.metadata.get("wordpress_tags_count", 0),
    "error": response.metadata.get("wordpress_error")
}
```

### View Publishing Logs
Publishing activities are logged to your Supabase database in the `wordpress_publications` table for audit trails.

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Failed**
   - ✅ Verify your WordPress Application Password
   - ✅ Check username is correct
   - ✅ Ensure WordPress REST API is enabled

2. **Wrong Categories**
   - ✅ Run discovery tool to get correct category IDs
   - ✅ Update category mapping in configuration

3. **Missing Custom Fields**
   - ✅ Check your WordPress custom field names
   - ✅ Update field mapping to match your site

4. **No Featured Image**
   - ✅ Ensure DataForSEO integration is enabled
   - ✅ Check image search results in logs

### Debug Mode
Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# This will show detailed WordPress publishing logs
```

## 🎯 Best Practices

### 1. Content Review Workflow
- ✅ Posts are created as **drafts** by default
- ✅ Review content before publishing
- ✅ Use the provided edit URLs to access WordPress admin

### 2. SEO Optimization
- ✅ Titles are automatically SEO-optimized
- ✅ Meta descriptions are generated from content
- ✅ Tags are relevant and comprehensive

### 3. Quality Assurance
- ✅ High-confidence responses get comprehensive metadata
- ✅ Source quality is tracked and included
- ✅ Fact-checking flags are set automatically

### 4. Category Strategy
- ✅ Use multiple categories for broader reach
- ✅ Include both general and specific categories
- ✅ Maintain consistent categorization

## 📞 Support

If you need help configuring WordPress publishing:

1. **Run the discovery tool** first to understand your site structure
2. **Check the generated configuration files** for your specific setup
3. **Test with a simple query** before using in production
4. **Monitor the publishing logs** for troubleshooting

The enhanced WordPress publishing system automatically handles all the complexity of casino review publishing while giving you full control over categories, metadata, and content structure. 