# Remaining Issues Investigation & Fixes

## ✅ Issue 1: Screenshot Schema Mismatch - FIXED

**Problem**: `'ScreenshotResult' object has no attribute 'screenshot_id'`

**Root Cause**: 
- `ScreenshotResult` dataclass doesn't have `screenshot_id`, `format`, `width`, `height`, or `storage_path` attributes
- Actual attributes: `success`, `screenshot_data`, `error_message`, `url`, `timestamp`, `file_size`, `viewport_size`, `element_info`

**Fix Applied**:
- Updated `screenshot_tool` in `src/agents/tools/research_tools.py` to use actual `ScreenshotResult` attributes
- Extract `width`/`height` from `viewport_size` dict
- Return `screenshot_data` (bytes) instead of non-existent fields
- Remove references to `screenshot_id`, `format`, `storage_path`

**Location**: `src/agents/tools/research_tools.py` lines 550-565

---

## ✅ Issue 2: RAG Retrieval Failure - FIXED

**Problem**: `Document retrieval failed:` error with empty results

**Root Causes Identified**:

1. **Similarity Threshold Logic Error**:
   - SupabaseVectorStore returns **cosine distance** (lower = more similar)
   - Code was filtering: `score >= similarity_threshold` (WRONG - higher scores = less similar)
   - Should filter: `score <= (1 - similarity_threshold)` (correct - lower distance = higher similarity)

2. **Vector Store May Be Empty**:
   - Documents need to be stored in Supabase `documents` table first
   - Research agent stores data but might not be chunking/storing properly

**Fixes Applied**:

1. **Fixed Similarity Threshold Logic**:
   ```python
   # Before (WRONG):
   filtered_results = [doc for doc, score in results if score >= self.config.similarity_threshold]
   
   # After (CORRECT):
   max_distance = 1.0 - self.config.similarity_threshold
   filtered_results = [doc for doc, score in results if score <= max_distance]
   ```

2. **Enhanced Error Logging**:
   - Better error messages to identify if vector store is empty vs query issue

**Location**: `src/chains/universal_rag_lcel.py` lines 450-457

**Additional Notes**:
- Vector store auto-initialization works correctly
- Need to ensure research data is being stored in Supabase during research phase
- Check that `_store_casino_intelligence_in_supabase` is properly chunking and storing documents

---

## ⚠️ Issue 3: Affiliate Links Database Empty

**Problem**: Query returns 0 affiliate links

**Investigation**:
- Database table `affiliate_links` exists (migration 006_affiliate_links.sql)
- Table structure is correct
- Query logic in `AffiliateLinkManager.get_affiliate_links()` is correct
- **Issue**: Table is empty - no affiliate links have been populated

**Solution Required**:
1. Populate `affiliate_links` table with CyBet affiliate links
2. Ensure category matches ('casino' vs 'casinos')
3. Set `active=True` for links to be retrieved

**SQL to Populate** (example):
```sql
INSERT INTO affiliate_links (
    casino_name, 
    affiliate_url, 
    category, 
    active, 
    description,
    utm_source,
    utm_medium,
    utm_campaign
) VALUES (
    'CyBet',
    'https://cybet.com/?ref=YOUR_AFFILIATE_ID',
    'casino',
    true,
    'CyBet Casino affiliate link',
    'crashcasino',
    'affiliate',
    'cybet_review'
);
```

**Location**: `src/integrations/affiliate_link_manager.py` lines 61-100

---

## ⚠️ Issue 4: Content Length (3,181 chars)

**Problem**: Content too short (should be 5,000-10,000+ chars)

**Possible Causes**:
1. **Empty Research Data**: Research returns 0 fields → limited context for generation
2. **RAG Retrieval Failing**: Vector store empty or similarity threshold too strict
3. **Template Selection**: May not be selecting casino_review template properly
4. **LLM Token Limits**: May be hitting max_tokens limit

**Investigation Steps**:
1. ✅ Fixed research field extraction (Issue #1 in previous fixes)
2. ✅ Fixed RAG similarity threshold logic
3. ✅ Fixed template selection enum issue
4. ⚠️ Need to verify research data is being stored in vector store
5. ⚠️ Check if LLM max_tokens is limiting content length

**Next Steps**:
- Verify research data storage during next test run
- Check if content generation is hitting token limits
- Ensure template is providing sufficient structure for long-form content

---

## Summary of Fixes

### Completed Fixes ✅
1. Screenshot schema mismatch - Fixed attribute access
2. RAG similarity threshold - Fixed distance comparison logic
3. Research field extraction - Fixed counting logic (previous session)
4. Image upload base64 - Fixed conversion and site config (previous session)
5. Template selection - Fixed enum value passing (previous session)
6. Authoritative links import - Fixed langchain_text_splitters import (previous session)

### Remaining Actions ⚠️
1. Populate affiliate_links table with CyBet affiliate link
2. Verify research data is being stored in Supabase vector store
3. Test content length after all fixes applied

---

## Testing Recommendations

1. **Run production test again** with all fixes applied
2. **Monitor logs** for:
   - Research field extraction counts
   - Vector store document counts
   - RAG retrieval success/failure
   - Content generation length
3. **Check Supabase**:
   - Verify documents table has content
   - Verify affiliate_links table populated
   - Check vector embeddings are created


