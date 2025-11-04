# Integration Test Setup - API Keys Needed

## ✅ Credentials Provided

1. **WordPress (crashcasino.io)**
   - URL: https://crashcasino.io
   - User: nmlwh
   - Password: KA8Z 0guj 18Lq GpnY etRw ner0

2. **Supabase**
   - Project: ambjsovdhizjxwhhnbtd
   - Publishable Key: sb_publishable_RLIhxePcAYDm2p1_l-SnZg_na9oUV0P
   - ⚠️ **NEED: Service Role Key** (for database writes)

3. **DataForSEO**
   - Base64 Auth: cGVldGVycy5wZXRlckB0ZWxlbmV0LmJlOjY1NGIxY2ZjY2EwODRkMTk=

4. **Anthropic (Claude)**
   - ✅ Already have key

---

## ❓ Additional Keys Needed

### Required for Full Test:

1. **Tavily API** (Research Agent - Web Search)
   - Used by: web_search_tool
   - Get from: https://tavily.com
   - Purpose: Web search for casino information
   - **Priority: HIGH** - Research agent won't work without this

2. **Supabase Service Role Key** (Affiliate Agent)
   - Currently have: Publishable key (read-only)
   - Need: Service role key (read-write)
   - Purpose: Query affiliate link database
   - Location: Supabase dashboard → Settings → API → service_role key
   - **Priority: MEDIUM** - Can test without affiliate links

3. **OpenAI API Key** (Optional - for gpt-4o-mini)
   - Currently have: Anthropic (Claude)
   - Purpose: Alternative LLM
   - **Priority: LOW** - Can use Claude instead

### Optional (Nice to Have):

4. **Perplexity API** (Enhanced Research)
   - Purpose: Additional research capabilities
   - **Priority: LOW** - Not critical for basic test

---

## 🎯 What We Can Test Right Now

With current credentials:

✅ **Writing Agent** - Uses Claude only
✅ **Publishing Agent** - Has WordPress credentials
✅ **Image Agent** - Has DataForSEO credentials

⚠️ **Research Agent** - Needs Tavily for web search
⚠️ **Affiliate Agent** - Needs Supabase service role key

---

## 📝 Integration Test Plan

### Minimal Test (Current Credentials)
```
Query: "Betway Casino Review"
1. ⚠️ Research (skip - needs Tavily)
2. ✅ Writing (use mock research data)
3. ⚠️ Affiliate (skip - needs Supabase service key)
4. ✅ Image (use DataForSEO)
5. ✅ Publishing (to crashcasino.io)
```

### Full Test (With All Keys)
```
Query: "Betway Casino Review"
1. ✅ Research (Tavily + Claude)
2. ✅ Writing (Claude + research data)
3. ✅ Affiliate (Supabase)
4. ✅ Image (DataForSEO)
5. ✅ Publishing (WordPress)
```

---

## 🔑 Please Provide

**To run full test, please provide:**

1. **Tavily API Key** (Most important)
   ```
   TAVILY_API_KEY=tvly-...
   ```

2. **Supabase Service Role Key**
   ```
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
   ```

**Alternative: Run partial test with what we have**
- I can create a mock research data and test writing → image → publishing workflow
