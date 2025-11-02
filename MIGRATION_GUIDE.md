# Database Migration Guide

## Quick Setup

### Option 1: Supabase Dashboard (Easiest - Recommended)

1. **Go to Supabase Dashboard**
   - Visit: https://supabase.com/dashboard
   - Select your project: `ambjsovdhizjxwhhnbtd`

2. **Navigate to SQL Editor**
   - Click "SQL Editor" in the left sidebar
   - Click "New query"

3. **Run Migration 006: Affiliate Links**
   - Copy the SQL from `database/migrations/006_affiliate_links.sql`
   - Paste into SQL Editor
   - Click "Run" or press Cmd/Ctrl + Enter
   - Wait for "Success" message

4. **Run Migration 007: WordPress Sites**
   - Copy the SQL from `database/migrations/007_wordpress_sites.sql`
   - Paste into SQL Editor
   - Click "Run" or press Cmd/Ctrl + Enter
   - Wait for "Success" message

### Option 2: Using Python Script (If you have database password)

If you have your Supabase database password:

```bash
# Set database password
export SUPABASE_DB_PASSWORD="your-database-password"

# Run migrations
python3 execute_migrations.py
```

**To get your database password:**
1. Go to Supabase Dashboard > Settings > Database
2. Find "Connection string" section
3. The password is shown there (or reset it if needed)

### Option 3: Using psql directly

```bash
# Get connection string from Supabase Dashboard > Settings > Database
# Format: postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres

psql "postgresql://postgres:[PASSWORD]@db.ambjsovdhizjxwhhnbtd.supabase.co:5432/postgres" \
  -f database/migrations/006_affiliate_links.sql

psql "postgresql://postgres:[PASSWORD]@db.ambjsovdhizjxwhhnbtd.supabase.co:5432/postgres" \
  -f database/migrations/007_wordpress_sites.sql
```

## Verify Migrations

After running migrations, verify tables exist:

```python
from supabase import create_client
import os

client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Check affiliate_links table
try:
    result = client.table("affiliate_links").select("*").limit(1).execute()
    print("✅ affiliate_links table exists")
except Exception as e:
    print(f"❌ affiliate_links table error: {e}")

# Check wordpress_sites table
try:
    result = client.table("wordpress_sites").select("*").limit(1).execute()
    print("✅ wordpress_sites table exists")
except Exception as e:
    print(f"❌ wordpress_sites table error: {e}")
```

## Migration Files

- **006_affiliate_links.sql**: Creates `affiliate_links` and `affiliate_link_insertions` tables
- **007_wordpress_sites.sql**: Creates `wordpress_sites` table

## Troubleshooting

### "relation does not exist" error
- Make sure migrations ran successfully
- Check Supabase Dashboard > Table Editor to see if tables exist

### "permission denied" error
- Ensure you're using the service role key (SUPABASE_SERVICE_KEY)
- Check RLS policies are set correctly

### "column does not exist" error
- Table might exist but schema is wrong
- Drop and recreate: `DROP TABLE IF EXISTS affiliate_links CASCADE;`
- Then run migration again

## Next Steps

After migrations:
1. Register WordPress site using `WordPressSiteRegistry`
2. Add affiliate links using `AffiliateLinkManager`
3. Run end-to-end test: `python3 test_agent_cms_e2e.py`

