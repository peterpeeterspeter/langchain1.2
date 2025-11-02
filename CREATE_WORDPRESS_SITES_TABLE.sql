-- Create WordPress Sites Table
-- Run this in Supabase SQL Editor
-- Then run: python3 register_wordpress_site.py

CREATE TABLE IF NOT EXISTS wordpress_sites (
    site_id VARCHAR(255) PRIMARY KEY,
    site_name VARCHAR(255) NOT NULL,
    site_url TEXT NOT NULL,
    username VARCHAR(255) NOT NULL,
    application_password TEXT NOT NULL,
    
    -- Publishing defaults
    default_status VARCHAR(20) DEFAULT 'publish',
    default_author_id INTEGER DEFAULT 1,
    default_category_ids INTEGER[] DEFAULT '{}',
    default_tags TEXT[] DEFAULT '{}',
    
    -- Site-specific settings
    content_adaptation BOOLEAN DEFAULT FALSE,
    featured_image_required BOOLEAN DEFAULT FALSE,
    max_content_length INTEGER,
    
    -- Metadata
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_status CHECK (default_status IN ('draft', 'publish', 'private', 'pending'))
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_wordpress_sites_active ON wordpress_sites(active);
CREATE INDEX IF NOT EXISTS idx_wordpress_sites_site_id ON wordpress_sites(site_id);

-- Enable Row Level Security
ALTER TABLE wordpress_sites ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
DROP POLICY IF EXISTS "Service role full access on wordpress_sites" ON wordpress_sites;
CREATE POLICY "Service role full access on wordpress_sites"
    ON wordpress_sites FOR ALL
    USING (auth.role() = 'service_role');

-- Comments
COMMENT ON TABLE wordpress_sites IS 'Registry of WordPress sites for multi-site publishing';

-- Verify table was created
SELECT 
    'wordpress_sites' as table_name,
    COUNT(*) as row_count
FROM wordpress_sites;

