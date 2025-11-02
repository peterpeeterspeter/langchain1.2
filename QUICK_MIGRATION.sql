-- Combined Migration Script for Agent-Based CMS
-- Run this in Supabase SQL Editor to create all required tables
-- Created: 2025-11-02

-- ============================================================================
-- Migration 006: Affiliate Links Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS affiliate_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant VARCHAR(255) NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    affiliate_url TEXT NOT NULL,
    commission_rate DECIMAL(5, 2) DEFAULT 0.0,
    keywords TEXT[] DEFAULT '{}',
    category VARCHAR(50) NOT NULL DEFAULT 'other',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    tracking_template TEXT NOT NULL DEFAULT '{url}?ref={tracking_id}',
    tracking_id VARCHAR(255) NOT NULL,
    
    -- Optional fields
    description TEXT,
    image_url TEXT,
    priority INTEGER DEFAULT 0,
    max_uses_per_article INTEGER DEFAULT 3,
    min_content_length INTEGER DEFAULT 500,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    
    -- Status
    active BOOLEAN DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT valid_commission_rate CHECK (commission_rate >= 0 AND commission_rate <= 100),
    CONSTRAINT valid_priority CHECK (priority >= 0),
    CONSTRAINT valid_max_uses CHECK (max_uses_per_article > 0),
    CONSTRAINT valid_category CHECK (category IN ('casino', 'sportsbook', 'crypto_casino', 'bonus', 'payment', 'software', 'other')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'pending', 'expired'))
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_affiliate_links_category ON affiliate_links(category);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_status ON affiliate_links(status);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_active ON affiliate_links(active);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_keywords ON affiliate_links USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_merchant ON affiliate_links(merchant);

-- Insertion tracking table
CREATE TABLE IF NOT EXISTS affiliate_link_insertions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    link_id UUID NOT NULL REFERENCES affiliate_links(id) ON DELETE CASCADE,
    article_id VARCHAR(255),
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    position INTEGER NOT NULL,
    anchor_text VARCHAR(255) NOT NULL,
    final_url TEXT NOT NULL,
    context TEXT,
    
    -- Indexes
    CONSTRAINT fk_affiliate_link FOREIGN KEY (link_id) REFERENCES affiliate_links(id)
);

CREATE INDEX IF NOT EXISTS idx_insertions_link_id ON affiliate_link_insertions(link_id);
CREATE INDEX IF NOT EXISTS idx_insertions_article_id ON affiliate_link_insertions(article_id);
CREATE INDEX IF NOT EXISTS idx_insertions_inserted_at ON affiliate_link_insertions(inserted_at);

-- Enable Row Level Security
ALTER TABLE affiliate_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE affiliate_link_insertions ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
DROP POLICY IF EXISTS "Service role full access on affiliate_links" ON affiliate_links;
CREATE POLICY "Service role full access on affiliate_links"
    ON affiliate_links FOR ALL
    USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "Service role full access on affiliate_link_insertions" ON affiliate_link_insertions;
CREATE POLICY "Service role full access on affiliate_link_insertions"
    ON affiliate_link_insertions FOR ALL
    USING (auth.role() = 'service_role');

-- Comments
COMMENT ON TABLE affiliate_links IS 'Registry of affiliate links for CMS content';
COMMENT ON TABLE affiliate_link_insertions IS 'Tracking of affiliate link insertions in articles';

-- ============================================================================
-- Migration 007: WordPress Sites Registry
-- ============================================================================

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

-- ============================================================================
-- Verification
-- ============================================================================

-- Verify tables were created
SELECT 
    'affiliate_links' as table_name,
    COUNT(*) as row_count
FROM affiliate_links
UNION ALL
SELECT 
    'wordpress_sites' as table_name,
    COUNT(*) as row_count
FROM wordpress_sites;

