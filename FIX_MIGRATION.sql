-- Complete Fix for Affiliate Links Table
-- Run this if you get "column category does not exist" error
-- This will ensure all required columns exist

-- Drop and recreate table properly (WARNING: This will delete existing data)
-- Only use this if you don't have important data in the table

-- Step 1: Drop dependent table first
DROP TABLE IF EXISTS affiliate_link_insertions CASCADE;

-- Step 2: Drop main table
DROP TABLE IF EXISTS affiliate_links CASCADE;

-- Step 3: Recreate affiliate_links table with all columns
CREATE TABLE affiliate_links (
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
CREATE INDEX idx_affiliate_links_category ON affiliate_links(category);
CREATE INDEX idx_affiliate_links_status ON affiliate_links(status);
CREATE INDEX idx_affiliate_links_active ON affiliate_links(active);
CREATE INDEX idx_affiliate_links_keywords ON affiliate_links USING GIN(keywords);
CREATE INDEX idx_affiliate_links_merchant ON affiliate_links(merchant);

-- Recreate insertion tracking table
CREATE TABLE affiliate_link_insertions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    link_id UUID NOT NULL REFERENCES affiliate_links(id) ON DELETE CASCADE,
    article_id VARCHAR(255),
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    position INTEGER NOT NULL,
    anchor_text VARCHAR(255) NOT NULL,
    final_url TEXT NOT NULL,
    context TEXT,
    
    CONSTRAINT fk_affiliate_link FOREIGN KEY (link_id) REFERENCES affiliate_links(id)
);

CREATE INDEX idx_insertions_link_id ON affiliate_link_insertions(link_id);
CREATE INDEX idx_insertions_article_id ON affiliate_link_insertions(article_id);
CREATE INDEX idx_insertions_inserted_at ON affiliate_link_insertions(inserted_at);

-- Enable Row Level Security
ALTER TABLE affiliate_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE affiliate_link_insertions ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
CREATE POLICY "Service role full access on affiliate_links"
    ON affiliate_links FOR ALL
    USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access on affiliate_link_insertions"
    ON affiliate_link_insertions FOR ALL
    USING (auth.role() = 'service_role');

-- Comments
COMMENT ON TABLE affiliate_links IS 'Registry of affiliate links for CMS content';
COMMENT ON TABLE affiliate_link_insertions IS 'Tracking of affiliate link insertions in articles';

-- Verify columns exist
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'affiliate_links' 
ORDER BY ordinal_position;

