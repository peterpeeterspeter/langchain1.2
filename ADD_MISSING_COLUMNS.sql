-- Add Missing Columns to Existing affiliate_links Table
-- Safe migration that preserves existing data
-- Run this in Supabase SQL Editor

-- Add category column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'category'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN category VARCHAR(50) NOT NULL DEFAULT 'other';
        RAISE NOTICE 'Added category column';
    ELSE
        RAISE NOTICE 'category column already exists';
    END IF;
END $$;

-- Add status column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'status'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active';
        RAISE NOTICE 'Added status column';
    ELSE
        RAISE NOTICE 'status column already exists';
    END IF;
END $$;

-- Add active column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'active'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN active BOOLEAN DEFAULT TRUE;
        RAISE NOTICE 'Added active column';
    ELSE
        RAISE NOTICE 'active column already exists';
    END IF;
END $$;

-- Add other missing columns that might be needed
DO $$
BEGIN
    -- Add merchant if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'merchant'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN merchant VARCHAR(255);
        RAISE NOTICE 'Added merchant column';
    END IF;
    
    -- Add product_name if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'product_name'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN product_name VARCHAR(255);
        RAISE NOTICE 'Added product_name column';
    END IF;
    
    -- Add affiliate_url if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'affiliate_url'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN affiliate_url TEXT;
        RAISE NOTICE 'Added affiliate_url column';
    END IF;
    
    -- Add tracking_template if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'tracking_template'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN tracking_template TEXT DEFAULT '{url}?ref={tracking_id}';
        RAISE NOTICE 'Added tracking_template column';
    END IF;
    
    -- Add tracking_id if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'tracking_id'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN tracking_id VARCHAR(255);
        RAISE NOTICE 'Added tracking_id column';
    END IF;
    
    -- Add keywords if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'keywords'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN keywords TEXT[] DEFAULT '{}';
        RAISE NOTICE 'Added keywords column';
    END IF;
    
    -- Add commission_rate if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'commission_rate'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN commission_rate DECIMAL(5, 2) DEFAULT 0.0;
        RAISE NOTICE 'Added commission_rate column';
    END IF;
    
    -- Add optional fields
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'description'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN description TEXT;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'image_url'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN image_url TEXT;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'priority'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN priority INTEGER DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'max_uses_per_article'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN max_uses_per_article INTEGER DEFAULT 3;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'min_content_length'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN min_content_length INTEGER DEFAULT 500;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'usage_count'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN usage_count INTEGER DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'affiliate_links' 
        AND column_name = 'last_used'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN last_used TIMESTAMP WITH TIME ZONE;
    END IF;
END $$;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_affiliate_links_category ON affiliate_links(category);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_status ON affiliate_links(status);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_active ON affiliate_links(active);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_keywords ON affiliate_links USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_affiliate_links_merchant ON affiliate_links(merchant);

-- Add constraints if they don't exist (drop and recreate to ensure they're correct)
DO $$
BEGIN
    -- Drop existing constraint if it exists (to update it)
    ALTER TABLE affiliate_links DROP CONSTRAINT IF EXISTS valid_category;
    ALTER TABLE affiliate_links ADD CONSTRAINT valid_category 
        CHECK (category IN ('casino', 'sportsbook', 'crypto_casino', 'bonus', 'payment', 'software', 'other'));
    
    ALTER TABLE affiliate_links DROP CONSTRAINT IF EXISTS valid_status;
    ALTER TABLE affiliate_links ADD CONSTRAINT valid_status 
        CHECK (status IN ('active', 'inactive', 'pending', 'expired'));
END $$;

-- Verify all columns exist
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = 'affiliate_links'
ORDER BY ordinal_position;

