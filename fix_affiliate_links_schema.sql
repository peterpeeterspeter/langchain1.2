-- Fix Affiliate Links Table Schema
-- Run this if you get "column category does not exist" error
-- This will add missing columns or recreate the table properly

-- First, check if table exists and what columns it has
-- (This is just for reference - run SELECT * FROM information_schema.columns WHERE table_name = 'affiliate_links';)

-- Option 1: Add missing columns if table exists but is incomplete
DO $$
BEGIN
    -- Add category column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'affiliate_links' AND column_name = 'category'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN category VARCHAR(50) NOT NULL DEFAULT 'other';
        CREATE INDEX IF NOT EXISTS idx_affiliate_links_category ON affiliate_links(category);
    END IF;
    
    -- Add status column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'affiliate_links' AND column_name = 'status'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'active';
        CREATE INDEX IF NOT EXISTS idx_affiliate_links_status ON affiliate_links(status);
    END IF;
    
    -- Add active column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'affiliate_links' AND column_name = 'active'
    ) THEN
        ALTER TABLE affiliate_links ADD COLUMN active BOOLEAN DEFAULT TRUE;
        CREATE INDEX IF NOT EXISTS idx_affiliate_links_active ON affiliate_links(active);
    END IF;
END $$;

-- Add constraints if they don't exist
DO $$
BEGIN
    -- Add category constraint
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'valid_category' AND table_name = 'affiliate_links'
    ) THEN
        ALTER TABLE affiliate_links ADD CONSTRAINT valid_category 
            CHECK (category IN ('casino', 'sportsbook', 'crypto_casino', 'bonus', 'payment', 'software', 'other'));
    END IF;
    
    -- Add status constraint
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'valid_status' AND table_name = 'affiliate_links'
    ) THEN
        ALTER TABLE affiliate_links ADD CONSTRAINT valid_status 
            CHECK (status IN ('active', 'inactive', 'pending', 'expired'));
    END IF;
END $$;

-- Verify table structure
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'affiliate_links'
ORDER BY ordinal_position;

