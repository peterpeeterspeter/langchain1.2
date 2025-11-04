-- Populate CyBet affiliate link for testing
-- Match the actual schema from 006_affiliate_links.sql

INSERT INTO affiliate_links (
    merchant, 
    product_name,
    affiliate_url, 
    category, 
    active, 
    status,
    description,
    tracking_id,
    tracking_template,
    keywords,
    max_uses_per_article,
    priority,
    commission_rate
) VALUES (
    'CyBet',
    'CyBet Casino',
    'https://cybetplay.com/tv7enau46',
    'casino',
    true,
    'active',
    'CyBet Casino affiliate link - Promo code: tv7enau46, Commission: Default 25%',
    'tv7enau46',
    '{url}',
    ARRAY['casino', 'gaming', 'betting', 'gambling', 'play', 'sign up', 'cybet', 'cybetplay'],
    3,
    10,
    25.0
)
ON CONFLICT DO NOTHING;

-- Verify insertion
SELECT id, merchant, product_name, category, active, status FROM affiliate_links WHERE merchant = 'CyBet';
