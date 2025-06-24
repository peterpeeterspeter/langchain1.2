<?php
/**
 * 🎰 ENABLE ALL MT CASINO POST TYPES REST API
 * Exposes all Coinflip theme MT Casino post types via REST API
 * 
 * CONFIRMED POST TYPE NAMES from WordPress admin URLs:
 * - mt_listing (MT Casinos) 
 * - mt_bonus (MT Bonuses)
 * - mt_slot (MT Slots - singular!)
 * - mt_bookmaker (MT Bookmakers)
 * - mt_reviews (MT Reviews)
 * 
 * ADD THIS CODE TO THE END OF YOUR functions.php FILE
 */
function enable_all_mt_casino_rest_api() {
    // ✅ CONFIRMED MT Casino post types from crashcasino.io admin
    $mt_post_types = [
        'mt_listing',      // MT Casinos (main casino listings)
        'mt_bonus',        // MT Bonuses  
        'mt_slot',         // MT Slots (singular, not plural!)
        'mt_bookmaker',    // MT Bookmakers
        'mt_reviews',      // MT Reviews
        // Optional: Add other MT post types if they exist
        'mt_games',        // Games (if exists)
        'mt_software',     // Software providers (if exists)
    ];
    
    foreach ($mt_post_types as $post_type) {
        // Check if post type exists before enabling REST API
        if (post_type_exists($post_type)) {
            // Get the post type object
            $post_type_object = get_post_type_object($post_type);
            
            if ($post_type_object) {
                // Enable REST API for this post type
                $post_type_object->show_in_rest = true;
                $post_type_object->rest_base = $post_type;
                $post_type_object->rest_controller_class = 'WP_REST_Posts_Controller';
                
                // Log success for debugging
                error_log("✅ MT Casino REST API enabled for: {$post_type}");
            }
        } else {
            // Log if post type doesn't exist
            error_log("⚠️ MT Casino post type not found: {$post_type}");
        }
    }
}

/**
 * 🔧 ENABLE MT CASINO CUSTOM FIELDS IN REST API
 * Exposes MT Casino custom fields (meta fields) via REST API
 */
function enable_mt_casino_meta_fields_rest_api() {
    $mt_post_types = ['mt_listing', 'mt_bonus', 'mt_slot', 'mt_bookmaker', 'mt_reviews'];
    
    // Common MT Casino meta fields based on Coinflip theme
    $meta_fields = [
        // Casino specific fields
        'mt_casino_logo',
        'mt_casino_website',
        'mt_casino_rating',
        'mt_casino_bonus_code',
        'mt_casino_license',
        'mt_casino_currencies',
        'mt_casino_languages',
        'mt_casino_countries',
        'mt_casino_software',
        'mt_casino_games',
        'mt_casino_payment_methods',
        'mt_casino_withdrawal_time',
        'mt_casino_min_deposit',
        'mt_casino_max_withdrawal',
        'mt_casino_customer_support',
        'mt_casino_live_chat',
        'mt_casino_mobile_app',
        'mt_casino_vip_program',
        'mt_casino_responsible_gambling',
        'mt_casino_security',
        
        // Bonus specific fields  
        'mt_bonus_amount',
        'mt_bonus_percentage',
        'mt_bonus_wagering',
        'mt_bonus_code',
        'mt_bonus_terms',
        'mt_bonus_expiry',
        'mt_bonus_type',
        
        // Slot specific fields
        'mt_slot_rtp',
        'mt_slot_volatility',
        'mt_slot_provider',
        'mt_slot_theme',
        'mt_slot_paylines',
        'mt_slot_min_bet',
        'mt_slot_max_bet',
        'mt_slot_jackpot',
        
        // Review specific fields
        'mt_review_overall_rating',
        'mt_review_pros',
        'mt_review_cons',
        'mt_review_final_verdict',
        'mt_review_date',
        'mt_review_author',
        
        // General meta fields
        'mt_featured',
        'mt_popular',
        'mt_new',
        'mt_recommended',
        'mt_affiliate_link',
        'mt_terms_conditions',
    ];
    
    foreach ($mt_post_types as $post_type) {
        if (post_type_exists($post_type)) {
            foreach ($meta_fields as $meta_key) {
                register_rest_field($post_type, $meta_key, [
                    'get_callback' => function($object) use ($meta_key) {
                        return get_post_meta($object['id'], $meta_key, true);
                    },
                    'update_callback' => function($value, $object) use ($meta_key) {
                        return update_post_meta($object->ID, $meta_key, $value);
                    },
                    'schema' => [
                        'description' => "MT Casino meta field: {$meta_key}",
                        'type' => 'string',
                        'context' => ['view', 'edit'],
                    ],
                ]);
            }
        }
    }
}

/**
 * 🚀 INITIALIZE MT CASINO REST API
 * Hook the functions to WordPress initialization
 */
function init_mt_casino_rest_api() {
    // Enable REST API for post types
    enable_all_mt_casino_rest_api();
    
    // Enable custom fields in REST API
    enable_mt_casino_meta_fields_rest_api();
    
    // Log initialization
    error_log("🎰 MT Casino REST API initialization complete");
}

// Hook to WordPress init action
add_action('init', 'init_mt_casino_rest_api', 20);

/**
 * 🔍 DEBUG FUNCTION: Check MT Casino REST API Status
 * Add ?mt_casino_debug=1 to any page to see REST API status
 */
function debug_mt_casino_rest_api() {
    if (isset($_GET['mt_casino_debug']) && current_user_can('administrator')) {
        echo "<h2>🎰 MT Casino REST API Debug Info</h2>";
        
        $mt_post_types = ['mt_listing', 'mt_bonus', 'mt_slot', 'mt_bookmaker', 'mt_reviews'];
        
        foreach ($mt_post_types as $post_type) {
            $post_type_object = get_post_type_object($post_type);
            
            echo "<h3>{$post_type}</h3>";
            if ($post_type_object) {
                echo "<p>✅ Post type exists</p>";
                echo "<p>REST Enabled: " . ($post_type_object->show_in_rest ? '✅ Yes' : '❌ No') . "</p>";
                echo "<p>REST Base: " . ($post_type_object->rest_base ?: 'Not set') . "</p>";
                echo "<p>REST Endpoint: /wp-json/wp/v2/{$post_type_object->rest_base}</p>";
                
                // Count posts
                $count = wp_count_posts($post_type);
                echo "<p>Total Posts: " . ($count->publish ?? 0) . "</p>";
            } else {
                echo "<p>❌ Post type does not exist</p>";
            }
            echo "<hr>";
        }
        
        echo "<h3>🔗 Test REST API Endpoints:</h3>";
        $site_url = home_url();
        foreach ($mt_post_types as $post_type) {
            echo "<p><a href='{$site_url}/wp-json/wp/v2/{$post_type}' target='_blank'>{$site_url}/wp-json/wp/v2/{$post_type}</a></p>";
        }
        
        die();
    }
}
add_action('wp', 'debug_mt_casino_rest_api');

/**
 * 🎯 ENSURE MT CASINO TAXONOMIES ARE REST ENABLED
 * Enable REST API for MT Casino taxonomies (categories, tags, etc.)
 */
function enable_mt_casino_taxonomies_rest_api() {
    $mt_taxonomies = [
        'mt_casino_category',
        'mt_casino_tag',
        'mt_casino_country',
        'mt_casino_software',
        'mt_casino_payment',
        'mt_casino_license',
        'mt_bonus_category',
        'mt_slot_category',
        'mt_slot_provider',
        'mt_bookmaker_category',
    ];
    
    foreach ($mt_taxonomies as $taxonomy) {
        if (taxonomy_exists($taxonomy)) {
            $taxonomy_object = get_taxonomy($taxonomy);
            if ($taxonomy_object) {
                $taxonomy_object->show_in_rest = true;
                $taxonomy_object->rest_base = $taxonomy;
                $taxonomy_object->rest_controller_class = 'WP_REST_Terms_Controller';
                error_log("✅ MT Casino taxonomy REST API enabled for: {$taxonomy}");
            }
        }
    }
}

// Hook taxonomies to init
add_action('init', 'enable_mt_casino_taxonomies_rest_api', 25);

?> 