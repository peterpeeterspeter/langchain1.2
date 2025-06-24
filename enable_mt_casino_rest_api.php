<?php
/**
 * 🎰 ENABLE MT CASINO REST API SUPPORT
 * 
 * This script enables REST API access for Coinflip theme's MT Casino post types.
 * Add this to your theme's functions.php or create as a custom plugin.
 * 
 * INSTRUCTIONS:
 * 1. Upload this as a plugin OR add to functions.php
 * 2. This will expose MT Casino post types via WordPress REST API
 * 3. Allows our Universal RAG Chain to publish directly as MT Casino posts
 */

// Enable REST API support for MT Casino post types
function enable_mt_casino_rest_api() {
    
    // List of MT Casino post types to enable
    $mt_post_types = [
        'mt_casinos',
        'mt_slots', 
        'mt_bonuses',
        'mt_bookmakers',
        'mt_reviews',
        'mt_knowledge',
        'mt_testimonials',
        'mt_clients',
        'mt_members'
    ];
    
    foreach ($mt_post_types as $post_type) {
        // Check if post type exists
        if (post_type_exists($post_type)) {
            // Enable REST API support
            global $wp_post_types;
            if (isset($wp_post_types[$post_type])) {
                $wp_post_types[$post_type]->show_in_rest = true;
                $wp_post_types[$post_type]->rest_base = $post_type;
                $wp_post_types[$post_type]->rest_controller_class = 'WP_REST_Posts_Controller';
                
                error_log("✅ Enabled REST API for: {$post_type}");
            }
        } else {
            error_log("❌ Post type not found: {$post_type}");
        }
    }
}

// Hook into WordPress initialization
add_action('init', 'enable_mt_casino_rest_api', 20);

// Also try after theme setup
add_action('after_setup_theme', 'enable_mt_casino_rest_api', 20);

// Add custom REST API fields for MT Casino metadata
function add_mt_casino_rest_fields() {
    
    // MT Casino specific meta fields
    $mt_casino_meta_fields = [
        'mt_casino_rating',
        'mt_casino_website', 
        'mt_casino_established',
        'mt_casino_license',
        'mt_casino_games_count',
        'mt_casino_bonus_rating',
        'mt_casino_payment_rating',
        'mt_casino_support_rating',
        'mt_casino_mobile_rating',
        'mt_casino_security_rating',
        'mt_casino_features',
        'mt_casino_currencies',
        'mt_casino_languages',
        'mt_casino_payment_methods',
        'mt_casino_pros',
        'mt_casino_cons'
    ];
    
    foreach ($mt_casino_meta_fields as $field) {
        register_rest_field('mt_casinos', $field, [
            'get_callback' => function($post) use ($field) {
                return get_post_meta($post['id'], $field, true);
            },
            'update_callback' => function($value, $post) use ($field) {
                return update_post_meta($post->ID, $field, $value);
            },
            'schema' => [
                'description' => "MT Casino field: {$field}",
                'type' => 'string',
                'context' => ['view', 'edit']
            ]
        ]);
    }
}

add_action('rest_api_init', 'add_mt_casino_rest_fields');

// Add support for custom taxonomies
function enable_mt_casino_taxonomies_rest() {
    
    $mt_taxonomies = [
        'mt_casino_category',
        'mt_casino_features', 
        'mt_casino_licenses',
        'mt_casino_payment_methods',
        'mt_casino_software'
    ];
    
    foreach ($mt_taxonomies as $taxonomy) {
        if (taxonomy_exists($taxonomy)) {
            global $wp_taxonomies;
            if (isset($wp_taxonomies[$taxonomy])) {
                $wp_taxonomies[$taxonomy]->show_in_rest = true;
                $wp_taxonomies[$taxonomy]->rest_base = $taxonomy;
                
                error_log("✅ Enabled REST API for taxonomy: {$taxonomy}");
            }
        }
    }
}

add_action('init', 'enable_mt_casino_taxonomies_rest', 21);

// Debug function to check what's available
function debug_mt_casino_post_types() {
    if (defined('WP_DEBUG') && WP_DEBUG) {
        error_log("🔍 DEBUG: Available post types:");
        $post_types = get_post_types(['public' => true], 'objects');
        foreach ($post_types as $post_type) {
            $rest_enabled = $post_type->show_in_rest ? '✅' : '❌';
            error_log("   {$rest_enabled} {$post_type->name} - REST: {$post_type->rest_base}");
        }
    }
}

add_action('wp_loaded', 'debug_mt_casino_post_types');

// Custom endpoint to verify MT Casino functionality
function register_mt_casino_test_endpoint() {
    register_rest_route('mt-casino/v1', '/test', [
        'methods' => 'GET',
        'callback' => function() {
            return [
                'status' => 'success',
                'message' => 'MT Casino REST API is working!',
                'available_post_types' => array_keys(get_post_types(['show_in_rest' => true])),
                'mt_casino_active' => post_type_exists('mt_casinos'),
                'timestamp' => current_time('mysql')
            ];
        },
        'permission_callback' => '__return_true'
    ]);
}

add_action('rest_api_init', 'register_mt_casino_test_endpoint');

?>

<!-- 
INSTALLATION INSTRUCTIONS:

METHOD 1: As WordPress Plugin
1. Save this file as: wp-content/plugins/mt-casino-rest-api/mt-casino-rest-api.php
2. Add plugin header at the top
3. Activate via WordPress admin

METHOD 2: Add to Theme Functions
1. Copy the PHP code (without <?php tags)
2. Add to your theme's functions.php file
3. Upload via FTP or WordPress admin

METHOD 3: Upload via WordPress Admin
1. Go to WordPress Admin > Appearance > Theme Editor
2. Select functions.php
3. Add this code at the bottom
4. Save changes

VERIFICATION:
After installation, test with:
curl -H "Authorization: Basic [credentials]" https://www.crashcasino.io/wp-json/wp/v2/mt_casinos

EXPECTED RESULT:
Should return MT Casino posts instead of 404 error
--> 