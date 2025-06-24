<?php
/**
 * Plugin Name: MT Casino REST API Enabler
 * Plugin URI: https://crashcasino.io
 * Description: Enables REST API access for Coinflip theme's MT Casino custom post types. Required for Universal RAG Chain integration.
 * Version: 1.0.0
 * Author: Universal RAG CMS
 * License: GPL v2 or later
 */

// Prevent direct access
if (!defined('ABSPATH')) {
    exit;
}

/**
 * 🎰 MT CASINO REST API ENABLER PLUGIN
 * 
 * This plugin enables REST API access for all Coinflip theme MT Casino post types.
 * Allows external applications like Universal RAG Chain to publish directly
 * to MT Casino, MT Slots, MT Bonuses, etc.
 */

class MTCasinoRestAPIEnabler {
    
    public function __construct() {
        add_action('init', [$this, 'enable_mt_casino_rest_api'], 20);
        add_action('after_setup_theme', [$this, 'enable_mt_casino_rest_api'], 20);
        add_action('rest_api_init', [$this, 'add_mt_casino_rest_fields']);
        add_action('init', [$this, 'enable_mt_casino_taxonomies_rest'], 21);
        add_action('wp_loaded', [$this, 'debug_mt_casino_post_types']);
        add_action('rest_api_init', [$this, 'register_mt_casino_test_endpoint']);
        
        // Add admin notice
        add_action('admin_notices', [$this, 'admin_notice']);
    }
    
    /**
     * Enable REST API support for MT Casino post types
     */
    public function enable_mt_casino_rest_api() {
        
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
                    
                    error_log("✅ MT Casino REST API: Enabled {$post_type}");
                }
            } else {
                error_log("❌ MT Casino REST API: Post type not found - {$post_type}");
            }
        }
    }
    
    /**
     * Add custom REST API fields for MT Casino metadata
     */
    public function add_mt_casino_rest_fields() {
        
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
    
    /**
     * Add support for custom taxonomies
     */
    public function enable_mt_casino_taxonomies_rest() {
        
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
                    
                    error_log("✅ MT Casino REST API: Enabled taxonomy {$taxonomy}");
                }
            }
        }
    }
    
    /**
     * Debug function to check what's available
     */
    public function debug_mt_casino_post_types() {
        if (defined('WP_DEBUG') && WP_DEBUG) {
            error_log("🔍 MT Casino REST API Debug: Available post types:");
            $post_types = get_post_types(['public' => true], 'objects');
            foreach ($post_types as $post_type) {
                $rest_enabled = $post_type->show_in_rest ? '✅' : '❌';
                error_log("   {$rest_enabled} {$post_type->name} - REST: {$post_type->rest_base}");
            }
        }
    }
    
    /**
     * Custom endpoint to verify MT Casino functionality
     */
    public function register_mt_casino_test_endpoint() {
        register_rest_route('mt-casino/v1', '/test', [
            'methods' => 'GET',
            'callback' => [$this, 'test_endpoint_callback'],
            'permission_callback' => '__return_true'
        ]);
    }
    
    public function test_endpoint_callback() {
        return [
            'status' => 'success',
            'message' => 'MT Casino REST API is working!',
            'plugin_version' => '1.0.0',
            'available_post_types' => array_keys(get_post_types(['show_in_rest' => true])),
            'mt_casino_active' => post_type_exists('mt_casinos'),
            'mt_slots_active' => post_type_exists('mt_slots'),
            'mt_bonuses_active' => post_type_exists('mt_bonuses'),
            'timestamp' => current_time('mysql')
        ];
    }
    
    /**
     * Admin notice to confirm plugin is working
     */
    public function admin_notice() {
        if (get_transient('mt_casino_rest_api_notice')) {
            return;
        }
        
        $mt_casino_exists = post_type_exists('mt_casinos');
        
        if ($mt_casino_exists) {
            $class = 'notice notice-success';
            $message = '🎰 MT Casino REST API Enabler: Successfully enabled REST API access for MT Casino post types!';
        } else {
            $class = 'notice notice-warning';
            $message = '⚠️ MT Casino REST API Enabler: MT Casino post types not found. Make sure Coinflip theme is active.';
        }
        
        echo "<div class='{$class}'><p>{$message}</p></div>";
        
        // Show notice only once
        set_transient('mt_casino_rest_api_notice', true, HOUR_IN_SECONDS);
    }
}

// Initialize the plugin
new MTCasinoRestAPIEnabler();

/**
 * Plugin activation hook
 */
function mt_casino_rest_api_activation() {
    // Clear permalinks to ensure REST routes are registered
    flush_rewrite_rules();
    
    // Log activation
    error_log('🎰 MT Casino REST API Enabler: Plugin activated');
}
register_activation_hook(__FILE__, 'mt_casino_rest_api_activation');

/**
 * Plugin deactivation hook
 */
function mt_casino_rest_api_deactivation() {
    // Clear permalinks
    flush_rewrite_rules();
    
    // Clean up transients
    delete_transient('mt_casino_rest_api_notice');
    
    // Log deactivation
    error_log('🎰 MT Casino REST API Enabler: Plugin deactivated');
}
register_deactivation_hook(__FILE__, 'mt_casino_rest_api_deactivation');

?> 