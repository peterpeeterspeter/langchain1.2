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
        'mt_knowledgebase', // MT Knowledge Base (if exists)
        'mt_testimonials',  // MT Testimonials (if exists)
    ];
    
    foreach ($mt_post_types as $post_type) {
        if (post_type_exists($post_type)) {
            global $wp_post_types;
            $wp_post_types[$post_type]->show_in_rest = true;
            $wp_post_types[$post_type]->rest_base = $post_type;
            $wp_post_types[$post_type]->rest_controller_class = 'WP_REST_Posts_Controller';
            
            // Also enable REST for associated taxonomies
            $taxonomies = get_object_taxonomies($post_type);
            foreach ($taxonomies as $taxonomy) {
                if (taxonomy_exists($taxonomy)) {
                    global $wp_taxonomies;
                    $wp_taxonomies[$taxonomy]->show_in_rest = true;
                    $wp_taxonomies[$taxonomy]->rest_base = $taxonomy;
                }
            }
        }
    }
    
    // Log success for debugging
    if (function_exists('error_log')) {
        error_log('✅ MT Casino REST API enabled for post types: ' . implode(', ', $mt_post_types));
    }
}
add_action('init', 'enable_all_mt_casino_rest_api', 20);

/**
 * 🔧 ENABLE CUSTOM FIELDS IN REST API
 * Ensures MT Casino custom fields are accessible via REST API
 */
function enable_mt_casino_custom_fields_rest() {
    $meta_fields = [
        'casino_logo',
        'bonus_code', 
        'website_url',
        'rating',
        'license_info',
        'payment_methods',
        'game_providers',
        'bonus_terms',
        'min_deposit',
        'max_withdrawal'
    ];
    
    foreach ($meta_fields as $field) {
        register_rest_field(['mt_listing', 'mt_bonus', 'mt_slot', 'mt_bookmaker', 'mt_reviews'], 
            $field, [
                'get_callback' => function($post) use ($field) {
                    return get_post_meta($post['id'], $field, true);
                },
                'update_callback' => function($value, $post) use ($field) {
                    return update_post_meta($post->ID, $field, $value);
                },
                'schema' => [
                    'description' => 'MT Casino custom field: ' . $field,
                    'type' => 'string',
                    'context' => ['view', 'edit']
                ]
            ]
        );
    }
}
add_action('rest_api_init', 'enable_mt_casino_custom_fields_rest');
?> 