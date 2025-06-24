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
    ];

    foreach ($mt_post_types as $post_type) {
        // Check if post type exists before modifying
        if (post_type_exists($post_type)) {
            // Get the post type object
            $post_type_object = get_post_type_object($post_type);
            
            if ($post_type_object) {
                // Enable REST API support
                $post_type_object->show_in_rest = true;
                $post_type_object->rest_base = $post_type;
                $post_type_object->rest_controller_class = 'WP_REST_Posts_Controller';
                
                // Enable public queries
                $post_type_object->publicly_queryable = true;
                $post_type_object->public = true;
                
                error_log("✅ MT Casino REST API enabled for: $post_type");
            }
        } else {
            error_log("❌ MT Casino post type not found: $post_type");
        }
    }
    
    // Also enable custom fields in REST API
    add_filter('rest_prepare_' . 'mt_listing', 'add_mt_casino_custom_fields_to_rest', 10, 3);
    add_filter('rest_prepare_' . 'mt_bonus', 'add_mt_casino_custom_fields_to_rest', 10, 3);
    add_filter('rest_prepare_' . 'mt_slot', 'add_mt_casino_custom_fields_to_rest', 10, 3);
    add_filter('rest_prepare_' . 'mt_bookmaker', 'add_mt_casino_custom_fields_to_rest', 10, 3);
    add_filter('rest_prepare_' . 'mt_reviews', 'add_mt_casino_custom_fields_to_rest', 10, 3);
}

/**
 * Add custom fields to REST API response for MT Casino post types
 */
function add_mt_casino_custom_fields_to_rest($response, $post, $request) {
    // Get all custom fields for this post
    $custom_fields = get_post_meta($post->ID);
    
    // Clean up the meta values (remove arrays for single values)
    $cleaned_fields = [];
    foreach ($custom_fields as $key => $value) {
        if (is_array($value) && count($value) === 1) {
            $cleaned_fields[$key] = $value[0];
        } else {
            $cleaned_fields[$key] = $value;
        }
    }
    
    // Add custom fields to the response
    $response->data['mt_casino_meta'] = $cleaned_fields;
    
    return $response;
}

// Hook the function to run after theme setup
add_action('init', 'enable_all_mt_casino_rest_api', 20);

/**
 * Enable taxonomies for MT Casino post types in REST API
 */
function enable_mt_casino_taxonomies_rest_api() {
    $taxonomies = [
        'mt_casino_categories',
        'mt_casino_software',
        'mt_casino_licenses',
        'mt_casino_countries',
        'mt_casino_features',
    ];
    
    foreach ($taxonomies as $taxonomy) {
        if (taxonomy_exists($taxonomy)) {
            $taxonomy_object = get_taxonomy($taxonomy);
            if ($taxonomy_object) {
                $taxonomy_object->show_in_rest = true;
                $taxonomy_object->rest_base = $taxonomy;
                error_log("✅ MT Casino taxonomy REST API enabled: $taxonomy");
            }
        }
    }
}

// Enable taxonomies in REST API
add_action('init', 'enable_mt_casino_taxonomies_rest_api', 25); 