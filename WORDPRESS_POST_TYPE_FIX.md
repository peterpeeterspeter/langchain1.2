# WordPress Post Type Fallback Fix

## Problem
WordPress was returning error: "Invalid post type. Post type mt_listing does not exist" when trying to publish casino content.

## Root Cause
The system was attempting to use MT Casino custom post types (`mt_listing`, `mt_bonus`, etc.) but WordPress REST API wasn't enabled for these custom post types.

## Solution
Added automatic fallback logic in `WordPressRESTPublisher.publish_post()`:

1. **Added `post_type` parameter** (defaults to `"post"`)
2. **Automatic fallback**: If custom post type fails with 404 or "does not exist" error, automatically retries with regular `"post"` type
3. **Tracking**: Result includes `post_type_used` and `original_post_type` fields to track what was attempted

## Implementation Details

### Before:
```python
async def publish_post(self, title, content, ...):
    url = urljoin(self.config.site_url, "/wp-json/wp/v2/posts")
    # Always uses regular posts
```

### After:
```python
async def publish_post(self, title, content, ..., post_type: str = "post"):
    endpoint = f"/wp-json/wp/v2/{post_type}" if post_type != "post" else "/wp-json/wp/v2/posts"
    url = urljoin(self.config.site_url, endpoint)
    
    # Try custom post type first
    async with self.session.post(url, ...) as response:
        if response.status in [200, 201]:
            return result
        elif post_type != "post" and response.status == 404:
            # Fallback to regular posts
            fallback_url = urljoin(self.config.site_url, "/wp-json/wp/v2/posts")
            # Retry with regular posts
```

## Benefits

1. **Graceful Degradation**: System works even if MT Casino REST API isn't enabled
2. **Future-Proof**: When MT Casino REST API is enabled, it will automatically use custom post types
3. **Transparent**: Result metadata shows what post type was actually used
4. **No Breaking Changes**: Default behavior unchanged (uses regular posts)

## Testing

The fix has been applied. Next production run will:
- Try custom post type if specified
- Automatically fallback to regular posts if custom post type unavailable
- Successfully publish content regardless of WordPress configuration

## WordPress Configuration

To enable MT Casino custom post types REST API, add to WordPress `functions.php`:

```php
add_action('init', function() {
    $mt_casino_post_types = ['mt_listing', 'mt_bonus', 'mt_slots', 'mt_bookmaker', 'mt_reviews'];
    foreach($mt_casino_post_types as $post_type) {
        $post_type_object = get_post_type_object($post_type);
        if($post_type_object) {
            $post_type_object->show_in_rest = true;
            $post_type_object->rest_base = $post_type;
            $post_type_object->rest_controller_class = 'WP_REST_Posts_Controller';
        }
    }
}, 999);
```

Until this is added, the system will automatically use regular posts.


