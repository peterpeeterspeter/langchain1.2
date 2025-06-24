#!/usr/bin/env python3
"""
Debug the WordPress publisher to find the exact error
"""
import asyncio
import sys
import traceback
sys.path.append('/Users/Peter/LANGCHAIN 1.2/langchain')

async def debug_wordpress_publisher():
    """Debug the WordPress publisher directly"""
    
    print("🔧 Debugging WordPress Publisher Directly")
    print("=" * 60)
    
    try:
        from src.integrations.wordpress_publisher import WordPressRESTPublisher, WordPressConfig
        
        # Create config
        config = WordPressConfig()
        print(f"✅ WordPress config created: {config.site_url}")
        
        # Create publisher
        async with WordPressRESTPublisher(config) as publisher:
            print("✅ WordPress publisher created")
            
            # Test with simple content that has embedded images
            content = '''
            <h1>Test Article</h1>
            <p>This is a test article with an embedded image:</p>
            <figure class="image-container">
                <img src="https://games.bitcoin.com/images/uploads/gambling/roobet-logo.png" alt="Test Image" />
                <figcaption>Test image caption</figcaption>
            </figure>
            <p>More content here.</p>
            '''
            
            print("📝 Testing with content containing embedded images")
            
            # Test the new _process_embedded_images method directly
            processed_content, embedded_count = await publisher._process_embedded_images(content)
            
            print(f"✅ Embedded images processed: {embedded_count}")
            print(f"📝 Processed content length: {len(processed_content)}")
            
            # Now test full publishing
            result = await publisher.publish_post(
                title="Test Article with Embedded Images",
                content=content,
                status="draft"  # Use draft to avoid publishing live
            )
            
            print(f"✅ WordPress publishing successful!")
            print(f"📊 Result type: {type(result)}")
            print(f"📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            if isinstance(result, dict):
                print(f"📊 Post ID: {result.get('id')}")
                print(f"📊 Embedded media count: {result.get('embedded_media_count', 'NOT SET')}")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_wordpress_publisher())
