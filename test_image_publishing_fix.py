#!/usr/bin/env python3
"""
Test the image publishing fix
"""
import asyncio
import os
from bs4 import BeautifulSoup

async def test_image_publishing():
    """Test the complete image publishing pipeline"""
    
    print("🧪 Testing Image Publishing Fix")
    print("=" * 50)
    
    # Sample content with embedded images (similar to what your RAG chain produces)
    test_content = '''
    <h1>Test Casino Review - Image Publishing Fix</h1>
    <p>This is a test article with embedded images to verify the fix.</p>
    
    <figure class="image-container">
        <img src="https://via.placeholder.com/800x400/FF6B6B/FFFFFF?text=Casino+Lobby" 
             alt="Casino Lobby" 
             title="Amazing Casino Lobby"
             loading="lazy"
             style="max-width: 100%; height: auto;">
        <figcaption>Casino Lobby View</figcaption>
    </figure>
    
    <p>More content here about the casino features...</p>
    
    <figure class="image-container">
        <img src="https://via.placeholder.com/600x300/4ECDC4/FFFFFF?text=Slot+Games" 
             alt="Slot Games" 
             title="Popular Slot Games">
        <figcaption>Top Slot Games</figcaption>
    </figure>
    
    <p>Final thoughts and conclusion...</p>
    '''
    
    print("🔍 Original content analysis:")
    soup = BeautifulSoup(test_content, 'html.parser')
    img_tags = soup.find_all('img')
    print(f"  📊 Found {len(img_tags)} images to process")
    
    for i, img in enumerate(img_tags, 1):
        print(f"  🖼️ Image {i}: {img.get('src')}")
        print(f"    Alt: {img.get('alt')}")
        print(f"    Title: {img.get('title')}")
    
    # Test the WordPress integration
    print("\n🔧 Testing WordPress Publisher Integration...")
    
    try:
        # Import the updated WordPress publisher
        from src.integrations.wordpress_publisher import WordPressRESTPublisher, WordPressConfig
        
        # Check if the new method exists
        if hasattr(WordPressRESTPublisher, 'process_embedded_images_in_content'):
            print("✅ process_embedded_images_in_content method found")
        else:
            print("❌ process_embedded_images_in_content method NOT found")
            return
        
        # Test with environment variables
        config = WordPressConfig(
            site_url=os.getenv("WORDPRESS_SITE_URL", "https://www.crashcasino.io"),
            username=os.getenv("WORDPRESS_USERNAME", ""),
            application_password=os.getenv("WORDPRESS_APP_PASSWORD", "")
        )
        
        if not config.username or not config.application_password:
            print("⚠️ WordPress credentials not found in environment")
            print("   Set WORDPRESS_USERNAME and WORDPRESS_APP_PASSWORD to test publishing")
            print("   Testing method signature only...")
            
            # Just test that the method exists and can be called
            publisher = WordPressRESTPublisher(config)
            print("✅ WordPressRESTPublisher initialized successfully")
            print("✅ Ready for image processing when credentials are provided")
            return
        
        print("✅ WordPress credentials found")
        
        # Test with actual publishing (if credentials available)
        async with WordPressRESTPublisher(config) as publisher:
            print("✅ WordPress connection established")
            
            # Test the embedded image processing
            print("\n🔄 Processing embedded images...")
            processed_content, media_ids = await publisher.process_embedded_images_in_content(test_content)
            
            print(f"✅ Processed {len(media_ids)} images")
            if media_ids:
                print(f"📸 Uploaded media IDs: {media_ids}")
            
            # Check the processed content
            processed_soup = BeautifulSoup(processed_content, 'html.parser')
            processed_imgs = processed_soup.find_all('img')
            
            print("\n🎯 After processing:")
            for i, img in enumerate(processed_imgs, 1):
                src = img.get('src', '')
                classes = img.get('class', [])
                data_id = img.get('data-id', '')
                
                print(f"  🖼️ Image {i}: {src}")
                print(f"    Classes: {classes}")
                if data_id:
                    print(f"    WordPress Media ID: {data_id}")
                
                # Check if URL was updated to WordPress
                if 'wp-content' in src or config.site_url in src:
                    print("    ✅ URL updated to WordPress-hosted")
                else:
                    print("    ⚠️ Still using external URL")
            
            # Publish a test post (as draft)
            print("\n📝 Publishing test post...")
            result = await publisher.publish_post(
                title="Image Publishing Fix Test",
                content=processed_content,
                status="draft",  # Use draft for testing
                meta_description="Testing the image publishing fix"
            )
            
            print(f"✅ Test post published: {result.get('link')}")
            print(f"📊 Embedded media count: {result.get('embedded_media_count', 0)}")
            print(f"🆔 Post ID: {result.get('id')}")
            
            if result.get('embedded_media_ids'):
                print(f"📸 Media IDs: {result.get('embedded_media_ids')}")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the WordPress publisher is properly installed")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

async def test_universal_rag_integration():
    """Test the Universal RAG Chain integration"""
    
    print("\n🔗 Testing Universal RAG Chain Integration")
    print("=" * 50)
    
    try:
        # Test if the Universal RAG Chain has been updated
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        print("✅ Universal RAG Chain imported successfully")
        
        # Check if the _optional_wordpress_publishing method mentions embedded_images
        import inspect
        source = inspect.getsource(UniversalRAGChain._optional_wordpress_publishing)
        
        if 'embedded_images' in source:
            print("✅ Universal RAG Chain updated with embedded_images support")
        else:
            print("⚠️ Universal RAG Chain may not have embedded_images support")
        
        if '_last_images' in source:
            print("✅ Universal RAG Chain passes _last_images to WordPress")
        else:
            print("⚠️ Universal RAG Chain may not pass _last_images")
        
        print("✅ Integration test complete")
        
    except ImportError as e:
        print(f"❌ Could not import Universal RAG Chain: {e}")
    except Exception as e:
        print(f"❌ Error testing integration: {e}")

async def main():
    """Run all tests"""
    await test_image_publishing()
    await test_universal_rag_integration()
    
    print("\n🎉 Image Publishing Fix Testing Complete!")
    print("\n📋 Next Steps:")
    print("1. Set WordPress credentials in environment variables")
    print("2. Run your Universal RAG Chain with publish_to_wordpress=True")
    print("3. Check published articles for properly hosted images")
    print("4. Verify images appear in WordPress Media Library")

if __name__ == "__main__":
    asyncio.run(main())
