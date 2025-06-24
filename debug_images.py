#!/usr/bin/env python3
"""
Debug Images in Universal RAG Chain
Check why images are not being embedded in the content
"""

import sys
import asyncio
import os
sys.path.append('src')

async def debug_image_system():
    """Debug the image system step by step"""
    
    print("🔍 DEBUG: Universal RAG Chain Image System")
    print("=" * 60)
    
    # Step 1: Check environment variables
    print("Step 1: Checking DataForSEO credentials...")
    dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
    dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")
    
    print(f"✅ DATAFORSEO_LOGIN: {'SET' if dataforseo_login else 'NOT SET'}")
    print(f"✅ DATAFORSEO_PASSWORD: {'SET' if dataforseo_password else 'NOT SET'}")
    
    if not dataforseo_login or not dataforseo_password:
        print("❌ DataForSEO credentials missing - this is why no images are found!")
        print("💡 Images require DataForSEO API credentials to work")
        
        # Test with fake images
        print("\n🧪 Testing with placeholder images...")
        fake_images = [
            {
                "url": "https://via.placeholder.com/400x300/0066cc/ffffff?text=Betway+Casino",
                "alt_text": "Betway Casino Official Logo",
                "title": "Betway Casino",
                "width": 400,
                "height": 300
            },
            {
                "url": "https://via.placeholder.com/600x400/ff6600/ffffff?text=Casino+Games",
                "alt_text": "Casino Games Collection",
                "title": "Popular Casino Games",
                "width": 600,
                "height": 400
            }
        ]
        
        # Test image embedding function
        from chains.universal_rag_lcel import UniversalRAGChain
        chain = UniversalRAGChain()
        
        test_content = """
# Betway Casino Review

Betway Casino is a popular online casino offering a wide range of games.

## Games Selection
- Slots
- Table Games  
- Live Casino

## Conclusion
Overall rating: 8.5/10
"""
        
        enhanced_content = chain._embed_images_in_content(test_content, fake_images)
        
        print(f"📝 Original content: {len(test_content)} characters")
        print(f"📝 Enhanced content: {len(enhanced_content)} characters")
        print(f"📸 Images embedded: {len(fake_images)}")
        
        if len(enhanced_content) > len(test_content):
            print("✅ Image embedding function works!")
            print("\n📋 Enhanced content preview:")
            print(enhanced_content[-500:])  # Show last 500 chars where images should be
        else:
            print("❌ Image embedding function failed")
        
        return
    
    # Step 2: Test DataForSEO service
    print("\n🧪 Testing DataForSEO service...")
    try:
        from integrations.dataforseo_image_search import EnhancedDataForSEOImageSearch, DataForSEOConfig, ImageSearchRequest
        
        config = DataForSEOConfig()
        service = EnhancedDataForSEOImageSearch(config)
        
        request = ImageSearchRequest(
            keyword="Betway Casino",
            max_results=3
        )
        
        print("🔍 Searching for 'Betway Casino' images...")
        result = await service.search_images(request)
        
        print(f"✅ Found {len(result.images)} images")
        for i, img in enumerate(result.images):
            print(f"  {i+1}. {img.url} - {img.alt_text}")
            
    except Exception as e:
        print(f"❌ DataForSEO service failed: {e}")
    
    # Step 3: Test Universal RAG Chain with images
    print("\n🧪 Testing Universal RAG Chain image integration...")
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        chain = create_universal_rag_chain(
            enable_dataforseo_images=True,
            enable_wordpress_publishing=False  # Focus on images only
        )
        
        # Manually set some test images
        chain._last_images = fake_images
        
        inputs = {'query': 'Write a short review of Betway Casino'}
        result = await chain.ainvoke(inputs)
        
        print(f"✅ Chain completed: {len(result.answer)} characters")
        
        # Check if images were embedded
        if "Related Images" in result.answer or "<img" in result.answer:
            print("✅ Images were embedded in content!")
        else:
            print("❌ Images were NOT embedded in content")
            
        # Check metadata
        images_found = result.metadata.get('images_found', 0)
        images_embedded = result.metadata.get('dataforseo_images_used', False)
        
        print(f"📊 Images found: {images_found}")
        print(f"📊 Images embedded: {images_embedded}")
        
    except Exception as e:
        print(f"❌ Chain test failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_image_system()) 