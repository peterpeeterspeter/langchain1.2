#!/usr/bin/env python3
"""
Test the image publishing fix with a real article generation
"""
import asyncio
import os
import sys
sys.path.append('/Users/Peter/LANGCHAIN 1.2/langchain')

async def test_crashino_with_images():
    """Generate a Crashino article with the image fix"""
    
    print("🎰 Testing Crashino Article Generation with Image Fix")
    print("=" * 60)
    
    try:
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize the chain
        chain = UniversalRAGChain(
            enable_wordpress_publishing=True,  # Enable WordPress publishing
            enable_dataforseo_images=True,     # Enable DataForSEO images
            enable_web_research=True,          # Enable web research
            enable_hyperlink_generation=True, # Enable hyperlinks
        )
        
        print("✅ Universal RAG Chain initialized")
        
        # Test query for Crashino Casino
        query = "Write a comprehensive review of Crashino Casino covering games, bonuses, licensing, and user experience"
        
        print(f"🔍 Query: {query}")
        print("\n🚀 Generating article with images...")
        
        # Generate the article
        result = await chain.ainvoke({
            "query": query,
            "publish_to_wordpress": True,  # Ensure WordPress publishing
        })
        
        print("\n📊 Results:")
        print(f"✅ Content Length: {len(result.get('final_content', ''))} characters")
        print(f"✅ Confidence Score: {result.get('confidence_score', 'N/A')}")
        
        # Check for images
        if hasattr(chain, '_last_images') and chain._last_images:
            print(f"🖼️ DataForSEO Images Found: {len(chain._last_images)}")
            for i, img in enumerate(chain._last_images[:3], 1):
                print(f"  📸 Image {i}: {img.get('url', 'No URL')[:80]}...")
        else:
            print("⚠️ No DataForSEO images found")
        
        # Check WordPress publishing
        wordpress_info = result.get('wordpress_metadata', {})
        if wordpress_info.get('published'):
            print(f"\n📝 WordPress Publishing:")
            print(f"  ✅ Published: {wordpress_info.get('published')}")
            print(f"  🆔 Post ID: {wordpress_info.get('post_id')}")
            print(f"  🔗 URL: {wordpress_info.get('post_url')}")
            print(f"  🖼️ Embedded Images: {wordpress_info.get('embedded_images_processed', 0)}")
            
            if wordpress_info.get('embedded_media_ids'):
                print(f"  📸 WordPress Media IDs: {wordpress_info.get('embedded_media_ids')}")
        else:
            print("⚠️ WordPress publishing failed or disabled")
        
        print(f"\n🎉 Test completed successfully!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_crashino_with_images())
