#!/usr/bin/env python3
"""
Complete Image Integration Test
Test Universal RAG Chain with DataForSEO image integration
"""

import sys
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append('src')

async def test_complete_image_integration():
    """Test the complete image integration pipeline"""
    
    print("🎯 COMPLETE IMAGE INTEGRATION TEST")
    print("=" * 60)
    
    # Step 1: Verify DataForSEO credentials
    print("Step 1: Verifying DataForSEO credentials...")
    dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
    dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")
    
    print(f"✅ DATAFORSEO_LOGIN: {'✅ SET' if dataforseo_login else '❌ NOT SET'}")
    print(f"✅ DATAFORSEO_PASSWORD: {'✅ SET' if dataforseo_password else '❌ NOT SET'}")
    
    if not dataforseo_login or not dataforseo_password:
        print("❌ DataForSEO credentials missing! Cannot test real image integration.")
        return
    
    # Step 2: Test DataForSEO service directly
    print("\n🔍 Step 2: Testing DataForSEO service...")
    try:
        from integrations.dataforseo_image_search import EnhancedDataForSEOImageSearch, DataForSEOConfig, ImageSearchRequest
        
        config = DataForSEOConfig()
        service = EnhancedDataForSEOImageSearch(config)
        
        request = ImageSearchRequest(
            keyword="Betway Casino",
            max_results=5,
            engine="google"
        )
        
        print(f"🔍 Searching for '{request.keyword}' images...")
        result = await service.search_images(request)
        
        print(f"✅ DataForSEO found {len(result.images)} images!")
        for i, img in enumerate(result.images[:3]):
            print(f"  {i+1}. {img.url[:50]}... - {img.alt_text}")
            
    except Exception as e:
        print(f"❌ DataForSEO service failed: {e}")
        print("💡 Continuing with Universal RAG Chain test...")
    
    # Step 3: Test Universal RAG Chain with full image integration
    print("\n🚀 Step 3: Testing Universal RAG Chain with image integration...")
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain with ALL features enabled including images
        chain = create_universal_rag_chain(
            enable_dataforseo_images=True,
            enable_wordpress_publishing=False,  # Disable publishing for this test
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_hyperlink_generation=True
        )
        
        print("🎯 Testing query: 'Write a comprehensive review of Betway Casino'")
        
        inputs = {'query': 'Write a comprehensive review of Betway Casino focusing on games and bonuses'}
        result = await chain.ainvoke(inputs)
        
        print(f"\n✅ Universal RAG Chain completed!")
        print(f"📄 Content length: {len(result.answer):,} characters")
        print(f"📊 Confidence score: {result.confidence_score:.3f}")
        print(f"🔗 Sources found: {len(result.sources)}")
        print(f"⏱️ Response time: {result.response_time:.2f}s")
        
        # Check for image integration
        image_indicators = [
            '<img ' in result.answer,
            '<figure' in result.answer,
            'Related Images' in result.answer,
            'src=' in result.answer
        ]
        
        images_embedded = any(image_indicators)
        print(f"📸 Images embedded: {'✅ YES' if images_embedded else '❌ NO'}")
        
        # Check metadata for image details
        metadata = result.metadata
        images_found = metadata.get('images_found', 0)
        images_used = metadata.get('dataforseo_images_used', False)
        hyperlinks_added = metadata.get('hyperlinks_added', 0)
        
        print(f"📊 Images found: {images_found}")
        print(f"📊 DataForSEO images used: {images_used}")
        print(f"🔗 Hyperlinks added: {hyperlinks_added}")
        
        # Show a sample of the content with images
        if images_embedded:
            print("\n📋 Content sample with images:")
            # Find the first image in content
            img_start = result.answer.find('<img') if '<img' in result.answer else result.answer.find('<figure')
            if img_start > -1:
                sample = result.answer[max(0, img_start-100):img_start+300]
                print(sample + "...")
        
        # Save full content for inspection
        with open('betway_review_with_images.md', 'w') as f:
            f.write(f"# Betway Casino Review (Generated with Images)\n\n")
            f.write(f"**Generated:** {result.metadata.get('timestamp', 'Unknown')}\n")
            f.write(f"**Confidence:** {result.confidence_score:.3f}\n")
            f.write(f"**Images Found:** {images_found}\n")
            f.write(f"**Sources:** {len(result.sources)}\n\n")
            f.write("---\n\n")
            f.write(result.answer)
            
        print(f"\n💾 Full review saved to: betway_review_with_images.md")
        
        if images_embedded:
            print("🎉 SUCCESS: Universal RAG Chain is successfully integrating images!")
        else:
            print("⚠️ WARNING: Images not found in final content. Check image embedding logic.")
            
    except Exception as e:
        print(f"❌ Universal RAG Chain test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_image_integration()) 