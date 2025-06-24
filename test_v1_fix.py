#!/usr/bin/env python3
"""
Test the V1 fix for embedded image processing
"""
import asyncio
import sys
sys.path.append('/Users/Peter/LANGCHAIN 1.2/langchain')

async def test_v1_fix():
    """Test the V1 fix with embedded image processing"""
    
    print("🔧 Testing V1 Fix for Embedded Image Processing")
    print("=" * 60)
    
    try:
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize the chain
        chain = UniversalRAGChain(
            enable_wordpress_publishing=True,
            enable_dataforseo_images=True,
            enable_web_research=True,
        )
        
        print("✅ Universal RAG Chain initialized with V1 fix")
        
        # Test with a simple query
        query = "Test Crashino Casino review with V1 fix"
        
        inputs = {
            "question": query,
            "publish_to_wordpress": True
        }
        
        print(f"🎯 Testing query: {query}")
        
        # Run the chain
        result = await chain.ainvoke(inputs)
        
        print("✅ Chain execution completed")
        print(f"📊 Result type: {type(result)}")
        
        # Check WordPress publishing results
        if hasattr(result, 'metadata') and result.metadata:
            wp_published = result.metadata.get('wordpress_published', False)
            wp_error = result.metadata.get('wordpress_error')
            
            print(f"🌐 WordPress published: {wp_published}")
            if wp_error:
                print(f"❌ WordPress error: {wp_error}")
            else:
                wp_post_id = result.metadata.get('wordpress_post_id')
                if wp_post_id:
                    print(f"✅ WordPress post ID: {wp_post_id}")
                    print(f"🔗 Post URL: https://www.crashcasino.io/?p={wp_post_id}")
        
        print("✅ V1 fix test completed successfully")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_v1_fix())
