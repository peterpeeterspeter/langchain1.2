#!/usr/bin/env python3
"""
Debug the WordPress publishing flow to find the comparison error
"""
import asyncio
import sys
import traceback
sys.path.append('/Users/Peter/LANGCHAIN 1.2/langchain')

async def debug_wordpress_flow():
    """Debug the exact WordPress publishing flow"""
    
    print("🔍 Debugging WordPress Publishing Flow")
    print("=" * 60)
    
    try:
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize the chain with WordPress publishing enabled
        chain = UniversalRAGChain(
            enable_wordpress_publishing=True,
            enable_dataforseo_images=True,
            enable_web_research=True,
            enable_hyperlink_generation=True,
        )
        
        print("✅ Universal RAG Chain initialized")
        
        # Test with a simple query that will trigger publishing
        query = "Test Crashino Casino review for debugging"
        
        # Create inputs that will trigger WordPress publishing
        inputs = {
            "question": query,
            "publish_to_wordpress": True  # This triggers the publishing
        }
        
        print(f"🎯 Testing query: {query}")
        print("📝 WordPress publishing enabled")
        
        # Run the chain
        result = await chain.ainvoke(inputs)
        
        print("✅ Chain execution completed")
        print(f"📊 Result type: {type(result)}")
        
        # Check if it's a Pydantic model
        if hasattr(result, 'model_dump'):
            print("📋 Result is a Pydantic model")
            result_dict = result.model_dump()
            print(f"📋 Model dump keys: {list(result_dict.keys())}")
        elif hasattr(result, '__dict__'):
            print("📋 Result has __dict__")
            print(f"📋 Dict keys: {list(result.__dict__.keys())}")
        else:
            print("📋 Result is a standard dict")
            print(f"📋 Keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
        
        # Check WordPress publishing results
        if hasattr(result, 'metadata') and result.metadata:
            wp_data = result.metadata.get('wordpress_published', False)
            print(f"🌐 WordPress published: {wp_data}")
        
        print("✅ Debug completed successfully")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        
        # Check if it's the comparison error
        if "'>=' not supported between instances of 'NoneType' and 'int'" in str(e):
            print("\n🎯 FOUND THE COMPARISON ERROR!")
            print("This means somewhere in the code, None is being compared to an integer")
            print("Let's check the WordPress publisher result handling...")

if __name__ == "__main__":
    asyncio.run(debug_wordpress_flow())
