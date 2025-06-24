#!/usr/bin/env python3
"""
Trace the exact location of the >= comparison error
"""
import asyncio
import sys
import traceback
import logging
sys.path.append('/Users/Peter/LANGCHAIN 1.2/langchain')

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

async def trace_exact_error():
    """Trace where the >= comparison error occurs"""
    
    print("🔍 Tracing Exact Location of >= Comparison Error")
    print("=" * 60)
    
    try:
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize the chain
        chain = UniversalRAGChain(
            enable_wordpress_publishing=True,
            enable_dataforseo_images=True,
            enable_web_research=False,  # Disable to simplify
            enable_comprehensive_web_research=False,  # Disable to simplify
        )
        
        print("✅ Universal RAG Chain initialized")
        
        # Test with minimal query
        query = "Simple test"
        
        inputs = {
            "question": query,
            "publish_to_wordpress": True
        }
        
        print(f"🎯 Testing query: {query}")
        
        # Wrap in try-catch to get exact traceback
        try:
            result = await chain.ainvoke(inputs)
            print("✅ Chain execution completed successfully")
            
        except TypeError as e:
            if "'>=' not supported between instances of 'NoneType' and 'int'" in str(e):
                print("🎯 FOUND THE EXACT ERROR!")
                print(f"Error: {e}")
                print("\n�� Full traceback:")
                traceback.print_exc()
                
                print("\n🔍 This error is happening in the chain execution, not WordPress publisher")
                print("Let's check where in the chain this comparison is happening...")
                
            else:
                print(f"❌ Different TypeError: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ Other error: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trace_exact_error())
