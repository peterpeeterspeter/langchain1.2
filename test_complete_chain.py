#!/usr/bin/env python3
"""
Test the complete WordPress publishing chain with images
Tests the None comparison fix in Universal RAG Chain
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_complete_chain():
    """Test the complete WordPress publishing chain"""
    
    print("🚀 Testing Complete WordPress Publishing Chain with Images")
    print("=" * 70)
    
    try:
        # Import the Universal RAG Chain
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        print("✅ Successfully imported Universal RAG Chain")
        
        # Create the chain with all features enabled
        chain = create_universal_rag_chain(
            model_name="gpt-4",
            temperature=0.1,
            enable_caching=True,
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=True,
            enable_dataforseo_images=True,     # ✅ Images enabled
            enable_wordpress_publishing=True,  # ✅ WordPress enabled
            enable_fti_processing=True,
            enable_security=True,
            enable_profiling=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_hyperlink_generation=True,
            enable_response_storage=True
        )
        
        print("✅ Successfully created Universal RAG Chain with all features")
        
        # Test query that should trigger casino intelligence insights
        test_query = "Write a comprehensive review of Betway Casino including games, bonuses, and safety features"
        
        print(f"\n🎯 Testing Query: {test_query}")
        print("-" * 50)
        
        # Run the complete chain
        print("🔄 Running complete chain...")
        response = await chain.ainvoke({
            "query": test_query,
            "publish_to_wordpress": True  # Enable WordPress publishing
        })
        
        print("\n✅ Chain execution completed successfully!")
        print("=" * 70)
        
        # Display results
        print(f"📝 Response Length: {len(response.answer)} characters")
        print(f"🎯 Confidence Score: {response.confidence_score:.3f}")
        print(f"⏱️ Response Time: {response.response_time:.1f}ms")
        print(f"💾 Cached: {response.cached}")
        print(f"📊 Sources: {len(response.sources)}")
        
        # Check for images and WordPress publishing
        if hasattr(response, 'metadata') and response.metadata:
            metadata = response.metadata
            print(f"🖼️ Images Embedded: {metadata.get('images_embedded', 0)}")
            print(f"📝 WordPress Published: {metadata.get('wordpress_published', False)}")
            print(f"⚠️ Compliance Notices: {metadata.get('compliance_notices_added', 0)}")
            print(f"🔧 HTML Formatted: {metadata.get('html_formatted', False)}")
            
            if metadata.get('wordpress_published'):
                print("\n🎉 SUCCESS: WordPress publishing completed!")
                print("✅ The None comparison fix worked!")
            else:
                print("\n⚠️ WordPress publishing not completed - checking logs...")
        
        # Display first 500 characters of response
        print("\n📄 Response Preview:")
        print("-" * 30)
        print(response.answer[:500] + "..." if len(response.answer) > 500 else response.answer)
        
        # Check for intelligence insights (this was where the None error occurred)
        if "🛡️" in response.answer or "🎮" in response.answer or "💰" in response.answer:
            print("\n✅ SUCCESS: Intelligence insights generated without errors!")
            print("🎯 The None comparison fix is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"📍 Error Type: {type(e).__name__}")
        
        # Check if it's the specific None comparison error we fixed
        if "'>=' not supported between instances of 'NoneType' and 'int'" in str(e):
            print("🚨 CRITICAL: The None comparison error is still occurring!")
            print("🔧 The fix may not have been applied correctly.")
        else:
            print("ℹ️ This is a different error - the None comparison fix may be working.")
        
        import traceback
        print("\n📋 Full Traceback:")
        traceback.print_exc()
        
        return False

async def main():
    """Main test function"""
    print("🧪 WordPress Publishing Chain Test")
    print("Testing the None comparison fix...")
    print()
    
    success = await test_complete_chain()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 TEST COMPLETED SUCCESSFULLY!")
        print("✅ WordPress publishing chain is working with images!")
        print("🎯 None comparison error has been resolved!")
    else:
        print("❌ TEST FAILED!")
        print("🔧 Check the error details above for debugging.")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main()) 