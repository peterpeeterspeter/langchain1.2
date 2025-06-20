#!/usr/bin/env python3
"""
Test script for Authoritative Hyperlink Generation integration
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_hyperlink_integration():
    """Test the hyperlink generation integration"""
    print("🔗 Testing Authoritative Hyperlink Generation Integration")
    print("=" * 60)
    
    try:
        # Test hyperlink engine standalone
        print("\n1. Testing standalone hyperlink engine...")
        
        from src.chains.authoritative_hyperlink_engine import (
            AuthoritativeHyperlinkEngine,
            LinkGenerationConfig
        )
        from src.chains.authority_links_config import AuthorityLinkPresets
        
        # Create hyperlink engine
        config = LinkGenerationConfig(**AuthorityLinkPresets.seo_optimized())
        engine = AuthoritativeHyperlinkEngine(config)
        
        # Test content
        test_content = """
        This casino is regulated by the UK Gambling Commission and offers responsible gambling tools. 
        Players can enjoy slots from Microgaming and NetEnt. The site uses SSL encryption for secure payments.
        """
        
        result = await engine.generate_hyperlinks(
            content=test_content,
            query="casino review safety"
        )
        
        print(f"✅ Links added: {result['links_added']}")
        print(f"✅ Enhanced content length: {len(result['enhanced_content'])} chars")
        if result['links_added'] > 0:
            print("✅ Hyperlink generation working!")
        
        # Test integration with Universal RAG Chain
        print("\n2. Testing Universal RAG Chain integration...")
        
        from src.chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain with hyperlink generation enabled
        chain = create_universal_rag_chain(
            model_name="gpt-4o-mini",
            enable_hyperlink_generation=True,
            enable_caching=False,  # Disable for testing
            enable_contextual_retrieval=False,  # Disable for testing
            enable_comprehensive_web_research=False  # Disable for testing
        )
        
        print("✅ Chain created with hyperlink generation enabled")
        
        # Check if hyperlink engine is initialized
        if hasattr(chain, 'hyperlink_engine') and chain.hyperlink_engine:
            print("✅ Hyperlink engine initialized in chain")
        else:
            print("❌ Hyperlink engine not initialized")
        
        print("\n3. Testing simple query...")
        
        # Simple test without full RAG (to avoid dependencies)
        simple_response = await chain._generate_with_all_features({
            "question": "Is this casino safe?",
            "enhanced_context": "This casino is licensed by UKGC and uses SSL encryption",
            "final_template": "",
            "query_analysis": None,
            "resources": {}
        })
        
        if "href=" in simple_response:
            print("✅ Hyperlinks found in generated content!")
            link_count = simple_response.count('<a href=')
            print(f"✅ {link_count} hyperlinks generated")
        else:
            print("⚠️ No hyperlinks found in content")
        
        print(f"\nGenerated content preview:")
        print(simple_response[:300] + "..." if len(simple_response) > 300 else simple_response)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hyperlink_integration()) 