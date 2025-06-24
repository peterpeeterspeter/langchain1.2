#!/usr/bin/env python3
"""
Crashino Casino Review Test
Simple test script for the Universal RAG Chain with Crashino Casino
"""

import sys
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append('src')

async def crashino_test():
    """Test the Universal RAG Chain with Crashino Casino review"""
    
    print("🎰 CRASHINO CASINO REVIEW TEST")
    print("=" * 60)
    
    # Step 1: Verify credentials
    print("Step 1: Verifying credentials...")
    
    # Check key credentials
    openai_key = os.getenv("OPENAI_API_KEY", "")
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
    dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
    dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")
    
    print(f"✅ OPENAI_API_KEY: {'✅ SET' if openai_key else '❌ NOT SET'}")
    print(f"✅ SUPABASE_URL: {'✅ SET' if supabase_url else '❌ NOT SET'}")
    print(f"✅ SUPABASE_ANON_KEY: {'✅ SET' if supabase_key else '❌ NOT SET'}")
    print(f"✅ DATAFORSEO_LOGIN: {'✅ SET' if dataforseo_login else '❌ NOT SET'}")
    print(f"✅ DATAFORSEO_PASSWORD: {'✅ SET' if dataforseo_password else '❌ NOT SET'}")
    
    # Step 2: Import and create chain with safe settings
    print(f"\nStep 2: Creating Universal RAG Chain...")
    
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create chain with conservative settings to avoid issues
        chain = create_universal_rag_chain(
            model_name="gpt-4.1-mini",
            temperature=0.1,
            enable_caching=True,
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=False,  # Disable to avoid schema issues
            enable_dataforseo_images=True,
            enable_wordpress_publishing=False,  # Disable for this test
            enable_fti_processing=False,  # Disable to avoid issues
            enable_security=True,
            enable_profiling=True,
            enable_web_search=True,
            enable_comprehensive_web_research=False,  # Disable to avoid schema issues
            enable_hyperlink_generation=True,
            enable_response_storage=True
        )
        
        print("✅ Universal RAG Chain created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating chain: {e}")
        return
    
    # Step 3: Run the Crashino review
    print(f"\nStep 3: Generating Crashino Casino review...")
    
    query = "Write a comprehensive review of Crashino Casino focusing on games and bonuses"
    
    try:
        print(f"🔍 Query: {query}")
        print(f"⏱️ Starting generation...")
        
        response = await chain.ainvoke(query)
        
        print(f"\n🎉 SUCCESS! Generated Crashino review")
        print(f"📊 Content length: {len(response.answer):,} characters")
        print(f"🎯 Confidence score: {response.confidence_score:.3f}")
        print(f"⏱️ Response time: {response.response_time:.1f}ms")
        print(f"💾 Cached: {response.cached}")
        print(f"📚 Sources found: {len(response.sources)} sources")
        
        # Step 4: Save the review
        filename = "crashino_review_complete.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.answer)
        
        print(f"\n💾 Review saved to: {filename}")
        
        # Show first 500 characters as preview
        print(f"\n📄 PREVIEW:")
        print("=" * 60)
        preview = response.answer[:500] + "..." if len(response.answer) > 500 else response.answer
        print(preview)
        print("=" * 60)
        
        # Show sources
        if response.sources:
            print(f"\n📚 SOURCES:")
            for i, source in enumerate(response.sources[:5], 1):
                url = source.get('url', 'N/A')
                title = source.get('title', 'N/A')
                print(f"  {i}. {title}")
                print(f"     URL: {url}")
                
        # Show query analysis if available
        if response.query_analysis:
            print(f"\n🔍 QUERY ANALYSIS:")
            print(f"  Query Type: {response.query_analysis.get('query_type', 'N/A')}")
            print(f"  Expertise Level: {response.query_analysis.get('expertise_level', 'N/A')}")
            print(f"  Response Format: {response.query_analysis.get('response_format', 'N/A')}")
        
        print(f"\n🎉 CRASHINO REVIEW GENERATION COMPLETE!")
        
    except Exception as e:
        print(f"❌ Error generating review: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(crashino_test()) 