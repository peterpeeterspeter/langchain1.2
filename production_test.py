#!/usr/bin/env python3
"""
Production Test: Universal RAG Chain with WordPress Publishing
Test the complete pipeline with all features including publishing to WordPress
"""

import sys
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append('src')

async def production_test():
    """Run the complete production pipeline with WordPress publishing"""
    
    print("🚀 PRODUCTION TEST: Universal RAG CMS")
    print("=" * 60)
    
    # Step 1: Verify all credentials
    print("Step 1: Verifying all credentials...")
    
    # DataForSEO credentials
    dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
    dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")
    print(f"📸 DataForSEO: {'✅ SET' if dataforseo_login and dataforseo_password else '❌ NOT SET'}")
    
    # WordPress credentials
    wp_url = os.getenv("WORDPRESS_URL", "")
    wp_username = os.getenv("WORDPRESS_USERNAME", "")
    wp_password = os.getenv("WORDPRESS_APP_PASSWORD", "")
    print(f"📝 WordPress: {'✅ SET' if wp_url and wp_username and wp_password else '❌ NOT SET'}")
    
    # API keys
    openai_key = os.getenv("OPENAI_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_ANON_KEY", "")
    
    print(f"🤖 OpenAI: {'✅ SET' if openai_key else '❌ NOT SET'}")
    print(f"🔍 Tavily: {'✅ SET' if tavily_key else '❌ NOT SET'}")
    print(f"🗄️ Supabase: {'✅ SET' if supabase_url and supabase_key else '❌ NOT SET'}")
    
    if not all([dataforseo_login, dataforseo_password, wp_url, wp_username, wp_password, openai_key]):
        print("\n❌ Missing required credentials! Check your .env file.")
        return
    
    print("\n✅ All credentials verified! Ready for production test.")
    
    # Step 2: Initialize Universal RAG Chain with ALL features
    print("\n🏗️ Step 2: Initializing Universal RAG CMS...")
    try:
        from chains.universal_rag_lcel import create_universal_rag_chain
        
        # Create production chain with ALL features enabled
        chain = create_universal_rag_chain(
            enable_dataforseo_images=True,           # ✅ DataForSEO image integration
            enable_wordpress_publishing=True,       # ✅ WordPress publishing
            enable_web_search=True,                  # ✅ Tavily web search
            enable_comprehensive_web_research=True, # ✅ 95-field casino analysis
            enable_hyperlink_generation=True,       # ✅ Authoritative links
            enable_template_system_v2=True,         # ✅ Advanced templates
            enable_enhanced_confidence=True,        # ✅ Enhanced confidence scoring
            enable_prompt_optimization=True,        # ✅ Prompt optimization
            enable_contextual_retrieval=True,       # ✅ Smart retrieval
            enable_fti_processing=True,             # ✅ FTI processing
            enable_security=True,                   # ✅ Security features
            enable_profiling=True,                  # ✅ Performance profiling
            enable_response_storage=True            # ✅ Response storage
        )
        
        print("✅ Universal RAG CMS initialized with ALL features!")
        
    except Exception as e:
        print(f"❌ Failed to initialize chain: {e}")
        return
    
    # Step 3: Generate and publish content
    print("\n🎯 Step 3: Generating and publishing content...")
    
    test_queries = [
        "Write a comprehensive review of LeoVegas Casino focusing on mobile gaming and live casino",
        # Add more queries if you want to test multiple publications
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        try:
            inputs = {'query': query}
            result = await chain.ainvoke(inputs)
            
            print(f"\n✅ Content generated successfully!")
            print(f"📄 Content length: {len(result.answer):,} characters")
            print(f"📊 Confidence score: {result.confidence_score:.3f}")
            print(f"🔗 Sources found: {len(result.sources)}")
            print(f"⏱️ Response time: {result.response_time:.2f}s")
            
            # Check what was accomplished
            metadata = result.metadata
            
            # Image integration
            images_found = metadata.get('images_found', 0)
            images_used = metadata.get('dataforseo_images_used', False)
            print(f"📸 Images found: {images_found}")
            print(f"📸 Images integrated: {'✅ YES' if images_used else '❌ NO'}")
            
            # WordPress publishing
            wp_published = metadata.get('wordpress_published', False)
            wp_url_published = metadata.get('wordpress_url', '')
            wp_post_id = metadata.get('wordpress_post_id', '')
            
            print(f"📝 WordPress published: {'✅ YES' if wp_published else '❌ NO'}")
            if wp_published:
                print(f"🌐 Published URL: {wp_url_published}")
                print(f"🆔 Post ID: {wp_post_id}")
            
            # Hyperlinks
            hyperlinks_added = metadata.get('hyperlinks_added', 0)
            print(f"🔗 Hyperlinks added: {hyperlinks_added}")
            
            # Research sources
            web_sources = metadata.get('web_sources_found', 0)
            comprehensive_sources = metadata.get('comprehensive_sources_found', 0)
            print(f"🔍 Web sources: {web_sources}")
            print(f"📊 Comprehensive research sources: {comprehensive_sources}")
            
            # Save local copy
            timestamp = metadata.get('timestamp', 'unknown')
            filename = f"leovegas_production_test_{timestamp.replace(':', '').replace(' ', '_')}.md"
            
            with open(filename, 'w') as f:
                f.write(f"# LeoVegas Casino Review (Production Test)\n\n")
                f.write(f"**Generated:** {timestamp}\n")
                f.write(f"**Confidence:** {result.confidence_score:.3f}\n")
                f.write(f"**WordPress Published:** {'YES' if wp_published else 'NO'}\n")
                if wp_published:
                    f.write(f"**WordPress URL:** {wp_url_published}\n")
                    f.write(f"**Post ID:** {wp_post_id}\n")
                f.write(f"**Images Found:** {images_found}\n")
                f.write(f"**Sources:** {len(result.sources)}\n\n")
                f.write("---\n\n")
                f.write(result.answer)
            
            print(f"💾 Local copy saved: {filename}")
            
            if wp_published:
                print(f"\n🎉 SUCCESS! Article published to WordPress:")
                print(f"🌐 Live URL: {wp_url_published}")
                print(f"📸 With {images_found} images")
                print(f"🔗 With {hyperlinks_added} authoritative links")
                print(f"📊 Based on {comprehensive_sources} research sources")
            else:
                print(f"\n⚠️ Content generated but not published to WordPress")
                print(f"Check WordPress credentials and permissions")
            
        except Exception as e:
            print(f"❌ Error processing query {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Production test completed!")

if __name__ == "__main__":
    asyncio.run(production_test()) 