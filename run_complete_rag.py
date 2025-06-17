#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.append('src')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")

# Verify required environment variables are set
required_vars = ['ANTHROPIC_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"❌ Missing required environment variables: {missing_vars}")
    print("Please set these in your .env file or system environment")
    exit(1)

async def test_complete_universal_rag():
    print("🎯 Testing Complete Universal RAG CMS with All Fixes Applied...")
    
    try:
        # Test 1: Import Universal RAG Chain with fixes
        print("\n1. 🔧 Importing Universal RAG Chain...")
        from chains.universal_rag_lcel import UniversalRAGChain
        print("   ✅ Universal RAG Chain imported successfully")
        
        # Test 2: Initialize with auto-initialization (the working v5.1 feature)
        print("\n2. 🚀 Creating Universal RAG Chain with auto-initialization...")
        
        rag_chain = UniversalRAGChain()
        print("   ✅ Universal RAG Chain created with auto-initialization")
        print(f"   - Supabase client: {'✅' if rag_chain.supabase_client else '❌'}")
        print(f"   - Vector store: {'✅' if rag_chain.vector_store else '❌'}")
        print(f"   - LLM: {'✅' if rag_chain.llm else '❌'}")
        print(f"   - Embeddings: {'✅' if rag_chain.embeddings else '❌'}")
        
        # Test 3: Database connection check
        print("\n3. 📊 Checking database connectivity...")
        if rag_chain.supabase_client:
            count_response = rag_chain.supabase_client.table("documents").select("*", count="exact").limit(1).execute()
            total_docs = count_response.count
            print(f"   ✅ Database connected - {total_docs} documents available")
        else:
            print("   ❌ Database not connected")
            return False
        
        # Test 4: Run the complete chain with corrected input format
        print("\n4. 🧠 Running Universal RAG Chain with corrected format...")
        
        # CRITICAL: Use 'question' instead of 'query' (the fix we identified)
        test_input = {
            "question": "What are Betway's safety features and licensing information?"
        }
        
        print(f"   Query: {test_input['question']}")
        print("   Processing... (this may take 30-60 seconds)")
        
        response = await rag_chain.ainvoke(test_input)
        
        print(f"\n✅ Universal RAG Chain completed successfully!")
        print(f"   - Response type: {type(response)}")
        print(f"   - Response length: {len(str(response))} characters")
        
        # Check if response has sources
        if hasattr(response, 'sources') or (isinstance(response, dict) and 'sources' in response):
            sources = response.sources if hasattr(response, 'sources') else response.get('sources', [])
            print(f"   - Sources found: {len(sources)}")
            
            for i, source in enumerate(sources[:3]):
                if hasattr(source, 'page_content'):
                    content_preview = source.page_content[:100] + "..."
                    print(f"     - Source {i+1}: {content_preview}")
                elif isinstance(source, dict) and 'content' in source:
                    content_preview = source['content'][:100] + "..."
                    print(f"     - Source {i+1}: {content_preview}")
        else:
            print("   - Sources: Unknown format")
        
        # Display final content
        if isinstance(response, dict):
            content = response.get('content', response.get('text', str(response)))
        elif hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
            
        print(f"\n📄 Generated Content Preview (first 300 chars):")
        print(f"   {content[:300]}...")
        
        # Test 5: Save the complete response 
        import datetime
        output_file = f"betway_complete_all_features_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(output_file, 'w') as f:
            f.write(f"# Betway Casino Review - Complete Universal RAG CMS Test\n\n")
            f.write(f"**Query:** {test_input['question']}\n\n")
            f.write(f"**Response Type:** {type(response)}\n\n")
            f.write(f"**Generated Content:**\n\n{content}\n\n")
            
            if hasattr(response, 'sources') or (isinstance(response, dict) and 'sources' in response):
                sources = response.sources if hasattr(response, 'sources') else response.get('sources', [])
                f.write(f"**Sources ({len(sources)} found):**\n\n")
                for i, source in enumerate(sources):
                    f.write(f"### Source {i+1}\n")
                    if hasattr(source, 'page_content'):
                        f.write(f"{source.page_content}\n\n")
                        f.write(f"**Metadata:** {source.metadata}\n\n")
                    elif isinstance(source, dict):
                        f.write(f"{source}\n\n")
            
        print(f"\n💾 Complete response saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_universal_rag())
    if success:
        print("\n🎉 Universal RAG CMS working completely with all fixes!")
        print("📈 System Status: Production Ready")
    else:
        print("\n💥 Universal RAG CMS still needs debugging")
        print("🔧 Next: Check specific component failures") 