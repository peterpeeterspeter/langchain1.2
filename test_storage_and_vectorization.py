#!/usr/bin/env python3
"""
Test script to verify web search and response storage/vectorization
Tests that both web search results and AI responses are stored in Supabase
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables only")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src to Python path
sys.path.append('src')

async def test_storage_vectorization():
    """Test that web search results and responses are stored and vectorized"""
    
    print("="*80)
    print("📚 TESTING STORAGE & VECTORIZATION")
    print("="*80)
    
    try:
        # Test 1: Check Supabase connection
        print("\n1. 🔗 Testing Supabase connection...")
        
        try:
            from supabase import create_client
            
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            
            if not supabase_url or not supabase_key:
                print("❌ Missing Supabase credentials")
                return
            
            supabase = create_client(supabase_url, supabase_key)
            print("✅ Supabase client created successfully")
            
            # Check current document count
            result = supabase.table("documents").select("count", count="exact").execute()
            initial_count = result.count
            print(f"📊 Current documents in database: {initial_count}")
            
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return
            
        # Test 2: Create Universal RAG Chain with storage enabled
        print("\n2. 🏗️ Creating Universal RAG Chain with storage enabled...")
        
        try:
            from chains.universal_rag_lcel import create_universal_rag_chain
            
            chain = create_universal_rag_chain(
                enable_web_search=True,
                enable_response_storage=True,
                enable_contextual_retrieval=True,
                enable_dataforseo_images=True,
                supabase_client=supabase
            )
            
            print("✅ Universal RAG Chain created with storage enabled")
            print(f"📊 Active features: {chain._count_active_features()}/11")
            
        except Exception as e:
            print(f"❌ Chain creation failed: {e}")
            return
            
        # Test 3: Run a test query that should generate web search AND store results
        print("\n3. 🔍 Running test query to generate and store web search results...")
        
        test_query = "What are the latest Betway casino bonuses for 2024?"
        
        try:
            print(f"Query: {test_query}")
            start_time = datetime.now()
            
            response = await chain.ainvoke({"question": test_query})
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ Response generated in {duration:.2f} seconds")
            print(f"📊 Response length: {len(response.answer)} characters")
            print(f"📊 Confidence score: {response.confidence_score:.2f}")
            print(f"📊 Sources found: {len(response.sources)}")
            
            # Show source types
            source_types = {}
            for source in response.sources:
                source_type = source.get("source", "unknown")
                source_types[source_type] = source_types.get(source_type, 0) + 1
            
            print("📊 Source breakdown:")
            for source_type, count in source_types.items():
                print(f"  - {source_type}: {count}")
            
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return
            
        # Test 4: Check if new documents were stored
        print("\n4. 📚 Checking if new documents were stored in Supabase...")
        
        try:
            # Wait a moment for async storage to complete
            await asyncio.sleep(2)
            
            # Check new document count
            result = supabase.table("documents").select("count", count="exact").execute()
            final_count = result.count
            new_documents = final_count - initial_count
            
            print(f"📊 Documents after query: {final_count}")
            print(f"📊 New documents added: {new_documents}")
            
            if new_documents > 0:
                print("✅ Documents were successfully stored!")
                
                # Get recent documents to check types
                recent_docs = supabase.table("documents")\
                    .select("metadata")\
                    .order("id", desc=True)\
                    .limit(10)\
                    .execute()
                
                if recent_docs.data:
                    print("📊 Recent document types:")
                    for doc in recent_docs.data[:5]:
                        metadata = doc.get("metadata", {})
                        source = metadata.get("source", "unknown")
                        content_type = metadata.get("content_type", "unknown")
                        print(f"  - {source} ({content_type})")
                        
            else:
                print("⚠️ No new documents were stored")
                
        except Exception as e:
            print(f"❌ Document count check failed: {e}")
            
        # Test 5: Verify web search results were stored
        print("\n5. 🌐 Checking for stored web search results...")
        
        try:
            # Query for web search results
            web_search_docs = supabase.table("documents")\
                .select("metadata")\
                .eq("metadata->>source", "tavily_web_search")\
                .limit(5)\
                .execute()
            
            if web_search_docs.data:
                print(f"✅ Found {len(web_search_docs.data)} web search results in storage")
                for doc in web_search_docs.data:
                    metadata = doc.get("metadata", {})
                    title = metadata.get("title", "No title")
                    print(f"  - {title[:50]}...")
            else:
                print("⚠️ No web search results found in storage")
                
        except Exception as e:
            print(f"❌ Web search check failed: {e}")
            
        # Test 6: Verify RAG responses were stored
        print("\n6. 💬 Checking for stored RAG responses...")
        
        try:
            # Query for RAG responses
            rag_response_docs = supabase.table("documents")\
                .select("metadata")\
                .eq("metadata->>source", "rag_conversation")\
                .limit(3)\
                .execute()
            
            if rag_response_docs.data:
                print(f"✅ Found {len(rag_response_docs.data)} RAG responses in storage")
                for doc in rag_response_docs.data:
                    metadata = doc.get("metadata", {})
                    query = metadata.get("query", "No query")
                    confidence = metadata.get("confidence_score", 0)
                    print(f"  - Query: {query[:40]}... (confidence: {confidence})")
            else:
                print("⚠️ No RAG responses found in storage")
                
        except Exception as e:
            print(f"❌ RAG response check failed: {e}")
            
        print("\n" + "="*80)
        print("🎉 STORAGE & VECTORIZATION TEST COMPLETE")
        print("="*80)
        
        if new_documents > 0:
            print("✅ SUCCESS: New content was stored and vectorized!")
            print("📚 Your web search results and responses are now part of the knowledge base")
            print("🔄 Future queries can now retrieve this stored content for improved answers")
        else:
            print("⚠️ WARNING: No new content was stored")
            print("🔧 Check the logs above for any storage errors")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_storage_vectorization()) 