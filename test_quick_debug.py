#!/usr/bin/env python3
"""
Quick debug script to identify what's causing the hang
"""
import sys
import os
import signal
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def timeout_handler(signum, frame):
    print(f"\n🚨 TIMEOUT: Script stuck at {datetime.now()}")
    sys.exit(1)

async def test_components():
    print("🔍 Starting component testing...")
    
    try:
        print("1. Testing basic imports...")
        from chains.universal_rag_lcel import create_universal_rag_chain
        print("✅ Universal RAG import successful")
        
        print("2. Testing environment variables...")
        required_vars = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'SUPABASE_URL', 'WORDPRESS_SITE_URL']
        for var in required_vars:
            value = os.getenv(var)
            print(f"   {var}: {'✅ Set' if value else '❌ Missing'}")
        
        print("3. Testing chain creation...")
        # Chain creation is SYNCHRONOUS, not async
        signal.alarm(10)  # 10 second timeout
        chain = create_universal_rag_chain()  # Remove await
        signal.alarm(0)  # Cancel timeout
        print("✅ Chain creation successful")
        
        print("4. Testing simple query...")
        signal.alarm(15)  # 15 second timeout
        test_query = "What is Bitcoin?"
        response = await chain.ainvoke({'query': test_query})  # This is async
        signal.alarm(0)  # Cancel timeout
        print(f"✅ Simple query successful: {len(response.answer)} chars")
        
        print("5. Testing casino-specific query...")
        signal.alarm(20)  # 20 second timeout
        casino_query = "Review BitStarz Casino focusing on Bitcoin payments and provably fair games"
        casino_response = await chain.ainvoke({'query': casino_query})
        signal.alarm(0)  # Cancel timeout
        print(f"✅ Casino query successful: {len(casino_response.answer)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in component testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print(f"🚀 Quick Debug Test - {datetime.now()}")
    
    # Set overall timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)  # 90 second overall timeout
    
    try:
        # Run async test
        success = asyncio.run(test_components())
        
        if success:
            print("\n✅ All components working - ready for MT Casino publishing test")
        else:
            print("\n❌ Found component issues")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)  # Cancel any remaining timeout

if __name__ == "__main__":
    main() 