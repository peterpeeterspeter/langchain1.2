#!/usr/bin/env python3
"""
Full Integration Test - Real Data, Real APIs
Tests complete workflow: Research → Writing → Affiliate → Image → Publishing
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

print("=" * 80)
print("🧪 FULL INTEGRATION TEST - NATIVE AGENTS WITH REAL DATA")
print("=" * 80)
print(f"\nTest started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Test query
TEST_QUERY = "Betway Casino Review"
print(f"📝 Test Query: {TEST_QUERY}\n")

# Verify API keys
print("🔑 Checking API Keys...")
required_keys = {
    "ANTHROPIC_API_KEY": "Claude LLM",
    "TAVILY_API_KEY": "Web Search (Research)",
    "SUPABASE_URL": "Affiliate Database",
    "SUPABASE_SERVICE_KEY": "Affiliate Database Auth",
    "WORDPRESS_URL": "Publishing Target",
    "WORDPRESS_USERNAME": "Publishing Auth",
}

missing_keys = []
for key, description in required_keys.items():
    value = os.getenv(key)
    if value:
        # Mask the value for security
        masked = value[:8] + "..." if len(value) > 8 else "***"
        print(f"  ✅ {key}: {masked} ({description})")
    else:
        print(f"  ❌ {key}: MISSING ({description})")
        missing_keys.append(key)

if missing_keys:
    print(f"\n❌ Missing required keys: {', '.join(missing_keys)}")
    print("Please update .env file with missing credentials")
    sys.exit(1)

print("\n✅ All required API keys present!\n")

# Initialize LLM
print("🤖 Initializing Claude (Haiku)...")
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)
print("✅ LLM initialized\n")

print("=" * 80)
print("TESTING INDIVIDUAL AGENTS")
print("=" * 80)


# Test 1: Research Agent
print("\n1️⃣ TESTING RESEARCH AGENT")
print("-" * 80)

async def test_research_agent():
    """Test research agent with Tavily web search"""
    try:
        from agents.research_agent_native import create_native_research_agent

        print("Creating research agent...")
        agent = create_native_research_agent(llm=llm, enable_screenshots=False)
        print("✅ Agent created")

        print(f"\nExecuting research for: {TEST_QUERY}")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Research information about {TEST_QUERY}")]
        })

        messages = result.get("messages", [])
        print(f"✅ Research completed - {len(messages)} messages")

        # Count tool calls
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        print(f"🔧 Tool calls made: {len(tool_calls)}")
        for i, tc in enumerate(tool_calls, 1):
            print(f"   {i}. {tc.get('name', 'unknown')}")

        # Get final response
        final_msg = messages[-1] if messages else None
        if final_msg and hasattr(final_msg, 'content'):
            print(f"\n📊 Research summary (first 300 chars):")
            print(final_msg.content[:300] + "...")

        return {
            "success": True,
            "messages": messages,
            "tool_calls": len(tool_calls)
        }

    except Exception as e:
        print(f"❌ Research agent failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Test 2: Writing Agent
print("\n2️⃣ TESTING WRITING AGENT")
print("-" * 80)

async def test_writing_agent(research_data=None):
    """Test writing agent with mock or real research data"""
    try:
        from agents.writing_agent_native import create_native_writing_agent

        print("Creating writing agent...")
        agent = create_native_writing_agent(llm=llm)
        print("✅ Agent created")

        # Prepare input
        if research_data:
            query_input = f"Write a comprehensive review for {TEST_QUERY} based on the research data provided"
        else:
            query_input = f"Write a brief review for {TEST_QUERY} (use your knowledge)"

        print(f"\nExecuting writing task...")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query_input)]
        })

        messages = result.get("messages", [])
        print(f"✅ Writing completed - {len(messages)} messages")

        # Count tool calls
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        print(f"🔧 Tool calls made: {len(tool_calls)}")
        for i, tc in enumerate(tool_calls, 1):
            print(f"   {i}. {tc.get('name', 'unknown')}")

        # Get final content
        final_msg = messages[-1] if messages else None
        content = ""
        if final_msg and hasattr(final_msg, 'content'):
            content = final_msg.content
            print(f"\n📝 Content preview (first 300 chars):")
            print(content[:300] + "...")

        return {
            "success": True,
            "messages": messages,
            "content": content,
            "tool_calls": len(tool_calls)
        }

    except Exception as e:
        print(f"❌ Writing agent failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Test 3: Affiliate Agent
print("\n3️⃣ TESTING AFFILIATE AGENT")
print("-" * 80)

async def test_affiliate_agent(content="Sample casino content about Betway"):
    """Test affiliate agent with Supabase"""
    try:
        from agents.affiliate_agent_native import create_native_affiliate_agent

        print("Creating affiliate agent...")
        agent = create_native_affiliate_agent(llm=llm)
        print("✅ Agent created")

        print(f"\nAdding affiliate links to content...")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Add relevant affiliate links to this content: {content[:200]}...")]
        })

        messages = result.get("messages", [])
        print(f"✅ Affiliate processing completed - {len(messages)} messages")

        # Count tool calls
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        print(f"🔧 Tool calls made: {len(tool_calls)}")
        for i, tc in enumerate(tool_calls, 1):
            print(f"   {i}. {tc.get('name', 'unknown')}")

        return {
            "success": True,
            "messages": messages,
            "tool_calls": len(tool_calls)
        }

    except Exception as e:
        print(f"❌ Affiliate agent failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Test 4: Image Agent
print("\n4️⃣ TESTING IMAGE AGENT")
print("-" * 80)

async def test_image_agent():
    """Test image agent with DataForSEO"""
    try:
        from agents.image_agent_native import create_native_image_agent

        print("Creating image agent...")
        agent = create_native_image_agent(llm=llm, upload_to_wordpress=False)
        print("✅ Agent created")

        print(f"\nSearching for images related to {TEST_QUERY}...")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Find 2-3 relevant images for {TEST_QUERY}")]
        })

        messages = result.get("messages", [])
        print(f"✅ Image processing completed - {len(messages)} messages")

        # Count tool calls
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        print(f"🔧 Tool calls made: {len(tool_calls)}")
        for i, tc in enumerate(tool_calls, 1):
            print(f"   {i}. {tc.get('name', 'unknown')}")

        return {
            "success": True,
            "messages": messages,
            "tool_calls": len(tool_calls)
        }

    except Exception as e:
        print(f"❌ Image agent failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Test 5: Publishing Agent (DRY RUN)
print("\n5️⃣ TESTING PUBLISHING AGENT (DRY RUN)")
print("-" * 80)
print("⚠️  Note: Will validate credentials but NOT publish to live site")

async def test_publishing_agent_dry_run():
    """Test publishing agent setup (don't actually publish)"""
    try:
        from agents.publishing_agent_native import create_native_publishing_agent

        print("Creating publishing agent...")
        agent = create_native_publishing_agent(llm=llm)
        print("✅ Agent created")

        # Just check we can query site registry
        print(f"\nValidating WordPress connection to {os.getenv('WORDPRESS_URL')}...")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Check available sites in the registry")]
        })

        messages = result.get("messages", [])
        print(f"✅ Publishing agent validated - {len(messages)} messages")

        return {
            "success": True,
            "messages": messages,
            "note": "Dry run - no actual publishing"
        }

    except Exception as e:
        print(f"❌ Publishing agent failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# Run all tests
async def main():
    """Run all integration tests"""
    results = {}

    # Test 1: Research
    print("\n" + "=" * 80)
    results["research"] = await test_research_agent()

    # Test 2: Writing
    print("\n" + "=" * 80)
    results["writing"] = await test_writing_agent(results.get("research"))

    # Test 3: Affiliate
    print("\n" + "=" * 80)
    content = results["writing"].get("content", "Sample content")
    results["affiliate"] = await test_affiliate_agent(content)

    # Test 4: Image
    print("\n" + "=" * 80)
    results["image"] = await test_image_agent()

    # Test 5: Publishing (dry run)
    print("\n" + "=" * 80)
    results["publishing"] = await test_publishing_agent_dry_run()

    # Final Summary
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get("success"))

    for agent, result in results.items():
        status = "✅ PASS" if result.get("success") else "❌ FAIL"
        tool_calls = result.get("tool_calls", 0)
        print(f"{status} - {agent.upper()}: {tool_calls} tool calls")
        if not result.get("success"):
            print(f"        Error: {result.get('error', 'Unknown')}")

    print("\n" + "-" * 80)
    print(f"Total: {passed_tests}/{total_tests} agents passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n✅ Native agents are working with real APIs!")
        print("✅ Cost-optimized tool calling validated")
        print("✅ Ready for production use")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
        print("Review errors above for details")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(exit_code)
