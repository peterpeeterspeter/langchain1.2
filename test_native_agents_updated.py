"""
Test script for updated native agents using create_react_agent
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_research_agent():
    """Test the native research agent"""
    print("=" * 80)
    print("TESTING NATIVE RESEARCH AGENT (create_react_agent)")
    print("=" * 80)

    try:
        from agents.research_agent_native import create_native_research_agent
        from langchain_core.messages import HumanMessage

        # Create agent
        print("\n✓ Creating native research agent...")
        agent = create_native_research_agent(verbose=True)
        print("✓ Agent created successfully")

        # Test with a simple query
        print("\n✓ Testing with query: 'What is Betway Casino?'")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="What is Betway Casino?")]
        })

        # Check result structure
        print(f"\n✓ Agent returned {len(result.get('messages', []))} messages")

        # Extract tool calls
        messages = result.get('messages', [])
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        print(f"✓ Agent made {len(tool_calls)} tool calls")
        for i, tc in enumerate(tool_calls, 1):
            print(f"  {i}. {tc.get('name', 'unknown')}")

        # Get final response
        final_message = messages[-1] if messages else None
        if final_message:
            output = final_message.content if hasattr(final_message, 'content') else str(final_message)
            print(f"\n✓ Final response length: {len(output)} characters")
            print(f"✓ Preview: {output[:200]}...")

        print("\n" + "=" * 80)
        print("✅ RESEARCH AGENT TEST PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n🧪 Testing Updated Native Agents\n")

    success = await test_research_agent()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
