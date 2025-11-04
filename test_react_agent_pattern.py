"""
Standalone test for create_react_agent pattern
Tests the agent architecture without requiring all project dependencies
"""
import asyncio
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()


# Mock tools that don't require external dependencies
@tool
def mock_web_search(query: str) -> dict:
    """Search the web for information"""
    return {
        "results": [
            {"title": "Betway Casino", "url": "https://betway.com", "snippet": "Leading online casino"}
        ]
    }


@tool
def mock_casino_intel(casino_name: str) -> dict:
    """Extract casino intelligence data"""
    return {
        "name": casino_name,
        "license": "Malta Gaming Authority",
        "games": 500,
        "bonuses": ["Welcome bonus", "Free spins"]
    }


@tool
def mock_screenshot(url: str) -> dict:
    """Take screenshot of casino website"""
    return {
        "success": True,
        "url": url,
        "path": f"/screenshots/{url.replace('https://', '')}.png"
    }


async def test_react_agent():
    """Test create_react_agent pattern"""
    print("=" * 80)
    print("TESTING create_react_agent PATTERN")
    print("=" * 80)

    # Create tools
    tools = [mock_web_search, mock_casino_intel, mock_screenshot]
    print(f"\n✓ Created {len(tools)} mock tools")

    # Create LLM (will use API if available, or skip if not)
    try:
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)
        print("✓ Created Anthropic LLM (claude-3-haiku)")
    except Exception as e:
        print(f"⚠️  Could not create LLM: {e}")
        print("⚠️  Test requires ANTHROPIC_API_KEY to be set")
        return False

    # Create system message
    system_message = SystemMessage(content="""You are a research agent specialized in casino information.

Available tools:
- mock_web_search: Search for casino information
- mock_casino_intel: Extract detailed casino data
- mock_screenshot: Capture casino website screenshots

Strategy:
1. Use web search for overview
2. Extract detailed intelligence
3. Capture screenshots if needed

Be thorough but efficient - only call necessary tools.""")

    print("✓ Created system message")

    # Create native agent using create_react_agent
    try:
        agent = create_react_agent(
            llm,
            tools,
            prompt=system_message
        )
        print("✓ Created native agent with create_react_agent()")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test agent execution
    try:
        print("\n✓ Testing agent with query: 'Research Betway Casino'")
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="Research Betway Casino")]
        })
        print("✓ Agent execution completed")
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze results
    messages = result.get("messages", [])
    print(f"\n✓ Agent returned {len(messages)} messages")

    # Count tool calls
    tool_calls = []
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    print(f"✓ Agent made {len(tool_calls)} tool calls:")
    for i, tc in enumerate(tool_calls, 1):
        print(f"  {i}. {tc.get('name', 'unknown')}")

    # Get final response
    final_message = messages[-1] if messages else None
    if final_message and hasattr(final_message, 'content'):
        output = final_message.content
        print(f"\n✓ Final response length: {len(output)} characters")
        print(f"✓ Preview:\n{output[:300]}...")

    # Validate agent behavior
    if len(tool_calls) == 0:
        print("\n⚠️  WARNING: Agent made no tool calls - expected at least 1")
        print("  This might indicate the agent isn't reasoning properly")
        return False

    if len(tool_calls) > 10:
        print(f"\n⚠️  WARNING: Agent made {len(tool_calls)} tool calls - seems excessive")
        print("  Native agents should be more efficient")

    print("\n" + "=" * 80)
    print("✅ create_react_agent PATTERN TEST PASSED")
    print("=" * 80)
    print(f"""
Key Findings:
- Agent successfully uses create_react_agent() ✓
- LLM reasoning determines tool calls ✓
- Made {len(tool_calls)} tool calls (adaptive)
- Full message history available for tracing ✓
""")
    return True


async def main():
    """Run tests"""
    print("\n🧪 Testing Native Agent Pattern (create_react_agent)\n")

    success = await test_react_agent()

    if success:
        print("\n✅ TEST SUITE PASSED!")
        print("\nThis confirms that:")
        print("  1. create_react_agent() is working correctly")
        print("  2. The agent makes LLM-driven decisions about tool calls")
        print("  3. The ReAct pattern is functioning as expected")
        print("  4. We can proceed with migrating all agents to this pattern")
        return 0
    else:
        print("\n❌ TEST SUITE FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
