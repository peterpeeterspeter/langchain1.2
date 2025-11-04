#!/usr/bin/env python3
"""
Test that all native agents have the correct structure
"""
import os

print("🧪 Testing All Native Agents Structure\n")
print("=" * 80)

base_dir = os.path.dirname(os.path.abspath(__file__))

agents_to_check = [
    ("Research Agent", "src/agents/research_agent_native.py", "create_native_research_agent", "_extract_research_data_from_messages"),
    ("Writing Agent", "src/agents/writing_agent_native.py", "create_native_writing_agent", "_extract_writing_data_from_messages"),
    ("Affiliate Agent", "src/agents/affiliate_agent_native.py", "create_native_affiliate_agent", "_extract_affiliate_data_from_messages"),
    ("Image Agent", "src/agents/image_agent_native.py", "create_native_image_agent", "_extract_image_data_from_messages"),
    ("Publishing Agent", "src/agents/publishing_agent_native.py", "create_native_publishing_agent", "_extract_publishing_data_from_messages"),
]

all_passed = True

for i, (name, filepath, func_name, extract_func) in enumerate(agents_to_check, 1):
    print(f"\n{i}. Testing {name}...")

    full_path = os.path.join(base_dir, filepath)

    if not os.path.exists(full_path):
        print(f"   ❌ {name}: File not found")
        all_passed = False
        continue

    # Read file and check for key patterns
    with open(full_path, 'r') as f:
        content = f.read()

    tests = [
        ('from langgraph.prebuilt import create_react_agent', 'Uses create_react_agent import'),
        ('from langchain_core.messages import SystemMessage', 'Uses SystemMessage'),
        (f'def {func_name}(', f'Has {func_name} function'),
        ('create_react_agent(', 'Calls create_react_agent()'),
        ('HumanMessage(content=', 'Uses message-based invocation'),
        (f'def {extract_func}(messages:', f'Has {extract_func} function'),
        ('for message in messages:', 'Iterates over messages'),
        ("hasattr(message, 'tool_calls')", 'Checks for tool calls'),
        ("hasattr(message, 'name')", 'Checks for tool responses'),
    ]

    agent_passed = True
    for pattern, description in tests:
        if pattern in content:
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - MISSING")
            agent_passed = False
            all_passed = False

    if agent_passed:
        print(f"   🎉 {name} is correctly updated!")

print("\n" + "=" * 80)

if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("\nSummary:")
    print("  ✅ All 5 agents use create_react_agent()")
    print("  ✅ All 5 agents use SystemMessage")
    print("  ✅ All 5 agents have creation functions")
    print("  ✅ All 5 agents use message-based invocation")
    print("  ✅ All 5 agents have message-based extraction")
    print("\n🎉 Native agents are ready for production!")
    exit(0)
else:
    print("❌ SOME TESTS FAILED")
    print("\nPlease review the failures above and fix the issues.")
    exit(1)
