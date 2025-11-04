#!/usr/bin/env python3
"""
Batch update remaining native agents to use create_react_agent
"""
import re
from pathlib import Path

def update_agent_file(filepath):
    """Update a single agent file"""
    print(f"\nUpdating {filepath.name}...")

    content = filepath.read_text()
    original_content = content

    # 1. Fix return type annotation
    content = re.sub(
        r'\) -> AgentExecutor:',
        r'):',
        content
    )

    # 2. Update docstring
    content = content.replace(
        'using create_tool_calling_agent()',
        'using create_react_agent()'
    )
    content = content.replace(
        'AgentExecutor configured with',
        'Compiled LangGraph agent'
    )

    # 3. Replace ChatPromptTemplate with SystemMessage
    content = re.sub(
        r'prompt = ChatPromptTemplate\.from_messages\(\[\s*\("system",\s*(""".*?""")\),\s*\("human", "\{input\}"\),\s*\("placeholder", "\{agent_scratchpad\}"\),\s*\]\)',
        r'system_message = SystemMessage(content=\1)',
        content,
        flags=re.DOTALL
    )

    # 4. Replace create_tool_calling_agent + AgentExecutor with create_react_agent
    content = re.sub(
        r'agent = create_tool_calling_agent\(llm, tools, prompt\)\s*'
        r'agent_executor = AgentExecutor\(\s*agent=agent,\s*tools=tools,\s*verbose=verbose,\s*max_iterations=max_iterations,\s*handle_parsing_errors=True,\s*return_intermediate_steps=True,\s*\)\s*'
        r'logger\.info\(f"Created native .* agent with \{len\(tools\)\} tools"\)\s*'
        r'return agent_executor',
        r'agent = create_react_agent(\n        llm,\n        tools,\n        prompt=system_message\n    )\n\n    logger.info(f"Created native agent with {len(tools)} tools")\n    return agent',
        content,
        flags=re.DOTALL
    )

    # 5. Update node function: agent_executor -> agent
    content = content.replace('agent_executor = create_native_', 'agent = create_native_')
    content = content.replace('agent = create_native_', 'agent = create_native_')

    # 6. Update invocation to use messages
    content = re.sub(
        r'result = await agent(_executor)?\.ainvoke\(\{\s*"input": (.*?),\s*\}\)',
        r'from langchain_core.messages import HumanMessage\n        result = await agent.ainvoke({\n            "messages": [HumanMessage(content=\2)]\n        })',
        content,
        flags=re.DOTALL
    )

    # 7. Update result extraction
    content = re.sub(
        r'output = result\.get\("output", ""\)\s*intermediate_steps = result\.get\("intermediate_steps", \[\]\)',
        r'messages = result.get("messages", [])\n        final_message = messages[-1] if messages else None\n        output = final_message.content if final_message and hasattr(final_message, \'content\') else ""',
        content
    )

    # 8. Update extraction helper function calls
    content = re.sub(
        r'_extract_(\w+)_data_from_steps\(intermediate_steps\)',
        r'_extract_\1_data_from_messages(messages)',
        content
    )

    # 9. Update state to use messages instead of intermediate_steps
    content = re.sub(
        r'state\["(\w+)_intermediate_steps"\] = intermediate_steps',
        r'state["\1_messages"] = messages',
        content
    )

    # 10. Update extraction helper function definitions
    content = re.sub(
        r'def _extract_(\w+)_data_from_steps\(intermediate_steps: (?:list|List)\) -> Dict\[str, Any\]:',
        r'def _extract_\1_data_from_messages(messages: list) -> Dict[str, Any]:',
        content
    )

    # 11. Update extraction helper docstrings
    content = content.replace(
        'Extract structured data from agent\'s intermediate steps',
        'Extract structured data from agent\'s message history'
    )
    content = content.replace(
        'intermediate_steps: List of (AgentAction, observation) tuples',
        'messages: List of messages from agent execution'
    )

    # 12. Update extraction helper body
    content = re.sub(
        r'for action, observation in intermediate_steps:\s*tool_name = action\.tool\s*tool_input = action\.tool_input',
        r'''for message in messages:
        # Check for tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', '')
                tool_input = tool_call.get('args', {})
                # Store tool call info

        # Check for tool responses
        if hasattr(message, 'name') and message.name:
            tool_name = message.name
            tool_output = message.content''',
        content,
        flags=re.DOTALL
    )

    # 13. Replace observation with tool_output in extraction logic
    content = content.replace('if isinstance(observation, dict):', 'if isinstance(tool_output, dict):')
    content = content.replace('observation.get(', 'tool_output.get(')

    if content != original_content:
        filepath.write_text(content)
        print(f"✓ Updated {filepath.name}")
        return True
    else:
        print(f"⚠ No changes made to {filepath.name}")
        return False

def main():
    agents_dir = Path('src/agents')

    files_to_update = [
        agents_dir / 'affiliate_agent_native.py',
        agents_dir / 'image_agent_native.py',
        agents_dir / 'publishing_agent_native.py',
    ]

    updated_count = 0
    for filepath in files_to_update:
        if filepath.exists():
            if update_agent_file(filepath):
                updated_count += 1
        else:
            print(f"❌ File not found: {filepath}")

    print(f"\n✅ Updated {updated_count}/{len(files_to_update)} files")

if __name__ == '__main__':
    main()
