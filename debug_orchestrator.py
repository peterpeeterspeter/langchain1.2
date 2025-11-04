#!/usr/bin/env python3
"""
Debug script to identify missing components and function calls
"""

import os
import sys
import asyncio
from pathlib import Path

# Load credentials from environment variables
# Required: OPENAI_API_KEY, GOOGLE_API_KEY, TAVILY_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY
# Set these in your .env file or export them before running
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("DEBUGGING ORCHESTRATOR - Missing Components & Function Calls")
print("=" * 80)
print()

# Test 1: Check RAG Chain initialization
print("1. Testing RAG Chain Initialization...")
try:
    from src.chains.universal_rag_lcel import create_universal_rag_chain
    chain = create_universal_rag_chain(
        model_name="gpt-4o-mini",
        enable_wordpress_publishing=False,
        enable_comprehensive_web_research=True,
        enable_web_search=True
    )
    print("   ✅ RAG Chain created successfully")
    print(f"   ✅ Chain type: {type(chain)}")
    print(f"   ✅ Has ainvoke: {hasattr(chain, 'ainvoke')}")
except Exception as e:
    print(f"   ❌ RAG Chain creation failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 2: Check RAG Chain response format
print("2. Testing RAG Chain Response Format...")
try:
    async def test_rag_response():
        response = await chain.ainvoke({"query": "test query"})
        print(f"   ✅ Response type: {type(response)}")
        print(f"   ✅ Response attributes: {dir(response)}")
        print(f"   ✅ Has 'answer': {hasattr(response, 'answer')}")
        print(f"   ✅ Has 'confidence_score': {hasattr(response, 'confidence_score')}")
        print(f"   ✅ Has 'sources': {hasattr(response, 'sources')}")
        if hasattr(response, 'answer'):
            print(f"   ✅ Answer type: {type(response.answer)}")
            print(f"   ✅ Answer length: {len(str(response.answer))}")
    asyncio.run(test_rag_response())
except Exception as e:
    print(f"   ❌ RAG Chain response test failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Check Writing Tools
print("3. Testing Writing Tools...")
try:
    from src.agents.tools.writing_tools import content_generation_tool, _get_rag_chain
    rag_chain = _get_rag_chain()
    print(f"   ✅ _get_rag_chain() returned: {type(rag_chain)}")
    if rag_chain:
        print("   ✅ RAG chain is initialized")
    else:
        print("   ❌ RAG chain is None!")
except Exception as e:
    print(f"   ❌ Writing tools test failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 4: Check Writing Agent
print("4. Testing Writing Agent...")
try:
    from langchain_openai import ChatOpenAI
    from src.agents.writing_agent import WritingAgent
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    writing_agent = WritingAgent(llm=llm)
    print("   ✅ Writing Agent created")
    print(f"   ✅ Agent has execute: {hasattr(writing_agent, 'execute')}")
    print(f"   ✅ Agent has run: {hasattr(writing_agent, 'run')}")
    print(f"   ✅ Agent tools: {[t.name for t in writing_agent.tools]}")
except Exception as e:
    print(f"   ❌ Writing Agent test failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 5: Check Orchestrator LCEL Chains
print("5. Testing Orchestrator LCEL Chains...")
try:
    from src.agents.orchestrator import ArticleCMSOrchestrator
    from src.agents.research_agent import ResearchAgent
    from src.agents.writing_agent import WritingAgent
    
    research_agent = ResearchAgent(llm=llm)
    writing_agent = WritingAgent(llm=llm)
    
    orchestrator = ArticleCMSOrchestrator(
        research_agent=research_agent,
        writing_agent=writing_agent,
        enable_checkpoints=False
    )
    print("   ✅ Orchestrator created")
    print(f"   ✅ Has research_chain: {hasattr(orchestrator, 'research_chain')}")
    print(f"   ✅ Has writing_chain: {hasattr(orchestrator, 'writing_chain')}")
    if hasattr(orchestrator, 'research_chain'):
        print(f"   ✅ Research chain type: {type(orchestrator.research_chain)}")
    if hasattr(orchestrator, 'writing_chain'):
        print(f"   ✅ Writing chain type: {type(orchestrator.writing_chain)}")
except Exception as e:
    print(f"   ❌ Orchestrator test failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 6: Test actual content generation tool call
print("6. Testing Content Generation Tool Call...")
try:
    async def test_content_gen():
        result = await content_generation_tool.ainvoke({
            "query": "Coincasino Review 2025",
            "research_data": {"test": "data"},
            "context": "test context"
        })
        print(f"   ✅ Tool returned: {type(result)}")
        print(f"   ✅ Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        print(f"   ✅ Has content: {'content' in result if isinstance(result, dict) else False}")
        if isinstance(result, dict) and 'content' in result:
            content_len = len(result['content'])
            print(f"   ✅ Content length: {content_len}")
            if content_len == 0:
                print(f"   ⚠️  Content is empty!")
                if 'error' in result:
                    print(f"   ⚠️  Error: {result['error']}")
    asyncio.run(test_content_gen())
except Exception as e:
    print(f"   ❌ Content generation tool test failed: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)

