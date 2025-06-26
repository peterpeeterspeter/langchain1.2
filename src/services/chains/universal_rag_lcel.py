"""
Universal RAG LCEL Chain - Native LangChain Cache Integration.

This shows the NATIVE LangChain way to add semantic caching to RAG chains.
Just set_llm_cache() once and ALL LLM calls are automatically cached!
"""

import sys
import os
from typing import List, Any
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_redis.cache import RedisSemanticCache
from langchain_core.globals import set_llm_cache


# Simple RAG prompt for casino reviews
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert casino reviewer. Based on the provided context about {casino_name}, 
write a comprehensive and professional casino review.

Context: {context}

Question: Please provide a detailed review of {casino_name} casino, covering:
- Overview and reputation
- Game selection and quality
- User experience and interface
- Security and licensing
- Payment methods and processing times
- Customer support
- Bonuses and promotions
- Overall rating and recommendation

Casino to review: {casino_name}

Professional Casino Review:
""")


def format_docs(docs: List[Any]) -> str:
    """Format retrieved documents."""
    return "\n\n".join(doc.page_content for doc in docs)


def setup_cache(embeddings, redis_url="redis://localhost:6379", content_type="default"):
    """
    NATIVE LANGCHAIN CACHE SETUP - This is it!
    """
    try:
        # Content-specific configurations (distance_threshold as per langchain-redis docs)
        configs = {
            'news': {'distance_threshold': 0.15},      # strict matching
            'reviews': {'distance_threshold': 0.2},    # moderate matching  
            'regulatory': {'distance_threshold': 0.25}, # relaxed matching
            'default': {'distance_threshold': 0.2}     # moderate matching
        }
        
        config = configs.get(content_type, configs['default'])
        
        # THE NATIVE LANGCHAIN WAY - Just this!
        set_llm_cache(RedisSemanticCache(
            embeddings=embeddings,  # Note: 'embeddings' parameter name
            redis_url=redis_url,
            distance_threshold=config['distance_threshold']  # Note: 'distance_threshold' parameter
        ))
        return True
    except Exception as e:
        print(f"⚠️  Redis cache setup failed: {e}")
        print("💡 Continuing without caching for demo purposes")
        return False


def create_casino_review_chain(llm, casino_name, context=""):
    """
    Create a casino review chain. Cache is handled globally by set_llm_cache().
    """
    return (
        {
            "context": lambda _: context,
            "casino_name": lambda _: casino_name
        }
        | RAG_PROMPT
        | llm  # Automatically cached via set_llm_cache()!
        | StrOutputParser()
    )


def generate_casino_review(casino_name):
    """Generate a casino review using the native LangChain cached chain."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        print(f"🎰 Setting up native LangChain cache for {casino_name} review...")
        
        # Initialize components
        embeddings = OpenAIEmbeddings()
        
        # 1. Set up cache ONCE globally - Native LangChain way!
        cache_enabled = setup_cache(embeddings, content_type="reviews")
        if cache_enabled:
            print("✅ Cache configured with native RedisSemanticCache")
        else:
            print("⚠️  Running without cache for this demo")
        
        # 2. Create LLM normally
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # 3. Sample context for the casino (in a real app, this would come from a vectorstore)
        context = f"""
        {casino_name} is a cryptocurrency casino that has been operating since 2018.
        It offers a wide variety of games including slots, table games, and live dealer options.
        The casino is licensed and regulated, ensuring fair play and secure transactions.
        Popular games include various slot machines, blackjack, roulette, and baccarat.
        The platform supports multiple cryptocurrencies for deposits and withdrawals.
        Customer support is available 24/7 through live chat and email.
        The casino offers various bonuses and promotions for new and existing players.
        """
        
        # 4. Create and use the chain - caching happens automatically!
        chain = create_casino_review_chain(llm, casino_name, context)
        
        print(f"🔄 Generating review for {casino_name}...")
        print("💡 Note: Similar queries will be automatically cached!")
        
        response = chain.invoke({})
        
        print(f"\n📝 Generated Casino Review for {casino_name}:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        print("✅ Review generation complete!")
        print("🚀 Next similar query will hit the cache automatically!")
        
        return response
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("💡 Please install: pip install langchain-openai")
        return None
    except Exception as e:
        print(f"❌ Error generating review: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        casino_name = sys.argv[1]
    else:
        casino_name = "TrustDice"
    
    print(f"🎯 Universal RAG LCEL Chain - Native LangChain Caching Demo")
    print(f"🎰 Casino: {casino_name}")
    print("=" * 60)
    
    review = generate_casino_review(casino_name)
    
    if review:
        print(f"\n🎉 Successfully generated review for {casino_name}!")
    else:
        print(f"\n❌ Failed to generate review for {casino_name}")


# Example usage - THIS IS THE NATIVE WAY:
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize components
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("casino_docs", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 1. Set up cache ONCE globally
setup_cache(embeddings, content_type="news")

# 2. Create LLM and chain normally
llm = ChatOpenAI(model="gpt-4")
chain = create_rag_chain(llm, retriever)

# 3. Use it - caching happens automatically!
response = chain.invoke("What's the latest casino news?")
response2 = chain.invoke("Tell me recent casino updates")  # Cache hit if similar!
""" 