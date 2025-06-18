#!/usr/bin/env python3
"""
🚀 STREAMLIT UNIVERSAL RAG CMS FRONTEND
Professional web interface for Universal RAG CMS with ALL 11 features preserved
"""

import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from src.chains.universal_rag_lcel import create_universal_rag_chain

# Page configuration
st.set_page_config(
    page_title="Universal RAG CMS",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-badge {
    background-color: #28a745;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-size: 0.8rem;
    margin: 0.1rem;
    display: inline-block;
}
.confidence-score {
    font-size: 1.5rem;
    font-weight: bold;
    text-align: center;
}
.source-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    enable_live_tracking = st.checkbox("Live Progress Tracking", value=True)
    show_confidence_breakdown = st.checkbox("Detailed Confidence Analysis", value=True)
    show_sources = st.checkbox("Display Sources & Images", value=True)
    show_metadata = st.checkbox("Technical Metadata", value=False)
    
    # Feature status display
    st.subheader("✅ Active Features")
    features = [
        "Advanced Prompt Optimization",
        "Enhanced Confidence Scoring", 
        "Template System v2.0",
        "Contextual Retrieval System",
        "DataForSEO Image Integration",
        "WordPress Publishing",
        "FTI Content Processing",
        "Security & Compliance",
        "Performance Profiling",
        "Web Search Research",
        "Response Storage & Vectorization"
    ]
    
    for feature in features:
        st.markdown(f'<span class="feature-badge">✅</span> {feature}', unsafe_allow_html=True)

# Main interface
st.markdown('<h1 class="main-header">🚀 Universal RAG CMS</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Professional Content Generation with 11 Advanced Features
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_chain" not in st.session_state:
    with st.spinner("🔄 Initializing Universal RAG Chain with all features..."):
        st.session_state.rag_chain = create_universal_rag_chain()
    st.success("✅ All 11 features loaded and ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display additional data for assistant messages
        if message["role"] == "assistant" and "result" in message:
            result = message["result"]
            
            # Confidence score display
            if show_confidence_breakdown:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="confidence-score">🎯 {result.confidence_score:.3f}</div>', unsafe_allow_html=True)
                    st.caption("Confidence Score")
                with col2:
                    st.markdown(f'<div class="confidence-score">📊 {len(result.sources)}</div>', unsafe_allow_html=True)
                    st.caption("Sources Found")
                with col3:
                    features_count = result.metadata.get('advanced_features_count', 11)
                    st.markdown(f'<div class="confidence-score">⚡ {features_count}/11</div>', unsafe_allow_html=True)
                    st.caption("Features Active")
            
            # Sources and images
            if show_sources and result.sources:
                with st.expander(f"📚 View {len(result.sources)} Sources & Images"):
                    for i, source in enumerate(result.sources):
                        st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
                        st.markdown(f"**Source {i+1}:** {source.get('content', 'N/A')}")
                        if 'metadata' in source and 'url' in source['metadata']:
                            url = source['metadata']['url']
                            if url:
                                st.markdown(f"🔗 [View Source]({url})")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Technical metadata
            if show_metadata:
                with st.expander("🔧 Technical Metadata"):
                    st.json(result.metadata)

# Chat input
if prompt := st.chat_input("Enter your content query..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if enable_live_tracking:
            # Create StreamlitCallbackHandler for live tracking
            progress_container = st.container()
            st_callback = StreamlitCallbackHandler(progress_container)
            
            with st.spinner("🚀 Processing with all 11 advanced features..."):
                # Run the Universal RAG Chain with live tracking
                result = asyncio.run(
                    st.session_state.rag_chain.ainvoke(
                        {'query': prompt},
                        config={'callbacks': [st_callback]}
                    )
                )
        else:
            with st.spinner("🚀 Processing with all 11 advanced features..."):
                result = asyncio.run(
                    st.session_state.rag_chain.ainvoke({'query': prompt})
                )
        
        # Display the main response
        st.markdown(result.answer)
        
        # Display confidence and features info
        if show_confidence_breakdown:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="confidence-score">🎯 {result.confidence_score:.3f}</div>', unsafe_allow_html=True)
                st.caption("Confidence Score")
            with col2:
                st.markdown(f'<div class="confidence-score">📊 {len(result.sources)}</div>', unsafe_allow_html=True)
                st.caption("Sources Found")
            with col3:
                features_count = result.metadata.get('advanced_features_count', 11)
                st.markdown(f'<div class="confidence-score">⚡ {features_count}/11</div>', unsafe_allow_html=True)
                st.caption("Features Active")
        
        # Display sources and images
        if show_sources and result.sources:
            with st.expander(f"📚 View {len(result.sources)} Sources & Images"):
                for i, source in enumerate(result.sources):
                    st.markdown(f'<div class="source-card">', unsafe_allow_html=True)
                    st.markdown(f"**Source {i+1}:** {source.get('content', 'N/A')}")
                    if 'metadata' in source and 'url' in source['metadata']:
                        url = source['metadata']['url']
                        if url:
                            st.markdown(f"🔗 [View Source]({url})")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Display technical metadata
        if show_metadata:
            with st.expander("🔧 Technical Metadata"):
                st.json(result.metadata)
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result.answer,
            "result": result
        })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🚀 Universal RAG CMS v6.0 | All 11 Features Active | 
    <a href="https://github.com/peterpeeterspeter/langchain1.2.git" target="_blank">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True) 