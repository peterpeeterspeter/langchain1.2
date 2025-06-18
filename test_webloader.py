#!/usr/bin/env python3
"""
Test Native LangChain WebBaseLoader for Casino Research
"""

import sys
import os
sys.path.append('src')

from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin
import time

def test_webbase_loader():
    """Test basic WebBaseLoader functionality"""
    print('🎯 Testing Native LangChain WebBaseLoader for Casino Research...')
    
    # Test URLs for different data categories
    test_urls = [
        'https://betway.com/about',
        'https://betway.com/casino',
        'https://betway.com/promotions',
        'https://betway.com/banking'
    ]
    
    # Configure WebBaseLoader with casino-friendly settings
    loader = WebBaseLoader(
        web_paths=test_urls,
        header_template={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        },
        requests_per_second=0.5,  # Very conservative for testing
        continue_on_failure=True,
        raise_for_status=False
    )
    
    try:
        print(f'📥 Loading {len(test_urls)} URLs...')
        start_time = time.time()
        
        docs = loader.load()
        
        load_time = time.time() - start_time
        print(f'✅ Successfully loaded {len(docs)} documents in {load_time:.2f}s')
        
        # Analyze loaded content
        total_content = 0
        data_points_found = {
            'licensing': 0,
            'games': 0, 
            'bonuses': 0,
            'payments': 0,
            'support': 0
        }
        
        for i, doc in enumerate(docs):
            content = doc.page_content.lower()
            total_content += len(doc.page_content)
            
            print(f'\\n📄 Document {i+1}:')
            print(f'  Source: {doc.metadata.get("source", "Unknown")}')
            print(f'  Title: {doc.metadata.get("title", "No title")}')
            print(f'  Content Length: {len(doc.page_content):,} chars')
            
            # Check for data points
            if any(term in content for term in ['license', 'regulation', 'authority', 'mga', 'ukgc']):
                data_points_found['licensing'] += 1
            if any(term in content for term in ['slot', 'casino', 'game', 'rtp', 'provider']):
                data_points_found['games'] += 1
            if any(term in content for term in ['bonus', 'promotion', 'welcome', 'free spin']):
                data_points_found['bonuses'] += 1
            if any(term in content for term in ['payment', 'deposit', 'withdrawal', 'banking']):
                data_points_found['payments'] += 1
            if any(term in content for term in ['support', 'help', 'contact', 'chat']):
                data_points_found['support'] += 1
                
            # Show content preview
            preview = doc.page_content[:200].replace('\\n', ' ')
            print(f'  Preview: {preview}...')
        
        print(f'\\n🎯 DATA POINTS ANALYSIS:')
        print(f'Total Content Collected: {total_content:,} characters')
        for category, count in data_points_found.items():
            print(f'  {category.title()}: Found in {count}/{len(docs)} documents')
            
        # Calculate potential for 95 data points
        potential_score = sum(data_points_found.values()) / len(data_points_found) * 100 / len(docs)
        print(f'\\n📊 Potential for 95 Data Points: {potential_score:.1f}% coverage')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_webbase_loader()
    if success:
        print('\\n✅ WebBaseLoader test completed successfully!')
        print('🚀 Ready to implement full 95 data points collection!')
    else:
        print('\\n❌ WebBaseLoader test failed - check network connectivity and URLs') 