#!/usr/bin/env python3
"""
🚀 UNIVERSAL RAG CHAIN DEMO WITH BULLETPROOF IMAGE INTEGRATION
Demonstrates the complete solution working end-to-end
"""

import sys
import os
import asyncio
sys.path.insert(0, 'src')

from chains.universal_rag_lcel import create_universal_rag_chain

async def main():
    print('🚀 RUNNING UNIVERSAL RAG CHAIN WITH BULLETPROOF IMAGE INTEGRATION')
    print('=' * 70)
    
    try:
        # Create the chain (includes bulletproof image integration)
        print('📝 Creating Universal RAG Chain...')
        chain = create_universal_rag_chain()
        
        # Query about a casino (triggers DataForSEO + image embedding)
        query = 'Review Betway Casino focusing on games and user experience'
        
        print(f'🎯 Query: {query}')
        print('⏳ Processing with enhanced features...')
        
        # Run the complete chain
        response = await chain.ainvoke({'query': query})
        
        print('✅ RESULTS:')
        
        # Handle different response types
        if hasattr(response, 'content'):
            content = response.content
            confidence = getattr(response, 'confidence_score', 0.0)
        elif isinstance(response, str):
            content = response  
            confidence = 0.0
        else:
            content = str(response)
            confidence = 0.0
            
        print(f'📄 Content length: {len(content)} characters')
        print(f'🖼️  Images embedded: {"YES" if "<img" in content else "NO"}')
        print(f'🔍 Confidence score: {confidence:.3f}')
        
        # Show preview of content
        preview_length = 800
        preview = content[:preview_length] + '...' if len(content) > preview_length else content
        
        print('\n📖 CONTENT PREVIEW:')
        print('-' * 50)
        print(preview)
        
        # Check for bulletproof features
        print('\n🔧 BULLETPROOF FEATURES CHECK:')
        features = {
            'Hero Image': 'hero-image' in content,
            'Responsive Design': 'max-width' in content,
            'Lazy Loading': 'loading=' in content,
            'Image Captions': 'image-caption' in content,
            'Professional HTML': '<div class=' in content
        }
        
        for feature, enabled in features.items():
            status = '✅' if enabled else '❌'
            print(f'{status} {feature}: {"Enabled" if enabled else "Disabled"}')
            
        if any(features.values()):
            print('\n🎉 BULLETPROOF IMAGE INTEGRATION WORKING!')
        else:
            print('\n⚠️  No advanced image features detected')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        print('💡 Make sure environment variables are set (.env file)')

if __name__ == "__main__":
    asyncio.run(main()) 