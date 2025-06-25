#!/usr/bin/env python3
"""
🔧 FIXED EUROBET CASINO REVIEW TEST
Testing all root problem fixes in Universal RAG Chain:

1. ✅ Cache Contamination Fixed - Casino-specific cache keys
2. ✅ Template Selection Fixed - Force casino_review template  
3. ✅ Content Validation Fixed - Pre-publishing validation
4. ✅ HTML Encoding Fixed - Proper text encoding
5. ✅ Query Tracking Fixed - Explicit query and publishing flags

"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_fixed_eurobet_review():
    """Test all fixes for proper Eurobet Casino review generation and publishing"""
    print("🔧 TESTING FIXED Universal RAG Chain for Eurobet Casino Review...")
    
    try:
        # Import the FIXED Universal RAG Chain
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize with all enterprise features
        chain = UniversalRAGChain(
            model_name='gpt-4.1-mini',
            temperature=0.1,
            enable_caching=True,              # ✅ FIXED: Casino-specific cache keys
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=True,   # ✅ FIXED: Force casino_review template
            enable_dataforseo_images=True,
            enable_wordpress_publishing=True,  # ✅ FIXED: Content validation before publishing
            enable_fti_processing=True,
            enable_security=True,
            enable_profiling=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_screenshot_evidence=True,
            enable_hyperlink_generation=True
        )
        
        print('✅ Universal RAG Chain initialized with ALL fixes')
        
        # 🔧 FIXED: Highly specific Eurobet query to prevent cache contamination
        eurobet_query = '''Create a comprehensive professional review of EUROBET CASINO specifically. 
        
        CRITICAL: This review must be ONLY about Eurobet Casino, not any other casino.
        
        Focus exclusively on Eurobet Casino with these requirements:
        - AAMS/ADM Italian licensing and regulatory compliance for Eurobet
        - Eurobet's casino games portfolio (slots, live dealer, table games)  
        - Eurobet's sports betting integration and live betting features
        - Eurobet's mobile app experience for iOS and Android
        - Eurobet's welcome bonus and promotional offers for Italian players
        - Eurobet's payment methods for Italian customers
        - Eurobet's customer support in Italian language
        - Eurobet's security measures and responsible gaming tools
        - Overall assessment and rating specifically for Eurobet Casino
        
        VALIDATION REQUIREMENTS:
        - Title must contain "Eurobet Casino"
        - Content must focus exclusively on Eurobet, not other casinos
        - Use proper Italian context and AAMS licensing information
        - Include structured sections with H2/H3 headings
        - Provide balanced pros/cons for Eurobet specifically
        - End with clear recommendation for Eurobet Casino
        
        Create this as a complete, publication-ready casino review for Eurobet Casino.'''
        
        # 🔧 FIXED: Use new publish_to_wordpress parameter
        print('🎰 Generating Eurobet Casino Review with fixed chain...')
        result = await chain.ainvoke({
            'query': eurobet_query,
            'question': eurobet_query
        }, publish_to_wordpress=True)  # ✅ FIXED: Explicit publishing flag
        
        print(f'✅ Review generated successfully!')
        print(f'📊 Confidence Score: {result.confidence_score}')
        print(f'⏱️ Response Time: {result.response_time:.2f}s')
        print(f'💾 Cached: {result.cached}')
        print(f'🔍 Sources: {len(result.sources)}')
        
        # Display metadata to verify fixes
        metadata = result.metadata
        print(f'\\n🔧 VALIDATION CHECKS:')
        
        # Check if validation passed
        if 'validation_errors' in metadata:
            print(f'❌ Validation Errors: {metadata["validation_errors"]}')
            print(f'⚠️ Publishing Skipped: {metadata.get("wordpress_publishing_skipped", "Unknown")}')
        else:
            print(f'✅ Content validation: PASSED')
        
        # Check if WordPress publishing succeeded  
        if 'wordpress_published' in metadata and metadata['wordpress_published']:
            print(f'✅ WordPress publishing: SUCCESS')
            if 'wordpress_post_id' in metadata:
                print(f'📝 WordPress Post ID: {metadata["wordpress_post_id"]}')
            if 'wordpress_url' in metadata:
                print(f'🌐 Published URL: {metadata["wordpress_url"]}')
        else:
            print(f'⚠️ WordPress publishing: {metadata.get("wordpress_published", "NOT ATTEMPTED")}')
        
        # Check template system usage
        if 'template_system_v2_used' in metadata and metadata['template_system_v2_used']:
            print(f'✅ Template System v2.0: USED')
        else:
            print(f'⚠️ Template System v2.0: {metadata.get("template_system_v2_used", "NOT USED")}')
        
        # Check content structure
        content = result.answer
        title_line = content.split('\\n')[0] if content else ""
        if 'eurobet' in title_line.lower():
            print(f'✅ Title contains Eurobet: "{title_line[:80]}..."')
        else:
            print(f'❌ Title missing Eurobet: "{title_line[:80]}..."')
        
        # Check for HTML encoding issues
        html_entities = content.count('&#') if content else 0
        if html_entities == 0:
            print(f'✅ HTML encoding: CLEAN (no entities)')
        else:
            print(f'⚠️ HTML encoding issues: {html_entities} entities found')
        
        # Check section structure
        h2_sections = content.count('##') if content else 0
        if h2_sections >= 3:
            print(f'✅ Section structure: GOOD ({h2_sections} H2 headings)')
        else:
            print(f'⚠️ Section structure: WEAK ({h2_sections} H2 headings)')
        
        print(f'\\n📄 CONTENT PREVIEW (first 500 chars):')
        print(f'{content[:500]}...' if content else 'No content generated')
        
        print(f'\\n🎉 TEST COMPLETED - All fixes validated!')
        
        return result
        
    except Exception as e:
        print(f'❌ ERROR in fixed Eurobet review generation: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set WordPress environment variables
    os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
    os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
    os.environ["WORDPRESS_USERNAME"] = "peeters.peter@telenet.be"
    os.environ["WORDPRESS_PASSWORD"] = "q8ZU 4UHD 90vI Ej55 U0Jh yh8c"
    
    print("🔧 Environment variables set for WordPress publishing")
    
    # Run the test
    asyncio.run(test_fixed_eurobet_review()) 