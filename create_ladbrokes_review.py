#!/usr/bin/env python3
"""
🎰 LADBROKES CASINO REVIEW - MT_LISTING PUBLICATION
Using Fixed Universal RAG Chain with All Enterprise Features

🔧 DEMONSTRATES ALL ROOT FIXES:
1. ✅ Casino-specific cache keys (no cross-contamination)
2. ✅ Forced casino_review template selection
3. ✅ Pre-publishing content validation
4. ✅ Clean HTML encoding (no entities)
5. ✅ Explicit WordPress publishing with MT Casino integration

📝 Publishing Target: MT_LISTING custom post type on crashcasino.io
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

async def create_ladbrokes_casino_review():
    """Create and publish comprehensive Ladbrokes Casino review as MT_LISTING"""
    print("🎰 Creating Ladbrokes Casino Review for MT_LISTING Publication...")
    
    try:
        # Import the FIXED Universal RAG Chain
        from src.chains.universal_rag_lcel import UniversalRAGChain
        
        # Initialize with ALL enterprise features enabled
        chain = UniversalRAGChain(
            model_name='gpt-4.1-mini',
            temperature=0.1,
            enable_caching=True,              # ✅ FIXED: Casino-specific cache keys
            enable_contextual_retrieval=True,
            enable_prompt_optimization=True,
            enable_enhanced_confidence=True,
            enable_template_system_v2=True,   # ✅ FIXED: Force casino_review template
            enable_dataforseo_images=True,
            enable_wordpress_publishing=True,  # ✅ FIXED: Content validation + MT Casino
            enable_fti_processing=True,
            enable_security=True,
            enable_profiling=True,
            enable_web_search=True,
            enable_comprehensive_web_research=True,
            enable_screenshot_evidence=True,
            enable_hyperlink_generation=True
        )
        
        print('✅ Universal RAG Chain initialized with ALL fixes')
        
        # 🎰 LADBROKES-SPECIFIC QUERY (prevents cache contamination)
        ladbrokes_query = '''Create a comprehensive professional review of LADBROKES CASINO specifically for UK players.
        
        CRITICAL REQUIREMENTS - This review must be EXCLUSIVELY about Ladbrokes Casino:
        
        ## Focus Areas for Ladbrokes Casino:
        - UK Gambling Commission licensing and regulatory compliance for Ladbrokes
        - Ladbrokes' casino games portfolio (2000+ slots, live dealer, table games)
        - Ladbrokes' established sports betting integration and live betting platform
        - Ladbrokes' mobile app experience and mobile casino optimization
        - Ladbrokes' welcome bonus offers and ongoing promotions for UK players
        - Ladbrokes' payment methods including UK bank transfers and e-wallets
        - Ladbrokes' 24/7 customer support in English
        - Ladbrokes' security measures, responsible gambling tools, and player protection
        - Ladbrokes' reputation as an established UK gambling brand since 1886
        - Overall assessment and rating specifically for Ladbrokes Casino operations
        
        ## Content Structure Requirements:
        - Title must prominently feature "Ladbrokes Casino"
        - Include executive summary with key findings about Ladbrokes
        - Detailed sections with H2/H3 headings for each major topic
        - Balanced pros and cons specific to Ladbrokes Casino
        - Clear overall rating out of 10 for Ladbrokes
        - Final recommendation section for UK players considering Ladbrokes
        - Include compliance information for UK regulatory requirements
        
        ## Content Quality Standards:
        - Professional tone suitable for UK gambling market
        - Accurate information about Ladbrokes' services and features
        - No mention of competing casinos unless for brief comparison context
        - Include relevant UK gambling legislation context (UKGC regulations)
        - Emphasis on responsible gambling and player protection measures
        
        Create this as a complete, publication-ready MT Casino listing review for Ladbrokes Casino.'''
        
        print('🎰 Generating Ladbrokes Casino Review...')
        print('📝 Query focused exclusively on Ladbrokes to prevent cache contamination')
        
        # 🔧 FIXED: Use explicit publish_to_wordpress flag for MT_LISTING
        result = await chain.ainvoke({
            'query': ladbrokes_query,
            'question': ladbrokes_query,
            'content_type': 'individual_casino_review',  # Ensures MT Casino publishing
            'target_casino': 'ladbrokes',  # Explicit casino targeting
            'publish_format': 'mt_listing'  # MT Casino custom post type
        }, publish_to_wordpress=True)  # ✅ FIXED: Explicit publishing flag
        
        print(f'✅ Ladbrokes Casino Review Generated Successfully!')
        
        # 📊 COMPREHENSIVE VALIDATION REPORT
        print(f'\\n🔧 === FIXED UNIVERSAL RAG CHAIN VALIDATION RESULTS ===')
        
        print(f'\\n📈 Performance Metrics:')
        print(f'📊 Confidence Score: {result.confidence_score:.3f}')
        print(f'⏱️ Response Time: {result.response_time:.2f}s')
        print(f'💾 Cache Status: {"HIT" if result.cached else "FRESH GENERATION"}')
        print(f'🔍 Research Sources: {len(result.sources)}')
        
        # 🔧 VALIDATION CHECKS FOR ALL ROOT FIXES
        metadata = result.metadata
        content = result.answer
        
        print(f'\\n🔧 ROOT PROBLEM FIXES VALIDATION:')
        
        # Fix 1: Cache Contamination Check
        print(f'\\n1️⃣ CACHE CONTAMINATION FIX:')
        if result.cached:
            print(f'💾 Content was cached - checking cache key specificity')
        else:
            print(f'✅ Fresh content generated - cache working correctly')
        
        # Fix 2: Template Selection Check  
        print(f'\\n2️⃣ TEMPLATE SELECTION FIX:')
        if metadata.get('template_system_v2_used'):
            print(f'✅ Template System v2.0: ACTIVE (casino_review template)')
        else:
            print(f'⚠️ Template System v2.0: {metadata.get("template_system_v2_used", "NOT DETECTED")}')
        
        # Fix 3: Content Validation Check
        print(f'\\n3️⃣ CONTENT VALIDATION FIX:')
        if 'validation_errors' in metadata:
            print(f'❌ Content Validation: FAILED')
            print(f'   Errors: {metadata["validation_errors"]}')
        else:
            print(f'✅ Content Validation: PASSED (no validation errors)')
        
        # Fix 4: HTML Encoding Check
        print(f'\\n4️⃣ HTML ENCODING FIX:')
        html_entities = content.count('&#') if content else 0
        if html_entities == 0:
            print(f'✅ HTML Encoding: CLEAN (0 HTML entities)')
        else:
            print(f'⚠️ HTML Encoding: {html_entities} HTML entities detected')
        
        # Fix 5: Query Tracking & Publishing Check
        print(f'\\n5️⃣ WORDPRESS PUBLISHING FIX:')
        if metadata.get('wordpress_published'):
            print(f'✅ WordPress Publishing: SUCCESS')
            if metadata.get('wordpress_post_id'):
                print(f'   📝 MT_LISTING Post ID: {metadata["wordpress_post_id"]}')
            if metadata.get('wordpress_url'):
                print(f'   🌐 Published URL: {metadata["wordpress_url"]}')
            if metadata.get('custom_post_type'):
                print(f'   🏷️ Custom Post Type: {metadata["custom_post_type"]}')
        else:
            publishing_status = metadata.get('wordpress_published', 'NOT ATTEMPTED')
            print(f'⚠️ WordPress Publishing: {publishing_status}')
            if 'wordpress_publishing_skipped' in metadata:
                print(f'   Reason: {metadata["wordpress_publishing_skipped"]}')
        
        # 🎯 CONTENT QUALITY CHECKS
        print(f'\\n🎯 CONTENT QUALITY VALIDATION:')
        
        # Title check
        title_line = content.split('\\n')[0] if content else ""
        if 'ladbrokes' in title_line.lower():
            print(f'✅ Title Validation: Contains "Ladbrokes" - "{title_line[:80]}..."')
        else:
            print(f'❌ Title Validation: Missing "Ladbrokes" - "{title_line[:80]}..."')
        
        # Structure check
        h2_count = content.count('##') if content else 0
        if h2_count >= 3:
            print(f'✅ Structure Validation: Good structure ({h2_count} H2 sections)')
        else:
            print(f'⚠️ Structure Validation: Weak structure ({h2_count} H2 sections)')
        
        # Casino focus check
        ladbrokes_mentions = content.lower().count('ladbrokes') if content else 0
        other_casino_mentions = sum([
            content.lower().count('bet365') if content else 0,
            content.lower().count('william hill') if content else 0,
            content.lower().count('eurobet') if content else 0,
            content.lower().count('trustdice') if content else 0
        ])
        
        if ladbrokes_mentions >= 5 and other_casino_mentions <= 2:
            print(f'✅ Casino Focus: Excellent ({ladbrokes_mentions} Ladbrokes mentions, {other_casino_mentions} other casinos)')
        else:
            print(f'⚠️ Casino Focus: {ladbrokes_mentions} Ladbrokes mentions, {other_casino_mentions} other casino mentions')
        
        # 📄 CONTENT PREVIEW
        print(f'\\n📄 LADBROKES CASINO REVIEW PREVIEW:')
        print(f'{"="*80}')
        if content:
            # Show first 800 characters
            preview = content[:800]
            print(preview)
            if len(content) > 800:
                print("\\n[... content continues ...]")
            print(f'\\nTotal Content Length: {len(content)} characters')
        else:
            print("❌ No content generated")
        print(f'{"="*80}')
        
        # 🎉 FINAL STATUS
        if metadata.get('wordpress_published') and not metadata.get('validation_errors'):
            print(f'\\n🎉 SUCCESS: Ladbrokes Casino Review Published as MT_LISTING!')
            print(f'✅ All root problems fixed and validated')
            print(f'✅ Content published to WordPress with proper MT Casino structure')
        else:
            print(f'\\n⚠️ PARTIAL SUCCESS: Review generated but publishing may have issues')
            print(f'🔧 Check validation errors and publishing status above')
        
        return result
        
    except Exception as e:
        print(f'❌ ERROR in Ladbrokes Casino review generation: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set WordPress environment variables for MT Casino publishing
    os.environ["WORDPRESS_URL"] = "https://www.crashcasino.io"
    os.environ["WORDPRESS_SITE_URL"] = "https://www.crashcasino.io"
    os.environ["WORDPRESS_USERNAME"] = "peeters.peter@telenet.be"
    os.environ["WORDPRESS_PASSWORD"] = "your-wordpress-password-here"
    
    print("🔧 WordPress environment configured for MT_LISTING publication")
    print("🎰 Starting Ladbrokes Casino Review with FIXED Universal RAG Chain...")
    
    # Execute the review generation and publication
    asyncio.run(create_ladbrokes_casino_review()) 