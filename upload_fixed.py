#!/usr/bin/env python3
"""
Fixed Upload Script for All 32+ Templates
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Import LangChain hub
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# Add project path
sys.path.append(str(Path(__file__).parent))

# Import our systems
from src.chains.advanced_prompt_system import DomainSpecificPrompts, QueryType, ExpertiseLevel
from src.templates.improved_template_manager import IMPROVED_UNIVERSAL_RAG_TEMPLATE, IMPROVED_FTI_GENERATION_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def upload_all_templates():
    """Upload all 32+ templates to LangChain Hub"""
    
    # Initialize prompts system
    domain_prompts = DomainSpecificPrompts()
    uploaded = {}
    failed = []
    
    logger.info("🎯 Uploading 32+ Template System v2.0 Templates")
    logger.info("=" * 60)
    
    # Upload domain-specific templates (32 templates)
    query_types = list(QueryType)
    expertise_levels = list(ExpertiseLevel)
    
    for query_type in query_types:
        for expertise_level in expertise_levels:
            try:
                # Get template string from base_prompts
                template_string = domain_prompts.base_prompts.get(query_type, {}).get(expertise_level)
                
                if not template_string:
                    logger.warning(f"⚠️ No template found for {query_type.value}_{expertise_level.value}")
                    continue
                
                # Create template
                template = ChatPromptTemplate.from_template(template_string)
                
                # Create hub ID
                hub_id = f"{query_type.value}-{expertise_level.value}-template"
                
                # Upload to hub
                logger.info(f"🚀 Uploading {hub_id}...")
                response = hub.push(hub_id, template)
                
                uploaded[f"{query_type.value}_{expertise_level.value}"] = hub_id
                logger.info(f"✅ Success: {hub_id}")
                logger.info(f"   URL: {response}")
                
            except Exception as e:
                logger.error(f"❌ Failed {query_type.value}_{expertise_level.value}: {e}")
                failed.append(f"{query_type.value}_{expertise_level.value}: {e}")
    
    # Upload universal templates
    try:
        # Universal RAG
        universal_template = ChatPromptTemplate.from_template(IMPROVED_UNIVERSAL_RAG_TEMPLATE)
        response = hub.push('universal-rag-template-v2', universal_template)
        uploaded['universal_rag'] = 'universal-rag-template-v2'
        logger.info(f"✅ Universal RAG: {response}")
        
        # FTI Template
        fti_template = ChatPromptTemplate.from_template(IMPROVED_FTI_GENERATION_TEMPLATE)
        response = hub.push('fti-generation-template-v2', fti_template)
        uploaded['fti_generation'] = 'fti-generation-template-v2'
        logger.info(f"✅ FTI Generation: {response}")
        
    except Exception as e:
        logger.error(f"❌ Universal templates failed: {e}")
        failed.append(f"Universal templates: {e}")
    
    # Summary
    logger.info("🎉 Upload Summary")
    logger.info("=" * 40)
    logger.info(f"✅ Uploaded: {len(uploaded)} templates")
    logger.info(f"❌ Failed: {len(failed)} templates")
    
    if uploaded:
        logger.info("📋 Successful uploads:")
        for key, hub_id in uploaded.items():
            logger.info(f"   • {key} → {hub_id}")
    
    if failed:
        logger.info("❌ Failed uploads:")
        for failure in failed:
            logger.info(f"   • {failure}")
    
    return uploaded, failed

if __name__ == "__main__":
    # Check API key
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.error("❌ LANGCHAIN_API_KEY not set!")
        sys.exit(1)
    
    # Run upload
    asyncio.run(upload_all_templates()) 