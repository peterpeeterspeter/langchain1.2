#!/usr/bin/env python3
"""
Upload ALL Template System v2.0 Templates to LangChain Hub
========================================================

This script uploads our complete 32+ template system including:
- 8 Query Types × 4 Expertise Levels = 32 specialized templates
- Plus universal and FTI templates

🔧 Setup Required:
LANGCHAIN_API_KEY must be set (already configured)

📋 Templates to Upload:
✅ Casino Review Templates (4 expertise levels)
✅ Game Guide Templates (4 expertise levels)  
✅ Promotion Analysis Templates (4 expertise levels)
✅ Comparison Templates (4 expertise levels)
✅ News Update Templates (4 expertise levels)
✅ General Info Templates (4 expertise levels)
✅ Troubleshooting Templates (4 expertise levels)
✅ Regulatory Templates (4 expertise levels)
✅ Universal RAG Template
✅ FTI Generation Template

Total: 34+ professional templates for LangChain community!
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any, List
from pathlib import Path

# Import native LangChain hub functions
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# Import our comprehensive template system
sys.path.append(str(Path(__file__).parent.parent))
from src.templates.improved_template_manager import (
    ImprovedTemplateManager, QueryType, ExpertiseLevel,
    IMPROVED_UNIVERSAL_RAG_TEMPLATE, IMPROVED_FTI_GENERATION_TEMPLATE
)
from src.chains.advanced_prompt_system import DomainSpecificPrompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveHubUploader:
    """Upload all Template System v2.0 templates to LangChain Hub"""
    
    def __init__(self):
        self.template_manager = ImprovedTemplateManager()
        self.domain_prompts = DomainSpecificPrompts()
        self.uploaded_templates = {}
        self.failed_uploads = []
        
    def create_template_from_string(self, template_string: str) -> ChatPromptTemplate:
        """Convert template string to ChatPromptTemplate"""
        return ChatPromptTemplate.from_template(template_string)
    
    async def upload_single_template(self, hub_id: str, template: ChatPromptTemplate, metadata: Dict[str, Any]) -> bool:
        """Upload a single template to LangChain Hub"""
        
        logger.info(f"🚀 Uploading to LangChain Hub...")
        logger.info(f"   Hub ID: {hub_id}")
        logger.info(f"   Name: {metadata.get('name', 'N/A')}")
        logger.info(f"   Description: {metadata.get('description', 'N/A')}")
        
        try:
            # ✅ Use native LangChain hub.push() 
            response = hub.push(hub_id, template)
            logger.info(f"✅ Successfully uploaded {hub_id}")
            logger.info(f"   Response: {response}")
            logger.info(f"   Available via: hub.pull('{hub_id}')")
            logger.info("-" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed for {hub_id}: {e}")
            self.failed_uploads.append({
                'hub_id': hub_id,
                'error': str(e),
                'metadata': metadata
            })
            return False
    
    async def upload_domain_specific_templates(self) -> Dict[str, str]:
        """Upload all 32 domain-specific templates (8 query types × 4 expertise levels)"""
        
        logger.info("📋 Uploading 32 Domain-Specific Templates...")
        logger.info("=" * 80)
        
        uploaded = {}
        
        # Get all query types and expertise levels
        query_types = list(QueryType)
        expertise_levels = list(ExpertiseLevel)
        
        for query_type in query_types:
            for expertise_level in expertise_levels:
                try:
                    # Get template string from domain prompts
                    template_string = self.domain_prompts.get_prompt(query_type, expertise_level)
                    
                    # Create ChatPromptTemplate
                    template = self.create_template_from_string(template_string)
                    
                    # Create hub ID
                    hub_id = f"{query_type.value}-{expertise_level.value}-template"
                    
                    # Create metadata
                    metadata = {
                        'name': f"{query_type.value.replace('_', ' ').title()} - {expertise_level.value.title()} Level",
                        'description': f"Professional {query_type.value} template for {expertise_level.value} level content creation",
                        'tags': [query_type.value, expertise_level.value, 'content-creation', 'professional', 'rag'],
                        'query_type': query_type.value,
                        'expertise_level': expertise_level.value,
                        'word_count': self._get_word_count_for_level(expertise_level),
                        'version': '2.0'
                    }
                    
                    # Upload template
                    success = await self.upload_single_template(hub_id, template, metadata)
                    if success:
                        uploaded[f"{query_type.value}_{expertise_level.value}"] = hub_id
                        
                except Exception as e:
                    logger.error(f"❌ Failed to process {query_type.value}_{expertise_level.value}: {e}")
                    
        return uploaded
    
    async def upload_universal_templates(self) -> Dict[str, str]:
        """Upload universal and FTI templates"""
        
        logger.info("📋 Uploading Universal Templates...")
        logger.info("=" * 60)
        
        uploaded = {}
        
        # 1. Universal RAG Template
        try:
            universal_template = self.create_template_from_string(IMPROVED_UNIVERSAL_RAG_TEMPLATE)
            universal_metadata = {
                'name': 'Universal RAG Template v2.0',
                'description': 'Enhanced universal template for high-quality content generation with SEO optimization',
                'tags': ['universal', 'rag', 'content-creation', 'seo', 'professional'],
                'version': '2.0',
                'word_count': '800-2000'
            }
            
            success = await self.upload_single_template('universal-rag-template-v2', universal_template, universal_metadata)
            if success:
                uploaded['universal_rag'] = 'universal-rag-template-v2'
                
        except Exception as e:
            logger.error(f"❌ Failed to upload universal RAG template: {e}")
        
        # 2. FTI Generation Template
        try:
            fti_template = self.create_template_from_string(IMPROVED_FTI_GENERATION_TEMPLATE)
            fti_metadata = {
                'name': 'FTI Content Generation Template v2.0',
                'description': 'Advanced template for Feature-Training-Inference content pipeline with engagement optimization',
                'tags': ['fti', 'content-generation', 'engagement', 'professional', 'pipeline'],
                'version': '2.0',
                'word_count': '1200-1800'
            }
            
            success = await self.upload_single_template('fti-generation-template-v2', fti_template, fti_metadata)
            if success:
                uploaded['fti_generation'] = 'fti-generation-template-v2'
                
        except Exception as e:
            logger.error(f"❌ Failed to upload FTI template: {e}")
            
        return uploaded
    
    def _get_word_count_for_level(self, expertise_level: ExpertiseLevel) -> str:
        """Get word count requirements for expertise level"""
        word_counts = {
            ExpertiseLevel.BEGINNER: "800+",
            ExpertiseLevel.INTERMEDIATE: "1200+", 
            ExpertiseLevel.ADVANCED: "1500+",
            ExpertiseLevel.EXPERT: "2000+"
        }
        return word_counts.get(expertise_level, "800+")
    
    async def upload_all_templates(self) -> Dict[str, Any]:
        """Upload all Template System v2.0 templates"""
        
        logger.info("🎯 Comprehensive Template System v2.0 Upload")
        logger.info("=" * 80)
        logger.info("📊 Templates to upload:")
        logger.info("   • 8 Query Types × 4 Expertise Levels = 32 specialized templates")
        logger.info("   • 2 Universal templates (RAG + FTI)")
        logger.info("   • Total: 34+ professional templates")
        logger.info("=" * 80)
        
        # Upload domain-specific templates (32 templates)
        domain_uploaded = await self.upload_domain_specific_templates()
        
        # Upload universal templates (2 templates)
        universal_uploaded = await self.upload_universal_templates()
        
        # Combine results
        all_uploaded = {**domain_uploaded, **universal_uploaded}
        
        return {
            'uploaded_templates': all_uploaded,
            'total_uploaded': len(all_uploaded),
            'failed_uploads': self.failed_uploads,
            'total_failed': len(self.failed_uploads)
        }
    
    def generate_comprehensive_production_code(self, uploaded_templates: Dict[str, str]):
        """Generate production code for all uploaded templates"""
        
        production_code = '''
"""
Production LangChain Hub Integration - Complete Template System v2.0
====================================================================

This code provides access to all 34+ uploaded templates from LangChain Hub.
Each template is professionally crafted for specific content types and expertise levels.

Usage Examples:
- hub.pull('casino-review-intermediate-template')
- hub.pull('game-guide-beginner-template') 
- hub.pull('universal-rag-template-v2')
"""

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional
import logging

class TemplateSystemV2Hub:
    """Access to complete Template System v2.0 via LangChain Hub"""
    
    def __init__(self):
        # Domain-specific template mappings
        self.domain_templates = {
'''
        
        # Add all domain template mappings
        for template_key, hub_id in uploaded_templates.items():
            if template_key not in ['universal_rag', 'fti_generation']:
                production_code += f'            "{template_key}": "{hub_id}",\n'
        
        production_code += '''        }
        
        # Universal template mappings
        self.universal_templates = {
'''
        
        # Add universal template mappings
        if 'universal_rag' in uploaded_templates:
            production_code += f'            "universal_rag": "{uploaded_templates["universal_rag"]}",\n'
        if 'fti_generation' in uploaded_templates:
            production_code += f'            "fti_generation": "{uploaded_templates["fti_generation"]}",\n'
        
        production_code += '''        }
    
    async def get_template(self, query_type: str, expertise_level: str = "intermediate") -> ChatPromptTemplate:
        """Get template from LangChain Hub based on query type and expertise level"""
        
        # Try domain-specific template first
        template_key = f"{query_type}_{expertise_level}"
        hub_id = self.domain_templates.get(template_key)
        
        if not hub_id:
            # Fallback to universal template
            hub_id = self.universal_templates.get("universal_rag")
        
        if not hub_id:
            raise ValueError(f"No template found for {query_type}_{expertise_level}")
        
        try:
            template = hub.pull(hub_id)
            logging.info(f"✅ Using {template_key} from LangChain Hub (ID: {hub_id})")
            return template
        except Exception as e:
            logging.error(f"❌ Hub pull failed for {hub_id}: {e}")
            # Fallback to basic template
            return ChatPromptTemplate.from_template(
                "Based on the context: {context}\\n\\nAnswer the question: {question}"
            )
    
    def get_available_templates(self) -> Dict[str, Any]:
        """Get list of all available templates"""
        return {
            "domain_templates": list(self.domain_templates.keys()),
            "universal_templates": list(self.universal_templates.keys()),
            "total_count": len(self.domain_templates) + len(self.universal_templates)
        }

# Example usage:
# template_hub = TemplateSystemV2Hub()
# template = await template_hub.get_template("casino_review", "intermediate")
# available = template_hub.get_available_templates()
'''

        # Save production code
        output_file = Path(__file__).parent / "complete_hub_integration.py"
        with open(output_file, "w") as f:
            f.write(production_code)
        
        logger.info(f"📝 Complete production code saved to {output_file}")
        return production_code

async def main():
    """Main upload process for all Template System v2.0 templates"""
    
    # Check API key
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.error("❌ LANGCHAIN_API_KEY not found!")
        logger.info("🔧 Please set: export LANGCHAIN_API_KEY='your_api_key'")
        sys.exit(1)
    
    logger.info(f"✅ LANGCHAIN_API_KEY found (length: {len(api_key)})")
    
    try:
        # Initialize uploader
        uploader = ComprehensiveHubUploader()
        
        # Upload all templates
        results = await uploader.upload_all_templates()
        
        # Show summary
        logger.info("🎉 Upload Complete!")
        logger.info("=" * 80)
        logger.info(f"✅ Successfully uploaded: {results['total_uploaded']} templates")
        logger.info(f"❌ Failed uploads: {results['total_failed']} templates")
        
        if results['uploaded_templates']:
            logger.info("📋 Uploaded Templates:")
            for template_key, hub_id in results['uploaded_templates'].items():
                logger.info(f"   ✅ {template_key} → {hub_id}")
        
        if results['failed_uploads']:
            logger.info("❌ Failed Uploads:")
            for failure in results['failed_uploads']:
                logger.info(f"   ❌ {failure['hub_id']}: {failure['error']}")
        
        # Generate production code
        uploader.generate_comprehensive_production_code(results['uploaded_templates'])
        
        logger.info("🔄 Next Steps:")
        logger.info("   1. Update universal_rag_lcel.py with complete hub integration")
        logger.info("   2. Test all template types in production")
        logger.info("   3. Share templates with LangChain community!")
        logger.info("   4. Document template usage examples")
        
    except Exception as e:
        logger.error(f"❌ Upload process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 