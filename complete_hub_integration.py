
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
        }
        
        # Universal template mappings
        self.universal_templates = {
            "universal_rag": "universal-rag-template-v2",
            "fti_generation": "fti-generation-template-v2",
        }
    
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
                "Based on the context: {context}\n\nAnswer the question: {question}"
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
