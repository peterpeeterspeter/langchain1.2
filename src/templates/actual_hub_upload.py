#!/usr/bin/env python3
"""
Actual LangChain Hub Upload Script - Template System v2.0
========================================================

This script uploads our production-ready templates to the official LangChain Hub.
Follow the setup instructions below before running.

🔧 Setup Required:
1. Create account at https://smith.langchain.com/
2. Generate API key in account settings
3. Set environment variable: export LANGCHAIN_API_KEY="your_api_key"
4. Run this script: python actual_hub_upload.py

📋 Templates to Upload:
- Casino Review Template (95-field casino intelligence)
- Gaming Guide Template (step-by-step tutorials)  
- Comparison Template (detailed analysis)
- Default RAG Template (general purpose)

🎯 After Upload:
Your templates will be available via:
hub.pull("your-username/casino-review-template")
hub.pull("your-username/game-guide-template")
hub.pull("your-username/comparison-template")
hub.pull("your-username/default-rag-template")
"""

import os
import sys
import logging
from typing import Dict, Any
from pathlib import Path

# Import native LangChain hub functions
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# Import our templates
sys.path.append(str(Path(__file__).parent))
from langchain_hub_templates import get_langchain_hub_templates, HUB_METADATA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_api_key() -> bool:
    """Check if LANGCHAIN_API_KEY is properly configured"""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.error("❌ LANGCHAIN_API_KEY not found!")
        logger.info("🔧 Setup Instructions:")
        logger.info("   1. Visit https://smith.langchain.com/")
        logger.info("   2. Create account and generate API key")
        logger.info("   3. Run: export LANGCHAIN_API_KEY='your_api_key'")
        logger.info("   4. Re-run this script")
        return False
    
    logger.info(f"✅ LANGCHAIN_API_KEY found (length: {len(api_key)})")
    return True

async def upload_template_to_hub(
    template_name: str, 
    template: ChatPromptTemplate,
    username: str = None
) -> str:
    """Upload a single template to LangChain Hub"""
    
    # Format hub ID - try without tenant first for personal account
    if username:
        hub_id = f"{username}/{template_name.replace('_', '-')}-template"
    else:
        # Upload to personal space without tenant prefix
        hub_id = f"{template_name.replace('_', '-')}-template"
    
    logger.info(f"🚀 Uploading {template_name} to LangChain Hub...")
    logger.info(f"   Hub ID: {hub_id}")
    logger.info(f"   Template: {template}")
    
    try:
        # ✅ Use native LangChain hub.push() 
        response = hub.push(hub_id, template)
        logger.info(f"✅ Successfully uploaded {template_name}")
        logger.info(f"   Response: {response}")
        logger.info(f"   Available via: hub.pull('{hub_id}')")
        
        return hub_id
        
    except Exception as e:
        logger.error(f"❌ Upload failed for {template_name}: {e}")
        logger.info(f"💡 Troubleshooting:")
        logger.info(f"   - Verify API key is valid")
        logger.info(f"   - Check network connection")
        logger.info(f"   - Ensure template format is correct")
        return None

async def upload_all_templates(username: str = None) -> Dict[str, str]:
    """Upload all Template System v2.0 templates to LangChain Hub"""
    
    logger.info("📋 Starting Template System v2.0 Hub Upload...")
    logger.info("=" * 60)
    
    # Get our production templates
    templates = get_langchain_hub_templates()
    uploaded = {}
    
    for template_name, template in templates.items():
        hub_id = await upload_template_to_hub(template_name, template, username)
        if hub_id:
            uploaded[template_name] = hub_id
            
            # Show metadata
            metadata = HUB_METADATA.get(template_name, {})
            logger.info(f"📊 Template metadata:")
            logger.info(f"   Name: {metadata.get('name', 'N/A')}")
            logger.info(f"   Description: {metadata.get('description', 'N/A')}")
            logger.info(f"   Tags: {', '.join(metadata.get('tags', []))}")
            logger.info("-" * 40)
    
    return uploaded

def generate_production_code(uploaded_templates: Dict[str, str], username: str = "peter-rag"):
    """Generate production code that uses the uploaded hub templates"""
    
    hub_mappings = {
        "casino_review": uploaded_templates.get("casino_review", f"{username}/casino-review-template"),
        "game_guide": uploaded_templates.get("game_guide", f"{username}/game-guide-template"), 
        "comparison": uploaded_templates.get("comparison", f"{username}/comparison-template"),
        "default": uploaded_templates.get("default", f"{username}/default-rag-template")
    }
    
    production_code = f'''
async def _select_optimal_template(self, inputs: Dict[str, Any]) -> ChatPromptTemplate:
    """✅ Production LangChain Hub Integration - Using uploaded Template System v2.0"""
    from langchain import hub
    
    # ✅ Simple selection logic based on query analysis
    query = inputs.get("question", "").lower()
    
    if "casino" in query or "gambling" in query or "review" in query:
        template_key = "casino_review"
    elif "guide" in query or "tutorial" in query or "how to" in query:
        template_key = "game_guide"
    elif "compare" in query or "comparison" in query or "vs" in query:
        template_key = "comparison"
    else:
        template_key = "default"
    
    # ✅ Hub mappings for uploaded Template System v2.0
    hub_mappings = {hub_mappings}
    
    hub_id = hub_mappings.get(template_key, hub_mappings["default"])
    
    try:
        # ✅ Pull from LangChain Hub using native API
        template = hub.pull(hub_id)
        logging.info(f"✅ Using {{template_key}} template from LangChain Hub (ID: {{hub_id}})")
        return template
    except Exception as hub_error:
        logging.warning(f"⚠️ Hub pull failed for {{hub_id}}: {{hub_error}}")
        # Fallback to basic template
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_template(
            "Based on the context: {{context}}\\n\\nAnswer the question: {{question}}"
        )
'''
    
    # Save production code
    output_file = Path(__file__).parent / "production_hub_code.py"
    with open(output_file, "w") as f:
        f.write(production_code)
    
    logger.info(f"📝 Production code saved to {output_file}")
    return production_code

async def main():
    """Main upload process"""
    
    logger.info("🎯 LangChain Hub Upload - Template System v2.0")
    logger.info("=" * 60)
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    try:
        # Upload all templates
        uploaded = await upload_all_templates()
        
        if uploaded:
            logger.info("🎉 Upload Summary:")
            for template_name, hub_id in uploaded.items():
                logger.info(f"   ✅ {template_name} → {hub_id}")
                
            # Generate production code
            generate_production_code(uploaded)
            
            logger.info("🔄 Next Steps:")
            logger.info("   1. Update universal_rag_lcel.py with new hub IDs")
            logger.info("   2. Test hub.pull() calls in production")
            logger.info("   3. Remove local template fallbacks")
            
        else:
            logger.error("❌ No templates uploaded successfully")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Upload process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 