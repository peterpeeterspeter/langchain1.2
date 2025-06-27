
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
    hub_mappings = {'casino_review': 'casino-review-template', 'game_guide': 'game-guide-template', 'comparison': 'comparison-template', 'default': 'default-template'}
    
    hub_id = hub_mappings.get(template_key, hub_mappings["default"])
    
    try:
        # ✅ Pull from LangChain Hub using native API
        template = hub.pull(hub_id)
        logging.info(f"✅ Using {template_key} template from LangChain Hub (ID: {hub_id})")
        return template
    except Exception as hub_error:
        logging.warning(f"⚠️ Hub pull failed for {hub_id}: {hub_error}")
        # Fallback to basic template
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_template(
            "Based on the context: {context}\n\nAnswer the question: {question}"
        )
