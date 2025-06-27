
# ===== PRODUCTION CODE: LangChain Hub Integration =====
# Replace the _select_optimal_template method with this code:

async def _select_optimal_template(self, inputs: Dict[str, Any]) -> ChatPromptTemplate:
    """✅ LangChain Hub Integration - Using community-tested prompts"""
    from langchain import hub
    
    # ✅ Simple selection logic
    query = inputs.get("question", "").lower()
    
    # Map query types to hub IDs
    hub_mappings = {
        "casino_review": "peter-rag/casino-review-template",  # When "casino" in query and ("review" in query or "analysis" in query)
        "game_guide": "peter-rag/game-guide-template",  # When "game" in query and ("guide" in query or "how to" in query)
        "comparison": "peter-rag/comparison-template",  # When any(word in query for word in ["vs", "versus", "compare", "comparison"])
        "default": "peter-rag/default-template",  # When True  # Default case
    }
    
    # Determine template type
    if "casino" in query and ("review" in query or "analysis" in query):
        template_key = "casino_review"
    elif "game" in query and ("guide" in query or "how to" in query):
        template_key = "game_guide"
    elif any(word in query for word in ["vs", "versus", "compare", "comparison"]):
        template_key = "comparison"
    else:
        template_key = "default"
    
    # ✅ Pull from LangChain Hub
    hub_id = hub_mappings.get(template_key, hub_mappings["default"])
    
    try:
        template = hub.pull(hub_id)
        logging.info(f"✅ Using {template_key} template from LangChain Hub (ID: {hub_id})")
        return template
    except Exception as e:
        logging.error(f"❌ Hub pull failed for {hub_id}: {e}")
        # Fallback to basic template
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_template(
            "Based on the context: {context}\n\nAnswer the question: {question}"
        )
