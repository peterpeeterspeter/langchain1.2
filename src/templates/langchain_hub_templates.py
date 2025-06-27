#!/usr/bin/env python3
"""
LangChain Hub Templates - Production-Ready Prompts for Casino Intelligence System

These templates are designed for import to LangChain Hub and use by the community.
Each template follows LangChain Hub best practices and includes comprehensive testing.

🎯 Templates Included:
- casino_review: Comprehensive casino analysis and review generation
- game_guide: Step-by-step gaming guides and tutorials  
- comparison: Detailed casino/game comparison analysis
- default: General purpose RAG template

📋 Hub Import Instructions:
1. Save each template as separate .yaml files
2. Use `langchain hub push` to upload to LangChain Hub
3. Replace local_hub with hub.pull() calls in production code
"""

from langchain_core.prompts import ChatPromptTemplate

# ===== CASINO REVIEW TEMPLATE =====
# Hub ID: [your-username]/casino-review-template
CASINO_REVIEW_TEMPLATE = """You are an expert casino analyst providing comprehensive reviews using structured data.

Based on the comprehensive casino analysis data provided, create a detailed, structured review that leverages all available information.

Context: {context}
Query: {question}

## Content Structure:
1. **Executive Summary** - Key findings and overall rating
2. **Licensing & Trustworthiness** - License authority data, security
3. **Games & Software** - Counts, providers, live casino details
4. **Bonuses & Promotions** - Welcome bonuses, wagering requirements
5. **Payment Methods** - Deposit/withdrawal options and times
6. **User Experience** - Mobile app, customer support
7. **Final Assessment** - Ratings, recommendations, pros/cons

Instructions:
- Use specific data points from context when available
- Include quantitative metrics (game counts, RTP percentages, withdrawal times)
- Highlight unique features and competitive advantages
- Address security, licensing, and player protection measures
- Provide balanced assessment with both pros and cons
- Include final rating out of 10 with justification

Response:"""

# ===== GAME GUIDE TEMPLATE =====
# Hub ID: [your-username]/game-guide-template  
GAME_GUIDE_TEMPLATE = """You are an expert gaming guide creator focusing on clear, actionable instructions.

Create a comprehensive guide based on the provided context and question.

Context: {context}
Query: {question}

## Guide Structure:
1. **Quick Overview** - What players will learn
2. **Getting Started** - Basic requirements and setup
3. **Step-by-Step Instructions** - Detailed gameplay steps
4. **Advanced Tips** - Pro strategies and optimizations
5. **Common Questions** - FAQ section
6. **Next Steps** - What to do after mastering this

Instructions:
- Use clear, numbered steps for actionable content
- Include screenshots or visual references when mentioned in context
- Explain terminology for beginners
- Provide specific examples and scenarios
- Address common mistakes and how to avoid them
- Include success metrics and progression indicators

Response:"""

# ===== COMPARISON TEMPLATE =====
# Hub ID: [your-username]/comparison-template
COMPARISON_TEMPLATE = """You are an expert analyst specializing in detailed comparisons.

Compare the options based on the provided context and answer the comparison question thoroughly.

Context: {context}
Query: {question}

## Comparison Structure:
1. **Overview** - What's being compared
2. **Key Differences** - Main distinguishing factors
3. **Pros & Cons** - Advantages and disadvantages of each
4. **Performance Metrics** - Quantitative comparisons
5. **Use Cases** - Best scenarios for each option
6. **Recommendation** - Which is better for different needs

Instructions:
- Create side-by-side comparisons using tables when appropriate
- Use specific metrics and data points from context
- Address different user personas and their needs
- Highlight unique selling points for each option
- Provide clear decision-making framework
- Include final recommendation with reasoning

Response:"""

# ===== DEFAULT RAG TEMPLATE =====
# Hub ID: [your-username]/default-rag-template
DEFAULT_RAG_TEMPLATE = """Based on the following context, answer the question accurately and comprehensively.

Context: {context}
Question: {question}

Instructions:
- Provide accurate, fact-based answers using the context provided
- If context is insufficient, clearly state limitations
- Structure responses with clear headings when appropriate
- Include specific details and examples from the context
- Maintain professional, informative tone
- End with actionable next steps when relevant

Answer:"""

# ===== LANGCHAIN HUB READY TEMPLATES =====
def get_langchain_hub_templates():
    """
    Returns dictionary of ChatPromptTemplate objects ready for LangChain Hub
    
    Usage:
    templates = get_langchain_hub_templates()
    # Save each template to .yaml files for hub upload
    """
    return {
        "casino_review": ChatPromptTemplate.from_template(CASINO_REVIEW_TEMPLATE),
        "game_guide": ChatPromptTemplate.from_template(GAME_GUIDE_TEMPLATE),
        "comparison": ChatPromptTemplate.from_template(COMPARISON_TEMPLATE),
        "default": ChatPromptTemplate.from_template(DEFAULT_RAG_TEMPLATE)
    }

# ===== HUB METADATA =====
HUB_METADATA = {
    "casino_review": {
        "name": "Casino Review Template",
        "description": "Comprehensive casino analysis and review generation with 95-field intelligence",
        "tags": ["casino", "review", "gambling", "analysis", "structured"],
        "use_cases": ["Casino reviews", "Gambling site analysis", "Platform comparison"],
        "input_variables": ["context", "question"],
        "version": "1.0.0"
    },
    "game_guide": {
        "name": "Gaming Guide Template", 
        "description": "Step-by-step gaming guides and tutorials with clear instructions",
        "tags": ["gaming", "guide", "tutorial", "instructions", "education"],
        "use_cases": ["Game tutorials", "Strategy guides", "How-to content"],
        "input_variables": ["context", "question"],
        "version": "1.0.0"
    },
    "comparison": {
        "name": "Comparison Analysis Template",
        "description": "Detailed comparison analysis for multiple options with structured evaluation", 
        "tags": ["comparison", "analysis", "evaluation", "decision-making"],
        "use_cases": ["Product comparisons", "Service evaluations", "Option analysis"],
        "input_variables": ["context", "question"],
        "version": "1.0.0"
    },
    "default": {
        "name": "Default RAG Template",
        "description": "General purpose RAG template for accurate context-based answers",
        "tags": ["rag", "qa", "general", "context", "default"],
        "use_cases": ["General Q&A", "Information retrieval", "Context-based answers"],
        "input_variables": ["context", "question"],
        "version": "1.0.0"
    }
}

if __name__ == "__main__":
    """
    Export templates for LangChain Hub upload
    """
    import yaml
    from pathlib import Path
    
    # Create hub-ready directory
    hub_dir = Path("langchain_hub_export")
    hub_dir.mkdir(exist_ok=True)
    
    templates = get_langchain_hub_templates()
    
    for template_name, template_obj in templates.items():
        # Create hub-compatible YAML
        hub_content = {
            "template": template_obj.messages[0].prompt.template,  # Access template string correctly
            "input_variables": template_obj.input_variables,
            "metadata": HUB_METADATA[template_name]
        }
        
        # Save to YAML file
        output_file = hub_dir / f"{template_name}_template.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(hub_content, f, default_flow_style=False)
        
        print(f"✅ Exported {template_name} template to {output_file}")
    
    print(f"\n🚀 All templates exported to {hub_dir}/")
    print("📋 Next steps:")
    print("1. Upload each template to LangChain Hub using 'langchain hub push'")
    print("2. Update Universal RAG Chain to use hub.pull() calls")
    print("3. Test with production workloads") 