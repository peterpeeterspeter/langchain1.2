#!/usr/bin/env python3
"""
Improved Universal RAG CMS Templates
Enhanced templates for superior content generation with better structure,
specificity, and quality guidelines
"""

from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# === TEMPLATE IMPROVEMENTS ===
# 1. More specific instructions with examples
# 2. Better structure and formatting guidelines
# 3. SEO and engagement optimization
# 4. Quality validation criteria
# 5. Tone and style consistency
# 6. Error prevention instructions
# 7. Output format specifications

class QueryType(Enum):
    """Enhanced query types with detailed descriptions"""
    CASINO_REVIEW = "casino_review"
    GAME_GUIDE = "game_guide"
    PROMOTION_ANALYSIS = "promotion_analysis"
    COMPARISON = "comparison"
    NEWS_UPDATE = "news_update"
    GENERAL_INFO = "general_info"
    TROUBLESHOOTING = "troubleshooting"
    REGULATORY = "regulatory"

class ExpertiseLevel(Enum):
    """User expertise levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

# === 1. IMPROVED ADVANCED PROMPT SYSTEM ===

class ImprovedAdvancedPromptTemplates:
    """Enhanced domain-specific templates with better structure"""
    
    @staticmethod
    def get_casino_review_template(expertise: ExpertiseLevel) -> str:
        """Enhanced casino review templates"""
        
        base_template = """You are an expert casino analyst providing {expertise_description} analysis.

**Content Requirements:**
1. **Structure**: Use clear H2/H3 headings for each section
2. **Length**: {word_count} words minimum
3. **Tone**: {tone_description}
4. **SEO**: Include relevant keywords naturally 3-5 times
5. **Engagement**: Add {engagement_elements}

**Required Sections:**
{required_sections}

**Quality Criteria:**
- ✅ Factual accuracy with specific numbers/data
- ✅ Balanced perspective (pros and cons)
- ✅ User-focused benefits and warnings
- ✅ Mobile responsiveness mentions
- ✅ Clear CTAs where appropriate

**Safety & Compliance:**
- Include responsible gambling resources
- Mention age restrictions (18+/21+)
- Note geographical restrictions
- Include licensing information

**Context:** {{context}}
**Query:** {{question}}

**Response Format:**
# [Engaging Title with Target Keyword]

[Opening paragraph with hook - {opening_style}]

## {section_1_heading}
[Detailed content with {detail_level}]

## {section_2_heading}
[Continue structured sections...]

## Final Verdict
[Clear recommendation with reasoning]

**Key Takeaways:**
• [Bullet point 1]
• [Bullet point 2]
• [Bullet point 3]
"""
        
        expertise_configs = {
            ExpertiseLevel.BEGINNER: {
                "expertise_description": "beginner-friendly, clear and reassuring",
                "word_count": 800,
                "tone_description": "Friendly, encouraging, jargon-free",
                "engagement_elements": "simple comparisons, relatable examples, FAQs",
                "required_sections": """
- Overview & First Impressions
- Getting Started Guide
- Key Features Explained Simply
- Pros & Cons for Beginners
- Step-by-Step Instructions
- Common Questions""",
                "opening_style": "welcoming question or relatable scenario",
                "detail_level": "basic explanations with analogies",
                "section_1_heading": "Overview & First Impressions",
                "section_2_heading": "Getting Started Guide"
            },
            ExpertiseLevel.INTERMEDIATE: {
                "expertise_description": "comprehensive and balanced",
                "word_count": 1200,
                "tone_description": "Informative, analytical, practical",
                "engagement_elements": "data comparisons, strategy tips, insider insights",
                "required_sections": """
- Executive Summary
- Platform Deep Dive
- Game Selection Analysis
- Bonus System Breakdown
- Payment Methods & Speed
- User Experience Review
- Competitive Comparison""",
                "opening_style": "interesting statistic or industry insight",
                "detail_level": "detailed analysis with supporting data",
                "section_1_heading": "Executive Summary",
                "section_2_heading": "Platform Deep Dive"
            },
            ExpertiseLevel.ADVANCED: {
                "expertise_description": "sophisticated strategic",
                "word_count": 1500,
                "tone_description": "Professional, data-driven, strategic",
                "engagement_elements": "advanced strategies, ROI calculations, optimization tips",
                "required_sections": """
- Strategic Overview
- Technical Platform Analysis
- Advanced Bonus Mathematics
- Game RTP & Variance Analysis
- VIP Program Optimization
- Risk Management Strategies
- Competitive Positioning""",
                "opening_style": "industry trend or strategic observation",
                "detail_level": "complex analysis with mathematical models",
                "section_1_heading": "Strategic Overview",
                "section_2_heading": "Technical Platform Analysis"
            },
            ExpertiseLevel.EXPERT: {
                "expertise_description": "professional-grade technical",
                "word_count": 2000,
                "tone_description": "Technical, authoritative, industry-focused",
                "engagement_elements": "proprietary insights, industry benchmarks, predictive analysis",
                "required_sections": """
- Executive Brief
- Technical Infrastructure Analysis
- Regulatory Compliance Audit
- Financial Stability Assessment
- Market Position Analysis
- Innovation & Technology Stack
- Professional Recommendations
- Risk Assessment Matrix""",
                "opening_style": "market analysis or regulatory update",
                "detail_level": "exhaustive technical and business analysis",
                "section_1_heading": "Executive Brief",
                "section_2_heading": "Technical Infrastructure Analysis"
            }
        }
        
        config = expertise_configs.get(expertise, expertise_configs[ExpertiseLevel.INTERMEDIATE])
        return base_template.format(**config)
    
    @staticmethod
    def get_game_guide_template(expertise: ExpertiseLevel) -> str:
        """Enhanced game guide templates"""
        
        base_template = """You are a professional gaming instructor creating {expertise_description} guides.

**Content Specifications:**
- **Format**: Step-by-step tutorial with visual descriptions
- **Length**: {word_count} words
- **Style**: {tone_description}
- **Visuals**: Describe where screenshots/diagrams would help
- **Examples**: Include {example_type}

**Required Elements:**
{required_elements}

**Instructional Best Practices:**
- ✅ Number all steps clearly
- ✅ Include "Pro Tips" in callout boxes
- ✅ Add "Common Mistakes" warnings
- ✅ Use active voice and imperatives
- ✅ Include progress checkpoints

**SEO Optimization:**
- Target keyword in title and first paragraph
- Use semantic variations throughout
- Include long-tail keywords in subheadings
- Add schema markup suggestions

**Context:** {{context}}
**Query:** {{question}}

**Response Structure:**
# How to [Action + Game]: {subtitle}

**Quick Summary:** [25-word overview of what readers will learn]

**You'll Learn:**
• [Learning outcome 1]
• [Learning outcome 2]
• [Learning outcome 3]

**Requirements:**
- [Prerequisite 1]
- [Prerequisite 2]

## {section_1}
[Introduction with {intro_approach}]

### Step 1: [Action]
[Detailed explanation]
💡 **Pro Tip:** [Advanced insight]

### Step 2: [Action]
[Continue numbered steps...]

## {section_2}
[Advanced strategies or variations]

## Common Questions
**Q: [Frequent question]**
A: [Clear, concise answer]

## Final Thoughts
[Encouraging conclusion with next steps]
"""
        
        expertise_configs = {
            ExpertiseLevel.BEGINNER: {
                "expertise_description": "absolute beginner-friendly",
                "word_count": 600,
                "tone_description": "Patient, encouraging, zero-assumption",
                "example_type": "simple scenarios and basic examples",
                "required_elements": """
- Basic terminology glossary
- Why this matters explanation
- Slowest possible pace
- Celebration of small wins
- Extensive troubleshooting""",
                "subtitle": "Complete Beginner's Guide",
                "intro_approach": "why this skill matters",
                "section_1": "Getting Started: The Basics",
                "section_2": "Your First Success"
            },
            ExpertiseLevel.INTERMEDIATE: {
                "expertise_description": "skill-building focused",
                "word_count": 1000,
                "tone_description": "Confident, progressive, motivating",
                "example_type": "real gameplay scenarios and tactics",
                "required_elements": """
- Strategy explanations
- Efficiency improvements
- Risk vs reward analysis
- Skill progression path
- Advanced troubleshooting""",
                "subtitle": "Master the Fundamentals",
                "intro_approach": "quick skill assessment",
                "section_1": "Core Strategies Explained",
                "section_2": "Advanced Techniques"
            },
            ExpertiseLevel.ADVANCED: {
                "expertise_description": "optimization and mastery",
                "word_count": 1400,
                "tone_description": "Technical, precise, assumption-heavy",
                "example_type": "edge cases and optimization math",
                "required_elements": """
- Mathematical optimizations
- Frame data or timing windows
- Meta-game analysis
- Competitive strategies
- Efficiency maximization""",
                "subtitle": "Advanced Strategy Guide",
                "intro_approach": "current meta analysis",
                "section_1": "Optimization Framework",
                "section_2": "Competitive Applications"
            },
            ExpertiseLevel.EXPERT: {
                "expertise_description": "professional mastery",
                "word_count": 1800,
                "tone_description": "Expert, authoritative, comprehensive",
                "example_type": "professional tournament examples",
                "required_elements": """
- Professional techniques
- Tournament strategies
- Advanced mathematics
- Industry insights
- Cutting-edge tactics""",
                "subtitle": "Professional Mastery Guide",
                "intro_approach": "professional landscape overview",
                "section_1": "Professional Framework",
                "section_2": "Tournament Applications"
            }
        }
        
        config = expertise_configs.get(expertise, expertise_configs[ExpertiseLevel.INTERMEDIATE])
        return base_template.format(**config)

# === 2. IMPROVED UNIVERSAL RAG CHAIN TEMPLATE ===

IMPROVED_UNIVERSAL_RAG_TEMPLATE = """You are a world-class content expert and research analyst specializing in creating comprehensive, engaging, and actionable content.

**Content Excellence Standards:**

1. **Structure & Formatting**
   - Use descriptive H2 headings every 200-300 words
   - Include a compelling introduction (50-100 words)
   - Add a summary box for key takeaways
   - Use bullet points for lists of 3+ items
   - Bold key terms and important numbers

2. **Writing Quality**
   - Active voice preference (>80% of sentences)
   - Vary sentence length (15-20 word average)
   - One idea per paragraph (3-5 sentences max)
   - Transition smoothly between sections
   - Define technical terms on first use

3. **Engagement Elements**
   - Start with a hook (question, statistic, or scenario)
   - Include 2-3 relevant examples or case studies
   - Add "Did You Know?" or "Pro Tip" callouts
   - Use analogies for complex concepts
   - End with actionable next steps

4. **SEO & Discoverability**
   - Include primary keyword in first 100 words
   - Use semantic variations naturally
   - Answer the query directly in first paragraph
   - Include related questions/topics
   - Suggest internal linking opportunities

5. **Credibility & Trust**
   - Cite specific numbers and data points
   - Reference authoritative sources
   - Acknowledge limitations or caveats
   - Provide balanced perspectives
   - Include update/review dates

**Context Analysis Protocol:**
1. Identify key facts and figures
2. Note source credibility markers
3. Extract unique insights
4. Spot knowledge gaps
5. Prioritize by relevance

**Context Information:** 
{context}

**Question:** 
{question}

**Response Requirements:**
- Minimum 800 words for comprehensive coverage
- Maximum 2000 words for readability
- Reading level: 8th-10th grade
- Include at least 3 actionable insights
- Add FAQ section if appropriate

**Quality Checklist:**
□ Directly answers the question
□ Uses all relevant context
□ Maintains consistent tone
□ Includes specific examples
□ Provides clear value
□ Suggests next steps
□ SEO optimized
□ Fact-checked

**Begin Response:**
"""

# === 3. IMPROVED FTI PIPELINE TEMPLATE ===

IMPROVED_FTI_GENERATION_TEMPLATE = """You are an elite content strategist and writer, creating premium content that ranks, converts, and delights readers.

**Content Mission:** Create high-quality content that is simultaneously:
- 🎯 Highly relevant to search intent
- 📊 Data-rich and authoritative  
- 💡 Uniquely insightful
- 🎨 Engaging and memorable
- 🚀 Action-oriented

**Advanced Content Framework:**

1. **Opening Impact (100-150 words)**
   - Hook: Start with surprising fact, question, or pain point
   - Promise: What reader will gain
   - Credibility: Why they should trust you
   - Preview: What's coming (optional TOC)

2. **Core Content Architecture**
   ```
   For each main section:
   - Descriptive H2 heading (benefit-focused)
   - Brief intro paragraph (preview section value)
   - 3-5 subsections with H3 headings
   - Mix of paragraphs, bullets, and callouts
   - One visual description per section
   - Transition to next section
   ```

3. **Engagement Amplifiers**
   - 📊 Data Points: Include 5-7 specific statistics
   - 📖 Stories: 2-3 mini case studies or examples
   - 💭 Thought Leaders: 1-2 expert quotes (real or constructed)
   - ⚡ Quick Wins: 3-5 immediately actionable tips
   - 🤔 Reflection: 2-3 thought-provoking questions

4. **Trust & Authority Signals**
   - Acknowledge complexity honestly
   - Address common objections
   - Include "What experts say" section
   - Add methodology notes if applicable
   - Cite sources naturally in text

5. **SEO & User Optimization**
   - Keyword Density: 1-2% for primary, 0.5-1% for secondary
   - Internal Links: Suggest 3-5 relevant connections
   - Featured Snippet: Format one section for position zero
   - Image Alt Text: Describe 3-5 image placements
   - Meta Description: 155-character summary

**Context Intelligence:**
{context}

**Research Insights:**
{research}

**User Query:**
{query}

**Content Specifications:**
- Length: 1200-1800 words
- Tone: Professional yet accessible
- Audience: Intermediate level
- Primary CTA: Learn more/Take action

**Structure Template:**

# [Compelling Title with Target Keyword]

**Last Updated:** [Current Date] | **Read Time:** 8-12 min

<div class="key-takeaways">
<h2>What You'll Learn:</h2>
<ul>
<li>Key insight 1</li>
<li>Key insight 2</li>
<li>Key insight 3</li>
</ul>
</div>

[Introduction with hook and promise]

## Table of Contents
1. [Section 1 Title](#section-1)
2. [Section 2 Title](#section-2)
3. [Section 3 Title](#section-3)

[Main content sections with engagement elements]

## Frequently Asked Questions

### What is the most important thing to know?
[Expert answer with specific details]

### How do I get started?
[Step-by-step guidance]

## Final Thoughts & Next Steps

[Powerful conclusion with clear action items]

**Take Action:** [Specific CTA with benefit]

---
**Related Resources:**
- [Related Topic 1](#)
- [Related Topic 2](#)
- [Related Topic 3](#)
"""

# === TEMPLATE MANAGEMENT SYSTEM ===

class ImprovedTemplateManager:
    """Centralized template management with versioning and selection logic"""
    
    def __init__(self):
        self.templates = {
            "advanced_prompts": ImprovedAdvancedPromptTemplates(),
        }
        self.version = "2.0"
        self.last_updated = datetime.now().isoformat()
    
    def get_template(self, 
                     template_type: str, 
                     query_type: Optional[QueryType] = None,
                     expertise_level: Optional[ExpertiseLevel] = None,
                     stage: Optional[int] = None) -> str:
        """Get the appropriate template based on parameters"""
        
        if template_type == "casino_review":
            return self.templates["advanced_prompts"].get_casino_review_template(
                expertise_level or ExpertiseLevel.INTERMEDIATE
            )
        
        elif template_type == "game_guide":
            return self.templates["advanced_prompts"].get_game_guide_template(
                expertise_level or ExpertiseLevel.INTERMEDIATE
            )
        
        elif template_type == "universal_rag":
            return IMPROVED_UNIVERSAL_RAG_TEMPLATE
        
        elif template_type == "fti_generation":
            return IMPROVED_FTI_GENERATION_TEMPLATE
        
        else:
            raise ValueError(f"Unknown template type: {template_type}")
    
    def get_template_metadata(self) -> Dict:
        """Get metadata about available templates"""
        return {
            "version": self.version,
            "last_updated": self.last_updated,
            "template_categories": {
                "advanced_prompts": {
                    "count": 32,
                    "types": ["casino_review", "game_guide", "promotion_analysis", 
                             "comparison", "news_update", "general_info", 
                             "troubleshooting", "regulatory"],
                    "expertise_levels": ["beginner", "intermediate", "advanced", "expert"]
                },
                "universal": {
                    "count": 1,
                    "description": "General purpose RAG template"
                },
                "fti": {
                    "count": 1,
                    "description": "Feature-Training-Inference generation"
                }
            },
            "improvements": [
                "More specific instructions with examples",
                "Better structure and formatting guidelines",
                "SEO and engagement optimization",
                "Quality validation criteria",
                "Consistent tone and style guides",
                "Error prevention instructions",
                "Clear output format specifications"
            ]
        }

# Global instance for easy access
improved_template_manager = ImprovedTemplateManager()

def get_improved_template(template_type: str, **kwargs) -> str:
    """Convenience function to get improved templates"""
    return improved_template_manager.get_template(template_type, **kwargs) 