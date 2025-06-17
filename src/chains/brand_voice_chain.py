#!/usr/bin/env python3
"""
Brand Voice Management Chain - Native LangChain Implementation
Applies consistent brand voice across content using native LangChain patterns

✅ NATIVE LANGCHAIN PATTERNS:
- RunnablePassthrough for voice configuration flow
- RunnableLambda for voice adaptation logic
- RunnableBranch for voice type selection
- Composable with existing content generation chains

🎯 USAGE: content_data | brand_voice_chain → voice-adapted content
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ✅ PYDANTIC MODELS FOR BRAND VOICE

class BrandVoiceConfig(BaseModel):
    """Brand voice configuration"""
    voice_name: str = Field(description="Voice configuration name")
    tone: str = Field(description="Overall tone (professional, casual, authoritative, etc.)")
    personality_traits: List[str] = Field(default=[], description="Personality characteristics")
    vocabulary_style: str = Field(description="Vocabulary complexity (simple, moderate, advanced)")
    sentence_structure: str = Field(description="Sentence style (short, varied, complex)")
    content_approach: str = Field(description="Content approach (analytical, conversational, directive)")
    expertise_level: str = Field(description="Expertise demonstration (beginner, intermediate, expert)")
    call_to_action_style: str = Field(description="CTA style (soft, direct, urgent)")
    brand_values: List[str] = Field(default=[], description="Core brand values to reflect")
    target_audience: str = Field(description="Primary target audience")
    content_type_adaptations: Dict[str, Dict[str, str]] = Field(default={}, description="Type-specific adaptations")

class VoiceAdaptedContent(BaseModel):
    """Content adapted with brand voice"""
    original_content: str = Field(description="Original content")
    adapted_content: str = Field(description="Voice-adapted content")
    voice_config: BrandVoiceConfig = Field(description="Applied voice configuration")
    adaptation_summary: Dict[str, Any] = Field(description="Summary of changes made")
    quality_score: float = Field(description="Voice consistency score")
    timestamp: datetime = Field(default_factory=datetime.now)

# ✅ PREDEFINED BRAND VOICE CONFIGURATIONS

BRAND_VOICE_PROFILES = {
    "casino_expert_authoritative": BrandVoiceConfig(
        voice_name="Casino Expert - Authoritative",
        tone="professional and authoritative",
        personality_traits=["knowledgeable", "trustworthy", "analytical", "experienced"],
        vocabulary_style="advanced",
        sentence_structure="varied with complex analysis",
        content_approach="analytical with data-driven insights",
        expertise_level="expert",
        call_to_action_style="direct but responsible",
        brand_values=["transparency", "expertise", "responsible gambling", "accuracy"],
        target_audience="experienced casino players and serious researchers",
        content_type_adaptations={
            "casino_review": {
                "opening": "comprehensive analysis approach",
                "structure": "detailed methodical evaluation",
                "conclusion": "definitive expert recommendation"
            },
            "slot_review": {
                "opening": "technical gameplay analysis",
                "structure": "mechanics and features breakdown",
                "conclusion": "strategic play recommendation"
            }
        }
    ),
    
    "casino_casual_friendly": BrandVoiceConfig(
        voice_name="Casino Guide - Friendly",
        tone="casual and approachable",
        personality_traits=["helpful", "encouraging", "relatable", "honest"],
        vocabulary_style="moderate",
        sentence_structure="short and conversational",
        content_approach="conversational with practical tips",
        expertise_level="intermediate",
        call_to_action_style="soft and encouraging",
        brand_values=["helpfulness", "honesty", "fun", "accessibility"],
        target_audience="casual players and beginners",
        content_type_adaptations={
            "casino_review": {
                "opening": "friendly introduction and overview",
                "structure": "easy-to-understand breakdown",
                "conclusion": "personal recommendation with caveats"
            },
            "game_guide": {
                "opening": "welcoming and encouraging",
                "structure": "step-by-step guidance",
                "conclusion": "encouraging next steps"
            }
        }
    ),
    
    "casino_news_balanced": BrandVoiceConfig(
        voice_name="Casino News - Balanced",
        tone="objective and informative",
        personality_traits=["factual", "balanced", "insightful", "current"],
        vocabulary_style="moderate to advanced",
        sentence_structure="varied journalistic style",
        content_approach="balanced reporting with analysis",
        expertise_level="expert",
        call_to_action_style="informative with no pressure",
        brand_values=["objectivity", "timeliness", "accuracy", "insight"],
        target_audience="industry professionals and informed players",
        content_type_adaptations={
            "news_article": {
                "opening": "factual headline approach",
                "structure": "inverted pyramid journalism",
                "conclusion": "industry implications summary"
            },
            "analysis": {
                "opening": "context-setting introduction",
                "structure": "multi-perspective analysis",
                "conclusion": "forward-looking insights"
            }
        }
    )
}

# ✅ BRAND VOICE TRANSFORMATION FUNCTIONS

def create_voice_selector() -> RunnableBranch:
    """Select appropriate brand voice based on content type and context"""
    
    def is_expert_content(input_data: Dict[str, Any]) -> bool:
        """Check if content requires expert voice"""
        return (
            input_data.get("content_type") == "detailed_review" or
            input_data.get("expertise_required", False) or
            "comprehensive" in input_data.get("title", "").lower()
        )
    
    def is_casual_content(input_data: Dict[str, Any]) -> bool:
        """Check if content should use casual voice"""
        return (
            input_data.get("content_type") == "beginner_guide" or
            input_data.get("target_audience") == "beginners" or
            "guide" in input_data.get("title", "").lower()
        )
    
    def is_news_content(input_data: Dict[str, Any]) -> bool:
        """Check if content is news/analysis"""
        return (
            input_data.get("content_type") == "news" or
            input_data.get("content_type") == "analysis" or
            "news" in input_data.get("title", "").lower()
        )
    
    # Voice assignment functions
    expert_voice_assigner = RunnableLambda(lambda x: {
        **x,
        "selected_voice": "casino_expert_authoritative",
        "voice_config": BRAND_VOICE_PROFILES["casino_expert_authoritative"]
    })
    
    casual_voice_assigner = RunnableLambda(lambda x: {
        **x,
        "selected_voice": "casino_casual_friendly",
        "voice_config": BRAND_VOICE_PROFILES["casino_casual_friendly"]
    })
    
    news_voice_assigner = RunnableLambda(lambda x: {
        **x,
        "selected_voice": "casino_news_balanced",
        "voice_config": BRAND_VOICE_PROFILES["casino_news_balanced"]
    })
    
    default_voice_assigner = RunnableLambda(lambda x: {
        **x,
        "selected_voice": "casino_expert_authoritative",
        "voice_config": BRAND_VOICE_PROFILES["casino_expert_authoritative"]
    })
    
    return RunnableBranch(
        (is_expert_content, expert_voice_assigner),
        (is_casual_content, casual_voice_assigner),
        (is_news_content, news_voice_assigner),
        default_voice_assigner
    )

def create_voice_adaptation_prompt() -> ChatPromptTemplate:
    """Create prompt for voice adaptation"""
    
    return ChatPromptTemplate.from_template("""
You are a brand voice specialist. Adapt the provided content to match the specified brand voice configuration.

BRAND VOICE CONFIGURATION:
- Voice Name: {voice_name}
- Tone: {tone}
- Personality Traits: {personality_traits}
- Vocabulary Style: {vocabulary_style}
- Sentence Structure: {sentence_structure}
- Content Approach: {content_approach}
- Expertise Level: {expertise_level}
- CTA Style: {call_to_action_style}
- Brand Values: {brand_values}
- Target Audience: {target_audience}

CONTENT TYPE: {content_type}
CONTENT TYPE ADAPTATIONS: {content_adaptations}

ORIGINAL CONTENT:
{original_content}

ADAPTATION REQUIREMENTS:
1. Maintain all factual information and key points
2. Adjust tone, vocabulary, and sentence structure to match voice profile
3. Ensure personality traits are reflected throughout
4. Apply content type specific adaptations
5. Include appropriate calls-to-action in the specified style
6. Reflect brand values naturally in the content
7. Tailor language complexity for the target audience

ADAPTED CONTENT:
""")

def create_voice_adapter(llm: ChatOpenAI) -> RunnableLambda:
    """Create voice adaptation function"""
    
    voice_prompt = create_voice_adaptation_prompt()
    adaptation_chain = voice_prompt | llm
    
    def adapt_content_voice(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt content to match brand voice"""
        
        voice_config = input_data.get("voice_config")
        original_content = input_data.get("content", "")
        content_type = input_data.get("content_type", "general")
        
        if not voice_config or not original_content:
            return {
                **input_data,
                "adapted_content": original_content,
                "adaptation_summary": {"status": "no_adaptation", "reason": "missing_voice_config_or_content"}
            }
        
        # Get content type specific adaptations
        content_adaptations = voice_config.content_type_adaptations.get(content_type, {})
        
        # Prepare prompt variables
        prompt_vars = {
            "voice_name": voice_config.voice_name,
            "tone": voice_config.tone,
            "personality_traits": ", ".join(voice_config.personality_traits),
            "vocabulary_style": voice_config.vocabulary_style,
            "sentence_structure": voice_config.sentence_structure,
            "content_approach": voice_config.content_approach,
            "expertise_level": voice_config.expertise_level,
            "call_to_action_style": voice_config.call_to_action_style,
            "brand_values": ", ".join(voice_config.brand_values),
            "target_audience": voice_config.target_audience,
            "content_type": content_type,
            "content_adaptations": str(content_adaptations),
            "original_content": original_content
        }
        
        try:
            # Apply voice adaptation
            adapted_content = adaptation_chain.invoke(prompt_vars)
            
            # Calculate basic quality metrics
            original_length = len(original_content)
            adapted_length = len(adapted_content.content if hasattr(adapted_content, 'content') else str(adapted_content))
            length_ratio = adapted_length / original_length if original_length > 0 else 1.0
            
            # Calculate voice consistency score (simplified)
            consistency_score = min(1.0, 0.8 + (0.2 * (1 - abs(1 - length_ratio))))
            
            adaptation_summary = {
                "status": "success",
                "original_length": original_length,
                "adapted_length": adapted_length,
                "length_change_ratio": length_ratio,
                "voice_applied": voice_config.voice_name,
                "content_type_adaptations_applied": len(content_adaptations) > 0
            }
            
            return {
                **input_data,
                "adapted_content": adapted_content.content if hasattr(adapted_content, 'content') else str(adapted_content),
                "adaptation_summary": adaptation_summary,
                "voice_consistency_score": consistency_score
            }
            
        except Exception as e:
            logger.error(f"Voice adaptation failed: {e}")
            return {
                **input_data,
                "adapted_content": original_content,
                "adaptation_summary": {"status": "error", "error": str(e)},
                "voice_consistency_score": 0.0
            }
    
    return RunnableLambda(adapt_content_voice)

def create_voice_validator() -> RunnableLambda:
    """Create voice validation function"""
    
    def validate_voice_adaptation(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that voice adaptation meets quality standards"""
        
        adapted_content = input_data.get("adapted_content", "")
        voice_config = input_data.get("voice_config")
        adaptation_summary = input_data.get("adaptation_summary", {})
        
        validation_results = {
            "content_preserved": len(adapted_content) > 0,
            "voice_applied": "voice_applied" in adaptation_summary,
            "length_reasonable": 0.5 <= adaptation_summary.get("length_change_ratio", 1.0) <= 2.0,
            "adaptation_successful": adaptation_summary.get("status") == "success"
        }
        
        overall_quality = sum(validation_results.values()) / len(validation_results)
        
        return {
            **input_data,
            "voice_validation": validation_results,
            "overall_voice_quality": overall_quality,
            "ready_for_publication": overall_quality >= 0.8
        }
    
    return RunnableLambda(validate_voice_adaptation)

# ✅ MAIN BRAND VOICE CHAIN

def create_brand_voice_chain(
    llm: Optional[ChatOpenAI] = None
) -> RunnableSequence:
    """
    Create brand voice management chain using native LangChain patterns
    
    ✅ NATIVE LANGCHAIN COMPONENTS:
    - RunnablePassthrough for voice configuration flow
    - RunnableBranch for voice type selection
    - RunnableLambda for voice adaptation and validation
    - Structured outputs with Pydantic models
    
    Args:
        llm: Language model for voice adaptation
    
    Returns:
        RunnableSequence: Composable chain for brand voice application
        
    Usage:
        content_data | brand_voice_chain → voice-adapted content
    """
    
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)  # Slightly higher temp for voice variation
    
    # Create chain components
    voice_selector = create_voice_selector()
    voice_adapter = create_voice_adapter(llm)
    voice_validator = create_voice_validator()
    
    # Create the complete voice chain
    input_processor = RunnableLambda(lambda x: {
        "content": x.get("content", ""),
        "content_type": x.get("content_type", "general"),
        "title": x.get("title", ""),
        "target_audience": x.get("target_audience", "general"),
        "expertise_required": x.get("expertise_required", False),
        **x
    })
    
    output_formatter = RunnableLambda(lambda result: VoiceAdaptedContent(
        original_content=result.get("content", ""),
        adapted_content=result.get("adapted_content", ""),
        voice_config=result.get("voice_config"),
        adaptation_summary=result.get("adaptation_summary", {}),
        quality_score=result.get("overall_voice_quality", 0.0)
    ))
    
    brand_voice_chain = (
        input_processor |
        voice_selector |
        voice_adapter |
        voice_validator |
        output_formatter
    )
    
    return brand_voice_chain

# ✅ INTEGRATION HELPERS

class BrandVoiceEnhancer:
    """Helper for integrating brand voice with v2 systems"""
    
    @staticmethod
    def enhance_content_generation(content_chain, llm=None):
        """Add brand voice to content generation pipeline"""
        voice_chain = create_brand_voice_chain(llm)
        return content_chain.pipe(voice_chain)
    
    @staticmethod
    def create_multi_voice_adapter(voice_profiles: List[str]) -> RunnableLambda:
        """Create multi-voice adaptation for A/B testing"""
        
        def adapt_multiple_voices(content_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate content variations with different voices"""
            
            voice_chain = create_brand_voice_chain()
            variations = {}
            
            for voice_name in voice_profiles:
                if voice_name in BRAND_VOICE_PROFILES:
                    voice_specific_data = {
                        **content_data,
                        "voice_config": BRAND_VOICE_PROFILES[voice_name],
                        "selected_voice": voice_name
                    }
                    
                    try:
                        # Skip the voice selector and go directly to adaptation
                        adapter = create_voice_adapter(ChatOpenAI(model="gpt-4o", temperature=0.3))
                        result = adapter.invoke(voice_specific_data)
                        
                        variations[voice_name] = {
                            "adapted_content": result.get("adapted_content"),
                            "quality_score": result.get("voice_consistency_score", 0.0),
                            "adaptation_summary": result.get("adaptation_summary", {})
                        }
                    except Exception as e:
                        variations[voice_name] = {
                            "error": str(e),
                            "quality_score": 0.0
                        }
            
            return {
                "original_content": content_data.get("content", ""),
                "voice_variations": variations,
                "total_variations": len(variations),
                "successful_adaptations": len([v for v in variations.values() if "error" not in v])
            }
        
        return RunnableLambda(adapt_multiple_voices)
    
    @staticmethod
    def create_voice_consistency_checker() -> RunnableLambda:
        """Create voice consistency validation across content pieces"""
        
        def check_voice_consistency(content_pieces: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Check voice consistency across multiple content pieces"""
            
            voice_profiles_used = set()
            quality_scores = []
            
            for piece in content_pieces:
                if "voice_config" in piece:
                    voice_profiles_used.add(piece["voice_config"].voice_name)
                if "voice_consistency_score" in piece:
                    quality_scores.append(piece["voice_consistency_score"])
            
            consistency_report = {
                "total_pieces": len(content_pieces),
                "unique_voices_used": len(voice_profiles_used),
                "voice_profiles": list(voice_profiles_used),
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                "min_quality_score": min(quality_scores) if quality_scores else 0.0,
                "consistency_rating": "high" if len(voice_profiles_used) <= 1 else "medium" if len(voice_profiles_used) <= 2 else "low"
            }
            
            return consistency_report
        
        return RunnableLambda(check_voice_consistency)

# ✅ TESTING

async def test_brand_voice_chain():
    """Test the brand voice chain"""
    print("🧪 Testing Brand Voice Management Chain")
    
    try:
        # Create the chain
        voice_chain = create_brand_voice_chain()
        
        # Test input - casino review content
        test_content = {
            "title": "Comprehensive Betway Casino Review",
            "content": "Betway Casino offers a wide range of games. The platform has good security. Players can deposit and withdraw using various methods. The customer support is available 24/7. Overall, it's a decent casino for players.",
            "content_type": "casino_review",
            "target_audience": "experienced players",
            "expertise_required": True
        }
        
        # Run the chain
        result = await voice_chain.ainvoke(test_content)
        
        print(f"✅ Brand voice chain executed successfully")
        print(f"🎯 Voice Applied: {result.voice_config.voice_name}")
        print(f"📊 Quality Score: {result.quality_score:.2f}")
        print(f"📝 Original Length: {len(result.original_content)} chars")
        print(f"📝 Adapted Length: {len(result.adapted_content)} chars")
        print(f"✨ Voice Tone: {result.voice_config.tone}")
        
        # Test multi-voice adapter
        multi_voice_adapter = BrandVoiceEnhancer.create_multi_voice_adapter([
            "casino_expert_authoritative",
            "casino_casual_friendly"
        ])
        
        multi_result = multi_voice_adapter.invoke(test_content)
        print(f"🔄 Multi-voice variations: {multi_result['successful_adaptations']}/{multi_result['total_variations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Brand voice chain test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_brand_voice_chain()) 