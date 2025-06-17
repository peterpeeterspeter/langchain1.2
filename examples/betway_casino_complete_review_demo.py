#!/usr/bin/env python3
"""
Betway Casino Complete Review Demo
Using Full Enterprise Pipeline with Native LangChain Chains

This demo showcases:
1. Comprehensive Research Chain (95+ fields)
2. Brand Voice Chain (professional adaptation)
3. WordPress Publishing Chain (complete XML)
4. Full integration with v2 architecture
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chains.comprehensive_research_chain import create_comprehensive_research_chain
from src.chains.wordpress_publishing_chain import create_wordpress_publishing_chain
from src.chains.brand_voice_chain import create_brand_voice_chain
from src.chains.enhanced_confidence_scoring_system import EnhancedConfidenceCalculator
from src.retrieval.contextual_retrieval import create_contextual_retrieval_system

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class BetwayReviewGenerator:
    """Complete Betway Casino review generator using enterprise pipeline"""
    
    def __init__(self):
        print("🚀 Initializing Betway Casino Review Generator")
        print("=" * 80)
        
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=4000
        )
        
        self.embeddings = OpenAIEmbeddings()
        
        # Create sample casino documents for retrieval
        self.setup_casino_knowledge_base()
        
        # Initialize enterprise chains
        self.research_chain = create_comprehensive_research_chain(self.retriever, self.llm)
        self.voice_chain = create_brand_voice_chain(self.llm)
        self.wordpress_chain = create_wordpress_publishing_chain(self.llm)
        
        # Initialize confidence scoring
        self.confidence_calculator = EnhancedConfidenceCalculator()
        
        print("✅ All enterprise chains initialized successfully!")
        print()
    
    def setup_casino_knowledge_base(self):
        """Setup comprehensive Betway Casino knowledge base"""
        casino_docs = [
            Document(
                page_content="Betway Casino is licensed by the Malta Gaming Authority (MGA) and UK Gambling Commission (UKGC). Operating since 2006, Betway Group is headquartered in Malta with offices in London and Guernsey. The casino has established itself as a trusted brand in the online gambling industry with over 15 years of experience.",
                metadata={"source": "licensing", "category": "trustworthiness", "last_updated": "2024-01-15"}
            ),
            Document(
                page_content="Betway Casino offers over 500 slot games from top providers including NetEnt, Microgaming, Pragmatic Play, Evolution Gaming, and Red Tiger. The casino features popular titles like Starburst, Gonzo's Quest, Book of Dead, and Mega Moolah. Live casino games include blackjack, roulette, baccarat, and game shows powered by Evolution Gaming.",
                metadata={"source": "games", "category": "entertainment", "last_updated": "2024-01-10"}
            ),
            Document(
                page_content="Betway offers a welcome bonus of 100% up to £250 plus 50 free spins on Starburst. Regular promotions include reload bonuses, free spins, and the Betway Plus loyalty program with exclusive rewards. VIP players receive personalized bonuses, faster withdrawals, and dedicated account managers.",
                metadata={"source": "promotions", "category": "bonuses", "last_updated": "2024-01-12"}
            ),
            Document(
                page_content="Betway Casino supports multiple payment methods including Visa, Mastercard, PayPal, Skrill, Neteller, bank transfers, and Apple Pay. Minimum deposit is £10 with instant processing. Withdrawals range from £10 to £4,000 per transaction with processing times of 1-3 business days for e-wallets and 3-5 days for bank transfers.",
                metadata={"source": "banking", "category": "payments", "last_updated": "2024-01-08"}
            ),
            Document(
                page_content="Customer support is available 24/7 via live chat, email, and phone. The support team is multilingual and highly trained. Average response time for live chat is under 2 minutes. The casino also features a comprehensive FAQ section and responsible gambling tools including deposit limits, session timers, and self-exclusion options.",
                metadata={"source": "support", "category": "customer_service", "last_updated": "2024-01-14"}
            ),
            Document(
                page_content="Betway Casino uses 128-bit SSL encryption to protect player data and transactions. The casino is regularly audited by eCOGRA for fair gaming practices. All games use certified random number generators (RNG). The casino holds certifications for responsible gambling and data protection compliance (GDPR).",
                metadata={"source": "security", "category": "safety", "last_updated": "2024-01-09"}
            ),
            Document(
                page_content="The Betway mobile app is available for iOS and Android devices with over 300 games optimized for mobile play. The responsive website works seamlessly on all devices. Mobile features include touch-friendly navigation, quick deposits, live chat support, and full access to promotions and account management.",
                metadata={"source": "mobile", "category": "accessibility", "last_updated": "2024-01-11"}
            )
        ]
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(casino_docs, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        
        print("📚 Knowledge base created with 7 comprehensive documents")
    
    async def generate_comprehensive_research(self, keyword: str = "Betway Casino") -> Dict[str, Any]:
        """Step 1: Generate comprehensive research using 95+ field extraction"""
        print(f"🔬 STEP 1: COMPREHENSIVE RESEARCH EXTRACTION")
        print(f"Analyzing {keyword} with 95+ field parallel extraction...")
        
        start_time = time.time()
        
        research_result = await self.research_chain.ainvoke({
            "keyword": keyword,
            "content_type": "casino_review"
        })
        
        duration = time.time() - start_time
        
        print(f"✅ Research completed in {duration:.2f}s")
        print(f"📊 Fields populated: {research_result['extraction_summary']['populated_fields']}")
        print(f"🎯 Quality score: {research_result['extraction_summary']['quality_score']:.1f}%")
        print()
        
        return research_result
    
    def generate_review_content(self, research_data: Dict[str, Any]) -> str:
        """Step 2: Generate comprehensive review content based on research"""
        print("📝 STEP 2: CONTENT GENERATION")
        print("Creating comprehensive Betway Casino review...")
        
        comprehensive_data = research_data['comprehensive_data']
        
        # Extract key information
        trustworthiness = comprehensive_data.trustworthiness
        games = comprehensive_data.games
        bonuses = comprehensive_data.bonuses
        payments = comprehensive_data.payments
        ux = comprehensive_data.user_experience
        
        content = f"""# Betway Casino Review 2024: Complete Analysis & Expert Rating

## Executive Summary

Betway Casino stands as one of the most established and trusted names in online gambling, operating under licenses from the {', '.join(trustworthiness.license_authorities or ['Malta Gaming Authority', 'UK Gambling Commission'])}. With {'over 15 years' if not trustworthiness.years_in_operation else f'{trustworthiness.years_in_operation} years'} of experience in the industry, this casino has built a reputation for reliability, comprehensive game selection, and player-focused services.

**Overall Rating: 4.3/5** ⭐⭐⭐⭐

## 🏛️ Licensing & Trustworthiness

### Regulatory Compliance
Betway Casino operates under some of the most stringent licensing authorities in the gambling industry:
- **Malta Gaming Authority (MGA)**: Ensures fair gaming practices and player protection
- **UK Gambling Commission (UKGC)**: Provides additional oversight for UK players
- **Parent Company**: {trustworthiness.parent_company or 'Betway Group Limited'}

### Security & Fair Play
- **SSL Encryption**: 128-bit encryption protects all transactions and personal data
- **Game Fairness**: All games certified by eCOGRA with regular audits
- **RNG Certification**: Random Number Generators tested for true randomness
- **Responsible Gambling**: Comprehensive tools including deposit limits and self-exclusion

## 🎮 Games & Software

### Game Portfolio
Betway Casino offers an impressive selection of over 500 games from industry-leading providers:

**Slot Games**: {len(games.providers or ['NetEnt', 'Microgaming', 'Pragmatic Play'])} top-tier providers
- **Featured Providers**: {', '.join(games.providers or ['NetEnt', 'Microgaming', 'Pragmatic Play', 'Evolution Gaming'])}
- **Popular Titles**: Starburst, Gonzo's Quest, Book of Dead, Mega Moolah
- **Progressive Jackpots**: Multi-million jackpot slots available

**Live Casino**: {'Available' if games.live_casino else 'Not Available'}
- **Live Dealers**: Professional dealers streaming in HD
- **Game Variety**: Blackjack, Roulette, Baccarat, Game Shows
- **Operating Hours**: 24/7 availability

**Mobile Gaming**: {'Optimized' if games.mobile_compatibility else 'Limited'}
- **Mobile App**: Available for iOS and Android
- **Game Selection**: 300+ mobile-optimized games
- **Performance**: Smooth gameplay on all devices

## 💰 Bonuses & Promotions

### Welcome Offer
**New Player Bonus**: {bonuses.welcome_bonus_amount or '100% up to £250 + 50 Free Spins'}
- **Bonus Terms**: Competitive wagering requirements
- **Free Spins**: On popular Starburst slot
- **Time Limit**: Reasonable timeframe for completion

### Ongoing Promotions
- **Reload Bonuses**: Regular deposit bonuses for existing players
- **Free Spins**: Weekly free spin offers
- **Loyalty Program**: Betway Plus rewards system
- **VIP Treatment**: Exclusive bonuses for high-roller players

## 💳 Banking & Payments

### Deposit Methods
Betway Casino supports a comprehensive range of payment options:
- **Credit Cards**: Visa, Mastercard
- **E-Wallets**: PayPal, Skrill, Neteller
- **Mobile Payments**: Apple Pay, Google Pay
- **Bank Transfers**: Direct bank deposits

### Withdrawal Options
- **Processing Times**: 1-3 business days for e-wallets, 3-5 days for bank transfers
- **Withdrawal Limits**: £10 minimum, £4,000 maximum per transaction
- **Verification**: Standard KYC procedures for security

## 📱 User Experience

### Website Design
- **Navigation**: Intuitive and user-friendly interface
- **Search Functionality**: Easy game discovery
- **Loading Times**: Fast page loading and game launches
- **Visual Design**: Modern and professional appearance

### Mobile Experience
- **Responsive Design**: Seamless mobile website
- **App Availability**: Dedicated mobile apps
- **Touch Optimization**: Optimized for touchscreen devices
- **Feature Parity**: Full functionality on mobile

## 🎧 Customer Support

### Support Channels
- **Live Chat**: 24/7 availability with quick response times
- **Email Support**: Comprehensive email assistance
- **Phone Support**: Direct phone line for urgent issues
- **FAQ Section**: Extensive self-help resources

### Support Quality
- **Response Time**: Average 2 minutes for live chat
- **Languages**: Multilingual support team
- **Expertise**: Well-trained support agents
- **Availability**: Round-the-clock assistance

## ✅ Pros and Cons

### Advantages
✅ **Strong Licensing**: MGA and UKGC licensed for maximum protection
✅ **Game Variety**: 500+ games from top providers
✅ **Mobile Optimized**: Excellent mobile casino experience
✅ **Fast Payouts**: Quick withdrawal processing
✅ **24/7 Support**: Round-the-clock customer assistance
✅ **Responsible Gambling**: Comprehensive player protection tools

### Areas for Improvement
❌ **Bonus Terms**: Some wagering requirements could be more player-friendly
❌ **Geographic Restrictions**: Not available in all countries
❌ **Live Chat**: Occasional wait times during peak hours

## 🎯 Final Verdict

Betway Casino delivers a comprehensive and trustworthy online gambling experience that caters to both casual players and serious gamblers. The combination of strong regulatory oversight, diverse game selection, and reliable customer support makes it a standout choice in the competitive online casino market.

**Who Should Play at Betway Casino?**
- Players seeking a trusted, licensed casino
- Slot enthusiasts looking for variety
- Mobile gamers wanting optimized gameplay
- UK players wanting UKGC protection

**Overall Rating: 4.3/5** ⭐⭐⭐⭐

Betway Casino earns its reputation as a premium online gambling destination through consistent performance across all key areas. While there's always room for improvement, the casino's strengths significantly outweigh any minor shortcomings.

## 📋 Quick Facts

- **Founded**: 2006
- **Licenses**: MGA, UKGC
- **Games**: 500+ slots, live casino, table games
- **Welcome Bonus**: 100% up to £250 + 50 Free Spins
- **Mobile**: iOS and Android apps available
- **Support**: 24/7 live chat, email, phone
- **Withdrawal Time**: 1-3 business days

---

*This review was last updated on {datetime.now().strftime('%B %d, %Y')} and reflects current terms and conditions. Please gamble responsibly.*"""

        print(f"✅ Generated comprehensive review ({len(content):,} characters)")
        print()
        
        return content
    
    async def apply_brand_voice(self, content: str) -> Dict[str, Any]:
        """Step 3: Apply professional brand voice adaptation"""
        print("🎭 STEP 3: BRAND VOICE ADAPTATION")
        print("Applying expert authoritative voice...")
        
        start_time = time.time()
        
        voice_result = await self.voice_chain.ainvoke({
            "content": content,
            "content_type": "casino_review",
            "target_audience": "experienced casino players",
            "expertise_required": True
        })
        
        duration = time.time() - start_time
        
        print(f"✅ Voice adaptation completed in {duration:.2f}s")
        print(f"🎭 Voice applied: {getattr(voice_result.voice_config, 'voice_type', 'Professional')}")
        print(f"📊 Quality score: {voice_result.quality_score:.2f}")
        print()
        
        return voice_result
    
    async def generate_wordpress_xml(self, content: str, title: str = "Betway Casino Review 2024") -> Dict[str, Any]:
        """Step 4: Generate WordPress XML for publishing"""
        print("📄 STEP 4: WORDPRESS XML GENERATION")
        print("Creating complete WXR export...")
        
        start_time = time.time()
        
        wordpress_result = await self.wordpress_chain.ainvoke({
            "title": title,
            "content": content,
            "content_type": "casino_review",
            "author": "Casino Expert",
            "category": "Casino Reviews",
            "tags": ["Betway", "Casino Review", "Online Gambling", "2024"]
        })
        
        duration = time.time() - start_time
        
        print(f"✅ WordPress XML generated in {duration:.2f}s")
        print(f"📊 File size: {wordpress_result['file_size_kb']:.1f} KB")
        print(f"📋 Export format: {wordpress_result['export_format']}")
        print()
        
        return wordpress_result
    
    def save_review_outputs(self, research_data, voice_result, wordpress_result):
        """Save all outputs for review"""
        print("💾 STEP 5: SAVING OUTPUTS")
        
        # Save research data
        with open("betway_research_data.json", "w") as f:
            json.dump({
                "extraction_summary": research_data['extraction_summary'],
                "comprehensive_data": {
                    "trustworthiness": research_data['comprehensive_data'].trustworthiness.__dict__,
                    "games": research_data['comprehensive_data'].games.__dict__,
                    "bonuses": research_data['comprehensive_data'].bonuses.__dict__,
                    "payments": research_data['comprehensive_data'].payments.__dict__,
                    "user_experience": research_data['comprehensive_data'].user_experience.__dict__,
                }
            }, f, indent=2, default=str)
        
        # Save final review
        with open("betway_casino_review_final.md", "w") as f:
            f.write(voice_result.adapted_content)
        
        # Save WordPress XML
        with open("betway_casino_wordpress.xml", "w") as f:
            f.write(wordpress_result['wordpress_xml'])
        
        print("✅ Saved outputs:")
        print("   📊 betway_research_data.json - Research extraction results")
        print("   📝 betway_casino_review_final.md - Final review content")
        print("   📄 betway_casino_wordpress.xml - WordPress import file")
        print()
    
    async def generate_complete_review(self):
        """Generate complete Betway Casino review using full pipeline"""
        print("🎯 STARTING COMPLETE BETWAY CASINO REVIEW GENERATION")
        print("=" * 80)
        print()
        
        total_start = time.time()
        
        try:
            # Step 1: Comprehensive research
            research_data = await self.generate_comprehensive_research("Betway Casino")
            
            # Step 2: Generate content
            review_content = self.generate_review_content(research_data)
            
            # Step 3: Apply brand voice
            voice_result = await self.apply_brand_voice(review_content)
            
            # Step 4: Generate WordPress XML
            wordpress_result = await self.generate_wordpress_xml(
                voice_result.adapted_content,
                "Betway Casino Review 2024: Complete Expert Analysis"
            )
            
            # Step 5: Save outputs
            self.save_review_outputs(research_data, voice_result, wordpress_result)
            
            total_duration = time.time() - total_start
            
            print("=" * 80)
            print("🎉 BETWAY CASINO REVIEW GENERATION COMPLETED!")
            print("=" * 80)
            print(f"⏱️  Total processing time: {total_duration:.2f}s")
            print(f"📊 Research quality: {research_data['extraction_summary']['quality_score']:.1f}%")
            print(f"🎭 Voice quality: {voice_result.quality_score:.2f}")
            print(f"📄 WordPress XML: {wordpress_result['file_size_kb']:.1f} KB")
            print()
            
            print("🎯 ENTERPRISE FEATURES DEMONSTRATED:")
            print("   ✅ 95+ Field Research Extraction (RunnableParallel)")
            print("   ✅ Professional Brand Voice Adaptation")
            print("   ✅ WordPress WXR XML Generation")
            print("   ✅ Complete Native LangChain Integration")
            print("   ✅ Modular Architecture (No Monoliths)")
            print()
            
            print("📝 FINAL REVIEW PREVIEW:")
            print("-" * 50)
            preview = voice_result.adapted_content[:500] + "..." if len(voice_result.adapted_content) > 500 else voice_result.adapted_content
            print(preview)
            print("-" * 50)
            
            return {
                "research_data": research_data,
                "voice_result": voice_result,
                "wordpress_result": wordpress_result,
                "total_duration": total_duration
            }
            
        except Exception as e:
            print(f"❌ Error during review generation: {str(e)}")
            raise

async def main():
    """Main execution function"""
    generator = BetwayReviewGenerator()
    await generator.generate_complete_review()

if __name__ == "__main__":
    asyncio.run(main()) 