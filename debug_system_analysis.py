#!/usr/bin/env python3
"""
🔧 UNIVERSAL RAG CHAIN DIAGNOSTIC ANALYSIS
==========================================

Systematic debugging to identify what's actually broken in the Universal RAG Chain.
Tests each component individually to isolate failure points.

Based on:
- Task 17: 95-field casino intelligence extraction (COMPLETE)
- Task 18: Enhanced WordPress publishing system  
- Universal RAG Chain: All 12 features implemented

Author: AI Assistant
Date: 2025-01-20
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append('src')

class SystemDiagnostic:
    """Comprehensive system diagnostic for Universal RAG Chain"""
    
    def __init__(self):
        self.results = {}
        self.errors = {}
        
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        
        self.results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    def log_error(self, test_name: str, error: Exception):
        """Log error details"""
        error_msg = f"{type(error).__name__}: {str(error)}"
        self.errors[test_name] = {
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        print(f"   ERROR: {error_msg}")

    def test_1_basic_imports(self):
        """Test 1: Basic Python imports"""
        print("\n🔍 TEST 1: Basic Python Imports")
        print("-" * 40)
        
        try:
            # Test core LangChain imports
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import PydanticOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import ChatOpenAI
            from langchain_community.embeddings import OpenAIEmbeddings
            
            self.log_result("Core LangChain imports", True, "All core imports successful")
            
        except Exception as e:
            self.log_result("Core LangChain imports", False)
            self.log_error("Core LangChain imports", e)

    def test_2_casino_schema_import(self):
        """Test 2: Casino Intelligence Schema Import"""
        print("\n🔍 TEST 2: Casino Intelligence Schema")
        print("-" * 40)
        
        try:
            # Test schema import
            from schemas.casino_intelligence_schema import CasinoIntelligence
            
            # Test schema instantiation
            test_casino = CasinoIntelligence(casino_name="Test Casino")
            
            # Check field count
            total_fields = test_casino.calculate_completeness_score()
            
            self.log_result("Casino schema import", True, f"Schema loaded, {95} fields available")
            
            # Test Pydantic parser
            parser = PydanticOutputParser(pydantic_object=CasinoIntelligence)
            format_instructions = parser.get_format_instructions()
            
            self.log_result("Pydantic parser creation", True, "Parser created successfully")
            
        except Exception as e:
            self.log_result("Casino schema import", False)
            self.log_error("Casino schema import", e)

    def test_3_environment_variables(self):
        """Test 3: Environment Variables"""
        print("\n🔍 TEST 3: Environment Variables")
        print("-" * 40)
        
        # Check critical environment variables
        env_vars = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'SUPABASE_URL': os.getenv('SUPABASE_URL'),
            'SUPABASE_SERVICE_KEY': os.getenv('SUPABASE_SERVICE_KEY'),
            'TAVILY_API_KEY': os.getenv('TAVILY_API_KEY'),
            'DATAFORSEO_LOGIN': os.getenv('DATAFORSEO_LOGIN'),
            'DATAFORSEO_PASSWORD': os.getenv('DATAFORSEO_PASSWORD'),
        }
        
        for var_name, var_value in env_vars.items():
            if var_value:
                # Mask the key for security
                masked_value = var_value[:8] + "..." if len(var_value) > 8 else "***"
                self.log_result(f"ENV: {var_name}", True, f"Value: {masked_value}")
            else:
                self.log_result(f"ENV: {var_name}", False, "Not set")

    def test_4_openai_model_access(self):
        """Test 4: OpenAI Model Access"""
        print("\n🔍 TEST 4: OpenAI Model Access")
        print("-" * 40)
        
        try:
            from langchain_openai import ChatOpenAI
            
            # Test with gpt-4.1-mini (the model used in the chain)
            llm = ChatOpenAI(
                model_name="gpt-4.1-mini",
                temperature=0.1,
                max_tokens=100
            )
            
            # Simple test query
            test_response = llm.invoke("Say 'OpenAI connection successful'")
            
            self.log_result("OpenAI gpt-4.1-mini access", True, f"Response: {test_response.content[:50]}...")
            
        except Exception as e:
            self.log_result("OpenAI gpt-4.1-mini access", False)
            self.log_error("OpenAI gpt-4.1-mini access", e)

    def test_5_supabase_connection(self):
        """Test 5: Supabase Database Connection"""
        print("\n🔍 TEST 5: Supabase Connection")
        print("-" * 40)
        
        try:
            from supabase import create_client, Client
            
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
            
            if not supabase_url or not supabase_key:
                self.log_result("Supabase connection", False, "Missing environment variables")
                return
            
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # Test basic query
            result = supabase.table('documents').select('id').limit(1).execute()
            
            self.log_result("Supabase connection", True, f"Connected, found {len(result.data)} test records")
            
        except Exception as e:
            self.log_result("Supabase connection", False)
            self.log_error("Supabase connection", e)

    def test_6_universal_rag_chain_creation(self):
        """Test 6: Universal RAG Chain Creation"""
        print("\n🔍 TEST 6: Universal RAG Chain Creation")
        print("-" * 40)
        
        try:
            from chains.universal_rag_lcel import create_universal_rag_chain
            
            # Create chain with minimal features first
            chain = create_universal_rag_chain(
                model_name='gpt-4.1-mini',
                temperature=0.1,
                enable_caching=False,  # Disable caching for testing
                enable_contextual_retrieval=False,  # Disable complex features
                enable_comprehensive_web_research=False,  # Disable web research
                enable_dataforseo_images=False,  # Disable external APIs
                enable_wordpress_publishing=False,  # Disable WordPress
            )
            
            active_features = chain._count_active_features()
            
            self.log_result("RAG Chain creation", True, f"Created with {active_features} active features")
            
        except Exception as e:
            self.log_result("RAG Chain creation", False)
            self.log_error("RAG Chain creation", e)

    async def test_7_simple_rag_query(self):
        """Test 7: Simple RAG Query Execution"""
        print("\n🔍 TEST 7: Simple RAG Query")
        print("-" * 40)
        
        try:
            from chains.universal_rag_lcel import create_universal_rag_chain
            
            # Create minimal chain
            chain = create_universal_rag_chain(
                model_name='gpt-4.1-mini',
                temperature=0.1,
                enable_caching=False,
                enable_contextual_retrieval=False,
                enable_comprehensive_web_research=False,
                enable_dataforseo_images=False,
                enable_wordpress_publishing=False,
            )
            
            # Simple test query
            response = await chain.ainvoke({'question': 'What is a casino?'})
            
            self.log_result("Simple RAG query", True, f"Response length: {len(response.answer)} chars")
            
        except Exception as e:
            self.log_result("Simple RAG query", False)
            self.log_error("Simple RAG query", e)

    async def test_8_casino_intelligence_extraction(self):
        """Test 8: 95-Field Casino Intelligence Extraction"""
        print("\n🔍 TEST 8: Casino Intelligence Extraction")
        print("-" * 40)
        
        try:
            from chains.universal_rag_lcel import UniversalRAGChain
            
            # Create chain instance
            chain = UniversalRAGChain(
                model_name='gpt-4.1-mini',
                enable_comprehensive_web_research=True  # Enable 95-field extraction
            )
            
            # Test the extraction method directly
            test_sources = [
                {
                    'url': 'https://betway.com',
                    'content': 'Betway Casino is licensed by the Malta Gaming Authority. Offers slots, table games, and live dealer games. Accepts credit cards and e-wallets.',
                    'title': 'Betway Casino'
                }
            ]
            
            casino_data = chain._extract_structured_casino_data(test_sources)
            
            if casino_data and isinstance(casino_data, dict):
                casino_name = casino_data.get('casino_name', 'Unknown')
                license_info = casino_data.get('license_status', 'Unknown')
                
                self.log_result("Casino intelligence extraction", True, f"Extracted data for {casino_name}, License: {license_info}")
            else:
                self.log_result("Casino intelligence extraction", False, "No structured data returned")
            
        except Exception as e:
            self.log_result("Casino intelligence extraction", False)
            self.log_error("Casino intelligence extraction", e)

    def test_9_file_structure_check(self):
        """Test 9: File Structure Check"""
        print("\n🔍 TEST 9: File Structure")
        print("-" * 40)
        
        required_files = [
            'src/chains/universal_rag_lcel.py',
            'src/schemas/casino_intelligence_schema.py',
            'src/integrations/wordpress_publisher.py',
            'src/integrations/enhanced_casino_wordpress_publisher.py',
            '.env',
            'betway_review_demo.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.log_result(f"File: {file_path}", True, f"Size: {file_size:,} bytes")
            else:
                self.log_result(f"File: {file_path}", False, "File not found")

    def generate_summary_report(self):
        """Generate summary diagnostic report"""
        print("\n" + "=" * 60)
        print("🎯 DIAGNOSTIC SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n🔥 FAILED TESTS:")
            for test_name, result in self.results.items():
                if not result['success']:
                    print(f"   ❌ {test_name}")
                    if test_name in self.errors:
                        print(f"      Error: {self.errors[test_name]['error']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if failed_tests == 0:
            print("   🎉 All tests passed! The system should be working correctly.")
            print("   🔍 If you're still experiencing issues, the problem may be:")
            print("      - Network connectivity during execution")
            print("      - API rate limiting")
            print("      - Timeout issues in terminal execution")
        else:
            if not self.results.get('ENV: OPENAI_API_KEY', {}).get('success'):
                print("   🔑 Set OPENAI_API_KEY environment variable")
            if not self.results.get('ENV: SUPABASE_URL', {}).get('success'):
                print("   🗄️  Set Supabase environment variables")
            if not self.results.get('OpenAI gpt-4.1-mini access', {}).get('success'):
                print("   🤖 Check OpenAI API access and billing")
            if not self.results.get('Casino schema import', {}).get('success'):
                print("   📦 Fix Python import paths and dependencies")

async def main():
    """Run comprehensive system diagnostic"""
    print("🔧 UNIVERSAL RAG CHAIN SYSTEM DIAGNOSTIC")
    print("=" * 50)
    print("🎯 Testing all components to identify issues...")
    print("⚡ Running systematic analysis...")
    print()
    
    diagnostic = SystemDiagnostic()
    
    # Run all diagnostic tests
    diagnostic.test_1_basic_imports()
    diagnostic.test_2_casino_schema_import()
    diagnostic.test_3_environment_variables()
    diagnostic.test_4_openai_model_access()
    diagnostic.test_5_supabase_connection()
    diagnostic.test_6_universal_rag_chain_creation()
    await diagnostic.test_7_simple_rag_query()
    await diagnostic.test_8_casino_intelligence_extraction()
    diagnostic.test_9_file_structure_check()
    
    # Generate summary
    diagnostic.generate_summary_report()
    
    return diagnostic

if __name__ == "__main__":
    # Run the diagnostic
    diagnostic_result = asyncio.run(main()) 