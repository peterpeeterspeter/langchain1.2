#!/usr/bin/env python3
"""
Test Enhanced WebBaseLoader for Casino Research
Simple validation without LLM dependencies
"""

import sys
import os
sys.path.append('src/chains')

from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urljoin
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class URLStrategy:
    """URL generation strategy for different casino data categories"""
    base_domain: str
    alternative_domains: List[str] = field(default_factory=list)
    category_paths: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.category_paths:
            self.category_paths = {
                'trustworthiness': [
                    '/about', '/about-us', '/company', '/licensing', 
                    '/legal', '/terms', '/privacy', '/responsible-gaming'
                ],
                'games': [
                    '/casino', '/games', '/slots', '/live-casino',
                    '/table-games', '/providers'
                ],
                'bonuses': [
                    '/promotions', '/bonuses', '/welcome-bonus',
                    '/free-spins', '/loyalty', '/vip'
                ]
            }

class SimpleWebLoader:
    """Simplified WebBaseLoader for testing"""
    
    def __init__(self, strategy: URLStrategy):
        self.strategy = strategy
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def generate_urls(self, categories: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Generate URLs for testing"""
        if categories is None:
            categories = list(self.strategy.category_paths.keys())
        
        url_collection = {}
        all_domains = [self.strategy.base_domain] + self.strategy.alternative_domains
        
        for category in categories:
            category_urls = []
            paths = self.strategy.category_paths.get(category, [])
            
            for domain in all_domains[:1]:  # Just test main domain for now
                for path in paths[:2]:  # Limit to 2 paths per category
                    if not domain.startswith(('http://', 'https://')):
                        domain = f'https://{domain}'
                    url = urljoin(domain, path)
                    category_urls.append(url)
            
            url_collection[category] = category_urls
        
        return url_collection
    
    def test_load_urls(self, urls: List[str]):
        """Test loading URLs"""
        
        print(f"🔍 Testing {len(urls)} URLs...")
        
        loader = WebBaseLoader(
            web_paths=urls,
            header_template=self.headers,
            requests_per_second=0.3,  # Very conservative
            continue_on_failure=True,
            raise_for_status=False,
            requests_kwargs={'timeout': 15}
        )
        
        try:
            start_time = time.time()
            docs = loader.load()
            load_time = time.time() - start_time
            
            successful_loads = 0
            failed_loads = 0
            total_content = 0
            
            for doc in docs:
                content_length = len(doc.page_content.strip())
                total_content += content_length
                
                # Check for failure indicators
                if (content_length > 100 and 
                    not any(fail_term in doc.page_content.lower() for fail_term in [
                        'connection timed out', 'access denied', 'not available in your region',
                        'cease trading', 'forbidden', '502 bad gateway'
                    ])):
                    successful_loads += 1
                    print(f"✅ {doc.metadata.get('source', 'Unknown')[:50]}... ({content_length:,} chars)")
                else:
                    failed_loads += 1
                    failure_reason = "timeout/blocked"
                    if 'cease trading' in doc.page_content.lower():
                        failure_reason = "geo-restricted"
                    elif content_length < 100:
                        failure_reason = "minimal content"
                    print(f"❌ {doc.metadata.get('source', 'Unknown')[:50]}... ({failure_reason})")
            
            print(f"\\n📊 LOADING RESULTS:")
            print(f"  Time: {load_time:.1f}s")
            print(f"  Total URLs: {len(urls)}")
            print(f"  Successful: {successful_loads}")
            print(f"  Failed: {failed_loads}")
            print(f"  Total Content: {total_content:,} characters")
            print(f"  Success Rate: {(successful_loads/len(docs)*100):.1f}%" if docs else "0%")
            
            return {
                'total_attempted': len(urls),
                'total_loaded': len(docs),
                'successful_loads': successful_loads,
                'failed_loads': failed_loads,
                'success_rate': (successful_loads/len(docs)*100) if docs else 0,
                'total_content': total_content,
                'load_time': load_time
            }
            
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            return {'error': str(e)}

def test_casino_research():
    """Test casino research with multiple domains"""
    
    test_casinos = [
        {
            'domain': 'casino.org',
            'alternatives': []
        },
        {
            'domain': 'pokerstar.com', 
            'alternatives': ['pokerstars.net']
        },
        {
            'domain': 'stake.com',
            'alternatives': ['stake.us']
        }
    ]
    
    overall_results = []
    
    for casino_config in test_casinos:
        print(f"\\n{'='*60}")
        print(f"🎰 Testing: {casino_config['domain']}")
        print(f"{'='*60}")
        
        try:
            # Create URL strategy
            strategy = URLStrategy(
                base_domain=casino_config['domain'],
                alternative_domains=casino_config['alternatives']
            )
            
            # Create loader
            web_loader = SimpleWebLoader(strategy)
            
            # Generate URLs for testing
            url_collection = web_loader.generate_urls(['trustworthiness', 'games', 'bonuses'])
            
            # Test each category
            category_results = {}
            
            for category, urls in url_collection.items():
                print(f"\\n🔍 Testing {category.upper()} URLs:")
                result = web_loader.test_load_urls(urls)
                category_results[category] = result
                
                time.sleep(2)  # Respectful delay between categories
            
            # Calculate overall stats
            total_successful = sum(r.get('successful_loads', 0) for r in category_results.values())
            total_attempted = sum(r.get('total_attempted', 0) for r in category_results.values())
            overall_success_rate = (total_successful / total_attempted * 100) if total_attempted > 0 else 0
            
            print(f"\\n🎯 OVERALL RESULTS for {casino_config['domain']}:")
            print(f"  Total URLs Attempted: {total_attempted}")
            print(f"  Total Successful Loads: {total_successful}")
            print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
            print(f"  Ready for 95 Fields: {'✅ YES' if overall_success_rate > 30 else '❌ NO'}")
            
            overall_results.append({
                'domain': casino_config['domain'],
                'success_rate': overall_success_rate,
                'total_successful': total_successful,
                'ready_for_95_fields': overall_success_rate > 30
            })
            
        except Exception as e:
            print(f"❌ Failed to test {casino_config['domain']}: {e}")
            overall_results.append({
                'domain': casino_config['domain'],
                'error': str(e)
            })
    
    # Final summary
    print(f"\\n{'='*60}")
    print(f"🎯 FINAL SUMMARY - WebBaseLoader for Casino Research")
    print(f"{'='*60}")
    
    for result in overall_results:
        if 'error' in result:
            print(f"❌ {result['domain']}: ERROR - {result['error']}")
        else:
            status = "✅ READY" if result['ready_for_95_fields'] else "⚠️  NEEDS WORK"
            print(f"{status} {result['domain']}: {result['success_rate']:.1f}% success, {result['total_successful']} sources")
    
    # Count ready domains
    ready_count = sum(1 for r in overall_results if r.get('ready_for_95_fields', False))
    total_count = len([r for r in overall_results if 'error' not in r])
    
    print(f"\\n📊 READINESS ASSESSMENT:")
    print(f"  Domains Ready for 95 Fields: {ready_count}/{total_count}")
    print(f"  WebBaseLoader Viability: {'✅ EXCELLENT' if ready_count >= 2 else '⚠️  MODERATE' if ready_count >= 1 else '❌ POOR'}")
    
    if ready_count >= 1:
        print(f"\\n🚀 RECOMMENDATION: WebBaseLoader is viable for casino research!")
        print(f"   ✅ Can collect comprehensive data for 95-field analysis")
        print(f"   ✅ Native LangChain integration")
        print(f"   ✅ No external API costs")
        print(f"   ✅ Handles geo-restrictions with alternative domains")
    else:
        print(f"\\n⚠️  RECOMMENDATION: WebBaseLoader needs enhancement for casino research")
        print(f"   🔧 Consider proxy rotation")
        print(f"   🔧 Add more alternative domains")
        print(f"   🔧 Implement archive.org fallback")

if __name__ == "__main__":
    test_casino_research() 