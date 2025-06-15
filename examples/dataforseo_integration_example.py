#!/usr/bin/env python3
"""
Enhanced DataForSEO Integration Example
Demonstrates the complete integration between DataForSEO Image Search and the Universal RAG CMS

This example shows:
1. Basic image search functionality
2. Integration with existing Supabase infrastructure
3. Enhanced metadata extraction and quality scoring
4. Batch processing capabilities
5. Caching and rate limiting
6. Error handling and retry mechanisms
"""

import asyncio
import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Set up environment variables (replace with your actual credentials)
os.environ.setdefault("DATAFORSEO_LOGIN", "peeters.peter@telenet.be")
os.environ.setdefault("DATAFORSEO_PASSWORD", "654b1cfcca084d19")
os.environ.setdefault("SUPABASE_URL", "https://ambjsovdhizjxwhhnbtd.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFtYmpzb3ZkaGl6anh3aGhuYnRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NzYzNzY0NiwiZXhwIjoyMDYzMjEzNjQ2fQ.ZSgK7qEdhCUkbAcAgeeDz23t-TrkX_m7H9O-WH5z5xs")

# Import the enhanced DataForSEO integration
from src.integrations.dataforseo_image_search import (
    EnhancedDataForSEOImageSearch,
    ImageSearchRequest,
    ImageSearchType,
    ImageSize,
    ImageType,
    ImageColor,
    create_dataforseo_image_search
)

class DataForSEOIntegrationDemo:
    """Comprehensive demonstration of DataForSEO integration capabilities"""
    
    def __init__(self):
        """Initialize the demo with enhanced DataForSEO search"""
        self.search_engine = create_dataforseo_image_search()
        print("🚀 DataForSEO Integration Demo initialized")
    
    async def demo_basic_search(self):
        """Demonstrate basic image search functionality"""
        print("\n" + "="*60)
        print("📸 DEMO 1: Basic Image Search")
        print("="*60)
        
        # Create a search request
        request = ImageSearchRequest(
            keyword="casino slot machines",
            search_engine=ImageSearchType.GOOGLE_IMAGES,
            max_results=10,
            image_size=ImageSize.LARGE,
            image_type=ImageType.PHOTO,
            safe_search=True,
            download_images=False,  # Don't download for basic demo
            generate_alt_text=True,
            quality_filter=True
        )
        
        try:
            # Perform search
            result = await self.search_engine.search_images(request)
            
            # Display results
            print(f"✅ Search completed for '{request.keyword}'")
            print(f"   Total results: {result.total_results}")
            print(f"   Images found: {len(result.images)}")
            print(f"   High quality images: {result.high_quality_count}")
            print(f"   Average quality score: {result.average_quality_score:.2f}")
            print(f"   Search duration: {result.search_duration_ms:.2f}ms")
            print(f"   Estimated cost: ${result.api_cost_estimate:.4f}")
            print(f"   Cached result: {result.cached}")
            
            # Show top 3 images
            print(f"\n📋 Top 3 Images:")
            for i, image in enumerate(result.images[:3]):
                print(f"   {i+1}. {image.title or 'Untitled'}")
                print(f"      Size: {image.width}x{image.height}")
                print(f"      Quality: {image.quality_score:.2f}")
                print(f"      Domain: {image.source_domain}")
                print(f"      Alt text: {image.generated_alt_text}")
                print()
            
            return result
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
            return None
    
    async def demo_batch_search(self):
        """Demonstrate batch search capabilities"""
        print("\n" + "="*60)
        print("🔄 DEMO 2: Batch Image Search")
        print("="*60)
        
        # Create multiple search requests
        keywords = ["poker cards", "roulette wheel", "blackjack table", "casino chips"]
        requests = []
        
        for keyword in keywords:
            request = ImageSearchRequest(
                keyword=keyword,
                search_engine=ImageSearchType.GOOGLE_IMAGES,
                max_results=5,
                image_size=ImageSize.MEDIUM,
                image_type=ImageType.PHOTO,
                safe_search=True,
                download_images=False,
                generate_alt_text=True,
                quality_filter=True
            )
            requests.append(request)
        
        try:
            # Perform batch search
            start_time = datetime.now()
            results = await self.search_engine.batch_search(requests)
            end_time = datetime.now()
            
            batch_duration = (end_time - start_time).total_seconds() * 1000
            
            print(f"✅ Batch search completed")
            print(f"   Keywords processed: {len(keywords)}")
            print(f"   Total batch duration: {batch_duration:.2f}ms")
            print(f"   Average per search: {batch_duration/len(keywords):.2f}ms")
            
            # Display results summary
            total_images = sum(len(result.images) for result in results)
            total_high_quality = sum(result.high_quality_count for result in results)
            total_cost = sum(result.api_cost_estimate for result in results)
            
            print(f"\n📊 Batch Results Summary:")
            print(f"   Total images found: {total_images}")
            print(f"   High quality images: {total_high_quality}")
            print(f"   Total cost estimate: ${total_cost:.4f}")
            
            # Show results per keyword
            for i, (keyword, result) in enumerate(zip(keywords, results)):
                cached_indicator = "💾" if result.cached else "🔍"
                print(f"   {cached_indicator} {keyword}: {len(result.images)} images (avg quality: {result.average_quality_score:.2f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Batch search failed: {str(e)}")
            return []
    
    async def demo_advanced_filtering(self):
        """Demonstrate advanced filtering and search options"""
        print("\n" + "="*60)
        print("🎯 DEMO 3: Advanced Filtering")
        print("="*60)
        
        # Test different filter combinations
        filter_tests = [
            {
                "name": "Large Photos Only",
                "filters": {
                    "image_size": ImageSize.LARGE,
                    "image_type": ImageType.PHOTO,
                    "image_color": ImageColor.COLOR
                }
            },
            {
                "name": "Medium Clipart",
                "filters": {
                    "image_size": ImageSize.MEDIUM,
                    "image_type": ImageType.CLIPART,
                    "image_color": ImageColor.COLOR
                }
            },
            {
                "name": "Black & White Photos",
                "filters": {
                    "image_size": ImageSize.LARGE,
                    "image_type": ImageType.PHOTO,
                    "image_color": ImageColor.BLACK_AND_WHITE
                }
            }
        ]
        
        keyword = "casino gaming"
        
        for test in filter_tests:
            print(f"\n🔍 Testing: {test['name']}")
            
            request = ImageSearchRequest(
                keyword=keyword,
                search_engine=ImageSearchType.GOOGLE_IMAGES,
                max_results=8,
                safe_search=True,
                download_images=False,
                generate_alt_text=True,
                quality_filter=True,
                **test['filters']
            )
            
            try:
                result = await self.search_engine.search_images(request)
                
                print(f"   Results: {len(result.images)} images")
                print(f"   Quality: {result.average_quality_score:.2f} avg")
                print(f"   Duration: {result.search_duration_ms:.2f}ms")
                print(f"   Cached: {'Yes' if result.cached else 'No'}")
                
                # Show format distribution
                formats = {}
                for img in result.images:
                    fmt = img.format or 'unknown'
                    formats[fmt] = formats.get(fmt, 0) + 1
                
                if formats:
                    format_str = ", ".join([f"{fmt}: {count}" for fmt, count in formats.items()])
                    print(f"   Formats: {format_str}")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
    
    async def demo_supabase_integration(self):
        """Demonstrate Supabase storage integration"""
        print("\n" + "="*60)
        print("💾 DEMO 4: Supabase Storage Integration")
        print("="*60)
        
        # Search with download enabled
        request = ImageSearchRequest(
            keyword="casino bonus",
            search_engine=ImageSearchType.GOOGLE_IMAGES,
            max_results=3,  # Small number for demo
            image_size=ImageSize.MEDIUM,
            image_type=ImageType.PHOTO,
            safe_search=True,
            download_images=True,  # Enable download
            generate_alt_text=True,
            quality_filter=True
        )
        
        try:
            print("🔍 Searching and downloading images...")
            result = await self.search_engine.search_images(request)
            
            print(f"✅ Search completed")
            print(f"   Images found: {len(result.images)}")
            print(f"   Processing time: {result.processing_duration_ms:.2f}ms")
            
            # Check download results
            downloaded_count = sum(1 for img in result.images if img.downloaded)
            storage_paths = [img.storage_path for img in result.images if img.storage_path]
            
            print(f"\n💾 Storage Results:")
            print(f"   Successfully downloaded: {downloaded_count}/{len(result.images)}")
            print(f"   Storage paths created: {len(storage_paths)}")
            
            # Show storage details
            for i, image in enumerate(result.images):
                status = "✅ Downloaded" if image.downloaded else "❌ Failed"
                print(f"   Image {i+1}: {status}")
                if image.storage_path:
                    print(f"      Path: {image.storage_path}")
                if image.supabase_id:
                    print(f"      DB ID: {image.supabase_id}")
                if image.processing_errors:
                    print(f"      Errors: {', '.join(image.processing_errors)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Supabase integration failed: {str(e)}")
            return None
    
    async def demo_caching_performance(self):
        """Demonstrate caching and performance optimization"""
        print("\n" + "="*60)
        print("⚡ DEMO 5: Caching & Performance")
        print("="*60)
        
        keyword = "online casino"
        
        # First search (cache miss)
        print("🔍 First search (cache miss expected)...")
        request = ImageSearchRequest(
            keyword=keyword,
            search_engine=ImageSearchType.GOOGLE_IMAGES,
            max_results=5,
            image_size=ImageSize.MEDIUM,
            safe_search=True,
            download_images=False,
            quality_filter=True
        )
        
        try:
            start_time = datetime.now()
            result1 = await self.search_engine.search_images(request)
            end_time = datetime.now()
            
            first_duration = (end_time - start_time).total_seconds() * 1000
            
            print(f"   Duration: {first_duration:.2f}ms")
            print(f"   Cached: {result1.cached}")
            print(f"   Images: {len(result1.images)}")
            
            # Second search (cache hit expected)
            print("\n🔍 Second search (cache hit expected)...")
            
            start_time = datetime.now()
            result2 = await self.search_engine.search_images(request)
            end_time = datetime.now()
            
            second_duration = (end_time - start_time).total_seconds() * 1000
            
            print(f"   Duration: {second_duration:.2f}ms")
            print(f"   Cached: {result2.cached}")
            print(f"   Images: {len(result2.images)}")
            
            # Performance comparison
            if result2.cached:
                speedup = first_duration / second_duration if second_duration > 0 else float('inf')
                print(f"\n⚡ Performance Improvement:")
                print(f"   Cache speedup: {speedup:.1f}x faster")
                print(f"   Time saved: {first_duration - second_duration:.2f}ms")
            
            return result1, result2
            
        except Exception as e:
            print(f"❌ Caching demo failed: {str(e)}")
            return None, None
    
    async def demo_analytics_monitoring(self):
        """Demonstrate analytics and monitoring capabilities"""
        print("\n" + "="*60)
        print("📊 DEMO 6: Analytics & Monitoring")
        print("="*60)
        
        try:
            # Get comprehensive analytics
            analytics = await self.search_engine.get_search_analytics()
            
            print("📈 Current System Analytics:")
            
            # Cache statistics
            cache_stats = analytics.get('cache_stats', {})
            print(f"\n💾 Cache Statistics:")
            print(f"   Current size: {cache_stats.get('size', 0)} entries")
            print(f"   Max size: {cache_stats.get('max_size', 0)} entries")
            print(f"   Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            
            # Rate limiter statistics
            rate_stats = analytics.get('rate_limiter_stats', {})
            print(f"\n🚦 Rate Limiter Statistics:")
            print(f"   Current requests: {rate_stats.get('current_requests', 0)}")
            print(f"   Max per minute: {rate_stats.get('max_per_minute', 0)}")
            print(f"   Max concurrent: {rate_stats.get('max_concurrent', 0)}")
            
            # Configuration
            config = analytics.get('configuration', {})
            print(f"\n⚙️ Configuration:")
            print(f"   Batch size: {config.get('batch_size', 0)}")
            print(f"   Cache enabled: {config.get('cache_enabled', False)}")
            print(f"   Supabase enabled: {config.get('supabase_enabled', False)}")
            
            return analytics
            
        except Exception as e:
            print(f"❌ Analytics demo failed: {str(e)}")
            return None
    
    async def run_comprehensive_demo(self):
        """Run all demonstration scenarios"""
        print("🎯 Enhanced DataForSEO Integration - Comprehensive Demo")
        print("=" * 80)
        
        # Run all demos
        demos = [
            ("Basic Search", self.demo_basic_search),
            ("Batch Search", self.demo_batch_search),
            ("Advanced Filtering", self.demo_advanced_filtering),
            ("Supabase Integration", self.demo_supabase_integration),
            ("Caching Performance", self.demo_caching_performance),
            ("Analytics Monitoring", self.demo_analytics_monitoring)
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            try:
                print(f"\n🚀 Running {demo_name}...")
                result = await demo_func()
                results[demo_name] = result
                print(f"✅ {demo_name} completed successfully")
            except Exception as e:
                print(f"❌ {demo_name} failed: {str(e)}")
                results[demo_name] = None
        
        # Final summary
        print("\n" + "="*80)
        print("📋 DEMO SUMMARY")
        print("="*80)
        
        successful_demos = sum(1 for result in results.values() if result is not None)
        total_demos = len(demos)
        
        print(f"✅ Successful demos: {successful_demos}/{total_demos}")
        print(f"📊 Success rate: {successful_demos/total_demos:.1%}")
        
        for demo_name, result in results.items():
            status = "✅ Success" if result is not None else "❌ Failed"
            print(f"   {demo_name}: {status}")
        
        print(f"\n🎉 Demo completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results

# === MAIN EXECUTION ===

async def main():
    """Main execution function"""
    try:
        # Create demo instance
        demo = DataForSEOIntegrationDemo()
        
        # Run comprehensive demonstration
        results = await demo.run_comprehensive_demo()
        
        # Save results to file for reference
        output_file = f"dataforseo_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if value is not None:
                try:
                    # Try to convert to dict if it has a dict method
                    if hasattr(value, 'dict'):
                        json_results[key] = value.dict()
                    elif hasattr(value, '__dict__'):
                        json_results[key] = str(value)
                    else:
                        json_results[key] = str(value)
                except:
                    json_results[key] = "Result available but not serializable"
            else:
                json_results[key] = None
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Demo execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main()) 