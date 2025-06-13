#!/usr/bin/env python3
"""
Example: Using IntegratedRAGChain with Full Monitoring and Configuration

This example demonstrates how to use the IntegratedRAGChain with all its features:
- Real-time monitoring and analytics
- Dynamic configuration management  
- Feature flags and A/B testing
- Performance profiling
- Enhanced logging

Prerequisites:
- Set environment variables: SUPABASE_URL, SUPABASE_KEY
- Run database migrations if not already done
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from supabase import create_client
from src.chains import IntegratedRAGChain, create_integrated_rag_chain
from src.utils.integration_helpers import (
    quick_setup_integrated_rag,
    IntegrationHealthChecker,
    ConfigurationValidator
)
from src.config.prompt_config import PromptOptimizationConfig

async def main():
    """Main example function demonstrating IntegratedRAGChain usage."""
    
    print("🚀 IntegratedRAGChain Example - Full Monitoring & Configuration")
    print("=" * 70)
    
    # Step 1: Validate environment
    print("\n1. Validating Environment...")
    is_valid, issues = ConfigurationValidator.validate_runtime_environment()
    
    if not is_valid:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease set SUPABASE_URL and SUPABASE_KEY environment variables.")
        return
    
    print("✅ Environment validated successfully")
    
    # Step 2: Quick setup with all systems
    print("\n2. Initializing Integrated Systems...")
    try:
        managers = await quick_setup_integrated_rag()
        print("✅ All systems initialized successfully")
        
        # Health check
        health = await IntegrationHealthChecker.check_all_systems(managers)
        print(f"🔍 System health: {health['overall_health']}")
        
        for system, status in health['systems'].items():
            status_icon = "✅" if status['status'] == 'healthy' else "⚠️"
            print(f"   {status_icon} {system}: {status['message']}")
            
    except Exception as e:
        print(f"❌ Failed to initialize systems: {e}")
        print("Note: This example requires a working Supabase instance")
        return
    
    # Step 3: Create IntegratedRAGChain
    print("\n3. Creating IntegratedRAGChain...")
    try:
        chain = create_integrated_rag_chain(
            model_name="gpt-4",
            temperature=0.1,
            supabase_client=managers['supabase_client'],
            enable_all_features=True,
            # vector_store=your_vector_store  # Add your vector store here
        )
        print("✅ IntegratedRAGChain created with all features enabled")
        
    except Exception as e:
        print(f"❌ Failed to create chain: {e}")
        return
    
    # Step 4: Check feature flags
    print("\n4. Checking Feature Flags...")
    user_context = {"user_id": "demo_user_123", "region": "US"}
    
    flags_to_check = [
        "enable_hybrid_search",
        "enable_advanced_prompts", 
        "enable_streaming_responses",
        "enable_multilingual_support"
    ]
    
    for flag in flags_to_check:
        try:
            enabled = await chain.check_feature_flag(flag, user_context)
            status_icon = "🟢" if enabled else "🔴"
            print(f"   {status_icon} {flag}: {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            print(f"   ❌ {flag}: error checking ({e})")
    
    # Step 5: Configuration Management
    print("\n5. Configuration Management...")
    try:
        # Get current configuration
        config_manager = managers['config_manager']
        current_config = await config_manager.get_active_config()
        print(f"📝 Current config version: {current_config.version}")
        print(f"   Confidence threshold: {current_config.query_classification.confidence_threshold}")
        print(f"   Max context length: {current_config.context_formatting.max_context_length}")
        print(f"   Cache TTL (general): {current_config.cache_config.general_ttl}h")
        
        # Demonstrate configuration update
        print("\n   Updating configuration...")
        updated_config = current_config
        updated_config.performance.response_time_warning_ms = 1500  # Reduce warning threshold
        
        await config_manager.save_config(
            updated_config,
            "ExampleScript",
            "Demo configuration update - reduced response time warning"
        )
        
        # Reload in chain
        await chain.reload_configuration()
        print("✅ Configuration updated and reloaded")
        
    except Exception as e:
        print(f"❌ Configuration management error: {e}")
    
    # Step 6: Example Query with Full Monitoring
    print("\n6. Running Example Query with Full Monitoring...")
    
    example_queries = [
        "What are the best online casinos for slot games?",
        "How do I get started with poker online?",
        "What bonuses are available for new players?"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        try:
            # Execute query with monitoring
            start_time = datetime.now()
            
            response = await chain.ainvoke(
                query,
                user_context=user_context
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # Display results
            print(f"   ✅ Response generated in {execution_time:.0f}ms")
            print(f"      Query ID: {response.metadata.get('query_id', 'N/A')}")
            print(f"      Confidence: {response.confidence_score:.2f}")
            print(f"      Cached: {'Yes' if response.cached else 'No'}")
            print(f"      Sources: {len(response.sources)}")
            print(f"      Response preview: {response.answer[:100]}...")
            
            # Show monitoring metadata
            if 'total_pipeline_time_ms' in response.metadata:
                print(f"      Pipeline time: {response.metadata['total_pipeline_time_ms']:.0f}ms")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Step 7: Performance Analytics
    print("\n7. Performance Analytics...")
    try:
        analytics = managers['analytics']
        metrics = await analytics.get_real_time_metrics(window_minutes=5)
        
        if metrics:
            print("📊 Recent metrics (last 5 minutes):")
            
            # Classification metrics
            if 'classification' in metrics:
                class_metrics = metrics['classification']
                print(f"   📋 Classification accuracy: {class_metrics.get('overall_accuracy', 0):.1%}")
            
            # Performance metrics  
            if 'performance' in metrics:
                perf_metrics = metrics['performance']
                avg_time = perf_metrics.get('avg_response_time_ms', 0)
                print(f"   ⏱️  Average response time: {avg_time:.0f}ms")
            
            # Cache metrics
            if 'cache' in metrics:
                cache_metrics = metrics['cache']
                hit_rate = cache_metrics.get('overall_hit_rate', 0)
                print(f"   💾 Cache hit rate: {hit_rate:.1%}")
            
            # Quality metrics
            if 'quality' in metrics:
                quality_metrics = metrics['quality']
                avg_quality = quality_metrics.get('avg_quality_score', 0)
                print(f"   ⭐ Average quality score: {avg_quality:.2f}")
        else:
            print("📊 No recent metrics available (queries needed to generate metrics)")
            
    except Exception as e:
        print(f"❌ Analytics error: {e}")
    
    # Step 8: System Monitoring Stats
    print("\n8. System Monitoring Stats...")
    try:
        stats = chain.get_monitoring_stats()
        
        print("📈 Current system status:")
        print(f"   Monitoring: {'✅ Enabled' if stats.get('monitoring_enabled') else '❌ Disabled'}")
        print(f"   Profiling: {'✅ Enabled' if stats.get('profiling_enabled') else '❌ Disabled'}")  
        print(f"   Feature Flags: {'✅ Enabled' if stats.get('feature_flags_enabled') else '❌ Disabled'}")
        
        if 'cache_stats' in stats and stats['cache_stats']:
            cache_stats = stats['cache_stats']
            print(f"   Cache entries: {cache_stats.get('size', 0)}")
            print(f"   Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
        
    except Exception as e:
        print(f"❌ Monitoring stats error: {e}")
    
    # Step 9: Optimization Report (if profiling enabled)
    print("\n9. Performance Optimization...")
    try:
        if chain.enable_profiling:
            report = await chain.get_optimization_report(hours=1)
            
            if 'top_bottlenecks' in report:
                bottlenecks = report['top_bottlenecks']
                if bottlenecks:
                    print("🔍 Top performance bottlenecks:")
                    for bottleneck in bottlenecks[:3]:
                        print(f"   - {bottleneck.get('operation', 'Unknown')}: {bottleneck.get('avg_duration_ms', 0):.0f}ms")
                else:
                    print("🔍 No bottlenecks detected")
            else:
                print("🔍 Profiling data not available")
        else:
            print("🔍 Profiling disabled - enable for optimization recommendations")
            
    except Exception as e:
        print(f"❌ Optimization report error: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 IntegratedRAGChain Example Completed Successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Environment validation and system health checks")
    print("✅ Integrated system initialization with all managers")
    print("✅ Feature flag management and A/B testing")
    print("✅ Dynamic configuration management with live updates")
    print("✅ Query execution with comprehensive monitoring")
    print("✅ Real-time analytics and performance metrics")
    print("✅ System monitoring and optimization insights")
    print("\nThe IntegratedRAGChain provides production-ready observability,")
    print("configuration management, and feature control for your RAG system!")


# Helper functions for demonstration
async def demonstrate_migration():
    """Demonstrate migration from UniversalRAGChain to IntegratedRAGChain."""
    
    print("\n🔄 Migration Example")
    print("-" * 50)
    
    # This would normally be your existing UniversalRAGChain
    from src.chains import UniversalRAGChain
    from src.utils.integration_helpers import MigrationHelper
    
    # Show migration guide
    guide = MigrationHelper.create_migration_guide()
    
    print("📋 Migration Steps:")
    for step in guide['migration_steps']:
        print(f"\n{step['step']}. {step['description']}")
        print(f"   Old: {step['old_code']}")
        print(f"   New: {step['new_code']}")
    
    print(f"\n✨ New Features Available:")
    for feature in guide['new_features']:
        print(f"   • {feature}")
    
    print(f"\n🛡️  Backward Compatible: {guide['backward_compatibility']}")
    print(f"💥 Breaking Changes: {len(guide['breaking_changes'])} (none!)")


async def run_example():
    """Run the full example with error handling."""
    
    try:
        await main()
        
        # Demonstrate migration
        await demonstrate_migration()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Example interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting IntegratedRAGChain Example...")
    print("Press Ctrl+C to interrupt")
    
    asyncio.run(run_example()) 