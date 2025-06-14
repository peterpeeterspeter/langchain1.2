"""
Comprehensive Test Runner
Universal RAG CMS - Task 3.8 Testing Framework Demonstration

This script runs the complete testing suite for the contextual retrieval system.
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from testing.test_contextual_retrieval import (
    run_contextual_retrieval_tests,
    TestConfig,
    ContextualRetrievalTestSuite
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run comprehensive testing suite."""
    print("🧪 Universal RAG CMS - Comprehensive Testing Suite")
    print("="*60)
    print("Running complete test suite for contextual retrieval system...")
    
    # Configure test parameters
    test_config = TestConfig(
        target_response_time_ms=500.0,
        target_precision_at_5=0.8,
        target_cache_hit_rate=0.6,
        target_diversity_score=0.7,
        test_queries_count=20,
        test_documents_count=50,
        benchmark_iterations=5
    )
    
    try:
        # Run comprehensive tests
        test_results = await run_contextual_retrieval_tests(test_config)
        
        # Save test results
        results_file = Path("test_results.json")
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to: {results_file}")
        
        # Print final assessment
        if test_results["overall_passed"]:
            print("\n🎉 ALL TESTS PASSED!")
            print("The contextual retrieval system is ready for production deployment.")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("Review the test results and address any issues before deployment.")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ Test execution failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main()) 