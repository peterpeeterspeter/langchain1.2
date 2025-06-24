#!/usr/bin/env python3
"""
Production Memory Leak Detection Runner for Task 22.4

This script demonstrates the complete memory leak detection and resource cleanup
verification system for screenshot operations in a production-like environment.

✅ TASK 22.4 PRODUCTION FEATURES:
- Extended memory profiling during screenshot operations
- Automated memory leak detection with statistical analysis
- Resource cleanup verification for browser processes and temp files
- Memory usage alerts with configurable thresholds
- Comprehensive reporting for production monitoring
- CI/CD integration support
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('memory_leak_detection.log')
    ]
)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from test_task_22_4_memory_leak_detection import (
    MemoryLeakDetector,
    MemoryProfiler,
    ResourceCleanupVerifier
)


class ProductionMemoryLeakRunner:
    """
    Production-ready memory leak detection and monitoring system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path("memory_leak_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize detector with production settings
        self.detector = MemoryLeakDetector(
            sample_interval=0.5,  # Sample every 500ms
            alert_threshold_mb=1024.0,  # Alert at 1GB
            output_dir=str(self.results_dir)
        )
    
    async def run_comprehensive_memory_tests(self):
        """Run comprehensive memory leak detection tests"""
        self.logger.info("🧠 Starting Comprehensive Memory Leak Detection Tests")
        self.logger.info("=" * 70)
        
        test_scenarios = [
            ("Quick Stability Test", 2, 30),      # 2 minutes, 30 ops/min
            ("Medium Load Test", 5, 20),          # 5 minutes, 20 ops/min  
            ("Extended Endurance Test", 10, 15),  # 10 minutes, 15 ops/min
        ]
        
        overall_results = {
            "test_start": datetime.now().isoformat(),
            "scenarios": {},
            "summary": {},
            "recommendations": []
        }
        
        for scenario_name, duration_minutes, ops_per_minute in test_scenarios:
            self.logger.info(f"🔍 Running {scenario_name}...")
            self.logger.info(f"   Duration: {duration_minutes} minutes")
            self.logger.info(f"   Operations: {ops_per_minute} per minute")
            
            try:
                scenario_results = await self.detector.run_extended_memory_test(
                    duration_minutes=duration_minutes,
                    operations_per_minute=ops_per_minute
                )
                
                overall_results["scenarios"][scenario_name] = scenario_results
                
                # Log scenario summary
                self._log_scenario_summary(scenario_name, scenario_results)
                
            except Exception as e:
                self.logger.error(f"❌ {scenario_name} failed: {e}")
                overall_results["scenarios"][scenario_name] = {
                    "error": str(e),
                    "overall_success": False
                }
        
        # Analyze overall results
        overall_results["summary"] = self._analyze_overall_results(overall_results["scenarios"])
        overall_results["recommendations"] = self._generate_recommendations(overall_results["summary"])
        
        # Save comprehensive results
        self._save_comprehensive_results(overall_results)
        
        # Generate final report
        self._generate_final_report(overall_results)
        
        return overall_results
    
    def _log_scenario_summary(self, scenario_name: str, results: dict):
        """Log summary of a scenario test"""
        if "error" in results:
            self.logger.error(f"   ❌ {scenario_name} FAILED: {results['error']}")
            return
        
        success = results.get("overall_success", False)
        status_icon = "✅" if success else "❌"
        
        self.logger.info(f"   {status_icon} {scenario_name} {'PASSED' if success else 'FAILED'}")
        self.logger.info(f"      Operations: {results.get('operations_completed', 0)} completed, "
                        f"{results.get('operations_failed', 0)} failed")
        
        # Memory analysis summary
        memory_analysis = results.get("memory_analysis", {})
        if "memory_trend" in memory_analysis:
            trend = memory_analysis["memory_trend"]
            leak_severity = trend.get("leak_severity", "unknown")
            slope = trend.get("slope_mb_per_second", 0)
            
            severity_icon = {
                "none": "✅",
                "minor": "⚠️",
                "moderate": "🔶",
                "critical": "🔴"
            }.get(leak_severity, "❓")
            
            self.logger.info(f"      Memory: {severity_icon} {leak_severity} leak severity "
                           f"({slope:.4f} MB/sec growth)")
        
        # Cleanup verification summary
        cleanup = results.get("cleanup_verification", {})
        cleanup_success = cleanup.get("cleanup_successful", False)
        cleanup_icon = "✅" if cleanup_success else "❌"
        issues_count = len(cleanup.get("issues_found", []))
        
        self.logger.info(f"      Cleanup: {cleanup_icon} {'Success' if cleanup_success else 'Issues'} "
                        f"({issues_count} issues found)")
        
        self.logger.info("")
    
    def _analyze_overall_results(self, scenarios: dict) -> dict:
        """Analyze results across all scenarios"""
        analysis = {
            "total_scenarios": len(scenarios),
            "successful_scenarios": 0,
            "failed_scenarios": 0,
            "total_operations": 0,
            "total_failures": 0,
            "memory_leak_issues": 0,
            "cleanup_issues": 0,
            "overall_health": "healthy"
        }
        
        for scenario_name, results in scenarios.items():
            if results.get("overall_success", False):
                analysis["successful_scenarios"] += 1
            else:
                analysis["failed_scenarios"] += 1
            
            analysis["total_operations"] += results.get("operations_completed", 0)
            analysis["total_failures"] += results.get("operations_failed", 0)
            
            # Check for memory leaks
            memory_analysis = results.get("memory_analysis", {})
            if "memory_trend" in memory_analysis:
                leak_severity = memory_analysis["memory_trend"].get("leak_severity", "none")
                if leak_severity in ["moderate", "critical"]:
                    analysis["memory_leak_issues"] += 1
            
            # Check for cleanup issues
            cleanup = results.get("cleanup_verification", {})
            if not cleanup.get("cleanup_successful", True):
                analysis["cleanup_issues"] += 1
        
        # Determine overall health
        if analysis["failed_scenarios"] > 0 or analysis["memory_leak_issues"] > 0:
            analysis["overall_health"] = "critical"
        elif analysis["cleanup_issues"] > 0:
            analysis["overall_health"] = "warning"
        else:
            analysis["overall_health"] = "healthy"
        
        return analysis
    
    def _generate_recommendations(self, summary: dict) -> list:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Overall health recommendations
        if summary["overall_health"] == "critical":
            recommendations.append("🔴 CRITICAL: Immediate investigation required for memory leaks or test failures")
            recommendations.append("📋 Review browser instance lifecycle and cleanup procedures")
            recommendations.append("🔍 Implement more frequent memory profiling in production")
        elif summary["overall_health"] == "warning":
            recommendations.append("⚠️ WARNING: Resource cleanup issues detected")
            recommendations.append("🧹 Review temporary file and process cleanup procedures")
        else:
            recommendations.append("✅ HEALTHY: Memory usage and resource cleanup appear stable")
        
        # Specific recommendations based on issues
        if summary["memory_leak_issues"] > 0:
            recommendations.append("🧠 Consider implementing memory limits and automatic browser recycling")
            recommendations.append("📊 Set up continuous memory monitoring with alerting")
        
        if summary["cleanup_issues"] > 0:
            recommendations.append("🗑️ Implement more aggressive resource cleanup strategies")
            recommendations.append("⏰ Add timeout mechanisms for cleanup operations")
        
        if summary["total_failures"] > 0:
            failure_rate = (summary["total_failures"] / summary["total_operations"]) * 100
            if failure_rate > 5:
                recommendations.append(f"📉 High failure rate ({failure_rate:.1f}%) - investigate screenshot reliability")
        
        # Production deployment recommendations
        recommendations.append("📈 Consider implementing these memory monitoring thresholds in production:")
        recommendations.append("   - Memory growth rate > 0.1 MB/sec: Warning")
        recommendations.append("   - Memory growth rate > 0.5 MB/sec: Critical")
        recommendations.append("   - Browser process orphaned for > 30 seconds: Alert")
        recommendations.append("   - Temp files not cleaned within 60 seconds: Alert")
        
        return recommendations
    
    def _save_comprehensive_results(self, results: dict):
        """Save comprehensive results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"comprehensive_memory_test_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = self.detector._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"📁 Comprehensive results saved to: {results_file}")
    
    def _generate_final_report(self, results: dict):
        """Generate a final summary report"""
        self.logger.info("=" * 70)
        self.logger.info("📋 FINAL MEMORY LEAK DETECTION REPORT")
        self.logger.info("=" * 70)
        
        summary = results["summary"]
        
        # Overall status
        health_icons = {
            "healthy": "✅",
            "warning": "⚠️", 
            "critical": "🔴"
        }
        health_icon = health_icons.get(summary["overall_health"], "❓")
        
        self.logger.info(f"🎯 Overall Health: {health_icon} {summary['overall_health'].upper()}")
        self.logger.info("")
        
        # Test statistics
        self.logger.info("📊 Test Statistics:")
        self.logger.info(f"   • Scenarios executed: {summary['total_scenarios']}")
        self.logger.info(f"   • Successful scenarios: {summary['successful_scenarios']}")
        self.logger.info(f"   • Failed scenarios: {summary['failed_scenarios']}")
        self.logger.info(f"   • Total operations: {summary['total_operations']}")
        self.logger.info(f"   • Operation failures: {summary['total_failures']}")
        
        if summary['total_operations'] > 0:
            success_rate = ((summary['total_operations'] - summary['total_failures']) / 
                          summary['total_operations']) * 100
            self.logger.info(f"   • Success rate: {success_rate:.1f}%")
        
        self.logger.info("")
        
        # Issue summary
        self.logger.info("🔍 Issue Summary:")
        self.logger.info(f"   • Memory leak issues: {summary['memory_leak_issues']}")
        self.logger.info(f"   • Resource cleanup issues: {summary['cleanup_issues']}")
        self.logger.info("")
        
        # Recommendations
        self.logger.info("💡 Recommendations:")
        for recommendation in results["recommendations"]:
            self.logger.info(f"   {recommendation}")
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("🎉 Memory Leak Detection Test Suite Completed!")
        self.logger.info("=" * 70)


async def main():
    """Main entry point for production memory leak detection"""
    runner = ProductionMemoryLeakRunner()
    
    try:
        results = await runner.run_comprehensive_memory_tests()
        
        # Exit with appropriate code for CI/CD integration
        summary = results.get("summary", {})
        overall_health = summary.get("overall_health", "unknown")
        
        if overall_health == "critical":
            sys.exit(1)  # Critical issues found
        elif overall_health == "warning":
            sys.exit(2)  # Warnings found
        else:
            sys.exit(0)  # All tests passed
            
    except Exception as e:
        logging.error(f"❌ Memory leak detection runner failed: {e}")
        sys.exit(3)  # Test runner error


if __name__ == "__main__":
    asyncio.run(main()) 