#!/usr/bin/env python3
"""
Task 22.3: CI/CD Integration for Performance Testing Framework

This module provides utilities for integrating the performance testing framework
into CI/CD pipelines for continuous monitoring and automated performance regression detection.

Usage Examples:
- GitHub Actions integration
- Jenkins pipeline integration  
- GitLab CI integration
- Standalone monitoring scripts

Features:
- Automated benchmark execution
- Performance regression detection
- CI/CD friendly reporting
- Exit code-based pass/fail determination
- JSON and Markdown report generation
"""

import asyncio
import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the tests directory to the path for imports
sys.path.append(os.path.dirname(__file__))
from test_task_22_3_performance_testing_framework import (
    PerformanceBenchmarkSuite,
    ContinuousPerformanceMonitor,
    PerformanceTestCategory,
    ScreenshotComplexity
)


class CICDPerformanceRunner:
    """
    CI/CD Performance Testing Runner
    
    Integrates performance testing into CI/CD pipelines with automated
    benchmark execution, regression detection, and reporting.
    """
    
    def __init__(self, output_dir: str = None, config: Dict[str, Any] = None):
        self.output_dir = output_dir or "./performance_reports"
        self.config = config or self._default_config()
        self.benchmark_suite = PerformanceBenchmarkSuite(output_dir=self.output_dir)
        self.monitor = ContinuousPerformanceMonitor()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for CI/CD performance testing"""
        return {
            "benchmarks": {
                "browser_initialization": {
                    "enabled": True,
                    "iterations": 5
                },
                "screenshot_capture": {
                    "enabled": True,
                    "iterations": 10,
                    "test_scenarios": [
                        {"type": "full_page", "complexity": "simple"},
                        {"type": "viewport", "complexity": "simple"},
                        {"type": "full_page", "complexity": "medium"}
                    ]
                }
            },
            "regression_detection": {
                "enabled": True,
                "threshold": 1.3,  # 30% performance degradation threshold
                "fail_on_regression": True
            },
            "reporting": {
                "json_report": True,
                "markdown_report": True,
                "ci_summary": True
            },
            "performance_targets": {
                "browser_init_p95_ms": 6000,
                "screenshot_capture_p95_ms": 3000,
                "success_rate_min": 0.90
            }
        }
    
    async def run_ci_cd_performance_tests(self) -> Dict[str, Any]:
        """
        Execute complete CI/CD performance test suite
        
        Returns:
            Dict containing test results, analysis, and pass/fail status
        """
        print("🚀 Starting CI/CD Performance Test Suite...")
        
        all_results = []
        test_summary = {
            "start_time": datetime.now().isoformat(),
            "tests_executed": 0,
            "benchmarks_run": [],
            "overall_success": True,
            "performance_regressions": [],
            "recommendations": []
        }
        
        # Run browser initialization benchmarks
        if self.config["benchmarks"]["browser_initialization"]["enabled"]:
            print("📊 Running browser initialization benchmarks...")
            iterations = self.config["benchmarks"]["browser_initialization"]["iterations"]
            
            try:
                browser_results = await self.benchmark_suite.benchmark_browser_initialization(
                    iterations=iterations
                )
                all_results.extend(browser_results)
                test_summary["benchmarks_run"].append("browser_initialization")
                test_summary["tests_executed"] += len(browser_results)
                print(f"   ✅ Completed {len(browser_results)} browser initialization tests")
                
            except Exception as e:
                print(f"   ❌ Browser initialization tests failed: {e}")
                test_summary["overall_success"] = False
        
        # Run screenshot capture benchmarks
        if self.config["benchmarks"]["screenshot_capture"]["enabled"]:
            print("📊 Running screenshot capture benchmarks...")
            iterations = self.config["benchmarks"]["screenshot_capture"]["iterations"]
            scenarios = self.config["benchmarks"]["screenshot_capture"]["test_scenarios"]
            
            for scenario in scenarios:
                try:
                    complexity = ScreenshotComplexity(scenario["complexity"])
                    capture_results = await self.benchmark_suite.benchmark_screenshot_capture(
                        screenshot_type=scenario["type"],
                        complexity=complexity,
                        iterations=iterations
                    )
                    all_results.extend(capture_results)
                    test_summary["tests_executed"] += len(capture_results)
                    print(f"   ✅ Completed {len(capture_results)} {scenario['type']} ({scenario['complexity']}) tests")
                    
                except Exception as e:
                    print(f"   ❌ Screenshot capture tests failed for {scenario}: {e}")
                    test_summary["overall_success"] = False
            
            test_summary["benchmarks_run"].append("screenshot_capture")
        
        # Add results to monitoring system
        if all_results:
            self.monitor.add_performance_data(all_results)
        
        # Perform regression analysis
        if self.config["regression_detection"]["enabled"] and all_results:
            print("🔍 Analyzing performance regressions...")
            
            regression_threshold = self.config["regression_detection"]["threshold"]
            regression_analysis = self.benchmark_suite.detect_performance_regression(
                all_results, 
                regression_threshold=regression_threshold
            )
            
            if regression_analysis.get("regression_detected", False):
                test_summary["performance_regressions"] = regression_analysis["regression_details"]
                print(f"   ⚠️  Performance regressions detected: {len(regression_analysis['regression_details'])} issues")
                
                if self.config["regression_detection"]["fail_on_regression"]:
                    test_summary["overall_success"] = False
                    print("   ❌ CI/CD marked as FAILED due to performance regressions")
            else:
                print("   ✅ No performance regressions detected")
        
        # Check performance targets
        if all_results:
            print("🎯 Validating performance targets...")
            target_validation = self._validate_performance_targets(all_results)
            
            if not target_validation["all_targets_met"]:
                test_summary["overall_success"] = False
                test_summary["recommendations"].extend(target_validation["failed_targets"])
                print(f"   ❌ {len(target_validation['failed_targets'])} performance targets not met")
            else:
                print("   ✅ All performance targets met")
        
        # Generate CI/CD monitoring report
        ci_report = self.monitor.generate_ci_cd_report()
        
        test_summary.update({
            "end_time": datetime.now().isoformat(),
            "ci_cd_health_status": ci_report["health_status"],
            "performance_trends": ci_report.get("performance_trends", {}),
            "ci_recommendations": ci_report.get("recommendations", [])
        })
        
        print(f"🏁 CI/CD Performance Tests Completed!")
        print(f"   - Tests executed: {test_summary['tests_executed']}")
        print(f"   - Overall status: {'✅ PASSED' if test_summary['overall_success'] else '❌ FAILED'}")
        print(f"   - Health status: {ci_report['health_status']}")
        
        return test_summary
    
    def _validate_performance_targets(self, results: List) -> Dict[str, Any]:
        """Validate results against performance targets"""
        targets = self.config["performance_targets"]
        analysis = self.benchmark_suite.analyze_performance(results)
        
        validation = {
            "all_targets_met": True,
            "failed_targets": [],
            "target_results": {}
        }
        
        # Check success rate target
        if "success_rate_min" in targets:
            actual_success_rate = analysis.get("success_rate", 0)
            target_success_rate = targets["success_rate_min"]
            
            validation["target_results"]["success_rate"] = {
                "target": target_success_rate,
                "actual": actual_success_rate,
                "met": actual_success_rate >= target_success_rate
            }
            
            if actual_success_rate < target_success_rate:
                validation["all_targets_met"] = False
                validation["failed_targets"].append(
                    f"Success rate {actual_success_rate:.2%} below target {target_success_rate:.2%}"
                )
        
        # Check timing targets
        timing_analysis = analysis.get("timing_analysis", {})
        
        for metric_name, target_value in targets.items():
            if metric_name.endswith("_p95_ms") and "total_time_ms" in timing_analysis:
                actual_p95 = timing_analysis["total_time_ms"].get("p95", float('inf'))
                
                validation["target_results"][metric_name] = {
                    "target": target_value,
                    "actual": actual_p95,
                    "met": actual_p95 <= target_value
                }
                
                if actual_p95 > target_value:
                    validation["all_targets_met"] = False
                    validation["failed_targets"].append(
                        f"{metric_name}: {actual_p95:.0f}ms exceeds target {target_value}ms"
                    )
        
        return validation
    
    def generate_reports(self, test_summary: Dict[str, Any]) -> Dict[str, str]:
        """Generate various report formats for CI/CD integration"""
        reports = {}
        
        # JSON Report
        if self.config["reporting"]["json_report"]:
            json_path = os.path.join(self.output_dir, "performance_report.json")
            
            # Include full benchmark suite report
            full_report = self.benchmark_suite.generate_performance_report()
            
            combined_report = {
                "ci_cd_summary": test_summary,
                "detailed_analysis": full_report,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(json_path, 'w') as f:
                json.dump(combined_report, f, indent=2)
            
            reports["json"] = json_path
            print(f"📄 JSON report generated: {json_path}")
        
        # Markdown Report
        if self.config["reporting"]["markdown_report"]:
            md_path = os.path.join(self.output_dir, "performance_report.md")
            markdown_content = self._generate_markdown_report(test_summary)
            
            with open(md_path, 'w') as f:
                f.write(markdown_content)
            
            reports["markdown"] = md_path
            print(f"📄 Markdown report generated: {md_path}")
        
        # CI Summary (for GitHub Actions, etc.)
        if self.config["reporting"]["ci_summary"]:
            summary_path = os.path.join(self.output_dir, "ci_summary.txt")
            summary_content = self._generate_ci_summary(test_summary)
            
            with open(summary_path, 'w') as f:
                f.write(summary_content)
            
            reports["ci_summary"] = summary_path
            print(f"📄 CI summary generated: {summary_path}")
        
        return reports
    
    def _generate_markdown_report(self, test_summary: Dict[str, Any]) -> str:
        """Generate a markdown performance report"""
        status_emoji = "✅" if test_summary["overall_success"] else "❌"
        health_emoji = {"healthy": "💚", "warning": "⚠️", "critical": "🔴"}.get(
            test_summary.get("ci_cd_health_status", "unknown"), "❓"
        )
        
        md_content = f"""# Performance Test Report

## Summary
{status_emoji} **Overall Status**: {'PASSED' if test_summary["overall_success"] else 'FAILED'}
{health_emoji} **Health Status**: {test_summary.get("ci_cd_health_status", "Unknown")}

- **Tests Executed**: {test_summary["tests_executed"]}
- **Benchmarks Run**: {", ".join(test_summary["benchmarks_run"])}
- **Test Duration**: {test_summary.get("start_time", "")} to {test_summary.get("end_time", "")}

## Performance Regressions
"""
        
        if test_summary["performance_regressions"]:
            md_content += "\n⚠️ **Regressions Detected**:\n\n"
            for regression in test_summary["performance_regressions"]:
                md_content += f"- **{regression['metric']}**: {regression['current_value']:.2f} vs baseline {regression['baseline_value']:.2f} (ratio: {regression['regression_ratio']:.2f}x)\n"
        else:
            md_content += "\n✅ No performance regressions detected.\n"
        
        md_content += "\n## Recommendations\n\n"
        
        all_recommendations = test_summary["recommendations"] + test_summary.get("ci_recommendations", [])
        if all_recommendations:
            for rec in all_recommendations:
                md_content += f"- {rec}\n"
        else:
            md_content += "- No specific recommendations at this time.\n"
        
        md_content += f"""
## Next Steps

{'- 🔍 Investigate performance regressions and optimize code' if test_summary["performance_regressions"] else ''}
- 📊 Review detailed performance analysis in JSON report
- 🎯 Continue monitoring performance trends
- 🚀 Maintain or improve current performance levels

---
*Generated by Task 22.3 Performance Testing Framework*
"""
        
        return md_content
    
    def _generate_ci_summary(self, test_summary: Dict[str, Any]) -> str:
        """Generate a concise CI summary"""
        status = "PASSED" if test_summary["overall_success"] else "FAILED"
        
        summary = f"""PERFORMANCE TEST {status}
Tests: {test_summary["tests_executed"]}
Health: {test_summary.get("ci_cd_health_status", "unknown")}
Regressions: {len(test_summary["performance_regressions"])}
"""
        
        if test_summary["performance_regressions"]:
            summary += "\nREGRESSION DETAILS:\n"
            for reg in test_summary["performance_regressions"]:
                summary += f"- {reg['metric']}: {reg['regression_ratio']:.2f}x worse\n"
        
        return summary


async def main():
    """Main entry point for CI/CD performance testing"""
    parser = argparse.ArgumentParser(description="CI/CD Performance Testing Runner")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--output-dir", default="./performance_reports", help="Output directory for reports")
    parser.add_argument("--fail-on-regression", action="store_true", help="Fail CI if regressions detected")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per benchmark")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize runner
    runner = CICDPerformanceRunner(output_dir=args.output_dir, config=config)
    
    # Override config with command line arguments
    if args.fail_on_regression:
        runner.config["regression_detection"]["fail_on_regression"] = True
    
    if args.iterations:
        runner.config["benchmarks"]["browser_initialization"]["iterations"] = args.iterations
        runner.config["benchmarks"]["screenshot_capture"]["iterations"] = args.iterations
    
    try:
        # Run performance tests
        test_summary = await runner.run_ci_cd_performance_tests()
        
        # Generate reports
        reports = runner.generate_reports(test_summary)
        
        # Print final status
        if test_summary["overall_success"]:
            print(f"\n🎉 CI/CD Performance Tests PASSED!")
            exit_code = 0
        else:
            print(f"\n💥 CI/CD Performance Tests FAILED!")
            exit_code = 1
        
        print(f"\n📊 Reports generated in: {args.output_dir}")
        for report_type, path in reports.items():
            print(f"   - {report_type}: {path}")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n💥 CI/CD Performance Tests encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 