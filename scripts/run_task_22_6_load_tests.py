#!/usr/bin/env python3
"""
Task 22.6: Production Load Testing Runner

This script executes the comprehensive load testing framework for concurrent 
screenshot operations and generates detailed reports for production deployment.

Features:
- Automatic execution of all load test scenarios
- Performance metrics collection and analysis
- Production configuration recommendations
- CI/CD integration with proper exit codes
- JSON report generation for automation
- Performance visualization data

Usage:
    python scripts/run_task_22_6_load_tests.py
    
Exit Codes:
    0: All tests passed, system ready for production
    1: Load tests failed or performance degradation detected
    2: Critical system failure during testing

Author: Task Master AI
Created: 2024-12-19
"""

import asyncio
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the load testing framework
from scripts.task_22_6_load_testing_framework import (
    ScreenshotLoadTestingFramework,
    run_task_22_6_load_tests
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Task226ProductionRunner:
    """
    Production runner for Task 22.6 load testing with comprehensive reporting.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}
        self.exit_code = 0
        
    async def execute_load_testing_suite(self) -> Dict[str, Any]:
        """Execute the complete load testing suite with error handling."""
        
        logger.info("🚀 Task 22.6: Screenshot Load Testing - Production Runner")
        logger.info("=" * 60)
        
        try:
            # Execute comprehensive load tests
            logger.info("📊 Executing load testing scenarios...")
            test_results = await run_task_22_6_load_tests()
            
            if test_results.get('task_status') == 'FAILED':
                logger.error("❌ Load testing execution failed")
                self.exit_code = 2
                return test_results
            
            # Validate results
            validation_result = self._validate_test_results(test_results)
            test_results['validation'] = validation_result
            
            if not validation_result['passed']:
                logger.warning("⚠️ Load testing validation failed")
                self.exit_code = 1
            
            # Generate performance report
            report = self._generate_comprehensive_report(test_results)
            test_results['comprehensive_report'] = report
            
            # Save results
            self._save_test_results(test_results)
            
            # Print summary
            self._print_executive_summary(test_results)
            
            return test_results
            
        except Exception as e:
            logger.error(f"💥 Critical failure during load testing: {str(e)}")
            self.exit_code = 2
            return {
                'task_status': 'CRITICAL_FAILURE',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate load test results against production criteria."""
        
        validation_checks = []
        passed = True
        
        # Check if we have any successful test results
        execution_summary = results.get('execution_summary', {})
        successful_scenarios = execution_summary.get('successful_scenarios', 0)
        total_scenarios = execution_summary.get('total_scenarios', 0)
        
        if total_scenarios == 0:
            validation_checks.append({
                'check': 'scenario_execution',
                'passed': False,
                'message': 'No test scenarios were executed'
            })
            passed = False
        elif successful_scenarios < total_scenarios * 0.8:
            validation_checks.append({
                'check': 'scenario_success_rate',
                'passed': False,
                'message': f'Only {successful_scenarios}/{total_scenarios} scenarios succeeded'
            })
            passed = False
        else:
            validation_checks.append({
                'check': 'scenario_execution',
                'passed': True,
                'message': f'{successful_scenarios}/{total_scenarios} scenarios executed successfully'
            })
        
        # Check optimal concurrency determination
        analysis = results.get('performance_analysis', {})
        optimal_concurrency = analysis.get('optimal_concurrency', {})
        recommended_users = optimal_concurrency.get('recommended', 0)
        
        if recommended_users < 2:
            validation_checks.append({
                'check': 'minimum_concurrency',
                'passed': False,
                'message': f'Recommended concurrency too low: {recommended_users} users'
            })
            passed = False
        else:
            validation_checks.append({
                'check': 'minimum_concurrency',
                'passed': True,
                'message': f'Adequate concurrency support: {recommended_users} users'
            })
        
        # Check for breaking point detection
        breaking_point_detected = execution_summary.get('breaking_point_detected', False)
        if breaking_point_detected:
            system_limits = analysis.get('system_limits', {})
            breaking_point_users = system_limits.get('breaking_point_users', 0)
            
            validation_checks.append({
                'check': 'breaking_point_analysis',
                'passed': True,
                'message': f'Breaking point identified at {breaking_point_users} users'
            })
        else:
            validation_checks.append({
                'check': 'breaking_point_analysis',
                'passed': True,
                'message': 'No breaking point reached in tested range'
            })
        
        # Check production readiness
        summary = results.get('task_22_6_summary', {})
        production_ready = summary.get('production_ready', False)
        
        if not production_ready:
            validation_checks.append({
                'check': 'production_readiness',
                'passed': False,
                'message': 'System not ready for production deployment'
            })
            passed = False
        else:
            validation_checks.append({
                'check': 'production_readiness',
                'passed': True,
                'message': 'System validated for production deployment'
            })
        
        return {
            'passed': passed,
            'checks': validation_checks,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        # Extract key metrics
        analysis = results.get('performance_analysis', {})
        summary = results.get('task_22_6_summary', {})
        
        # Performance highlights
        performance_highlights = {
            'optimal_concurrent_users': summary.get('optimal_concurrent_users', 'Unknown'),
            'max_throughput_qps': summary.get('max_stable_throughput_qps', 'Unknown'),
            'recommended_browser_pool_size': summary.get('recommended_browser_pool_size', 'Unknown'),
            'production_ready': summary.get('production_ready', False)
        }
        
        # System characteristics
        system_characteristics = {
            'tested_concurrency_range': '1-50 users',
            'test_duration_total_seconds': execution_time,
            'scenarios_executed': summary.get('total_scenarios_tested', 0),
            'breaking_point_detected': summary.get('breaking_point_detected', False)
        }
        
        # Production deployment recommendations
        production_config = results.get('production_configuration', {})
        deployment_recommendations = {
            'browser_pool_configuration': production_config.get('browser_pool', {}),
            'queue_configuration': production_config.get('screenshot_queue', {}),
            'resource_limits': production_config.get('resource_limits', {}),
            'monitoring_setup': production_config.get('monitoring', {})
        }
        
        return {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'test_execution_duration_seconds': execution_time,
                'report_version': '1.0.0'
            },
            'performance_highlights': performance_highlights,
            'system_characteristics': system_characteristics,
            'deployment_recommendations': deployment_recommendations,
            'validation_status': results.get('validation', {})
        }
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to files for later analysis."""
        
        # Create reports directory
        reports_dir = Path('.taskmaster/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Main results file
        results_file = reports_dir / f'task_22_6_load_test_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Production configuration file
        if 'production_configuration' in results:
            config_file = reports_dir / f'task_22_6_production_config_{timestamp}.json'
            with open(config_file, 'w') as f:
                json.dump(results['production_configuration'], f, indent=2, default=str)
        
        # Latest symlinks for easy access
        latest_results = reports_dir / 'task_22_6_latest_results.json'
        latest_config = reports_dir / 'task_22_6_latest_config.json'
        
        try:
            if latest_results.exists():
                latest_results.unlink()
            latest_results.symlink_to(results_file.name)
            
            if 'production_configuration' in results:
                if latest_config.exists():
                    latest_config.unlink()
                latest_config.symlink_to(config_file.name)
        except OSError:
            # Symlink creation might fail on some systems, that's okay
            pass
        
        logger.info(f"📁 Test results saved to: {results_file}")
        if 'production_configuration' in results:
            logger.info(f"⚙️ Production config saved to: {config_file}")
    
    def _print_executive_summary(self, results: Dict[str, Any]):
        """Print executive summary of load testing results."""
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 TASK 22.6 LOAD TESTING - EXECUTIVE SUMMARY")
        logger.info("=" * 60)
        
        # Test execution summary
        execution_summary = results.get('execution_summary', {})
        logger.info(f"🔍 Test Execution:")
        logger.info(f"   • Total scenarios: {execution_summary.get('total_scenarios', 0)}")
        logger.info(f"   • Successful: {execution_summary.get('successful_scenarios', 0)}")
        logger.info(f"   • Failed: {execution_summary.get('failed_scenarios', 0)}")
        logger.info(f"   • Duration: {execution_summary.get('total_duration_seconds', 0):.1f}s")
        
        # Performance results
        summary = results.get('task_22_6_summary', {})
        logger.info(f"\n🚀 Performance Results:")
        logger.info(f"   • Optimal concurrency: {summary.get('optimal_concurrent_users', 'Unknown')} users")
        logger.info(f"   • Max throughput: {summary.get('max_stable_throughput_qps', 'Unknown')} QPS")
        logger.info(f"   • Recommended pool size: {summary.get('recommended_browser_pool_size', 'Unknown')} browsers")
        
        # System analysis
        logger.info(f"\n🔬 System Analysis:")
        breaking_point = summary.get('breaking_point_detected', False)
        logger.info(f"   • Breaking point detected: {'YES' if breaking_point else 'NO'}")
        production_ready = summary.get('production_ready', False)
        logger.info(f"   • Production ready: {'YES' if production_ready else 'NO'}")
        
        # Validation status
        validation = results.get('validation', {})
        validation_passed = validation.get('passed', False)
        logger.info(f"\n✅ Validation Status:")
        logger.info(f"   • Overall validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        # Final status
        logger.info(f"\n🎯 FINAL STATUS:")
        if self.exit_code == 0:
            logger.info("   ✅ LOAD TESTING SUCCESSFUL - SYSTEM READY FOR PRODUCTION")
        elif self.exit_code == 1:
            logger.info("   ⚠️ LOAD TESTING COMPLETED WITH WARNINGS")
        else:
            logger.info("   ❌ LOAD TESTING FAILED - SYSTEM NOT READY")
        
        logger.info("=" * 60)

async def main():
    """Main execution function."""
    
    runner = Task226ProductionRunner()
    
    try:
        # Execute load testing suite
        results = await runner.execute_load_testing_suite()
        
        # Exit with appropriate code
        sys.exit(runner.exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n⏸️ Load testing interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main()) 