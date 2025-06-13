#!/usr/bin/env python3
"""
Comprehensive test runner for configuration and monitoring systems.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=False)
    duration = time.time() - start_time
    
    print(f"\nCompleted in {duration:.2f} seconds")
    if result.returncode != 0:
        print(f"❌ Failed with exit code {result.returncode}")
        return False
    else:
        print("✅ Success")
        return True


def run_unit_tests():
    """Run unit tests."""
    return run_command([
        "python", "-m", "pytest", 
        "tests/unit/", 
        "-v", 
        "--tb=short",
        "-m", "unit"
    ], "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    return run_command([
        "python", "-m", "pytest", 
        "tests/integration/", 
        "-v", 
        "--tb=short",
        "-m", "integration"
    ], "Integration Tests")


def run_performance_tests():
    """Run performance tests."""
    return run_command([
        "python", "-m", "pytest", 
        "tests/performance/", 
        "-v", 
        "--tb=short",
        "-m", "performance",
        "-s"  # Don't capture output for performance summaries
    ], "Performance Tests")


def run_all_tests():
    """Run all tests."""
    return run_command([
        "python", "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short"
    ], "All Tests")


def run_coverage_tests():
    """Run tests with coverage reporting."""
    return run_command([
        "python", "-m", "pytest", 
        "tests/", 
        "--cov=src", 
        "--cov-report=term-missing", 
        "--cov-report=html:htmlcov",
        "--cov-fail-under=80"
    ], "Coverage Tests")


def run_smoke_tests():
    """Run smoke tests (quick verification)."""
    return run_command([
        "python", "-m", "pytest", 
        "tests/unit/config/test_prompt_config.py::TestQueryType::test_query_type_values",
        "tests/unit/config/test_prompt_config.py::TestCacheConfig::test_default_cache_config",
        "tests/unit/config/test_prompt_config.py::TestPromptOptimizationConfig::test_default_prompt_config",
        "-v"
    ], "Smoke Tests")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "pytest", "pytest-cov", "pytest-asyncio", 
        "pydantic", "supabase", "psutil"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies are installed")
    return True


def install_dependencies():
    """Install test dependencies."""
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "pytest-asyncio>=0.21.0",
        "psutil>=5.9.0"
    ]
    
    return run_command([
        "pip", "install"
    ] + dependencies, "Installing Test Dependencies")


def clean_test_artifacts():
    """Clean up test artifacts."""
    artifacts = [
        ".pytest_cache",
        "__pycache__",
        "htmlcov",
        ".coverage"
    ]
    
    for artifact in artifacts:
        artifact_path = project_root / artifact
        if artifact_path.exists():
            if artifact_path.is_dir():
                import shutil
                shutil.rmtree(artifact_path)
                print(f"Removed directory: {artifact}")
            else:
                artifact_path.unlink()
                print(f"Removed file: {artifact}")
    
    # Clean __pycache__ recursively
    for pycache in project_root.rglob("__pycache__"):
        import shutil
        shutil.rmtree(pycache)
        print(f"Removed: {pycache}")
    
    print("✅ Test artifacts cleaned")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test runner for configuration and monitoring systems")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "all", "coverage", "smoke"],
                       default="all", help="Type of tests to run")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--check-deps", action="store_true", help="Check test dependencies")
    parser.add_argument("--clean", action="store_true", help="Clean test artifacts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    
    print(f"🧪 Configuration & Monitoring Test Runner")
    print(f"Project Root: {project_root}")
    print(f"Python Version: {sys.version}")
    
    # Handle special commands
    if args.clean:
        clean_test_artifacts()
        return
    
    if args.install_deps:
        if not install_dependencies():
            return 1
    
    if args.check_deps:
        if not check_dependencies():
            return 1
        return 0
    
    # Check dependencies before running tests
    if not check_dependencies():
        print("\n💡 Run with --install-deps to install missing dependencies")
        return 1
    
    # Run tests based on type
    success = True
    
    if args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    elif args.type == "performance":
        success = run_performance_tests()
    elif args.type == "coverage":
        success = run_coverage_tests()
    elif args.type == "smoke":
        success = run_smoke_tests()
    elif args.type == "all":
        print("\n🚀 Running comprehensive test suite...")
        
        # Run tests in order
        tests = [
            ("Smoke Tests", run_smoke_tests),
            ("Unit Tests", run_unit_tests),
            ("Integration Tests", run_integration_tests),
            ("Performance Tests", run_performance_tests)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if not test_func():
                print(f"❌ {test_name} failed")
                success = False
                break
            print(f"✅ {test_name} passed")
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        print("\n📊 Test Coverage Report:")
        print("- HTML report available at: htmlcov/index.html")
        print("- Run 'python -m http.server 8000' in htmlcov/ to view")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 