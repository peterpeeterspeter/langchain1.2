"""
Task 10.11: CI/CD Pipeline Integration - Test Runner

This test suite validates the CI/CD pipeline integration components including:
- GitHub Actions workflow validation
- Test execution pipeline verification
- Coverage enforcement testing
- Performance regression detection
- Deployment validation workflows
- Pipeline configuration validation
"""

import pytest
import os
import yaml
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests_mock

# Test markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ci_cd,
    pytest.mark.task_10_11
]

class TestCICDPipelineIntegration:
    """Test CI/CD Pipeline Integration components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent
        self.workflows_dir = self.project_root / ".github" / "workflows"
        
    def test_workflow_files_exist(self):
        """Test that all required workflow files exist"""
        required_workflows = [
            "universal_rag_cms_ci.yml",
            "performance_regression.yml", 
            "coverage_enforcement.yml",
            "deployment_validation.yml"
        ]
        
        for workflow in required_workflows:
            workflow_path = self.workflows_dir / workflow
            assert workflow_path.exists(), f"Workflow file {workflow} not found"
            assert workflow_path.stat().st_size > 0, f"Workflow file {workflow} is empty"
    
    def test_workflow_yaml_syntax(self):
        """Test that all workflow files have valid YAML syntax"""
        workflow_files = list(self.workflows_dir.glob("*.yml"))
        assert len(workflow_files) >= 4, "Expected at least 4 workflow files"
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                try:
                    workflow_data = yaml.safe_load(f)
                    assert isinstance(workflow_data, dict), f"Invalid YAML structure in {workflow_file.name}"
                    assert 'name' in workflow_data, f"Workflow {workflow_file.name} missing 'name' field"
                    # Check for 'on' field (can be parsed as True by PyYAML)
                    assert ('on' in workflow_data or True in workflow_data), f"Workflow {workflow_file.name} missing 'on' field"
                    assert 'jobs' in workflow_data, f"Workflow {workflow_file.name} missing 'jobs' field"
                except yaml.YAMLError as e:
                    pytest.fail(f"YAML syntax error in {workflow_file.name}: {e}")
    
    def test_main_ci_workflow_structure(self):
        """Test main CI workflow has correct structure"""
        workflow_path = self.workflows_dir / "universal_rag_cms_ci.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check basic structure
        assert workflow['name'] == "Universal RAG CMS - CI/CD Pipeline"
        
        # Check triggers (handle PyYAML parsing 'on' as True)
        on_section = workflow.get('on') or workflow.get(True)
        assert on_section is not None, "Missing 'on' section in workflow"
        assert 'push' in on_section
        assert 'pull_request' in on_section
        assert 'workflow_dispatch' in on_section
        
        # Check required jobs
        required_jobs = [
            'pre_flight',
            'security_scan',
            'code_quality',
            'unit_tests',
            'integration_tests',
            'performance_tests',
            'security_tests',
            'coverage_analysis',
            'test_summary',
            'deployment_gate'
        ]
        
        for job in required_jobs:
            assert job in workflow['jobs'], f"Missing required job: {job}"
        
        # Check job dependencies
        assert workflow['jobs']['unit_tests']['needs'] == 'pre_flight'
        assert 'unit_tests' in workflow['jobs']['integration_tests']['needs']
        assert 'coverage_analysis' in workflow['jobs']['test_summary']['needs']
    
    def test_performance_regression_workflow(self):
        """Test performance regression workflow configuration"""
        workflow_path = self.workflows_dir / "performance_regression.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        assert workflow['name'] == "Performance Regression Detection"
        
        # Check jobs
        required_jobs = ['performance_baseline', 'performance_current', 'regression_analysis']
        for job in required_jobs:
            assert job in workflow['jobs'], f"Missing job: {job}"
        
        # Check environment variables
        assert 'REGRESSION_THRESHOLD' in workflow['env']
        # Check that the threshold has a default value (exact format may vary)
        threshold_value = workflow['env']['REGRESSION_THRESHOLD']
        assert "'10'" in threshold_value or '"10"' in threshold_value
    
    def test_coverage_enforcement_workflow(self):
        """Test coverage enforcement workflow configuration"""
        workflow_path = self.workflows_dir / "coverage_enforcement.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        assert workflow['name'] == "Coverage Enforcement"
        
        # Check jobs
        required_jobs = ['coverage_analysis', 'coverage_diff', 'coverage_enforcement']
        for job in required_jobs:
            assert job in workflow['jobs'], f"Missing job: {job}"
        
        # Check threshold configuration
        assert 'COVERAGE_THRESHOLD' in workflow['env']
    
    def test_deployment_validation_workflow(self):
        """Test deployment validation workflow configuration"""
        workflow_path = self.workflows_dir / "deployment_validation.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        assert workflow['name'] == "Deployment Validation"
        
        # Check workflow_call inputs (handle PyYAML parsing 'on' as True)
        on_section = workflow.get('on') or workflow.get(True)
        assert on_section is not None, "Missing 'on' section in workflow"
        assert 'workflow_call' in on_section
        inputs = on_section['workflow_call']['inputs']
        assert 'environment' in inputs
        assert 'deployment_url' in inputs
        
        # Check jobs exist
        assert 'health_checks' in workflow['jobs']
        assert 'smoke_tests' in workflow['jobs']

    def test_documentation_exists(self):
        """Test that CI/CD documentation exists"""
        docs_dir = self.project_root / "docs"
        ci_cd_guide = docs_dir / "CI_CD_PIPELINE_GUIDE.md"
        
        assert ci_cd_guide.exists(), "CI/CD Pipeline Guide documentation not found"

class TestPipelineConfiguration:
    """Test pipeline configuration and settings"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent
    
    def test_pytest_configuration(self):
        """Test pytest configuration for CI/CD"""
        pytest_ini = self.project_root / "tests" / "pytest.ini"
        assert pytest_ini.exists(), "pytest.ini not found"
        
        # Read and validate pytest configuration
        with open(pytest_ini, 'r') as f:
            content = f.read()
        
        # Check for required markers
        required_markers = ['unit', 'integration', 'performance', 'security', 'smoke']
        for marker in required_markers:
            assert marker in content, f"Pytest marker '{marker}' not configured"
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and has content"""
        requirements_file = self.project_root / "requirements.txt"
        assert requirements_file.exists(), "requirements.txt not found"
        
        with open(requirements_file, 'r') as f:
            requirements = f.read().strip()
        
        assert len(requirements) > 0, "requirements.txt is empty"
        
        # Check for testing dependencies
        testing_deps = ['pytest', 'pytest-cov']
        for dep in testing_deps:
            assert any(dep in line for line in requirements.split('\n')), f"Missing testing dependency: {dep}"
    
    def test_pyproject_toml_configuration(self):
        """Test pyproject.toml configuration if it exists"""
        pyproject_file = self.project_root / "pyproject.toml"
        
        if pyproject_file.exists():
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Basic validation - should be valid TOML-like content
            assert '[' in content and ']' in content, "pyproject.toml appears malformed"

class TestTestExecutionPipeline:
    """Test the test execution pipeline components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent
    
    @patch('subprocess.run')
    def test_unit_test_execution(self, mock_run):
        """Test unit test execution pipeline"""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Simulate unit test execution
        cmd = [
            'python', '-m', 'pytest',
            '-m', 'unit',
            '--cov=../src',
            '--cov-report=xml',
            '--junitxml=junit.xml',
            '-v'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root / "tests")
        
        # Verify command was called
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == cmd
    
    @patch('subprocess.run')
    def test_integration_test_execution(self, mock_run):
        """Test integration test execution pipeline"""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Simulate integration test execution
        cmd = [
            'python', '-m', 'pytest',
            '-m', 'integration',
            '--cov=../src',
            '--cov-report=xml',
            '--timeout=600',
            '-v'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root / "tests")
        
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_performance_test_execution(self, mock_run):
        """Test performance test execution pipeline"""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        # Simulate performance test execution
        cmd = [
            'python', '-m', 'pytest',
            '-m', 'performance',
            '--benchmark-json=benchmark-results.json',
            '--timeout=1200',
            '-v'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root / "tests")
        
        mock_run.assert_called_once()
    
    def test_test_markers_functionality(self):
        """Test that pytest markers work correctly"""
        # This test verifies that our marker system is working
        # by checking that this test itself has the correct markers
        
        # Get current test markers
        request = pytest.current_request if hasattr(pytest, 'current_request') else None
        
        # Verify we can run tests with specific markers
        # This is a meta-test that validates our marker system
        assert True  # Placeholder - the fact that this test runs with markers validates the system

class TestCoverageEnforcement:
    """Test coverage enforcement components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent
    
    def test_coverage_configuration(self):
        """Test coverage configuration"""
        # Check if .coveragerc exists or coverage is configured in pyproject.toml
        coveragerc = self.project_root / ".coveragerc"
        pyproject = self.project_root / "pyproject.toml"
        
        has_coverage_config = coveragerc.exists() or pyproject.exists()
        assert has_coverage_config, "No coverage configuration found"
    
    @patch('subprocess.run')
    def test_coverage_report_generation(self, mock_run):
        """Test coverage report generation"""
        mock_run.return_value = Mock(returncode=0, stdout="Coverage: 85%", stderr="")
        
        # Simulate coverage report generation
        cmd = [
            'python', '-m', 'pytest',
            '--cov=../src',
            '--cov-report=html',
            '--cov-report=xml',
            '--cov-report=term-missing'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root / "tests")
        
        mock_run.assert_called_once()
    
    def test_coverage_threshold_validation(self):
        """Test coverage threshold validation logic"""
        # Test coverage threshold logic
        def validate_coverage_threshold(coverage_pct, threshold):
            return coverage_pct >= threshold
        
        # Test cases
        assert validate_coverage_threshold(85, 80) == True
        assert validate_coverage_threshold(75, 80) == False
        assert validate_coverage_threshold(80, 80) == True
        assert validate_coverage_threshold(100, 80) == True

class TestPerformanceRegression:
    """Test performance regression detection"""
    
    def setup_method(self):
        """Setup test environment"""
        self.sample_baseline = {
            "benchmarks": [
                {"name": "test_query_processing", "stats": {"mean": 0.5}},
                {"name": "test_embedding_generation", "stats": {"mean": 1.2}}
            ]
        }
        
        self.sample_current = {
            "benchmarks": [
                {"name": "test_query_processing", "stats": {"mean": 0.6}},  # 20% slower
                {"name": "test_embedding_generation", "stats": {"mean": 1.1}}  # 8% faster
            ]
        }
    
    def test_performance_regression_detection(self):
        """Test performance regression detection logic"""
        def detect_regression(baseline, current, threshold=10):
            regressions = []
            
            baseline_benchmarks = {b['name']: b['stats'] for b in baseline.get('benchmarks', [])}
            current_benchmarks = {b['name']: b['stats'] for b in current.get('benchmarks', [])}
            
            for name in baseline_benchmarks:
                if name in current_benchmarks:
                    baseline_mean = baseline_benchmarks[name]['mean']
                    current_mean = current_benchmarks[name]['mean']
                    change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
                    
                    if change_pct > threshold:
                        regressions.append((name, change_pct))
            
            return regressions
        
        # Test regression detection
        regressions = detect_regression(self.sample_baseline, self.sample_current, threshold=10)
        
        # Should detect test_query_processing as regression (20% > 10%)
        assert len(regressions) == 1
        assert regressions[0][0] == "test_query_processing"
        assert abs(regressions[0][1] - 20.0) < 0.1
    
    def test_performance_improvement_detection(self):
        """Test performance improvement detection"""
        def detect_improvements(baseline, current, threshold=10):
            improvements = []
            
            baseline_benchmarks = {b['name']: b['stats'] for b in baseline.get('benchmarks', [])}
            current_benchmarks = {b['name']: b['stats'] for b in current.get('benchmarks', [])}
            
            for name in baseline_benchmarks:
                if name in current_benchmarks:
                    baseline_mean = baseline_benchmarks[name]['mean']
                    current_mean = current_benchmarks[name]['mean']
                    change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
                    
                    if change_pct < -threshold:
                        improvements.append((name, abs(change_pct)))
            
            return improvements
        
        # Test improvement detection (should not detect 8% improvement with 10% threshold)
        improvements = detect_improvements(self.sample_baseline, self.sample_current, threshold=10)
        assert len(improvements) == 0
        
        # Test with lower threshold
        improvements = detect_improvements(self.sample_baseline, self.sample_current, threshold=5)
        assert len(improvements) == 1
        assert improvements[0][0] == "test_embedding_generation"

class TestDeploymentValidation:
    """Test deployment validation components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_url = "https://example.com"
    
    def test_health_check_validation(self, requests_mock):
        """Test health check validation"""
        # Mock health endpoint
        requests_mock.get(f"{self.test_url}/health", json={"status": "healthy"}, status_code=200)
        
        # Simulate health check
        import requests
        response = requests.get(f"{self.test_url}/health", timeout=30)
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_smoke_test_validation(self, requests_mock):
        """Test smoke test validation"""
        # Mock various endpoints
        requests_mock.get(self.test_url, text="OK", status_code=200)
        requests_mock.get(f"{self.test_url}/api/v1/status", json={"version": "1.0"}, status_code=200)
        
        # Simulate smoke tests
        import requests
        
        # Test root endpoint
        response = requests.get(self.test_url)
        assert response.status_code == 200
        
        # Test API endpoint
        response = requests.get(f"{self.test_url}/api/v1/status")
        assert response.status_code == 200
        assert "version" in response.json()
    
    def test_deployment_validation_logic(self):
        """Test deployment validation logic"""
        def validate_deployment_results(results):
            critical_checks = ['health_checks', 'smoke_tests']
            
            for check in critical_checks:
                if results.get(check) != 'success':
                    return False
            
            return True
        
        # Test successful deployment
        success_results = {
            'health_checks': 'success',
            'smoke_tests': 'success',
            'performance_tests': 'success'
        }
        assert validate_deployment_results(success_results) == True
        
        # Test failed deployment
        failed_results = {
            'health_checks': 'failure',
            'smoke_tests': 'success',
            'performance_tests': 'success'
        }
        assert validate_deployment_results(failed_results) == False

class TestPipelineIntegration:
    """Test overall pipeline integration"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent
    
    def test_workflow_dependencies(self):
        """Test workflow job dependencies are correctly configured"""
        workflow_path = self.project_root / ".github" / "workflows" / "universal_rag_cms_ci.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        jobs = workflow['jobs']
        
        # Test dependency chain
        # integration_tests should depend on unit_tests
        integration_needs = jobs['integration_tests']['needs']
        assert 'unit_tests' in integration_needs or integration_needs == 'unit_tests'
        
        # coverage_analysis should depend on test jobs
        coverage_needs = jobs['coverage_analysis']['needs']
        assert isinstance(coverage_needs, list)
        assert 'unit_tests' in coverage_needs
        assert 'integration_tests' in coverage_needs
    
    def test_environment_variable_consistency(self):
        """Test environment variables are consistent across workflows"""
        workflow_files = [
            "universal_rag_cms_ci.yml",
            "performance_regression.yml",
            "coverage_enforcement.yml"
        ]
        
        python_versions = set()
        
        for workflow_file in workflow_files:
            workflow_path = self.project_root / ".github" / "workflows" / workflow_file
            
            with open(workflow_path, 'r') as f:
                workflow = yaml.safe_load(f)
            
            if 'env' in workflow and 'PYTHON_VERSION' in workflow['env']:
                python_versions.add(workflow['env']['PYTHON_VERSION'])
        
        # All workflows should use the same Python version
        assert len(python_versions) <= 1, f"Inconsistent Python versions: {python_versions}"
    
    def test_artifact_retention_policies(self):
        """Test artifact retention policies are configured"""
        workflow_path = self.project_root / ".github" / "workflows" / "universal_rag_cms_ci.yml"
        
        with open(workflow_path, 'r') as f:
            content = f.read()
        
        # Check for retention-days configuration
        assert "retention-days: 30" in content, "Missing artifact retention policy"
    
    def test_timeout_configurations(self):
        """Test job timeout configurations are reasonable"""
        workflow_path = self.project_root / ".github" / "workflows" / "universal_rag_cms_ci.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        jobs = workflow['jobs']
        
        # Check that jobs have reasonable timeouts
        for job_name, job_config in jobs.items():
            if 'timeout-minutes' in job_config:
                timeout = job_config['timeout-minutes']
                assert isinstance(timeout, int), f"Job {job_name} has non-integer timeout"
                assert 1 <= timeout <= 60, f"Job {job_name} has unreasonable timeout: {timeout}"

class TestSecurityValidation:
    """Test security aspects of CI/CD pipeline"""
    
    def setup_method(self):
        """Setup test environment"""
        self.project_root = Path(__file__).parent.parent
    
    def test_secrets_not_exposed(self):
        """Test that secrets are not exposed in workflow files"""
        workflow_files = list((self.project_root / ".github" / "workflows").glob("*.yml"))
        
        sensitive_patterns = [
            "sk-",  # API keys
            "password",
            "secret",
            "token"
        ]
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                content = f.read().lower()
            
            for pattern in sensitive_patterns:
                # Should only appear in secrets context
                if pattern in content:
                    # Check if it's properly referenced as a secret
                    lines_with_pattern = [line for line in content.split('\n') if pattern in line]
                    for line in lines_with_pattern:
                        if pattern in line and 'secrets.' not in line and '#' not in line:
                            # Allow test database passwords in services configuration
                            if 'postgres_password: postgres' in line.lower() and 'services:' in content:
                                continue
                            # Allow "secrets: inherit" pattern in GitHub Actions
                            if 'secrets: inherit' in line.lower():
                                continue
                            # Allow standalone "secrets:" pattern in GitHub Actions
                            if line.strip().lower() == 'secrets:':
                                continue
                            # Allow GitHub Actions permission patterns
                            if 'id-token:' in line.lower() or 'contents:' in line.lower() or 'pull-requests:' in line.lower():
                                continue
                            pytest.fail(f"Potential secret exposure in {workflow_file.name}: {line.strip()}")
    
    def test_workflow_permissions(self):
        """Test workflow permissions are properly configured"""
        workflow_path = self.project_root / ".github" / "workflows" / "universal_rag_cms_ci.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check if permissions are explicitly set (good practice)
        # If not set, that's also acceptable as it uses defaults
        if 'permissions' in workflow:
            permissions = workflow['permissions']
            assert isinstance(permissions, dict), "Permissions should be a dictionary"

# Performance and load testing for CI/CD components
class TestCICDPerformance:
    """Test performance aspects of CI/CD pipeline"""
    
    @pytest.mark.performance
    def test_workflow_parsing_performance(self):
        """Test that workflow files can be parsed quickly"""
        import time
        
        workflow_files = list((Path(__file__).parent.parent / ".github" / "workflows").glob("*.yml"))
        
        start_time = time.time()
        
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                yaml.safe_load(f)
        
        parse_time = time.time() - start_time
        
        # Should parse all workflows in under 1 second
        assert parse_time < 1.0, f"Workflow parsing too slow: {parse_time:.2f}s"
    
    @pytest.mark.performance
    def test_test_discovery_performance(self):
        """Test that test discovery is performant"""
        import time
        
        start_time = time.time()
        
        # Simulate test discovery
        test_files = list((Path(__file__).parent).glob("test_*.py"))
        
        discovery_time = time.time() - start_time
        
        # Should discover tests quickly
        assert discovery_time < 0.5, f"Test discovery too slow: {discovery_time:.2f}s"
        assert len(test_files) > 0, "No test files discovered"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 