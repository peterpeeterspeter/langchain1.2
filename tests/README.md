# Configuration & Monitoring Testing Framework

A comprehensive testing framework for the configuration and monitoring systems in the Universal RAG CMS project.

## 🧪 Overview

This testing framework provides:
- **Unit Tests**: Component-level testing with mocking
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmarking and performance validation
- **Coverage Reports**: Code coverage analysis
- **Mock Infrastructure**: Comprehensive mocking for external dependencies

## 📁 Directory Structure

```
tests/
├── conftest.py                     # Pytest configuration and global fixtures
├── pytest.ini                     # Pytest settings
├── run_tests.py                    # Main test runner script
├── README.md                       # This documentation
├── fixtures/
│   └── test_configs.py            # Test fixtures and mock data
├── unit/
│   ├── config/
│   │   └── test_prompt_config.py  # Configuration system unit tests
│   └── monitoring/
│       └── test_monitoring_systems.py  # Monitoring systems unit tests
├── integration/
│   └── config_monitoring/
│       └── test_config_integration.py  # Integration tests
└── performance/
    └── profiling/
        └── test_performance_benchmarks.py  # Performance benchmarks
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install test dependencies
python tests/run_tests.py --install-deps

# Or manually install
pip install pytest pytest-cov pytest-asyncio psutil
```

### 2. Run Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type performance
python tests/run_tests.py --type coverage

# Quick smoke tests
python tests/run_tests.py --type smoke
```

### 3. Check Coverage

```bash
# Run with coverage reporting
python tests/run_tests.py --type coverage

# View HTML coverage report
open htmlcov/index.html
```

## 📊 Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation with comprehensive mocking.

**Configuration Tests:**
- QueryType enum validation
- CacheConfig TTL calculations
- QueryClassificationConfig validation
- ContextFormattingConfig weight validation
- PerformanceConfig threshold validation
- FeatureFlags percentage validation
- PromptOptimizationConfig serialization
- ConfigurationManager database operations

**Monitoring Tests:**
- Query metrics data structure validation
- Performance profile analysis
- Alert threshold evaluation
- Feature flag evaluation logic
- Cache analytics calculations

```bash
# Run only unit tests
pytest tests/unit/ -v -m unit
```

### Integration Tests (`tests/integration/`)

Test complete workflows and component interactions.

**Configuration Integration:**
- Full configuration lifecycle (create → save → retrieve → update → rollback)
- Configuration validation edge cases
- Caching behavior validation
- Error handling scenarios
- Configuration history tracking

**Monitoring Integration:**
- Query metrics collection and aggregation
- Performance profiling data flow
- Feature flag integration with configuration
- A/B testing assignment logic

```bash
# Run only integration tests
pytest tests/integration/ -v -m integration
```

### Performance Tests (`tests/performance/`)

Benchmark system performance and validate performance requirements.

**Configuration Performance:**
- Cold vs warm configuration loading
- Configuration validation speed
- Serialization/deserialization benchmarks
- Concurrent access performance
- Memory usage analysis

**Monitoring Performance:**
- Metrics processing speed
- Large dataset handling (10k+ records)
- Profile analysis performance
- Memory efficiency validation

```bash
# Run only performance tests
pytest tests/performance/ -v -m performance -s
```

## 🛠️ Test Infrastructure

### Mock Systems

**MockSupabaseClient** (`tests/fixtures/test_configs.py`):
- Simulates Supabase database operations
- Configurable failure modes for error testing
- Supports all CRUD operations
- Maintains in-memory state for testing

**TestConfigFixtures**:
- Default configuration templates
- Custom configuration variants
- Invalid configuration test cases
- Edge case configurations (min/max values)

**PerformanceTestData**:
- Sample query metrics (100 records)
- Performance profiles (50 records)
- Realistic data distributions
- Time-series test data

### Performance Benchmarking

The performance test suite includes:

- **Baseline Measurements**: Establish performance baselines
- **Regression Detection**: Detect performance regressions
- **Threshold Validation**: Ensure performance meets requirements
- **Statistical Analysis**: Calculate percentiles and statistics
- **Memory Profiling**: Track memory usage patterns

**Performance Thresholds:**
- Configuration loading (cold): < 100ms
- Configuration loading (warm): < 1ms
- Configuration validation: < 10ms
- Serialization operations: < 1ms
- Large dataset processing: < 10s
- Concurrent operations: < 10ms per operation

## 🎯 Usage Examples

### Running Specific Tests

```bash
# Test only configuration validation
pytest tests/unit/config/test_prompt_config.py::TestQueryClassificationConfig -v

# Test configuration lifecycle
pytest tests/integration/config_monitoring/test_config_integration.py::TestConfigurationIntegration::test_full_config_lifecycle -v

# Benchmark configuration loading
pytest tests/performance/profiling/test_performance_benchmarks.py::TestConfigurationPerformance::test_config_loading_performance -v -s
```

### Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only unit tests
pytest -m unit

# Run only slow tests
pytest -m slow

# Run only performance tests
pytest -m performance

# Skip slow tests
pytest -m "not slow"
```

### Coverage Analysis

```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Require minimum coverage
pytest --cov=src --cov-fail-under=80

# Coverage for specific modules
pytest --cov=src.config --cov-report=term-missing
```

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

- Minimum pytest version: 6.0
- Coverage reporting enabled
- Strict marker enforcement
- HTML and terminal coverage reports
- 80% minimum coverage requirement

### Global Fixtures (`conftest.py`)

- Async event loop configuration
- Test environment setup/teardown
- Performance test configuration
- Automatic test marking based on location

## 📈 Continuous Integration

### Test Pipeline

1. **Dependency Check**: Verify all required packages
2. **Smoke Tests**: Quick validation of core functionality
3. **Unit Tests**: Comprehensive component testing
4. **Integration Tests**: Workflow and interaction testing
5. **Performance Tests**: Benchmark validation
6. **Coverage Analysis**: Code coverage reporting

### Performance Monitoring

The test suite automatically tracks:
- Test execution times
- Memory usage during tests
- Performance regression detection
- Benchmark result comparison

## 🐛 Debugging Tests

### Verbose Output

```bash
# Maximum verbosity
pytest -vvv

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s
```

### Test Selection

```bash
# Run tests matching pattern
pytest -k "test_config"

# Run last failed tests
pytest --lf

# Run only failed tests, then all
pytest --ff
```

## 📝 Writing New Tests

### Unit Test Template

```python
import pytest
from src.config.prompt_config import YourComponent
from tests.fixtures.test_configs import TestConfigFixtures

class TestYourComponent:
    """Test YourComponent functionality."""
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        component = YourComponent()
        result = component.do_something()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async component functionality."""
        component = YourComponent()
        result = await component.async_operation()
        assert result is not None
```

### Integration Test Template

```python
@pytest.mark.asyncio
async def test_component_integration(self):
    """Test integration between components."""
    # Setup
    mock_client = MockSupabaseClient()
    component_a = ComponentA(mock_client)
    component_b = ComponentB()
    
    # Execute
    result = await component_a.interact_with(component_b)
    
    # Verify
    assert result.success is True
    assert len(mock_client.configs) > 0
```

### Performance Test Template

```python
@pytest.mark.asyncio
async def test_performance_benchmark(self, benchmark_suite):
    """Benchmark component performance."""
    component = YourComponent()
    iterations = 100
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        result = await component.operation()
        assert result is not None
    duration = time.perf_counter() - start_time
    
    benchmark_suite.record_benchmark("operation_name", duration, iterations)
    
    # Verify performance requirement
    avg_time = duration / iterations
    assert avg_time < 0.01, f"Operation too slow: {avg_time:.4f}s"
```

## 🎯 Best Practices

### Test Organization
- One test class per component
- Clear, descriptive test names
- Arrange-Act-Assert pattern
- Minimal setup/teardown

### Mocking Strategy
- Mock external dependencies (Supabase, APIs)
- Use dependency injection for testability
- Verify mock interactions
- Avoid over-mocking internal logic

### Performance Testing
- Establish baseline performance
- Test with realistic data sizes
- Include memory usage validation
- Monitor for performance regressions

### Coverage Goals
- Aim for 80%+ code coverage
- Focus on critical business logic
- Test error paths and edge cases
- Exclude trivial getter/setter methods

## 🚀 Automation

### Pre-commit Hooks

```bash
# Run smoke tests before commit
git config core.hooksPath .githooks
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: python tests/run_tests.py --install-deps
      - run: python tests/run_tests.py --type coverage
```

---

## 📞 Support

For questions about the testing framework:
1. Check test output for specific error messages
2. Review mock configurations in `tests/fixtures/`
3. Validate test environment setup
4. Check dependency installation

Happy testing! 🧪✨ 