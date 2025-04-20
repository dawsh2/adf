# Testing Guide for Algorithmic Trading System

This guide explains how to effectively test the algorithmic trading system using the comprehensive test suite we've developed.

## Test Structure

The testing framework is organized into two main categories:

1. **Unit Tests**: Tests for individual components in isolation
2. **Integration Tests**: Tests for how components work together

Tests are further organized by module:

- `core`: Event system and infrastructure components
- `data`: Data sources, handlers, and transformers
- `models`: Indicators, features, and rules
- `strategy`: Trading strategies and managers
- `execution`: Order execution, portfolio, and risk management

## Running Tests

### Prerequisites

Make sure you have all required packages installed:

```bash
pip install -r test_requirements.txt
```

### Using the Test Runner

The easiest way to run tests is using the `run_tests.py` script:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit-only

# Run only integration tests
python run_tests.py --integration-only

# Run tests for a specific component
python run_tests.py --component=data

# Generate HTML test report
python run_tests.py --html-report

# Show more detailed output
python run_tests.py --verbose
```

### Using pytest

You can also use pytest directly:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run integration tests only
pytest tests/integration

# Run tests for a specific component
pytest tests/unit/data tests/integration/data

# Run a specific test file
pytest tests/unit/data/test_csv_handler.py
```

## Writing New Tests

### Unit Tests

When writing unit tests, follow these guidelines:

1. Create one test class per component
2. Use descriptive test method names that explain what you're testing
3. Mock dependencies to isolate the component under test
4. Test both success and failure paths
5. Test edge cases and boundary conditions

Example:

```python
import unittest
from unittest.mock import MagicMock

class TestSomeComponent(unittest.TestCase):
    def setUp(self):
        # Setup test environment with mocks
        self.dependency = MagicMock()
        self.component = SomeComponent(self.dependency)
    
    def test_normal_operation(self):
        # Test normal operation
        result = self.component.do_something(input_data)
        self.assertEqual(result, expected_result)
    
    def test_error_handling(self):
        # Test error handling
        self.dependency.method.side_effect = Exception("Test error")
        result = self.component.do_something(input_data)
        self.assertIsNone(result)
```

### Integration Tests

For integration tests:

1. Test real interactions between components
2. Use temporary files and fixtures instead of mocks when possible
3. Test realistic scenarios that cover multiple components
4. Verify intermediate state and final results

Example:

```python
class TestComponentsIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create components with real dependencies
        self.data_source = CSVDataSource(self.temp_dir.name)
        self.data_handler = HistoricalDataHandler(self.data_source)
        self.strategy = SomeStrategy()
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        # Clean up
        self.temp_dir.cleanup()
    
    def _create_test_data(self):
        # Create test data files
        pass
    
    def test_end_to_end_flow(self):
        # Test the entire flow from data to strategy
        pass
```

## Coverage Reports

To generate a test coverage report:

```bash
# Generate coverage report
pytest --cov=src

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

This will show how much of your code is covered by tests.

## Continuous Integration

Set up GitHub Actions or another CI system to run tests automatically on pushes and pull requests. Example GitHub Actions workflow:

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r test_requirements.txt
    - name: Run tests
      run: |
        python run_tests.py --html-report
    - name: Upload test report
      uses: actions/upload-artifact@v2
      with:
        name: test-report
        path: test_report.html
```

## Best Practices

1. **Test Isolation**: Each test should run independently
2. **Fast Tests**: Tests should execute quickly for rapid feedback
3. **Deterministic Results**: Tests should produce the same result every time
4. **Clear Failures**: Failure messages should be descriptive
5. **Test Maintenance**: Update tests when code changes
6. **Test-Driven Development**: Consider writing tests before implementation

Remember that tests are an investment in code quality and maintainability. The time spent writing good tests pays off in reduced debugging time and fewer production issues.