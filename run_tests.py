#!/usr/bin/env python
"""
Test Runner for Algorithmic Trading System

This script runs all the unit and integration tests for the algorithmic trading system.
It discovers tests in the specified directories and runs them, generating a report.

Usage:
    python run_tests.py [options]

Options:
    --unit-only         Run only unit tests
    --integration-only  Run only integration tests
    --component=NAME    Run tests for a specific component
    --verbose           Show more detailed output
    --html-report       Generate HTML report of test results
"""

import os
import sys
import unittest
import argparse
import time
import importlib
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path so importing works correctly
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Define test directories
UNIT_TEST_DIR = 'tests/unit'
INTEGRATION_TEST_DIR = 'tests/integration'
COMPONENT_DIRS = {
    'core': ['tests/unit/core', 'tests/integration/core'],
    'data': ['tests/unit/data', 'tests/integration/data'],
    'models': ['tests/unit/models', 'tests/integration/models'],
    'strategy': ['tests/unit/strategy', 'tests/integration/strategy'],
    'execution': ['tests/unit/execution', 'tests/integration/execution']
}


def discover_tests(start_dirs: List[str]) -> unittest.TestSuite:
    """
    Discover tests in the given directories.
    
    Args:
        start_dirs: List of directories to start discovery
        
    Returns:
        TestSuite containing all discovered tests
    """
    suite = unittest.TestSuite()
    
    for start_dir in start_dirs:
        if os.path.exists(start_dir):
            logger.info(f"Discovering tests in {start_dir}")
            pattern = 'test_*.py'  # Standard test file pattern
            
            # Create loader and discover tests
            loader = unittest.defaultTestLoader
            discovered_tests = loader.discover(start_dir, pattern=pattern)
            
            # Debug output
            logger.debug(f"Found {discovered_tests.countTestCases()} test cases in {start_dir}")
            
            # Add them to the suite
            suite.addTest(discovered_tests)
        else:
            logger.warning(f"Test directory not found: {start_dir}")
    
    return suite


def run_tests(suite: unittest.TestSuite, verbosity: int = 1) -> unittest.TestResult:
    """
    Run the test suite.
    
    Args:
        suite: TestSuite to run
        verbosity: Verbosity level (1=normal, 2=verbose)
        
    Returns:
        TestResult with results of tests
    """
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def generate_html_report(result: unittest.TestResult, suite: unittest.TestSuite, output_file: str = 'test_report.html'):
    """
    Generate an HTML report of test results.
    
    Args:
        result: TestResult to generate report from
        suite: TestSuite with all tests
        output_file: File to write report to
    """
    try:
        import jinja2
    except ImportError:
        logger.error("jinja2 package not installed, cannot generate HTML report")
        logger.info("Install it with: pip install jinja2")
        return
    
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Algorithmic Trading System Test Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .summary { margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
            .passed { color: green; }
            .failed { color: red; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .error-details { margin-left: 20px; white-space: pre-wrap; font-family: monospace; background-color: #f8f8f8; padding: 10px; border-left: 3px solid #e74c3c; }
        </style>
    </head>
    <body>
        <h1>Algorithmic Trading System Test Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Run: {{ result.testsRun }}, Errors: {{ result.errors|length }}, Failures: {{ result.failures|length }}, Skipped: {{ result.skipped|length }}</p>
            <p class="{% if result.wasSuccessful() %}passed{% else %}failed{% endif %}">
                {% if result.wasSuccessful() %}All tests passed!{% else %}Tests failed!{% endif %}
            </p>
        </div>
        
        {% if result.failures %}
        <h2>Failures</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Error</th>
            </tr>
            {% for test, error in result.failures %}
            <tr>
                <td>{{ test }}</td>
                <td>
                    {{ error.splitlines()[0] }}
                    <div class="error-details">{{ error }}</div>
                </td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if result.errors %}
        <h2>Errors</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Error</th>
            </tr>
            {% for test, error in result.errors %}
            <tr>
                <td>{{ test }}</td>
                <td>
                    {{ error.splitlines()[0] }}
                    <div class="error-details">{{ error }}</div>
                </td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if result.skipped %}
        <h2>Skipped</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Reason</th>
            </tr>
            {% for test, reason in result.skipped %}
            <tr>
                <td>{{ test }}</td>
                <td>{{ reason }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        <h2>All Tests</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Status</th>
            </tr>
            {% for test in all_tests %}
            <tr>
                <td>{{ test }}</td>
                <td class="{% if test in failed_tests %}failed{% else %}passed{% endif %}">
                    {% if test in failed_tests %}Failed{% else %}Passed{% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    
    # Extract all test names
    all_tests = []
    failed_tests = []
    
    # This is a bit of a hack, but it works
    for test, _ in result.failures + result.errors:
        failed_tests.append(str(test))
    
    # Get all tests by iterating the suite
    def get_test_names(test_suite):
        test_names = []
        for test in test_suite:
            if isinstance(test, unittest.TestCase):
                test_names.append(str(test))
            else:
                # This is a test suite, recurse
                test_names.extend(get_test_names(test))
        return test_names
    
    all_tests = get_test_names(suite)
    
    # Render template
    template = jinja2.Template(template_str)
    html = template.render(
        result=result,
        all_tests=sorted(all_tests),
        failed_tests=failed_tests
    )
    
    # Write report
    with open(output_file, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report written to {output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run algorithmic trading system tests.')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--component', help='Run tests for a specific component')
    parser.add_argument('--verbose', action='store_true', help='Show more detailed output')
    parser.add_argument('--html-report', action='store_true', help='Generate HTML report of test results')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    
    return parser.parse_args()


def get_test_dirs(args) -> List[str]:
    """Get test directories based on command line arguments."""
    if args.component:
        if args.component not in COMPONENT_DIRS:
            logger.error(f"Invalid component: {args.component}")
            logger.info(f"Available components: {', '.join(COMPONENT_DIRS.keys())}")
            sys.exit(1)
        
        dirs = COMPONENT_DIRS[args.component]
        
        if args.unit_only:
            dirs = [d for d in dirs if 'unit' in d]
        elif args.integration_only:
            dirs = [d for d in dirs if 'integration' in d]
        
        return dirs
    
    if args.unit_only:
        return [UNIT_TEST_DIR]
    elif args.integration_only:
        return [INTEGRATION_TEST_DIR]
    else:
        return [UNIT_TEST_DIR, INTEGRATION_TEST_DIR]


def main():
    """Main function."""
    args = parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get test directories
    test_dirs = get_test_dirs(args)
    
    # Show debugging info
    if args.debug:
        logger.debug(f"Project root: {project_root}")
        logger.debug(f"Python path: {sys.path}")
        for test_dir in test_dirs:
            logger.debug(f"Looking for tests in: {os.path.abspath(test_dir)}")
    
    # Discover tests
    start_time = time.time()
    suite = discover_tests(test_dirs)
    
    # Count tests
    test_count = suite.countTestCases()
    logger.info(f"Discovered {test_count} tests")
    
    if test_count == 0:
        logger.warning("No tests found!")
        return
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    result = run_tests(suite, verbosity)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Test Summary:")
    print(f"  Ran {result.testsRun} tests in {execution_time:.2f} seconds")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    # Generate HTML report if requested
    if args.html_report:
        generate_html_report(result, suite)
    
    # Return with appropriate exit code
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
