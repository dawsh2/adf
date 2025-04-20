#!/usr/bin/env python
"""
Custom Test Runner for Algorithmic Trading System

This script runs all the unit and integration tests for the algorithmic trading system,
directly importing and running the test files rather than using module imports.

Usage:
    python custom_test_runner.py [options]

Options:
    --unit-only         Run only unit tests
    --integration-only  Run only integration tests
    --component=NAME    Run tests for a specific component
    --verbose           Show more detailed output
    --list              Just list the tests that would run
"""

import os
import sys
import unittest
import argparse
import importlib.util
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Define test directories
UNIT_TEST_DIR = os.path.join(project_root, 'tests/unit')
INTEGRATION_TEST_DIR = os.path.join(project_root, 'tests/integration')
COMPONENT_DIRS = {
    'core': [
        os.path.join(UNIT_TEST_DIR, 'core'),
        os.path.join(INTEGRATION_TEST_DIR, 'core')
    ],
    'data': [
        os.path.join(UNIT_TEST_DIR, 'data'),
        os.path.join(INTEGRATION_TEST_DIR, 'data')
    ],
    'models': [
        os.path.join(UNIT_TEST_DIR, 'models'),
        os.path.join(INTEGRATION_TEST_DIR, 'models')
    ],
    'strategy': [
        os.path.join(UNIT_TEST_DIR, 'strategy'),
        os.path.join(INTEGRATION_TEST_DIR, 'strategy')
    ],
    'execution': [
        os.path.join(UNIT_TEST_DIR, 'execution'),
        os.path.join(INTEGRATION_TEST_DIR, 'execution')
    ]
}


def find_test_files(start_dirs):
    """Find all test Python files in the given directories."""
    test_files = []
    
    for start_dir in start_dirs:
        if not os.path.exists(start_dir):
            logger.warning(f"Directory not found: {start_dir}")
            continue
            
        logger.info(f"Searching for tests in {start_dir}")
        
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(start_dir):
            # Find test files
            for filename in files:
                if filename.startswith('test_') and filename.endswith('.py'):
                    test_file = os.path.join(root, filename)
                    test_files.append(test_file)
    
    return test_files


def load_tests_from_file(file_path):
    """Load tests from a Python file directly."""
    logger.debug(f"Loading tests from {file_path}")
    
    # Generate a module name
    module_name = os.path.basename(file_path)[:-3]  # Remove .py extension
    
    try:
        # Create module spec
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.error(f"Could not create spec for {file_path}")
            return None
            
        # Create module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Find all test cases in the module
        test_suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        return test_suite
        
    except Exception as e:
        logger.error(f"Error loading tests from {file_path}: {e}")
        return None


def run_tests(test_files, verbosity=1, just_list=False):
    """Run tests from the given files."""
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Load tests from each file
    total_tests = 0
    successful_loads = 0
    
    for test_file in test_files:
        test_suite = load_tests_from_file(test_file)
        if test_suite:
            suite.addTest(test_suite)
            successful_loads += 1
            file_test_count = test_suite.countTestCases()
            total_tests += file_test_count
            logger.info(f"Loaded {file_test_count} tests from {os.path.basename(test_file)}")
    
    logger.info(f"Successfully loaded {successful_loads} of {len(test_files)} test files")
    logger.info(f"Total tests to run: {total_tests}")
    
    if just_list:
        # Just print tests that would be run
        def print_test(test):
            if isinstance(test, unittest.TestCase):
                print(f"  - {test.id()}")
            else:
                for t in test:
                    print_test(t)
        
        print("\nTests that would be run:")
        print_test(suite)
        return None
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Test Summary:")
    print(f"  Ran {result.testsRun} tests in {end_time - start_time:.2f} seconds")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run algorithmic trading system tests.')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--component', help='Run tests for a specific component')
    parser.add_argument('--verbose', action='store_true', help='Show more detailed output')
    parser.add_argument('--list', action='store_true', help='Just list the tests that would run')
    
    return parser.parse_args()


def get_test_dirs(args):
    """Get test directories based on command line arguments."""
    if args.component:
        if args.component not in COMPONENT_DIRS:
            logger.error(f"Invalid component: {args.component}")
            logger.info(f"Available components: {', '.join(COMPONENT_DIRS.keys())}")
            sys.exit(1)
        
        dirs = COMPONENT_DIRS[args.component]
        
        if args.unit_only:
            dirs = [d for d in dirs if '/unit/' in d]
        elif args.integration_only:
            dirs = [d for d in dirs if '/integration/' in d]
        
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
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get test directories
    test_dirs = get_test_dirs(args)
    
    # Find test files
    test_files = find_test_files(test_dirs)
    logger.info(f"Found {len(test_files)} test files")
    
    if not test_files:
        logger.warning("No test files found!")
        return
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    result = run_tests(test_files, verbosity, args.list)
    
    # Exit with appropriate code if tests were run
    if not args.list:
        sys.exit(0 if result and result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
