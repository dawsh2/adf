#!/bin/bash
# Run tests for the algorithmic trading system

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure we're in the project root directory
cd "$SCRIPT_DIR"

# Add src to Python path
export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# Create necessary __init__.py files if they don't exist
echo "Ensuring all test directories have __init__.py files..."
mkdir -p tests/unit/core
mkdir -p tests/unit/data
mkdir -p tests/integration

for dir in tests tests/unit tests/unit/core tests/unit/data tests/integration; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "Creating $dir/__init__.py"
        touch "$dir/__init__.py"
    fi
done

# Check if we want to run specific tests
if [ "$1" == "core" ]; then
    # Run core event system tests
    echo "Running core event system tests..."
    python -m unittest tests.unit.core.test_events tests.unit.core.test_async_events
    
elif [ "$1" == "data" ]; then
    # Run data module tests
    echo "Running data module tests..."
    python -m unittest discover tests/unit/data
    
elif [ "$1" == "integration" ]; then
    # Run integration tests
    echo "Running integration tests..."
    python -m unittest discover tests/integration
    
elif [ "$1" == "data-integration" ]; then
    # Run data integration tests specifically
    echo "Running data integration tests..."
    python -m unittest tests.integration.test_data_event_integration
    
elif [ "$1" == "help" ] || [ "$1" == "-h" ]; then
    # Show help
    echo "Usage: ./run_tests.sh [option]"
    echo ""
    echo "Options:"
    echo "  core             Run core event system tests"
    echo "  data             Run data module unit tests"
    echo "  integration      Run all integration tests"
    echo "  data-integration Run data integration tests"
    echo "  help, -h         Show this help message"
    echo "  (no option)      Run all tests"
    
else
    # Run all tests
    echo "Running all tests..."
    python -m unittest discover tests
fi

echo "All tests completed."
