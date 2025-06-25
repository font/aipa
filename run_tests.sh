#!/bin/bash
set -e

echo "🧪 Running AIPA Test Suite"
echo "=========================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected. Activating venv..."
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        echo "❌ No venv found. Please create one with: python -m venv venv && source venv/bin/activate"
        exit 1
    fi
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -e ".[dev]" > /dev/null 2>&1

# Run linting
echo ""
echo "🔍 Running code quality checks..."
echo "  - Black formatting check..."
black --check --diff src tests || {
    echo "❌ Black formatting issues found. Run: black src tests"
    exit 1
}

echo "  - Import sorting check..."
isort --check-only --diff src tests || {
    echo "❌ Import sorting issues found. Run: isort src tests"
    exit 1
}

# Run tests
echo ""
echo "🧪 Running tests..."

echo "  - Unit tests (index optimization)..."
pytest tests/test_index_optimization.py -v -q

echo "  - Integration tests (API)..."
pytest tests/test_api_integration.py -v -q

echo "  - All tests with coverage..."
pytest --cov=src --cov-report=term-missing --cov-report=html -q

echo ""
echo "✅ All tests passed!"
echo ""
echo "📊 Coverage report generated in htmlcov/index.html"
echo "🎉 Test suite completed successfully!" 