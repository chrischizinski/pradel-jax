# Pradel-JAX Test Suite

This directory contains the complete test suite for Pradel-JAX, organized by test type and scope.

## 🧪 Test Structure

### `unit/` - Unit Tests
Fast, isolated tests for individual components:
- `test_models.py` - Model implementation tests
- `test_optimization.py` - Optimization strategy tests  
- `test_formulas.py` - Formula parsing and design matrix tests
- `test_data.py` - Data loading and validation tests
- `test_utils.py` - Utility function tests

### `integration/` - Integration Tests  
End-to-end tests that verify complete workflows:
- `test_optimization_minimal.py` - Basic optimization integration
- `test_optimization_framework.py` - Full framework testing
- `test_optimization_integration.py` - Complex integration scenarios
- `test_simple_integration.py` - Simple workflow validation
- `test_pradel_jax_integration.py` - Complete system integration

### `benchmarks/` - Performance Tests
Performance and scalability testing:
- `benchmark_optimization.py` - Optimization performance
- `benchmark_likelihood.py` - Likelihood calculation speed
- `benchmark_large_data.py` - Large dataset handling

### `fixtures/` - Test Data and Utilities
Shared test utilities and data:
- `sample_data.py` - Synthetic test datasets
- `test_utilities.py` - Common test helper functions
- `mock_objects.py` - Mock objects for testing

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=pradel_jax --cov-report=html

# Run specific test type
python -m pytest tests/unit/          # Fast unit tests only
python -m pytest tests/integration/  # Integration tests only
```

### Detailed Test Commands

```bash
# Run tests with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/unit/test_models.py -v

# Run specific test function
python -m pytest tests/unit/test_models.py::test_pradel_model_creation -v

# Run tests matching pattern
python -m pytest -k "optimization" -v

# Run tests and stop on first failure
python -m pytest -x

# Run tests in parallel (if pytest-xdist installed)
python -m pytest -n auto
```

### Performance Testing
```bash
# Run benchmark tests
python -m pytest tests/benchmarks/ --benchmark-only

# Compare with baseline
python -m pytest tests/benchmarks/ --benchmark-compare=baseline

# Save benchmark results
python -m pytest tests/benchmarks/ --benchmark-save=current
```

## 🔧 Test Configuration

### `pytest.ini`
Configuration settings for pytest:
- Test discovery patterns
- Coverage settings  
- Marker definitions
- Output formatting

### `conftest.py`
Shared pytest fixtures and configuration:
- Sample data fixtures
- Mock objects
- Test utilities
- Parameterized test data

## 📊 Test Categories

### By Test Type

**🏃‍♂️ Fast Tests** (< 1 second each)
- Unit tests for individual functions
- Mock-based tests
- Simple integration tests

**🚶‍♂️ Medium Tests** (1-10 seconds each) 
- Integration tests with real data
- End-to-end workflow tests
- Small optimization problems

**🐌 Slow Tests** (> 10 seconds each)
- Large dataset tests
- Complex optimization scenarios
- Performance benchmarks
- RMark comparison tests

### By Scope

**Component Tests**
- Individual module testing
- Interface compliance
- Error handling

**Integration Tests** 
- Cross-module interaction
- Complete workflows
- Data pipeline tests

**System Tests**
- End-to-end scenarios
- Performance validation
- External tool integration

## 🎯 Test Guidelines

### Writing Good Tests

```python
# Good test structure
def test_specific_behavior():
    """Test one specific behavior clearly."""
    # Arrange
    data = create_test_data()
    model = PradelModel()
    
    # Act
    result = model.fit(data)
    
    # Assert
    assert result.success
    assert result.log_likelihood < 0
    assert len(result.parameters) == 3
```

### Test Naming Conventions

- `test_<function_name>_<scenario>` - Unit tests
- `test_<workflow_name>_integration` - Integration tests
- `benchmark_<operation>_performance` - Performance tests

### Test Documentation

```python
def test_optimization_convergence():
    """Test that optimization converges for well-behaved problems.
    
    This test verifies that:
    1. L-BFGS optimization converges
    2. Final gradient norm is small
    3. Parameter estimates are reasonable
    """
    # Test implementation...
```

## 🔍 Continuous Integration

### GitHub Actions Integration

Tests are automatically run on:
- Pull requests
- Pushes to main branch  
- Nightly schedules
- Release creation

### Test Matrix

Tests run across:
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **Operating systems**: Ubuntu, macOS, Windows
- **JAX versions**: Latest stable, development
- **Dependencies**: Minimum and latest versions

### Coverage Requirements

- **Minimum coverage**: 80% overall
- **New code coverage**: 90% for new features
- **Critical paths**: 95% coverage required
- **Integration tests**: All major workflows covered

## 🐛 Debugging Tests

### Common Issues

**Tests fail locally but pass in CI**
```bash
# Use same Python version as CI
pyenv install 3.9.18
pyenv local 3.9.18

# Install exact dependencies
pip install -r requirements-dev.txt
```

**Flaky tests (intermittent failures)**
```bash
# Run test multiple times
python -m pytest tests/test_flaky.py --count=10

# Use fixed random seeds in tests
np.random.seed(42)
```

**Slow test debugging**
```bash
# Profile test performance
python -m pytest --durations=10

# Run with minimal output
python -m pytest -q
```

### Test Data Issues

- Use synthetic data when possible
- Keep test datasets small (< 1MB)
- Mock external dependencies
- Ensure tests are deterministic

## 📈 Test Metrics

### Coverage Tracking

```bash
# Generate coverage report
python -m pytest --cov=pradel_jax --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

### Performance Tracking

```bash
# Run performance benchmarks
python -m pytest tests/benchmarks/ --benchmark-only

# Compare performance over time
python -m pytest tests/benchmarks/ --benchmark-compare

# Generate performance report
python -m pytest tests/benchmarks/ --benchmark-json=results.json
```

## 🤝 Contributing Tests

### Before Submitting

- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Test coverage maintained/improved
- [ ] Tests are documented
- [ ] Performance impact assessed

### Test Review Checklist

- [ ] Tests are focused and specific
- [ ] Edge cases are covered
- [ ] Error conditions are tested
- [ ] Tests are deterministic
- [ ] Test data is appropriate

---

## 📞 Test Support

**Having trouble with tests?**
- Check our [Testing Guide](../docs/development/testing.md)
- Review [Contributing Guidelines](../docs/development/contributing.md)  
- Ask in [GitHub Discussions](https://github.com/chrischizinski/pradel-jax/discussions)
- Open an [issue](https://github.com/chrischizinski/pradel-jax/issues) for test bugs

*Well-tested code is reliable code! 🧪*