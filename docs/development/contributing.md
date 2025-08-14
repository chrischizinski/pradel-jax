# Contributing to Pradel-JAX

Thank you for your interest in contributing to Pradel-JAX! This guide will help you get started with contributing code, documentation, or other improvements.

## 🎯 Ways to Contribute

### 🐛 Report Bugs
- Use our [bug report template](../../.github/ISSUE_TEMPLATE/bug_report.md)
- Include minimal reproducible examples
- Specify your environment (OS, Python version, JAX version)

### 💡 Suggest Features
- Check existing [issues](https://github.com/chrischizinski/pradel-jax/issues) first
- Use our [feature request template](../../.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the use case and expected behavior

### 📖 Improve Documentation
- Fix typos, clarify explanations, add examples
- Add missing documentation for new features
- Translate documentation to other languages

### 🔧 Contribute Code
- Fix bugs, add features, improve performance
- Add new model types or optimization strategies
- Improve test coverage and quality

## 🚀 Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/pradel-jax.git
cd pradel-jax

# Add upstream remote
git remote add upstream https://github.com/chrischizinski/pradel-jax.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .  # Install package in development mode

# Verify setup
python -c "import pradel_jax; print('Setup successful!')"
```

### 3. Run Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_optimization.py

# Run with coverage
python -m pytest --cov=pradel_jax --cov-report=html
```

### 4. Make Your Changes

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, add tests, update docs
# ...

# Run tests and linting
python -m pytest
python -m flake8 pradel_jax/
python -m black pradel_jax/
```

### 5. Submit Pull Request

```bash
# Commit your changes
git add .
git commit -m "Add descriptive commit message

- Detailed explanation of changes
- Reference any issues (#123)
- Breaking changes noted if any"

# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## 📋 Development Workflow

### Branch Management

- **`main`** - Stable releases, protected branch
- **`feature/description`** - New features and enhancements
- **`fix/description`** - Bug fixes
- **`docs/description`** - Documentation improvements

### Commit Guidelines

Follow conventional commit format:

```
type(scope): short description

Longer explanation if needed.

- Bullet points for multiple changes
- Reference issues: Fixes #123
- Note breaking changes: BREAKING CHANGE: ...
```

**Types:**
- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation changes
- `test:` - Test improvements
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes
- `ci:` - CI/CD changes

### Code Quality Standards

#### Python Code Style
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Type hints encouraged for public APIs

```bash
# Format code
black pradel_jax/

# Check linting
flake8 pradel_jax/

# Type checking (optional)
mypy pradel_jax/
```

#### Documentation Standards
- Docstrings in [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Examples in docstrings when helpful
- Update relevant documentation files

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Brief description of function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter with default
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    return len(param1) > param2
```

#### Testing Standards
- Unit tests for all public functions
- Integration tests for workflows
- Test coverage >90% for new code
- Use pytest fixtures for setup

```python
import pytest
from pradel_jax import PradelModel

class TestPradelModel:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_data()
    
    def test_model_creation(self):
        """Test basic model creation."""
        model = PradelModel()
        assert model is not None
        
    def test_likelihood_calculation(self, sample_data):
        """Test likelihood calculation with sample data."""
        model = PradelModel()
        # Test implementation...
```

## 🏗️ Architecture for Contributors

### Adding New Models

1. **Create model class** inheriting from `CaptureRecaptureModel`
2. **Implement required methods**: `log_likelihood`, `validate_data`, etc.
3. **Add comprehensive tests**
4. **Document with examples**
5. **Register in model registry**

```python
from pradel_jax.models.base import CaptureRecaptureModel

class MyNewModel(CaptureRecaptureModel):
    def __init__(self):
        super().__init__(ModelType.CUSTOM)
    
    def log_likelihood(self, parameters, data_context, design_matrices):
        # Implementation here
        pass
        
    def validate_data(self, data_context):
        # Validation logic
        super().validate_data(data_context)
```

### Adding Optimization Strategies

1. **Create optimizer class** inheriting from `BaseOptimizer`
2. **Implement optimization method**
3. **Add to strategy enum and factory**
4. **Test with various problem types**

```python
from pradel_jax.optimization.optimizers import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def optimize(self, objective_fn, initial_params, bounds=None):
        # Implementation here
        return OptimizationResult(...)
```

### Adding Data Formats

1. **Create adapter class** inheriting from `DataFormatAdapter`
2. **Implement parsing and validation**
3. **Add format detection logic**
4. **Test with various data files**

## 🧪 Testing Guidelines

### Test Organization

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_models.py
│   ├── test_optimization.py
│   └── test_formulas.py
├── integration/          # End-to-end integration tests
│   ├── test_full_workflow.py
│   └── test_rmark_comparison.py
└── fixtures/             # Test data and utilities
    ├── sample_data.py
    └── test_datasets/
```

### Running Different Test Types

```bash
# Quick unit tests
python -m pytest tests/unit/ -v

# Full integration tests (slower)
python -m pytest tests/integration/ -v

# Performance benchmarks
python -m pytest tests/benchmarks/ -v --benchmark-only

# Test specific functionality
python -m pytest -k "test_optimization" -v
```

### Test Data

- Use small, synthetic datasets for unit tests
- Include edge cases (empty data, single individual, etc.)
- Real data for integration tests (ensure it's safe to publish)
- Mock external dependencies (R, file systems)

## 📖 Documentation Guidelines

### Writing Good Documentation

1. **Start with why** - Explain the purpose and use case
2. **Provide examples** - Show real code that works
3. **Be concise but complete** - Cover all necessary details
4. **Link related topics** - Help users discover relevant information
5. **Update with changes** - Keep docs in sync with code

### Documentation Structure

- **User Guide** - How to use features (user perspective)
- **Tutorials** - Step-by-step walkthroughs with examples
- **API Reference** - Technical details (generated from docstrings)
- **Development** - Information for contributors

### Documentation Workflow

1. **Write docs alongside code** - Don't leave it for later
2. **Review docs in PRs** - Documentation is part of the feature
3. **Test examples** - Ensure code examples actually work
4. **Get feedback** - Have others review for clarity

## 🚀 Release Process

### Version Numbering

We use [semantic versioning](https://semver.org/):
- **Major** (1.0.0) - Breaking changes
- **Minor** (1.1.0) - New features, backward compatible  
- **Patch** (1.1.1) - Bug fixes

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version number updated
- [ ] Changelog updated
- [ ] Release notes prepared
- [ ] Security audit if needed

## 🤝 Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Be respectful, inclusive, and professional in all interactions.

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Questions, design discussions
- **Pull Requests** - Code review and collaboration
- **Email** - Private or sensitive matters

### Recognition

Contributors are recognized in:
- Release notes
- Contributors file
- Documentation credits
- Annual contributor highlights

## 📞 Getting Help

### For Contributors

- **Development questions** - GitHub Discussions
- **Code review help** - Tag maintainers in PRs
- **Architecture questions** - Create discussion thread
- **Blocked on something** - Reach out to maintainers

### Resources

- [Development Setup](setup.md) - Detailed environment setup
- [Testing Guide](testing.md) - Comprehensive testing information
- [Code Style Guide](style.md) - Coding standards and best practices
- [Architecture Overview](../user-guide/architecture.md) - System design

---

**Thank you for contributing to Pradel-JAX!** Every contribution, no matter how small, helps improve the framework for the entire community.

*Questions? Open an issue or discussion - we're here to help!* 🚀