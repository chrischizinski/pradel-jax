# Development Environment Setup

Complete guide for setting up a Pradel-JAX development environment, including tools, dependencies, and development workflows.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Installation](#development-installation)
3. [Development Tools](#development-tools)
4. [Project Structure](#project-structure)
5. [Development Workflow](#development-workflow)
6. [Testing Setup](#testing-setup)
7. [Code Quality Tools](#code-quality-tools)
8. [IDE Configuration](#ide-configuration)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python 3.8+** (3.9+ recommended for best performance)
- **Git** for version control
- **Modern terminal** (Terminal on macOS, WSL2 on Windows, or Linux terminal)

### Recommended Tools

```bash
# Check Python version
python --version  # Should be 3.8+

# Check Git version
git --version     # Any recent version is fine

# Install build tools (if not already available)
# macOS: Xcode command line tools
xcode-select --install

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

## Development Installation

### 1. Repository Setup

```bash
# Fork the repository on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/pradel-jax.git
cd pradel-jax

# Add upstream remote for syncing
git remote add upstream https://github.com/chrischizinski/pradel-jax.git

# Verify remotes
git remote -v
# origin    https://github.com/YOUR_USERNAME/pradel-jax.git (fetch)
# origin    https://github.com/YOUR_USERNAME/pradel-jax.git (push)
# upstream  https://github.com/chrischizinski/pradel-jax.git (fetch)
# upstream  https://github.com/chrischizinski/pradel-jax.git (push)
```

### 2. Virtual Environment Setup

```bash
# Create isolated virtual environment
python -m venv pradel_dev_env
source pradel_dev_env/bin/activate  # On Windows: pradel_dev_env\Scripts\activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Verify activation
which python  # Should point to your virtual environment
python --version
```

### 3. Development Dependencies Installation

```bash
# Install development dependencies (includes everything needed)
pip install -r requirements-dev.txt

# Install the package in development mode (editable install)
pip install -e .

# Verify installation
python -c "import pradel_jax; print(pradel_jax.__version__)"
python -c "from pradel_jax.optimization import optimize_model; print('✓ Development installation successful')"
```

### 4. Pre-commit Hooks Setup

```bash
# Install pre-commit hooks for code quality
pre-commit install

# Test pre-commit setup
pre-commit run --all-files

# This will run:
# - Black (code formatting)
# - isort (import sorting)
# - flake8 (linting)
# - mypy (type checking)
# - pytest (tests)
```

### 5. Verify Development Setup

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run optimization performance tests
python -m pytest tests/benchmarks/test_optimization_performance.py -v

# Check code quality
python -m flake8 pradel_jax/
python -m black --check pradel_jax/
python -m mypy pradel_jax/

# Run an integration test
python tests/integration/test_optimization_minimal.py
```

## Development Tools

### Core Development Stack

| Tool | Purpose | Configuration File |
|------|---------|-------------------|
| **Black** | Code formatting | `pyproject.toml` |
| **isort** | Import sorting | `pyproject.toml` |
| **flake8** | Linting/style | `.flake8` |
| **mypy** | Type checking | `pyproject.toml` |
| **pytest** | Testing framework | `pytest.ini` |
| **pre-commit** | Git hooks | `.pre-commit-config.yaml` |

### Additional Tools (Optional but Recommended)

```bash
# Documentation tools
pip install sphinx sphinx-rtd-theme  # For generating docs

# Performance profiling
pip install line_profiler memory_profiler  # Performance analysis

# Advanced testing
pip install pytest-xdist pytest-benchmark  # Parallel testing, benchmarks

# Code coverage visualization
pip install coverage[toml] pytest-cov  # Coverage reporting
```

## Project Structure

```
pradel-jax/
├── 📁 pradel_jax/           # Main package
│   ├── 📁 core/             # Core APIs and abstractions
│   │   ├── api.py           # High-level user interface
│   │   ├── exceptions.py    # Exception classes
│   │   └── export.py        # Results export functionality
│   ├── 📁 data/             # Data loading and management
│   │   ├── adapters.py      # Data format adapters
│   │   ├── sampling.py      # Data sampling utilities
│   │   └── validation.py    # Data validation
│   ├── 📁 formulas/         # R-style formula system
│   │   ├── parser.py        # Formula parsing
│   │   ├── spec.py          # Formula specifications
│   │   └── design_matrix.py # Design matrix construction
│   ├── 📁 models/           # Model implementations
│   │   ├── base.py          # Abstract base classes
│   │   ├── pradel.py        # Pradel model implementation
│   │   └── results.py       # Model results classes
│   ├── 📁 optimization/     # Optimization framework
│   │   ├── strategy.py      # Strategy selection
│   │   ├── optimizers.py    # Optimizer implementations
│   │   ├── orchestrator.py  # High-level coordination
│   │   └── monitoring.py    # Performance monitoring
│   ├── 📁 inference/        # Statistical inference (WF-007)
│   │   ├── standard_errors.py    # Hessian-based standard errors
│   │   ├── confidence_intervals.py # CI computation
│   │   ├── bootstrap.py     # Bootstrap methods
│   │   └── hypothesis_tests.py # Statistical tests
│   ├── 📁 validation/       # RMark validation framework
│   │   ├── rmark_interface.py     # R interface
│   │   ├── parameter_comparison.py # Parameter validation
│   │   └── statistical_tests.py   # Concordance analysis
│   ├── 📁 config/           # Configuration management
│   │   └── settings.py      # Global configuration
│   └── 📁 utils/            # Utilities and helpers
│       ├── logging.py       # Structured logging
│       ├── transformations.py # Mathematical functions
│       └── monitoring.py    # Resource monitoring
├── 📁 tests/                # Test suite
│   ├── 📁 unit/             # Unit tests
│   ├── 📁 integration/      # Integration tests
│   ├── 📁 benchmarks/       # Performance benchmarks
│   └── 📁 fixtures/         # Test data and utilities
├── 📁 docs/                 # Documentation
│   ├── 📁 user-guide/       # User documentation
│   ├── 📁 tutorials/        # Step-by-step guides
│   ├── 📁 api/              # API reference
│   └── 📁 development/      # Developer docs
├── 📁 examples/             # Usage examples
│   ├── 📁 benchmarks/       # Performance examples
│   ├── 📁 nebraska/         # Real-data analyses
│   └── 📁 validation/       # RMark comparison
└── 📁 scripts/              # Utility scripts
    ├── 📁 analysis/         # Data analysis scripts
    ├── 📁 benchmarks/       # Performance testing
    └── 📁 validation/       # RMark validation scripts
```

### Key Development Files

```bash
# Configuration files
pyproject.toml          # Python project configuration
requirements.txt        # Production dependencies  
requirements-dev.txt    # Development dependencies
pytest.ini             # Pytest configuration
.pre-commit-config.yaml # Pre-commit hooks
.flake8                 # Flake8 configuration
.gitignore             # Git ignore patterns

# Setup and deployment
setup.py               # Package setup (legacy)
quickstart.sh          # Quick setup script
```

## Development Workflow

### 1. Feature Development Workflow

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes, commit regularly
git add .
git commit -m "feat: add new optimization strategy"

# 4. Run tests and quality checks
pre-commit run --all-files
python -m pytest tests/

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Create pull request on GitHub
```

### 2. Bug Fix Workflow

```bash
# 1. Create bug fix branch
git checkout -b fix/issue-number-description

# 2. Write failing test first (TDD approach)
# Edit tests/unit/test_relevant_module.py
# Add test that reproduces the bug

# 3. Run test to confirm it fails
python -m pytest tests/unit/test_relevant_module.py::test_bug_case -v

# 4. Fix the bug
# Edit relevant source files

# 5. Verify test now passes
python -m pytest tests/unit/test_relevant_module.py::test_bug_case -v

# 6. Run full test suite
python -m pytest tests/
```

### 3. Documentation Update Workflow

```bash
# 1. Create documentation branch
git checkout -b docs/update-user-guide

# 2. Update relevant documentation files
# Edit files in docs/

# 3. Test documentation builds (if applicable)
# Build and review documentation locally

# 4. Commit and push
git add docs/
git commit -m "docs: improve user guide clarity"
git push origin docs/update-user-guide
```

## Testing Setup

### Test Categories

1. **Unit Tests** (`tests/unit/`) - Test individual functions and classes
2. **Integration Tests** (`tests/integration/`) - Test complete workflows
3. **Benchmarks** (`tests/benchmarks/`) - Performance testing
4. **Fixtures** (`tests/fixtures/`) - Shared test data and utilities

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/unit/ -v           # Fast unit tests
python -m pytest tests/integration/ -v   # Slower integration tests
python -m pytest tests/benchmarks/ -v    # Performance tests

# Run specific test file
python -m pytest tests/unit/test_optimization.py -v

# Run specific test function
python -m pytest tests/unit/test_optimization.py::test_lbfgs_optimization -v

# Run tests with coverage
python -m pytest tests/ --cov=pradel_jax --cov-report=html

# Run tests in parallel (faster)
python -m pytest tests/ -n auto  # Uses all CPU cores
```

### Test Data Management

```bash
# Test data location
tests/fixtures/
├── small_dataset.csv      # Small test dataset (< 100 individuals)
├── medium_dataset.csv     # Medium test dataset (1000-5000 individuals)
├── dipper_dataset.csv     # Classic ecological dataset
└── synthetic_data.py      # Synthetic data generation

# Generate test data
python tests/fixtures/synthetic_data.py --n_individuals 100 --n_occasions 5
```

### Writing Tests

Example test structure:

```python
# tests/unit/test_new_feature.py
import pytest
import numpy as np
import pradel_jax as pj


class TestNewFeature:
    """Test suite for new feature."""
    
    def setup_method(self):
        """Set up test fixtures before each test."""
        self.data = pj.load_data("tests/fixtures/small_dataset.csv")
        self.formula = pj.create_formula_spec(phi="~1", p="~1", f="~1")
    
    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        result = pj.fit_model(formula=self.formula, data=self.data)
        assert result.success
        assert result.aic > 0
        
    def test_error_handling(self):
        """Test that errors are handled appropriately."""
        with pytest.raises(pj.ModelSpecificationError):
            pj.create_formula_spec(phi="~invalid_covariate")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal data
        minimal_data = self.data.sample(n_individuals=2)
        result = pj.fit_model(formula=self.formula, data=minimal_data)
        # Should handle gracefully
        
    @pytest.mark.parametrize("strategy", ["lbfgs", "slsqp", "adam"])
    def test_multiple_strategies(self, strategy):
        """Test feature works with different optimization strategies."""
        result = pj.fit_model(
            formula=self.formula, 
            data=self.data, 
            strategy=strategy
        )
        assert result.success
```

## Code Quality Tools

### Black - Code Formatting

```bash
# Format code automatically
python -m black pradel_jax/
python -m black tests/

# Check formatting without changing files
python -m black --check pradel_jax/

# Configuration in pyproject.toml:
# [tool.black]
# line-length = 88
# target-version = ['py38']
```

### isort - Import Sorting

```bash
# Sort imports automatically  
python -m isort pradel_jax/
python -m isort tests/

# Check import sorting
python -m isort --check-only pradel_jax/

# Configuration in pyproject.toml:
# [tool.isort]
# profile = "black"
# line_length = 88
```

### flake8 - Linting

```bash
# Check code quality
python -m flake8 pradel_jax/

# Configuration in .flake8:
# [flake8]
# max-line-length = 88
# extend-ignore = E203, W503
```

### mypy - Type Checking

```bash
# Check type annotations
python -m mypy pradel_jax/

# Configuration in pyproject.toml:
# [tool.mypy]
# python_version = "3.8"
# warn_return_any = true
# warn_unused_configs = true
```

### Pytest - Testing

```bash
# Configuration in pytest.ini:
# [tool:pytest]
# testpaths = tests
# python_files = test_*.py
# python_classes = Test*
# python_functions = test_*
```

## IDE Configuration

### VS Code Setup

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./pradel_dev_env/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true
    }
}
```

### PyCharm Setup

1. **Interpreter**: Set to `./pradel_dev_env/bin/python`
2. **Code Style**: Import Black style settings
3. **Testing**: Configure pytest as test runner
4. **Version Control**: Configure Git integration

### Vim/Neovim Setup

For vim users, consider using:
- **Python-mode** or **Jedi-vim** for Python development
- **ALE** for linting integration
- **Fugitive** for Git integration

## Troubleshooting

### Common Setup Issues

#### 1. Virtual Environment Problems

```bash
# If virtual environment activation fails:
python -m venv --clear pradel_dev_env
source pradel_dev_env/bin/activate

# Verify Python path
which python
python -c "import sys; print(sys.prefix)"
```

#### 2. Dependency Installation Failures

```bash
# Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install dependencies one by one to isolate issues
pip install jax jaxlib
pip install scipy numpy pandas

# Clear pip cache if needed
pip cache purge
```

#### 3. JAX Installation Issues

```bash
# For CPU-only systems
pip install jax[cpu]

# For CUDA systems
pip install jax[cuda11_pip]  # or cuda12_pip

# Check JAX installation
python -c "import jax; print(jax.devices())"
```

#### 4. Pre-commit Hook Failures

```bash
# Update pre-commit
pre-commit autoupdate

# Clear pre-commit cache
pre-commit clean

# Reinstall hooks
pre-commit uninstall
pre-commit install
```

#### 5. Test Failures

```bash
# Run specific failing test with more verbosity
python -m pytest tests/path/to/failing_test.py::test_function -v -s

# Check test dependencies
pip install -r requirements-dev.txt

# Clear test cache
python -m pytest --cache-clear
```

### Getting Help

If you encounter setup issues:

1. **Check existing issues** - Search GitHub issues for similar problems
2. **Check documentation** - Review this guide and other docs
3. **Ask for help** - Create a discussion or issue on GitHub
4. **Include details** - OS, Python version, error messages, steps to reproduce

### Performance Optimization

For development performance:

```bash
# Enable JAX 64-bit mode for higher precision
export JAX_ENABLE_X64=True

# Use multiple CPU cores for testing
python -m pytest tests/ -n auto

# Profile code during development
python -m cProfile -o profile.prof your_script.py
python -m snakeviz profile.prof  # Visualize profile
```

---

**Next Steps:**
- [Contributing Guidelines](contributing.md) - Code contribution standards and process
- [Testing Guide](testing.md) - Comprehensive testing documentation  
- [Code Style Guide](style.md) - Coding standards and best practices
- [Release Process](releases.md) - How releases are managed and deployed