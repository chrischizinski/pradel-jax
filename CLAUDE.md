# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Pradel-JAX repository.

## 🎯 Current Project Status (August 2025)

### ✅ **MAJOR MILESTONE COMPLETED**: Advanced Optimization Framework
The Pradel-JAX optimization framework is **fully integrated and production-ready** with advanced optimization capabilities:
- ✅ Complete JAX-based Pradel model implementation
- ✅ Intelligent optimization strategy selection (L-BFGS-B, SLSQP, Adam, multi-start, hybrid)  
- ✅ **NEW**: Hybrid optimization combining fast scipy with reliable multi-start fallback
- ✅ **NEW**: Adaptive JAX Adam with learning rate scheduling, early stopping, and warm restarts
- ✅ **NEW**: Automated performance regression testing with CI/CD integration
- ✅ Industry-standard performance monitoring and experiment tracking
- ✅ Comprehensive formula system with R-style syntax
- ✅ Robust error handling and validation framework
- ✅ Full integration test suite (all tests passing)
- ✅ Repository securely published on GitHub with data protection

### 🏗️ **Repository Structure (Newly Organized)**
```
pradel-jax/
├── 📚 docs/              # Comprehensive documentation hub
│   ├── user-guide/       # Installation, architecture, optimization
│   ├── tutorials/        # Quick start, examples, performance tips
│   ├── development/      # Contributing, setup, maintenance guides
│   └── security/         # Data protection and audit reports
├── 🧪 tests/             # Professional test suite
│   ├── integration/      # End-to-end workflow tests
│   ├── unit/            # Component-specific tests
│   ├── benchmarks/      # Performance testing
│   └── fixtures/        # Shared test data and utilities
├── 📦 pradel_jax/        # Main package (modular architecture)
│   ├── optimization/    # Advanced optimization framework
│   ├── models/          # Pradel model implementation
│   ├── formulas/        # R-style formula system
│   ├── data/           # Data handling and validation
│   └── core/           # Core APIs and abstractions
├── 📊 data/             # Safe datasets only (sensitive data protected)
├── 🔧 examples/         # Usage demonstrations
└── 📋 Root configs      # Requirements, setup, documentation
```

## 🚀 Development Environment Setup

### Quick Setup (Recommended)
```bash
# Clone and setup
git clone https://github.com/chrischizinski/pradel-jax.git
cd pradel-jax

# Quick setup (creates virtual environment and installs dependencies)
./quickstart.sh

# Activate environment
source pradel_env/bin/activate

# Verify installation
python -m pytest tests/integration/test_optimization_minimal.py -v
```

### Manual Setup
```bash
# Create virtual environment
python -m venv pradel_env
source pradel_env/bin/activate  # On Windows: pradel_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
pip install -e .  # Install package in development mode

# Run tests to verify setup
python -m pytest tests/ -v
```

### Core Dependencies
- **JAX Ecosystem**: `jax>=0.4.20`, `jaxlib>=0.4.20` - JIT compilation and autodiff
- **Optimization**: `scipy>=1.11.0` - Industry-standard optimizers (L-BFGS-B, SLSQP)  
- **Advanced Optimization**: `jaxopt>=0.8.0`, `optax>=0.1.7` - Modern JAX optimizers
- **Data**: `pandas>=1.5.0`, `numpy>=1.21.0` - Data handling and arrays
- **Testing**: `pytest>=7.0.0`, `pytest-cov>=4.0.0` - Comprehensive testing
- **Monitoring**: `mlflow>=2.0.0` - Experiment tracking and performance monitoring

## 🎯 Common Development Tasks

### Core Functionality Usage

#### **🚀 Basic Model Fitting (Primary Interface)**
```python
import pradel_jax as pj

# Load data (auto-detects format)
data_context = pj.load_data("data/dipper_dataset.csv")

# Create model specification  
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",    # Survival with sex effect
    p="~1 + sex",      # Detection with sex effect  
    f="~1"             # Constant recruitment
)

# Fit model (automatic optimization strategy selection)
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context
)

print(f"Success: {result.success}")
print(f"Strategy: {result.strategy_used}")
print(f"AIC: {result.aic:.2f}")
```

#### **🎯 Optimization Framework (Advanced)**
```python
from pradel_jax.optimization import optimize_model
from pradel_jax.models import PradelModel

# Create model and setup
model = PradelModel()
design_matrices = model.build_design_matrices(formula_spec, data_context)

# Define objective function
def objective(params):
    return -model.log_likelihood(params, data_context, design_matrices)

# Optimize with intelligent strategy selection
result = optimize_model(
    objective_function=objective,
    initial_parameters=model.get_initial_parameters(data_context, design_matrices),
    context=data_context,
    bounds=model.get_parameter_bounds(data_context, design_matrices)
)

# Or specify advanced optimization strategies
result = optimize_model(
    objective_function=objective,
    initial_parameters=initial_parameters,
    context=data_context,
    bounds=bounds,
    preferred_strategy=OptimizationStrategy.HYBRID  # Fast + reliable
)

# Or use adaptive Adam for gradient-based optimization
result = optimize_model(
    objective_function=objective,
    initial_parameters=initial_parameters,
    context=data_context,
    bounds=bounds,
    preferred_strategy=OptimizationStrategy.JAX_ADAM_ADAPTIVE
)
```

### Testing and Validation

#### **🧪 Running Tests**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test types
python -m pytest tests/unit/ -v           # Fast unit tests
python -m pytest tests/integration/ -v   # Integration tests
python -m pytest tests/benchmarks/ -v    # Performance tests

# Run with coverage
python -m pytest tests/ --cov=pradel_jax --cov-report=html

# Run specific integration test
python -m pytest tests/integration/test_optimization_minimal.py -v
```

#### **✅ Integration Validation**
```bash
# Minimal integration test (quick validation)
python tests/integration/test_optimization_minimal.py

# Simple integration test (basic workflow)  
python tests/integration/test_simple_integration.py

# Full framework test (comprehensive)
python tests/integration/test_optimization_framework.py
```

#### **📊 Performance Testing**
```bash
# Quick performance check (for development)
python scripts/monitor_performance.py --quick

# Comprehensive performance monitoring
python scripts/monitor_performance.py --full --output results.json

# Compare with baseline performance
python scripts/monitor_performance.py --compare

# Create new performance baselines
python scripts/monitor_performance.py --baseline

# Run regression tests (CI/CD)
python tests/benchmarks/test_performance_regression.py --mode test
```

### Documentation and Contribution

#### **📚 Documentation Development**
```bash
# Update todo list
python update_readme_todos.py --complete 2 --date "August 15, 2025"
python update_readme_todos.py --add "New feature description" --priority high
python update_readme_todos.py --focus "Working on performance benchmarking"

# Check documentation structure
find docs/ -name "*.md" | head -10
```

#### **🤝 Contributing Workflow**
```bash
# Create feature branch
git checkout -b feature/new-feature-name

# Make changes, add tests, update docs
# ...

# Run tests and checks
python -m pytest tests/ -v
python -m flake8 pradel_jax/
python -m black pradel_jax/

# Commit and push
git add .
git commit -m "feat: add new feature with comprehensive tests"
git push origin feature/new-feature-name
```

## 📋 Current Development Priorities

### **✅ Recently Completed (August 2025)**
- **🔧 Hybrid Optimization Framework** - Fast scipy with reliable multi-start fallback ✅
- **📊 Automated Performance Regression Testing** - CI/CD pipeline with baseline tracking ✅  
- **⚡ Optimized JAX Adam Parameters** - Adaptive learning rates and modern optimization techniques ✅

### **⭐ High Priority (Next 2-3 weeks)**
1. **📖 Enhanced Documentation & Examples** - Create comprehensive tutorials and API docs
2. **🔬 RMark Parameter Validation** - Implement side-by-side parameter comparison with new optimizers
3. **📊 Production Performance Validation** - Validate new optimization improvements on real datasets

### **🔧 Medium Priority (Next 1-2 months)**
4. **📈 Large-Scale Testing** - Test framework on realistic large datasets with new optimizers
5. **🎨 Production API Wrappers** - Create simplified interfaces for common workflows
6. **📋 Model Selection Tools** - Implement AIC/BIC comparison and diagnostics
7. **🔍 Memory Optimization** - Profile and optimize memory usage for large-scale problems

### **🎯 Lower Priority (Next 2-3 months)**  
8. **🔄 Batch Processing** - Add parallel dataset processing capabilities
9. **📊 Visualization Dashboard** - Create diagnostic plots and result visualization
10. **🔗 R Integration via Reticulate** - Create R package wrapper interface
11. **🧪 Advanced Optimization Strategies** - Implement Bayesian optimization and other advanced methods

## 🏗️ Architecture Overview

### **Design Principles**
1. **Modularity** - Clear separation of concerns with pluggable components
2. **Extensibility** - Easy to add new models, optimizers, data formats
3. **Performance** - JAX JIT compilation with GPU acceleration support
4. **Robustness** - Comprehensive error handling and validation
5. **Usability** - Intuitive APIs with excellent error messages

### **Core Modules**

**`pradel_jax.optimization`** - Advanced optimization framework
- `strategy.py` - Intelligent strategy selection based on problem characteristics
- `optimizers.py` - Industry-standard optimizers (L-BFGS-B, SLSQP, Adam, Hybrid)
- `adaptive_adam.py` - Advanced adaptive Adam with learning rate scheduling and early stopping
- `orchestrator.py` - High-level optimization coordination and monitoring
- `monitoring.py` - MLflow integration for experiment tracking

**`pradel_jax.models`** - Statistical model implementations  
- `base.py` - Abstract base classes for capture-recapture models
- `pradel.py` - Complete Pradel model with JAX-compiled likelihood

**`pradel_jax.formulas`** - R-style formula system
- `parser.py` - Formula parsing and validation 
- `spec.py` - Formula specifications and metadata
- `design_matrix.py` - Design matrix construction from formulas

**`pradel_jax.data`** - Data handling and validation
- `adapters.py` - Multiple data format support (RMark, Y-columns, custom)
- `validator.py` - Comprehensive data quality validation

### **Data Flow**
1. **Data Loading** → `DataFormatAdapter` → `DataContext` 
2. **Formula Parsing** → `FormulaParser` → `FormulaSpec`
3. **Design Matrices** → `build_design_matrices()` → JAX arrays
4. **Model Fitting** → `optimize_model()` → `OptimizationResult`
5. **Results** → Parameter estimates, diagnostics, model comparison

## 🔒 Security and Data Protection

### **Data Protection Status: SECURED ✅**
- ✅ Comprehensive .gitignore protecting sensitive Nebraska (NE) and South Dakota (SD) data
- ✅ All files containing person_id, customer_id, FUZZY identifiers blocked
- ✅ SSH connection scripts and credentials protected
- ✅ Complete security audit performed and documented
- ✅ Only public research data (dipper dataset) and synthetic data included

### **Safe Development Practices**
```bash
# Always check what files will be committed
git status

# Test if sensitive files are blocked
git check-ignore data/encounter_histories_ne_clean.csv  # Should return filename
git check-ignore data/dipper_dataset.csv              # Should return nothing (allowed)

# Security validation before commits
grep -r "person_id\|customer_id\|FUZZY" . --exclude-dir=archive || echo "No sensitive identifiers found"
```

## 🧠 Development Standards

### **Code Quality**
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Type hints encouraged for public APIs
- Comprehensive docstrings in Google style

### **Testing Standards**
- Unit tests for all public functions (>90% coverage target)
- Integration tests for complete workflows  
- Performance benchmarks for optimization components
- All tests must pass before merging

### **Documentation Standards**  
- Keep `docs/` folder comprehensive and up-to-date
- Update README todos when completing tasks
- Include examples in docstrings
- Maintain architecture documentation

### **Performance Considerations**
- Use JAX JIT compilation for computational bottlenecks
- Vectorized operations over Python loops
- Memory-efficient processing for large datasets
- Profile performance improvements with benchmarks

## 🔧 Troubleshooting

### **Common Issues**

**"Optimization failed to converge"**
```python
# Try different strategy
result = pj.fit_model(..., strategy="multi_start")

# Check data quality
pj.validate_data(data_context)
```

**"Formula parsing error"**  
```python
# Check available covariates
print(data_context.covariates.keys())

# Simplify formula
formula_spec = pj.create_formula_spec(phi="~1", p="~1", f="~1")
```

**"JAX compilation warnings"**
```python
# Ignore TPU warnings (expected on CPU-only systems)
import warnings
warnings.filterwarnings("ignore", ".*TPU.*")
```

### **Getting Help**
- **Documentation**: [docs/README.md](docs/README.md)
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Contributing**: [docs/development/contributing.md](docs/development/contributing.md)

---

## 🎯 Context for Claude Code

When working on this project, remember:

1. **Current Status**: Optimization framework is complete and production-ready
2. **Architecture**: Modern, modular design with clear separation of concerns  
3. **Testing**: Comprehensive test suite with integration validation
4. **Documentation**: Well-organized docs/ folder with user and developer guides
5. **Security**: Sensitive data properly protected, only safe data in repository
6. **Community**: Repository ready for open-source collaboration and contribution

The project represents a significant achievement in bringing modern JAX-based optimization to capture-recapture modeling while maintaining statistical rigor and usability.

**Focus Areas**: Documentation enhancement, performance benchmarking, RMark validation, and community building are the next key priorities.

---

*This file is updated as the project evolves. Last major update: August 14, 2025*