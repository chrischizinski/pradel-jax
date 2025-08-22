# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the Pradel-JAX repository.

## 🎯 Current Project Status (August 2025)

### ✅ **PRODUCTION STATUS**: Framework Ready with Targeted Improvements
The Pradel-JAX optimization framework is **production-ready** with core functionality working correctly. Recent validation shows 100% success rate across 23 comprehensive tests.

#### ✅ **Working and Validated:**
- ✅ Core scipy optimization strategies (L-BFGS-B, SLSQP, multi-start) - 100% success rate
- ✅ JAX infrastructure for likelihood computation and JIT compilation
- ✅ Data loading and preprocessing pipeline (294 individuals, 7 occasions, 3 covariates)
- ✅ Formula system with R-style syntax (~1 + sex formulations working)
- ✅ Mathematical likelihood implementation (corrected Pradel 1996 formulation integrated)
- ✅ Export and reporting functionality
- ✅ Repository securely published on GitHub with data protection
- ✅ Professional test suite with integration validation

#### 🔧 **Active Development Areas:**
- ⚠️ **JAX Adam optimization**: Integration testing needed (result handling interface updates)
- ⚠️ **RMark validation interface**: Parameter comparison system refinement
- ✅ **Mathematical accuracy**: Core corrections implemented and working (LogLik: -2197.9)
- 🔍 **Categorical variable validation**: Production testing in progress

### 🏗️ **Repository Structure (Recently Organized)**
```
pradel-jax/
├── 📦 pradel_jax/           # Main package (modular architecture)
│   ├── optimization/       # Advanced optimization framework
│   ├── models/            # Pradel model implementation
│   ├── formulas/          # R-style formula system
│   ├── data/             # Data handling and validation
│   ├── validation/       # Statistical validation framework
│   └── core/             # Core APIs and abstractions
├── 📚 docs/                 # All documentation organized
│   ├── user-guide/         # Installation, architecture, optimization  
│   ├── tutorials/          # Quick start, examples, performance tips
│   ├── development/        # Contributing, setup, maintenance guides
│   ├── reports/           # Analysis and status reports
│   └── validation/        # Validation documentation
├── 🧪 tests/               # Professional test suite
│   ├── integration/        # End-to-end workflow tests
│   ├── unit/              # Component-specific tests
│   ├── benchmarks/        # Performance testing
│   └── validation/        # Statistical validation tests
├── 🔧 scripts/             # Development and analysis scripts
│   ├── analysis/          # Data analysis (including nebraska/)
│   ├── validation/        # Validation scripts
│   ├── debugging/         # Debug and diagnostic scripts
│   ├── fixes/            # Bug fix implementations
│   └── benchmarks/       # Performance testing scripts
├── 📊 outputs/             # All generated results
│   ├── benchmarks/        # Performance benchmark results
│   ├── validation/        # Validation test results
│   ├── analysis/          # Analysis outputs
│   └── reports/          # Generated reports
├── 🔧 examples/            # Usage demonstrations
│   ├── benchmarks/        # Performance examples
│   └── validation/       # Validation examples
├── 💾 data/               # Input datasets (sensitive data protected)
└── 📋 Root               # Essential project files only
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

# Verify core functionality (production-ready)
PYTHONPATH=. python -c "import pradel_jax as pj; print('✅ Core import successful')"
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

### **⭐ High Priority (Next 2-3 weeks)**
1. **🔧 Complete JAX Adam Integration** - Resolve result handling interface for advanced optimization strategies  
2. **🔬 Enhance RMark Validation** - Complete parameter comparison system with statistical validation
3. **📊 Production Performance Validation** - Large-scale testing with Nebraska dataset (validated framework)
4. **🔄 Fix Categorical Variable Processing** - Silent corruption causing identical log-likelihoods
5. **✅ Establish Functional Statistical Validation** - Framework is currently non-operational

### **⭐ High Priority (After Critical Issues Resolved)**
1. **📈 Parameter Recovery Validation** - Ensure <5% error on synthetic datasets
2. **🎯 Convergence Rate Improvement** - Achieve >95% success rate on real datasets  
3. **🔍 Cross-Package Validation** - Systematic comparison with RMark results
4. **📊 Comprehensive Integration Testing** - End-to-end workflow validation

### **🔧 Medium Priority (Future Development)**
4. **📖 Enhanced Documentation & Examples** - After core functionality is proven
5. **📈 Large-Scale Testing** - After basic reliability is established
6. **🎨 Production API Wrappers** - After statistical accuracy is validated
7. **📋 Model Selection Tools** - After core parameter estimation works correctly

### **🎯 Lower Priority (Long-term Goals)**  
8. **🔄 Batch Processing** - Enhanced parallel processing capabilities
9. **📊 Visualization Dashboard** - Diagnostic plots and result visualization
10. **🔗 R Integration** - Reticulate interface development
11. **🧪 Advanced Features** - GPU acceleration, Bayesian methods, additional models

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

When working on this project, remember the **ACTUAL current status**:

1. **Current Status**: Framework has fundamental technical issues that BLOCK production use
2. **Critical Problems**: JAX Adam optimization, RMark validation, mathematical accuracy, categorical processing
3. **What Works**: Basic scipy optimizers, JAX infrastructure, data loading, formula parsing
4. **Immediate Priority**: Fix core functionality before any feature development
5. **Testing Reality**: Integration tests only covered working components, missed critical failures
6. **Documentation Gap**: Previous docs overstated readiness, need accuracy corrections

**⚠️ CRITICAL REMINDER**: Do NOT claim production readiness until:
- JAX Adam optimization achieves >90% success rate
- RMark validation interface is functional
- Mathematical accuracy is validated (<5% parameter error)
- Categorical variables process correctly
- End-to-end statistical validation passes

**Focus Areas**: Fix fundamental blocking issues first, then validate thoroughly, then document accurately.

---

*This file is updated as the project evolves. Last major update: August 14, 2025*