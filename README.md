# Pradel-JAX: Flexible Capture-Recapture Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/user/pradel-jax)

A modern, extensible framework for capture-recapture analysis using JAX, designed to be robust, flexible, and user-friendly.

## 📚 Documentation

**👉 [Complete Documentation](docs/README.md)** | **🚀 [Quick Start](docs/tutorials/quickstart.md)** | **🏗️ [Architecture](docs/user-guide/architecture.md)**

- **[User Guide](docs/user-guide/)** - Installation, model specification, optimization strategies
- **[Tutorials](docs/tutorials/)** - Step-by-step examples and walkthroughs  
- **[API Reference](docs/api/)** - Technical documentation for all modules
- **[Development](docs/development/)** - Contributing guidelines and setup instructions

## 🚀 Features

### ✅ Implemented (v2.0.0-alpha)

- **🔧 Robust Data Handling**: Automatically detects and handles multiple data formats
  - RMark format (ch column + covariates)
  - Y-column format (Y2016, Y2017, etc.) 
  - Generic format with custom column specifications
  - Intelligent capture history parsing (preserves leading zeros)
  - Automatic covariate detection and processing

- **⚙️ Flexible Configuration System**: Hierarchical configuration with multiple sources
  - YAML configuration files
  - Environment variables
  - Runtime configuration updates
  - User-specific settings

- **🛡️ Rich Error Handling**: Informative errors with actionable suggestions
  - Structured error messages with context
  - Specific suggestions for resolution
  - Documentation links
  - Error codes for programmatic handling

- **📊 Comprehensive Logging**: Structured logging with multiple outputs
  - Colored console output
  - File logging
  - Performance tracking
  - Context-aware messages

- **✅ Robust Validation**: Multi-level data validation with clear feedback
  - Capture matrix validation
  - Covariate compatibility checking
  - Array dimension validation
  - Parameter constraint validation

### 🚧 In Progress

- **📝 Formula System**: R-style formula parsing and evaluation
- **🏗️ Model Registry**: Plugin-based model system
- **🎯 Optimization Framework**: Multiple optimization strategies
- **🔬 Validation Framework**: Comparison with RMark/MARK

### 📋 Planned Features

- **Multi-model Support**: CJS, Multi-state, Robust design models
- **Advanced Formulas**: Splines, interactions, custom functions
- **GPU Acceleration**: JAX-based GPU computing
- **Model Selection**: AIC, cross-validation, model averaging
- **Diagnostic Tools**: Residual analysis, goodness-of-fit
- **R Integration**: Seamless RMark comparison and validation

## 🏗️ Architecture

The new pradel-jax follows a clean, modular architecture:

```
pradel_jax/
├── config/          # Configuration management
├── core/            # Core abstractions and APIs
├── data/            # Data format adapters
├── models/          # Model implementations
├── optimization/    # Optimization strategies  
├── validation/      # Validation framework
└── utils/           # Utilities and helpers
```

### Key Design Principles

1. **Modularity**: Clear separation of concerns with pluggable components
2. **Extensibility**: Easy to add new data formats, models, and optimizers
3. **Robustness**: Comprehensive error handling and validation
4. **Performance**: JAX-based computation with GPU support
5. **Usability**: Clear APIs with excellent error messages

## 🧪 Quick Test

```bash
# Run the test suite
python -m pytest tests/

# Run a specific integration test  
python -m pytest tests/integration/test_optimization_minimal.py -v

# Check test coverage
python -m pytest --cov=pradel_jax --cov-report=html
```

Or test the framework directly:
```python
import pradel_jax as pj

# Test with sample data
data_context = pj.load_data("data/test_datasets/dipper_dataset.csv")
print(f"Loaded {data_context.n_individuals} individuals with {data_context.n_occasions} occasions")

# Run a simple optimization
result = pj.fit_simple_model(data_context)
print(f"Optimization successful: {result.success}")
```

## 📊 Architecture Comparison

| Feature | Old System | New System |
|---------|------------|------------|
| **Data Formats** | Hardcoded Y-columns only | Extensible adapter system |
| **Error Handling** | Cryptic messages | Rich errors with suggestions |
| **Configuration** | Scattered constants | Unified config system |
| **Validation** | Basic checks | Comprehensive validation |
| **Extensibility** | Monolithic | Modular plugin architecture |
| **User Experience** | Developer-focused | User-friendly with guidance |

## 🔧 Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd pradel-jax

# Setup environment
./quickstart.sh

# Test installation
python examples/test_new_architecture.py
```

## 📖 Current Status

### ✅ Phase 1: Foundation (Complete)
- [x] Clean directory structure
- [x] Configuration system with Pydantic validation
- [x] Rich exception handling with suggestions
- [x] Structured logging with colored output
- [x] Data format abstraction layer
- [x] RMark format adapter with robust parsing
- [x] Comprehensive validation utilities

### 🚧 Phase 2: Core Functionality (In Progress)
- [ ] Formula system with R-style syntax
- [ ] Model registry framework
- [ ] Basic Pradel model implementation
- [ ] Optimization strategy framework

### 📋 Phase 3: Advanced Features (Planned)
- [ ] Multi-start optimization
- [ ] GPU acceleration
- [ ] RMark validation framework
- [ ] Model selection tools
- [ ] Performance benchmarking

## 🎯 Next 10 Priority Tasks

> **📅 Last Updated:** August 14, 2025n framework integration complete, repository published securely

### ⭐ High Priority (Next 2-3 weeks)

6. **📊 Performance Benchmarking** - Validate optimization results against existing RMark validation data and historical test cases
7. **🔬 RMark Parameter Validation** - Implement side-by-side parameter estimation comparison with RMark to ensure statistical consistency

### 🔧 Medium Priority (Next 1-2 months)

8. **📈 Large-Scale Testing** - Test optimization framework performance and memory usage on realistic large datasets (wf.dat and similar)
9. **🚀 CI/CD Pipeline** - Set up GitHub Actions for automated testing, linting, and continuous integration
10. **🎨 Production API Wrappers** - Create simplified interfaces and convenience functions for common modeling workflows
11. **📋 Model Selection Tools** - Implement AIC/BIC comparison, convergence diagnostics, and automated model selection

### 🎯 Lower Priority (Next 2-3 months)

12. **🔄 Batch Processing** - Add capabilities for processing multiple datasets and model comparisons in parallel
13. **📊 Visualization Dashboard** - Create diagnostic plots, convergence monitoring, and result visualization tools
14. **🔗 R Integration via Reticulate** - Create R package wrapper to use Pradel-JAX from R through reticulate interface
15. **🌐 Community Features** - Add discussion templates, contribution guidelines, and issue templates for GitHub collaboration

### 📝 Progress Tracking

**✅ Recently Completed:**
- Create comprehensive tutorials, API docs, and practical usage examples for the optimization framework *(August 14, 2025)*
- Full integration test suite validation *(August 14, 2025)*

**🚧 Current Focus:** Enhanced documentation and testing organizationnhanced documentation and performance validation

---

> **📌 Note:** This todo list is actively maintained and updated as work progresses. Check commit history for latest changes and completed items.

## 🚨 Breaking Changes from v1.x

The v2.0 redesign introduces breaking changes for better long-term maintainability:

- **New Import Structure**: `import pradel_jax as pj` 
- **Configuration System**: Settings now managed through `PradelJaxConfig`
- **Data Loading**: `pj.load_data()` instead of `DataHandler`
- **Error Handling**: Rich exceptions instead of simple errors

## 🤝 Contributing

This redesign creates a solid foundation for contributions:

- **Add Data Formats**: Implement new `DataFormatAdapter` subclasses
- **Add Models**: Register new model types in the model registry
- **Add Optimizers**: Implement `OptimizationStrategy` subclasses
- **Improve Validation**: Add new validation checks and diagnostics

## 📄 License

MIT License - see LICENSE file for details.

---

**Status**: Active development of robust, extensible capture-recapture analysis framework
**Version**: 2.0.0-alpha (Foundation complete, core functionality in progress)