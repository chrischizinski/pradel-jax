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

### ✅ Recently Completed

- **🔬 RMark Parameter Validation**: Industry-standard statistical validation framework
  - **Phase 1**: Core validation framework and secure execution ✅
  - **Phase 2**: Advanced statistical testing and model concordance analysis ✅
    - Bootstrap confidence intervals (Basic, Percentile, BCa, Studentized)
    - Comprehensive concordance analysis (Lin's CCC, Bland-Altman)
    - Cross-validation stability testing
    - Publication-ready statistical reporting
  - Multi-environment execution (SSH, local R, mock validation)
  - Statistical equivalence testing (TOST methodology) 
  - Parameter comparison with confidence interval analysis
  - Model ranking concordance testing

### ✅ Major Milestones Completed

- **🎯 Optimization Framework**: Complete JAX-based optimization with intelligent strategy selection ✅
- **📝 Formula System**: R-style formula parsing and evaluation ✅
- **🏗️ Model Registry**: Plugin-based model system ✅
- **🔬 RMark Parameter Validation Framework**: Complete 3-phase implementation ✅
  - **Phase 1**: Core validation framework and secure execution ✅
  - **Phase 2**: Advanced statistical testing and model concordance analysis ✅
  - **Phase 3**: Automated pipeline with quality gates ✅
    - Flexible configuration system with environment-specific settings
    - End-to-end validation orchestration with parallel processing
    - Quality gate evaluation framework with configurable criteria
    - Comprehensive error handling and recovery mechanisms
    - Publication-ready reporting and monitoring capabilities

### 🚧 Current Focus

- **📊 Performance Benchmarking**: Validating optimization results against RMark validation data
- **📈 Large-Scale Testing**: Framework performance on realistic datasets
- **🚀 Production Deployment**: Comprehensive documentation and deployment preparation

### 📋 Planned Features

- **Multi-model Support**: CJS, Multi-state, Robust design models
- **Advanced Formulas**: Splines, interactions, custom functions
- **GPU Acceleration**: JAX-based GPU computing
- **Model Selection**: AIC, cross-validation, model averaging
- **Diagnostic Tools**: Residual analysis, goodness-of-fit
- **Advanced R Integration**: Enhanced RMark integration beyond validation

## 🏗️ Architecture

The new pradel-jax follows a clean, modular architecture:

```
pradel_jax/
├── config/          # Configuration management
├── core/            # Core abstractions and APIs
├── data/            # Data format adapters
├── models/          # Model implementations
├── optimization/    # Optimization strategies  
├── validation/      # RMark parameter validation framework (NEW)
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

### ✅ Phase 3: Advanced Features (Complete)
- [x] Automated pipeline orchestration with quality gates
- [x] Parallel processing and resource management
- [x] Comprehensive error handling and recovery
- [x] Flexible configuration system
- [x] Production-ready validation framework

## 🎯 Development Roadmap

> **📅 Last Updated:** August 14, 2025 - Phase 3 validation framework complete, moving to production readiness

### ⭐ High Priority (Next 2-3 weeks)

1. **📊 Performance Benchmarking** - Validate optimization results against existing RMark validation data and historical test cases
2. **📈 Large-Scale Testing** - Test framework performance and memory usage on realistic large datasets (wf.dat and similar)
3. **🚀 Production Deployment** - Comprehensive documentation and deployment preparation

### 🔧 Medium Priority (Next 1-2 months)

4. **🚀 CI/CD Pipeline** - Set up GitHub Actions for automated testing, linting, and continuous integration
5. **🎨 Production API Wrappers** - Create simplified interfaces and convenience functions for common modeling workflows
6. **📋 Model Selection Tools** - Implement AIC/BIC comparison, convergence diagnostics, and automated model selection
7. **🌐 Community Features** - Add discussion templates, contribution guidelines, and issue templates for GitHub collaboration

### 🎯 Future Enhancements (Next 2-3 months)

8. **🔄 Batch Processing** - Enhanced capabilities for processing multiple datasets and model comparisons in parallel
9. **📊 Visualization Dashboard** - Create diagnostic plots, convergence monitoring, and result visualization tools
10. **🔗 R Integration via Reticulate** - Create R package wrapper to use Pradel-JAX from R through reticulate interface
11. **🌍 Multi-model Support** - Extend to CJS, Multi-state, and Robust design models

### 📝 Major Achievements

**✅ Phase 3 Complete (August 14, 2025):**
- ✅ Automated pipeline with quality gates implementation
- ✅ Comprehensive error handling and recovery framework
- ✅ Parallel processing with intelligent resource management
- ✅ Production-ready validation framework

**✅ Framework Foundations:**
- ✅ JAX-based optimization framework with intelligent strategy selection
- ✅ Industry-standard performance monitoring and experiment tracking
- ✅ Complete 3-phase RMark parameter validation system
- ✅ Repository securely published on GitHub with data protection

**🚧 Current Focus:** Production deployment preparation and performance validation

---

> **📌 Note:** The validation framework is now production-ready. Focus has shifted from core development to deployment preparation, performance optimization, and community building.

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

**Status**: Production-ready validation framework with complete 3-phase implementation
**Version**: 2.0.0-rc (Release candidate - validation framework complete, preparing for production deployment)

## 🏆 Framework Status

**🎉 VALIDATION FRAMEWORK COMPLETE!** 

The Pradel-JAX validation framework now represents a **world-class parameter validation system** that:

- ✅ **Statistically Rigorous**: Industry-standard validation with bioequivalence and ecological significance thresholds
- ✅ **Production Ready**: Comprehensive error handling, quality gates, and automated pipeline orchestration
- ✅ **High Performance**: Parallel processing with intelligent resource management and optimization
- ✅ **Scientifically Credible**: Publication-quality validation suitable for peer-reviewed research
- ✅ **Community Focused**: Ready for open-source collaboration and contribution

**Ready for**: Production deployment, large-scale testing, and community collaboration