# API Reference

Complete technical documentation for all Pradel-JAX modules, classes, and functions.

## Module Overview

| Module | Description | Key Classes/Functions |
|--------|-------------|----------------------|
| [**Core API**](core.md) | High-level user interface | `fit_model()`, `create_formula_spec()` |
| [**Models**](models.md) | Model implementations | `PradelModel`, `ModelResult` |
| [**Data**](data.md) | Data loading and management | `load_data()`, `DataContext` |
| [**Formulas**](formulas.md) | Formula parsing and design matrices | `FormulaSpec`, `ParameterFormula` |
| [**Optimization**](optimization.md) | Optimization strategies and engines | `optimize_model()`, `OptimizationStrategy` |
| [**Inference**](inference.md) | Statistical inference framework | Standard errors, confidence intervals |
| [**Validation**](validation.md) | Data and model validation | `validate_data()`, `ValidationResult` |
| [**Configuration**](configuration.md) | Settings and configuration | `PradelJaxConfig` |
| [**Utilities**](utilities.md) | Helper functions and tools | Logging, transformations |

## Quick Navigation

### Most Used APIs
- [`pradel_jax.fit_model()`](core.md#fit_model) - Fit capture-recapture models
- [`pradel_jax.load_data()`](data.md#load_data) - Load and validate data
- [`pradel_jax.create_formula_spec()`](core.md#create_formula_spec) - Create model specifications

### Core Classes
- [`PradelModel`](models.md#pradelmodel) - Main model implementation
- [`DataContext`](data.md#datacontext) - Data container with rich metadata
- [`ModelResult`](models.md#modelresult) - Complete model fitting results
- [`FormulaSpec`](formulas.md#formulaspec) - Model specification container

### Advanced Features
- [**Optimization Strategies**](optimization.md#strategies) - L-BFGS-B, SLSQP, Adam, Multi-start
- [**Statistical Inference**](inference.md) - Hessian-based standard errors, bootstrap CIs
- [**Time-Varying Covariates**](data.md#time-varying-covariates) - Temporal covariate support
- [**Model Comparison**](models.md#model-comparison) - AIC/BIC comparison tools

## API Conventions

### Parameter Naming
- **snake_case** for function names and parameters: `fit_model()`, `data_context`
- **PascalCase** for class names: `PradelModel`, `DataContext`
- **UPPER_CASE** for constants and enums: `ParameterType.PHI`

### Return Types
- **Results objects** for complex returns: `ModelResult`, `ValidationResult`
- **Tuples** for simple multiple returns: `(train_data, test_data)`
- **None** for in-place operations: `configure()`

### Error Handling
- **Structured exceptions** with context: `PradelJaxError`, `OptimizationError`
- **Rich error messages** with suggestions for resolution
- **Validation before operations** to catch issues early

### Optional Dependencies
- **Graceful degradation** when optional packages unavailable
- **Clear warnings** about missing functionality
- **Alternative implementations** where possible

## Common Patterns

### Basic Model Fitting
```python
import pradel_jax as pj

# 1. Load data
data = pj.load_data("data.csv")

# 2. Specify model  
formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")

# 3. Fit model
result = pj.fit_model(model=pj.PradelModel(), formula=formula, data=data)

# 4. Extract results
if result.success:
    print(f"AIC: {result.aic:.2f}")
    print("Parameters:", result.parameter_estimates)
```

### Advanced Configuration
```python
# Custom optimization configuration
config = pj.OptimizationConfig(
    strategy="multi_start",
    max_iterations=2000,
    tolerance=1e-8
)

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula,
    data=data,
    optimization_config=config,
    compute_standard_errors=True,
    confidence_intervals=True
)
```

### Error Handling Pattern
```python
try:
    result = pj.fit_model(model, formula, data)
    if result.success:
        # Use results
        pass
    else:
        print(f"Optimization failed: {result.optimization_message}")
        
except pj.DataFormatError as e:
    print(f"Data format issue: {e}")
    print(f"Suggestion: {e.suggestion}")
    
except pj.ModelSpecificationError as e:
    print(f"Model specification issue: {e}")
    print(f"Available covariates: {e.available_covariates}")
    
except pj.OptimizationError as e:
    print(f"Optimization failed: {e}")
    print(f"Try strategy: {e.suggested_strategy}")
```

## Type Hints and Validation

All public APIs include comprehensive type hints:

```python
from typing import Optional, Union, Dict, List
from pathlib import Path

def fit_model(
    model: Optional[CaptureRecaptureModel] = None,
    formula: Optional[FormulaSpec] = None,
    data: Optional[Union[DataContext, str, Path]] = None,
    strategy: Optional[Union[str, OptimizationStrategy]] = None,
    **kwargs
) -> ModelResult:
    """Fit a capture-recapture model to data."""
    # Implementation with runtime validation
```

Runtime validation ensures type safety:

```python
# These will raise informative TypeErrors:
pj.fit_model(model="invalid_model_type")
pj.load_data(123)  # Not a valid path/DataFrame
pj.create_formula_spec(phi=123)  # Not a string formula
```

## Extensibility Points

### Custom Models
```python
from pradel_jax.models.base import CaptureRecaptureModel

class CustomModel(CaptureRecaptureModel):
    def log_likelihood(self, parameters, data_context, design_matrices):
        # Implement custom likelihood
        pass
    
    def get_initial_parameters(self, data_context, design_matrices):
        # Implement parameter initialization
        pass

# Register for use
pj.register_model(ModelType.CUSTOM, CustomModel)
```

### Custom Data Formats
```python
from pradel_jax.data.adapters import DataFormatAdapter

class CustomFormatAdapter(DataFormatAdapter):
    def can_handle(self, data_source) -> bool:
        # Check if this adapter can handle the data
        pass
    
    def load(self, data_source, **kwargs) -> DataContext:
        # Load and convert to DataContext
        pass

# Register adapter
pj.register_data_format_adapter(CustomFormatAdapter())
```

### Custom Optimizers
```python
from pradel_jax.optimization.strategy import OptimizationStrategy

class CustomOptimizer(OptimizationStrategy):
    def optimize(self, objective_function, initial_parameters, **kwargs):
        # Implement custom optimization algorithm
        pass

# Register strategy
pj.register_optimization_strategy("custom", CustomOptimizer())
```

## Performance Considerations

### Memory Usage
- **DataContext objects** hold data in memory - consider sampling for large datasets
- **Design matrices** can be large with many covariates - use caching judiciously
- **Bootstrap operations** create temporary copies - monitor memory usage

### Computation Speed
- **JAX compilation** has overhead - first model fit is slower than subsequent fits
- **Optimization strategy** choice affects speed - L-BFGS-B fastest for most problems
- **Time-varying covariates** increase computation - use only when necessary

### Caching
- **Design matrices** cached automatically for identical formula/data combinations
- **JAX functions** cached after first compilation
- **Bootstrap samples** not cached - recomputed each time

## Compatibility

### Python Versions
- **Python 3.8+** required
- **Type hints** use modern syntax (3.8+ features)
- **Pathlib** used throughout for cross-platform compatibility

### JAX Compatibility
- **JAX 0.4.20+** required for latest features
- **GPU support** automatic if available
- **64-bit precision** configurable via `JAX_ENABLE_X64`

### Optional Dependencies
- **MLflow 2.19.0+** for experiment tracking
- **Optuna 3.0+** for hyperparameter optimization  
- **Matplotlib/Seaborn** for built-in plotting
- **Scikit-optimize** for Bayesian optimization

## Version Compatibility

### API Stability Promise
- **Major versions** (2.0 → 3.0) may break backwards compatibility
- **Minor versions** (2.0 → 2.1) maintain backwards compatibility
- **Patch versions** (2.0.0 → 2.0.1) only fix bugs

### Deprecation Policy
- **Features marked deprecated** work for one major version
- **Deprecation warnings** issued in advance
- **Migration guides** provided for breaking changes

---

## Module Documentation

Click the links below for detailed documentation of each module:

- [**Core API**](core.md) - High-level interface functions
- [**Models Module**](models.md) - Model implementations and results
- [**Data Module**](data.md) - Data loading and management
- [**Formulas Module**](formulas.md) - Formula specification system
- [**Optimization Module**](optimization.md) - Optimization strategies and configuration
- [**Inference Module**](inference.md) - Statistical inference framework
- [**Validation Module**](validation.md) - Data and model validation
- [**Configuration Module**](configuration.md) - Settings and configuration management
- [**Utilities Module**](utilities.md) - Helper functions and utilities