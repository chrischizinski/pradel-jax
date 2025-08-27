# Models Module API Reference

The `pradel_jax.models` module provides model implementations and result handling for capture-recapture analysis.

## Overview

This module contains:
- **Base Classes**: Abstract interfaces for all capture-recapture models
- **Model Implementations**: Concrete model classes (Pradel, CJS, etc.)
- **Result Classes**: Rich result objects with statistical inference
- **Model Registry**: Plugin system for custom models

## Core Classes

### PradelModel

The main implementation of the Pradel temporal symmetry model.

```python
class PradelModel(CaptureRecaptureModel):
    """Pradel temporal symmetry model for capture-recapture analysis.
    
    Supports survival (phi), detection (p), and recruitment (f) parameters
    with flexible formula-based covariate specification.
    """
```

#### Constructor

```python
def __init__(
    self,
    link_functions: Optional[Dict[str, str]] = None,
    parameter_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    name: Optional[str] = None
):
    """Initialize Pradel model.
    
    Parameters
    ----------
    link_functions : dict, optional
        Link functions for each parameter type.
        Default: {"phi": "logit", "p": "logit", "f": "log"}
    parameter_constraints : dict, optional
        Parameter bounds for optimization.
        Default: {"phi": (0.0, 1.0), "p": (0.0, 1.0), "f": (0.0, 10.0)}
    name : str, optional
        Model name for identification. Default: "Pradel"
    
    Examples
    --------
    >>> model = PradelModel()  # Use defaults
    >>> model = PradelModel(name="Custom Pradel")
    >>> model = PradelModel(
    ...     link_functions={"phi": "cloglog", "p": "logit", "f": "log"},
    ...     parameter_constraints={"f": (0.0, 5.0)}
    ... )
    """
```

#### Key Methods

```python
def log_likelihood(
    self,
    parameters: jnp.ndarray,
    data_context: DataContext,
    design_matrices: Dict[str, DesignMatrixInfo]
) -> float:
    """Compute log-likelihood for given parameters.
    
    Parameters
    ----------
    parameters : jnp.ndarray
        Model parameters in optimization space
    data_context : DataContext
        Data container with capture histories and covariates
    design_matrices : dict
        Design matrices for each parameter type
        
    Returns
    -------
    float
        Log-likelihood value (negative for minimization)
    """

def get_initial_parameters(
    self,
    data_context: DataContext,
    design_matrices: Dict[str, DesignMatrixInfo]
) -> jnp.ndarray:
    """Generate reasonable initial parameter values.
    
    Returns
    -------
    jnp.ndarray
        Initial parameter vector for optimization
    """

def get_parameter_bounds(
    self,
    data_context: DataContext,
    design_matrices: Dict[str, DesignMatrixInfo]
) -> List[Tuple[float, float]]:
    """Get parameter bounds for constrained optimization.
    
    Returns
    -------
    List[Tuple[float, float]]
        Bounds for each parameter
    """

def build_design_matrices(
    self,
    formula_spec: FormulaSpec,
    data_context: DataContext
) -> Dict[str, DesignMatrixInfo]:
    """Build design matrices from formula specification.
    
    Parameters
    ----------
    formula_spec : FormulaSpec
        Formula specification for phi, p, f parameters
    data_context : DataContext
        Data container with covariates
        
    Returns
    -------
    Dict[str, DesignMatrixInfo]
        Design matrices with metadata for each parameter
    """
```

#### Usage Examples

```python
import pradel_jax as pj

# Basic usage
model = pj.PradelModel()
data = pj.load_data("data.csv")
formula = pj.create_formula_spec(phi="~sex", p="~1", f="~1")

result = pj.fit_model(model, formula, data)

# Advanced usage with custom configuration
custom_model = pj.PradelModel(
    link_functions={"phi": "cloglog", "p": "logit", "f": "log"},
    parameter_constraints={"phi": (0.001, 0.999), "f": (0.0, 5.0)},
    name="Custom Pradel with Cloglog"
)

result = pj.fit_model(custom_model, formula, data)
```

### ModelResult

Rich result container with comprehensive model fitting information and statistical inference.

```python
@dataclass
class ModelResult:
    """Complete results from model fitting with statistical inference.
    
    Provides access to parameter estimates, standard errors, confidence
    intervals, model comparison statistics, and diagnostics.
    """
```

#### Core Attributes

```python
# Model identification
model_type: ModelType                           # Type of model fitted
formula_spec: FormulaSpec                      # Formula specification used
model_name: Optional[str] = None               # Custom model name

# Optimization results
status: OptimizationStatus = OptimizationStatus.FAILED
parameters: Optional[jnp.ndarray] = None       # Raw parameter vector
log_likelihood: Optional[float] = None          # Final log-likelihood
aic: Optional[float] = None                    # Akaike Information Criterion
bic: Optional[float] = None                    # Bayesian Information Criterion

# Parameter information  
parameter_names: Optional[List[str]] = None     # Human-readable names
parameter_se: Optional[jnp.ndarray] = None      # Standard errors
parameter_ci: Optional[Dict[str, jnp.ndarray]] = None  # Confidence intervals

# Design matrix information
design_matrices: Optional[Dict[str, DesignMatrixInfo]] = None

# Statistical inference
inference_result: Optional[InferenceResult] = None  # Full inference results

# Optimization metadata
n_parameters: Optional[int] = None              # Number of parameters
n_iterations: Optional[int] = None              # Optimization iterations  
optimizer_used: Optional[str] = None            # Strategy that succeeded
strategy_used: Optional[str] = None             # Strategy identifier
convergence_tolerance: Optional[float] = None   # Final tolerance achieved

# Diagnostics
gradient_norm: Optional[float] = None           # Final gradient norm
hessian_condition: Optional[float] = None       # Hessian condition number
warnings: List[str] = field(default_factory=list)  # Warnings/issues

# Timing
fit_time: Optional[float] = None                # Total fitting time
```

#### Key Properties

```python
@property
def success(self) -> bool:
    """Whether optimization succeeded."""
    return self.status == OptimizationStatus.SUCCESS

@property  
def parameter_estimates(self) -> Dict[str, float]:
    """Parameter estimates as name-value dictionary.
    
    Returns
    -------
    Dict[str, float]
        Parameter names mapped to their estimated values
        
    Examples
    --------
    >>> result.parameter_estimates
    {'phi_intercept': 0.546, 'phi_sex': 0.034, 'p_intercept': -1.012, ...}
    """

@property
def confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
    """95% confidence intervals for all parameters.
    
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Parameter names mapped to (lower, upper) bounds
        
    Examples
    --------
    >>> result.confidence_intervals
    {'phi_intercept': (0.356, 0.736), 'phi_sex': (-0.244, 0.312), ...}
    """

@property
def standard_errors(self) -> Dict[str, float]:
    """Standard errors for all parameters.
    
    Returns
    -------
    Dict[str, float]
        Parameter names mapped to standard error values
    """

@property
def z_scores(self) -> Dict[str, float]:
    """Z-scores for hypothesis testing (parameter != 0).
    
    Returns
    -------
    Dict[str, float]
        Parameter names mapped to z-score values
    """

@property  
def p_values(self) -> Dict[str, float]:
    """P-values for hypothesis testing (parameter != 0).
    
    Returns
    -------
    Dict[str, float]
        Parameter names mapped to p-value values
    """
```

#### Key Methods

```python
def get_parameter_table(
    self,
    confidence_level: float = 0.95,
    format: str = "dataframe"
) -> Union[pd.DataFrame, str]:
    """Get formatted parameter table with inference.
    
    Parameters
    ----------
    confidence_level : float, default 0.95
        Confidence level for intervals
    format : {'dataframe', 'latex', 'markdown'}, default 'dataframe'
        Output format
        
    Returns
    -------
    pd.DataFrame or str
        Formatted parameter table
        
    Examples
    --------
    >>> table = result.get_parameter_table()
    >>> print(table)
                    Estimate    SE   CI_Lower  CI_Upper  Z_Score  P_Value
    phi_intercept      0.546  0.097      0.356     0.736    5.629    0.000
    phi_sex            0.034  0.142     -0.244     0.312    0.239    0.811
    ...
    """

def get_model_summary(self) -> Dict[str, Any]:
    """Get comprehensive model summary.
    
    Returns
    -------
    Dict[str, Any]
        Summary statistics and diagnostics
        
    Examples
    --------
    >>> summary = result.get_model_summary()
    >>> summary['model_fit']
    {'aic': 235.47, 'bic': 248.52, 'log_likelihood': -112.74, 'n_parameters': 5}
    """

def compare_models(
    self,
    other: Union['ModelResult', List['ModelResult']],
    criteria: str = "aic"
) -> pd.DataFrame:
    """Compare this model with others.
    
    Parameters
    ----------
    other : ModelResult or list of ModelResult
        Model(s) to compare against
    criteria : {'aic', 'bic', 'log_likelihood'}, default 'aic'
        Comparison criterion
        
    Returns
    -------
    pd.DataFrame
        Model comparison table with rankings and statistics
        
    Examples
    --------
    >>> comparison = result1.compare_models([result2, result3])
    >>> print(comparison)
             Model         AIC  Delta_AIC  AIC_Weight  Evidence_Ratio
    0  Sex_Model      235.47       0.00        0.73           1.00
    1  Constant_Model 238.92       3.45        0.27           2.70
    """

def plot_diagnostics(
    self,
    plots: List[str] = ["residuals", "qq", "influence"],
    figsize: Tuple[int, int] = (12, 8)
) -> matplotlib.figure.Figure:
    """Generate diagnostic plots.
    
    Parameters
    ----------
    plots : list of str
        Types of plots to generate
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with diagnostic plots
    """

def bootstrap_confidence_intervals(
    self,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_seed: Optional[int] = None
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap confidence intervals.
    
    Parameters
    ----------
    n_bootstrap : int, default 1000
        Number of bootstrap samples
    confidence_level : float, default 0.95
        Confidence level
    method : {'percentile', 'bca', 'studentized'}, default 'percentile'
        Bootstrap CI method
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, Tuple[float, float]]
        Bootstrap confidence intervals
    """

def export_results(
    self,
    filename: str,
    format: str = "csv",
    include: Optional[List[str]] = None
) -> None:
    """Export results to file.
    
    Parameters
    ----------
    filename : str
        Output filename
    format : {'csv', 'json', 'excel', 'pickle'}, default 'csv'
        Export format
    include : list of str, optional
        Components to include. Default: all available
        
    Examples
    --------
    >>> result.export_results("pradel_results.csv")
    >>> result.export_results("results.json", format="json")
    >>> result.export_results("analysis.xlsx", format="excel", 
    ...                       include=["parameters", "inference", "diagnostics"])
    """
```

#### Usage Examples

```python
# Basic result access
if result.success:
    print(f"AIC: {result.aic:.2f}")
    print("Parameter estimates:")
    for name, value in result.parameter_estimates.items():
        print(f"  {name}: {value:.4f}")

# Statistical inference
print("95% Confidence Intervals:")
for name, (lower, upper) in result.confidence_intervals.items():
    print(f"  {name}: [{lower:.4f}, {upper:.4f}]")

# Hypothesis testing
print("Significance tests (α = 0.05):")
for name, p_val in result.p_values.items():
    significant = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"  {name}: p = {p_val:.4f} {significant}")

# Publication-ready table
table = result.get_parameter_table(format="latex")
print(table)  # Ready for LaTeX documents

# Model comparison
comparison = result.compare_models([other_result1, other_result2])
print(comparison)

# Export for further analysis
result.export_results("pradel_analysis.csv")
result.export_results("full_results.json", format="json")
```

## Base Classes

### CaptureRecaptureModel

Abstract base class for all capture-recapture models.

```python
class CaptureRecaptureModel(ABC):
    """Abstract base class for capture-recapture models.
    
    Defines the interface that all model implementations must follow.
    Provides common functionality for parameter transformations, 
    bounds handling, and optimization integration.
    """
```

#### Abstract Methods

All model implementations must provide:

```python
@abstractmethod
def log_likelihood(
    self, 
    parameters: jnp.ndarray,
    data_context: DataContext, 
    design_matrices: Dict[str, DesignMatrixInfo]
) -> float:
    """Compute log-likelihood for given parameters."""

@abstractmethod  
def get_initial_parameters(
    self,
    data_context: DataContext,
    design_matrices: Dict[str, DesignMatrixInfo]
) -> jnp.ndarray:
    """Generate initial parameter values for optimization."""

@abstractmethod
def get_parameter_bounds(
    self,
    data_context: DataContext, 
    design_matrices: Dict[str, DesignMatrixInfo]
) -> List[Tuple[float, float]]:
    """Get parameter bounds for constrained optimization."""
```

#### Common Methods

Base class provides default implementations for:

```python
def build_design_matrices(
    self,
    formula_spec: FormulaSpec,
    data_context: DataContext
) -> Dict[str, DesignMatrixInfo]:
    """Build design matrices from formulas using patsy."""

def transform_parameters(
    self,
    parameters: jnp.ndarray,
    parameter_type: str,
    direction: str = "to_natural"
) -> jnp.ndarray:
    """Apply link function transformations."""

def get_parameter_names(
    self,
    design_matrices: Dict[str, DesignMatrixInfo]
) -> List[str]:
    """Generate human-readable parameter names."""
```

### ModelType

Enumeration of available model types.

```python
class ModelType(str, Enum):
    """Types of capture-recapture models supported."""
    
    PRADEL = "pradel"                    # Pradel temporal symmetry model
    CJS = "cjs"                         # Cormack-Jolly-Seber model  
    POPAN = "popan"                     # POPAN superpopulation model
    MULTI_STATE = "multi_state"         # Multi-state models
    ROBUST_DESIGN = "robust_design"     # Robust design models
```

## Model Registry

The model registry provides a plugin system for custom models.

### Registration Functions

```python
def register_model(model_type: ModelType, model_class: Type[CaptureRecaptureModel]) -> None:
    """Register a model implementation.
    
    Parameters
    ----------
    model_type : ModelType
        Model type identifier
    model_class : class
        Model implementation class
        
    Examples
    --------
    >>> class MyCustomModel(CaptureRecaptureModel):
    ...     # Implementation
    ...     pass
    >>> register_model(ModelType.PRADEL, MyCustomModel)
    """

def get_model(model_type: ModelType) -> Type[CaptureRecaptureModel]:
    """Get registered model class.
    
    Parameters
    ---------- 
    model_type : ModelType
        Model type to retrieve
        
    Returns
    -------
    Type[CaptureRecaptureModel]
        Model class
    """

def list_available_models() -> Dict[ModelType, Type[CaptureRecaptureModel]]:
    """List all registered models.
    
    Returns
    -------
    Dict[ModelType, Type[CaptureRecaptureModel]]
        Available model types and their implementations
    """
```

### Usage Examples

```python
# Check available models
available = pj.list_available_models()
print("Available models:", list(available.keys()))

# Get model class
model_class = pj.get_model(ModelType.PRADEL)
model = model_class()

# Register custom model
class CustomPradelModel(pj.CaptureRecaptureModel):
    # Custom implementation
    pass

pj.register_model(ModelType.PRADEL, CustomPradelModel)
```

## Optimization Status

```python
class OptimizationStatus(str, Enum):
    """Optimization outcome status codes."""
    
    SUCCESS = "success"                  # Optimization succeeded
    FAILED = "failed"                   # General failure  
    MAX_ITER = "max_iterations"         # Maximum iterations reached
    NUMERICAL_ERROR = "numerical_error" # Numerical issues
    CONVERGENCE_ERROR = "convergence_error"  # Convergence problems
```

## Statistical Inference Integration

The models module integrates closely with the inference module to provide comprehensive statistical analysis:

```python
# Standard errors computed automatically when requested
result = pj.fit_model(model, formula, data, compute_se=True)
print(result.standard_errors)

# Confidence intervals using multiple methods
result = pj.fit_model(model, formula, data, confidence_intervals=True)
print(result.confidence_intervals)  # Asymptotic CIs

# Bootstrap confidence intervals
bootstrap_cis = result.bootstrap_confidence_intervals(n_bootstrap=2000)
print(bootstrap_cis)

# Full inference result with additional statistics
inference = result.inference_result
print(inference.covariance_matrix)
print(inference.correlation_matrix)
print(inference.confidence_ellipses)
```

## Performance Notes

### Memory Usage
- **ModelResult objects** contain all fitting information and can be large
- **Design matrices** cached to avoid recomputation with same formulas
- **Bootstrap operations** temporarily increase memory usage

### Computation Speed  
- **First model fit** slower due to JAX compilation
- **Subsequent fits** with same model structure much faster
- **Parameter initialization** uses smart defaults based on data

### Caching Behavior
- **Design matrices** cached by (formula, data) hash
- **JAX functions** compiled once per model structure
- **Parameter bounds** recomputed each time (cheap operation)

---

*This documentation covers the models module API. For usage examples, see the [User Guide](../user-guide/) and [Tutorials](../tutorials/).*