# Core API Reference

The core API provides the main user interface for Pradel-JAX. These are the functions most users will interact with for standard capture-recapture analysis.

## Module: `pradel_jax.core.api`

### Functions

## `fit_model()`

Fit a capture-recapture model to data using the specified optimization strategy.

**Signature:**
```python
def fit_model(
    model: Optional[CaptureRecaptureModel] = None,
    formula: Optional[FormulaSpec] = None,
    data: Optional[Union[DataContext, str, Path]] = None,
    strategy: Optional[Union[str, OptimizationStrategy]] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    compute_standard_errors: bool = True,
    confidence_intervals: bool = True,
    bootstrap_confidence_intervals: bool = False,
    bootstrap_config: Optional[Dict[str, Any]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    initial_parameters: Optional[Dict[str, float]] = None,
    **kwargs
) -> ModelResult
```

**Parameters:**

- **model** (*CaptureRecaptureModel*, optional): Model instance to fit. Defaults to `PradelModel()`.
- **formula** (*FormulaSpec*, optional): Formula specification for model parameters. Defaults to constant parameters `(phi="~1", p="~1", f="~1")`.
- **data** (*DataContext | str | Path*, optional): Data to fit model to. Can be a DataContext object, file path, or pandas DataFrame.
- **strategy** (*str | OptimizationStrategy*, optional): Optimization strategy to use. Options: `"auto"`, `"lbfgs"`, `"slsqp"`, `"adam"`, `"multi_start"`. Defaults to `"auto"`.
- **optimization_config** (*OptimizationConfig*, optional): Detailed optimization configuration.
- **compute_standard_errors** (*bool*, default True): Whether to compute Hessian-based standard errors.
- **confidence_intervals** (*bool*, default True): Whether to compute asymptotic 95% confidence intervals.
- **bootstrap_confidence_intervals** (*bool*, default False): Whether to compute bootstrap confidence intervals.
- **bootstrap_config** (*Dict[str, Any]*, optional): Bootstrap configuration parameters.
- **parameter_bounds** (*Dict[str, Tuple[float, float]]*, optional): Custom parameter bounds.
- **initial_parameters** (*Dict[str, float]*, optional): Custom starting parameter values.
- **kwargs**: Additional optimization parameters passed to the underlying optimizer.

**Returns:**

- **ModelResult**: Complete model fitting results including parameter estimates, standard errors, confidence intervals, and model diagnostics.

**Raises:**

- **DataFormatError**: If data cannot be loaded or is in wrong format.
- **ModelSpecificationError**: If formula specification is invalid.
- **OptimizationError**: If optimization fails to converge.
- **PradelJaxError**: For other model fitting errors.

**Examples:**

```python
import pradel_jax as pj

# Basic usage with defaults
data = pj.load_data("data.csv")
result = pj.fit_model(data=data)

# Specify model formula
formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
result = pj.fit_model(formula=formula, data=data)

# Custom optimization
result = pj.fit_model(
    formula=formula,
    data=data,
    strategy="multi_start",
    compute_standard_errors=True,
    confidence_intervals=True
)

# With bootstrap confidence intervals
bootstrap_config = {
    "n_bootstrap": 1000,
    "confidence_level": 0.95,
    "method": "bca",
    "parallel": True
}

result = pj.fit_model(
    formula=formula,
    data=data,
    bootstrap_confidence_intervals=True,
    bootstrap_config=bootstrap_config
)

# Check results
if result.success:
    print(f"AIC: {result.aic:.2f}")
    print("Parameter estimates:")
    for param, estimate in result.parameter_estimates.items():
        se = result.standard_errors.get(param, 0.0)
        print(f"  {param}: {estimate:.3f} ± {se:.3f}")
else:
    print(f"Optimization failed: {result.optimization_message}")
```

---

## `create_formula_spec()`

Create a formula specification for Pradel model parameters using R-style syntax.

**Signature:**
```python
def create_formula_spec(
    phi: Optional[str] = None,
    p: Optional[str] = None,
    f: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
) -> FormulaSpec
```

**Parameters:**

- **phi** (*str*, optional): Formula for survival probability φ. Default: `"~1"` (constant).
- **p** (*str*, optional): Formula for detection probability p. Default: `"~1"` (constant).
- **f** (*str*, optional): Formula for recruitment rate f. Default: `"~1"` (constant).
- **name** (*str*, optional): Name for this formula specification.
- **description** (*str*, optional): Description of the model specification.
- **kwargs**: Additional parameter formulas for extended models.

**Returns:**

- **FormulaSpec**: Complete formula specification object ready for model fitting.

**Raises:**

- **ModelSpecificationError**: If any formula has invalid syntax.
- **ValueError**: If required parameters are missing.

**Examples:**

```python
# Constant parameters (default)
formula = pj.create_formula_spec()
# Equivalent to: phi="~1", p="~1", f="~1"

# Sex effect on survival and detection
formula = pj.create_formula_spec(
    phi="~1 + sex",
    p="~1 + sex", 
    f="~1"
)

# Complex model with interactions
formula = pj.create_formula_spec(
    phi="~1 + sex * age",
    p="~1 + effort + weather",
    f="~1 + habitat_quality",
    name="Complex Model",
    description="Sex-age interaction with environmental effects"
)

# Time-varying covariates
formula = pj.create_formula_spec(
    phi="~1 + sex + age_tv",      # age varies over time
    p="~1 + tier_tv",             # tier varies over time
    f="~1"
)

# Access formula properties
print(formula.phi.formula_string)  # "~1 + sex + age_tv"
print(formula.get_required_covariates())  # {'sex', 'age_tv', 'tier_tv'}
```

**Formula Syntax:**

The formula system supports R-style model specification:

- **Intercept**: `~1` (constant parameter)
- **Main effects**: `~1 + sex + age`
- **Interactions**: `~1 + sex * age` (equivalent to `~1 + sex + age + sex:age`)
- **Interaction only**: `~1 + sex:age` (no main effects)
- **Transformations**: `~1 + I(age**2)`, `~1 + np.log(weight)`
- **Factors**: `~1 + factor(region)` (categorical variables)
- **Time-varying**: `~1 + age_tv` (time-varying covariates)

---

## Module: `pradel_jax.core.export`

### Functions

## `export_model_results()`

Export model results to CSV files with publication-ready formatting.

**Signature:**
```python
def export_model_results(
    results: Union[ModelResult, Dict[str, ModelResult]],
    output_dir: Union[str, Path] = ".",
    prefix: str = "",
    include_timestamp: bool = True,
    export_parameters: bool = True,
    export_model_comparison: bool = True,
    export_confidence_intervals: bool = True,
    export_bootstrap_results: bool = False
) -> Dict[str, Path]
```

**Parameters:**

- **results** (*ModelResult | Dict[str, ModelResult]*): Model results to export. Can be single result or dictionary of multiple results.
- **output_dir** (*str | Path*, default "."): Directory to save export files.
- **prefix** (*str*, default ""): Prefix for output filenames.
- **include_timestamp** (*bool*, default True): Whether to include timestamp in filenames.
- **export_parameters** (*bool*, default True): Export parameter estimates table.
- **export_model_comparison** (*bool*, default True): Export model comparison table (if multiple results).
- **export_confidence_intervals** (*bool*, default True): Export confidence interval tables.
- **export_bootstrap_results** (*bool*, default False): Export bootstrap results (if available).

**Returns:**

- **Dict[str, Path]**: Dictionary mapping export type to output file path.

**Examples:**

```python
# Export single model results
result = pj.fit_model(formula=formula, data=data)
export_paths = pj.export_model_results(
    result,
    output_dir="results/",
    prefix="bird_analysis"
)

print("Exported files:")
for export_type, path in export_paths.items():
    print(f"  {export_type}: {path}")

# Export model comparison
models = {
    "null": pj.fit_model(formula=pj.create_formula_spec(), data=data),
    "sex_effect": pj.fit_model(formula=pj.create_formula_spec(phi="~1 + sex"), data=data),
    "full_model": pj.fit_model(formula=pj.create_formula_spec(phi="~1 + sex + age"), data=data)
}

export_paths = pj.export_model_results(
    models,
    output_dir="comparison_results/",
    prefix="model_selection",
    export_bootstrap_results=True
)
```

---

## `create_timestamped_export()`

Create timestamped export directory and export model results.

**Signature:**
```python
def create_timestamped_export(
    results: Union[ModelResult, Dict[str, ModelResult]],
    base_dir: Union[str, Path] = "results",
    analysis_name: str = "analysis",
    **export_kwargs
) -> Tuple[Path, Dict[str, Path]]
```

**Parameters:**

- **results**: Model results to export.
- **base_dir** (*str | Path*, default "results"): Base directory for exports.
- **analysis_name** (*str*, default "analysis"): Name for this analysis.
- **export_kwargs**: Additional arguments passed to `export_model_results()`.

**Returns:**

- **Tuple[Path, Dict[str, Path]]**: Export directory path and dictionary of exported files.

**Example:**

```python
# Create timestamped export
result = pj.fit_model(formula=formula, data=data)
export_dir, export_files = pj.create_timestamped_export(
    result,
    analysis_name="nebraska_birds",
    export_bootstrap_results=True
)

print(f"Results exported to: {export_dir}")
# Output: Results exported to: results/nebraska_birds_20250826_143052/
```

---

## Module: `pradel_jax.core.exceptions`

### Exception Classes

## `PradelJaxError`

Base exception class for all Pradel-JAX errors.

**Attributes:**
- **message** (*str*): Error message
- **suggestion** (*str*, optional): Suggested solution
- **context** (*Dict[str, Any]*, optional): Additional context information

## `DataFormatError`

Raised when data cannot be loaded or is in wrong format.

**Additional Attributes:**
- **available_formats** (*List[str]*): List of supported data formats
- **detected_format** (*str*, optional): Format that was detected

**Example:**
```python
try:
    data = pj.load_data("invalid_data.txt")
except pj.DataFormatError as e:
    print(f"Error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    print(f"Available formats: {e.available_formats}")
```

## `ModelSpecificationError`

Raised when model specification is invalid.

**Additional Attributes:**
- **invalid_formula** (*str*, optional): The problematic formula
- **available_covariates** (*List[str]*, optional): Available covariates in data

## `OptimizationError`

Raised when optimization fails.

**Additional Attributes:**
- **strategy_used** (*str*): Optimization strategy that failed
- **suggested_strategy** (*str*, optional): Alternative strategy to try
- **optimization_result** (*OptimizationResult*, optional): Raw optimization result

---

## Configuration

## Module: `pradel_jax.config.settings`

### `PradelJaxConfig`

Global configuration class for Pradel-JAX settings.

**Signature:**
```python
class PradelJaxConfig(BaseSettings):
    # Optimization settings
    default_optimization_strategy: str = "auto"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Statistical inference settings
    default_confidence_level: float = 0.95
    compute_standard_errors_by_default: bool = True
    
    # Performance settings
    use_jax_jit: bool = True
    jax_enable_x64: bool = False
    parallel_bootstrap: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[Path] = None
```

**Global Configuration Functions:**

```python
# Get current configuration
config = pj.get_config()
print(f"Default strategy: {config.default_optimization_strategy}")

# Update configuration
pj.configure(
    default_optimization_strategy="lbfgs",
    max_iterations=2000,
    tolerance=1e-8
)

# Environment variable override
# Set JAX_ENABLE_X64=True in environment to enable 64-bit precision
import os
os.environ["JAX_ENABLE_X64"] = "True"
```

---

## Utility Functions

### Model Comparison

```python
def compare_models(
    results: Dict[str, ModelResult],
    criterion: str = "aic"
) -> pd.DataFrame:
    """Compare multiple models using information criteria."""
    pass
```

### Parameter Transformations

```python
def logit_inverse(x: float) -> float:
    """Convert logit-scale value to probability scale."""
    return 1.0 / (1.0 + np.exp(-x))

def logit_to_prob_effect(logit_effect: float, baseline_prob: float = 0.5) -> float:
    """Convert logit-scale effect to probability-scale effect."""
    baseline_logit = np.log(baseline_prob / (1 - baseline_prob))
    new_prob = logit_inverse(baseline_logit + logit_effect)
    return new_prob - baseline_prob
```

**Examples:**

```python
# Convert parameter estimates to probability scale
result = pj.fit_model(formula=formula, data=data)

phi_intercept = result.parameter_estimates["phi_intercept"]
phi_baseline = pj.logit_inverse(phi_intercept)
print(f"Baseline survival probability: {phi_baseline:.3f}")

if "phi_sex" in result.parameter_estimates:
    sex_effect_logit = result.parameter_estimates["phi_sex"]
    sex_effect_prob = pj.logit_to_prob_effect(sex_effect_logit, phi_baseline)
    print(f"Sex effect on survival probability: {sex_effect_prob:+.3f}")
```

---

## Usage Patterns

### Complete Analysis Workflow

```python
import pradel_jax as pj

# 1. Load and validate data
data = pj.load_data("capture_data.csv")
print(f"Loaded {data.n_individuals} individuals, {data.n_occasions} occasions")

# 2. Create model specification
formula = pj.create_formula_spec(
    phi="~1 + sex + age",
    p="~1 + effort",
    f="~1"
)

# 3. Fit model with full statistical inference
result = pj.fit_model(
    formula=formula,
    data=data,
    compute_standard_errors=True,
    confidence_intervals=True,
    bootstrap_confidence_intervals=True,
    bootstrap_config={"n_bootstrap": 1000, "method": "bca"}
)

# 4. Check convergence and results
if result.success:
    print(f"✅ Model converged (AIC: {result.aic:.2f})")
    
    # 5. Export results
    export_paths = pj.export_model_results(
        result,
        output_dir="analysis_results/",
        prefix="final_analysis"
    )
    
    print("Results exported:")
    for export_type, path in export_paths.items():
        print(f"  {export_type}: {path}")
        
else:
    print(f"❌ Model failed: {result.optimization_message}")
```

### Model Selection Workflow

```python
# Define candidate models
candidates = {
    "null": pj.create_formula_spec(),
    "sex_only": pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1"),
    "age_only": pj.create_formula_spec(phi="~1 + age", p="~1", f="~1"),
    "additive": pj.create_formula_spec(phi="~1 + sex + age", p="~1", f="~1"),
    "interactive": pj.create_formula_spec(phi="~1 + sex * age", p="~1", f="~1")
}

# Fit all models
results = {}
for name, formula in candidates.items():
    print(f"Fitting {name} model...")
    results[name] = pj.fit_model(formula=formula, data=data)

# Compare models
comparison = pj.compare_models(results, criterion="aic")
print("\nModel Rankings:")
print(comparison.round(2))

# Export comparison results
pj.export_model_results(
    results,
    output_dir="model_selection/",
    prefix="comparison"
)
```

---

**See Also:**
- [Models API Reference](models.md) - Model classes and results
- [Data API Reference](data.md) - Data loading and management
- [Optimization API Reference](optimization.md) - Optimization strategies and configuration