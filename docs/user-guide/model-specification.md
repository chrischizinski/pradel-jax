# Model Specification Guide

This guide covers how to specify and configure capture-recapture models in Pradel-JAX, including the Pradel model implementation, formula system, and parameter estimation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Pradel Model](#understanding-the-pradel-model)
3. [Formula System Overview](#formula-system-overview)
4. [Parameter Specification](#parameter-specification)
5. [Model Configuration](#model-configuration)
6. [Statistical Inference](#statistical-inference)
7. [Model Comparison](#model-comparison)
8. [Advanced Specifications](#advanced-specifications)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

## Quick Start

```python
import pradel_jax as pj

# Load your data
data_context = pj.load_data("data.csv")

# Create simple constant parameter model
formula_spec = pj.create_formula_spec(
    phi="~1",  # Constant survival
    p="~1",    # Constant detection  
    f="~1"     # Constant recruitment
)

# Fit the model
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context
)

# View results
print(f"Model converged: {result.success}")
print(f"AIC: {result.aic:.2f}")
print("Parameter estimates:")
for param, estimate in result.parameter_estimates.items():
    print(f"  {param}: {estimate:.3f} ± {result.standard_errors[param]:.3f}")
```

## Understanding the Pradel Model

The Pradel model is a robust open population capture-recapture model that estimates three key parameters:

### Model Parameters

1. **φ (phi) - Survival Probability**
   - Probability that an individual present at time *i* survives to time *i+1*
   - Range: [0, 1]
   - Interpretation: Individual survival between capture occasions

2. **p - Detection Probability** 
   - Probability that an individual present at time *i* is captured
   - Range: [0, 1]
   - Interpretation: Sampling efficiency or capture effort

3. **f - Recruitment Rate**
   - Number of new individuals entering the population per existing individual
   - Range: [0, ∞]
   - Interpretation: Population growth through births, immigration, or maturation

### Model Assumptions

The Pradel model assumes:
- **Population closure between occasions**: No entry/exit during sampling periods
- **Random sampling**: All individuals have equal capture probability (unless modeled otherwise)
- **Independent captures**: Capture at one occasion doesn't affect capture at another
- **No tag loss**: Marks are permanent and correctly read
- **Homogeneity**: All individuals have the same demographic rates (unless modeled with covariates)

### Identifiability Constraints

- The first and last occasion parameters may have identifiability constraints
- Time-varying models require careful parameterization
- Covariate models need sufficient data per covariate level

## Formula System Overview

Pradel-JAX uses R-style formulas to specify how parameters depend on covariates. This provides a flexible and familiar interface for ecological modeling.

### Basic Formula Syntax

```python
# Constant parameter (intercept only)
"~1"

# Single covariate effect
"~1 + sex"
"~sex"  # Equivalent (intercept implied)

# Multiple covariates  
"~1 + sex + age"
"~sex + age"  # Equivalent

# Interaction terms
"~1 + sex + age + sex:age"
"~sex * age"  # Equivalent (includes main effects and interaction)

# Polynomial terms
"~1 + age + I(age**2)"  # Quadratic age effect

# Factor variables (automatic dummy coding)
"~1 + region"  # Creates dummy variables for each region level
```

### Advanced Formula Features

```python
# Transformation functions
"~1 + np.log(weight)"      # Log transformation
"~1 + np.sqrt(body_size)"  # Square root transformation  
"~1 + standardize(age)"    # Standardization (mean=0, sd=1)

# Spline functions (planned feature)
"~1 + spline(age, df=3)"   # Natural spline with 3 degrees of freedom

# Custom functions
"~1 + custom_transformation(covariate)"
```

### Formula Validation

The formula system validates specifications during model fitting:

```python
# This will raise informative errors:
formula_spec = pj.create_formula_spec(
    phi="~1 + nonexistent_covariate",  # Error: covariate not found
    p="~1 + sex + sex",                # Warning: duplicate terms
    f="~1 + I(age**0.5"                # Error: syntax error
)
```

## Parameter Specification

### Creating Formula Specifications

```python
# Method 1: High-level API (recommended)
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",
    p="~1 + effort", 
    f="~1"
)

# Method 2: Direct construction
from pradel_jax.formulas import FormulaSpec, ParameterFormula, ParameterType

phi_formula = ParameterFormula(parameter=ParameterType.PHI, formula_string="~1 + sex")
p_formula = ParameterFormula(parameter=ParameterType.P, formula_string="~1 + effort")
f_formula = ParameterFormula(parameter=ParameterType.F, formula_string="~1")

formula_spec = FormulaSpec(phi=phi_formula, p=p_formula, f=f_formula)
```

### Common Parameter Modeling Patterns

#### 1. Time Effects

```python
# Time-varying parameters (each occasion gets its own estimate)
formula_spec = pj.create_formula_spec(
    phi="~1 + time",  # Linear time trend
    p="~1 + time",    # Detection changes over time
    f="~1"            # Constant recruitment
)

# Categorical time effects (separate estimate per occasion)
formula_spec = pj.create_formula_spec(
    phi="~1 + factor(occasion)",  # Different phi each occasion
    p="~1 + factor(occasion)",    # Different p each occasion  
    f="~1"
)
```

#### 2. Individual Covariates

```python
# Sex effects
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",  # Males and females have different survival
    p="~1 + sex",    # Sex affects detection probability
    f="~1"           # Recruitment independent of sex
)

# Age effects
formula_spec = pj.create_formula_spec(
    phi="~1 + age",              # Linear age effect on survival
    p="~1 + I(age**2)",          # Quadratic age effect on detection
    f="~1 + age"                 # Age affects recruitment
)

# Multiple covariates
formula_spec = pj.create_formula_spec(
    phi="~1 + sex + age + weight",  # Additive effects
    p="~1 + effort + weather",      # Environmental effects on detection
    f="~1 + habitat_quality"        # Habitat affects recruitment
)
```

#### 3. Interaction Effects

```python
# Sex-age interaction
formula_spec = pj.create_formula_spec(
    phi="~1 + sex * age",      # Sex, age, and sex:age interaction
    p="~1 + sex + age",        # Additive sex and age effects
    f="~1"
)

# Environmental interactions
formula_spec = pj.create_formula_spec(
    phi="~1 + temperature",
    p="~1 + effort * weather",  # Effort effectiveness depends on weather
    f="~1 + rainfall * season"  # Recruitment depends on seasonal rainfall
)
```

#### 4. Time-Varying Covariates

```python
# For data with age_2016, age_2017, etc.
formula_spec = pj.create_formula_spec(
    phi="~1 + age_tv",  # Time-varying age effect
    p="~1 + sex",       # Constant sex effect
    f="~1 + tier_tv"    # Time-varying tier effect
)

# Multiple time-varying covariates
formula_spec = pj.create_formula_spec(
    phi="~1 + age_tv + weight_tv",
    p="~1 + sex + effort_tv",
    f="~1"
)
```

## Model Configuration

### Optimization Configuration

```python
from pradel_jax.optimization import OptimizationConfig

# Create model with custom optimization settings
config = OptimizationConfig(
    strategy="auto",           # Let framework choose best strategy
    max_iterations=1000,       # Maximum optimization iterations
    tolerance=1e-6,            # Convergence tolerance
    use_bounds=True,           # Apply parameter bounds
    multi_start_attempts=5     # Multiple starting points for robustness
)

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context,
    optimization_config=config
)
```

### Parameter Bounds and Constraints

```python
# Custom parameter bounds
bounds_config = {
    "phi": (0.01, 0.99),  # Survival bounded away from 0 and 1
    "p": (0.001, 0.8),    # Detection probability constraints
    "f": (0.0, 10.0)      # Recruitment rate upper limit
}

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context,
    parameter_bounds=bounds_config
)
```

### Initial Parameter Values

```python
# Custom starting values
initial_values = {
    "phi_intercept": 0.8,     # Start with high survival
    "phi_sex": 0.1,           # Small sex effect
    "p_intercept": 0.3,       # Moderate detection
    "f_intercept": 0.2        # Low recruitment
}

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context,
    initial_parameters=initial_values
)
```

## Statistical Inference

Pradel-JAX provides comprehensive statistical inference capabilities implemented through the WF-007 framework.

### Standard Errors and Confidence Intervals

```python
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context,
    compute_standard_errors=True,      # Hessian-based standard errors
    confidence_intervals=True,         # 95% normal approximation CIs
    bootstrap_confidence_intervals=True # Bootstrap CIs (more robust)
)

# Access statistical inference results
print("Parameter Estimates:")
for param in result.parameter_names:
    est = result.parameter_estimates[param]
    se = result.standard_errors[param]
    ci_lower = result.confidence_intervals[param]["lower"]
    ci_upper = result.confidence_intervals[param]["upper"]
    
    print(f"{param:15} = {est:6.3f} ± {se:6.3f} [{ci_lower:6.3f}, {ci_upper:6.3f}]")
```

### Hypothesis Testing

```python
# Statistical significance testing
print("Hypothesis Tests (H0: parameter = 0):")
for param in result.parameter_names:
    z_score = result.z_scores[param]
    p_value = result.p_values[param]
    significant = result.significance[param]
    
    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"{param:15}: Z = {z_score:6.2f}, p = {p_value:6.4f} {sig_marker}")
```

### Bootstrap Confidence Intervals

```python
# Configure bootstrap parameters
bootstrap_config = {
    "n_bootstrap": 1000,        # Number of bootstrap samples
    "confidence_level": 0.95,   # CI level
    "method": "bca",            # BCa method (bias-corrected accelerated)
    "parallel": True,           # Use parallel processing
    "random_state": 42          # Reproducible results
}

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context,
    bootstrap_config=bootstrap_config
)

# Access bootstrap results
bootstrap_cis = result.bootstrap_confidence_intervals
for param, ci in bootstrap_cis.items():
    print(f"{param}: [{ci['lower']:.3f}, {ci['upper']:.3f}] (bootstrap BCa)")
```

## Model Comparison

### Information Criteria

```python
# Fit multiple models for comparison
models = {}

# Model 1: Constant parameters
models['constant'] = pj.fit_model(
    model=pj.PradelModel(),
    formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
    data=data_context
)

# Model 2: Sex effect on survival and detection
models['sex_effect'] = pj.fit_model(
    model=pj.PradelModel(),
    formula=pj.create_formula_spec(phi="~1 + sex", p="~1 + sex", f="~1"),
    data=data_context
)

# Model 3: Full covariate model
models['full_model'] = pj.fit_model(
    model=pj.PradelModel(),
    formula=pj.create_formula_spec(phi="~1 + sex + age", p="~1 + sex + age", f="~1 + sex"),
    data=data_context
)

# Compare models
comparison_table = pj.compare_models(models)
print(comparison_table)
```

### Model Selection

```python
# Automatic model selection based on AIC
from pradel_jax.model_selection import ModelSelector

selector = ModelSelector(
    candidate_formulas=[
        pj.create_formula_spec(phi="~1", p="~1", f="~1"),
        pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1"),
        pj.create_formula_spec(phi="~1", p="~1 + sex", f="~1"),
        pj.create_formula_spec(phi="~1 + sex", p="~1 + sex", f="~1"),
    ],
    selection_criterion="aic"  # or "bic", "aicc"
)

best_model = selector.select_best_model(data_context)
print(f"Best model: {best_model.formula_spec}")
print(f"AIC: {best_model.aic:.2f}")
```

## Advanced Specifications

### Custom Link Functions

```python
# Specify alternative link functions
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",
    p="~1 + effort",
    f="~1",
    link_functions={
        "phi": "logit",    # Default for probabilities
        "p": "logit",      # Default for probabilities
        "f": "log"         # Default for rates
    }
)
```

### Random Effects (Planned Feature)

```python
# Mixed-effects modeling (future enhancement)
formula_spec = pj.create_formula_spec(
    phi="~1 + sex + (1|site)",        # Random site intercepts
    p="~1 + effort + (effort|year)",  # Random year slopes
    f="~1"
)
```

### Non-linear Parameter Relationships

```python
# Complex functional forms
formula_spec = pj.create_formula_spec(
    phi="~1 + poly(age, 2)",              # Polynomial age effect
    p="~1 + spline(day_of_year, df=4)",   # Seasonal detection pattern
    f="~1 + threshold(temperature, 15)"   # Threshold temperature effect
)
```

## Examples

### Example 1: Basic Analysis

```python
import pradel_jax as pj

# Load data
data = pj.load_data("dipper_data.csv")

# Simple model with sex effect
formula = pj.create_formula_spec(
    phi="~1 + sex",  # Sex affects survival
    p="~1 + sex",    # Sex affects detection
    f="~1"           # Constant recruitment
)

# Fit model
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula,
    data=data
)

# Results summary
if result.success:
    print(f"✅ Model converged (AIC: {result.aic:.2f})")
    print("\nParameter estimates:")
    print(result.parameter_summary())
else:
    print("❌ Model failed to converge")
    print(f"Error: {result.optimization_message}")
```

### Example 2: Time-Varying Covariates

```python
# Load data with time-varying age and tier
data = pj.load_data("nebraska_data.csv")

# Check time-varying structure
print("Time-varying covariates detected:")
for cov, occasions in data.time_varying_covariates.items():
    print(f"  {cov}: {occasions} occasions")

# Model with time-varying covariates
formula = pj.create_formula_spec(
    phi="~1 + age_tv + sex",     # Age varies over time
    p="~1 + tier_tv + sex",      # Tier varies over time
    f="~1"
)

# Fit with statistical inference
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula,
    data=data,
    compute_standard_errors=True,
    confidence_intervals=True
)

# Detailed results
print(result.statistical_summary())
```

### Example 3: Model Comparison Study

```python
# Define candidate models
candidate_models = {
    "null": pj.create_formula_spec(phi="~1", p="~1", f="~1"),
    "sex_phi": pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1"),
    "sex_p": pj.create_formula_spec(phi="~1", p="~1 + sex", f="~1"), 
    "sex_both": pj.create_formula_spec(phi="~1 + sex", p="~1 + sex", f="~1"),
    "additive": pj.create_formula_spec(phi="~1 + sex + age", p="~1 + sex", f="~1"),
    "interactive": pj.create_formula_spec(phi="~1 + sex * age", p="~1 + sex", f="~1")
}

# Fit all models
results = {}
for name, formula in candidate_models.items():
    print(f"Fitting {name} model...")
    results[name] = pj.fit_model(
        model=pj.PradelModel(),
        formula=formula,
        data=data
    )

# Compare models
comparison = pj.compare_models(results)
print("\nModel Comparison:")
print(comparison.sort_values('aic'))

# Best model analysis
best_model_name = comparison.sort_values('aic').index[0]
best_result = results[best_model_name]
print(f"\n🏆 Best model: {best_model_name}")
print(f"AIC: {best_result.aic:.2f}")
print("\nParameter estimates:")
print(best_result.parameter_summary())
```

## Troubleshooting

### Common Model Specification Issues

1. **Convergence Problems**
   ```python
   # Try different optimization strategy
   result = pj.fit_model(
       model=pj.PradelModel(),
       formula=formula_spec,
       data=data_context,
       strategy="multi_start"  # More robust optimization
   )
   
   # Or adjust tolerance
   result = pj.fit_model(
       model=pj.PradelModel(),
       formula=formula_spec,
       data=data_context,
       tolerance=1e-4  # Less strict convergence
   )
   ```

2. **Parameter Bounds Issues**
   ```python
   # Check parameter bounds are reasonable
   result = pj.fit_model(
       model=pj.PradelModel(),
       formula=formula_spec,
       data=data_context,
       check_bounds=True,
       verbose=True  # See optimization details
   )
   ```

3. **Formula Parsing Errors**
   ```python
   # Validate formula before fitting
   try:
       formula_spec = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
       # Check formula against data
       formula_spec.validate(data_context)
   except pj.ModelSpecificationError as e:
       print(f"Formula error: {e}")
   ```

4. **Insufficient Data**
   ```python
   # Check if data is adequate for model complexity
   model_complexity = formula_spec.count_parameters(data_context)
   data_points = data_context.total_captures
   
   if model_complexity > data_points / 10:
       print("⚠️ Model may be overparameterized")
       print(f"Parameters: {model_complexity}, Data points: {data_points}")
   ```

### Performance Issues

1. **Slow Convergence**
   ```python
   # Use faster optimization strategy for development
   result = pj.fit_model(
       model=pj.PradelModel(),
       formula=formula_spec,
       data=data_context,
       strategy="lbfgs",  # Usually fastest
       max_iterations=500  # Limit iterations for testing
   )
   ```

2. **Memory Issues**
   ```python
   # For very large datasets, sample for model development
   sample_data = pj.stratified_sample(data_context, n_samples=1000)
   result = pj.fit_model(
       model=pj.PradelModel(),
       formula=formula_spec,
       data=sample_data
   )
   ```

### Interpretation Issues

1. **Parameter Scaling**
   ```python
   # Standardize continuous covariates for interpretability
   formula_spec = pj.create_formula_spec(
       phi="~1 + standardize(age) + sex",
       p="~1 + standardize(weight)",
       f="~1"
   )
   ```

2. **Effect Size Interpretation**
   ```python
   # Convert logit-scale effects to probability scale
   logit_effect = result.parameter_estimates["phi_sex"]
   prob_effect = pj.logit_to_prob_effect(logit_effect)
   print(f"Sex increases survival probability by {prob_effect:.3f}")
   ```

---

**Next Steps:**
- [Optimization Guide](optimization.md) - Understanding optimization strategies and performance tuning
- [Formula System Guide](formulas.md) - Deep dive into the R-style formula syntax
- [Statistical Inference Guide](statistical-inference.md) - Comprehensive guide to statistical testing and confidence intervals