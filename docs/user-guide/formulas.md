# Formula System Guide

Pradel-JAX uses a powerful R-style formula system that allows flexible specification of how model parameters depend on covariates. This guide covers the complete formula syntax, advanced features, and best practices.

## Table of Contents

1. [Formula Basics](#formula-basics)
2. [Parameter Types](#parameter-types)
3. [Covariate Types](#covariate-types)
4. [Formula Operators](#formula-operators)
5. [Transformations](#transformations)
6. [Time-Varying Covariates](#time-varying-covariates)
7. [Design Matrix Construction](#design-matrix-construction)
8. [Advanced Features](#advanced-features)
9. [Formula Validation](#formula-validation)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)

## Formula Basics

### R-Style Formula Syntax

Pradel-JAX formulas follow R's model formula syntax, which expresses relationships between parameters and covariates using a simple, intuitive language.

```python
import pradel_jax as pj

# Basic formula structure: parameter ~ covariates
# Left side: parameter name (implicit in Pradel-JAX)
# Right side: covariate specification

# Constant parameter (intercept only)
formula = "~1"

# Single covariate
formula = "~sex"          # Equivalent to "~1 + sex"
formula = "~1 + sex"      # Explicit intercept

# Multiple covariates
formula = "~sex + age + weight"
```

### Creating Formula Specifications

```python
# High-level API (recommended)
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",     # Survival depends on sex
    p="~1 + effort",    # Detection depends on effort
    f="~1"              # Constant recruitment
)

# Access individual formulas
print(formula_spec.phi.formula_string)  # "~1 + sex"
print(formula_spec.p.formula_string)   # "~1 + effort" 
print(formula_spec.f.formula_string)   # "~1"
```

## Parameter Types

### Pradel Model Parameters

1. **phi (φ) - Survival Probability**
   ```python
   phi="~1"              # Constant survival
   phi="~1 + sex"        # Sex-specific survival
   phi="~1 + age + sex"  # Age and sex effects
   ```

2. **p - Detection Probability**
   ```python
   p="~1"                # Constant detection
   p="~1 + effort"       # Effort-dependent detection
   p="~1 + weather"      # Weather effects on sampling
   ```

3. **f - Recruitment Rate**
   ```python
   f="~1"                # Constant recruitment
   f="~1 + habitat"      # Habitat quality effects
   f="~1 + density"      # Density-dependent recruitment
   ```

### Parameter Constraints and Links

```python
# Parameters are automatically transformed using appropriate link functions:
# phi, p: logit link (ensures 0 < parameter < 1)
# f: log link (ensures parameter > 0)

# Custom link functions (advanced)
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",
    p="~1 + effort", 
    f="~1",
    link_functions={
        "phi": "logit",  # Default
        "p": "probit",   # Alternative for detection
        "f": "log"       # Default for rates
    }
)
```

## Covariate Types

### 1. Categorical Variables

```python
# Categorical covariates are automatically dummy-coded
formula = "~1 + sex"        # Creates dummy for Male (if Female is reference)
formula = "~1 + region"     # Creates dummies for each region except reference

# View design matrix to understand dummy coding
data = pj.load_data("data.csv")
formula_spec = pj.create_formula_spec(phi="~1 + sex + region", p="~1", f="~1")
design_matrices = pj.create_design_matrices(formula_spec, data)
print("Phi design matrix columns:", design_matrices["phi"].columns)
```

### 2. Continuous Variables

```python
# Continuous covariates enter linearly by default
formula = "~1 + age"        # Linear age effect
formula = "~1 + weight"     # Linear weight effect
formula = "~1 + temperature" # Linear temperature effect

# Check for appropriate scaling
print("Age range:", data.covariates["age"].min(), "to", data.covariates["age"].max())
```

### 3. Factor Variables

```python
# Explicit factor specification for categorical variables
formula = "~1 + factor(year)"      # Treat year as categorical
formula = "~1 + factor(site_id)"   # Each site gets separate parameter

# Factor with custom reference level
formula = "~1 + factor(treatment, ref='control')"
```

## Formula Operators

### Basic Operators

```python
# Addition: Include multiple terms
formula = "~1 + sex + age"

# Subtraction: Remove specific terms (rarely used)
formula = "~sex + age - 1"    # Remove intercept

# Multiplication: Include main effects and interactions
formula = "~sex * age"        # Equivalent to "~sex + age + sex:age"

# Interaction only (no main effects)
formula = "~1 + sex:age"      # Only interaction term

# Power: Multiple interactions
formula = "~(sex + age + site)^2"  # All two-way interactions
```

### Advanced Operators

```python
# Nested effects
formula = "~1 + region/site"       # Sites nested within regions
# Equivalent to "~1 + region + region:site"

# Crossing with exclusions
formula = "~(sex + age + weight)^2 - sex:weight"  # All interactions except sex:weight

# As-is transformations
formula = "~1 + I(age^2)"          # Quadratic age (protect ^ from interpretation)
formula = "~1 + I(weight/1000)"    # Rescale weight to kg
```

## Transformations

### Built-in Transformations

```python
# Polynomial terms
formula = "~1 + age + I(age**2)"              # Quadratic
formula = "~1 + age + I(age**2) + I(age**3)"  # Cubic

# Mathematical functions
formula = "~1 + np.log(weight)"                # Natural logarithm
formula = "~1 + np.sqrt(body_size)"           # Square root
formula = "~1 + np.exp(temperature/10)"       # Exponential transformation

# Standardization (mean=0, sd=1)
formula = "~1 + standardize(age)"             # Built-in standardization
formula = "~1 + scale(weight)"                # Alternative name
```

### Custom Transformation Functions

```python
import numpy as np

# Define custom transformations
def log_plus_one(x):
    return np.log(x + 1)

def threshold(x, cutoff):
    return (x > cutoff).astype(float)

# Use in formulas
formula = "~1 + log_plus_one(count)"
formula = "~1 + threshold(temperature, 15)"   # Temperature above 15°C

# Register transformations for reuse
pj.register_transformation("log1p", log_plus_one)
formula = "~1 + log1p(count)"
```

### Spline Functions (Planned Feature)

```python
# Smooth functions of continuous covariates (future enhancement)
formula = "~1 + spline(age, df=3)"           # Natural spline with 3 df
formula = "~1 + smooth(day_of_year, k=5)"    # Cyclical smooth for seasonality
formula = "~1 + tensor(lat, lon, k=10)"      # Spatial smooth
```

## Time-Varying Covariates

### Structure and Naming

```python
# Time-varying covariates follow naming conventions:
# covariate_year: age_2016, age_2017, age_2018, ...
# covariate_occasion: weight_1, weight_2, weight_3, ...

# Load data with time-varying covariates
data = pj.load_data("data_with_time_varying.csv")

# Check detected time-varying structure
print("Time-varying covariates:")
for cov, occasions in data.time_varying_covariates.items():
    print(f"  {cov}: {occasions} occasions")
    
# Example output:
# Time-varying covariates:
#   age: 9 occasions (2016-2024)
#   tier: 9 occasions (2016-2024)
```

### Using Time-Varying Covariates in Formulas

```python
# Reference time-varying covariates with _tv suffix
formula_spec = pj.create_formula_spec(
    phi="~1 + age_tv",        # Age varies over time
    p="~1 + sex + tier_tv",   # Sex constant, tier time-varying
    f="~1"
)

# Multiple time-varying covariates
formula_spec = pj.create_formula_spec(
    phi="~1 + age_tv + weight_tv",
    p="~1 + tier_tv + effort_tv",
    f="~1 + sex"  # Sex is time-constant
)

# Interactions with time-varying covariates
formula_spec = pj.create_formula_spec(
    phi="~1 + sex * age_tv",      # Sex-age interaction (age time-varying)
    p="~1 + tier_tv + effort_tv", 
    f="~1"
)
```

### Time-Varying Covariate Validation

```python
# Validate temporal consistency
validation = data.validate_time_varying_covariates()

if validation.has_warnings:
    print("Time-varying covariate warnings:")
    for warning in validation.warnings:
        print(f"  - {warning}")

# Check covariate progression (e.g., age should increase)
age_matrix = data.get_time_varying_covariate("age")
age_changes = age_matrix[:, 1:] - age_matrix[:, :-1]
print(f"Age changes per occasion: mean={age_changes.mean():.2f}")

# Expected output: mean ≈ 1.0 for yearly age progression
```

## Design Matrix Construction

### Understanding Design Matrices

Design matrices translate formulas into numerical arrays suitable for optimization:

```python
# Create design matrices from formula specification
data = pj.load_data("data.csv")
formula_spec = pj.create_formula_spec(
    phi="~1 + sex + age",
    p="~1 + sex",
    f="~1"
)

model = pj.PradelModel()
design_matrices = model.build_design_matrices(formula_spec, data)

# Examine design matrix structure
print("Design matrix shapes:")
print(f"  phi: {design_matrices['phi'].shape}")  # (n_individuals * n_occasions, n_phi_params)
print(f"  p: {design_matrices['p'].shape}")      # (n_individuals * n_occasions, n_p_params)
print(f"  f: {design_matrices['f'].shape}")      # (n_individuals * n_occasions, n_f_params)

print("Parameter names:")
print(f"  phi: {design_matrices['phi_param_names']}")
print(f"  p: {design_matrices['p_param_names']}")  
print(f"  f: {design_matrices['f_param_names']}")
```

### Parameter Naming Convention

```python
# Parameters are named based on formula terms:
# Format: {parameter}_{term}

# Example with sex and age:
formula = "~1 + sex + age"
# Creates parameters: phi_intercept, phi_sex, phi_age

# Example with interactions:
formula = "~1 + sex * age" 
# Creates parameters: phi_intercept, phi_sex, phi_age, phi_sex:age

# Factor variables:
formula = "~1 + factor(region)"
# Creates parameters: phi_intercept, phi_region_north, phi_region_south
# (if 'east' is the reference level)
```

### Design Matrix Caching

```python
# Design matrices are automatically cached for efficiency
# Reusing the same formula_spec and data reuses cached matrices

# Clear cache if data changes
pj.clear_design_matrix_cache()

# Control caching behavior
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",
    p="~1",
    f="~1",
    use_cache=True  # Default: True
)
```

## Advanced Features

### Offset Terms

```python
# Include offset terms (fixed effects with coefficient = 1)
formula = "~1 + sex + offset(log_effort)"
# log_effort enters with coefficient fixed at 1.0
```

### Constrained Parameters

```python
# Set parameter constraints through bounds
bounds = {
    "phi_age": (0.0, 0.1),      # Age effect bounded
    "p_effort": (0.0, 2.0),     # Effort effect upper bound
    "f_intercept": (-5.0, 2.0)  # Recruitment intercept range
}

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data,
    parameter_bounds=bounds
)
```

### Formula Composition

```python
# Build complex formulas programmatically
base_terms = ["1", "sex", "age"]
interaction_terms = [f"{t1}:{t2}" for t1 in base_terms[1:] for t2 in base_terms[1:] if t1 < t2]
full_formula = "~" + " + ".join(base_terms + interaction_terms)
print(full_formula)  # "~1 + sex + age + age:sex"

# Conditional formula construction
def build_formula(include_sex=True, include_age=True, include_interaction=False):
    terms = ["1"]
    if include_sex:
        terms.append("sex")
    if include_age:
        terms.append("age")
    if include_interaction and include_sex and include_age:
        terms.append("sex:age")
    return "~" + " + ".join(terms)

formula = build_formula(include_interaction=True)
```

## Formula Validation

### Automatic Validation

```python
# Formulas are validated when creating formula specifications
try:
    formula_spec = pj.create_formula_spec(
        phi="~1 + nonexistent_covariate",  # Error: covariate not found
        p="~1 + sex + sex",                # Warning: duplicate terms
        f="~1 + I(age**"                   # Error: syntax error
    )
except pj.ModelSpecificationError as e:
    print(f"Formula validation failed: {e}")
```

### Manual Validation

```python
# Validate formula against data
data = pj.load_data("data.csv")
formula_spec = pj.create_formula_spec(phi="~1 + sex + age", p="~1", f="~1")

validation_result = formula_spec.validate(data)
print(f"Valid: {validation_result.is_valid}")

if validation_result.warnings:
    print("Warnings:")
    for warning in validation_result.warnings:
        print(f"  - {warning}")

if validation_result.errors:
    print("Errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Common Validation Issues

```python
# Check for common problems before model fitting

# 1. Missing covariates
available_covariates = set(data.covariates.keys())
formula_covariates = formula_spec.get_required_covariates()
missing = formula_covariates - available_covariates
if missing:
    print(f"Missing covariates: {missing}")

# 2. Insufficient factor levels
for covariate in formula_spec.get_categorical_covariates():
    levels = data.covariates[covariate].nunique()
    if levels < 2:
        print(f"Warning: {covariate} has only {levels} level(s)")

# 3. Model complexity vs. data size
n_parameters = formula_spec.count_parameters(data)
n_observations = data.total_captures
if n_parameters > n_observations / 10:
    print(f"Warning: Model may be overparameterized ({n_parameters} params, {n_observations} observations)")
```

## Examples

### Example 1: Basic Categorical Analysis

```python
import pradel_jax as pj

# Load data
data = pj.load_data("bird_data.csv")
print(f"Available covariates: {list(data.covariates.keys())}")
print(f"Sex levels: {data.covariates['sex'].unique()}")

# Sex effect on all parameters
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",  # Different survival for males/females
    p="~1 + sex",    # Different detection for males/females
    f="~1 + sex"     # Different recruitment for males/females
)

# Fit model
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data
)

# Interpret results
if result.success:
    # Parameter estimates are on logit scale for phi and p, log scale for f
    phi_female = pj.logit_inverse(result.parameter_estimates["phi_intercept"])
    phi_male = pj.logit_inverse(result.parameter_estimates["phi_intercept"] + 
                               result.parameter_estimates["phi_sex"])
    
    print(f"Female survival: {phi_female:.3f}")
    print(f"Male survival: {phi_male:.3f}")
    print(f"Survival difference: {phi_male - phi_female:.3f}")
```

### Example 2: Continuous Covariate Analysis

```python
# Age effect analysis
data = pj.load_data("mammal_data.csv")

# Check age distribution
print(f"Age range: {data.covariates['age'].min():.1f} to {data.covariates['age'].max():.1f}")
print(f"Age mean: {data.covariates['age'].mean():.1f}")

# Standardize age for better numerical stability and interpretation
formula_spec = pj.create_formula_spec(
    phi="~1 + standardize(age)",      # Linear age effect on survival
    p="~1 + I(standardize(age)**2)",  # Quadratic age effect on detection
    f="~1"                            # Constant recruitment
)

result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data,
    compute_standard_errors=True
)

if result.success:
    age_effect = result.parameter_estimates["phi_standardize(age)"]
    age_se = result.standard_errors["phi_standardize(age)"]
    
    print(f"Standardized age effect on survival: {age_effect:.3f} ± {age_se:.3f}")
    print(f"Interpretation: One SD increase in age changes survival logit by {age_effect:.3f}")
```

### Example 3: Complex Interaction Model

```python
# Sex-age interaction with environmental effects
formula_spec = pj.create_formula_spec(
    phi="~1 + sex * age + temperature",     # Sex-age interaction plus temperature
    p="~1 + effort + weather",              # Sampling effort and weather effects
    f="~1 + habitat_quality"                # Habitat affects recruitment
)

# Check model complexity
n_params = formula_spec.count_parameters(data)
print(f"Model complexity: {n_params} parameters")
print(f"Data points: {data.total_captures}")
print(f"Ratio: {data.total_captures / n_params:.1f} observations per parameter")

# Fit with robust optimization
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data,
    strategy="multi_start",  # More robust for complex models
    compute_standard_errors=True
)

# Examine interaction effect
if result.success:
    interaction_effect = result.parameter_estimates.get("phi_sex:age", 0.0)
    interaction_se = result.standard_errors.get("phi_sex:age", 0.0)
    
    if interaction_effect != 0.0:
        z_score = interaction_effect / interaction_se
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))
        
        print(f"Sex-age interaction: {interaction_effect:.3f} ± {interaction_se:.3f}")
        print(f"Z-score: {z_score:.2f}, p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Significant interaction: age effects differ between sexes")
        else:
            print("Non-significant interaction: age effects similar between sexes")
```

### Example 4: Time-Varying Covariate Model

```python
# Nebraska data analysis with time-varying age and tier
data = pj.load_data("nebraska_data.csv")

# Examine time-varying structure
print("Time-varying covariates:")
for cov, occasions in data.time_varying_covariates.items():
    print(f"  {cov}: {occasions} occasions")

# Complex model with time-varying and constant effects
formula_spec = pj.create_formula_spec(
    phi="~1 + sex + age_tv",              # Sex constant, age time-varying
    p="~1 + sex + tier_tv",               # Sex constant, tier time-varying  
    f="~1 + sex * age_tv"                 # Sex-age interaction (age time-varying)
)

# Validate before fitting
validation = formula_spec.validate(data)
if not validation.is_valid:
    print("Formula validation issues:")
    for error in validation.errors:
        print(f"  - {error}")
else:
    # Fit model
    result = pj.fit_model(
        model=pj.PradelModel(),
        formula=formula_spec,
        data=data,
        compute_standard_errors=True,
        bootstrap_confidence_intervals=True
    )
    
    if result.success:
        print(f"✅ Model converged (AIC: {result.aic:.2f})")
        print("\nTime-varying age effect:")
        
        age_effect = result.parameter_estimates["phi_age_tv"]
        age_ci = result.confidence_intervals["phi_age_tv"]
        
        print(f"Age effect: {age_effect:.3f} [{age_ci['lower']:.3f}, {age_ci['upper']:.3f}]")
        
        # Biological interpretation
        prob_change = pj.logit_to_prob_effect(age_effect)
        print(f"Each year of age changes survival probability by {prob_change:.3f}")
```

## Troubleshooting

### Common Formula Problems

1. **Covariate Not Found**
   ```python
   # Error: KeyError when covariate doesn't exist in data
   # Check available covariates
   print("Available covariates:", list(data.covariates.keys()))
   
   # Check for typos in formula
   formula_spec = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")  # Correct
   # formula_spec = pj.create_formula_spec(phi="~1 + Sex", p="~1", f="~1")  # Wrong case
   ```

2. **Syntax Errors**
   ```python
   # Common syntax mistakes:
   # Wrong: formula = "~1 + I(age^2)"     # ^ is interpreted as interaction
   # Right: formula = "~1 + I(age**2)"   # ** for power
   
   # Wrong: formula = "~1 + log(age)"    # log function not available
   # Right: formula = "~1 + np.log(age)" # Use numpy functions
   ```

3. **Factor Reference Level Issues**
   ```python
   # Check factor levels and reference
   print("Sex levels:", data.covariates["sex"].value_counts())
   
   # Explicit reference level specification
   formula = "~1 + factor(sex, ref='Female')"
   
   # Or use pandas categorical with explicit categories
   data.covariates["sex"] = pd.Categorical(
       data.covariates["sex"], 
       categories=["Female", "Male"]  # Female will be reference
   )
   ```

4. **Model Complexity Issues**
   ```python
   # Too many parameters for available data
   n_params = formula_spec.count_parameters(data)
   n_observations = data.total_captures
   
   if n_params > n_observations / 10:
       print("Consider simplifying model:")
       print("- Remove interactions")
       print("- Combine factor levels") 
       print("- Remove less important covariates")
   ```

5. **Time-Varying Covariate Problems**
   ```python
   # Check if time-varying covariates detected properly
   print("Detected time-varying:", list(data.time_varying_covariates.keys()))
   
   # If not detected, check naming convention:
   # Should be: age_2016, age_2017, age_2018 (or age_1, age_2, age_3)
   # Not: age2016, age16, ageYear1
   
   # Manual specification if needed
   data.specify_time_varying_covariates(
       age=["age_2016", "age_2017", "age_2018"],
       tier=["tier_2016", "tier_2017", "tier_2018"]
   )
   ```

### Performance Issues

1. **Large Design Matrices**
   ```python
   # For large datasets with many covariates, monitor memory usage
   import psutil
   
   before_memory = psutil.virtual_memory().used / 1e9
   design_matrices = model.build_design_matrices(formula_spec, data)
   after_memory = psutil.virtual_memory().used / 1e9
   
   print(f"Design matrix memory usage: {after_memory - before_memory:.2f} GB")
   ```

2. **Complex Formula Parsing**
   ```python
   # For very complex formulas, parse once and reuse
   formula_spec = pj.create_formula_spec(
       phi="~1 + sex * age * region + I(weight**2)",
       p="~1 + effort + weather + sex:effort",
       f="~1 + habitat + density"
   )
   
   # Cache the design matrices
   design_matrices = model.build_design_matrices(formula_spec, data, cache=True)
   ```

### Getting Help

For formula-related issues:

1. **Check formula syntax** - Use simple formulas first, then add complexity
2. **Validate against data** - Use `formula_spec.validate(data)` before fitting
3. **Examine design matrices** - Look at the actual numerical matrices created
4. **Start simple** - Begin with constant models, add covariates incrementally
5. **Consult R documentation** - The syntax closely follows R's model formulas

---

**Next Steps:**
- [Statistical Inference Guide](statistical-inference.md) - Understanding parameter estimation and hypothesis testing
- [Model Comparison Guide](model-comparison.md) - Comparing and selecting between different model formulations
- [Advanced Modeling Techniques](advanced-modeling.md) - Complex modeling scenarios and custom extensions