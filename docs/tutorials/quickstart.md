# Quick Start Guide

Get up and running with Pradel-JAX in just 5 minutes! This guide walks you through your first capture-recapture analysis.

## 🚀 Prerequisites

Before starting, make sure you have:
- Python 3.8+ installed
- The pradel-jax package installed ([Installation Guide](../user-guide/installation.md))
- Basic familiarity with capture-recapture concepts

**💡 New User?** Run our automated setup:
```bash
git clone https://github.com/chrischizinski/pradel-jax.git
cd pradel-jax
./quickstart.sh           # Basic setup + demo
# OR
./quickstart_parallel.sh  # Performance demo + benchmarks
source pradel_env/bin/activate
```

## 📊 Your First Analysis

### Step 1: Load Data

```python
import pradel_jax as pj

# Load sample dipper dataset (included with package)
data_context = pj.load_data("data/dipper_dataset.csv")

print(f"Loaded {data_context.n_individuals} individuals")
print(f"Capture occasions: {data_context.n_occasions}")
print(f"Available covariates: {list(data_context.covariates.keys())}")
```

Expected output:
```
Loaded 294 individuals
Capture occasions: 7
Available covariates: ['sex']
```

### Step 2: Specify Your Model

```python
# Create a simple Pradel model with sex effects on survival and detection
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",    # Survival: intercept + sex effect
    p="~1 + sex",      # Detection: intercept + sex effect  
    f="~1"             # Recruitment: constant
)

print(f"Model formula: {formula_spec}")
```

### Step 3: Fit the Model

```python
# Create and fit the model
model = pj.PradelModel()

# The optimization framework automatically selects the best strategy
result = pj.fit_model(
    model=model,
    formula=formula_spec,
    data=data_context
)

print(f"Optimization successful: {result.success}")
print(f"Strategy used: {result.strategy_used}")
print(f"Final log-likelihood: {result.log_likelihood:.2f}")
```

Expected output:
```
Optimization successful: True
Strategy used: scipy_lbfgs
Final log-likelihood: -879.06
```

### Step 4: Examine Results

```python
# Get parameter estimates
estimates = result.get_parameter_estimates()
print("Parameter Estimates:")
for param, value in estimates.items():
    print(f"  {param}: {value:.4f}")

# Get model summary
summary = result.summary()
print(f"\nModel Summary:")
print(f"AIC: {summary.aic:.2f}")
print(f"Number of parameters: {summary.n_parameters}")
```

Expected output:
```
Parameter Estimates:
  phi_(Intercept): 0.7854
  phi_sex: -0.1234
  p_(Intercept): 0.3456
  p_sex: 0.0987
  f_(Intercept): -1.6789

Model Summary:
AIC: 1768.12
Number of parameters: 5
```

## 🎉 Congratulations!

You've just fitted your first Pradel model! Here's what happened:

1. **Data Loading**: Pradel-JAX automatically detected the data format and loaded it
2. **Formula Parsing**: The R-style formulas were parsed and validated
3. **Optimization**: The framework selected L-BFGS-B as the best strategy
4. **Results**: You got parameter estimates and model fit statistics

## 🔍 What's Next?

Now that you have a working model, explore these next steps:

### Compare Models
```python
# Fit a simpler constant model for comparison
constant_formula = pj.create_formula_spec(
    phi="~1", p="~1", f="~1"
)

constant_result = pj.fit_model(model, constant_formula, data_context)

# Compare AIC values
print(f"Sex model AIC: {result.summary().aic:.2f}")
print(f"Constant model AIC: {constant_result.summary().aic:.2f}")
```

### Model Validation
```python
# Check convergence diagnostics
diagnostics = result.get_diagnostics()
print(f"Convergence: {diagnostics.converged}")
print(f"Gradient norm: {diagnostics.gradient_norm:.6f}")

# Residual analysis
residuals = result.get_residuals()
# Plot residuals (requires matplotlib)
```

### Export Results
```python
# Save results for later analysis
result.save("my_pradel_results.pkl")

# Export to CSV for R/Excel
estimates_df = result.to_dataframe()
estimates_df.to_csv("parameter_estimates.csv")
```

## 📚 Learn More

**Ready for more advanced topics?**

- [**Model Specification**](../user-guide/model-specification.md) - Learn about complex formulas
- [**Optimization Framework**](../user-guide/optimization.md) - Understand optimization strategies  
- [**Data Loading**](../user-guide/data-loading.md) - Work with your own data formats
- [**Multi-model Comparison**](model-comparison.md) - Compare multiple candidate models

**Need help with your data?**

- [**RMark Integration**](rmark-integration.md) - Validate against RMark results
- [**Large-scale Analysis**](large-scale.md) - Handle big datasets efficiently
- [**Performance Tips**](performance.md) - Optimize for speed and memory

## 🐛 Troubleshooting

### Common Issues

**"Data format not recognized"**
```python
# Specify format explicitly
data_context = pj.load_data("mydata.csv", format="rmark")
```

**"Optimization failed to converge"**
```python
# Try a different optimization strategy
result = pj.fit_model(
    model, formula_spec, data_context,
    strategy="multi_start"
)
```

**"Formula parsing error"**
```python
# Check available covariates
print(data_context.covariates.keys())

# Simplify formula
formula_spec = pj.create_formula_spec(phi="~1", p="~1", f="~1")
```

### Getting Help

- Check our [**Common Issues**](../user-guide/troubleshooting.md) guide
- Search [**GitHub Issues**](https://github.com/chrischizinski/pradel-jax/issues)
- Ask in [**GitHub Discussions**](https://github.com/chrischizinski/pradel-jax/discussions)

---

*This quick start guide gets you modeling in minutes. For comprehensive documentation, see the full [User Guide](../user-guide/README.md).*