# Troubleshooting Guide

This guide helps diagnose and resolve common issues when using Pradel-JAX. Issues are organized by category with step-by-step solutions and prevention strategies.

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Installation Issues](#installation-issues)
3. [Data Loading Problems](#data-loading-problems)
4. [Model Specification Errors](#model-specification-errors)
5. [Optimization Failures](#optimization-failures)
6. [Performance Issues](#performance-issues)
7. [Memory Problems](#memory-problems)
8. [Statistical Issues](#statistical-issues)
9. [Platform-Specific Issues](#platform-specific-issues)
10. [Getting Help](#getting-help)

## Quick Diagnostic Tools

### System Information Check

```python
import pradel_jax as pj
import sys
import platform

def system_diagnostics():
    """Run comprehensive system diagnostics."""
    print("=== Pradel-JAX System Diagnostics ===")
    
    # Basic system info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Pradel-JAX: {pj.__version__}")
    
    # JAX info
    try:
        import jax
        print(f"JAX: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
        print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    except Exception as e:
        print(f"JAX issue: {e}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.total / 1e9:.1f}GB total, {memory.available / 1e9:.1f}GB available")
    except Exception as e:
        print(f"Memory info unavailable: {e}")
    
    # Test basic functionality
    try:
        # Test data loading
        test_data = pj.load_data("tests/fixtures/small_dataset.csv")
        print(f"✓ Data loading works")
        
        # Test model fitting
        result = pj.fit_model(data=test_data)
        print(f"✓ Model fitting works")
        
    except Exception as e:
        print(f"✗ Basic functionality failed: {e}")

# Run diagnostics
system_diagnostics()
```

### Installation Verification

```python
def verify_installation():
    """Verify Pradel-JAX installation completeness."""
    
    required_modules = [
        'pradel_jax.core.api',
        'pradel_jax.models',
        'pradel_jax.data.adapters',
        'pradel_jax.formulas',
        'pradel_jax.optimization',
    ]
    
    optional_modules = [
        'mlflow',
        'optuna',
        'scikit_optimize',
        'seaborn'
    ]
    
    print("=== Installation Verification ===")
    
    # Check required modules
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
    
    # Check optional modules
    print("\nOptional modules:")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"⚠ {module} (optional)")

verify_installation()
```

## Installation Issues

### JAX Installation Problems

**Problem**: JAX fails to install or import

**Solutions**:

```bash
# 1. Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# 2. Install JAX explicitly
pip install --upgrade jax jaxlib

# 3. For CPU-only systems
pip install jax[cpu]

# 4. For CUDA systems (check CUDA version first)
nvidia-smi  # Check CUDA version
pip install jax[cuda11_pip]  # For CUDA 11.x
pip install jax[cuda12_pip]  # For CUDA 12.x

# 5. Clear pip cache if still failing
pip cache purge
pip install jax jaxlib --no-cache-dir
```

**Verification**:
```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# Test basic JAX functionality
x = jax.random.normal(jax.random.PRNGKey(42), (10,))
print(f"JAX test: {jax.numpy.sum(x)}")  # Should work without error
```

### Dependency Version Conflicts

**Problem**: Conflicting package versions

**Solution**:

```bash
# 1. Create fresh virtual environment
python -m venv fresh_pradel_env
source fresh_pradel_env/bin/activate  # On Windows: fresh_pradel_env\Scripts\activate

# 2. Install requirements in order
pip install --upgrade pip
pip install -r requirements.txt

# 3. If conflicts persist, check versions
pip list | grep -E "(jax|scipy|numpy|pandas)"

# 4. Use compatibility versions
pip install "jax>=0.4.20,<0.5.0" "scipy>=1.7.0,<2.0.0"
```

### Import Errors

**Problem**: `ImportError` or `ModuleNotFoundError` for pradel_jax

**Solutions**:

```python
# 1. Check Python path
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# 2. Check if package is installed
try:
    import pradel_jax
    print(f"Package location: {pradel_jax.__file__}")
except ImportError as e:
    print(f"Import error: {e}")

# 3. Development installation issue
# Run from project directory:
pip install -e .

# 4. Virtual environment not activated
# Make sure virtual environment is active:
which python  # Should point to virtual environment
```

## Data Loading Problems

### File Format Not Recognized

**Problem**: "Format not detected" or "Unsupported format" errors

**Diagnosis**:
```python
# Check file format
import pandas as pd
df = pd.read_csv("your_data.csv")
print("Columns:", df.columns.tolist())
print("First few rows:")
print(df.head())
```

**Solutions**:

```python
# 1. Explicit format specification
data = pj.load_data("data.csv", format_type="rmark")  # or "y_column", "generic"

# 2. Check for RMark format
# Need 'ch' column for RMark format
if 'ch' in df.columns:
    data = pj.load_data("data.csv", format_type="rmark")

# 3. Check for Y-column format
y_columns = [col for col in df.columns if col.startswith('Y')]
if y_columns:
    data = pj.load_data("data.csv", format_type="y_column")

# 4. Generic format with explicit columns
data = pj.load_data(
    "data.csv",
    format_type="generic",
    capture_columns=["Y2016", "Y2017", "Y2018"],  # Specify explicitly
    covariate_columns=["sex", "age", "region"]
)
```

### Capture History Parsing Errors

**Problem**: Issues with capture history strings

**Diagnosis**:
```python
# Check capture history format
df = pd.read_csv("data.csv")
if 'ch' in df.columns:
    print("Capture history examples:")
    print(df['ch'].head(10))
    print(f"Unique lengths: {df['ch'].str.len().unique()}")
    print(f"Unique characters: {set(''.join(df['ch'].dropna()))}")
```

**Solutions**:

```python
# 1. Handle different separators
data = pj.load_data(
    "data.csv",
    capture_separator=",",    # For "1,1,0,1,0" format
    format_type="rmark"
)

# 2. Preserve leading zeros
data = pj.load_data(
    "data.csv",
    preserve_leading_zeros=True,  # Keep "01010" instead of "1010"
    format_type="rmark"
)

# 3. Handle missing values in capture histories
data = pj.load_data(
    "data.csv",
    na_values=["", "NA", "NULL", "-"],
    format_type="rmark"
)

# 4. Clean data before loading
df_clean = df.copy()
df_clean['ch'] = df_clean['ch'].str.replace(' ', '').str.replace(',', '')
data = pj.load_data(df_clean, format_type="rmark")
```

### Covariate Issues

**Problem**: Covariate loading or validation errors

**Solutions**:

```python
# 1. Check covariate data types
df = pd.read_csv("data.csv")
print("Data types:")
print(df.dtypes)

# 2. Handle missing values
data = pj.load_data(
    "data.csv",
    na_values=["", "NA", "NULL", -999, "missing"],
    fill_missing_method="mode"  # or "mean", "median", "drop"
)

# 3. Explicit data type specification
data = pj.load_data(
    "data.csv",
    dtype={
        "sex": "category",
        "age": "float64", 
        "weight": "float32",
        "region": "category"
    }
)

# 4. Handle categorical variables
df['sex'] = df['sex'].astype('category')
df['region'] = df['region'].astype('category')
data = pj.load_data(df)
```

### Time-Varying Covariate Problems

**Problem**: Time-varying covariates not detected

**Diagnosis**:
```python
# Check column names for time-varying pattern
df = pd.read_csv("data.csv")
columns = df.columns.tolist()
print("All columns:", columns)

# Look for time-varying patterns
import re
tv_patterns = [
    r'.*_\d{4}$',    # name_2016 pattern
    r'.*_\d+$',      # name_1 pattern  
]

for pattern in tv_patterns:
    matches = [col for col in columns if re.match(pattern, col)]
    if matches:
        print(f"Potential time-varying ({pattern}): {matches}")
```

**Solutions**:

```python
# 1. Check naming convention
# Should be: age_2016, age_2017, age_2018
# Not: age2016, age_16, ageYear1

# 2. Manual specification
data = pj.load_data("data.csv")
data.specify_time_varying_covariates(
    age=["age_2016", "age_2017", "age_2018"],
    tier=["tier_2016", "tier_2017", "tier_2018"]
)

# 3. Rename columns if needed
df = pd.read_csv("data.csv")
df = df.rename(columns={"age2016": "age_2016", "age2017": "age_2017"})
data = pj.load_data(df)
```

## Model Specification Errors

### Formula Syntax Errors

**Problem**: Formula parsing failures

**Diagnosis**:
```python
# Test formula parsing
try:
    formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
    print("✓ Formula syntax is valid")
except pj.ModelSpecificationError as e:
    print(f"Formula error: {e}")
    print(f"Available covariates: {e.available_covariates}")
```

**Solutions**:

```python
# 1. Check covariate names
data = pj.load_data("data.csv")
print("Available covariates:", list(data.covariates.keys()))

# 2. Fix common syntax errors
# Wrong: formula = "~1 + Sex"  (wrong case)
# Right: formula = "~1 + sex"

# Wrong: formula = "~1 + I(age^2)"  (^ interpreted as interaction)
# Right: formula = "~1 + I(age**2)"

# Wrong: formula = "~1 + log(age)"  (log not available)
# Right: formula = "~1 + np.log(age)"

# 3. Validate before use
formula_spec = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
validation = formula_spec.validate(data)
if not validation.is_valid:
    for error in validation.errors:
        print(f"Validation error: {error}")
```

### Missing Covariates

**Problem**: Formula references covariates not in data

**Solutions**:

```python
# 1. Check available vs required covariates
data = pj.load_data("data.csv")
formula_spec = pj.create_formula_spec(phi="~1 + sex + age", p="~1", f="~1")

available = set(data.covariates.keys())
required = formula_spec.get_required_covariates()
missing = required - available

if missing:
    print(f"Missing covariates: {missing}")
    print(f"Available covariates: {available}")

# 2. Simplify formula
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",  # Remove missing 'age'
    p="~1",
    f="~1"
)

# 3. Add missing covariates to data
if 'age' not in data.covariates and 'Age' in data.covariates:
    data.covariates['age'] = data.covariates['Age']  # Fix case
```

### Model Complexity Issues

**Problem**: Model too complex for available data

**Diagnosis**:
```python
# Check model complexity vs data size
formula_spec = pj.create_formula_spec(phi="~1 + sex * age * region", p="~1 + sex", f="~1")
n_parameters = formula_spec.count_parameters(data)
n_observations = data.total_captures

print(f"Parameters: {n_parameters}")
print(f"Observations: {n_observations}")
print(f"Ratio: {n_observations / n_parameters:.1f} observations per parameter")

if n_parameters > n_observations / 10:
    print("⚠ Model may be overparameterized")
```

**Solutions**:

```python
# 1. Simplify model
simple_formula = pj.create_formula_spec(
    phi="~1 + sex",      # Remove interactions
    p="~1",              # Constant detection
    f="~1"               # Constant recruitment
)

# 2. Combine factor levels
if 'region' in data.covariates:
    # Combine rare regions
    region_counts = data.covariates['region'].value_counts()
    rare_regions = region_counts[region_counts < 10].index
    data.covariates['region_combined'] = data.covariates['region'].replace(
        dict.fromkeys(rare_regions, 'Other')
    )

# 3. Use continuous instead of categorical
# Instead of factor(age_class), use continuous age
formula_spec = pj.create_formula_spec(
    phi="~1 + age",      # Continuous age instead of age categories
    p="~1 + sex",
    f="~1"
)
```

## Optimization Failures

### Convergence Problems

**Problem**: "Optimization failed to converge" errors

**Diagnosis**:
```python
# Try fitting with verbose output
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    verbose=True,
    return_trace=True  # Get optimization trace
)

if not result.success:
    print(f"Failure reason: {result.optimization_message}")
    print(f"Final parameters: {result.final_parameters}")
    print(f"Function evaluations: {result.n_function_evaluations}")
```

**Solutions**:

```python
# 1. Try different optimization strategy
strategies = ["lbfgs", "slsqp", "multi_start", "adam"]
for strategy in strategies:
    result = pj.fit_model(
        formula=formula_spec,
        data=data_context,
        strategy=strategy
    )
    if result.success:
        print(f"✓ Success with {strategy}")
        break
    else:
        print(f"✗ Failed with {strategy}: {result.optimization_message}")

# 2. Use multi-start for robustness
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="multi_start",
    multi_start_attempts=20  # Try many starting points
)

# 3. Adjust optimization parameters
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    max_iterations=5000,    # More iterations
    tolerance=1e-4,         # Relaxed tolerance
    learning_rate=0.001     # For Adam optimizer
)

# 4. Start with simpler model
simple_result = pj.fit_model(
    formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
    data=data_context
)

# Use simple model parameters as starting point
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    initial_parameters=simple_result.parameter_estimates
)
```

### Parameter Bounds Issues

**Problem**: Parameters hitting bounds or going to infinity

**Solutions**:

```python
# 1. Check parameter bounds
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    check_bounds=True,
    verbose=True
)

# 2. Custom parameter bounds
custom_bounds = {
    "phi_intercept": (-5.0, 5.0),     # Reasonable logit range
    "phi_sex": (-2.0, 2.0),           # Effect size bounds
    "p_intercept": (-3.0, 3.0),       # Detection probability bounds
    "f_intercept": (-2.0, 2.0)        # Recruitment rate bounds
}

result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    parameter_bounds=custom_bounds
)

# 3. Standardize continuous covariates
# This helps with parameter scaling
formula_with_standardization = pj.create_formula_spec(
    phi="~1 + sex + standardize(age)",    # Standardized age
    p="~1 + standardize(weight)",         # Standardized weight
    f="~1"
)
```

### Gradient/Hessian Issues

**Problem**: "Gradient computation failed" or "Hessian singular" errors

**Solutions**:

```python
# 1. Check for data issues
validation = data_context.validate()
if not validation.is_valid:
    print("Data validation errors:")
    for error in validation.errors:
        print(f"  - {error}")

# 2. Use finite differences for gradients
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    gradient_method="finite_differences",  # Instead of automatic differentiation
    hessian_method="finite_differences"    # For standard errors
)

# 3. Remove perfect correlations
# Check for highly correlated covariates
correlation_matrix = data_context.covariates.corr()
high_corr = correlation_matrix[correlation_matrix.abs() > 0.95]
print("High correlations:", high_corr)

# 4. Add regularization (experimental)
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    regularization=1e-6  # Small L2 penalty
)
```

## Performance Issues

### Slow Model Fitting

**Problem**: Model fitting takes too long

**Solutions**:

```python
# 1. Use faster optimization strategy
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="lbfgs"  # Usually fastest
)

# 2. Sample data for development
sample_data = pj.stratified_sample(data_context, n_samples=1000)
result = pj.fit_model(formula=formula_spec, data=sample_data)

# 3. Warm-start JAX compilation
pj.warm_up_jax()  # Pre-compile functions
result = pj.fit_model(formula=formula_spec, data=data_context)

# 4. Disable expensive features during development
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    compute_standard_errors=False,         # Skip during development
    bootstrap_confidence_intervals=False   # Skip bootstrap
)
```

### Memory Usage Problems

**Problem**: High memory usage or out-of-memory errors

**Solutions**:

```python
# 1. Monitor memory usage
import psutil

def check_memory():
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}% used ({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)")

check_memory()

# 2. Use memory-efficient options
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    low_memory_mode=True,              # Reduce memory usage
    cache_design_matrices=False,       # Don't cache large matrices
    precision="float32"                # Use 32-bit instead of 64-bit
)

# 3. Sample large datasets
if data_context.n_individuals > 10000:
    sample_data = pj.stratified_sample(data_context, n_samples=5000)
    result = pj.fit_model(formula=formula_spec, data=sample_data)

# 4. Process in chunks for very large datasets
def process_in_chunks(data_path, chunk_size=5000):
    results = []
    for chunk in pj.load_data_chunks(data_path, chunk_size=chunk_size):
        result = pj.fit_model(formula=formula_spec, data=chunk)
        results.append(result)
    return results
```

## Memory Problems

### Out of Memory Errors

**Problem**: System runs out of memory during analysis

**Immediate Solutions**:

```python
# 1. Clear Python memory
import gc
gc.collect()  # Force garbage collection

# 2. Clear JAX caches
import jax
jax.clear_caches()

# 3. Restart Python session
# Sometimes the only solution is to restart
```

**Prevention**:

```python
# 1. Monitor memory throughout analysis
import psutil

class MemoryMonitor:
    def __init__(self, threshold_percent=90):
        self.threshold = threshold_percent
    
    def check(self, label=""):
        memory = psutil.virtual_memory()
        if memory.percent > self.threshold:
            print(f"⚠ High memory usage {label}: {memory.percent}%")
        return memory.percent

monitor = MemoryMonitor()

# Use throughout analysis
monitor.check("after data loading")
result = pj.fit_model(formula=formula_spec, data=data_context)
monitor.check("after model fitting")

# 2. Use context managers for temporary objects
with pj.temporary_data_context(large_dataset) as temp_data:
    sample = temp_data.sample(n=1000)
    result = pj.fit_model(formula=formula_spec, data=sample)
    # temp_data automatically cleaned up

# 3. Explicit cleanup
def memory_efficient_analysis():
    data = pj.load_data("large_dataset.csv")
    
    # Fit model
    result = pj.fit_model(formula=formula_spec, data=data)
    
    # Extract only what you need
    summary = {
        "aic": result.aic,
        "parameters": dict(result.parameter_estimates)
    }
    
    # Clean up large objects
    del data, result
    gc.collect()
    
    return summary
```

### Design Matrix Memory Issues

**Problem**: Design matrices too large for memory

**Solutions**:

```python
# 1. Check design matrix size before creation
n_individuals = data_context.n_individuals
n_occasions = data_context.n_occasions
n_parameters = formula_spec.count_parameters(data_context)

matrix_size_gb = (n_individuals * n_occasions * n_parameters * 8) / 1e9
print(f"Estimated design matrix size: {matrix_size_gb:.2f}GB")

if matrix_size_gb > 2.0:  # Too large
    print("Design matrix too large, using alternatives:")
    
    # 2. Sample data
    sample_data = pj.stratified_sample(data_context, n_samples=2000)
    
    # 3. Simplify model
    simple_formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
    
    # 4. Use sparse matrices (if available)
    result = pj.fit_model(
        formula=simple_formula,
        data=sample_data,
        use_sparse_matrices=True  # Experimental feature
    )
```

## Statistical Issues

### Standard Error Computation Problems

**Problem**: "Hessian computation failed" or "Standard errors not available"

**Solutions**:

```python
# 1. Use finite differences instead of automatic differentiation
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    hessian_method="finite_differences",
    finite_difference_step=1e-6
)

# 2. Use bootstrap standard errors as fallback
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    compute_standard_errors=False,     # Skip Hessian-based
    bootstrap_confidence_intervals=True,  # Use bootstrap instead
    bootstrap_config={"n_bootstrap": 500}
)

# 3. Check model identifiability
# Simplify model if too complex
simple_formula = pj.create_formula_spec(phi="~1", p="~1", f="~1")
simple_result = pj.fit_model(formula=simple_formula, data=data_context)

if simple_result.success:
    print("Simple model works, issue is with model complexity")
```

### Confidence Interval Issues

**Problem**: Unrealistic or missing confidence intervals

**Diagnosis**:
```python
# Check parameter estimates and standard errors
for param in result.parameter_names:
    est = result.parameter_estimates[param]
    se = result.standard_errors.get(param, float('nan'))
    ci = result.confidence_intervals.get(param, {})
    
    print(f"{param}: {est:.3f} ± {se:.3f} [{ci.get('lower', 'NA')}, {ci.get('upper', 'NA')}]")
    
    # Flag potentially problematic parameters
    if abs(est) > 10:  # Very large estimate
        print(f"  ⚠ Large parameter estimate for {param}")
    if se > abs(est):  # Standard error larger than estimate
        print(f"  ⚠ Large uncertainty for {param}")
```

**Solutions**:

```python
# 1. Use bootstrap confidence intervals for robustness
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    bootstrap_confidence_intervals=True,
    bootstrap_config={
        "n_bootstrap": 1000,
        "method": "bca",        # Bias-corrected and accelerated
        "confidence_level": 0.95
    }
)

# 2. Check for model overparameterization
n_params = formula_spec.count_parameters(data_context)
n_obs = data_context.total_captures

if n_params > n_obs / 10:
    print("Model may be overparameterized, consider simplifying")

# 3. Transform parameters to natural scale for interpretation
phi_intercept_logit = result.parameter_estimates["phi_intercept"]
phi_intercept_prob = 1 / (1 + np.exp(-phi_intercept_logit))
print(f"Baseline survival probability: {phi_intercept_prob:.3f}")
```

### Model Comparison Issues

**Problem**: AIC/BIC values seem wrong or models don't rank sensibly

**Solutions**:

```python
# 1. Verify all models converged
models = {}
failed_models = []

for name, formula in candidate_formulas.items():
    result = pj.fit_model(formula=formula, data=data_context)
    if result.success:
        models[name] = result
    else:
        failed_models.append(name)
        print(f"⚠ Model {name} failed to converge")

print(f"Successfully fitted {len(models)} models")

# 2. Check for numerical issues
comparison = pj.compare_models(models)
print("Model comparison:")
print(comparison[['aic', 'bic', 'log_likelihood', 'n_parameters']])

# Look for suspiciously low AIC values or negative log-likelihoods
for name, result in models.items():
    if result.log_likelihood > 0:  # Should be negative
        print(f"⚠ Suspicious positive log-likelihood for {name}")

# 3. Use cross-validation for robust model comparison
cv_results = pj.cross_validate_models(models, data_context, cv_folds=5)
print("Cross-validation results:")
print(cv_results)
```

## Platform-Specific Issues

### Windows Issues

**Problem**: Path-related errors or performance issues on Windows

**Solutions**:

```python
# 1. Use pathlib for cross-platform paths
from pathlib import Path

data_path = Path("data") / "dataset.csv"  # Works on all platforms
data = pj.load_data(data_path)

# 2. Handle Windows-specific NumPy/SciPy issues
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Avoid conflicts

# 3. Use forward slashes in paths
data = pj.load_data("data/dataset.csv")  # Works on Windows too
```

### macOS Issues

**Problem**: Performance issues or installation problems on macOS

**Solutions**:

```bash
# 1. Ensure Xcode command line tools installed
xcode-select --install

# 2. Use conda for problematic packages
conda install numpy scipy

# 3. For M1/M2 Macs, ensure ARM64 versions
pip install --upgrade jax[cpu]  # Should automatically get ARM64 version
```

**macOS-specific JAX configuration**:
```python
# Check if running on Apple Silicon
import platform
if platform.processor() == 'arm':
    print("Running on Apple Silicon")
    # JAX should automatically use optimized ARM64 libraries
    
    # Enable Metal GPU acceleration (experimental)
    os.environ["JAX_PLATFORM_NAME"] = "metal"
```

### Linux Issues

**Problem**: Shared library or permission errors

**Solutions**:

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev

# For CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# 2. Fix shared library issues
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib

# 3. For CUDA issues on Linux
# Check CUDA installation
nvidia-smi
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
```

### Cloud/Server Issues

**Problem**: Issues running on cloud platforms or servers

**Solutions**:

```python
# 1. Headless operation (no display)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# 2. Limit resource usage
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    max_memory_gb=8,        # Limit memory usage
    max_cpu_cores=4,        # Limit CPU usage
    timeout_minutes=60      # Set timeout
)

# 3. Check for available resources
import psutil
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Memory: {psutil.virtual_memory().total / 1e9:.1f}GB")

# Adjust configuration based on resources
if psutil.cpu_count() < 4:
    parallel_config = {"parallel": False}
else:
    parallel_config = {"parallel": True, "n_jobs": psutil.cpu_count()}
```

## Getting Help

### Information to Include When Reporting Issues

When seeking help, please include:

1. **System Information**:
   ```python
   import pradel_jax as pj
   pj.system_diagnostics()
   ```

2. **Minimal Reproducible Example**:
   ```python
   # Simplest code that demonstrates the problem
   import pradel_jax as pj
   
   data = pj.load_data("small_test_dataset.csv")  # Share sample data if possible
   formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
   result = pj.fit_model(formula=formula, data=data)
   # Error occurs here
   ```

3. **Complete Error Trace**:
   ```python
   # Run with verbose error reporting
   import traceback
   
   try:
       result = pj.fit_model(formula=formula, data=data)
   except Exception as e:
       print(f"Error type: {type(e).__name__}")
       print(f"Error message: {str(e)}")
       print("Full traceback:")
       traceback.print_exc()
   ```

4. **What You've Tried**:
   - List the solutions you've already attempted
   - Include any partial successes or workarounds

### Where to Get Help

1. **GitHub Issues**: [https://github.com/chrischizinski/pradel-jax/issues](https://github.com/chrischizinski/pradel-jax/issues)
   - Bug reports and feature requests
   - Search existing issues first

2. **GitHub Discussions**: [https://github.com/chrischizinski/pradel-jax/discussions](https://github.com/chrischizinski/pradel-jax/discussions)
   - General questions and usage help
   - Community support

3. **Documentation**:
   - [User Guide](../user-guide/) - Comprehensive usage documentation
   - [API Reference](../api/) - Technical reference
   - [Examples](../tutorials/) - Practical examples

4. **Development Team**:
   - For sensitive issues or collaboration inquiries
   - Contact information in repository

### Self-Help Resources

Before reporting an issue, try:

1. **Search existing documentation** - Many common issues are covered
2. **Check GitHub issues** - Similar problems may have been solved
3. **Try the diagnostic tools** - Use the built-in diagnostic functions
4. **Start with a simple example** - Isolate the problem with minimal code
5. **Check dependencies** - Ensure all packages are up to date

---

**Related Documentation:**
- [Installation Guide](../user-guide/installation.md) - Detailed installation instructions
- [Performance Guide](performance.md) - Performance optimization and benchmarking
- [User Guide](../user-guide/) - Complete usage documentation