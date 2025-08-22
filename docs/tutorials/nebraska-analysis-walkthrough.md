# Nebraska Capture-Recapture Analysis Walkthrough

A comprehensive guide to running the Nebraska sample analysis script with Pradel-JAX.

## 📋 Overview

The `nebraska_sample_analysis.py` script demonstrates a complete capture-recapture analysis workflow using the Nebraska dataset. It randomly samples individuals, fits multiple Pradel models, and generates a comprehensive analysis package with MARK-compatible outputs.

## 🚀 Quick Start

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Navigate to project directory
   cd /path/to/pradel-jax
   
   # Activate environment (if using virtual environment)
   source pradel_env/bin/activate
   
   # Verify installation
   python -c "import pradel_jax as pj; print('✅ Pradel-JAX ready')"
   ```

2. **Data Requirements**:
   - Nebraska encounter history data: `data/encounter_histories_ne_clean.csv`
   - Data format: Individual encounter histories with Y-columns (Y2016-Y2024)
   - Covariates: gender, age, and other demographic variables

### Running the Analysis

```bash
# From the project root directory
cd examples/nebraska/
python nebraska_sample_analysis.py
```

## 📊 What the Script Does

### 1. Data Loading and Sampling

The script automatically:
- **Loads** the full Nebraska dataset (`encounter_histories_ne_clean.csv`)
- **Samples** 1000 random individuals (configurable)
- **Converts** to Pradel-JAX format using the GenericFormatAdapter
- **Reports** dataset dimensions and available covariates

```
🔬 Nebraska Capture-Recapture Analysis - Random Sample
============================================================
📂 Loading data from: data/encounter_histories_ne_clean.csv
   Full dataset shape: (50000, 25)
🎲 Randomly sampling 1000 rows...
   Sample shape: (1000, 25)
🔧 Converting to pradel-jax format...
   Data summary:
   - Number of individuals: 1000
   - Number of occasions: 9
   - Available covariates: ['gender', 'age', 'region', 'permit_type']
```

### 2. Intelligent Model Building

The script intelligently builds a model set based on available covariates:

**Base Model**: Always includes intercept-only
- φ(1) p(1) f(1) - Constant survival, detection, recruitment

**Covariate Effects**: Automatically detects and adds:
- **Gender effects**: φ(1 + gender) if gender covariate available
- **Age effects**: φ(1 + age) if age covariate available  
- **Additive models**: φ(1 + gender + age) if multiple covariates available

```
📊 Setting up Pradel models...
   Building model set with 4 survival models:
     1. φ~1
     2. φ~1 + gender
     3. φ~1 + age
     4. φ~1 + gender + age
   Available covariates: gender, age
   Created 4 model(s) for comparison:
   - Model 1: φ(1) p(1) f(1)
   - Model 2: φ(gender) p(1) f(1)
   - Model 3: φ(age) p(1) f(1)
   - Model 4: φ(gender + age) p(1) f(1)
```

### 3. Model Fitting with Advanced Optimization

Uses the hybrid optimization framework:
- **Strategy**: Automatic selection (L-BFGS-B → multi-start fallback)
- **Parallel processing**: Configurable workers (default: 1 for small datasets)
- **Error handling**: Robust convergence checking

```
⚡ Fitting model with automatic optimization...
   ✅ Model 1 converged (0.69s, hybrid_multistart_refined)
   ✅ Model 2 converged (0.13s, hybrid_multistart_refined)  
   ✅ Model 3 converged (0.07s, hybrid_multistart)
   ✅ Model 4 converged (0.20s, hybrid_multistart_refined)
```

### 4. Results Summary and Model Selection

Displays comprehensive results with AIC-based ranking:

```
🎯 Model Results (4 models fitted)
============================================================
✅ 4 model(s) converged successfully:

🥇 φ(gender) p(1) f(1)
   Log-likelihood: -1487.155
   AIC: 2982.310
   Parameters: 4
   Strategy: hybrid_multistart_refined
   Lambda (growth rate): 0.9294

🥈 φ(gender + age) p(1) f(1)
   Log-likelihood: -1487.049
   AIC: 2984.098
   Parameters: 5
   Strategy: hybrid_multistart_refined
   Lambda (growth rate): 0.5379

🥉 φ(1) p(1) f(1)
   Log-likelihood: -1505.781
   AIC: 3017.562
   Parameters: 3
   Strategy: hybrid_multistart_refined
   Lambda (growth rate): 0.6110
```

### 5. Comprehensive Export Package

Generates three publication-ready output files:

#### A. Full Results CSV (`nebraska_full_results_1000ind_TIMESTAMP.csv`)
- **40+ columns** of MARK-compatible statistical data
- **All parameters** with estimates and standard errors
- **Model diagnostics**: AIC, deviance, log-likelihood
- **Population metrics**: Lambda statistics, growth rates
- **Metadata**: Random seeds, data hashes, optimization details

```csv
model_name,converged,log_likelihood,aic,n_parameters,lambda_mean,phi_(Intercept),phi_gender,strategy_used,fit_time
φ(gender) p(1) f(1),True,-1487.155,2982.310,4,0.929,1.836,-1.708,hybrid_multistart_refined,0.129
```

#### B. Model Comparison Table (`nebraska_model_comparison_1000ind_TIMESTAMP.csv`)
- **AIC ranking** with delta-AIC values
- **AIC weights** and evidence ratios
- **Model support** indicators (substantial/some/little)
- **Publication-ready** format for tables

```csv
model_name,aic,delta_aic,aic_weight,evidence_ratio,substantial_support
φ(gender) p(1) f(1),2982.310,0.0,0.710,1.0,True
φ(gender + age) p(1) f(1),2984.098,1.787,0.290,2.44,True
```

#### C. Parameter Summary (`nebraska_parameters_1000ind_TIMESTAMP.csv`)
- **Simplified view** for quick reference
- **Key statistics** only (AIC, weights, main parameters)
- **Population growth rates** for ecological interpretation

## 📈 Interpreting Results

### Model Selection Metrics

**AIC Weights**: Probability that each model is the best
- Values sum to 1.0 across all models
- Higher weight = stronger support

**Evidence Ratios**: How much better the best model is
- Best model always has ratio = 1.0  
- Ratio of 2.44 means best model is 2.44× more likely to be correct

**Support Categories**:
- **Substantial**: ΔAIC ≤ 2 (strong evidence)
- **Some**: 2 < ΔAIC ≤ 7 (moderate evidence) 
- **Little**: ΔAIC > 7 (weak evidence)

### Population Dynamics

**Lambda (λ)**: Population growth rate
- **λ < 0.95**: 🔻 **DECLINING** population (-5% or more per year)
- **0.95 ≤ λ ≤ 1.05**: ➡️ **STABLE** population (±5% per year)
- **λ > 1.05**: 🔺 **INCREASING** population (+5% or more per year)

### Parameter Interpretation

**Survival (φ) Parameters**:
- **Intercept**: Baseline survival probability (logit scale)
- **Gender effect**: Difference between male/female survival
- **Age effect**: Change in survival per unit age increase

**Detection (p) Parameters**:
- Usually kept constant (~1) for simplicity
- Can be modeled with covariates if detection varies

## 📁 Output Files Structure

After successful analysis, you'll have:

```
examples/nebraska/
├── nebraska_full_results_1000ind_20250820_143022.csv      # Complete statistical data
├── nebraska_model_comparison_1000ind_20250820_143022.csv  # Model selection table  
└── nebraska_parameters_1000ind_20250820_143022.csv        # Parameter summary
```

## 🔧 Customization Options

### Modify Sample Size

```python
# In main() function, line ~40
sample_size = min(2000, len(full_data))  # Change from 1000 to 2000
```

### Add Different Covariates

```python
# In main() function, around line ~85
if 'region' in main_covariates:
    phi_formulas.append("~1 + region")
    covariate_effects.append("region")
```

### Change Model Complexity

```python
# Add interaction effects
if len(covariate_effects) >= 2:
    interaction_formula = f"~1 + {covariate_effects[0]} * {covariate_effects[1]}"
    phi_formulas.append(interaction_formula)
```

### Adjust Processing

```python
# Increase parallel workers for larger datasets
results = fit_models_parallel(
    model_specs=model_specs,
    data_context=data_context,
    n_workers=4  # Use 4 CPU cores
)
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Data File Not Found
```
❌ Error: Data file not found: data/encounter_histories_ne_clean.csv
```
**Solution**: Ensure you're running from the correct directory and the data file exists
```bash
# Check current directory
pwd
# Should be: /path/to/pradel-jax/examples/nebraska/

# Check for data file
ls ../../data/encounter_histories_ne_clean.csv
```

#### 2. No Models Converged
```
❌ No models converged successfully
```
**Solutions**:
- **Reduce sample size**: Try 100-500 individuals first
- **Simplify models**: Start with intercept-only model φ(1) p(1) f(1)
- **Check data quality**: Look for missing values or formatting issues

#### 3. Export Package Generation Failed
```
⚠️  Export package generation failed: "['delta_aic', 'aic_weight'] not in index"
```
**Solution**: This has been fixed in the latest version. The error occurred because model selection metrics weren't available before the comparison table was created.

#### 4. Memory Issues with Large Samples
```
MemoryError: Unable to allocate array
```
**Solutions**:
- **Reduce sample size**: Use 1000-5000 individuals maximum
- **Use single worker**: Set `n_workers=1` in `fit_models_parallel()`
- **Monitor RAM usage**: Large encounter history matrices require significant memory

#### 4. Optimization Failures
```
Optimization failed to converge
```
**Solutions**:
- **Check parameter bounds**: Ensure reasonable initial values
- **Try different strategies**: The hybrid optimizer usually handles this automatically
- **Simplify model**: Remove complex covariate interactions

### Debug Mode

Enable detailed output for troubleshooting:

```python
# Add at top of script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or add print statements for specific checks
print(f"Data shape after loading: {data_context.encounter_histories.shape}")
print(f"Available covariates: {list(data_context.covariates.keys())}")
```

## 🎯 Next Steps

### Production Analysis
- Run on full dataset (remove sampling)
- Add more sophisticated model structures
- Include temporal variation in parameters

### RMark Comparison  
- Export results to RMark format
- Compare parameter estimates
- Validate population growth rate calculations

### Publication Preparation
- Use model comparison tables directly in manuscripts
- Create diagnostic plots from parameter estimates  
- Report AIC weights and evidence ratios

## 📚 Additional Resources

- **Main Documentation**: `docs/user-guide/`
- **API Reference**: `docs/api/`
- **Performance Tips**: `docs/tutorials/performance-optimization.md`
- **MARK Compatibility**: `docs/tutorials/mark-integration.md`

---

*This walkthrough covers the complete Nebraska analysis workflow. For questions or issues, refer to the main documentation or create an issue on GitHub.*