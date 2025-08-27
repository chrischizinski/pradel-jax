# Complete Analysis Workflow with Pradel-JAX

A comprehensive tutorial walking through a real-world capture-recapture analysis from start to finish, demonstrating all major features of the Pradel-JAX framework.

## 🎯 What You'll Learn

By the end of this tutorial, you'll know how to:

1. **Load and validate** multi-format capture-recapture data
2. **Specify complex models** with time-varying covariates and interactions
3. **Optimize model fitting** with intelligent strategy selection
4. **Perform statistical inference** with confidence intervals and hypothesis testing
5. **Compare multiple models** using AIC/BIC and cross-validation
6. **Export results** for publication and further analysis
7. **Handle common challenges** in real-world data

## 📊 Example Dataset: Extended Dipper Study

We'll analyze an extended version of the classic European Dipper dataset with:

- **294 individuals** marked and recaptured over **7 occasions** (1981-1987)
- **Time-constant covariates**: sex (Male/Female)
- **Time-varying covariates**: age (Juvenile/Adult), environmental conditions
- **Real biological questions**: Does survival vary by sex and age? How does recruitment change over time?

## 🚀 Step 1: Environment Setup and Data Loading

### Install and Import

```python
# Ensure you have the latest version
# pip install --upgrade pradel-jax

import pradel_jax as pj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging to see what's happening
pj.configure(logging_level="INFO", console_output=True)

print(f"Pradel-JAX version: {pj.__version__}")
```

### Load and Examine Data

```python
# Load the example dataset (included with Pradel-JAX)
data_path = "data/extended_dipper_dataset.csv"
data = pj.load_data(data_path)

print(f"📊 Dataset Overview:")
print(f"  Individuals: {data.n_individuals}")
print(f"  Occasions: {data.n_occasions} (years {data.occasion_names})")
print(f"  Total captures: {data.total_captures}")
print(f"  Capture rate: {data.capture_rate:.3f}")

print(f"\n🔍 Available Covariates:")
for name, values in data.covariates.items():
    if hasattr(values, 'unique'):
        unique_vals = values.unique()
        print(f"  {name}: {unique_vals} (n = {len(unique_vals)})")
    else:
        print(f"  {name}: {values.shape} (time-varying)")
```

**Expected Output:**
```
📊 Dataset Overview:
  Individuals: 294
  Occasions: 7 (years ['1981', '1982', '1983', '1984', '1985', '1986', '1987'])
  Total captures: 982
  Capture rate: 0.477

🔍 Available Covariates:
  sex: ['M' 'F'] (n = 2)
  age: (294, 7) (time-varying)
  flood: [0 1] (n = 2)
  winter_severity: (294, 7) (time-varying)
```

### Data Quality Assessment

```python
# Comprehensive data validation
validation_result = data.validate()

print(f"✅ Data Validation:")
print(f"  Valid: {validation_result.is_valid}")
print(f"  Warnings: {len(validation_result.warnings)}")
print(f"  Errors: {len(validation_result.errors)}")

if validation_result.warnings:
    print("\n⚠️ Warnings:")
    for warning in validation_result.warnings:
        print(f"    - {warning}")

# Examine capture patterns
capture_summary = data.get_capture_summary()
print(f"\n📈 Capture Patterns:")
print(f"  Never captured: {capture_summary['never_captured']}")
print(f"  Captured once: {capture_summary['captured_once']}")  
print(f"  Captured 2-3 times: {capture_summary['captured_2_3']}")
print(f"  Captured 4+ times: {capture_summary['captured_4_plus']}")

# Visualize capture histories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Capture frequency by occasion
occasion_captures = data.capture_matrix.sum(axis=0)
ax1.bar(range(1, data.n_occasions + 1), occasion_captures)
ax1.set_xlabel('Occasion')
ax1.set_ylabel('Number of Captures')
ax1.set_title('Captures by Occasion')

# Individual capture frequencies
individual_captures = data.capture_matrix.sum(axis=1)
ax2.hist(individual_captures, bins=range(8), alpha=0.7)
ax2.set_xlabel('Number of Times Captured')
ax2.set_ylabel('Number of Individuals')
ax2.set_title('Distribution of Individual Capture Frequencies')

plt.tight_layout()
plt.show()
```

## 🎯 Step 2: Exploratory Data Analysis

### Covariate Exploration

```python
# Examine relationships between covariates and capture patterns
analysis_df = data.to_analysis_dataframe()

print("📊 Covariate Analysis:")

# Sex distribution
sex_summary = analysis_df.groupby('sex').agg({
    'total_captures': ['count', 'mean', 'std'],
    'first_capture': 'mean',
    'last_capture': 'mean'
}).round(3)
print(f"\nSex Summary:")
print(sex_summary)

# Age patterns (if time-varying age is available)
if 'age' in data.time_varying_covariates:
    age_data = data.get_time_varying_covariate('age')
    
    # Look at age transitions
    age_transitions = pd.DataFrame({
        'occasion': range(1, data.n_occasions + 1),
        'prop_adult': (age_data == 'Adult').mean(axis=0)
    })
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(age_transitions['occasion'], age_transitions['prop_adult'], 'o-')
    plt.xlabel('Occasion')
    plt.ylabel('Proportion Adult')
    plt.title('Age Structure Over Time')
    
    # Capture probability by age and sex
    plt.subplot(1, 2, 2)
    capture_by_demo = analysis_df.groupby(['sex', 'age_class'])['capture_probability'].mean()
    capture_by_demo.plot(kind='bar')
    plt.xlabel('Sex and Age')
    plt.ylabel('Capture Probability')
    plt.title('Capture Probability by Demographics')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
```

### Preliminary Model Assessment

```python
# Assess model identifiability before fitting
def assess_model_complexity(data, formula_spec):
    """Check if model is identifiable given the data."""
    design_info = pj.get_design_matrix_info(formula_spec, data)
    
    n_observations = data.total_captures
    n_parameters = sum(info.n_parameters for info in design_info.values())
    
    print(f"Model Complexity Assessment:")
    print(f"  Parameters: {n_parameters}")
    print(f"  Observations: {n_observations}")
    print(f"  Obs/Param ratio: {n_observations/n_parameters:.1f}")
    
    if n_parameters > n_observations / 10:
        print("  ⚠️ Model may be overparameterized")
        return False
    elif n_parameters > n_observations / 20:
        print("  ⚠️ Model is complex for available data")
        return True
    else:
        print("  ✅ Model complexity appropriate")
        return True

# Test different model complexities
models_to_test = {
    'Null': pj.create_formula_spec(phi="~1", p="~1", f="~1"),
    'Sex_Effects': pj.create_formula_spec(phi="~sex", p="~sex", f="~1"),
    'Time_Varying': pj.create_formula_spec(phi="~sex + age", p="~sex", f="~flood"),
    'Full_Model': pj.create_formula_spec(phi="~sex * age", p="~sex + flood", f="~flood + winter_severity")
}

for name, formula in models_to_test.items():
    print(f"\n{name} Model:")
    is_identifiable = assess_model_complexity(data, formula)
    models_to_test[name] = (formula, is_identifiable)
```

## 🔧 Step 3: Model Specification and Fitting

### Model 1: Null Model (Baseline)

```python
# Start with the simplest model
print("🎯 Fitting Null Model...")

null_formula = pj.create_formula_spec(
    phi="~1",    # Constant survival
    p="~1",      # Constant detection  
    f="~1"       # Constant recruitment
)

# Fit with automatic optimization strategy selection
null_result = pj.fit_model(
    model=pj.PradelModel(),
    formula=null_formula,
    data=data,
    compute_se=True,
    confidence_intervals=True
)

print(f"Results:")
print(f"  Success: {null_result.success}")
print(f"  Strategy: {null_result.strategy_used}")
print(f"  Log-likelihood: {null_result.log_likelihood:.3f}")
print(f"  AIC: {null_result.aic:.3f}")

if null_result.success:
    # Display results in natural scale
    params = null_result.parameter_estimates
    phi_prob = pj.logit_inverse(params['phi_intercept'])
    p_prob = pj.logit_inverse(params['p_intercept'])
    f_rate = np.exp(params['f_intercept'])
    
    print(f"\nParameter Estimates (Natural Scale):")
    print(f"  Survival probability: {phi_prob:.3f}")
    print(f"  Detection probability: {p_prob:.3f}")
    print(f"  Recruitment rate: {f_rate:.3f}")
    
    # Get confidence intervals
    ci = null_result.confidence_intervals
    print(f"\n95% Confidence Intervals:")
    for param, (lower, upper) in ci.items():
        print(f"  {param}: [{lower:.3f}, {upper:.3f}]")
```

### Model 2: Sex Effects Model

```python
print("\n🎯 Fitting Sex Effects Model...")

sex_formula = pj.create_formula_spec(
    phi="~sex",      # Survival varies by sex
    p="~sex",        # Detection varies by sex
    f="~1"           # Constant recruitment
)

sex_result = pj.fit_model(
    model=pj.PradelModel(),
    formula=sex_formula,
    data=data,
    compute_se=True,
    confidence_intervals=True,
    strategy="multi_start"  # Use robust optimization
)

print(f"Results:")
print(f"  Success: {sex_result.success}")
print(f"  Strategy: {sex_result.strategy_used}")
print(f"  Log-likelihood: {sex_result.log_likelihood:.3f}")
print(f"  AIC: {sex_result.aic:.3f}")
print(f"  Δ AIC vs Null: {sex_result.aic - null_result.aic:.3f}")

if sex_result.success:
    # Interpret sex effects
    params = sex_result.parameter_estimates
    
    # Male survival (reference category)
    phi_male = pj.logit_inverse(params['phi_intercept'])
    p_male = pj.logit_inverse(params['p_intercept'])
    
    # Female survival (reference + sex effect)
    phi_female = pj.logit_inverse(params['phi_intercept'] + params['phi_sex'])
    p_female = pj.logit_inverse(params['p_intercept'] + params['p_sex'])
    
    print(f"\nSurvival by Sex:")
    print(f"  Male: {phi_male:.3f}")
    print(f"  Female: {phi_female:.3f}")
    print(f"  Difference: {phi_female - phi_male:.3f}")
    
    print(f"\nDetection by Sex:")
    print(f"  Male: {p_male:.3f}")
    print(f"  Female: {p_female:.3f}")
    print(f"  Difference: {p_female - p_male:.3f}")
    
    # Statistical significance
    se = sex_result.standard_errors
    z_score_phi = params['phi_sex'] / se['phi_sex']
    z_score_p = params['p_sex'] / se['p_sex']
    
    print(f"\nSignificance Tests (H0: no sex effect):")
    print(f"  Survival sex effect: z = {z_score_phi:.3f}, p = {2 * (1 - stats.norm.cdf(abs(z_score_phi))):.3f}")
    print(f"  Detection sex effect: z = {z_score_p:.3f}, p = {2 * (1 - stats.norm.cdf(abs(z_score_p))):.3f}")
```

### Model 3: Time-Varying Covariates Model

```python
print("\n🎯 Fitting Time-Varying Model...")

# Model with time-varying age and environmental covariates
tv_formula = pj.create_formula_spec(
    phi="~sex + age",           # Survival by sex and age
    p="~sex",                   # Detection by sex
    f="~flood + winter_severity" # Recruitment by environmental conditions
)

tv_result = pj.fit_model(
    model=pj.PradelModel(),
    formula=tv_formula,
    data=data,
    compute_se=True,
    confidence_intervals=True,
    optimization_config=pj.OptimizationConfig(
        strategy="multi_start",
        max_iterations=2000,
        n_starts=5
    )
)

print(f"Results:")
print(f"  Success: {tv_result.success}")
print(f"  Iterations: {tv_result.n_iterations}")
print(f"  Log-likelihood: {tv_result.log_likelihood:.3f}")
print(f"  AIC: {tv_result.aic:.3f}")
print(f"  Δ AIC vs Sex Model: {tv_result.aic - sex_result.aic:.3f}")

if tv_result.success:
    # Create publication-ready parameter table
    param_table = tv_result.get_parameter_table(confidence_level=0.95)
    print("\n📊 Parameter Estimates:")
    print(param_table)
    
    # Calculate derived quantities with uncertainty
    # Example: Adult female survival probability
    phi_adult_female = tv_result.predict_parameter(
        parameter='phi',
        covariates={'sex': 'F', 'age': 'Adult'},
        confidence_intervals=True
    )
    print(f"\nAdult Female Survival: {phi_adult_female['estimate']:.3f} "
          f"[{phi_adult_female['ci_lower']:.3f}, {phi_adult_female['ci_upper']:.3f}]")
```

### Model 4: Complex Interactions Model

```python
print("\n🎯 Fitting Complex Model with Interactions...")

complex_formula = pj.create_formula_spec(
    phi="~sex * age",                    # Sex-age interaction for survival
    p="~sex + flood",                    # Detection by sex and flood conditions
    f="~flood + winter_severity + I(winter_severity**2)"  # Non-linear recruitment
)

# Use conservative settings for complex model
complex_result = pj.fit_model(
    model=pj.PradelModel(),
    formula=complex_formula,
    data=data,
    compute_se=True,
    confidence_intervals=True,
    optimization_config=pj.OptimizationConfig(
        strategy="multi_start", 
        max_iterations=3000,
        tolerance=1e-6,
        n_starts=10
    )
)

print(f"Results:")
print(f"  Success: {complex_result.success}")
if complex_result.success:
    print(f"  Log-likelihood: {complex_result.log_likelihood:.3f}")
    print(f"  AIC: {complex_result.aic:.3f}")
    print(f"  Parameters: {complex_result.n_parameters}")
else:
    print(f"  Failure reason: {complex_result.optimization_message}")
    print("  ℹ️ Complex model may be overparameterized for this dataset")
```

## 📈 Step 4: Model Comparison and Selection

### Comprehensive Model Comparison

```python
# Collect successful models
successful_models = {}
for name, result in [('Null', null_result), ('Sex', sex_result), ('TimeVarying', tv_result)]:
    if result.success:
        successful_models[name] = result

if complex_result.success:
    successful_models['Complex'] = complex_result

print(f"📊 Model Comparison ({len(successful_models)} models):")

# AIC/BIC comparison
comparison = pj.compare_models(successful_models, criterion='aic')
print("\nAIC-based Model Ranking:")
print(comparison[['model', 'aic', 'delta_aic', 'aic_weight', 'evidence_ratio']])

# Statistical support classification
for _, row in comparison.iterrows():
    if row['delta_aic'] <= 2:
        support = "Strong"
    elif row['delta_aic'] <= 7:
        support = "Moderate" 
    else:
        support = "Weak"
    print(f"  {row['model']}: {support} support (Δ AIC = {row['delta_aic']:.1f})")
```

### Cross-Validation for Robust Model Selection

```python
print("\n🔄 Cross-Validation Analysis...")

# Perform k-fold cross-validation
cv_results = pj.cross_validate_models(
    models=successful_models,
    data=data,
    k_folds=5,
    stratify_by='sex',  # Ensure balanced folds
    metrics=['log_likelihood', 'aic', 'prediction_accuracy'],
    random_seed=42
)

print("Cross-Validation Results:")
cv_summary = cv_results.groupby('model').agg({
    'cv_log_likelihood': ['mean', 'std'],
    'cv_aic': ['mean', 'std'],  
    'prediction_accuracy': ['mean', 'std']
}).round(3)

print(cv_summary)

# Plot CV results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['cv_log_likelihood', 'cv_aic', 'prediction_accuracy']
titles = ['Cross-Validation Log-Likelihood', 'Cross-Validation AIC', 'Prediction Accuracy']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    sns.boxplot(data=cv_results, x='model', y=metric, ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel('Model')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Model Selection Decision

```python
# Select best model based on multiple criteria
print("\n🏆 Model Selection Decision:")

# Get the top-ranking model
best_model_name = comparison.iloc[0]['model']
best_model = successful_models[best_model_name]

print(f"Selected Model: {best_model_name}")
print(f"  AIC: {best_model.aic:.3f}")
print(f"  Cross-validation score: {cv_summary.loc[best_model_name, ('cv_log_likelihood', 'mean')]:.3f}")
print(f"  Model weight: {comparison.iloc[0]['aic_weight']:.3f}")

# If top models are close, provide guidance
top_models = comparison[comparison['delta_aic'] <= 2]
if len(top_models) > 1:
    print(f"\n⚠️ Note: {len(top_models)} models have substantial support (Δ AIC ≤ 2)")
    print("Consider:")
    print("  - Model averaging for robust inference")
    print("  - Biological interpretability")
    print("  - Predictive performance for your specific goals")
```

## 📊 Step 5: Statistical Inference and Interpretation

### Comprehensive Results Analysis

```python
print("🔬 Statistical Inference for Selected Model")
selected_model = successful_models[best_model_name]

# Generate comprehensive results table
results_table = selected_model.get_parameter_table(
    confidence_level=0.95,
    include_significance=True,
    format='detailed'
)

print("\n📋 Complete Parameter Estimates:")
print(results_table)

# Effect sizes and biological interpretation
print("\n📈 Biological Interpretation:")

params = selected_model.parameter_estimates
se = selected_model.standard_errors

# Convert to natural scale with confidence intervals
for param_name, estimate in params.items():
    ci_lower, ci_upper = selected_model.confidence_intervals[param_name]
    
    if param_name.startswith('phi_'):
        # Survival parameters (logit scale)
        if param_name == 'phi_intercept':
            natural_est = pj.logit_inverse(estimate)
            natural_lower = pj.logit_inverse(ci_lower)
            natural_upper = pj.logit_inverse(ci_upper)
            print(f"  Baseline Survival: {natural_est:.3f} [{natural_lower:.3f}, {natural_upper:.3f}]")
        else:
            # For covariate effects, show effect on probability
            effect_size = pj.logit_inverse(params['phi_intercept'] + estimate) - pj.logit_inverse(params['phi_intercept'])
            print(f"  {param_name} effect: {effect_size:.3f} change in survival probability")
    
    elif param_name.startswith('p_'):
        # Detection parameters
        if param_name == 'p_intercept':
            natural_est = pj.logit_inverse(estimate)
            natural_lower = pj.logit_inverse(ci_lower)
            natural_upper = pj.logit_inverse(ci_upper)
            print(f"  Baseline Detection: {natural_est:.3f} [{natural_lower:.3f}, {natural_upper:.3f}]")
    
    elif param_name.startswith('f_'):
        # Recruitment parameters (log scale)
        if param_name == 'f_intercept':
            natural_est = np.exp(estimate)
            natural_lower = np.exp(ci_lower)
            natural_upper = np.exp(ci_upper)
            print(f"  Baseline Recruitment: {natural_est:.3f} [{natural_lower:.3f}, {natural_upper:.3f}]")
```

### Hypothesis Testing

```python
print("\n🧪 Hypothesis Testing:")

# Test specific biological hypotheses
hypotheses = [
    {
        'name': 'No sex difference in survival',
        'parameter': 'phi_sex',
        'null_value': 0,
        'alternative': 'two-sided'
    },
    {
        'name': 'Adult survival higher than juvenile',
        'parameter': 'phi_age',  # Assuming Adult is coded as 1
        'null_value': 0,
        'alternative': 'greater'
    },
    {
        'name': 'Flood events reduce recruitment',
        'parameter': 'f_flood',
        'null_value': 0,
        'alternative': 'less'
    }
]

from scipy import stats

for hypothesis in hypotheses:
    param_name = hypothesis['parameter']
    if param_name in params:
        estimate = params[param_name]
        standard_error = se[param_name]
        null_value = hypothesis['null_value']
        
        # Z-test
        z_score = (estimate - null_value) / standard_error
        
        if hypothesis['alternative'] == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        elif hypothesis['alternative'] == 'greater':
            p_value = 1 - stats.norm.cdf(z_score)
        elif hypothesis['alternative'] == 'less':
            p_value = stats.norm.cdf(z_score)
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"  {hypothesis['name']}:")
        print(f"    Estimate: {estimate:.4f} ± {standard_error:.4f}")
        print(f"    Z-score: {z_score:.3f}")
        print(f"    P-value: {p_value:.4f} {significance}")
        print(f"    Result: {'Reject' if p_value < 0.05 else 'Fail to reject'} null hypothesis")
        print()
```

### Bootstrap Confidence Intervals

```python
print("\n🔄 Bootstrap Confidence Intervals:")

# Compute bootstrap CIs for more robust inference
bootstrap_cis = selected_model.bootstrap_confidence_intervals(
    n_bootstrap=1000,
    confidence_level=0.95,
    method='bca',  # Bias-corrected and accelerated
    random_seed=42,
    parallel=True
)

print("Comparison of CI Methods:")
asymptotic_cis = selected_model.confidence_intervals

comparison_table = []
for param in params.keys():
    asymp_lower, asymp_upper = asymptotic_cis[param]
    boot_lower, boot_upper = bootstrap_cis[param]
    
    comparison_table.append({
        'Parameter': param,
        'Asymptotic_Lower': asymp_lower,
        'Asymptotic_Upper': asymp_upper,
        'Bootstrap_Lower': boot_lower,
        'Bootstrap_Upper': boot_upper,
        'Width_Diff': (boot_upper - boot_lower) - (asymp_upper - asymp_lower)
    })

comparison_df = pd.DataFrame(comparison_table)
print(comparison_df.round(4))
```

## 📈 Step 6: Model Diagnostics and Validation

### Residual Analysis

```python
print("\n🔍 Model Diagnostics:")

# Compute residuals and diagnostic statistics
diagnostics = selected_model.compute_diagnostics()

print("Goodness-of-fit Assessment:")
print(f"  Deviance: {diagnostics['deviance']:.3f}")
print(f"  Pearson χ²: {diagnostics['pearson_chi2']:.3f}")
print(f"  Overdispersion parameter: {diagnostics['overdispersion']:.3f}")

if diagnostics['overdispersion'] > 1.3:
    print("  ⚠️ Evidence of overdispersion - consider adding random effects")

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs fitted
axes[0,0].scatter(diagnostics['fitted_values'], diagnostics['residuals'], alpha=0.6)
axes[0,0].axhline(y=0, color='r', linestyle='--')
axes[0,0].set_xlabel('Fitted Values')
axes[0,0].set_ylabel('Residuals')
axes[0,0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(diagnostics['residuals'], dist="norm", plot=axes[0,1])
axes[0,1].set_title('Normal Q-Q Plot')

# Scale-location plot
standardized_residuals = np.sqrt(np.abs(diagnostics['residuals']))
axes[1,0].scatter(diagnostics['fitted_values'], standardized_residuals, alpha=0.6)
axes[1,0].set_xlabel('Fitted Values')
axes[1,0].set_ylabel('√|Residuals|')
axes[1,0].set_title('Scale-Location Plot')

# Cook's distance
axes[1,1].bar(range(len(diagnostics['cooks_distance'])), diagnostics['cooks_distance'])
axes[1,1].axhline(y=4/len(diagnostics['cooks_distance']), color='r', linestyle='--')
axes[1,1].set_xlabel('Observation')
axes[1,1].set_ylabel("Cook's Distance")
axes[1,1].set_title('Influence Plot')

plt.tight_layout()
plt.show()

# Flag influential observations
influential_threshold = 4 / data.n_individuals
influential_obs = np.where(diagnostics['cooks_distance'] > influential_threshold)[0]

if len(influential_obs) > 0:
    print(f"\n⚠️ Influential Observations (n = {len(influential_obs)}):")
    for obs_idx in influential_obs[:10]:  # Show first 10
        print(f"  Individual {obs_idx}: Cook's D = {diagnostics['cooks_distance'][obs_idx]:.4f}")
    
    if len(influential_obs) > 10:
        print(f"  ... and {len(influential_obs) - 10} more")
```

### Model Adequacy Tests

```python
print("\n🧪 Model Adequacy Tests:")

# Bootstrap goodness-of-fit test
bootstrap_gof = pj.bootstrap_goodness_of_fit(
    model=selected_model,
    data=data,
    n_bootstrap=500,
    test_statistics=['deviance', 'pearson_chi2', 'freeman_tukey'],
    random_seed=42
)

print("Bootstrap Goodness-of-Fit Tests:")
for test_name, result in bootstrap_gof.items():
    observed = result['observed']
    p_value = result['p_value']
    
    print(f"  {test_name}:")
    print(f"    Observed: {observed:.3f}")
    print(f"    P-value: {p_value:.3f}")
    print(f"    Interpretation: {'Poor fit' if p_value < 0.05 else 'Adequate fit'}")
    print()

# Component goodness-of-fit (if available)
if hasattr(selected_model, 'component_goodness_of_fit'):
    component_gof = selected_model.component_goodness_of_fit(data)
    
    print("Component-wise Goodness-of-Fit:")
    print(f"  Survival component: χ² = {component_gof['survival']['chi2']:.3f}, p = {component_gof['survival']['p_value']:.3f}")
    print(f"  Detection component: χ² = {component_gof['detection']['chi2']:.3f}, p = {component_gof['detection']['p_value']:.3f}")
    print(f"  Recruitment component: χ² = {component_gof['recruitment']['chi2']:.3f}, p = {component_gof['recruitment']['p_value']:.3f}")
```

### Sensitivity Analysis

```python
print("\n⚖️ Sensitivity Analysis:")

# Test sensitivity to outliers by removing influential observations
if len(influential_obs) > 0:
    # Create dataset without most influential observations
    robust_indices = np.setdiff1d(range(data.n_individuals), influential_obs[:5])
    robust_data = data.subset(robust_indices)
    
    print(f"Refitting model without {len(influential_obs[:5])} most influential observations...")
    
    robust_result = pj.fit_model(
        model=pj.PradelModel(),
        formula=selected_model.formula_spec,
        data=robust_data,
        compute_se=True
    )
    
    if robust_result.success:
        print("\nParameter Comparison (Full vs Robust):")
        for param in params.keys():
            full_est = params[param]
            robust_est = robust_result.parameter_estimates[param]
            relative_change = abs(robust_est - full_est) / abs(full_est) * 100
            
            print(f"  {param}:")
            print(f"    Full: {full_est:.4f}")
            print(f"    Robust: {robust_est:.4f}")
            print(f"    Change: {relative_change:.1f}%")
            
            if relative_change > 10:
                print(f"    ⚠️ Substantial change detected")
            print()

# Test sensitivity to prior assumptions (if applicable)
# This would involve refitting with different parameter bounds or initial values
```

## 💾 Step 7: Results Export and Reporting

### Publication-Ready Tables

```python
print("\n📊 Creating Publication-Ready Outputs:")

# Create comprehensive results table
publication_table = selected_model.create_publication_table(
    include_se=True,
    include_ci=True,
    include_significance=True,
    format='latex',  # Also supports 'markdown', 'html'
    caption="Parameter estimates for the selected Pradel model of European Dipper survival, detection, and recruitment."
)

print("LaTeX Table Code:")
print(publication_table)

# Create model comparison table
comparison_table = pj.create_model_comparison_table(
    models=successful_models,
    metrics=['aic', 'bic', 'log_likelihood', 'n_parameters'],
    format='markdown'
)

print("\nModel Comparison Table (Markdown):")
print(comparison_table)
```

### Export Data and Results

```python
# Export results to various formats
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# CSV exports
print(f"📁 Exporting Results (timestamp: {timestamp}):")

# Parameter estimates with uncertainty
param_df = selected_model.get_parameter_dataframe(include_inference=True)
param_file = f"dipper_parameters_{timestamp}.csv"
param_df.to_csv(param_file, index=False)
print(f"  ✅ Parameters: {param_file}")

# Model comparison
comparison_file = f"model_comparison_{timestamp}.csv"
comparison.to_csv(comparison_file, index=False)
print(f"  ✅ Model comparison: {comparison_file}")

# Fitted values and residuals
diagnostics_df = pd.DataFrame({
    'individual_id': range(data.n_individuals),
    'fitted_survival': diagnostics['fitted_survival'],
    'fitted_detection': diagnostics['fitted_detection'],
    'residuals': diagnostics['residuals'],
    'cooks_distance': diagnostics['cooks_distance']
})
diagnostics_file = f"model_diagnostics_{timestamp}.csv"
diagnostics_df.to_csv(diagnostics_file, index=False)
print(f"  ✅ Diagnostics: {diagnostics_file}")

# Complete results object (for reproducibility)
results_file = f"complete_results_{timestamp}.pkl"
selected_model.save(results_file)
print(f"  ✅ Complete results: {results_file}")

# Analysis metadata
metadata = {
    'analysis_date': pd.Timestamp.now().isoformat(),
    'pradel_jax_version': pj.__version__,
    'dataset': 'Extended European Dipper',
    'n_individuals': data.n_individuals,
    'n_occasions': data.n_occasions,
    'selected_model': best_model_name,
    'selected_formula': str(selected_model.formula_spec),
    'convergence': selected_model.success,
    'optimization_strategy': selected_model.strategy_used,
    'aic': selected_model.aic,
    'log_likelihood': selected_model.log_likelihood,
    'n_parameters': selected_model.n_parameters
}

metadata_file = f"analysis_metadata_{timestamp}.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ Metadata: {metadata_file}")
```

### Create Summary Report

```python
# Generate automatic summary report
report = pj.create_analysis_report(
    model_results=selected_model,
    data_context=data,
    model_comparison=comparison,
    diagnostics=diagnostics,
    title="European Dipper Capture-Recapture Analysis",
    author="Your Name",
    date=pd.Timestamp.now().strftime("%Y-%m-%d")
)

report_file = f"dipper_analysis_report_{timestamp}.html"
with open(report_file, 'w') as f:
    f.write(report)

print(f"  ✅ Summary report: {report_file}")
print(f"\n🎉 Analysis Complete! All results exported.")
```

## 🎯 Step 8: Advanced Topics and Extensions

### Prediction and Forecasting

```python
print("\n🔮 Population Predictions:")

# Use fitted model to predict future population dynamics
future_scenarios = {
    'baseline': {'flood': 0, 'winter_severity': 0},  # Average conditions
    'harsh': {'flood': 1, 'winter_severity': 1},     # Harsh conditions  
    'mild': {'flood': 0, 'winter_severity': -1}      # Mild conditions
}

predictions = {}
for scenario_name, conditions in future_scenarios.items():
    pred = selected_model.predict_population(
        initial_population=100,
        n_years=10,
        covariates=conditions,
        include_uncertainty=True,
        n_simulations=1000
    )
    predictions[scenario_name] = pred

# Plot predictions
plt.figure(figsize=(12, 8))

for scenario_name, pred in predictions.items():
    years = range(len(pred['mean']))
    
    plt.plot(years, pred['mean'], label=f"{scenario_name.title()} Scenario", linewidth=2)
    plt.fill_between(years, pred['lower'], pred['upper'], alpha=0.2)

plt.xlabel('Years into Future')
plt.ylabel('Population Size')
plt.title('Population Projections under Different Scenarios')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate extinction probabilities
for scenario_name, pred in predictions.items():
    extinction_prob = (pred['simulations'][:, -1] < 10).mean()  # <10 individuals = quasi-extinction
    print(f"{scenario_name.title()} scenario: {extinction_prob:.1%} quasi-extinction probability")
```

### Model Averaging

```python
print("\n⚖️ Model Averaging for Robust Inference:")

# If multiple models have substantial support, use model averaging
supported_models = comparison[comparison['delta_aic'] <= 7]

if len(supported_models) > 1:
    print(f"Using model averaging across {len(supported_models)} supported models")
    
    # Weighted average of parameter estimates
    averaged_params = pj.model_average_parameters(
        models={name: successful_models[name] for name in supported_models['model']},
        weights=supported_models['aic_weight'].values
    )
    
    print("\nModel-Averaged Parameter Estimates:")
    for param, result in averaged_params.items():
        print(f"  {param}: {result['estimate']:.4f} ± {result['se']:.4f}")
        print(f"    95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    
    # Model-averaged predictions
    avg_predictions = pj.model_average_predictions(
        models={name: successful_models[name] for name in supported_models['model']},
        weights=supported_models['aic_weight'].values,
        prediction_scenarios=future_scenarios
    )
    
    print("\nModel-averaged predictions incorporate model uncertainty")
else:
    print("Single model has overwhelming support - model averaging not needed")
```

### Power Analysis and Study Design

```python
print("\n📊 Retrospective Power Analysis:")

# Assess statistical power for detecting effects
power_analysis = pj.retrospective_power_analysis(
    model_result=selected_model,
    data_context=data,
    effects_to_test=['phi_sex', 'p_sex', 'f_flood'],
    alpha=0.05,
    n_simulations=1000
)

print("Statistical Power for Detected Effects:")
for effect, power in power_analysis.items():
    print(f"  {effect}: {power:.3f}")
    if power < 0.8:
        print(f"    ⚠️ Low power - effect may be difficult to detect")

# Prospective power analysis for future studies
print("\nPower Analysis for Future Study Design:")

future_power = pj.prospective_power_analysis(
    baseline_model=selected_model,
    effect_sizes={'phi_sex': 0.1, 'p_sex': 0.05},  # Hypothetical effect sizes
    sample_sizes=[100, 200, 300, 400, 500],
    n_occasions=[5, 7, 10],
    alpha=0.05,
    n_simulations=500
)

# Plot power curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for n_occ in [5, 7, 10]:
    power_curve = future_power[future_power['n_occasions'] == n_occ]
    axes[0].plot(power_curve['sample_size'], power_curve['power_phi_sex'], 
                label=f'{n_occ} occasions', marker='o')
    axes[1].plot(power_curve['sample_size'], power_curve['power_p_sex'], 
                label=f'{n_occ} occasions', marker='s')

axes[0].axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8')
axes[0].set_xlabel('Sample Size')
axes[0].set_ylabel('Power')
axes[0].set_title('Power to Detect Survival Sex Effect')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].axhline(y=0.8, color='r', linestyle='--', label='Power = 0.8')
axes[1].set_xlabel('Sample Size')
axes[1].set_ylabel('Power')
axes[1].set_title('Power to Detect Detection Sex Effect')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 🎓 Summary and Best Practices

### Key Takeaways from This Analysis

```python
print("🎯 Analysis Summary:")
print(f"  📊 Data: {data.n_individuals} individuals, {data.n_occasions} occasions")
print(f"  🏆 Best model: {best_model_name}")
print(f"  📈 Model fit: AIC = {selected_model.aic:.1f}, Log-likelihood = {selected_model.log_likelihood:.1f}")
print(f"  🔬 Key findings:")

# Extract key biological findings
key_findings = []

if 'phi_sex' in params and abs(params['phi_sex'] / se['phi_sex']) > 1.96:
    effect_direction = "higher" if params['phi_sex'] > 0 else "lower"
    key_findings.append(f"Significant sex effect on survival (females {effect_direction} than males)")

if 'f_flood' in params and abs(params['f_flood'] / se['f_flood']) > 1.96:
    effect_direction = "reduces" if params['f_flood'] < 0 else "increases"
    key_findings.append(f"Flood events significantly {effect_direction} recruitment")

for finding in key_findings:
    print(f"    • {finding}")

print(f"  ✅ Model validation: {'Passed' if diagnostics['overdispersion'] < 1.3 else 'Concerns about overdispersion'}")
```

### Best Practices Demonstrated

```python
print("\n📚 Best Practices Demonstrated in This Analysis:")

best_practices = [
    "✅ Comprehensive data validation and quality assessment",
    "✅ Exploratory data analysis to understand patterns",
    "✅ Multiple model comparison with biological justification", 
    "✅ Robust optimization with multiple strategies",
    "✅ Statistical inference with confidence intervals and hypothesis testing",
    "✅ Model diagnostics and adequacy testing",
    "✅ Bootstrap methods for robust uncertainty quantification",
    "✅ Sensitivity analysis for model robustness",
    "✅ Cross-validation for unbiased model comparison",
    "✅ Publication-ready output generation",
    "✅ Comprehensive results documentation",
    "✅ Power analysis for study design insights"
]

for practice in best_practices:
    print(f"  {practice}")
```

### Next Steps and Extensions

```python
print("\n🚀 Potential Extensions:")

extensions = [
    "🔬 Multi-state models for different life stages or breeding status",
    "🏞️ Spatial capture-recapture with location information", 
    "📊 Hierarchical models for multiple populations or species",
    "🌊 Environmental stochasticity with random effects",
    "🧬 Individual heterogeneity modeling",
    "📈 Integrated population models combining multiple data sources",
    "🤖 Machine learning integration for covariate selection",
    "🌐 Bayesian analysis for prior information incorporation"
]

for extension in extensions:
    print(f"  {extension}")

print(f"\n🎓 Congratulations! You've completed a comprehensive capture-recapture analysis with Pradel-JAX.")
print(f"The skills demonstrated here apply to any capture-recapture study using the Pradel framework.")
```

---

## 📚 Additional Resources

### Related Documentation
- [**Model Specification Guide**](../user-guide/model-specification.md) - Advanced formula syntax
- [**Optimization Framework**](../user-guide/optimization.md) - Choosing the right strategy
- [**Statistical Inference**](../user-guide/statistical-inference.md) - Understanding uncertainty
- [**Performance Guide**](../user-guide/performance.md) - Scaling to large datasets

### Example Scripts
- [**Nebraska Analysis**](nebraska-analysis-walkthrough.md) - Large-scale real-world example
- [**Multi-model Comparison**](model-comparison.md) - Advanced model selection
- [**RMark Integration**](rmark-integration.md) - Validating against established tools

### Research Applications
- [**Power Analysis Tutorial**](power-analysis.md) - Study design optimization
- [**Population Projections**](population-projections.md) - Forecasting with uncertainty
- [**Meta-analysis Framework**](meta-analysis.md) - Combining multiple studies

---

*This tutorial demonstrates the full power of Pradel-JAX for professional capture-recapture analysis. The framework provides all tools needed for publication-quality research while maintaining ease of use and statistical rigor.*