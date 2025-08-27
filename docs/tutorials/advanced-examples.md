# Advanced Examples and Use Cases

This guide provides comprehensive, real-world examples of using Pradel-JAX for complex capture-recapture analyses. These examples demonstrate advanced features and best practices for different research scenarios.

## Table of Contents

1. [Multi-Species Community Analysis](#multi-species-community-analysis)
2. [Long-Term Population Dynamics](#long-term-population-dynamics) 
3. [Environmental Effects Modeling](#environmental-effects-modeling)
4. [Large-Scale Dataset Analysis](#large-scale-dataset-analysis)
5. [Model Selection and Comparison](#model-selection-and-comparison)
6. [Bootstrap and Uncertainty Analysis](#bootstrap-and-uncertainty-analysis)
7. [Time-Varying Covariate Analysis](#time-varying-covariate-analysis)
8. [Production Pipeline Development](#production-pipeline-development)

## Multi-Species Community Analysis

### Scenario
Analyzing capture-recapture data for multiple bird species to compare survival rates and detection probabilities across species and habitats.

### Data Structure
```csv
species,habitat,ch,mass,sex,age_class
EABL,forest,110101,25.3,M,adult
EABL,forest,101011,24.1,F,adult
COYE,grassland,111001,11.2,M,juvenile
COYE,grassland,101101,10.8,F,adult
RBNU,forest,110011,9.8,M,adult
```

### Analysis Implementation

```python
import pradel_jax as pj
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Load and examine community data
data = pj.load_data("community_data.csv")

print(f"Community data: {data.n_individuals} individuals")
print(f"Species: {data.covariates['species'].value_counts()}")
print(f"Habitats: {data.covariates['habitat'].value_counts()}")

# Species-specific analysis function
def analyze_species(species_data_pair):
    """Analyze single species with multiple models."""
    species_name, species_data = species_data_pair
    
    print(f"Analyzing {species_name}: {species_data.n_individuals} individuals")
    
    models = {}
    
    # Model 1: Constant parameters
    models['constant'] = pj.fit_model(
        formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
        data=species_data,
        strategy="multi_start"  # Robust optimization
    )
    
    # Model 2: Sex effects
    if 'sex' in species_data.covariates and species_data.covariates['sex'].nunique() > 1:
        models['sex'] = pj.fit_model(
            formula=pj.create_formula_spec(phi="~1 + sex", p="~1 + sex", f="~1"),
            data=species_data
        )
    
    # Model 3: Habitat effects
    if 'habitat' in species_data.covariates and species_data.covariates['habitat'].nunique() > 1:
        models['habitat'] = pj.fit_model(
            formula=pj.create_formula_spec(phi="~1 + habitat", p="~1", f="~1"),
            data=species_data
        )
    
    # Model 4: Mass effects (if sufficient variation)
    if 'mass' in species_data.covariates:
        mass_cv = species_data.covariates['mass'].std() / species_data.covariates['mass'].mean()
        if mass_cv > 0.1:  # Sufficient variation
            models['mass'] = pj.fit_model(
                formula=pj.create_formula_spec(
                    phi="~1 + standardize(mass)",
                    p="~1 + standardize(mass)", 
                    f="~1"
                ),
                data=species_data
            )
    
    # Model selection
    successful_models = {name: result for name, result in models.items() if result.success}
    
    if successful_models:
        comparison = pj.compare_models(successful_models)
        best_model_name = comparison.sort_values('aic').index[0]
        best_model = successful_models[best_model_name]
        
        return {
            'species': species_name,
            'n_individuals': species_data.n_individuals,
            'best_model': best_model_name,
            'aic': best_model.aic,
            'parameters': dict(best_model.parameter_estimates),
            'standard_errors': dict(best_model.standard_errors),
            'models_tested': list(models.keys()),
            'models_successful': list(successful_models.keys()),
            'model_comparison': comparison
        }
    else:
        return {
            'species': species_name,
            'n_individuals': species_data.n_individuals,
            'error': 'No models converged'
        }

# Split data by species
species_datasets = {}
for species in data.covariates['species'].unique():
    species_mask = data.covariates['species'] == species
    species_datasets[species] = data.filter_individuals(species_mask)

# Parallel species analysis
with ProcessPoolExecutor(max_workers=4) as executor:
    species_results = list(executor.map(analyze_species, species_datasets.items()))

# Compile community results
community_results = pd.DataFrame([r for r in species_results if 'error' not in r])

print("\n=== Community Analysis Results ===")
print(community_results[['species', 'n_individuals', 'best_model', 'aic']])

# Extract survival estimates for comparison
survival_estimates = {}
for result in species_results:
    if 'error' not in result:
        species = result['species']
        phi_intercept = result['parameters'].get('phi_intercept', np.nan)
        phi_intercept_se = result['standard_errors'].get('phi_intercept', np.nan)
        
        # Convert to probability scale
        survival_prob = pj.logit_inverse(phi_intercept) if not np.isnan(phi_intercept) else np.nan
        survival_se_prob = phi_intercept_se * survival_prob * (1 - survival_prob) if not np.isnan(phi_intercept_se) else np.nan
        
        survival_estimates[species] = {
            'survival': survival_prob,
            'survival_se': survival_se_prob,
            'n_individuals': result['n_individuals']
        }

# Plot community survival rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Survival rates with error bars
species_list = list(survival_estimates.keys())
survival_vals = [survival_estimates[sp]['survival'] for sp in species_list]
survival_ses = [survival_estimates[sp]['survival_se'] for sp in species_list]
sample_sizes = [survival_estimates[sp]['n_individuals'] for sp in species_list]

ax1.errorbar(range(len(species_list)), survival_vals, yerr=survival_ses, 
             fmt='o', capsize=5, capthick=2, markersize=8)
ax1.set_xticks(range(len(species_list)))
ax1.set_xticklabels(species_list, rotation=45)
ax1.set_ylabel('Annual Survival Probability')
ax1.set_title('Species-Specific Survival Rates')
ax1.grid(True, alpha=0.3)

# Sample sizes
bars = ax2.bar(species_list, sample_sizes, alpha=0.7)
ax2.set_ylabel('Sample Size (Individuals)')
ax2.set_title('Sample Sizes by Species')
ax2.tick_params(axis='x', rotation=45)

# Add sample size labels on bars
for bar, size in zip(bars, sample_sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(size), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Statistical comparison of survival rates
print("\n=== Species Survival Rate Comparison ===")
for species, estimates in survival_estimates.items():
    print(f"{species}: {estimates['survival']:.3f} ± {estimates['survival_se']:.3f} (n={estimates['n_individuals']})")

# Test for significant differences (approximate)
species_pairs = [(s1, s2) for i, s1 in enumerate(species_list) for s2 in species_list[i+1:]]
print("\nPairwise survival rate differences:")
for s1, s2 in species_pairs:
    est1, se1 = survival_estimates[s1]['survival'], survival_estimates[s1]['survival_se']
    est2, se2 = survival_estimates[s2]['survival'], survival_estimates[s2]['survival_se']
    
    diff = est1 - est2
    se_diff = np.sqrt(se1**2 + se2**2)
    z_score = diff / se_diff if se_diff > 0 else np.nan
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score))) if not np.isnan(z_score) else np.nan
    
    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"{s1} vs {s2}: diff = {diff:+.3f} ± {se_diff:.3f}, p = {p_value:.4f} {sig_marker}")

# Export community results
export_dir = "community_analysis_results"
pj.export_model_results(
    {result['species']: result for result in species_results if 'error' not in result},
    output_dir=export_dir,
    prefix="community_analysis"
)

print(f"\nCommunity analysis results exported to {export_dir}/")
```

## Long-Term Population Dynamics

### Scenario
Analyzing 20-year capture-recapture dataset to understand population trends, environmental drivers, and demographic changes over time.

```python
import pradel_jax as pj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load long-term dataset
data = pj.load_data("longterm_population_data.csv")

print(f"Long-term dataset: {data.n_individuals} individuals over {data.n_occasions} years")
print(f"Year range: {data.year_range}")

# Environmental covariates over time
environmental_data = pd.read_csv("environmental_covariates.csv")
print("Environmental variables:", environmental_data.columns.tolist())

# Merge environmental data with capture data
# Add time-varying environmental covariates
for year in data.years:
    env_year = environmental_data[environmental_data['year'] == year]
    if not env_year.empty:
        data.add_occasion_covariate(f"temperature_{year}", env_year['mean_temp'].iloc[0])
        data.add_occasion_covariate(f"precipitation_{year}", env_year['total_precip'].iloc[0])
        data.add_occasion_covariate(f"nao_index_{year}", env_year['nao_index'].iloc[0])

# Time trend analysis
def analyze_temporal_trends():
    """Analyze population trends over time."""
    
    models = {}
    
    # Model 1: Constant parameters (null model)
    models['constant'] = pj.fit_model(
        formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
        data=data
    )
    
    # Model 2: Linear time trends
    models['linear_time'] = pj.fit_model(
        formula=pj.create_formula_spec(
            phi="~1 + standardize(year)",
            p="~1 + standardize(year)",
            f="~1 + standardize(year)"
        ),
        data=data
    )
    
    # Model 3: Quadratic time trends (detect non-linear changes)
    models['quadratic_time'] = pj.fit_model(
        formula=pj.create_formula_spec(
            phi="~1 + standardize(year) + I(standardize(year)**2)",
            p="~1 + standardize(year)",
            f="~1"
        ),
        data=data
    )
    
    # Model 4: Environmental effects
    models['environmental'] = pj.fit_model(
        formula=pj.create_formula_spec(
            phi="~1 + standardize(temperature_tv) + standardize(precipitation_tv)",
            p="~1 + standardize(year)",
            f="~1 + standardize(nao_index_tv)"
        ),
        data=data
    )
    
    # Model 5: Environmental + time interactions
    models['env_time_interaction'] = pj.fit_model(
        formula=pj.create_formula_spec(
            phi="~1 + standardize(year) * standardize(temperature_tv)",
            p="~1 + standardize(precipitation_tv)",
            f="~1 + standardize(nao_index_tv)"
        ),
        data=data,
        strategy="multi_start"  # More complex model needs robust optimization
    )
    
    return models

# Run temporal analysis
print("Analyzing temporal trends...")
temporal_models = analyze_temporal_trends()

# Model comparison
successful_temporal_models = {name: model for name, model in temporal_models.items() if model.success}
temporal_comparison = pj.compare_models(successful_temporal_models)
print("\n=== Temporal Model Comparison ===")
print(temporal_comparison[['aic', 'delta_aic', 'aic_weight', 'log_likelihood']].round(3))

best_temporal_model = successful_temporal_models[temporal_comparison.sort_values('aic').index[0]]

# Extract and visualize trends
def plot_population_trends(model_result):
    """Plot population trends from best model."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Survival probability over time
    years = data.years
    if 'phi_standardize(year)' in model_result.parameter_estimates:
        # Linear trend model
        phi_intercept = model_result.parameter_estimates['phi_intercept']
        phi_year_effect = model_result.parameter_estimates['phi_standardize(year)']
        
        # Standardized years for prediction
        scaler = StandardScaler()
        years_std = scaler.fit_transform(np.array(years).reshape(-1, 1)).flatten()
        
        # Predict survival over time
        phi_logit = phi_intercept + phi_year_effect * years_std
        phi_prob = pj.logit_inverse(phi_logit)
        
        ax1.plot(years, phi_prob, 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Annual Survival Probability')
        ax1.set_title('Survival Trend Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Add trend statistics
        trend_slope_per_year = phi_year_effect / len(years)  # Approximate per-year change
        ax1.text(0.05, 0.95, f'Trend: {trend_slope_per_year:+.4f}/year (logit scale)', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # 2. Detection probability over time
    if 'p_standardize(year)' in model_result.parameter_estimates:
        p_intercept = model_result.parameter_estimates['p_intercept'] 
        p_year_effect = model_result.parameter_estimates['p_standardize(year)']
        
        p_logit = p_intercept + p_year_effect * years_std
        p_prob = pj.logit_inverse(p_logit)
        
        ax2.plot(years, p_prob, 's-', color='orange', linewidth=2, markersize=6)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Detection Probability')
        ax2.set_title('Detection Probability Trend')
        ax2.grid(True, alpha=0.3)
    
    # 3. Recruitment over time
    if 'f_standardize(year)' in model_result.parameter_estimates:
        f_intercept = model_result.parameter_estimates['f_intercept']
        f_year_effect = model_result.parameter_estimates['f_standardize(year)']
        
        f_log = f_intercept + f_year_effect * years_std
        f_rate = np.exp(f_log)
        
        ax3.plot(years, f_rate, '^-', color='green', linewidth=2, markersize=6)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Recruitment Rate')
        ax3.set_title('Recruitment Trend Over Time')
        ax3.grid(True, alpha=0.3)
    
    # 4. Population growth rate (lambda = phi + f)
    if all(param in model_result.parameter_estimates for param in ['phi_intercept', 'f_intercept']):
        lambda_values = phi_prob + f_rate  # Approximate population growth rate
        
        ax4.plot(years, lambda_values, 'o-', color='red', linewidth=2, markersize=6)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Stable population')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Population Growth Rate (λ)')
        ax4.set_title('Population Growth Rate Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Highlight periods of decline/growth
        declining_years = years[lambda_values < 1]
        if len(declining_years) > 0:
            ax4.text(0.05, 0.95, f'Declining years: {len(declining_years)}/{len(years)}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

# Plot trends from best model
plot_population_trends(best_temporal_model)

# Environmental correlation analysis
def analyze_environmental_correlations():
    """Analyze correlations between environmental variables and demographic parameters."""
    
    # Extract annual parameter estimates using time-varying model
    if 'environmental' in successful_temporal_models:
        env_model = successful_temporal_models['environmental']
        
        print("\n=== Environmental Effects ===")
        
        # Temperature effects
        if 'phi_standardize(temperature_tv)' in env_model.parameter_estimates:
            temp_effect = env_model.parameter_estimates['phi_standardize(temperature_tv)']
            temp_se = env_model.standard_errors.get('phi_standardize(temperature_tv)', 0)
            print(f"Temperature effect on survival: {temp_effect:.4f} ± {temp_se:.4f}")
        
        # Precipitation effects
        if 'phi_standardize(precipitation_tv)' in env_model.parameter_estimates:
            precip_effect = env_model.parameter_estimates['phi_standardize(precipitation_tv)']
            precip_se = env_model.standard_errors.get('phi_standardize(precipitation_tv)', 0)
            print(f"Precipitation effect on survival: {precip_effect:.4f} ± {precip_se:.4f}")
        
        # NAO effects on recruitment
        if 'f_standardize(nao_index_tv)' in env_model.parameter_estimates:
            nao_effect = env_model.parameter_estimates['f_standardize(nao_index_tv)']
            nao_se = env_model.standard_errors.get('f_standardize(nao_index_tv)', 0)
            print(f"NAO effect on recruitment: {nao_effect:.4f} ± {nao_se:.4f}")

analyze_environmental_correlations()

# Bootstrap analysis for robust uncertainty estimates
print("\nRunning bootstrap analysis for temporal trends...")
temporal_bootstrap = pj.fit_model(
    formula=pj.create_formula_spec(
        phi="~1 + standardize(year)",
        p="~1 + standardize(year)",
        f="~1"
    ),
    data=data,
    bootstrap_confidence_intervals=True,
    bootstrap_config={
        "n_bootstrap": 1000,
        "method": "bca",
        "parallel": True,
        "confidence_level": 0.95
    }
)

if temporal_bootstrap.success:
    print("Bootstrap confidence intervals for temporal trends:")
    for param, ci in temporal_bootstrap.bootstrap_confidence_intervals.items():
        est = temporal_bootstrap.parameter_estimates[param]
        print(f"{param}: {est:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

# Export long-term analysis results
export_paths = pj.export_model_results(
    successful_temporal_models,
    output_dir="longterm_analysis_results",
    prefix="population_dynamics",
    include_timestamp=True
)

print(f"\nLong-term analysis results exported:")
for export_type, path in export_paths.items():
    print(f"  {export_type}: {path}")
```

## Environmental Effects Modeling

### Scenario
Investigating how weather patterns, habitat quality, and human disturbance affect survival and recruitment in a bird population.

```python
import pradel_jax as pj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load data with environmental covariates
data = pj.load_data("environmental_study_data.csv")
environmental_vars = pd.read_csv("environmental_variables.csv")

print(f"Environmental study: {data.n_individuals} individuals")
print(f"Environmental variables: {environmental_vars.columns.tolist()}")

# Merge environmental data
# Time-invariant environmental variables (habitat characteristics)
habitat_data = environmental_vars[['site_id', 'forest_cover', 'edge_distance', 'elevation']].drop_duplicates()

# Time-varying environmental variables  
weather_data = environmental_vars[['year', 'mean_temp', 'total_precip', 'extreme_weather_days']]
disturbance_data = environmental_vars[['year', 'logging_intensity', 'road_density', 'human_visits']]

# Add environmental covariates to data context
for _, row in habitat_data.iterrows():
    site_mask = data.covariates['site_id'] == row['site_id']
    if site_mask.any():
        for var in ['forest_cover', 'edge_distance', 'elevation']:
            data.covariates.loc[site_mask, var] = row[var]

# Add time-varying environmental covariates
for year in data.years:
    weather_year = weather_data[weather_data['year'] == year]
    disturbance_year = disturbance_data[disturbance_data['year'] == year]
    
    if not weather_year.empty:
        for var in ['mean_temp', 'total_precip', 'extreme_weather_days']:
            data.add_occasion_covariate(f"{var}_{year}", weather_year[var].iloc[0])
    
    if not disturbance_year.empty:
        for var in ['logging_intensity', 'road_density', 'human_visits']:
            data.add_occasion_covariate(f"{var}_{year}", disturbance_year[var].iloc[0])

# Environmental effects model suite
def build_environmental_models():
    """Build comprehensive suite of environmental effects models."""
    
    models = {}
    
    # Null model
    models['null'] = pj.create_formula_spec(phi="~1", p="~1", f="~1")
    
    # Habitat effects (time-invariant)
    models['habitat'] = pj.create_formula_spec(
        phi="~1 + standardize(forest_cover) + standardize(elevation)",
        p="~1 + standardize(edge_distance)",
        f="~1 + standardize(forest_cover)"
    )
    
    # Weather effects (time-varying)
    models['weather'] = pj.create_formula_spec(
        phi="~1 + standardize(mean_temp_tv) + standardize(total_precip_tv)",
        p="~1 + standardize(extreme_weather_days_tv)",
        f="~1 + standardize(total_precip_tv)"
    )
    
    # Human disturbance effects (time-varying)
    models['disturbance'] = pj.create_formula_spec(
        phi="~1 + standardize(logging_intensity_tv) + standardize(road_density_tv)",
        p="~1 + standardize(human_visits_tv)",
        f="~1 + standardize(logging_intensity_tv)"
    )
    
    # Combined habitat + weather
    models['habitat_weather'] = pj.create_formula_spec(
        phi="~1 + standardize(forest_cover) + standardize(mean_temp_tv) + standardize(total_precip_tv)",
        p="~1 + standardize(edge_distance) + standardize(extreme_weather_days_tv)",
        f="~1 + standardize(forest_cover) + standardize(total_precip_tv)"
    )
    
    # Combined weather + disturbance
    models['weather_disturbance'] = pj.create_formula_spec(
        phi="~1 + standardize(mean_temp_tv) + standardize(logging_intensity_tv)",
        p="~1 + standardize(extreme_weather_days_tv) + standardize(human_visits_tv)",
        f="~1 + standardize(total_precip_tv) * standardize(logging_intensity_tv)"  # Interaction
    )
    
    # Full environmental model
    models['full_environment'] = pj.create_formula_spec(
        phi="~1 + standardize(forest_cover) + standardize(mean_temp_tv) + standardize(logging_intensity_tv)",
        p="~1 + standardize(edge_distance) + standardize(extreme_weather_days_tv) + standardize(human_visits_tv)",
        f="~1 + standardize(forest_cover) + standardize(total_precip_tv) + standardize(logging_intensity_tv)"
    )
    
    # Interaction models (complex)
    models['weather_habitat_interaction'] = pj.create_formula_spec(
        phi="~1 + standardize(forest_cover) * standardize(mean_temp_tv)",
        p="~1 + standardize(edge_distance) * standardize(extreme_weather_days_tv)",
        f="~1 + standardize(forest_cover)"
    )
    
    return models

# Fit environmental models
environmental_model_specs = build_environmental_models()
environmental_results = {}

print("Fitting environmental effects models...")
for model_name, formula_spec in environmental_model_specs.items():
    print(f"  Fitting {model_name}...")
    
    try:
        result = pj.fit_model(
            formula=formula_spec,
            data=data,
            strategy="multi_start",  # Robust optimization for complex models
            multi_start_attempts=10,
            compute_standard_errors=True,
            confidence_intervals=True
        )
        
        if result.success:
            environmental_results[model_name] = result
            print(f"    ✓ Success (AIC: {result.aic:.1f})")
        else:
            print(f"    ✗ Failed: {result.optimization_message}")
            
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")

# Model comparison and selection
env_comparison = pj.compare_models(environmental_results)
print("\n=== Environmental Model Comparison ===")
print(env_comparison[['aic', 'delta_aic', 'aic_weight', 'n_parameters']].round(3))

best_env_model_name = env_comparison.sort_values('aic').index[0]
best_env_model = environmental_results[best_env_model_name]

print(f"\nBest model: {best_env_model_name}")
print(f"AIC: {best_env_model.aic:.2f}")

# Detailed analysis of environmental effects
def analyze_environmental_effects(model_result, model_name):
    """Analyze and interpret environmental effects from best model."""
    
    print(f"\n=== Environmental Effects Analysis: {model_name} ===")
    
    # Categorize parameters
    survival_effects = {}
    detection_effects = {}
    recruitment_effects = {}
    
    for param, estimate in model_result.parameter_estimates.items():
        se = model_result.standard_errors.get(param, 0)
        ci = model_result.confidence_intervals.get(param, {})
        z_score = estimate / se if se > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score))) if se > 0 else 1
        
        # Categorize by parameter type
        if param.startswith('phi_'):
            survival_effects[param] = {
                'estimate': estimate, 'se': se, 'ci': ci, 'p_value': p_value
            }
        elif param.startswith('p_'):
            detection_effects[param] = {
                'estimate': estimate, 'se': se, 'ci': ci, 'p_value': p_value
            }
        elif param.startswith('f_'):
            recruitment_effects[param] = {
                'estimate': estimate, 'se': se, 'ci': ci, 'p_value': p_value
            }
    
    # Print survival effects
    if survival_effects:
        print("\nSurvival Probability Effects:")
        for param, stats in survival_effects.items():
            if param != 'phi_intercept':  # Skip intercept
                clean_name = param.replace('phi_', '').replace('standardize(', '').replace(')', '').replace('_tv', ' (time-varying)')
                sig_marker = get_significance_marker(stats['p_value'])
                print(f"  {clean_name}: {stats['estimate']:+.4f} ± {stats['se']:.4f} {sig_marker}")
                
                # Biological interpretation
                effect_size = abs(stats['estimate'])
                if effect_size > 0.5:
                    magnitude = "large"
                elif effect_size > 0.2:
                    magnitude = "moderate"
                else:
                    magnitude = "small"
                
                direction = "positive" if stats['estimate'] > 0 else "negative"
                print(f"    → {magnitude} {direction} effect on survival")
    
    # Print detection effects  
    if detection_effects:
        print("\nDetection Probability Effects:")
        for param, stats in detection_effects.items():
            if param != 'p_intercept':
                clean_name = param.replace('p_', '').replace('standardize(', '').replace(')', '').replace('_tv', ' (time-varying)')
                sig_marker = get_significance_marker(stats['p_value'])
                print(f"  {clean_name}: {stats['estimate']:+.4f} ± {stats['se']:.4f} {sig_marker}")
    
    # Print recruitment effects
    if recruitment_effects:
        print("\nRecruitment Rate Effects:")
        for param, stats in recruitment_effects.items():
            if param != 'f_intercept':
                clean_name = param.replace('f_', '').replace('standardize(', '').replace(')', '').replace('_tv', ' (time-varying)')
                sig_marker = get_significance_marker(stats['p_value'])
                print(f"  {clean_name}: {stats['estimate']:+.4f} ± {stats['se']:.4f} {sig_marker}")

def get_significance_marker(p_value):
    """Get significance marker for p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return ""

# Analyze best environmental model
analyze_environmental_effects(best_env_model, best_env_model_name)

# Visualize environmental effects
def plot_environmental_effects(model_result):
    """Create comprehensive plots of environmental effects."""
    
    # Extract significant effects
    significant_effects = {}
    for param, estimate in model_result.parameter_estimates.items():
        if param.endswith('_intercept'):  # Skip intercepts
            continue
            
        se = model_result.standard_errors.get(param, 0)
        if se > 0:
            z_score = abs(estimate / se)
            if z_score > 1.96:  # Significant at α = 0.05
                significant_effects[param] = {
                    'estimate': estimate,
                    'se': se,
                    'parameter_type': param.split('_')[0]
                }
    
    if not significant_effects:
        print("No significant environmental effects to plot.")
        return
    
    # Create effect size plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Effect sizes with confidence intervals
    param_names = list(significant_effects.keys())
    estimates = [significant_effects[p]['estimate'] for p in param_names]
    ses = [significant_effects[p]['se'] for p in param_names]
    colors = ['blue' if p.startswith('phi_') else 'orange' if p.startswith('p_') else 'green' 
              for p in param_names]
    
    # Clean parameter names for plotting
    clean_names = []
    for name in param_names:
        clean = name.replace('phi_', 'Survival: ').replace('p_', 'Detection: ').replace('f_', 'Recruitment: ')
        clean = clean.replace('standardize(', '').replace(')', '').replace('_tv', ' (TV)')
        clean_names.append(clean)
    
    y_positions = range(len(param_names))
    ax1.barh(y_positions, estimates, xerr=np.array(ses) * 1.96, color=colors, alpha=0.7, capsize=5)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(clean_names)
    ax1.set_xlabel('Effect Size (standardized)')
    ax1.set_title('Significant Environmental Effects')
    ax1.grid(True, alpha=0.3)
    
    # Model comparison
    model_names = list(environmental_results.keys())
    aics = [environmental_results[name].aic for name in model_names]
    delta_aics = [aic - min(aics) for aic in aics]
    
    bars = ax2.bar(range(len(model_names)), delta_aics, alpha=0.7)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('ΔAIC')
    ax2.set_title('Environmental Model Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Highlight best model
    best_idx = delta_aics.index(0)
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(1.0)
    
    plt.tight_layout()
    plt.show()

plot_environmental_effects(best_env_model)

# Effect size interpretation and recommendations
def environmental_management_recommendations(model_result):
    """Generate management recommendations based on environmental effects."""
    
    print("\n=== Management Recommendations ===")
    
    # Analyze survival effects for management relevance
    management_priorities = []
    
    for param, estimate in model_result.parameter_estimates.items():
        if not param.startswith('phi_') or param == 'phi_intercept':
            continue
            
        se = model_result.standard_errors.get(param, 0)
        if se == 0:
            continue
            
        z_score = abs(estimate / se)
        if z_score < 1.96:  # Not significant
            continue
        
        effect_size = abs(estimate)
        is_negative = estimate < 0
        
        # Extract variable name
        var_name = param.replace('phi_', '').replace('standardize(', '').replace(')', '')
        
        # Management relevance
        if 'logging' in var_name and is_negative and effect_size > 0.2:
            management_priorities.append({
                'variable': 'Logging intensity',
                'effect': 'Strong negative effect on survival',
                'recommendation': 'Reduce logging intensity in breeding areas',
                'priority': 'High'
            })
        
        elif 'road_density' in var_name and is_negative and effect_size > 0.1:
            management_priorities.append({
                'variable': 'Road density',
                'effect': 'Negative effect on survival',
                'recommendation': 'Limit new road construction, consider road closures',
                'priority': 'Medium'
            })
        
        elif 'forest_cover' in var_name and not is_negative and effect_size > 0.1:
            management_priorities.append({
                'variable': 'Forest cover',
                'effect': 'Positive effect on survival',
                'recommendation': 'Prioritize forest conservation and restoration',
                'priority': 'High'
            })
        
        elif 'human_visits' in var_name and is_negative:
            management_priorities.append({
                'variable': 'Human disturbance',
                'effect': 'Negative effect on detection/survival',
                'recommendation': 'Establish buffer zones, limit access during breeding',
                'priority': 'Medium'
            })
    
    # Print recommendations
    high_priority = [p for p in management_priorities if p['priority'] == 'High']
    medium_priority = [p for p in management_priorities if p['priority'] == 'Medium']
    
    if high_priority:
        print("\nHigh Priority Actions:")
        for i, rec in enumerate(high_priority, 1):
            print(f"{i}. {rec['variable']}: {rec['effect']}")
            print(f"   → {rec['recommendation']}")
    
    if medium_priority:
        print("\nMedium Priority Actions:")
        for i, rec in enumerate(medium_priority, 1):
            print(f"{i}. {rec['variable']}: {rec['effect']}")
            print(f"   → {rec['recommendation']}")
    
    if not management_priorities:
        print("No clear management recommendations from current model.")
        print("Consider additional data collection or alternative model structures.")

environmental_management_recommendations(best_env_model)

# Export environmental analysis
export_paths = pj.export_model_results(
    environmental_results,
    output_dir="environmental_analysis_results",
    prefix="environmental_effects",
    include_timestamp=True,
    export_confidence_intervals=True
)

print(f"\nEnvironmental analysis results exported:")
for export_type, path in export_paths.items():
    print(f"  {export_type}: {path}")
```

This comprehensive guide provides advanced, production-ready examples for complex capture-recapture analyses using Pradel-JAX. Each example includes:

1. **Real-world scenarios** with biological context
2. **Complete data processing** workflows  
3. **Advanced statistical modeling** techniques
4. **Robust error handling** and diagnostics
5. **Professional visualization** and reporting
6. **Practical interpretation** and recommendations

The examples demonstrate best practices for:
- Multi-level analyses (species, populations, communities)
- Time series and trend analysis
- Environmental effects modeling
- Large-scale data processing
- Model selection and comparison
- Statistical inference and uncertainty quantification
- Production pipeline development

These examples serve as templates for researchers and practitioners working with capture-recapture data in various ecological and conservation contexts.