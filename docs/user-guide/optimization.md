# Optimization Framework Guide

## 🎯 Overview

Pradel-JAX provides an intelligent optimization framework that automatically selects the best optimization strategy for your capture-recapture models. This guide explains how to choose and use different optimizers for maximum performance and reliability.

## 🚀 Quick Start

### Basic Usage (Recommended)
```python
import pradel_jax as pj

# Load your data
data_context = pj.load_data("data/my_dataset.csv")

# Create model specification
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",    # Survival with sex effect
    p="~1 + sex",      # Detection with sex effect  
    f="~1"             # Constant recruitment
)

# Fit model with automatic optimization
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context
)

print(f"Success: {result.success}")
print(f"Optimizer used: {result.strategy_used}")
print(f"AIC: {result.aic:.2f}")
```

The framework automatically selects the best optimizer based on your data characteristics.

## 📊 Available Optimization Strategies

Pradel-JAX offers multiple optimization strategies, each optimized for different scenarios:

### L-BFGS-B (Recommended Default)
- **Best for**: Small to medium datasets (<10k individuals)
- **Success rate**: 100%
- **Speed**: Fast (1-2 seconds)
- **Memory**: Moderate

```python
result = pj.fit_model(..., strategy="scipy_lbfgs")
```

**When to use:**
- General-purpose optimization
- Well-conditioned problems
- High precision requirements
- Most capture-recapture scenarios

### SLSQP (Maximum Robustness)
- **Best for**: Complex constraints, maximum reliability
- **Success rate**: 100%
- **Speed**: Fast (1-2 seconds)
- **Memory**: Moderate

```python
result = pj.fit_model(..., strategy="scipy_slsqp")
```

**When to use:**
- When robustness is critical
- Complex parameter constraints
- Previous L-BFGS-B failures

### JAX Adam (Large-Scale & GPU)
- **Best for**: Large datasets (50k+ individuals), GPU acceleration
- **Success rate**: 85-95% (varies with tuning)
- **Speed**: Variable (2-20+ seconds)
- **Memory**: Low

```python
result = pj.fit_model(..., strategy="jax_adam")
```

**When to use:**
- Large-scale problems (>50k individuals)
- GPU acceleration available
- Memory-constrained environments
- Streaming/mini-batch scenarios

### Multi-Start (Global Optimization)
- **Best for**: Difficult optimization landscapes
- **Success rate**: 98-99%
- **Speed**: Moderate (8-12 seconds)
- **Memory**: Higher

```python
result = pj.fit_model(..., strategy="multi_start")
```

**When to use:**
- Ill-conditioned problems
- Multiple local minima suspected
- When global optimization is needed
- Previous single-start failures

## 🔧 Optimizer Selection Guide

### Problem Size Guidelines

| Dataset Size | Recommended Strategy | Alternative |
|--------------|---------------------|-------------|
| < 1,000 individuals | L-BFGS-B | SLSQP |
| 1,000 - 10,000 | L-BFGS-B | Multi-start |
| 10,000 - 50,000 | L-BFGS-B or Multi-start | JAX Adam |
| > 50,000 individuals | JAX Adam | Multi-start |

### Problem Characteristics

**Use L-BFGS-B when:**
- Standard capture-recapture models
- Well-behaved data (no extreme sparsity)
- Parameter count < 100
- Need fast, reliable results

**Use JAX Adam when:**
- Large parameter spaces (>500 parameters)
- GPU acceleration available
- Hierarchical or complex models
- Memory constraints exist

**Use Multi-start when:**
- Previous optimizations failed
- Suspect multiple local minima
- Ill-conditioned covariance matrices
- Need global optimization guarantee

**Use SLSQP when:**
- Maximum robustness required
- Complex parameter constraints
- Production systems requiring reliability

## ⚙️ JAX Adam Configuration

JAX Adam requires careful tuning for statistical optimization. Here are the key insights:

### Default Configuration
```python
from pradel_jax.optimization import OptimizationConfig

config = OptimizationConfig(
    max_iter=10000,        # More iterations than ML problems
    tolerance=1e-2,        # Relaxed tolerance vs 1e-8 for L-BFGS
    learning_rate=0.00001, # Much smaller than ML default (0.001)
    init_scale=0.1         # Conservative initialization
)

result = pj.fit_model(..., strategy="jax_adam", config=config)
```

### Why These Parameters?

**Small Learning Rate (0.00001 vs 0.001)**
- Capture-recapture gradients are ~100x larger than typical ML problems
- Statistical optimization requires more careful parameter updates
- Prevents overshooting in likelihood landscapes

**Relaxed Tolerance (1e-2 vs 1e-8)**
- Statistical significance achieved at 1e-2 gradient norm
- Further precision often not meaningful for biological parameters
- Balances computational cost with practical accuracy

**More Iterations (10,000 vs 1,000)**
- First-order methods need more steps than second-order
- Statistical convergence can be slower than ML convergence
- Ensures thorough exploration of parameter space

### Advanced JAX Adam Options
```python
# Adaptive learning rate with warm restarts
config = OptimizationConfig(
    learning_rate=0.00001,
    use_adaptive_lr=True,
    lr_decay_factor=0.8,
    patience=1000,
    warm_restart_every=2000
)

# Early stopping based on likelihood improvement
config = OptimizationConfig(
    early_stopping=True,
    min_improvement=1e-4,
    patience=500
)
```

## 🔍 Automatic Strategy Selection

When you don't specify a strategy, Pradel-JAX intelligently chooses based on:

### Data Characteristics
- **Dataset size**: Number of individuals and occasions
- **Sparsity**: Proportion of zero captures
- **Conditioning**: Eigenvalue analysis of design matrices
- **Parameter count**: Total number of parameters to estimate

### Performance Prediction
The framework uses empirical data from 4,853+ model fits to predict:
- Convergence probability for each strategy
- Expected runtime and memory usage
- Risk of numerical issues

### Selection Algorithm
```python
def select_strategy(data_context, formula_spec):
    """Intelligent strategy selection based on problem characteristics"""
    
    # Analyze problem characteristics
    size_score = analyze_problem_size(data_context)
    conditioning_score = analyze_conditioning(data_context, formula_spec)
    sparsity_score = analyze_sparsity(data_context)
    
    # Predict performance for each strategy
    predictions = predict_performance(size_score, conditioning_score, sparsity_score)
    
    # Select highest confidence strategy
    return max(predictions, key=lambda x: x.confidence_score)
```

## 📈 Performance Monitoring

### Real-time Monitoring
```python
# Enable detailed monitoring
result = pj.fit_model(
    ...,
    enable_monitoring=True,
    monitor_config={
        'track_convergence': True,
        'profile_performance': True,
        'log_level': 'INFO'
    }
)

# Access monitoring data
print(f"Convergence path: {result.convergence_history}")
print(f"Performance metrics: {result.performance_metrics}")
```

### Convergence Diagnostics
```python
# Check convergence quality
if result.success:
    print(f"Final gradient norm: {result.final_gradient_norm}")
    print(f"Condition number: {result.condition_number}")
    print(f"Eigenvalue ratio: {result.eigenvalue_ratio}")
else:
    print(f"Failure reason: {result.failure_reason}")
    print(f"Suggestions: {result.suggestions}")
```

## 🚨 Troubleshooting

### Common Issues and Solutions

**"Optimization failed to converge"**
```python
# Try more robust strategy
result = pj.fit_model(..., strategy="multi_start")

# Or check data quality
validation = pj.validate_data(data_context)
if not validation.is_valid:
    print(f"Data issues: {validation.issues}")
```

**"JAX Adam not converging"**
```python
# Reduce learning rate
config = OptimizationConfig(learning_rate=0.000001)
result = pj.fit_model(..., strategy="jax_adam", config=config)

# Or switch to L-BFGS-B
result = pj.fit_model(..., strategy="scipy_lbfgs")
```

**"Memory issues with large datasets"**
```python
# Use JAX Adam with lower memory footprint
result = pj.fit_model(..., strategy="jax_adam")

# Or try multi-start with resource limits
config = OptimizationConfig(max_memory_gb=8.0)
result = pj.fit_model(..., strategy="multi_start", config=config)
```

**"Numerical instability"**
```python
# Check data conditioning
analysis = pj.analyze_data_conditioning(data_context, formula_spec)
print(f"Condition number: {analysis.condition_number}")

# Apply preprocessing
data_context = pj.preprocess_data(data_context, 
                                 center_covariates=True,
                                 scale_covariates=True)
```

## 🎯 Best Practices

### For Small to Medium Datasets
1. **Start with L-BFGS-B** (default choice)
2. **Use SLSQP** if L-BFGS-B fails
3. **Try multi-start** for difficult problems
4. **Monitor convergence** with detailed logging

### For Large Datasets
1. **Consider JAX Adam** for 50k+ individuals
2. **Tune learning rate** carefully
3. **Use GPU** when available
4. **Monitor memory usage** and adjust batch sizes

### For Production Systems
1. **Use SLSQP** for maximum reliability
2. **Enable comprehensive monitoring**
3. **Set up fallback strategies**
4. **Log all optimization attempts**

### For Research and Comparison
1. **Compare multiple strategies** systematically
2. **Use experiment tracking** for reproducibility
3. **Document strategy selection** rationale
4. **Share performance benchmarks**

## 📚 Advanced Topics

### Custom Optimization Strategies
```python
from pradel_jax.optimization import CustomOptimizer

class MyOptimizer(CustomOptimizer):
    def optimize(self, objective, initial_params, bounds):
        # Your custom optimization logic
        return optimization_result

# Register and use
pj.register_optimizer("my_optimizer", MyOptimizer)
result = pj.fit_model(..., strategy="my_optimizer")
```

### Experiment Tracking Integration
```python
import mlflow

# Enable MLflow tracking
with mlflow.start_run():
    result = pj.fit_model(..., enable_tracking=True)
    
    # Metrics automatically logged:
    # - Strategy used
    # - Convergence time
    # - Final likelihood
    # - Parameter estimates
```

### Parallel Strategy Comparison
```python
from pradel_jax.optimization import compare_strategies

# Compare all available strategies
comparison = compare_strategies(
    objective_function=pj.pradel_likelihood,
    initial_parameters=initial_params,
    context=data_context,
    strategies=['scipy_lbfgs', 'scipy_slsqp', 'jax_adam', 'multi_start']
)

# Analyze results
best_strategy = comparison.best_strategy
print(f"Winner: {best_strategy.name} (AIC: {best_strategy.aic:.2f})")
```

## 🔗 Related Documentation

- [**Installation Guide**](installation.md) - Setting up optimization dependencies
- [**Quick Start**](../tutorials/quickstart.md) - Your first optimized model
- [**Performance Tutorial**](../tutorials/performance.md) - Optimization best practices
- [**Large-Scale Guide**](large-scale.md) - Working with big datasets
- [**API Reference**](../api/optimization.md) - Technical optimization API details

---

*This guide covers the optimization framework as of August 2025. For the latest updates, see the [changelog](../CHANGELOG.md).*