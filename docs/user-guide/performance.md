# Performance Guide

This guide covers optimization strategies, performance tuning, and benchmarking for Pradel-JAX. Learn how to get the best performance for your capture-recapture analyses.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Optimization Strategy Selection](#optimization-strategy-selection)
3. [Hardware Considerations](#hardware-considerations)
4. [Large Dataset Optimization](#large-dataset-optimization)
5. [Memory Management](#memory-management)
6. [JAX Performance Tuning](#jax-performance-tuning)
7. [Benchmarking and Profiling](#benchmarking-and-profiling)
8. [Performance Troubleshooting](#performance-troubleshooting)
9. [Best Practices](#best-practices)

## Performance Overview

Pradel-JAX is designed for high performance with JAX-based numerical computing and intelligent optimization strategy selection. Performance characteristics vary significantly based on problem size, data structure, and hardware configuration.

### Performance Factors

**Problem Size**
- **Small**: < 1,000 individuals, < 10 occasions → Sub-second optimization
- **Medium**: 1,000-10,000 individuals → Seconds to minutes  
- **Large**: 10,000-100,000 individuals → Minutes to hours
- **Very Large**: 100,000+ individuals → Hours (requires optimization)

**Model Complexity**
- **Simple models** (constant parameters): Fastest convergence
- **Covariate models**: Moderate overhead from design matrix construction
- **Time-varying models**: Higher memory usage, longer optimization
- **Interaction models**: Exponential growth in parameter space

**Hardware Impact**
- **CPU**: Multi-core benefits for bootstrap and parallel processing
- **Memory**: Critical for large design matrices and time-varying covariates  
- **GPU**: Potential acceleration for very large problems (experimental)

## Optimization Strategy Selection

Pradel-JAX provides multiple optimization strategies optimized for different scenarios.

### Strategy Performance Comparison

| Strategy | Best For | Typical Speed | Success Rate | Memory Usage |
|----------|----------|---------------|--------------|--------------|
| **L-BFGS-B** | Small-medium problems | Fast (1-5s) | 95%+ | Moderate |
| **SLSQP** | Constrained problems | Fast (1-5s) | 90%+ | Moderate |
| **Multi-start** | Difficult landscapes | Moderate (5-30s) | 99%+ | Higher |
| **JAX Adam** | Large-scale problems | Variable | 85%+ | Lower |

### Automatic Strategy Selection

```python
import pradel_jax as pj

# Default: automatic strategy selection based on problem characteristics
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="auto"  # Chooses best strategy automatically
)

print(f"Selected strategy: {result.strategy_used}")
print(f"Optimization time: {result.optimization_time:.2f}s")
```

### Manual Strategy Selection

```python
# For small-medium problems (< 10k individuals)
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="lbfgs"  # Usually fastest
)

# For robust optimization (global search)
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="multi_start",
    multi_start_attempts=10  # More attempts = more robust but slower
)

# For very large problems (experimental)
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="adam",
    max_iterations=5000,     # Adam needs more iterations
    learning_rate=0.001      # Tune learning rate for problem
)
```

### Strategy Selection Guidelines

#### Use L-BFGS-B when:
- Dataset < 10,000 individuals
- Well-conditioned problem (not too many parameters)
- Need fastest convergence
- Standard capture-recapture models

```python
# Optimal L-BFGS-B configuration
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="lbfgs",
    max_iterations=1000,     # Usually converges < 100 iterations
    tolerance=1e-6,          # High precision
    use_bounds=True          # Enforce parameter bounds
)
```

#### Use Multi-start when:
- Previous optimization failed
- Complex parameter landscape suspected
- Robustness more important than speed
- Model selection (need reliable global optimum)

```python
# Robust multi-start configuration
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="multi_start",
    multi_start_attempts=20,        # More attempts = more robust
    base_strategy="lbfgs",          # Use fast strategy for each attempt
    parallel_starts=True            # Use multiple CPU cores
)
```

#### Use JAX Adam when:
- Dataset > 50,000 individuals
- Many parameters (> 100)
- GPU available
- Can tolerate lower precision

```python
# Large-scale Adam configuration
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="adam",
    max_iterations=10000,           # Adam needs more iterations
    learning_rate=0.0001,           # Smaller LR for stability
    tolerance=1e-2,                 # Relaxed tolerance
    batch_processing=True           # Process data in batches
)
```

## Hardware Considerations

### CPU Optimization

```python
# Configure for optimal CPU usage
import os

# Use all available CPU cores for numerical operations
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count()) 

# Enable parallel bootstrap
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    bootstrap_confidence_intervals=True,
    bootstrap_config={
        "n_bootstrap": 1000,
        "parallel": True,           # Use multiprocessing
        "n_jobs": os.cpu_count()    # Use all CPU cores
    }
)
```

### Memory Optimization

```python
# Monitor memory usage during optimization
import psutil

def monitor_memory():
    memory = psutil.virtual_memory()
    return f"Memory: {memory.percent}% ({memory.used / 1e9:.1f}GB used)"

print(f"Before optimization: {monitor_memory()}")

# Use memory-efficient options for large datasets
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    low_memory_mode=True,          # Reduce memory usage
    cache_design_matrices=False,   # Don't cache large matrices
    precision="float32"            # Use 32-bit instead of 64-bit
)

print(f"After optimization: {monitor_memory()}")
```

### GPU Acceleration (Experimental)

```python
# Check for GPU availability
import jax
print("Available devices:", jax.devices())

# Enable GPU if available
if jax.devices("gpu"):
    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    
    # GPU-optimized configuration
    result = pj.fit_model(
        formula=formula_spec,
        data=data_context,
        strategy="adam",               # Works best on GPU
        use_gpu=True,                 # Explicit GPU usage
        batch_size=1000               # Process in batches for GPU
    )
else:
    print("No GPU detected, using CPU")
```

## Large Dataset Optimization

### Data Sampling Strategies

```python
# Stratified sampling for model development
sample_data = pj.stratified_sample(
    data_context,
    n_samples=5000,                # Manageable sample size
    stratify_by=["sex", "region"], # Maintain structure
    random_state=42
)

# Develop model on sample
result_sample = pj.fit_model(
    formula=formula_spec,
    data=sample_data,
    strategy="auto"
)

# Apply to full dataset with optimized parameters
result_full = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    initial_parameters=result_sample.parameter_estimates,  # Warm start
    strategy="lbfgs"               # Fast convergence from good start
)
```

### Parallel Processing

```python
# Process multiple models in parallel
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def fit_single_model(formula_name_pair, data):
    name, formula = formula_name_pair
    result = pj.fit_model(formula=formula, data=data)
    return name, result

# Define multiple models
models = {
    "null": pj.create_formula_spec(phi="~1", p="~1", f="~1"),
    "sex": pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1"),
    "age": pj.create_formula_spec(phi="~1 + age", p="~1", f="~1"),
    "full": pj.create_formula_spec(phi="~1 + sex + age", p="~1 + sex", f="~1")
}

# Fit models in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = dict(executor.map(
        partial(fit_single_model, data=data_context),
        models.items()
    ))

# Compare results
comparison = pj.compare_models(results)
print(comparison.sort_values('aic'))
```

### Chunked Processing

```python
# For extremely large datasets, process in chunks
def process_data_chunks(data_path, chunk_size=10000):
    """Process large dataset in chunks."""
    
    results = []
    
    for chunk_idx, data_chunk in enumerate(pj.load_data_chunks(data_path, chunk_size)):
        print(f"Processing chunk {chunk_idx + 1}...")
        
        result = pj.fit_model(
            formula=formula_spec,
            data=data_chunk,
            strategy="lbfgs"
        )
        
        results.append({
            "chunk": chunk_idx,
            "n_individuals": data_chunk.n_individuals,
            "aic": result.aic,
            "parameters": result.parameter_estimates
        })
    
    return results

# Process large dataset
chunk_results = process_data_chunks("very_large_dataset.csv")

# Combine results (meta-analysis approach)
combined_result = pj.combine_chunk_results(chunk_results)
```

## Memory Management

### Memory Usage Patterns

```python
# Monitor memory usage patterns
import tracemalloc

tracemalloc.start()

# Load data
data = pj.load_data("large_dataset.csv")
snapshot1 = tracemalloc.take_snapshot()
print(f"After data loading: {snapshot1.total_size / 1e6:.1f} MB")

# Create design matrices
model = pj.PradelModel()
design_matrices = model.build_design_matrices(formula_spec, data)
snapshot2 = tracemalloc.take_snapshot()
print(f"After design matrices: {snapshot2.total_size / 1e6:.1f} MB")

# Fit model
result = pj.fit_model(formula=formula_spec, data=data)
snapshot3 = tracemalloc.take_snapshot()
print(f"After optimization: {snapshot3.total_size / 1e6:.1f} MB")

tracemalloc.stop()
```

### Memory Optimization Techniques

```python
# 1. Use appropriate data types
data_optimized = pj.load_data(
    "dataset.csv",
    optimize_dtypes=True,          # Automatically optimize data types
    categorical_threshold=0.5      # Convert to categorical if < 50% unique
)

# 2. Clear intermediate results
def memory_efficient_fitting(formula, data):
    # Fit model
    result = pj.fit_model(formula=formula, data=data)
    
    # Extract only what you need
    summary = {
        "aic": result.aic,
        "parameters": dict(result.parameter_estimates),
        "success": result.success
    }
    
    # Clear large result object
    del result
    
    return summary

# 3. Use context managers for temporary data
with pj.temporary_data_context(large_dataset) as temp_data:
    # Temporary operations on large dataset
    sample = temp_data.sample(n=1000)
    result = pj.fit_model(formula=formula_spec, data=sample)
    
    # temp_data automatically cleaned up when exiting context
```

## JAX Performance Tuning

### JAX Configuration

```python
import jax
import os

# Enable 64-bit precision (higher accuracy, more memory)
os.environ["JAX_ENABLE_X64"] = "True"

# Configure JAX memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Use 80% of available memory

# Verify configuration
print(f"JAX precision: {jax.config.jax_enable_x64}")
print(f"JAX devices: {jax.devices()}")

# Test JAX performance
x = jax.random.normal(jax.random.PRNGKey(42), (10000,))
%timeit jax.numpy.dot(x, x)  # Benchmark basic operations
```

### JIT Compilation Optimization

```python
# Understand JIT compilation overhead
@jax.jit
def optimized_likelihood(parameters, data_matrices):
    """JIT-compiled likelihood function."""
    # First call has compilation overhead
    # Subsequent calls are very fast
    pass

# Warm up JIT compilation with small data
small_data = data_context.sample(n_individuals=100)
small_matrices = model.build_design_matrices(formula_spec, small_data)

# Compile with small data (faster compilation)
_ = optimized_likelihood(initial_params, small_matrices)

# Now use with full data (already compiled)
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    jit_compile=True,              # Use JIT-compiled functions
    warm_start=True                # Skip recompilation
)
```

### JAX Memory Management

```python
# Monitor JAX memory usage
def jax_memory_usage():
    """Report JAX memory usage."""
    try:
        from jax.lib import xla_bridge
        backend = xla_bridge.get_backend()
        if hasattr(backend, 'live_buffers'):
            buffers = backend.live_buffers()
            total_bytes = sum(buf.size * buf.dtype.itemsize for buf in buffers)
            return f"JAX memory: {total_bytes / 1e9:.2f} GB ({len(buffers)} buffers)"
    except:
        pass
    return "JAX memory info not available"

print(f"Before fitting: {jax_memory_usage()}")

# Use JAX memory clearing
result = pj.fit_model(formula=formula_spec, data=data_context)

# Clear JAX memory manually if needed
jax.clear_caches()
print(f"After clearing caches: {jax_memory_usage()}")
```

## Benchmarking and Profiling

### Built-in Benchmarking

```python
# Run performance benchmarks
benchmark_results = pj.run_performance_benchmark(
    data_sizes=[100, 500, 1000, 5000],      # Different dataset sizes
    strategies=["lbfgs", "slsqp", "multi_start"],  # Different optimizers
    n_trials=5,                             # Repeated trials
    verbose=True
)

print("Benchmark Results:")
print(benchmark_results.summary())

# Plot performance curves
benchmark_results.plot_performance_curves()
```

### Custom Benchmarking

```python
import time
import numpy as np

def benchmark_optimization_strategy(strategy, data, formula, n_trials=5):
    """Benchmark a specific optimization strategy."""
    
    times = []
    successes = []
    aics = []
    
    for trial in range(n_trials):
        start_time = time.time()
        
        result = pj.fit_model(
            formula=formula,
            data=data,
            strategy=strategy
        )
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        successes.append(result.success)
        if result.success:
            aics.append(result.aic)
    
    return {
        "strategy": strategy,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "success_rate": np.mean(successes),
        "mean_aic": np.mean(aics) if aics else None
    }

# Benchmark different strategies
strategies = ["lbfgs", "slsqp", "multi_start"]
results = []

for strategy in strategies:
    benchmark = benchmark_optimization_strategy(strategy, data_context, formula_spec)
    results.append(benchmark)
    print(f"{strategy}: {benchmark['mean_time']:.2f}s ± {benchmark['std_time']:.2f}s")
```

### Memory Profiling

```python
# Use memory profiler for detailed analysis
from memory_profiler import profile

@profile
def memory_intensive_analysis():
    """Profile memory usage of analysis workflow."""
    
    # Load large dataset
    data = pj.load_data("large_dataset.csv")
    
    # Create complex formula
    formula = pj.create_formula_spec(
        phi="~1 + sex * age + region",
        p="~1 + effort + weather",
        f="~1 + habitat"
    )
    
    # Fit model with bootstrap
    result = pj.fit_model(
        formula=formula,
        data=data,
        bootstrap_confidence_intervals=True,
        bootstrap_config={"n_bootstrap": 1000}
    )
    
    return result

# Run with memory profiling
# python -m memory_profiler your_script.py
```

### Line Profiling

```python
# Use line profiler for detailed timing analysis
from line_profiler import LineProfiler

def detailed_timing_analysis():
    """Profile line-by-line timing."""
    
    profiler = LineProfiler()
    
    # Add functions to profile
    profiler.add_function(pj.fit_model)
    profiler.add_function(pj.PradelModel.log_likelihood)
    
    profiler.enable()
    
    # Run analysis
    result = pj.fit_model(formula=formula_spec, data=data_context)
    
    profiler.disable()
    
    # Print results
    profiler.print_stats()

# Run: kernprof -l -v your_script.py
```

## Performance Troubleshooting

### Common Performance Issues

#### 1. Slow Initial Model Fit

```python
# Problem: First model fit is very slow
# Cause: JAX compilation overhead

# Solution: Warm up JAX compilation
pj.warm_up_jax()  # Pre-compile common functions

# Or use small dataset for first fit
small_sample = data_context.sample(n_individuals=100)
_ = pj.fit_model(formula=formula_spec, data=small_sample)  # Compile functions

# Now full dataset will be fast
result = pj.fit_model(formula=formula_spec, data=data_context)
```

#### 2. Memory Errors with Large Datasets

```python
# Problem: Out of memory errors
# Solutions:

# 1. Use sampling
sample_data = pj.stratified_sample(data_context, n_samples=10000)

# 2. Reduce precision
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    precision="float32"  # Instead of float64
)

# 3. Disable design matrix caching
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    cache_design_matrices=False
)
```

#### 3. Optimization Not Converging

```python
# Problem: Optimization fails or takes too long
# Solutions:

# 1. Try multi-start optimization
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    strategy="multi_start",
    multi_start_attempts=20
)

# 2. Simplify model temporarily
simple_formula = pj.create_formula_spec(phi="~1", p="~1", f="~1")
simple_result = pj.fit_model(formula=simple_formula, data=data_context)

# Use simple model parameters as starting point
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    initial_parameters=simple_result.parameter_estimates
)

# 3. Check data quality
validation = data_context.validate()
if not validation.is_valid:
    print("Data issues:", validation.errors)
```

#### 4. Bootstrap Taking Too Long

```python
# Problem: Bootstrap confidence intervals very slow
# Solutions:

# 1. Reduce bootstrap samples
bootstrap_config = {
    "n_bootstrap": 200,     # Instead of 1000
    "parallel": True,
    "n_jobs": os.cpu_count()
}

# 2. Use faster bootstrap method
bootstrap_config = {
    "method": "basic",      # Instead of "bca"
    "parallel": True
}

# 3. Skip bootstrap for model development
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    bootstrap_confidence_intervals=False  # Skip for speed
)
```

### Performance Monitoring

```python
# Set up performance monitoring
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        return self
        
    def __exit__(self, *args):
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        elapsed = end_time - self.start_time
        memory_change = (end_memory - self.start_memory) / 1e9
        
        print(f"Elapsed time: {elapsed:.2f}s")
        print(f"Memory change: {memory_change:+.2f}GB")

# Use performance monitor
with PerformanceMonitor():
    result = pj.fit_model(formula=formula_spec, data=data_context)
```

## Best Practices

### Development Workflow

1. **Start Small**: Use small datasets during model development
2. **Simple First**: Begin with simple models, add complexity gradually
3. **Profile Early**: Identify performance bottlenecks early
4. **Cache Results**: Save intermediate results to avoid recomputation

```python
# Example development workflow
def efficient_development_workflow():
    # 1. Start with small sample
    sample_data = pj.stratified_sample(full_data, n_samples=1000)
    
    # 2. Develop model on sample
    simple_formula = pj.create_formula_spec(phi="~1", p="~1", f="~1")
    simple_result = pj.fit_model(formula=simple_formula, data=sample_data)
    
    # 3. Add complexity gradually
    complex_formula = pj.create_formula_spec(phi="~1 + sex + age", p="~1 + sex", f="~1")
    complex_result = pj.fit_model(
        formula=complex_formula,
        data=sample_data,
        initial_parameters=simple_result.parameter_estimates  # Warm start
    )
    
    # 4. Scale to full dataset
    final_result = pj.fit_model(
        formula=complex_formula,
        data=full_data,
        initial_parameters=complex_result.parameter_estimates
    )
    
    return final_result
```

### Production Optimization

```python
# Production-ready configuration
def production_optimization_config():
    return {
        "strategy": "multi_start",          # Robust optimization
        "multi_start_attempts": 10,         # Good balance of speed/robustness
        "max_iterations": 2000,             # Allow sufficient iterations
        "tolerance": 1e-6,                  # High precision
        "compute_standard_errors": True,    # Full statistical inference
        "confidence_intervals": True,       # Publication-ready results
        "bootstrap_confidence_intervals": False,  # Skip if time-critical
        "parallel": True,                   # Use all available cores
        "cache_design_matrices": True,      # Reuse matrices if possible
        "verbose": False                    # Reduce output in production
    }

# Apply production configuration
result = pj.fit_model(
    formula=formula_spec,
    data=data_context,
    **production_optimization_config()
)
```

### Resource Management

```python
# Resource-aware fitting
def resource_aware_fitting(formula, data):
    """Fit model with resource monitoring and adaptive configuration."""
    
    # Check available resources
    memory_gb = psutil.virtual_memory().total / 1e9
    cpu_cores = os.cpu_count()
    
    print(f"Available resources: {memory_gb:.1f}GB RAM, {cpu_cores} CPU cores")
    
    # Adaptive configuration based on resources
    if data.n_individuals > 10000 and memory_gb < 8:
        print("Large dataset + low memory: using sampling")
        data = pj.stratified_sample(data, n_samples=5000)
    
    if cpu_cores >= 4:
        parallel_config = {"parallel": True, "n_jobs": cpu_cores}
    else:
        parallel_config = {"parallel": False}
    
    # Select strategy based on problem size
    if data.n_individuals < 1000:
        strategy = "lbfgs"
    elif data.n_individuals < 10000:
        strategy = "multi_start"
    else:
        strategy = "adam"
    
    return pj.fit_model(
        formula=formula,
        data=data,
        strategy=strategy,
        **parallel_config
    )
```

---

**Related Documentation:**
- [Optimization Guide](optimization.md) - Detailed optimization strategy documentation
- [Architecture Guide](../development/architecture.md) - Understanding the performance-oriented design
- [Troubleshooting Guide](../user-guide/troubleshooting.md) - Solutions to common performance problems