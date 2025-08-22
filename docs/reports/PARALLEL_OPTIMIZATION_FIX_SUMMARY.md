# Parallel Optimization DataContext Serialization Fix

## Issue Summary

The parallel optimization framework in `pradel_jax/optimization/parallel.py` had DataContext serialization issues that prevented proper cross-process communication. JAX arrays in the DataContext could not be pickled properly by Python's multiprocessing module, causing failures when worker processes tried to access the data.

## Root Cause

1. **JAX Array Serialization**: JAX arrays (`jax.numpy.ndarray`) cannot be directly pickled by Python's `pickle` module
2. **Cross-Process Communication**: The `ProcessPoolExecutor` uses pickle to serialize arguments passed to worker functions
3. **Complex Object Structure**: The `DataContext` class contains nested structures with JAX arrays and custom objects (`CovariateInfo`)

## Solution Implemented

### 1. Added Serialization Methods to DataContext

**File: `pradel_jax/data/adapters.py`**

```python
def to_dict(self) -> Dict[str, Any]:
    """Serialize DataContext to a pickle-safe dictionary."""
    # Convert JAX arrays to numpy arrays
    capture_matrix_np = np.array(self.capture_matrix)
    
    covariates_np = {}
    for name, value in self.covariates.items():
        if isinstance(value, jnp.ndarray):
            covariates_np[name] = np.array(value)
        else:
            covariates_np[name] = value
    
    # Convert CovariateInfo objects to dicts
    covariate_info_dict = {}
    for name, info in self.covariate_info.items():
        covariate_info_dict[name] = {
            'name': info.name,
            'dtype': info.dtype,
            'is_time_varying': info.is_time_varying,
            'is_categorical': info.is_categorical,
            'levels': info.levels,
            'time_occasions': info.time_occasions
        }
    
    return {
        'capture_matrix': capture_matrix_np,
        'covariates': covariates_np,
        'covariate_info': covariate_info_dict,
        'n_individuals': self.n_individuals,
        'n_occasions': self.n_occasions,
        'occasion_names': self.occasion_names,
        'individual_ids': self.individual_ids,
        'metadata': self.metadata
    }

@classmethod
def from_dict(cls, data_dict: Dict[str, Any]) -> 'DataContext':
    """Deserialize DataContext from dictionary."""
    # Convert numpy arrays back to JAX arrays
    capture_matrix = jnp.array(data_dict['capture_matrix'])
    
    covariates = {}
    for name, value in data_dict['covariates'].items():
        if isinstance(value, np.ndarray):
            covariates[name] = jnp.array(value)
        else:
            covariates[name] = value
    
    # Reconstruct CovariateInfo objects
    covariate_info = {}
    for name, info_dict in data_dict['covariate_info'].items():
        covariate_info[name] = CovariateInfo(**info_dict)
    
    return cls(
        capture_matrix=capture_matrix,
        covariates=covariates,
        covariate_info=covariate_info,
        n_individuals=data_dict['n_individuals'],
        n_occasions=data_dict['n_occasions'],
        occasion_names=data_dict['occasion_names'],
        individual_ids=data_dict['individual_ids'],
        metadata=data_dict['metadata']
    )
```

### 2. Updated Parallel Worker Function

**File: `pradel_jax/optimization/parallel.py`**

**Before:**
```python
def _fit_model_worker(args):
    model_spec, data_context_serialized, objective_func_name, bounds, strategy = args
    # Deserialize data context (passed as dict to avoid pickling issues)
    # This would need proper implementation based on DataContext structure
    # ...
```

**After:**
```python
def _fit_model_worker(args):
    model_spec, data_context_dict, objective_func_name, bounds, strategy = args
    
    try:
        # Deserialize data context from dict
        from ..data.adapters import DataContext
        data_context = DataContext.from_dict(data_context_dict)
        # ... rest of function uses properly reconstructed data_context
```

### 3. Updated Parallel Orchestrator

**File: `pradel_jax/optimization/parallel.py`**

**Before:**
```python
# Prepare arguments for worker processes
# Note: This is simplified - in practice you'd need proper DataContext serialization
worker_args = [
    (spec, data_context, "log_likelihood", bounds, strategy)
    for spec in batch_specs
]
```

**After:**
```python
# Prepare arguments for worker processes
# Serialize data context for cross-process communication
data_context_dict = data_context.to_dict()
worker_args = [
    (spec, data_context_dict, "log_likelihood", bounds, strategy)
    for spec in batch_specs
]
```

## Testing

### Test Results

1. **DataContext Serialization Test**: ✅ PASSED
   - Round-trip serialization preserves all data
   - JAX arrays properly converted to/from numpy
   - CovariateInfo objects correctly reconstructed

2. **Parallel Worker Simulation**: ✅ PASSED
   - DataContext survives process boundary
   - Design matrices build correctly in worker
   - Model operations work with reconstructed context

3. **Full Parallel Optimization**: ✅ PASSED
   - 4 models fitted successfully in parallel
   - Results include proper AIC values and parameter estimates
   - Checkpoint/resume functionality works
   - Performance logging shows ~0.4 models/second

### Example Test Output

```
Model Results:
------------------------------------------------------------
φ(.) p(.) f(.)       | AIC:   126.00 | Strategy: scipy_lbfgs
φ(sex) p(.) f(.)     | AIC:   127.98 | Strategy: scipy_lbfgs
φ(.) p(sex) f(.)     | AIC:   128.00 | Strategy: scipy_lbfgs
φ(sex) p(sex) f(.)   | AIC:   129.93 | Strategy: scipy_lbfgs

Best model: φ(.) p(.) f(.) (AIC: 126.00)
✅ Successful fits: 4/4
```

## Benefits

1. **True Parallelization**: Multiple models can now be fitted simultaneously across CPU cores
2. **Data Integrity**: All data structures are preserved exactly through serialization
3. **Scalability**: Framework can handle large model comparison studies
4. **Reliability**: Checkpoint/resume functionality for long-running jobs
5. **Performance**: Significant speedup for multiple model fitting workflows

## Usage

```python
from pradel_jax.optimization.parallel import ParallelOptimizer, ParallelModelSpec

# Create model specifications
model_specs = [...]

# Process data
data_context = pj.load_data("data.csv")

# Run parallel optimization
optimizer = ParallelOptimizer(n_workers=4)
results = optimizer.fit_models_parallel(
    model_specs=model_specs,
    data_context=data_context,
    strategy=OptimizationStrategy.HYBRID
)
```

## Files Modified

1. `pradel_jax/data/adapters.py` - Added serialization methods to DataContext
2. `pradel_jax/optimization/parallel.py` - Fixed worker function and orchestrator
3. Created comprehensive test suite for validation

## Status

✅ **COMPLETED** - DataContext serialization issues in parallel optimization framework have been resolved. The framework now supports reliable cross-process communication and can scale model fitting across multiple CPU cores.