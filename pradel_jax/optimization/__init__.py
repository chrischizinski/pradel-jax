"""
Optimization strategies for pradel-jax.

A comprehensive optimization framework providing:
- Intelligent strategy selection based on problem characteristics
- Industry-standard optimization algorithms (SciPy, JAX, Optuna, etc.)
- Comprehensive performance monitoring and experiment tracking
- Robust fallback mechanisms and error handling
- High-level orchestration for complex optimization workflows

Main entry points:
- optimize_model(): High-level optimization interface
- compare_optimization_strategies(): Strategy comparison and benchmarking
- OptimizationOrchestrator(): Full-featured orchestration class

Example usage:
    from pradel_jax.optimization import optimize_model
    
    response = optimize_model(
        objective_function=my_likelihood,
        initial_parameters=initial_params,
        context=model_context,
        bounds=parameter_bounds
    )
    
    if response.success:
        print(f"Optimized parameters: {response.result.x}")
        print(f"Final objective: {response.result.fun}")
"""

# Core strategy and configuration classes
from .strategy import (
    OptimizationStrategy,
    OptimizationConfig,
    StrategyRecommendation,
    StrategySelector,
    ProblemDifficulty,
    ModelCharacteristics,
    auto_optimize,
    recommend_strategy,
    diagnose_optimization_difficulty
)

# Optimizer implementations
from .optimizers import (
    BaseOptimizer,
    ScipyLBFGSOptimizer,
    ScipySLSQPOptimizer,
    JAXAdamOptimizer,
    MultiStartOptimizer,
    BayesianOptimizer,
    OptunaOptimizer,
    OptimizationResult,
    create_optimizer,
    minimize_with_strategy
)

# Monitoring and tracking
from .monitoring import (
    OptimizationMetrics,
    OptimizationSession,
    PerformanceMonitor,
    ExperimentTracker,
    OptimizationProfiler,
    start_monitoring_session,
    log_optimization_metrics,
    end_monitoring_session,
    optimization_experiment,
    create_optimization_report
)

# High-level orchestration
from .orchestrator import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationOrchestrator,
    optimize_model,
    compare_optimization_strategies
)

# Large-scale optimization
from .large_scale import (
    LargeScaleConfig,
    LargeScaleStrategySelector,
    MiniBatchSGDOptimizer,
    GPUAcceleratedOptimizer,
    DistributedOptimizer,
    optimize_large_dataset,
    create_large_scale_optimizer
)

# Main exports - the recommended public API
__all__ = [
    # High-level functions (recommended entry points)
    'optimize_model',
    'compare_optimization_strategies',
    'recommend_strategy',
    'diagnose_optimization_difficulty',
    
    # Core classes
    'OptimizationOrchestrator',
    'OptimizationRequest', 
    'OptimizationResponse',
    'OptimizationResult',
    'StrategyRecommendation',
    
    # Strategy and configuration
    'OptimizationStrategy',
    'OptimizationConfig',
    'ProblemDifficulty',
    'ModelCharacteristics',
    
    # Optimizers
    'BaseOptimizer',
    'ScipyLBFGSOptimizer',
    'ScipySLSQPOptimizer', 
    'JAXAdamOptimizer',
    'MultiStartOptimizer',
    'create_optimizer',
    'minimize_with_strategy',
    
    # Monitoring
    'PerformanceMonitor',
    'ExperimentTracker',
    'OptimizationMetrics',
    'optimization_experiment',
    'create_optimization_report',
    
    # Advanced features
    'auto_optimize',
    'BayesianOptimizer',
    'OptunaOptimizer',
    
    # Large-scale optimization
    'optimize_large_dataset',
    'LargeScaleConfig',
    'LargeScaleStrategySelector',
    'MiniBatchSGDOptimizer',
    'GPUAcceleratedOptimizer',
    'DistributedOptimizer'
]