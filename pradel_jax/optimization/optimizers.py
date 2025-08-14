"""
Industry-standard optimization implementations for pradel-jax.

Integrates with established optimization libraries:
- SciPy optimize for robust numerical optimization
- JAX optimizers for modern gradient-based methods  
- scikit-optimize for Bayesian optimization
- Optuna for hyperparameter optimization

Follows industry best practices from:
- scikit-learn estimator API patterns
- PyTorch/TensorFlow optimizer interfaces
- MLflow experiment tracking patterns
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import time
import warnings

# Industry-standard optimization libraries
try:
    import scipy.optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available - some optimizers will be disabled")

try:
    import optax  # JAX optimizers
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False
    warnings.warn("Optax not available - JAX optimizers will be limited")

try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .strategy import OptimizationConfig, OptimizationStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Standard optimization result following scipy.optimize conventions."""
    success: bool
    x: np.ndarray  # Final parameters
    fun: float     # Final objective value
    nit: int       # Number of iterations
    nfev: int      # Number of function evaluations
    message: str   # Convergence message
    jac: Optional[np.ndarray] = None  # Final gradient
    hess_inv: Optional[np.ndarray] = None  # Final Hessian inverse
    
    # Additional metadata
    optimization_time: float = 0.0
    strategy_used: str = ""
    convergence_history: Optional[List[float]] = None


class BaseOptimizer(ABC):
    """
    Base optimizer following scikit-learn estimator patterns.
    
    Provides consistent interface across all optimization strategies.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.is_fitted = False
        self.result_: Optional[OptimizationResult] = None
        
    @abstractmethod
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        hessian: Optional[Callable] = None
    ) -> OptimizationResult:
        """Minimize objective function."""
        pass
    
    def fit(self, objective: Callable, x0: np.ndarray, **kwargs) -> 'BaseOptimizer':
        """Scikit-learn style fit method."""
        self.result_ = self.minimize(objective, x0, **kwargs)
        self.is_fitted = True
        return self
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get optimizer parameters (scikit-learn compatible)."""
        return {
            'max_iter': self.config.max_iter,
            'tolerance': self.config.tolerance,
            'verbose': self.config.verbose
        }
    
    def set_params(self, **params) -> 'BaseOptimizer':
        """Set optimizer parameters (scikit-learn compatible)."""
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self


class ScipyLBFGSOptimizer(BaseOptimizer):
    """
    L-BFGS-B optimizer using scipy.optimize.minimize.
    
    Industry standard for bound-constrained optimization.
    Widely used in statistical software (R, MATLAB, etc).
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        if not HAS_SCIPY:
            raise ImportError("SciPy required for L-BFGS-B optimizer")
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using L-BFGS-B algorithm."""
        start_time = time.time()
        
        # Convert JAX functions to NumPy if needed
        if hasattr(objective, '__call__') and hasattr(objective, '__module__'):
            if 'jax' in str(objective.__module__):
                objective_np = lambda x: float(objective(x))
            else:
                objective_np = objective
        else:
            objective_np = objective
        
        # Handle gradient
        jac = None
        if gradient is not None:
            if hasattr(gradient, '__call__') and 'jax' in str(gradient.__module__):
                jac = lambda x: np.array(gradient(x))
            else:
                jac = gradient
        else:
            jac = '2-point'  # SciPy finite differences
        
        # Set up options following SciPy conventions
        options = {
            'maxiter': self.config.max_iter,
            'ftol': self.config.tolerance,
            'gtol': self.config.tolerance,
            'disp': self.config.verbose
        }
        
        try:
            # Run SciPy optimization
            scipy_result = scipy.optimize.minimize(
                fun=objective_np,
                x0=x0,
                method='L-BFGS-B',
                jac=jac,
                bounds=bounds,
                options=options
            )
            
            # Convert to our standard result format
            result = OptimizationResult(
                success=scipy_result.success,
                x=scipy_result.x,
                fun=scipy_result.fun,
                nit=scipy_result.nit,
                nfev=scipy_result.nfev,
                message=scipy_result.message,
                jac=getattr(scipy_result, 'jac', None),
                hess_inv=getattr(scipy_result, 'hess_inv', None),
                optimization_time=time.time() - start_time,
                strategy_used='scipy_lbfgs'
            )
            
            logger.info(f"L-BFGS-B converged in {result.nit} iterations, "
                       f"{result.nfev} function evaluations")
            
            return result
            
        except Exception as e:
            logger.error(f"L-BFGS-B optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float('inf'),
                nit=0,
                nfev=0,
                message=f"Optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used='scipy_lbfgs'
            )


class ScipySLSQPOptimizer(BaseOptimizer):
    """
    SLSQP optimizer using scipy.optimize.minimize.
    
    Sequential Least SQuares Programming - very robust for
    constrained optimization problems.
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        if not HAS_SCIPY:
            raise ImportError("SciPy required for SLSQP optimizer")
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        constraints: Optional[List[Dict]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using SLSQP algorithm."""
        start_time = time.time()
        
        # Convert functions as needed
        objective_np = objective
        if hasattr(objective, '__call__') and hasattr(objective, '__module__'):
            if 'jax' in str(objective.__module__):
                objective_np = lambda x: float(objective(x))
        
        # Handle gradient
        jac = None
        if gradient is not None:
            if hasattr(gradient, '__call__') and 'jax' in str(gradient.__module__):
                jac = lambda x: np.array(gradient(x))
            else:
                jac = gradient
        else:
            jac = '2-point'
        
        # SLSQP options
        options = {
            'maxiter': self.config.max_iter,
            'ftol': self.config.tolerance,
            'disp': self.config.verbose
        }
        
        try:
            scipy_result = scipy.optimize.minimize(
                fun=objective_np,
                x0=x0,
                method='SLSQP',
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            result = OptimizationResult(
                success=scipy_result.success,
                x=scipy_result.x,
                fun=scipy_result.fun,
                nit=scipy_result.nit,
                nfev=scipy_result.nfev,
                message=scipy_result.message,
                jac=getattr(scipy_result, 'jac', None),
                optimization_time=time.time() - start_time,
                strategy_used='scipy_slsqp'
            )
            
            logger.info(f"SLSQP converged in {result.nit} iterations")
            return result
            
        except Exception as e:
            logger.error(f"SLSQP optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float('inf'),
                nit=0,
                nfev=0,
                message=f"Optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used='scipy_slsqp'
            )


class JAXAdamOptimizer(BaseOptimizer):
    """
    Adam optimizer using JAX/Optax.
    
    Modern gradient-based optimization following PyTorch/TensorFlow patterns.
    Excellent for large-scale problems and GPU acceleration.
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        if not HAS_OPTAX:
            # Fallback to basic JAX implementation
            logger.warning("Optax not available, using basic JAX Adam")
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using Adam algorithm."""
        start_time = time.time()
        
        # Ensure functions are JAX-compatible
        if gradient is None:
            gradient = jit(grad(objective))
        
        if HAS_OPTAX:
            return self._minimize_with_optax(objective, gradient, x0, start_time)
        else:
            return self._minimize_basic_adam(objective, gradient, x0, start_time)
    
    def _minimize_with_optax(
        self, 
        objective: Callable, 
        gradient: Callable, 
        x0: np.ndarray,
        start_time: float
    ) -> OptimizationResult:
        """Use Optax Adam optimizer."""
        
        # Initialize Optax optimizer (follows PyTorch patterns)
        optimizer = optax.adam(learning_rate=self.config.learning_rate)
        opt_state = optimizer.init(x0)
        
        params = jnp.array(x0)
        loss_history = []
        
        @jit
        def update_step(params, opt_state):
            loss = objective(params)
            grads = gradient(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grads
        
        # Optimization loop
        for iteration in range(self.config.max_iter):
            params, opt_state, loss, grads = update_step(params, opt_state)
            loss_history.append(float(loss))
            
            # Check convergence
            grad_norm = jnp.linalg.norm(grads)
            if grad_norm < self.config.tolerance:
                logger.info(f"Adam converged at iteration {iteration}")
                break
            
            if self.config.verbose and iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: loss={loss:.6f}, grad_norm={grad_norm:.6e}")
        
        result = OptimizationResult(
            success=grad_norm < self.config.tolerance,
            x=np.array(params),
            fun=float(loss),
            nit=iteration + 1,
            nfev=iteration + 1,  # One function eval per iteration
            message="Adam optimization completed",
            jac=np.array(grads),
            optimization_time=time.time() - start_time,
            strategy_used='jax_adam',
            convergence_history=loss_history
        )
        
        return result
    
    def _minimize_basic_adam(
        self, 
        objective: Callable, 
        gradient: Callable, 
        x0: np.ndarray,
        start_time: float
    ) -> OptimizationResult:
        """Basic Adam implementation when Optax not available."""
        
        # Adam hyperparameters
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        lr = self.config.learning_rate
        
        params = jnp.array(x0)
        m = jnp.zeros_like(params)  # First moment
        v = jnp.zeros_like(params)  # Second moment
        
        loss_history = []
        
        for t in range(1, self.config.max_iter + 1):
            loss = objective(params)
            grads = gradient(params)
            loss_history.append(float(loss))
            
            # Adam updates
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * grads**2
            
            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            # Parameter update
            params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
            
            # Check convergence
            grad_norm = jnp.linalg.norm(grads)
            if grad_norm < self.config.tolerance:
                break
            
            if self.config.verbose and t % 100 == 0:
                logger.info(f"Iteration {t}: loss={loss:.6f}, grad_norm={grad_norm:.6e}")
        
        return OptimizationResult(
            success=grad_norm < self.config.tolerance,
            x=np.array(params),
            fun=float(loss),
            nit=t,
            nfev=t,
            message="Basic Adam optimization completed",
            jac=np.array(grads),
            optimization_time=time.time() - start_time,
            strategy_used='jax_adam_basic',
            convergence_history=loss_history
        )


class MultiStartOptimizer(BaseOptimizer):
    """
    Multi-start optimization for global optimization.
    
    Follows patterns from scikit-optimize and other global optimization libraries.
    Runs multiple local optimizations with different starting points.
    """
    
    def __init__(
        self, 
        config: OptimizationConfig, 
        base_optimizer: BaseOptimizer,
        n_starts: int = 5
    ):
        super().__init__(config)
        self.base_optimizer = base_optimizer
        self.n_starts = n_starts
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using multiple random starts."""
        start_time = time.time()
        
        results = []
        best_result = None
        best_fun = float('inf')
        
        # Generate starting points
        starting_points = self._generate_starting_points(x0, bounds)
        
        logger.info(f"Running multi-start optimization with {len(starting_points)} starts")
        
        for i, start_point in enumerate(starting_points):
            logger.debug(f"Start {i+1}/{len(starting_points)}")
            
            try:
                # Run local optimization
                result = self.base_optimizer.minimize(
                    objective=objective,
                    x0=start_point,
                    bounds=bounds,
                    **kwargs
                )
                
                results.append(result)
                
                # Track best result
                if result.success and result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Start {i+1} failed: {e}")
                continue
        
        if best_result is None:
            # All starts failed
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float('inf'),
                nit=0,
                nfev=0,
                message="All optimization starts failed",
                optimization_time=time.time() - start_time,
                strategy_used='multi_start'
            )
        
        # Enhance result with multi-start info
        best_result.optimization_time = time.time() - start_time
        best_result.strategy_used = 'multi_start'
        best_result.message += f" (best of {len(results)} starts)"
        
        success_rate = sum(r.success for r in results) / len(results)
        logger.info(f"Multi-start completed: {success_rate:.1%} success rate, "
                   f"best objective: {best_result.fun:.6f}")
        
        return best_result
    
    def _generate_starting_points(
        self, 
        x0: np.ndarray, 
        bounds: Optional[List[Tuple[float, float]]]
    ) -> List[np.ndarray]:
        """Generate diverse starting points."""
        points = [x0]  # Always include original starting point
        
        rng = np.random.RandomState(42)  # Reproducible
        
        for _ in range(self.n_starts - 1):
            if bounds is not None:
                # Sample uniformly within bounds
                point = np.array([
                    rng.uniform(low, high) for low, high in bounds
                ])
            else:
                # Sample around original point
                noise_scale = np.std(x0) if np.std(x0) > 0 else 1.0
                point = x0 + rng.normal(0, noise_scale, size=x0.shape)
            
            points.append(point)
        
        return points


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization using scikit-optimize.
    
    Excellent for expensive objective functions and automatic 
    hyperparameter tuning. Industry standard for ML hyperparameters.
    """
    
    def __init__(self, config: OptimizationConfig, n_calls: int = 50):
        super().__init__(config)
        self.n_calls = n_calls
        if not HAS_SKOPT:
            logger.warning("scikit-optimize not available, Bayesian optimization disabled")
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using Gaussian process-based Bayesian optimization."""
        
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        start_time = time.time()
        
        # Convert bounds to scikit-optimize format
        if bounds is None:
            # Create reasonable bounds around starting point
            scale = np.maximum(np.abs(x0), 1.0)
            bounds = [(x - 2*s, x + 2*s) for x, s in zip(x0, scale)]
        
        space = [Real(low, high) for low, high in bounds]
        
        try:
            # Run Bayesian optimization
            result = gp_minimize(
                func=objective,
                dimensions=space,
                x0=x0.tolist(),
                n_calls=self.n_calls,
                random_state=42
            )
            
            return OptimizationResult(
                success=True,
                x=np.array(result.x),
                fun=result.fun,
                nit=len(result.func_vals),
                nfev=len(result.func_vals),
                message="Bayesian optimization completed",
                optimization_time=time.time() - start_time,
                strategy_used='bayesian',
                convergence_history=result.func_vals
            )
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float('inf'),
                nit=0,
                nfev=0,
                message=f"Bayesian optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used='bayesian'
            )


class OptunaOptimizer(BaseOptimizer):
    """
    Hyperparameter optimization using Optuna.
    
    Industry-leading hyperparameter optimization framework used by
    many ML companies and research groups.
    """
    
    def __init__(self, config: OptimizationConfig, n_trials: int = 50):
        super().__init__(config)
        self.n_trials = n_trials
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, hyperparameter optimization disabled")
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using Optuna's advanced optimization algorithms."""
        
        if not HAS_OPTUNA:
            raise ImportError("Optuna required for hyperparameter optimization")
        
        start_time = time.time()
        
        if bounds is None:
            scale = np.maximum(np.abs(x0), 1.0)
            bounds = [(x - 2*s, x + 2*s) for x, s in zip(x0, scale)]
        
        def optuna_objective(trial):
            # Sample parameters
            x = np.array([
                trial.suggest_float(f'x_{i}', low, high) 
                for i, (low, high) in enumerate(bounds)
            ])
            return float(objective(x))
        
        try:
            # Create study with advanced sampler
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Optimize
            study.optimize(optuna_objective, n_trials=self.n_trials)
            
            # Extract best result
            best_params = np.array([study.best_params[f'x_{i}'] for i in range(len(x0))])
            
            return OptimizationResult(
                success=True,
                x=best_params,
                fun=study.best_value,
                nit=len(study.trials),
                nfev=len(study.trials),
                message="Optuna optimization completed",
                optimization_time=time.time() - start_time,
                strategy_used='optuna'
            )
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float('inf'),
                nit=0,
                nfev=0,
                message=f"Optuna optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used='optuna'
            )


# Factory function for creating optimizers
def create_optimizer(
    strategy: Union[OptimizationStrategy, str], 
    config: OptimizationConfig,
    **kwargs
) -> BaseOptimizer:
    """
    Factory function to create optimizer instances.
    
    Follows patterns from scikit-learn and other ML libraries.
    """
    
    if isinstance(strategy, str):
        try:
            strategy = OptimizationStrategy(strategy)
        except ValueError:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    # Standard optimizers
    optimizer_classes = {
        OptimizationStrategy.SCIPY_LBFGS: ScipyLBFGSOptimizer,
        OptimizationStrategy.SCIPY_SLSQP: ScipySLSQPOptimizer,
        OptimizationStrategy.JAX_ADAM: JAXAdamOptimizer,
    }
    
    if strategy in optimizer_classes:
        return optimizer_classes[strategy](config)
    elif strategy == OptimizationStrategy.MULTI_START:
        base_optimizer = ScipyLBFGSOptimizer(config)
        return MultiStartOptimizer(config, base_optimizer, **kwargs)
    
    # Large-scale optimizers - import here to avoid circular imports
    elif strategy in [OptimizationStrategy.MINI_BATCH_SGD, 
                      OptimizationStrategy.GPU_ACCELERATED,
                      OptimizationStrategy.DATA_PARALLEL,
                      OptimizationStrategy.STREAMING_ADAM]:
        try:
            from .large_scale import create_large_scale_optimizer, LargeScaleConfig
            
            # Convert config to large-scale config if needed
            if not isinstance(config, LargeScaleConfig):
                from .large_scale import LargeScaleConfig
                large_config = LargeScaleConfig(
                    max_iter=config.max_iter,
                    tolerance=config.tolerance,
                    learning_rate=getattr(config, 'learning_rate', 0.01),
                    verbose=config.verbose
                )
            else:
                large_config = config
                
            return create_large_scale_optimizer(strategy, large_config)
        except ImportError as e:
            raise ValueError(f"Large-scale optimizer dependencies not available: {e}")
    
    else:
        raise ValueError(f"Optimizer not implemented for strategy: {strategy}")


# Convenience functions following scikit-learn patterns

def minimize_with_strategy(
    strategy: Union[OptimizationStrategy, str],
    objective: Callable,
    x0: np.ndarray,
    config: Optional[OptimizationConfig] = None,
    **kwargs
) -> OptimizationResult:
    """
    Minimize objective using specified strategy.
    
    Convenience function following scipy.optimize.minimize patterns.
    """
    if config is None:
        config = OptimizationConfig()
    
    optimizer = create_optimizer(strategy, config, **kwargs)
    return optimizer.minimize(objective, x0, **kwargs)