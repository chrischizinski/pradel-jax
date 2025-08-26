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
    fun: float  # Final objective value
    nit: int  # Number of iterations
    nfev: int  # Number of function evaluations
    message: str  # Convergence message
    jac: Optional[np.ndarray] = None  # Final gradient
    hess_inv: Optional[np.ndarray] = None  # Final Hessian inverse

    # Additional metadata
    optimization_time: float = 0.0
    strategy_used: str = ""
    convergence_history: Optional[List[float]] = None

    # Statistical inference (computed from hess_inv when available)
    _parameter_names: Optional[List[str]] = None
    _sample_size: Optional[int] = None
    _objective_function: Optional[Callable] = None  # For fallback computations

    def set_statistical_info(
        self,
        parameter_names: List[str],
        sample_size: int,
        objective_function: Optional[Callable] = None,
    ) -> None:
        """Set information needed for statistical inference."""
        self._parameter_names = parameter_names
        self._sample_size = sample_size
        self._objective_function = objective_function

    @property
    def parameter_names(self) -> Optional[List[str]]:
        """Parameter names corresponding to x values."""
        return self._parameter_names

    @property
    def standard_errors(self) -> Optional[np.ndarray]:
        """Standard errors computed from Hessian inverse."""
        if self.hess_inv is None:
            return None

        try:
            # Handle different Hessian formats
            if isinstance(self.hess_inv, np.ndarray):
                if len(self.hess_inv.shape) == 2:
                    # Full Hessian inverse matrix
                    diagonal = np.diag(self.hess_inv)
                    # Standard errors are sqrt of diagonal elements
                    se = np.sqrt(np.maximum(diagonal, 0))  # Prevent sqrt of negative
                    return se
                elif len(self.hess_inv.shape) == 1:
                    # Diagonal approximation
                    return np.sqrt(np.maximum(self.hess_inv, 0))
            else:
                # Handle scipy LbfgsInvHessProduct or similar objects
                try:
                    # Get the shape
                    n = self.hess_inv.shape[0]

                    # Extract diagonal elements by multiplying with unit vectors (JAX-compatible)
                    diagonal_elements = []
                    for i in range(n):
                        unit_vector = np.zeros(n)
                        unit_vector = unit_vector.at[i].set(1.0) if hasattr(unit_vector, 'at') else np.array([1.0 if j == i else 0.0 for j in range(n)])
                        # Diagonal element is e_i^T * H^{-1} * e_i
                        diag_elem = unit_vector @ (self.hess_inv @ unit_vector)
                        diagonal_elements.append(diag_elem)
                    diagonal = np.array(diagonal_elements)

                    se = np.sqrt(
                        np.maximum(diagonal, 1e-12)
                    )  # Small minimum to prevent 0

                    # Check if this looks like a meaningful result (not unit approximation)
                    if not np.allclose(se, 1.0, rtol=1e-6):
                        return se
                    # If all values are ~1.0, continue to fallback methods

                except Exception as e:
                    # Last resort: try converting to dense matrix
                    try:
                        n = self.hess_inv.shape[0]
                        hess_inv_columns = []
                        for i in range(n):
                            unit_vector = np.array([1.0 if j == i else 0.0 for j in range(n)])
                            column = self.hess_inv @ unit_vector
                            hess_inv_columns.append(column)
                        hess_inv_dense = np.column_stack(hess_inv_columns)

                        diagonal = np.diag(hess_inv_dense)
                        se = np.sqrt(np.maximum(diagonal, 1e-12))

                        # Check if this looks meaningful
                        if not np.allclose(se, 1.0, rtol=1e-6):
                            return se
                        # If unit approximation, continue to fallback
                    except Exception:
                        pass
        except Exception:
            # Fallback if computation fails
            pass

        # Final fallback: finite difference Hessian if objective function available
        if self._objective_function is not None and self.x is not None:
            try:
                from .hessian_utils import (
                    compute_fallback_standard_errors,
                    validate_hessian_quality,
                )
                import logging

                logger = logging.getLogger(__name__)

                # First check if we should trust the existing Hessian
                if self.hess_inv is not None:
                    quality = validate_hessian_quality(self.hess_inv)
                    if not quality["meaningful"]:
                        # Use finite difference fallback
                        logger.info(
                            f"Using finite difference fallback for standard errors: {quality['issues']}"
                        )
                        return compute_fallback_standard_errors(
                            self._objective_function, self.x
                        )
                else:
                    # No Hessian provided, use finite difference
                    logger.info(
                        "No Hessian available, using finite difference for standard errors"
                    )
                    return compute_fallback_standard_errors(
                        self._objective_function, self.x
                    )
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Finite difference fallback failed: {e}")

        return None

    @property
    def log_likelihood(self) -> float:
        """Log-likelihood at optimum (negative of objective function)."""
        return -self.fun

    @property
    def aic(self) -> Optional[float]:
        """Akaike Information Criterion."""
        if self.x is None:
            return None

        k = len(self.x)  # Number of parameters
        log_lik = self.log_likelihood
        return 2 * k - 2 * log_lik

    @property
    def bic(self) -> Optional[float]:
        """Bayesian Information Criterion."""
        if self.x is None or self._sample_size is None:
            return None

        k = len(self.x)  # Number of parameters
        n = self._sample_size  # Sample size
        log_lik = self.log_likelihood
        return k * np.log(n) - 2 * log_lik

    @property
    def confidence_intervals(self, alpha: float = 0.05) -> Optional[np.ndarray]:
        """Wald confidence intervals for parameters."""
        se = self.standard_errors
        if se is None or self.x is None:
            return None

        from scipy import stats

        # Use t-distribution for small samples, normal for large
        if self._sample_size and self._sample_size - len(self.x) > 30:
            critical_value = stats.norm.ppf(1 - alpha / 2)
        else:
            # Conservative approach with t-distribution
            df = max(30, (self._sample_size or 100) - len(self.x))
            critical_value = stats.t.ppf(1 - alpha / 2, df)

        margin = critical_value * se
        lower = self.x - margin
        upper = self.x + margin

        return np.column_stack([lower, upper])

    def get_parameter_summary(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get comprehensive parameter summary with statistics."""
        if self.x is None:
            return None

        names = self.parameter_names or [f"param_{i}" for i in range(len(self.x))]
        se = self.standard_errors
        ci = self.confidence_intervals

        summary = {}
        for i, name in enumerate(names):
            param_info = {
                "estimate": float(self.x[i]),
                "log_likelihood": float(self.log_likelihood),
            }

            if se is not None:
                param_info["std_error"] = float(se[i])
                # Compute z-score / t-statistic
                if se[i] > 0:
                    param_info["z_score"] = float(self.x[i] / se[i])

            if ci is not None:
                param_info["ci_lower"] = float(ci[i, 0])
                param_info["ci_upper"] = float(ci[i, 1])

            summary[name] = param_info

        return summary


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
        hessian: Optional[Callable] = None,
    ) -> OptimizationResult:
        """Minimize objective function."""
        pass

    def fit(self, objective: Callable, x0: np.ndarray, **kwargs) -> "BaseOptimizer":
        """Scikit-learn style fit method."""
        self.result_ = self.minimize(objective, x0, **kwargs)
        self.is_fitted = True
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get optimizer parameters (scikit-learn compatible)."""
        return {
            "max_iter": self.config.max_iter,
            "tolerance": self.config.tolerance,
            "verbose": self.config.verbose,
        }

    def set_params(self, **params) -> "BaseOptimizer":
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
        **kwargs,
    ) -> OptimizationResult:
        """Minimize using L-BFGS-B algorithm."""
        start_time = time.time()

        # Convert JAX functions to NumPy if needed
        if hasattr(objective, "__call__") and hasattr(objective, "__module__"):
            if "jax" in str(objective.__module__):
                objective_np = lambda x: float(objective(x))
            else:
                objective_np = objective
        else:
            objective_np = objective

        # Handle gradient
        jac = None
        if gradient is not None:
            if hasattr(gradient, "__call__") and "jax" in str(gradient.__module__):
                jac = lambda x: np.array(gradient(x))
            else:
                jac = gradient
        else:
            jac = "2-point"  # SciPy finite differences

        # Set up options following SciPy conventions
        options = {
            "maxiter": self.config.max_iter,
            "ftol": self.config.tolerance,
            "gtol": self.config.tolerance,
            "disp": self.config.verbose,
        }

        try:
            # Run SciPy optimization
            scipy_result = scipy.optimize.minimize(
                fun=objective_np,
                x0=x0,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                options=options,
            )

            # Convert to our standard result format
            result = OptimizationResult(
                success=scipy_result.success,
                x=scipy_result.x,
                fun=scipy_result.fun,
                nit=scipy_result.nit,
                nfev=scipy_result.nfev,
                message=scipy_result.message,
                jac=getattr(scipy_result, "jac", None),
                hess_inv=getattr(scipy_result, "hess_inv", None),
                optimization_time=time.time() - start_time,
                strategy_used="scipy_lbfgs",
            )

            logger.info(
                f"L-BFGS-B converged in {result.nit} iterations, "
                f"{result.nfev} function evaluations"
            )

            return result

        except Exception as e:
            logger.error(f"L-BFGS-B optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float("inf"),
                nit=0,
                nfev=0,
                message=f"Optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used="scipy_lbfgs",
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
        **kwargs,
    ) -> OptimizationResult:
        """Minimize using SLSQP algorithm."""
        start_time = time.time()

        # Convert functions as needed
        objective_np = objective
        if hasattr(objective, "__call__") and hasattr(objective, "__module__"):
            if "jax" in str(objective.__module__):
                objective_np = lambda x: float(objective(x))

        # Handle gradient
        jac = None
        if gradient is not None:
            if hasattr(gradient, "__call__") and "jax" in str(gradient.__module__):
                jac = lambda x: np.array(gradient(x))
            else:
                jac = gradient
        else:
            jac = "2-point"

        # SLSQP options
        options = {
            "maxiter": self.config.max_iter,
            "ftol": self.config.tolerance,
            "disp": self.config.verbose,
        }

        try:
            scipy_result = scipy.optimize.minimize(
                fun=objective_np,
                x0=x0,
                method="SLSQP",
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )

            result = OptimizationResult(
                success=scipy_result.success,
                x=scipy_result.x,
                fun=scipy_result.fun,
                nit=scipy_result.nit,
                nfev=scipy_result.nfev,
                message=scipy_result.message,
                jac=getattr(scipy_result, "jac", None),
                optimization_time=time.time() - start_time,
                strategy_used="scipy_slsqp",
            )

            logger.info(f"SLSQP converged in {result.nit} iterations")
            return result

        except Exception as e:
            logger.error(f"SLSQP optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float("inf"),
                nit=0,
                nfev=0,
                message=f"Optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used="scipy_slsqp",
            )


class JAXAdamOptimizer(BaseOptimizer):
    """
    Adam optimizer using JAX/Optax.

    Modern gradient-based optimization following PyTorch/TensorFlow patterns.
    Excellent for large-scale problems and GPU acceleration.

    Note: For advanced features, use AdaptiveJAXAdamOptimizer from adaptive_adam module.
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
        **kwargs,
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
        self, objective: Callable, gradient: Callable, x0: np.ndarray, start_time: float
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
            loss_history.append(
                loss.item()
            )  # Use .item() instead of float() for JAX arrays

            # Check convergence
            grad_norm = jnp.linalg.norm(grads)
            if grad_norm < self.config.tolerance:
                logger.info(f"Adam converged at iteration {iteration}")
                break

            if self.config.verbose and iteration % 100 == 0:
                logger.info(
                    f"Iteration {iteration}: loss={loss:.6f}, grad_norm={grad_norm:.6e}"
                )

        result = OptimizationResult(
            success=grad_norm < self.config.tolerance,
            x=np.array(params),
            fun=loss.item(),
            nit=iteration + 1,
            nfev=iteration + 1,  # One function eval per iteration
            message="Adam optimization completed",
            jac=np.array(grads),
            optimization_time=time.time() - start_time,
            strategy_used="jax_adam",
            convergence_history=loss_history,
        )

        return result

    def _minimize_basic_adam(
        self, objective: Callable, gradient: Callable, x0: np.ndarray, start_time: float
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
            loss_history.append(
                loss.item()
            )  # Use .item() instead of float() for JAX arrays

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
                logger.info(
                    f"Iteration {t}: loss={loss:.6f}, grad_norm={grad_norm:.6e}"
                )

        return OptimizationResult(
            success=grad_norm < self.config.tolerance,
            x=np.array(params),
            fun=loss.item(),
            nit=t,
            nfev=t,
            message="Basic Adam optimization completed",
            jac=np.array(grads),
            optimization_time=time.time() - start_time,
            strategy_used="jax_adam_basic",
            convergence_history=loss_history,
        )


class JAXOPTLBFGSOptimizer(BaseOptimizer):
    """
    L-BFGS optimizer using JAXOpt.

    Modern JAX-native implementation with automatic differentiation.
    Often more stable than pure JAX Adam for optimization problems.
    """

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        try:
            global jaxopt
            import jaxopt

            self.has_jaxopt = True
        except ImportError:
            self.has_jaxopt = False
            logger.warning("JAXOpt not available, JAXOPT LBFGS optimizer unavailable")

    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Minimize using JAXOpt L-BFGS algorithm."""
        if not self.has_jaxopt:
            raise ImportError("JAXOpt required for JAXOPT LBFGS optimizer")

        start_time = time.time()

        try:
            # Convert to JAX arrays
            x0_jax = jnp.array(x0)

            # Create JAXOPT LBFGS solver
            solver = jaxopt.LBFGS(
                fun=objective,
                maxiter=self.config.max_iter,
                tol=self.config.tolerance,
                stepsize=getattr(self.config, "learning_rate", 1.0),
                linesearch="zoom",  # Use Zoom line search for robustness
            )

            # Run optimization
            if bounds is not None:
                # Use bounded LBFGS if bounds provided
                solver = jaxopt.LBFGSB(
                    fun=objective,
                    maxiter=self.config.max_iter,
                    tol=self.config.tolerance,
                )
                # Convert bounds for JAXOPT
                lower_bounds = jnp.array(
                    [b[0] if b[0] != -np.inf else -1e10 for b in bounds]
                )
                upper_bounds = jnp.array(
                    [b[1] if b[1] != np.inf else 1e10 for b in bounds]
                )

                result = solver.run(
                    init_params=x0_jax, bounds=(lower_bounds, upper_bounds)
                )
            else:
                result = solver.run(init_params=x0_jax)

            # Extract results
            success = result.state.error < self.config.tolerance
            final_params = np.array(result.params)
            final_objective = float(result.state.value)
            n_iterations = int(result.state.iter_num)
            n_function_evals = (
                int(result.state.num_fun_eval)
                if hasattr(result.state, "num_fun_eval")
                else n_iterations
            )

            # Get gradient if available
            final_gradient = None
            if hasattr(result.state, "grad"):
                final_gradient = np.array(result.state.grad)

            # Log results
            if self.config.verbose:
                logger.info(
                    f"JAXOPT L-BFGS completed: success={success}, "
                    f"iterations={n_iterations}, objective={final_objective:.6f}"
                )

            return OptimizationResult(
                success=success,
                x=final_params,
                fun=final_objective,
                nit=n_iterations,
                nfev=n_function_evals,
                message="JAXOPT L-BFGS optimization completed",
                jac=final_gradient,
                optimization_time=time.time() - start_time,
                strategy_used="jax_lbfgs",
            )

        except Exception as e:
            logger.error(f"JAXOPT L-BFGS optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float("inf"),
                nit=0,
                nfev=1,
                message=f"JAXOPT L-BFGS optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used="jax_lbfgs",
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
        n_starts: int = 5,
    ):
        super().__init__(config)
        self.base_optimizer = base_optimizer
        self.n_starts = n_starts

    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Minimize using multiple random starts."""
        start_time = time.time()

        results = []
        best_result = None
        best_fun = float("inf")

        # Generate starting points
        starting_points = self._generate_starting_points(x0, bounds)

        logger.info(
            f"Running multi-start optimization with {len(starting_points)} starts"
        )

        for i, start_point in enumerate(starting_points):
            logger.debug(f"Start {i+1}/{len(starting_points)}")

            try:
                # Run local optimization
                result = self.base_optimizer.minimize(
                    objective=objective, x0=start_point, bounds=bounds, **kwargs
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
                fun=float("inf"),
                nit=0,
                nfev=0,
                message="All optimization starts failed",
                optimization_time=time.time() - start_time,
                strategy_used="multi_start",
            )

        # Enhance result with multi-start info
        best_result.optimization_time = time.time() - start_time
        best_result.strategy_used = "multi_start"
        best_result.message += f" (best of {len(results)} starts)"

        success_rate = sum(r.success for r in results) / len(results)
        logger.info(
            f"Multi-start completed: {success_rate:.1%} success rate, "
            f"best objective: {best_result.fun:.6f}"
        )

        return best_result

    def _generate_starting_points(
        self, x0: np.ndarray, bounds: Optional[List[Tuple[float, float]]]
    ) -> List[np.ndarray]:
        """Generate diverse starting points."""
        points = [x0]  # Always include original starting point

        rng = np.random.RandomState(42)  # Reproducible

        for _ in range(self.n_starts - 1):
            if bounds is not None:
                # Sample uniformly within bounds
                point = np.array([rng.uniform(low, high) for low, high in bounds])
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
            logger.warning(
                "scikit-optimize not available, Bayesian optimization disabled"
            )

    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> OptimizationResult:
        """Minimize using Gaussian process-based Bayesian optimization."""

        if not HAS_SKOPT:
            raise ImportError("scikit-optimize required for Bayesian optimization")

        start_time = time.time()

        # Convert bounds to scikit-optimize format
        if bounds is None:
            # Create reasonable bounds around starting point
            scale = np.maximum(np.abs(x0), 1.0)
            bounds = [(x - 2 * s, x + 2 * s) for x, s in zip(x0, scale)]

        space = [Real(low, high) for low, high in bounds]

        try:
            # Run Bayesian optimization
            result = gp_minimize(
                func=objective,
                dimensions=space,
                x0=x0.tolist(),
                n_calls=self.n_calls,
                random_state=42,
            )

            return OptimizationResult(
                success=True,
                x=np.array(result.x),
                fun=result.fun,
                nit=len(result.func_vals),
                nfev=len(result.func_vals),
                message="Bayesian optimization completed",
                optimization_time=time.time() - start_time,
                strategy_used="bayesian",
                convergence_history=result.func_vals,
            )

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float("inf"),
                nit=0,
                nfev=0,
                message=f"Bayesian optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used="bayesian",
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
        **kwargs,
    ) -> OptimizationResult:
        """Minimize using Optuna's advanced optimization algorithms."""

        if not HAS_OPTUNA:
            raise ImportError("Optuna required for hyperparameter optimization")

        start_time = time.time()

        if bounds is None:
            scale = np.maximum(np.abs(x0), 1.0)
            bounds = [(x - 2 * s, x + 2 * s) for x, s in zip(x0, scale)]

        def optuna_objective(trial):
            # Sample parameters
            x = np.array(
                [
                    trial.suggest_float(f"x_{i}", low, high)
                    for i, (low, high) in enumerate(bounds)
                ]
            )
            return float(objective(x))

        try:
            # Create study with advanced sampler
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.MedianPruner(),
            )

            # Optimize
            study.optimize(optuna_objective, n_trials=self.n_trials)

            # Extract best result
            best_params = np.array(
                [study.best_params[f"x_{i}"] for i in range(len(x0))]
            )

            return OptimizationResult(
                success=True,
                x=best_params,
                fun=study.best_value,
                nit=len(study.trials),
                nfev=len(study.trials),
                message="Optuna optimization completed",
                optimization_time=time.time() - start_time,
                strategy_used="optuna",
            )

        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            return OptimizationResult(
                success=False,
                x=x0,
                fun=float("inf"),
                nit=0,
                nfev=0,
                message=f"Optuna optimization failed: {e}",
                optimization_time=time.time() - start_time,
                strategy_used="optuna",
            )


class HybridOptimizer(BaseOptimizer):
    """
    Hybrid optimization combining fast scipy methods with reliable multi-start fallback.

    Strategy:
    1. Quick attempt with L-BFGS-B (most reliable for well-behaved problems)
    2. If unsuccessful or poor convergence, fallback to multi-start optimization
    3. Optional final refinement with high-precision settings

    This approach balances speed and reliability, giving fast results for easy problems
    while ensuring robust convergence for difficult cases.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        quick_max_iter: int = 500,
        fallback_n_starts: int = 5,
        convergence_threshold: float = 1e-6,
        enable_refinement: bool = True,
    ):
        super().__init__(config)
        self.quick_max_iter = quick_max_iter
        self.fallback_n_starts = fallback_n_starts
        self.convergence_threshold = convergence_threshold
        self.enable_refinement = enable_refinement

        # Create optimizers for each phase
        quick_config = config.copy_with_overrides(
            max_iter=quick_max_iter,
            tolerance=convergence_threshold * 10,  # Relaxed for quick phase
        )
        self.quick_optimizer = ScipyLBFGSOptimizer(quick_config)

        fallback_config = config.copy_with_overrides(
            max_iter=config.max_iter, tolerance=convergence_threshold
        )
        base_optimizer = ScipyLBFGSOptimizer(fallback_config)
        self.fallback_optimizer = MultiStartOptimizer(
            fallback_config, base_optimizer, n_starts=fallback_n_starts
        )

        if enable_refinement:
            refinement_config = config.copy_with_overrides(
                max_iter=200, tolerance=config.tolerance  # Highest precision
            )
            self.refinement_optimizer = ScipySLSQPOptimizer(refinement_config)
        else:
            self.refinement_optimizer = None

    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Minimize using hybrid approach: quick attempt -> fallback -> refinement.
        """
        start_time = time.time()
        total_nfev = 0

        logger.info("Starting hybrid optimization: Phase 1 - Quick L-BFGS-B attempt")

        # Phase 1: Quick attempt with L-BFGS-B
        try:
            quick_result = self.quick_optimizer.minimize(
                objective=objective, x0=x0, bounds=bounds, gradient=gradient, **kwargs
            )
            total_nfev += quick_result.nfev

            # Check if quick result is satisfactory
            if self._is_satisfactory_result(quick_result):
                logger.info(
                    f"Phase 1 successful: converged in {quick_result.nit} iterations"
                )

                # Optional refinement
                if self.enable_refinement and self.refinement_optimizer:
                    logger.info("Phase 3 - Refinement with SLSQP")
                    refinement_result = self._refine_solution(
                        quick_result, objective, bounds, gradient, **kwargs
                    )
                    total_nfev += refinement_result.nfev

                    if (
                        refinement_result.success
                        and refinement_result.fun <= quick_result.fun
                    ):
                        logger.info(
                            f"Refinement improved solution: {quick_result.fun:.6f} -> {refinement_result.fun:.6f}"
                        )
                        return self._finalize_result(
                            refinement_result,
                            start_time,
                            total_nfev,
                            "hybrid_quick_refined",
                        )

                return self._finalize_result(
                    quick_result, start_time, total_nfev, "hybrid_quick"
                )

        except Exception as e:
            logger.warning(f"Phase 1 quick optimization failed: {e}")

        # Phase 2: Multi-start fallback
        logger.info("Phase 2 - Multi-start fallback optimization")
        try:
            fallback_result = self.fallback_optimizer.minimize(
                objective=objective, x0=x0, bounds=bounds, gradient=gradient, **kwargs
            )
            total_nfev += fallback_result.nfev

            if fallback_result.success:
                logger.info(
                    f"Phase 2 successful: converged with objective {fallback_result.fun:.6f}"
                )

                # Optional refinement
                if self.enable_refinement and self.refinement_optimizer:
                    logger.info("Phase 3 - Refinement with SLSQP")
                    refinement_result = self._refine_solution(
                        fallback_result, objective, bounds, gradient, **kwargs
                    )
                    total_nfev += refinement_result.nfev

                    if (
                        refinement_result.success
                        and refinement_result.fun <= fallback_result.fun
                    ):
                        logger.info(
                            f"Refinement improved solution: {fallback_result.fun:.6f} -> {refinement_result.fun:.6f}"
                        )
                        return self._finalize_result(
                            refinement_result,
                            start_time,
                            total_nfev,
                            "hybrid_multistart_refined",
                        )

                return self._finalize_result(
                    fallback_result, start_time, total_nfev, "hybrid_multistart"
                )
            else:
                logger.warning("Phase 2 multi-start optimization failed")

        except Exception as e:
            logger.error(f"Phase 2 multi-start optimization failed: {e}")

        # If all phases fail, return failure result
        logger.error("All hybrid optimization phases failed")
        return OptimizationResult(
            success=False,
            x=x0,
            fun=float("inf"),
            nit=0,
            nfev=total_nfev,
            message="All hybrid optimization phases failed",
            optimization_time=time.time() - start_time,
            strategy_used="hybrid_failed",
        )

    def _is_satisfactory_result(self, result: OptimizationResult) -> bool:
        """Check if optimization result meets satisfaction criteria."""
        if not result.success:
            return False

        # Check gradient norm if available
        if result.jac is not None:
            grad_norm = np.linalg.norm(result.jac)
            if grad_norm > self.convergence_threshold * 10:
                logger.debug(f"Quick result has high gradient norm: {grad_norm:.2e}")
                return False

        # Check for reasonable function value (not too high)
        if result.fun > 1e6:
            logger.debug(
                f"Quick result has suspiciously high objective value: {result.fun:.2e}"
            )
            return False

        return True

    def _refine_solution(
        self,
        initial_result: OptimizationResult,
        objective: Callable,
        bounds: Optional[List[Tuple[float, float]]],
        gradient: Optional[Callable],
        **kwargs,
    ) -> OptimizationResult:
        """Refine solution using high-precision optimizer."""
        try:
            return self.refinement_optimizer.minimize(
                objective=objective,
                x0=initial_result.x,
                bounds=bounds,
                gradient=gradient,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"Refinement phase failed: {e}")
            return initial_result  # Return original result if refinement fails

    def _finalize_result(
        self,
        result: OptimizationResult,
        start_time: float,
        total_nfev: int,
        strategy_used: str,
    ) -> OptimizationResult:
        """Finalize optimization result with hybrid-specific metadata."""
        return OptimizationResult(
            success=result.success,
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=total_nfev,  # Total function evaluations across all phases
            message=f"Hybrid optimization completed ({strategy_used})",
            jac=result.jac,
            hess_inv=result.hess_inv,
            optimization_time=time.time() - start_time,
            strategy_used=strategy_used,
            convergence_history=getattr(result, "convergence_history", None),
        )


# Factory function for creating optimizers
def create_optimizer(
    strategy: Union[OptimizationStrategy, str], config: OptimizationConfig, **kwargs
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

    # Apply strategy-specific default configurations for better convergence
    if strategy == OptimizationStrategy.JAX_LBFGS:
        # JAXOPT LBFGS works better with relaxed tolerance for statistical optimization
        config = config.copy_with_overrides(
            max_iter=max(config.max_iter, 2000), tolerance=max(config.tolerance, 1e-4)
        )
    elif strategy == OptimizationStrategy.JAX_ADAM:
        # JAX Adam needs lower learning rate and more iterations for statistical optimization
        config = config.copy_with_overrides(
            max_iter=max(config.max_iter, 5000),
            tolerance=max(config.tolerance, 1e-4),
            learning_rate=(
                min(config.learning_rate, 0.001)
                if config.learning_rate >= 0.01
                else config.learning_rate
            ),
        )
    elif strategy == OptimizationStrategy.JAX_ADAM_ADAPTIVE:
        # Adaptive Adam benefits from relaxed tolerance and sufficient iterations
        config = config.copy_with_overrides(
            max_iter=max(config.max_iter, 3000), tolerance=max(config.tolerance, 1e-4)
        )

    # Standard optimizers
    optimizer_classes = {
        OptimizationStrategy.SCIPY_LBFGS: ScipyLBFGSOptimizer,
        OptimizationStrategy.SCIPY_SLSQP: ScipySLSQPOptimizer,
        OptimizationStrategy.JAX_ADAM: JAXAdamOptimizer,
        OptimizationStrategy.JAX_LBFGS: JAXOPTLBFGSOptimizer,
    }

    if strategy in optimizer_classes:
        return optimizer_classes[strategy](config)
    elif strategy == OptimizationStrategy.MULTI_START:
        base_optimizer = ScipyLBFGSOptimizer(config)
        # Extract n_starts from kwargs if provided, ignore other kwargs like bounds
        n_starts = kwargs.get("n_starts", 5)
        return MultiStartOptimizer(config, base_optimizer, n_starts=n_starts)
    elif strategy == OptimizationStrategy.HYBRID:
        # Extract hybrid-specific parameters from kwargs
        quick_max_iter = kwargs.get("quick_max_iter", 500)
        fallback_n_starts = kwargs.get("fallback_n_starts", 5)
        convergence_threshold = kwargs.get("convergence_threshold", 1e-6)
        enable_refinement = kwargs.get("enable_refinement", True)
        return HybridOptimizer(
            config,
            quick_max_iter=quick_max_iter,
            fallback_n_starts=fallback_n_starts,
            convergence_threshold=convergence_threshold,
            enable_refinement=enable_refinement,
        )
    elif strategy == OptimizationStrategy.JAX_ADAM_ADAPTIVE:
        # Create adaptive Adam optimizer with optimized configuration
        from .adaptive_adam import (
            AdaptiveJAXAdamOptimizer,
            AdaptiveAdamConfig,
            get_optimized_adam_config,
        )

        # Convert to adaptive config if needed
        if isinstance(config, OptimizationConfig) and not isinstance(
            config, AdaptiveAdamConfig
        ):
            # Use problem characteristics if available in kwargs
            characteristics = kwargs.get("characteristics", None)
            if characteristics:
                adaptive_config = get_optimized_adam_config(characteristics)
                # Override with any specified config values
                adaptive_config.max_iter = config.max_iter
                adaptive_config.tolerance = config.tolerance
                adaptive_config.learning_rate = config.learning_rate
                adaptive_config.verbose = config.verbose
            else:
                # Create adaptive config from basic config
                adaptive_config = AdaptiveAdamConfig(
                    max_iter=config.max_iter,
                    tolerance=config.tolerance,
                    learning_rate=config.learning_rate,
                    init_scale=config.init_scale,
                    verbose=config.verbose,
                )
        else:
            adaptive_config = config

        return AdaptiveJAXAdamOptimizer(adaptive_config)

    # Large-scale optimizers - import here to avoid circular imports
    elif strategy in [
        OptimizationStrategy.MINI_BATCH_SGD,
        OptimizationStrategy.GPU_ACCELERATED,
        OptimizationStrategy.DATA_PARALLEL,
        OptimizationStrategy.STREAMING_ADAM,
    ]:
        try:
            from .large_scale import create_large_scale_optimizer, LargeScaleConfig

            # Convert config to large-scale config if needed
            if not isinstance(config, LargeScaleConfig):
                from .large_scale import LargeScaleConfig

                large_config = LargeScaleConfig(
                    max_iter=config.max_iter,
                    tolerance=config.tolerance,
                    learning_rate=getattr(config, "learning_rate", 0.01),
                    verbose=config.verbose,
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
    **kwargs,
) -> OptimizationResult:
    """
    Minimize objective using specified strategy.

    Convenience function following scipy.optimize.minimize patterns.
    """
    if config is None:
        config = OptimizationConfig()

    optimizer = create_optimizer(strategy, config, **kwargs)
    return optimizer.minimize(objective, x0, **kwargs)
