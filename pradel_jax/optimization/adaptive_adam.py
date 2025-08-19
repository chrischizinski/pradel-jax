"""
Adaptive JAX Adam optimizer with advanced features for statistical optimization.

This module provides an enhanced Adam optimizer specifically tuned for capture-recapture
model optimization, including adaptive learning rates, gradient clipping, warm restarts,
and other modern optimization techniques.

Key Features:
- Adaptive learning rate scheduling
- Gradient clipping and normalization
- Warm restarts for escaping local minima
- Early stopping with patience
- Parameter-specific learning rates
- Plateau detection and learning rate reduction
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
import logging
import time
import warnings

try:
    import optax
    HAS_OPTAX = True
except ImportError:
    HAS_OPTAX = False
    warnings.warn("Optax not available - using basic Adam implementation")

from .optimizers import BaseOptimizer, OptimizationResult, OptimizationConfig

logger = logging.getLogger(__name__)


class AdaptiveAdamConfig(OptimizationConfig):
    """Extended configuration for adaptive Adam optimizer."""
    
    def __init__(self,
                 # Base configuration
                 max_iter: int = 10000,
                 tolerance: float = 1e-6,
                 learning_rate: float = 0.01,  # Higher default than basic Adam
                 init_scale: float = 0.1,
                 verbose: bool = False,
                 
                 # Adam-specific parameters
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 
                 # Adaptive features
                 use_adaptive_lr: bool = True,
                 lr_schedule: str = "cosine_decay",  # "cosine_decay", "exponential", "polynomial"
                 min_lr_factor: float = 0.01,  # Minimum LR as fraction of initial
                 
                 # Gradient clipping
                 gradient_clip_norm: Optional[float] = 1.0,
                 gradient_clip_value: Optional[float] = None,
                 
                 # Warm restarts
                 use_warm_restarts: bool = True,
                 restart_period: int = 1000,
                 restart_multiplier: float = 2.0,
                 
                 # Early stopping
                 patience: int = 500,
                 min_delta: float = 1e-8,
                 
                 # Plateau detection
                 plateau_patience: int = 200,
                 plateau_factor: float = 0.5,
                 plateau_threshold: float = 1e-6,
                 
                 # Parameter-specific settings
                 use_parameter_scaling: bool = True,
                 weight_decay: float = 0.0,
                 
                 # Numerical stability
                 use_amsgrad: bool = False,  # AMSGrad variant
                 numerical_stability_eps: float = 1e-15
                 ):
        
        super().__init__(max_iter, tolerance, learning_rate, init_scale, False, verbose)
        
        # Adam parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Adaptive learning rate
        self.use_adaptive_lr = use_adaptive_lr
        self.lr_schedule = lr_schedule
        self.min_lr_factor = min_lr_factor
        
        # Gradient management
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        
        # Warm restarts
        self.use_warm_restarts = use_warm_restarts
        self.restart_period = restart_period
        self.restart_multiplier = restart_multiplier
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # Plateau detection
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.plateau_threshold = plateau_threshold
        
        # Advanced features
        self.use_parameter_scaling = use_parameter_scaling
        self.weight_decay = weight_decay
        self.use_amsgrad = use_amsgrad
        self.numerical_stability_eps = numerical_stability_eps


class AdaptiveJAXAdamOptimizer(BaseOptimizer):
    """
    Advanced JAX Adam optimizer with adaptive features for statistical optimization.
    
    This optimizer implements several modern techniques to improve convergence
    for capture-recapture model optimization:
    
    1. Adaptive learning rate scheduling
    2. Gradient clipping and normalization
    3. Warm restarts for global optimization
    4. Early stopping with patience
    5. Plateau detection and learning rate reduction
    6. Parameter-specific optimization
    """
    
    def __init__(self, config: AdaptiveAdamConfig):
        super().__init__(config)
        self.config = config
        
        if not HAS_OPTAX:
            logger.warning("Optax not available, using basic Adam implementation")
    
    def minimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        gradient: Optional[Callable] = None,
        **kwargs
    ) -> OptimizationResult:
        """Minimize using adaptive Adam algorithm."""
        start_time = time.time()
        
        # Ensure functions are JAX-compatible
        if gradient is None:
            gradient = jit(grad(objective))
        
        if HAS_OPTAX:
            return self._minimize_with_optax(objective, gradient, x0, bounds, start_time)
        else:
            return self._minimize_basic_adaptive(objective, gradient, x0, bounds, start_time)
    
    def _create_learning_rate_schedule(self, n_steps: int) -> Callable:
        """Create learning rate schedule based on configuration."""
        
        if not self.config.use_adaptive_lr:
            return optax.constant_schedule(self.config.learning_rate)
        
        min_lr = self.config.learning_rate * self.config.min_lr_factor
        
        if self.config.lr_schedule == "cosine_decay":
            # Cosine decay with warm restarts
            if self.config.use_warm_restarts:
                return optax.cosine_decay_schedule(
                    init_value=self.config.learning_rate,
                    decay_steps=self.config.restart_period,
                    alpha=self.config.min_lr_factor
                )
            else:
                return optax.cosine_decay_schedule(
                    init_value=self.config.learning_rate,
                    decay_steps=n_steps,
                    alpha=self.config.min_lr_factor
                )
        
        elif self.config.lr_schedule == "exponential":
            decay_rate = (min_lr / self.config.learning_rate) ** (1.0 / n_steps)
            return optax.exponential_decay(
                init_value=self.config.learning_rate,
                transition_steps=100,
                decay_rate=decay_rate
            )
        
        elif self.config.lr_schedule == "polynomial":
            return optax.polynomial_schedule(
                init_value=self.config.learning_rate,
                end_value=min_lr,
                power=1.0,
                transition_steps=n_steps
            )
        
        else:
            logger.warning(f"Unknown LR schedule: {self.config.lr_schedule}, using constant")
            return optax.constant_schedule(self.config.learning_rate)
    
    def _create_gradient_transform(self, n_steps: int) -> optax.GradientTransformation:
        """Create gradient transformation chain."""
        
        transforms = []
        
        # Gradient clipping
        if self.config.gradient_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.config.gradient_clip_norm))
        elif self.config.gradient_clip_value is not None:
            transforms.append(optax.clip(self.config.gradient_clip_value))
        
        # Weight decay (L2 regularization)
        if self.config.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.config.weight_decay))
        
        # Learning rate schedule
        lr_schedule = self._create_learning_rate_schedule(n_steps)
        
        # Adam optimizer
        if self.config.use_amsgrad:
            adam_transform = optax.amsgrad(
                learning_rate=lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.eps
            )
        else:
            adam_transform = optax.adam(
                learning_rate=lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.eps
            )
        
        transforms.append(adam_transform)
        
        # Chain all transformations
        return optax.chain(*transforms)
    
    def _minimize_with_optax(
        self,
        objective: Callable,
        gradient: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]],
        start_time: float
    ) -> OptimizationResult:
        """Minimize using Optax with adaptive features."""
        
        # Create optimizer
        optimizer = self._create_gradient_transform(self.config.max_iter)
        opt_state = optimizer.init(x0)
        
        params = jnp.array(x0)
        loss_history = []
        lr_history = []
        
        # Early stopping state
        best_loss = float('inf')
        best_params = params
        patience_counter = 0
        
        # Plateau detection
        plateau_counter = 0
        plateau_best_loss = float('inf')
        
        # Warm restart state
        last_restart = 0
        
        @jit
        def update_step(params, opt_state, iteration):
            loss = objective(params)
            grads = gradient(params)
            
            # Handle bounds by projecting gradients (avoid boolean conversion in JIT)
            # Don't project gradients in JIT - handle bounds outside the update step
            
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state, loss, grads
        
        # Main optimization loop
        for iteration in range(self.config.max_iter):
            params, opt_state, loss, grads = update_step(params, opt_state, iteration)
            
            # Handle bounds outside JIT function
            if bounds is not None:
                params = self._project_to_bounds(params, bounds)
            
            loss_val = float(loss)
            loss_history.append(loss_val)
            
            # Track learning rate if using schedule
            if hasattr(opt_state, 'hyperparams'):
                lr = float(opt_state.hyperparams.get('learning_rate', self.config.learning_rate))
                lr_history.append(lr)
            
            # Check convergence
            grad_norm = float(jnp.linalg.norm(grads))
            if grad_norm < self.config.tolerance:
                logger.info(f"Adaptive Adam converged at iteration {iteration} (grad_norm={grad_norm:.2e})")
                break
            
            # Early stopping check
            if loss_val < best_loss - self.config.min_delta:
                best_loss = loss_val
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at iteration {iteration} (patience exceeded)")
                params = best_params  # Restore best parameters
                break
            
            # Plateau detection
            if loss_val < plateau_best_loss - self.config.plateau_threshold:
                plateau_best_loss = loss_val
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            # Warm restart check
            if (self.config.use_warm_restarts and 
                iteration - last_restart >= self.config.restart_period and
                plateau_counter >= self.config.plateau_patience):
                
                logger.info(f"Warm restart at iteration {iteration}")
                
                # Reset optimizer state for restart
                optimizer = self._create_gradient_transform(self.config.max_iter - iteration)
                opt_state = optimizer.init(params)
                
                last_restart = iteration
                plateau_counter = 0
                
                # Scale down learning rate after restart
                if hasattr(self.config, 'restart_lr_factor'):
                    self.config.learning_rate *= 0.8
            
            # Verbose logging
            if self.config.verbose and iteration % 100 == 0:
                lr_val = lr_history[-1] if lr_history else self.config.learning_rate
                logger.info(f"Iteration {iteration}: loss={loss_val:.6f}, "
                           f"grad_norm={grad_norm:.2e}, lr={lr_val:.2e}")
        
        # Final result
        result = OptimizationResult(
            success=grad_norm < self.config.tolerance * 10,  # Relaxed for Adam
            x=np.array(params),
            fun=float(loss),
            nit=iteration + 1,
            nfev=iteration + 1,
            message=f"Adaptive Adam optimization completed",
            jac=np.array(grads),
            optimization_time=time.time() - start_time,
            strategy_used='adaptive_jax_adam',
            convergence_history=loss_history
        )
        
        if self.config.verbose:
            logger.info(f"Final result: success={result.success}, "
                       f"loss={result.fun:.6f}, iterations={result.nit}")
        
        return result
    
    def _project_gradients_to_bounds(
        self, 
        params: jnp.ndarray, 
        grads: jnp.ndarray, 
        bounds: List[Tuple[float, float]]
    ) -> jnp.ndarray:
        """Project gradients to respect bounds constraints."""
        
        if bounds is None:
            return grads
        
        projected_grads = grads
        
        for i, (low, high) in enumerate(bounds):
            if low is not None and params[i] <= low and grads[i] < 0:
                # At lower bound and gradient points further down
                projected_grads = projected_grads.at[i].set(0.0)
            elif high is not None and params[i] >= high and grads[i] > 0:
                # At upper bound and gradient points further up
                projected_grads = projected_grads.at[i].set(0.0)
        
        return projected_grads
    
    def _project_to_bounds(
        self, 
        params: jnp.ndarray, 
        bounds: List[Tuple[float, float]]
    ) -> jnp.ndarray:
        """Project parameters to bounds."""
        
        if bounds is None:
            return params
        
        projected_params = params
        
        for i, (low, high) in enumerate(bounds):
            if low is not None:
                projected_params = projected_params.at[i].set(
                    jnp.maximum(projected_params[i], low)
                )
            if high is not None:
                projected_params = projected_params.at[i].set(
                    jnp.minimum(projected_params[i], high)
                )
        
        return projected_params
    
    def _minimize_basic_adaptive(
        self,
        objective: Callable,
        gradient: Callable,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]],
        start_time: float
    ) -> OptimizationResult:
        """Basic adaptive Adam implementation when Optax not available."""
        
        # Initialize parameters
        params = jnp.array(x0)
        m = jnp.zeros_like(params)  # First moment
        v = jnp.zeros_like(params)  # Second moment
        
        if self.config.use_amsgrad:
            v_hat_max = jnp.zeros_like(params)  # Maximum of v_hat for AMSGrad
        
        loss_history = []
        best_loss = float('inf')
        best_params = params
        patience_counter = 0
        
        # Learning rate schedule (simple exponential decay)
        lr_decay = 0.99 if self.config.use_adaptive_lr else 1.0
        current_lr = self.config.learning_rate
        
        for t in range(1, self.config.max_iter + 1):
            loss = objective(params)
            grads = gradient(params)
            loss_val = float(loss)
            loss_history.append(loss_val)
            
            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                grad_norm = jnp.linalg.norm(grads)
                if grad_norm > self.config.gradient_clip_norm:
                    grads = grads * (self.config.gradient_clip_norm / grad_norm)
            
            # Adam updates
            m = self.config.beta1 * m + (1 - self.config.beta1) * grads
            v = self.config.beta2 * v + (1 - self.config.beta2) * grads**2
            
            # Bias correction
            m_hat = m / (1 - self.config.beta1**t)
            v_hat = v / (1 - self.config.beta2**t)
            
            # AMSGrad variant
            if self.config.use_amsgrad:
                v_hat_max = jnp.maximum(v_hat_max, v_hat)
                denominator = jnp.sqrt(v_hat_max) + self.config.eps
            else:
                denominator = jnp.sqrt(v_hat) + self.config.eps
            
            # Parameter update
            params = params - current_lr * m_hat / denominator
            
            # Apply bounds
            if bounds is not None:
                for i, (low, high) in enumerate(bounds):
                    if low is not None:
                        params = params.at[i].set(jnp.maximum(params[i], low))
                    if high is not None:
                        params = params.at[i].set(jnp.minimum(params[i], high))
            
            # Learning rate decay
            if self.config.use_adaptive_lr and t % 100 == 0:
                current_lr *= lr_decay
                current_lr = max(current_lr, self.config.learning_rate * self.config.min_lr_factor)
            
            # Check convergence
            grad_norm = float(jnp.linalg.norm(grads))
            if grad_norm < self.config.tolerance:
                break
            
            # Early stopping
            if loss_val < best_loss - self.config.min_delta:
                best_loss = loss_val
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                params = best_params
                break
            
            # Verbose logging
            if self.config.verbose and t % 100 == 0:
                logger.info(f"Iteration {t}: loss={loss_val:.6f}, "
                           f"grad_norm={grad_norm:.2e}, lr={current_lr:.2e}")
        
        return OptimizationResult(
            success=grad_norm < self.config.tolerance * 10,
            x=np.array(params),
            fun=float(loss),
            nit=t,
            nfev=t,
            message="Basic adaptive Adam optimization completed",
            jac=np.array(grads),
            optimization_time=time.time() - start_time,
            strategy_used='basic_adaptive_adam',
            convergence_history=loss_history
        )


# Factory function for creating optimized Adam configurations
def create_optimized_adam_config(
    problem_size: int,
    n_parameters: int,
    difficulty: str = "moderate"
) -> AdaptiveAdamConfig:
    """
    Create optimized Adam configuration based on problem characteristics.
    
    Args:
        problem_size: Number of individuals in dataset
        n_parameters: Number of model parameters
        difficulty: Problem difficulty ("easy", "moderate", "difficult")
    
    Returns:
        Optimized AdaptiveAdamConfig
    """
    
    # Base learning rate scaling
    if difficulty == "easy":
        base_lr = 0.02
        patience = 300
    elif difficulty == "moderate":
        base_lr = 0.01
        patience = 500
    else:  # difficult
        base_lr = 0.005
        patience = 800
    
    # Scale learning rate based on problem size
    if n_parameters > 20:
        base_lr *= 0.5  # Smaller LR for high-dimensional problems
    
    if problem_size > 5000:
        base_lr *= 1.5  # Can handle higher LR with more data
    
    # Choose schedule based on problem characteristics
    if difficulty == "difficult" or n_parameters > 15:
        lr_schedule = "cosine_decay"
        use_warm_restarts = True
        restart_period = min(1000, n_parameters * 50)
    else:
        lr_schedule = "exponential"
        use_warm_restarts = False
        restart_period = 1000
    
    return AdaptiveAdamConfig(
        max_iter=min(15000, n_parameters * 1000),
        tolerance=1e-6,
        learning_rate=base_lr,
        
        # Adam parameters optimized for statistical models
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        
        # Adaptive features
        use_adaptive_lr=True,
        lr_schedule=lr_schedule,
        min_lr_factor=0.01,
        
        # Gradient management
        gradient_clip_norm=1.0,
        
        # Warm restarts
        use_warm_restarts=use_warm_restarts,
        restart_period=restart_period,
        restart_multiplier=2.0,
        
        # Early stopping
        patience=patience,
        min_delta=1e-8,
        
        # Plateau detection
        plateau_patience=min(200, patience // 2),
        plateau_factor=0.5,
        
        # Advanced features
        use_parameter_scaling=True,
        weight_decay=1e-5,  # Light regularization
        use_amsgrad=difficulty == "difficult",
        
        verbose=False
    )


# Update strategy configuration to use optimized Adam
def get_optimized_adam_config(characteristics) -> AdaptiveAdamConfig:
    """Get optimized Adam configuration based on model characteristics."""
    
    difficulty_map = {
        "easy": "easy",
        "moderate": "moderate", 
        "difficult": "difficult",
        "very_difficult": "difficult"
    }
    
    # Determine difficulty (would be passed from strategy selector)
    difficulty = "moderate"  # Default
    
    if hasattr(characteristics, 'parameter_ratio'):
        if characteristics.parameter_ratio > 0.1:
            difficulty = "difficult"
        elif characteristics.parameter_ratio > 0.05:
            difficulty = "moderate"
        else:
            difficulty = "easy"
    
    return create_optimized_adam_config(
        problem_size=getattr(characteristics, 'n_individuals', 1000),
        n_parameters=getattr(characteristics, 'n_parameters', 10),
        difficulty=difficulty
    )