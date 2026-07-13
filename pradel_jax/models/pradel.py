"""
Pradel model implementation for pradel-jax.

Implements the Pradel capture-recapture model with JAX-based likelihood computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .base import CaptureRecaptureModel, ModelType, ModelResult
from ..formulas.spec import FormulaSpec, ParameterFormula, ParameterType
from ..formulas.design_matrix import DesignMatrixInfo, build_design_matrix
from ..data.adapters import DataContext
from ..core.exceptions import ModelSpecificationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@jax.jit
def logit(x: jnp.ndarray) -> jnp.ndarray:
    """Logit link function."""
    return jnp.log(x / (1 - x))


@jax.jit
def inv_logit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse logit (sigmoid) function."""
    return jax.nn.sigmoid(x)


@jax.jit
def log_link(x: jnp.ndarray) -> jnp.ndarray:
    """Log link function."""
    return jnp.log(x)


@jax.jit
def exp_link(x: jnp.ndarray) -> jnp.ndarray:
    """Exponential (inverse log) function."""
    return jnp.exp(x)


@jax.jit
def _log_beta_prior(
    probabilities: jnp.ndarray, alpha: float, beta: float, epsilon: float = 1e-6
) -> jnp.ndarray:
    """Log-prior contribution for a Beta(alpha, beta) distribution."""
    clipped_probs = jnp.clip(probabilities, epsilon, 1.0 - epsilon)
    return jnp.sum(
        (alpha - 1.0) * jnp.log(clipped_probs)
        + (beta - 1.0) * jnp.log1p(-clipped_probs)
    )


@jax.jit
def _log_lognormal_prior(
    values: jnp.ndarray,
    mode: float,
    sigma: float,
    epsilon: float = 1e-12,
) -> jnp.ndarray:
    """Log-prior contribution for a log-normal distribution parameterised by its mode."""
    clipped = jnp.clip(values, epsilon, jnp.inf)
    log_values = jnp.log(clipped)

    mu = jnp.log(mode) + sigma**2
    log_norm_const = jnp.log(sigma) + 0.5 * jnp.log(2.0 * jnp.pi)

    return jnp.sum(
        -0.5 * ((log_values - mu) / sigma) ** 2 - log_values - log_norm_const
    )


@jax.jit
def calculate_seniority_gamma(phi: float, f: float) -> float:
    """
    Calculate seniority probability γ from Pradel (1996).

    γ = φ / λ = φ / (φ + f)

    γ is the probability that an individual present at occasion i was already
    present (had not just recruited) at occasion i-1. This is the reverse-time
    analogue of survival φ and is what drives the recruitment side of the
    Pradel temporal-symmetry likelihood.
    """
    return phi / (phi + f)


@jax.jit
def _affine_iterate(x0: float, rate: float, p: float, n_steps: float) -> float:
    """
    Closed-form value of the affine recursion x_j = (1 - rate) + rate*(1-p)*x_{j-1}.

    This is the recursion shared by the two Pradel "tail" probabilities:

    - χ (chi), probability an individual is not detected again after its last
      capture: rate = φ, iterated over the occasions *after* the last capture.
      χ_0 = 1, χ_j = (1-φ) + φ(1-p) χ_{j-1}.

    - ξ (xi), probability of the (unobserved) history *before* first capture
      given the individual is present-and-first-detected at that occasion:
      rate = γ, iterated over the occasions *before* the first capture.
      ξ_0 = 1, ξ_j = (1-γ) + γ(1-p) ξ_{j-1}.

    The recursion is affine (x_j = a + b x_{j-1}, a = 1-rate, b = rate(1-p)),
    so the j-fold iterate starting from x0 = 1 has the closed form
        x_j = x* + b^j (x0 - x*),   x* = a / (1 - b),
    which is exact and avoids any data-dependent Python control flow.
    """
    a = 1.0 - rate
    b = rate * (1.0 - p)
    # 1 - b = 1 - rate(1-p) > 0 for rate in [0,1], p in [0,1] (only degenerate
    # at rate=1, p=0, which the parameter bounds/links exclude). Guard anyway.
    fixed_point = a / jnp.maximum(1.0 - b, 1e-12)
    return fixed_point + (b**n_steps) * (x0 - fixed_point)


@jax.jit
def _pradel_individual_likelihood(
    capture_history: jnp.ndarray, phi: float, p: float, f: float
) -> float:
    """
    Pradel (1996) individual log-likelihood contribution (temporal-symmetry form).

    Pradel's key insight is that a capture history contains information about both
    survival (reading time forwards) and recruitment/seniority (reading time
    backwards). The likelihood is therefore the product of a forward Cormack-Jolly-
    Seber (CJS) likelihood and a reverse-time CJS likelihood over the same history,
    sharing the detection probability p:

        P(h) = L_forward(φ, p) · L_reverse(γ, p)

    with, for a history first captured at occasion e and last captured at l,

        γ = φ/λ,   λ = φ + f            (seniority / population growth rate)

        L_forward = [Π_{i=e}^{l-1} φ] · [Π_{i=e+1}^{l} p^{h_i}(1-p)^{1-h_i}] · χ_l
        L_reverse = [Π_{i=e}^{l-1} γ] · [Π_{i=e}^{l-1} p^{h_i}(1-p)^{1-h_i}] · ξ_e

    where χ_l is the probability of not being detected after l (rate φ) and ξ_e is
    its reverse-time analogue before e (rate γ), both closed-form iterates of
    :func:`_affine_iterate`. Detections strictly inside (e, l) enter both products
    (the temporal-symmetry double-use); the forward product uses the detection at
    l and the reverse product the detection at e. The likelihood is conditional on
    each individual being captured at least once.

    Args:
        capture_history: Binary array (1=captured, 0=not) for one individual.
        phi: Survival probability (constant across occasions).
        p: Detection probability (constant across occasions).
        f: Per-capita recruitment rate (constant across occasions).

    Returns:
        Log-likelihood contribution for this individual. Individuals that are
        never captured are not part of the (conditional) Pradel likelihood and
        contribute 0.
    """
    n_occasions = len(capture_history)
    epsilon = 1e-12

    total_captures = jnp.sum(capture_history)
    indices = jnp.arange(n_occasions)

    # First and last capture occasions (0-based). Defined only when captured;
    # guarded by the final jnp.where for never-captured individuals.
    first_capture = jnp.min(jnp.where(capture_history == 1, indices, n_occasions))
    last_capture = jnp.max(jnp.where(capture_history == 1, indices, -1))

    lambda_pop = phi + f  # λ = φ + f
    gamma = phi / jnp.maximum(lambda_pop, epsilon)  # γ = φ/λ

    log_p = jnp.log(jnp.maximum(p, epsilon))
    log_1mp = jnp.log(jnp.maximum(1.0 - p, epsilon))
    log_phi = jnp.log(jnp.maximum(phi, epsilon))
    log_gamma = jnp.log(jnp.maximum(gamma, epsilon))

    # Per-occasion detection log-probability, h_i log p + (1-h_i) log(1-p).
    det_logprob = capture_history * log_p + (1.0 - capture_history) * log_1mp

    # Forward CJS detections cover occasions (e, l]; reverse CJS detections cover
    # [e, l). Interior detections (strictly between e and l) are used by both.
    fwd_mask = (indices > first_capture) & (indices <= last_capture)
    rev_mask = (indices >= first_capture) & (indices < last_capture)
    det_forward = jnp.sum(jnp.where(fwd_mask, det_logprob, 0.0))
    det_reverse = jnp.sum(jnp.where(rev_mask, det_logprob, 0.0))

    # Survival (forward) and seniority (reverse) each act over the l-e intervals.
    n_intervals = (last_capture - first_capture).astype(jnp.float32)
    survival_seniority = n_intervals * (log_phi + log_gamma)

    # ξ over the n_before occasions preceding first capture (reverse-time, rate γ)
    n_before = first_capture.astype(jnp.float32)
    xi = _affine_iterate(1.0, gamma, p, n_before)

    # χ over the n_after occasions following last capture (forward, rate φ)
    n_after = (n_occasions - 1 - last_capture).astype(jnp.float32)
    chi = _affine_iterate(1.0, phi, p, n_after)

    captured_ll = (
        survival_seniority
        + det_forward
        + det_reverse
        + jnp.log(jnp.maximum(xi, epsilon))
        + jnp.log(jnp.maximum(chi, epsilon))
    )

    # Never-captured individuals are not part of the conditional likelihood.
    return jnp.where(total_captures > 0, captured_ll, 0.0)


@jax.jit
def _pradel_vectorized_likelihood(
    phi: jnp.ndarray, p: jnp.ndarray, f: jnp.ndarray, capture_matrix: jnp.ndarray
) -> float:
    """
    JIT-compiled vectorized Pradel log-likelihood computation.

    FIXED: Now properly handles individual-specific parameters for gradient computation.
    For intercept-only models, all individuals have same parameters but gradients
    must be computed correctly through all parameter values.
    """
    n_individuals, n_occasions = capture_matrix.shape

    # Use vmap to vectorize individual likelihood computation
    # Pass individual-specific parameters (phi[i], p[i], f[i]) for each individual i
    individual_likelihoods = jax.vmap(
        lambda i: _pradel_individual_likelihood(capture_matrix[i], phi[i], p[i], f[i])
    )(jnp.arange(n_individuals))

    # Sum across all individuals
    return jnp.sum(individual_likelihoods)


class PradelModel(CaptureRecaptureModel):
    """
    Pradel capture-recapture model implementation.

    The Pradel model estimates:
    - φ (phi): Apparent survival probability
    - p: Detection/capture probability
    - f: Per-capita recruitment rate

    Uses logit link for φ and p (bounded 0-1), log link for f (positive).
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.PRADEL,
        boundary_prior_strength: float = 0.0,
        boundary_prior_alpha: float = 2.0,
        boundary_prior_beta: float = 2.0,
        recruitment_prior_strength: float = 0.0,
        recruitment_prior_mode: float = 0.05,
        recruitment_prior_sigma: float = 0.75,
    ):
        # NOTE: Priors are OFF by default so that log_likelihood() returns the
        # true Pradel log-likelihood and AIC/BIC/inference are valid. The soft
        # Beta/log-normal penalties remain available as an opt-in regularizer
        # (set the *_prior_strength arguments > 0), but when enabled the value
        # returned by log_likelihood is a penalized (MAP) objective, not the MLE
        # log-likelihood — do not use it for AIC/likelihood-ratio comparisons.
        super().__init__(model_type)
        self.parameter_order = ["phi", "p", "f"]
        self.boundary_prior_strength = float(boundary_prior_strength)
        self.boundary_prior_alpha = float(boundary_prior_alpha)
        self.boundary_prior_beta = float(boundary_prior_beta)
        self.recruitment_prior_strength = float(recruitment_prior_strength)
        self.recruitment_prior_mode = float(recruitment_prior_mode)
        self.recruitment_prior_sigma = float(recruitment_prior_sigma)

    def build_design_matrices(
        self, formula_spec: FormulaSpec, data_context: DataContext
    ) -> Dict[str, DesignMatrixInfo]:
        """Build design matrices for Pradel model parameters."""
        self.logger.debug("Building design matrices for Pradel model")

        # Get number of occasions for time-varying parameters
        n_occasions = data_context.n_occasions

        design_matrices = {}

        # Build design matrix for each parameter
        for param_name in self.parameter_order:
            param_formula = getattr(formula_spec, param_name)
            if param_formula is None:
                raise ModelSpecificationError(
                    formula=f"Missing {param_name} formula in specification",
                    suggestions=[
                        f"Pradel models require {param_name} formula",
                        "Provide all three formulas: phi, p, f",
                    ],
                )

            # For time-varying parameters, we might need different handling
            # For now, treat all as individual-level
            design_info = build_design_matrix(param_formula, data_context)
            design_matrices[param_name] = design_info

            self.logger.debug(
                f"Built {param_name} design matrix: {design_info.matrix.shape} "
                f"({design_info.parameter_count} parameters)"
            )

        return design_matrices

    def get_parameter_bounds(
        self, data_context: DataContext, design_matrices: Dict[str, DesignMatrixInfo]
    ) -> List[tuple]:
        """
        Get parameter bounds for optimization.

        FIXED: Use biologically reasonable bounds together with a soft boundary prior.
        """
        bounds = []

        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count

            if param_name in ["phi", "p"]:
                # Logit-scale bounds for survival/detection probabilities
                # Allow probabilities from 0.001 to 0.999 to avoid extreme logits
                param_bounds = [(logit(0.001), logit(0.999))] * n_params
            elif param_name == "f":
                # Log-scale bounds for recruitment rate
                # Allow recruitment from 1e-8 to 10.0 (wide range while avoiding underflow)
                param_bounds = [(log_link(1e-8), log_link(10.0))] * n_params
            else:
                # Default bounds (should not occur for Pradel model)
                param_bounds = [(-5.0, 5.0)] * n_params

            bounds.extend(param_bounds)

        return bounds

    def get_initial_parameters(
        self, data_context: DataContext, design_matrices: Dict[str, DesignMatrixInfo]
    ) -> jnp.ndarray:
        """Get initial parameter values based on data."""
        initial_params = []

        # Calculate simple empirical estimates for initialization
        capture_matrix = data_context.capture_matrix

        # Empirical capture probability (proportion of 1s)
        p_empirical = jnp.mean(capture_matrix)
        p_logit = logit(jnp.clip(p_empirical, 0.01, 0.99))

        # Empirical survival (naive estimate from consecutive captures)
        phi_empirical = self._estimate_empirical_survival(capture_matrix)
        phi_logit = logit(jnp.clip(phi_empirical, 0.01, 0.99))

        # Empirical recruitment (simple estimate)
        f_empirical = self._estimate_empirical_recruitment(capture_matrix)
        f_log = log_link(jnp.maximum(f_empirical, 0.01))

        self.logger.debug(
            f"Empirical estimates: p={float(p_empirical):.3f}, phi={float(phi_empirical):.3f}, f={float(f_empirical):.3f}"
        )

        # Initialize parameters for each design matrix
        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count

            if param_name == "phi":
                if design_info.has_intercept:
                    # Intercept gets empirical estimate, others get small non-zero values
                    params = jnp.concatenate(
                        [jnp.array([phi_logit]), jnp.ones(n_params - 1) * 0.1]
                    )
                else:
                    params = jnp.ones(n_params) * phi_logit

            elif param_name == "p":
                if design_info.has_intercept:
                    params = jnp.concatenate(
                        [jnp.array([p_logit]), jnp.ones(n_params - 1) * 0.1]
                    )
                else:
                    params = jnp.ones(n_params) * p_logit

            elif param_name == "f":
                if design_info.has_intercept:
                    params = jnp.concatenate(
                        [jnp.array([f_log]), jnp.ones(n_params - 1) * 0.1]
                    )
                else:
                    params = jnp.ones(n_params) * f_log

            initial_params.append(params)

        # Concatenate all parameters
        return jnp.concatenate(initial_params)

    def _estimate_empirical_survival(self, capture_matrix: jnp.ndarray) -> jnp.ndarray:
        """Estimate empirical survival probability using JAX-compatible operations."""
        # Simple approach: look at consecutive capture pairs
        n_individuals, n_occasions = capture_matrix.shape

        # Use jnp.where instead of if statement
        default_survival = 0.8

        # Count individuals captured in consecutive occasions using vectorized operations
        alive_and_captured = 0.0
        total_comparisons = 0.0

        for t in range(n_occasions - 1):
            captured_t = capture_matrix[:, t]
            captured_t1 = capture_matrix[:, t + 1]

            # Individuals captured at t
            n_captured_t = jnp.sum(captured_t)

            # Of those, how many were captured at t+1
            n_recaptured = jnp.sum(captured_t * captured_t1)

            # Use jnp.where to avoid control flow
            alive_and_captured += jnp.where(n_captured_t > 0, n_recaptured, 0.0)
            total_comparisons += jnp.where(n_captured_t > 0, n_captured_t, 0.0)

        # Calculate survival rate with JAX-compatible conditional
        survival_rate = jnp.where(
            total_comparisons > 0,
            alive_and_captured / total_comparisons,
            default_survival,
        )

        return survival_rate

    def _estimate_empirical_recruitment(
        self, capture_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """Estimate empirical recruitment rate using JAX-compatible operations."""
        # Simple approach: look at first captures as proxy for recruitment
        n_individuals, n_occasions = capture_matrix.shape

        default_recruitment = 0.1

        # Count new individuals in each period (first capture) using vectorized operations
        new_individuals = jnp.zeros(n_occasions - 1)

        for t in range(1, n_occasions):  # Start from second occasion
            # Individuals captured at t but not before
            captured_before = jnp.any(capture_matrix[:, :t], axis=1)
            captured_at_t = capture_matrix[:, t]

            new_at_t = captured_at_t * (1 - captured_before)
            new_individuals = new_individuals.at[t - 1].set(jnp.sum(new_at_t))

        # Average number of new individuals per occasion
        avg_new = jnp.mean(new_individuals)

        # Estimate recruitment rate relative to population size using JAX-compatible conditional
        population_estimate = n_individuals  # Rough proxy
        f_estimate = jnp.where(
            population_estimate > 0, avg_new / population_estimate, default_recruitment
        )

        return jnp.maximum(f_estimate, 0.01)

    def _split_parameters(
        self, parameters: jnp.ndarray, design_matrices: Dict[str, DesignMatrixInfo]
    ) -> Dict[str, jnp.ndarray]:
        """Split concatenated parameter vector by parameter type."""
        param_dict = {}
        start_idx = 0

        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count

            param_dict[param_name] = parameters[start_idx : start_idx + n_params]
            start_idx += n_params

        return param_dict

    def log_likelihood(
        self,
        parameters: jnp.ndarray,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo],
    ) -> float:
        """
        Calculate Pradel model log-likelihood using JAX-compatible operations.

        Args:
            parameters: Concatenated parameter vector
            data_context: Data and covariates
            design_matrices: Design matrices for each parameter

        Returns:
            Log-likelihood value
        """
        # Split parameters by type
        param_split = self._split_parameters(parameters, design_matrices)

        # Get design matrices
        X_phi = design_matrices["phi"].matrix
        X_p = design_matrices["p"].matrix
        X_f = design_matrices["f"].matrix

        # Calculate linear predictors
        eta_phi = X_phi @ param_split["phi"]
        eta_p = X_p @ param_split["p"]
        eta_f = X_f @ param_split["f"]

        # Apply link functions
        phi = inv_logit(eta_phi)  # Survival probability (0-1)
        p = inv_logit(eta_p)  # Detection probability (0-1)
        f = exp_link(eta_f)  # Recruitment rate (positive)

        # Get capture matrix
        capture_matrix = data_context.capture_matrix
        n_individuals, n_occasions = capture_matrix.shape

        # OPTIMIZED: Use JIT-compiled vectorized likelihood calculation
        base_likelihood = _pradel_vectorized_likelihood(phi, p, f, capture_matrix)

        # Apply a soft Beta prior to keep probabilities away from the boundaries.
        if self.boundary_prior_strength > 0.0:
            beta_prior = (
                _log_beta_prior(phi, self.boundary_prior_alpha, self.boundary_prior_beta)
                + _log_beta_prior(p, self.boundary_prior_alpha, self.boundary_prior_beta)
            )
            base_likelihood += self.boundary_prior_strength * beta_prior

        if self.recruitment_prior_strength > 0.0:
            base_likelihood += self.recruitment_prior_strength * _log_lognormal_prior(
                f, self.recruitment_prior_mode, self.recruitment_prior_sigma
            )

        return base_likelihood

    def validate_data(self, data_context: DataContext) -> None:
        """Validate data for Pradel model."""
        super().validate_data(data_context)

        # Pradel-specific validation
        if data_context.n_occasions < 3:
            raise ModelSpecificationError(
                formula="Pradel model data validation",
                suggestions=[
                    "Pradel models require at least 3 capture occasions",
                    "Use CJS model for 2-occasion data",
                ],
            )

        # Check for empty capture histories
        capture_matrix = data_context.capture_matrix
        empty_histories = jnp.sum(capture_matrix, axis=1) == 0
        n_empty = jnp.sum(empty_histories)

        if n_empty > 0:
            logger.warning(
                f"Found {n_empty} individuals with no captures - these will be ignored"
            )

    def calculate_lambda(
        self,
        parameters: jnp.ndarray,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo],
    ) -> jnp.ndarray:
        """
        Calculate lambda (population growth rate) from Pradel model parameters.

        In the Pradel (1996) recruitment (f) parameterization: λ = φ + f,
        with seniority γ = φ/λ. This matches program MARK's "Pradel recruitment"
        model, so estimates are comparable to RMark/MARK output.

        Args:
            parameters: Fitted parameter vector
            data_context: Data context
            design_matrices: Design matrices for each parameter

        Returns:
            Array of lambda values (one per individual)
        """
        # Split parameters by type
        param_split = self._split_parameters(parameters, design_matrices)

        # Get design matrices
        X_phi = design_matrices["phi"].matrix
        X_f = design_matrices["f"].matrix

        # Calculate linear predictors
        eta_phi = X_phi @ param_split["phi"]
        eta_f = X_f @ param_split["f"]

        # Apply link functions
        phi = inv_logit(eta_phi)  # Survival probability (0-1)
        f = exp_link(eta_f)  # Recruitment rate (positive)

        # Pradel (1996) recruitment parameterization: λ = φ + f
        lambda_values = phi + f

        return lambda_values

    def get_lambda_summary(self, lambda_values: jnp.ndarray) -> Dict[str, float]:
        """
        Get summary statistics for lambda values.

        Args:
            lambda_values: Array of lambda values

        Returns:
            Dictionary with summary statistics
        """
        return {
            "lambda_mean": float(jnp.mean(lambda_values)),
            "lambda_median": float(jnp.median(lambda_values)),
            "lambda_std": float(jnp.std(lambda_values)),
            "lambda_min": float(jnp.min(lambda_values)),
            "lambda_max": float(jnp.max(lambda_values)),
            "lambda_q25": float(jnp.percentile(lambda_values, 25)),
            "lambda_q75": float(jnp.percentile(lambda_values, 75)),
        }

    def predict(
        self,
        parameters: jnp.ndarray,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo],
        return_individual_predictions: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Make predictions using fitted Pradel model parameters.

        Args:
            parameters: Fitted parameter vector
            data_context: Data context for prediction
            design_matrices: Design matrices for prediction data
            return_individual_predictions: If True, return individual-level predictions

        Returns:
            Dictionary containing predictions and derived quantities
        """
        # Split parameters by type
        param_split = self._split_parameters(parameters, design_matrices)

        # Get design matrices
        X_phi = design_matrices["phi"].matrix
        X_p = design_matrices["p"].matrix
        X_f = design_matrices["f"].matrix

        # Calculate linear predictors
        eta_phi = X_phi @ param_split["phi"]
        eta_p = X_p @ param_split["p"]
        eta_f = X_f @ param_split["f"]

        # Apply link functions to get probabilities/rates
        phi = inv_logit(eta_phi)  # Survival probability (0-1)
        p = inv_logit(eta_p)  # Detection probability (0-1)
        f = exp_link(eta_f)  # Recruitment rate (positive)

        # Population growth rate - Pradel (1996) recruitment parameterization
        lambda_values = phi + f

        # Calculate log-likelihood for validation
        log_likelihood = self.log_likelihood(parameters, data_context, design_matrices)

        # Calculate AIC for validation data
        k = len(parameters)
        aic = 2 * k - 2 * log_likelihood

        predictions = {
            "phi": phi,
            "p": p,
            "f": f,
            "lambda": lambda_values,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "n_parameters": k,
        }

        # Add summary statistics
        predictions.update(
            {
                "phi_mean": float(jnp.mean(phi)),
                "phi_std": float(jnp.std(phi)),
                "p_mean": float(jnp.mean(p)),
                "p_std": float(jnp.std(p)),
                "f_mean": float(jnp.mean(f)),
                "f_std": float(jnp.std(f)),
            }
        )

        # Add lambda summary
        lambda_summary = self.get_lambda_summary(lambda_values)
        predictions.update(lambda_summary)

        # Optionally include individual-level predictions
        if not return_individual_predictions:
            # Remove individual-level arrays to save memory
            for key in ["phi", "p", "f", "lambda"]:
                del predictions[key]

        return predictions

    def predict_capture_probabilities(
        self,
        parameters: jnp.ndarray,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo],
    ) -> jnp.ndarray:
        """
        Predict capture probabilities for each individual and occasion.

        Args:
            parameters: Fitted parameter vector
            data_context: Data context
            design_matrices: Design matrices

        Returns:
            Matrix of capture probabilities (n_individuals x n_occasions)
        """
        # Get parameter estimates
        param_split = self._split_parameters(parameters, design_matrices)

        # Get design matrices
        X_phi = design_matrices["phi"].matrix
        X_p = design_matrices["p"].matrix

        # Calculate linear predictors
        eta_phi = X_phi @ param_split["phi"]
        eta_p = X_p @ param_split["p"]

        # Apply link functions
        phi = inv_logit(eta_phi)  # Survival probability
        p = inv_logit(eta_p)  # Detection probability

        # Calculate capture probabilities for each occasion
        n_individuals, n_occasions = data_context.capture_matrix.shape

        # For Pradel model, capture probability depends on survival to that occasion
        # This is a simplified version - could be more sophisticated
        capture_probs = jnp.broadcast_to(p[:, None], (n_individuals, n_occasions))

        # Adjust for survival probability (individuals must survive to be captured)
        for t in range(1, n_occasions):
            # Cumulative survival to occasion t
            survival_to_t = jnp.power(phi, t)
            capture_probs = capture_probs.at[:, t].multiply(survival_to_t)

        return capture_probs

    def calculate_validation_metrics(
        self,
        parameters: jnp.ndarray,
        train_context: DataContext,
        val_context: DataContext,
        design_matrices_train: Dict[str, DesignMatrixInfo],
        design_matrices_val: Dict[str, DesignMatrixInfo],
    ) -> Dict[str, float]:
        """
        Calculate validation metrics comparing training and validation performance.

        Args:
            parameters: Fitted parameters from training data
            train_context: Training data context
            val_context: Validation data context
            design_matrices_train: Training design matrices
            design_matrices_val: Validation design matrices

        Returns:
            Dictionary with validation metrics
        """
        # Training performance
        train_ll = self.log_likelihood(parameters, train_context, design_matrices_train)
        train_aic = 2 * len(parameters) - 2 * train_ll

        # Validation performance
        val_ll = self.log_likelihood(parameters, val_context, design_matrices_val)
        val_aic = 2 * len(parameters) - 2 * val_ll

        # Calculate per-individual metrics for fair comparison
        train_ll_per_ind = train_ll / train_context.n_individuals
        val_ll_per_ind = val_ll / val_context.n_individuals

        # Overfitting assessment
        ll_difference = train_ll_per_ind - val_ll_per_ind
        aic_difference = val_aic - train_aic

        return {
            "train_log_likelihood": float(train_ll),
            "train_aic": float(train_aic),
            "train_ll_per_individual": float(train_ll_per_ind),
            "val_log_likelihood": float(val_ll),
            "val_aic": float(val_aic),
            "val_ll_per_individual": float(val_ll_per_ind),
            "log_likelihood_difference": float(ll_difference),
            "aic_difference": float(aic_difference),
            "overfitting_ratio": (
                float(ll_difference / abs(train_ll_per_ind))
                if train_ll_per_ind != 0
                else 0.0
            ),
        }

    def get_parameter_names(
        self, design_matrices: Dict[str, DesignMatrixInfo]
    ) -> List[str]:
        """Get names of all parameters in order."""
        names = []

        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]

            # Add parameter prefix to column names
            param_names = [f"{param_name}_{col}" for col in design_info.column_names]
            names.extend(param_names)

        return names
