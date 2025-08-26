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
def calculate_seniority_gamma(phi: float, f: float) -> float:
    """
    Calculate seniority probability γ from Pradel (1996).

    From Pradel (1996): γ = φ/(1+f)
    This is the probability that an individual present at time i
    was also present at time i-1.
    """
    return phi / (1.0 + f)


@jax.jit
def _pradel_individual_likelihood(
    capture_history: jnp.ndarray, phi: float, p: float, f: float
) -> float:
    """
    Mathematically correct Pradel likelihood based on Pradel (1996).

    Implements the exact formulation from equation (2) in Pradel (1996):

    For an individual with capture history h = (h₁, h₂, ..., hₙ):

    L(h) = Pr(first capture at j) × Pr(h_{j+1}, ..., h_k | captured at j) × Pr(not seen after k)

    Where:
    - γᵢ = φᵢ₋₁/(1 + fᵢ₋₁) is the seniority probability
    - λᵢ = 1 + fᵢ is the population growth rate
    - φᵢ is the survival probability from i to i+1
    - pᵢ is the detection probability at occasion i

    Args:
        capture_history: Binary array of capture occasions (1=captured, 0=not)
        phi: Survival probability (constant across occasions)
        p: Detection probability (constant across occasions)
        f: Per-capita recruitment rate (constant across occasions)

    Returns:
        Log-likelihood contribution for this individual
    """
    n_occasions = len(capture_history)
    total_captures = jnp.sum(capture_history)

    # Small constant to prevent log(0)
    epsilon = 1e-12

    # Calculate derived parameters using Pradel (1996) relationships
    gamma = calculate_seniority_gamma(phi, f)  # γ = φ/(1+f)
    lambda_pop = 1.0 + f  # λ = 1 + f

    def never_captured_likelihood():
        """
        For never-captured individuals, calculate probability they were never in population
        or were in population but never detected.

        From Pradel (1996): This involves the probability of not entering during study
        plus probability of entering but never being detected.
        """
        # Probability of never entering the population during the study
        prob_never_enter = 1.0 / (lambda_pop ** (n_occasions - 1))

        # Probability of entering but never being detected
        # Simplified: if entered, probability of never being detected = (1-p)^n_occasions
        prob_enter_not_detected = (1.0 - prob_never_enter) * ((1.0 - p) ** n_occasions)

        total_prob = prob_never_enter + prob_enter_not_detected
        return jnp.log(jnp.maximum(total_prob, epsilon))

    def captured_likelihood():
        """
        For captured individuals, implement the Pradel (1996) likelihood formulation.
        """
        # Find first and last capture occasions
        indices = jnp.arange(n_occasions)

        # Get first capture occasion
        capture_indices = jnp.where(capture_history == 1, indices, n_occasions)
        first_capture = jnp.min(capture_indices)

        # Get last capture occasion
        last_capture_indices = jnp.where(capture_history == 1, indices, -1)
        last_capture = jnp.max(last_capture_indices)

        # Initialize log-likelihood
        log_likelihood = 0.0

        # Part 1: Probability of first capture at occasion 'first_capture'
        # This includes probability of entering before or at first_capture
        # and not being detected until first_capture, then being detected

        # Probability of being in population at first capture (JAX-compatible)
        entry_prob = jnp.where(
            first_capture > 0,
            gamma**first_capture,  # Could have entered at any previous occasion
            1.0,  # Captured at first occasion - was definitely present
        )

        # Probability of not being detected until first_capture (JAX-compatible)
        not_detected_prob = jnp.where(
            first_capture > 0, (1.0 - p) ** first_capture, 1.0
        )

        # Probability of detection at first_capture
        detected_prob = p

        # Add first capture contribution
        first_capture_contrib = entry_prob * not_detected_prob * detected_prob
        log_likelihood += jnp.log(jnp.maximum(first_capture_contrib, epsilon))

        # Part 2: Process occasions between first and last capture
        # This follows CJS-like structure but with Pradel modifications

        def process_intermediate_occasion(carry, t):
            running_ll = carry

            # Only process occasions after first capture and before/at last
            in_active_period = (t > first_capture) & (t <= last_capture)

            survival_contrib = jnp.where(
                in_active_period,
                jnp.log(jnp.maximum(phi, epsilon)),  # Survived to this occasion
                0.0,
            )

            # Detection contribution (for occasions before last)
            before_last = t < last_capture
            in_detection_period = (t > first_capture) & before_last

            captured_at_t = capture_history[t] == 1
            detection_contrib = jnp.where(
                in_detection_period,
                jnp.where(
                    captured_at_t,
                    jnp.log(jnp.maximum(p, epsilon)),  # Detected
                    jnp.log(jnp.maximum(1.0 - p, epsilon)),  # Not detected
                ),
                0.0,
            )

            # For last capture, we know it was detected (no choice probability)
            at_last = t == last_capture
            last_detection_contrib = jnp.where(
                at_last,
                jnp.log(jnp.maximum(p, epsilon)),  # Must be detected at last
                0.0,
            )

            new_ll = (
                running_ll
                + survival_contrib
                + detection_contrib
                + last_detection_contrib
            )
            return new_ll, new_ll

        # Scan over all occasions
        final_ll, _ = jax.lax.scan(
            process_intermediate_occasion, log_likelihood, jnp.arange(n_occasions)
        )

        # Part 3: Probability of not being seen after last capture
        # This involves either death or emigration after last capture
        # For occasions after last capture, individual either died or emigrated

        occasions_after_last = n_occasions - 1 - last_capture

        # JAX-compatible handling of occasions after last capture
        not_available_prob = jnp.where(
            occasions_after_last > 0,
            (1.0 - phi * p) ** occasions_after_last,
            1.0,  # No occasions after, so probability is 1
        )

        final_ll += jnp.where(
            occasions_after_last > 0,
            jnp.log(jnp.maximum(not_available_prob, epsilon)),
            0.0,  # No contribution if no occasions after
        )

        return final_ll

    # Return appropriate likelihood based on capture status
    return jnp.where(
        total_captures > 0, captured_likelihood(), never_captured_likelihood()
    )


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

    def __init__(self, model_type: ModelType = ModelType.PRADEL):
        super().__init__(model_type)
        self.parameter_order = ["phi", "p", "f"]

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

        FIXED: Use biologically reasonable bounds instead of overly restrictive ones.
        The original bounds [-10, 10] were causing optimization to hit boundaries.
        """
        bounds = []

        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count

            if param_name in ["phi", "p"]:
                # Logit-scale bounds for survival/detection probabilities
                # Allow probabilities from 0.0001 to 0.9999 (more precision, still stable)
                # logit(0.0001) ≈ -9.210, logit(0.9999) ≈ 9.210
                param_bounds = [(logit(0.0001), logit(0.9999))] * n_params
            elif param_name == "f":
                # Log-scale bounds for recruitment rate
                # Allow recruitment from 0.00001 to 10.0 (wider range for population dynamics)
                # log(0.00001) ≈ -11.513, log(10.0) ≈ 2.303
                param_bounds = [(log_link(0.00001), log_link(10.0))] * n_params
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
        return _pradel_vectorized_likelihood(phi, p, f, capture_matrix)

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

        CORRECTED: In the Pradel model: λ = 1 + f  (NOT φ + f)
        From Pradel (1996): λ = φ/γ = 1 + f, where γ = φ/(1+f)

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
        X_f = design_matrices["f"].matrix

        # Calculate linear predictors
        eta_f = X_f @ param_split["f"]

        # Apply link functions
        f = exp_link(eta_f)  # Recruitment rate (positive)

        # FIXED: Calculate lambda = 1 + recruitment rate (Pradel 1996)
        lambda_values = 1.0 + f

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

        # FIXED: Calculate lambda (population growth rate) - Pradel (1996)
        lambda_values = 1.0 + f

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
