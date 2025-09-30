"""
Parallel optimization framework for pradel-jax.

Provides parallel model fitting capabilities for running multiple models
simultaneously across CPU cores with checkpoint/resume functionality.
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .orchestrator import optimize_model, OptimizationRequest, OptimizationResponse
from .strategy import OptimizationStrategy
from ..data.adapters import DataContext
from ..formulas.spec import FormulaSpec
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelModelSpec:
    """Specification for a model to be fit in parallel."""

    name: str
    formula_spec: FormulaSpec
    index: int
    random_seed: Optional[int] = None


@dataclass
class ParallelOptimizationResult:
    """Result from parallel optimization."""

    success: bool
    model_name: str
    model_index: int
    parameters: Optional[List[float]] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    n_parameters: Optional[int] = None
    strategy_used: Optional[str] = None
    fit_time: Optional[float] = None
    error_message: Optional[str] = None
    # New fields for lambda estimates and reproducibility
    lambda_mean: Optional[float] = None
    lambda_median: Optional[float] = None
    lambda_std: Optional[float] = None
    lambda_min: Optional[float] = None
    lambda_max: Optional[float] = None
    lambda_q25: Optional[float] = None
    lambda_q75: Optional[float] = None
    random_seed: Optional[int] = None
    data_hash: Optional[str] = None


class CheckpointManager:
    """Manages checkpointing for long-running parallel jobs."""

    def __init__(self, checkpoint_file: Union[str, Path]):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_checkpoint_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to JSON-serializable format."""
        serializable_state = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                serializable_state[key] = {
                    '_type': 'numpy_array',
                    'data': value.tolist(),
                    'dtype': str(value.dtype),
                    'shape': value.shape
                }
            elif hasattr(value, 'to_dict'):  # For objects with dict conversion
                serializable_state[key] = {
                    '_type': 'object_dict',
                    'data': value.to_dict()
                }
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serializable_state[key] = value
            else:
                # Skip non-serializable objects with warning
                logger.warning(f"Skipping non-serializable checkpoint data: {key} ({type(value)})")
        return serializable_state

    def _deserialize_checkpoint_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON data back to original format."""
        deserialized_state = {}
        for key, value in state.items():
            if isinstance(value, dict) and '_type' in value:
                if value['_type'] == 'numpy_array':
                    deserialized_state[key] = np.array(
                        value['data'], 
                        dtype=value['dtype']
                    ).reshape(value['shape'])
                elif value['_type'] == 'object_dict':
                    # Note: Object reconstruction would require class knowledge
                    # For now, store as dict - applications should handle reconstruction
                    deserialized_state[key] = value['data']
                else:
                    deserialized_state[key] = value
            else:
                deserialized_state[key] = value
        return deserialized_state

    def save_checkpoint(self, state: Dict[str, Any]):
        """Save checkpoint state using secure JSON serialization."""
        logger.info(f"Saving checkpoint to {self.checkpoint_file}")
        try:
            serializable_state = self._serialize_checkpoint_data(state)
            with open(self.checkpoint_file, "w", encoding='utf-8') as f:
                json.dump(serializable_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state from secure JSON format."""
        if self.checkpoint_file.exists():
            logger.info(f"Loading checkpoint from {self.checkpoint_file}")
            try:
                with open(self.checkpoint_file, "r", encoding='utf-8') as f:
                    serialized_state = json.load(f)
                return self._deserialize_checkpoint_data(serialized_state)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None
        return None

    def delete_checkpoint(self):
        """Delete checkpoint file after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Deleted checkpoint file")


def _fit_model_worker(args):
    """
    Worker function for parallel model fitting.
    Runs in separate process to enable true parallelization.
    """
    # Unpack arguments (added worker_options for extended behaviors)
    if len(args) == 5:
        model_spec, data_context_dict, objective_func_name, bounds, strategy = args
        worker_options = {}
    else:
        model_spec, data_context_dict, objective_func_name, bounds, strategy, worker_options = args

    try:
        # Deserialize data context from dict
        from ..data.adapters import DataContext

        data_context = DataContext.from_dict(data_context_dict)

        start_time = time.time()

        # Import necessary modules in worker process
        import pradel_jax as pj
        from pradel_jax.optimization import optimize_model
        from pradel_jax.optimization.strategy import OptimizationStrategy

        # Create model and fit
        model = pj.PradelModel()
        design_matrices = model.build_design_matrices(
            model_spec.formula_spec, data_context
        )
        initial_params = model.get_initial_parameters(data_context, design_matrices)

        # Penalty options
        penalty = (worker_options or {}).get("penalty", "none")
        lambda_penalty = float((worker_options or {}).get("lambda_penalty", 0.0))
        include_intercept = bool((worker_options or {}).get("penalty_include_intercept", False))

        def _ridge_mask(design_mats):
            sizes = [
                design_mats["phi"].parameter_count,
                design_mats["p"].parameter_count,
                design_mats["f"].parameter_count,
            ]
            mask = np.ones(sum(sizes), dtype=float)
            if not include_intercept:
                idx = 0
                for pname in ["phi", "p", "f"]:
                    cols = design_mats[pname].column_names
                    if design_mats[pname].has_intercept:
                        try:
                            loc = cols.index("(Intercept)")
                            mask[idx + loc] = 0.0
                        except ValueError:
                            pass
                    idx += design_mats[pname].parameter_count
            return mask

        ridge_mask = _ridge_mask(design_matrices)

        # Define objective with optional ridge
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                nll = -ll if np.isfinite(ll) else 1e10
                if penalty == "ridge" and lambda_penalty > 0:
                    pen = lambda_penalty * float(np.sum((np.asarray(params) ** 2) * ridge_mask))
                    nll = nll + pen
                # Optional boundary-aware prior (worker_options)
                bprior = (worker_options or {}).get("boundary_prior", "none")
                bweight = float((worker_options or {}).get("boundary_weight", 0.0))
                if bprior != "none" and bweight > 0:
                    sizes = [
                        design_matrices["phi"].parameter_count,
                        design_matrices["p"].parameter_count,
                        design_matrices["f"].parameter_count,
                    ]
                    phi_params = np.asarray(params[:sizes[0]])
                    p_params = np.asarray(params[sizes[0]:sizes[0]+sizes[1]])
                    X_phi = np.asarray(design_matrices["phi"].matrix)
                    X_p = np.asarray(design_matrices["p"].matrix)
                    eta_phi = X_phi @ phi_params
                    eta_p = X_p @ p_params
                    phi_prob = 1.0 / (1.0 + np.exp(-eta_phi))
                    p_prob = 1.0 / (1.0 + np.exp(-eta_p))
                    eps = 1e-12
                    if bprior == "barrier":
                        prior_term = - (np.log(phi_prob*(1-phi_prob) + eps) + np.log(p_prob*(1-p_prob) + eps)).mean()
                    else:
                        prior_term = -0.5 * (np.log(phi_prob + eps) + np.log(1-phi_prob + eps) +
                                             np.log(p_prob + eps) + np.log(1-p_prob + eps)).mean()
                    nll = nll + bweight * float(prior_term)
                return nll
            except Exception:
                return 1e10

        # Convert strategy string to enum if needed
        if isinstance(strategy, str):
            try:
                strategy_enum = OptimizationStrategy(strategy)
            except ValueError:
                # Fallback to a default strategy
                strategy_enum = OptimizationStrategy.SCIPY_LBFGS
        else:
            strategy_enum = strategy

        # Optional warm-start from intercept-only
        warm_start = (worker_options or {}).get("warm_start", "none")
        warm_start_iter = int((worker_options or {}).get("warm_start_iter", 200))
        try:
            if warm_start == "intercept":
                ispec = pj.create_simple_spec(phi="~1", p="~1", f="~1")
                dm_i = model.build_design_matrices(ispec, data_context)
                init_i = model.get_initial_parameters(data_context, dm_i)
                bnd_i = model.get_parameter_bounds(data_context, dm_i)

                def obj_i(p):
                    return -model.log_likelihood(p, data_context, dm_i)

                warm_resp = optimize_model(
                    objective_function=obj_i,
                    initial_parameters=np.array(init_i),
                    context=data_context,
                    bounds=bnd_i,
                    preferred_strategy=strategy_enum,
                    config_overrides={"max_iter": warm_start_iter, "tolerance": 1e-6},
                )
                if warm_resp.success:
                    theta_i = np.array(warm_resp.result.x)
                    start = np.array(initial_params, dtype=float)
                    idx = 0
                    if design_matrices["phi"].has_intercept:
                        try:
                            loc = design_matrices["phi"].column_names.index("(Intercept)")
                            start[idx + loc] = theta_i[0]
                        except ValueError:
                            pass
                    idx += design_matrices["phi"].parameter_count
                    if design_matrices["p"].has_intercept:
                        try:
                            loc = design_matrices["p"].column_names.index("(Intercept)")
                            start[idx + loc] = theta_i[1]
                        except ValueError:
                            pass
                    idx += design_matrices["p"].parameter_count
                    if design_matrices["f"].has_intercept:
                        try:
                            loc = design_matrices["f"].column_names.index("(Intercept)")
                            start[idx + loc] = theta_i[2]
                        except ValueError:
                            pass
                    initial_params = start
        except Exception:
            pass

        # Optimize
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            bounds=bounds,
            preferred_strategy=strategy_enum,
        )

        fit_time = time.time() - start_time

        if result.success:
            opt_result = result.result
            ll = -opt_result.fun
            k = len(opt_result.x)
            aic = 2 * k - 2 * ll

            # Calculate lambda estimates for Pradel model
            lambda_estimates = None
            lambda_summary = {}

            try:
                if hasattr(model, "calculate_lambda"):
                    lambda_estimates = model.calculate_lambda(
                        opt_result.x, data_context, design_matrices
                    )
                    lambda_summary = model.get_lambda_summary(lambda_estimates)
            except Exception as e:
                logger.warning(f"Failed to calculate lambda estimates: {e}")

            # Generate data hash for reproducibility using secure SHA-256
            import hashlib

            data_str = str(data_context.capture_matrix.tolist())
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

            # Get random seed from model spec if available
            random_seed = getattr(model_spec, "random_seed", None)

            return ParallelOptimizationResult(
                success=True,
                model_name=model_spec.name,
                model_index=model_spec.index,
                parameters=opt_result.x.tolist(),
                log_likelihood=ll,
                aic=aic,
                n_parameters=k,
                strategy_used=result.strategy_used,
                fit_time=fit_time,
                lambda_mean=lambda_summary.get("lambda_mean"),
                lambda_median=lambda_summary.get("lambda_median"),
                lambda_std=lambda_summary.get("lambda_std"),
                lambda_min=lambda_summary.get("lambda_min"),
                lambda_max=lambda_summary.get("lambda_max"),
                lambda_q25=lambda_summary.get("lambda_q25"),
                lambda_q75=lambda_summary.get("lambda_q75"),
                random_seed=random_seed,
                data_hash=data_hash,
            )
        else:
            return ParallelOptimizationResult(
                success=False,
                model_name=model_spec.name,
                model_index=model_spec.index,
                fit_time=fit_time,
                error_message="Optimization failed",
            )

    except Exception as e:
        return ParallelOptimizationResult(
            success=False,
            model_name=model_spec.name,
            model_index=model_spec.index,
            error_message=str(e),
        )


class ParallelOptimizer:
    """
    Parallel optimization coordinator for fitting multiple models simultaneously.

    Features:
    - True parallel processing across CPU cores
    - Checkpoint/resume functionality for long jobs
    - Progress tracking and time estimation
    - Batch processing to manage memory usage
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ParallelOptimizer with {self.n_workers} workers")

    def fit_models_parallel(
        self,
        model_specs: List[ParallelModelSpec],
        data_context: DataContext,
        bounds: Optional[List[tuple]] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.HYBRID,
        batch_size: int = 8,
        checkpoint_interval: int = 5,
        checkpoint_name: Optional[str] = None,
        resume: bool = False,
        worker_options: Optional[Dict[str, Any]] = None,
    ) -> List[ParallelOptimizationResult]:
        """
        Fit multiple models in parallel with checkpointing.

        Args:
            model_specs: List of model specifications to fit
            data_context: Data context for all models
            bounds: Parameter bounds
            strategy: Optimization strategy to use
            batch_size: Number of models to process in parallel batches
            checkpoint_interval: Save checkpoint every N batches
            checkpoint_name: Name for checkpoint file
            resume: Whether to resume from checkpoint

        Returns:
            List of optimization results
        """
        if checkpoint_name is None:
            checkpoint_name = f"parallel_fit_{int(time.time())}"

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        checkpoint_manager = CheckpointManager(checkpoint_file)

        # Try to resume from checkpoint
        if resume:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                logger.info("Resuming from checkpoint")
                results = checkpoint_data["results"]
                completed_indices = set(checkpoint_data["completed_indices"])
                start_time = checkpoint_data["start_time"]
            else:
                logger.info("No checkpoint found, starting fresh")
                results = [None] * len(model_specs)
                completed_indices = set()
                start_time = time.time()
        else:
            results = [None] * len(model_specs)
            completed_indices = set()
            start_time = time.time()

        logger.info(f"Fitting {len(model_specs)} models with {self.n_workers} workers")
        logger.info(
            f"Batch size: {batch_size}, Checkpoint interval: {checkpoint_interval}"
        )

        # Process remaining models in batches
        remaining_specs = [
            spec for spec in model_specs if spec.index not in completed_indices
        ]

        for batch_start in range(0, len(remaining_specs), batch_size):
            batch_specs = remaining_specs[batch_start : batch_start + batch_size]
            batch_start_time = time.time()

            logger.info(
                f"Processing batch: models {batch_specs[0].index+1}-{batch_specs[-1].index+1}"
            )

            # Prepare arguments for worker processes
            # Serialize data context for cross-process communication
            data_context_dict = data_context.to_dict()
            worker_args = [
                (
                    spec,
                    data_context_dict,
                    "log_likelihood",
                    bounds,
                    strategy,
                    worker_options or {},
                )
                for spec in batch_specs
            ]

            # Submit batch to workers
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_spec = {
                    executor.submit(_fit_model_worker, args): args[0]
                    for args in worker_args
                }

                # Collect results as they complete
                for future in as_completed(future_to_spec):
                    try:
                        result = future.result()
                        results[result.model_index] = result
                        completed_indices.add(result.model_index)

                        if result.success:
                            logger.info(f"✅ {result.model_name}: AIC={result.aic:.1f}")
                        else:
                            logger.warning(
                                f"❌ {result.model_name}: {result.error_message}"
                            )

                    except Exception as e:
                        logger.error(f"Worker failed: {str(e)}")

            batch_time = time.time() - batch_start_time
            models_per_second = len(batch_specs) / batch_time

            # Progress reporting
            completed_count = len(completed_indices)
            elapsed_total = time.time() - start_time

            if completed_count > 0:
                remaining_models = len(model_specs) - completed_count
                eta_seconds = (
                    remaining_models / models_per_second if models_per_second > 0 else 0
                )

                logger.info(
                    f"Batch completed in {batch_time:.1f}s ({models_per_second:.1f} models/sec)"
                )
                logger.info(
                    f"Progress: {completed_count}/{len(model_specs)} ({completed_count/len(model_specs)*100:.1f}%)"
                )
                logger.info(f"ETA: {eta_seconds/3600:.1f} hours")

            # Save checkpoint periodically
            if len(completed_indices) % checkpoint_interval == 0:
                checkpoint_data = {
                    "results": results,
                    "completed_indices": list(completed_indices),
                    "start_time": start_time,
                    "model_specs": model_specs,
                }
                checkpoint_manager.save_checkpoint(checkpoint_data)

        # Final results
        total_time = time.time() - start_time
        successful_results = [r for r in results if r and r.success]

        logger.info(f"Parallel fitting completed in {total_time/3600:.1f} hours")
        logger.info(f"Successful models: {len(successful_results)}/{len(model_specs)}")

        # Clean up checkpoint
        checkpoint_manager.delete_checkpoint()

        return results


# Convenience functions
def fit_models_parallel(
    model_specs: List[ParallelModelSpec],
    data_context: DataContext,
    n_workers: Optional[int] = None,
    **kwargs,
) -> List[ParallelOptimizationResult]:
    """
    Convenience function for parallel model fitting.

    Args:
        model_specs: Models to fit
        data_context: Data context
        n_workers: Number of parallel workers
        **kwargs: Additional arguments for ParallelOptimizer

    Returns:
        List of optimization results
    """
    optimizer = ParallelOptimizer(n_workers=n_workers)
    return optimizer.fit_models_parallel(model_specs, data_context, **kwargs)


def create_model_specs_from_formulas(
    phi_formulas: List[str],
    p_formulas: List[str],
    f_formulas: List[str],
    random_seed_base: Optional[int] = None,
) -> List[ParallelModelSpec]:
    """
    Create model specifications from formula lists.

    Args:
        phi_formulas: List of survival formulas
        p_formulas: List of detection formulas
        f_formulas: List of recruitment formulas
        random_seed_base: Base seed for reproducible results (incremented per model)

    Returns:
        List of model specifications
    """
    import pradel_jax as pj

    def create_model_name(phi, p, f):
        def format_formula(formula):
            if formula == "~1":
                return "1"
            return (
                formula.replace("~1 + ", "")
                .replace("age_1", "age")
                .replace("tier_1", "tier")
            )

        return f"φ({format_formula(phi)}) p({format_formula(p)}) f({format_formula(f)})"

    specs = []
    index = 0

    for phi in phi_formulas:
        for p in p_formulas:
            for f in f_formulas:
                # Generate unique seed for this model
                model_seed = None
                if random_seed_base is not None:
                    model_seed = random_seed_base + index

                spec = ParallelModelSpec(
                    name=create_model_name(phi, p, f),
                    formula_spec=pj.create_simple_spec(
                        phi=phi, p=p, f=f, name=f"model_{index}"
                    ),
                    index=index,
                    random_seed=model_seed,
                )
                specs.append(spec)
                index += 1

    return specs
