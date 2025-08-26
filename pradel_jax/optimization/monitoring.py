"""
Performance monitoring and metrics collection for optimization.

Follows industry patterns from:
- MLflow for experiment tracking
- Weights & Biases for metrics logging
- TensorBoard for visualization
- Prometheus for monitoring systems

Provides comprehensive optimization monitoring including:
- Real-time performance metrics
- Convergence diagnostics
- Resource utilization tracking
- Experiment comparison and analysis
"""

import numpy as np
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import logging
import threading
from contextlib import contextmanager

# Optional integrations with monitoring systems
try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import wandb

    HAS_WANDB = False  # Disabled by default to avoid cloud dependencies
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from .optimizers import OptimizationResult
from .strategy import OptimizationStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Standard metrics for optimization monitoring."""

    iteration: int
    objective_value: float
    gradient_norm: Optional[float] = None
    parameter_norm: Optional[float] = None
    step_size: Optional[float] = None
    convergence_rate: Optional[float] = None
    elapsed_time: float = 0.0
    memory_usage_mb: float = 0.0

    # Statistical metrics
    condition_number: Optional[float] = None
    eigenvalue_ratio: Optional[float] = None

    # Convergence indicators
    relative_improvement: Optional[float] = None
    stagnation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class OptimizationSession:
    """Represents a complete optimization session."""

    session_id: str
    strategy: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    final_objective: Optional[float] = None
    total_iterations: int = 0
    total_function_evaluations: int = 0
    convergence_message: str = ""

    # Problem characteristics
    n_parameters: int = 0
    problem_difficulty: str = ""

    # Performance summary
    metrics_history: List[OptimizationMetrics] = None

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []

    @property
    def duration(self) -> float:
        """Total optimization duration."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        result = asdict(self)
        result["duration"] = self.duration
        return result


class PerformanceMonitor:
    """
    Real-time performance monitoring system.

    Follows patterns from monitoring systems like Prometheus,
    with real-time metrics collection and alerting.
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.sessions = {}
        self.active_session: Optional[str] = None
        self.callbacks = []
        self._lock = threading.Lock()

        # Performance thresholds (configurable)
        self.thresholds = {
            "max_stagnation": 50,
            "min_improvement_rate": 1e-8,
            "max_gradient_norm": 1e10,
            "memory_warning_mb": 8000,
        }

    def start_session(
        self,
        strategy: OptimizationStrategy,
        n_parameters: int,
        problem_difficulty: str = "unknown",
    ) -> str:
        """Start new optimization monitoring session."""
        session_id = str(uuid.uuid4())[:8]

        session = OptimizationSession(
            session_id=session_id,
            strategy=strategy.value,
            start_time=time.time(),
            n_parameters=n_parameters,
            problem_difficulty=problem_difficulty,
        )

        with self._lock:
            self.sessions[session_id] = session
            self.active_session = session_id

        logger.info(f"Started optimization session {session_id} with {strategy.value}")
        return session_id

    def log_metrics(self, metrics: OptimizationMetrics) -> None:
        """Log optimization metrics."""
        if self.active_session is None:
            logger.warning("No active session for metrics logging")
            return

        metrics.elapsed_time = time.time()

        with self._lock:
            session = self.sessions[self.active_session]
            session.metrics_history.append(metrics)
            self.metrics_buffer.append(metrics)

        # Check for warnings/alerts
        self._check_performance_alerts(metrics)

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Monitoring callback failed: {e}")

    def end_session(self, result: OptimizationResult) -> None:
        """End current optimization session."""
        if self.active_session is None:
            logger.warning("No active session to end")
            return

        with self._lock:
            session = self.sessions[self.active_session]
            session.end_time = time.time()
            session.success = result.success
            session.final_objective = result.fun
            session.total_iterations = result.nit
            session.total_function_evaluations = result.nfev
            session.convergence_message = result.message

        logger.info(
            f"Ended session {self.active_session}: "
            f"{'success' if result.success else 'failed'} "
            f"in {session.duration:.2f}s"
        )

        self.active_session = None

    def _check_performance_alerts(self, metrics: OptimizationMetrics) -> None:
        """Check for performance issues and log alerts."""
        alerts = []

        if metrics.stagnation_count > self.thresholds["max_stagnation"]:
            alerts.append(
                f"Optimization stagnating ({metrics.stagnation_count} iterations)"
            )

        if (
            metrics.gradient_norm
            and metrics.gradient_norm > self.thresholds["max_gradient_norm"]
        ):
            alerts.append(f"Very large gradient norm ({metrics.gradient_norm:.2e})")

        if metrics.memory_usage_mb > self.thresholds["memory_warning_mb"]:
            alerts.append(f"High memory usage ({metrics.memory_usage_mb:.0f}MB)")

        if (
            metrics.relative_improvement
            and metrics.relative_improvement < self.thresholds["min_improvement_rate"]
        ):
            alerts.append(f"Very slow convergence ({metrics.relative_improvement:.2e})")

        for alert in alerts:
            logger.warning(f"Performance alert: {alert}")

    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        if session_id is None:
            session_id = self.active_session

        if session_id is None or session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]
        metrics = session.metrics_history

        if not metrics:
            return session.to_dict()

        # Compute summary statistics
        objective_values = [m.objective_value for m in metrics]
        gradient_norms = [
            m.gradient_norm for m in metrics if m.gradient_norm is not None
        ]

        summary = session.to_dict()
        summary.update(
            {
                "convergence_stats": {
                    "best_objective": (
                        min(objective_values) if objective_values else None
                    ),
                    "worst_objective": (
                        max(objective_values) if objective_values else None
                    ),
                    "final_objective": (
                        objective_values[-1] if objective_values else None
                    ),
                    "total_improvement": (
                        objective_values[0] - objective_values[-1]
                        if len(objective_values) > 1
                        else 0
                    ),
                    "avg_gradient_norm": (
                        np.mean(gradient_norms) if gradient_norms else None
                    ),
                    "final_gradient_norm": (
                        gradient_norms[-1] if gradient_norms else None
                    ),
                },
                "performance_stats": {
                    "avg_time_per_iteration": session.duration
                    / max(session.total_iterations, 1),
                    "convergence_rate": self._estimate_convergence_rate(metrics),
                    "efficiency_score": self._calculate_efficiency_score(session),
                },
            }
        )

        return summary

    def _estimate_convergence_rate(
        self, metrics: List[OptimizationMetrics]
    ) -> Optional[float]:
        """Estimate exponential convergence rate."""
        if len(metrics) < 10:
            return None

        objective_values = [m.objective_value for m in metrics[-10:]]
        if len(set(objective_values)) <= 1:  # No variation
            return None

        # Fit exponential decay to recent objective values
        try:
            x = np.arange(len(objective_values))
            y = np.array(objective_values)
            if np.any(y <= 0):
                return None

            log_y = np.log(y - np.min(y) + 1e-10)
            poly = np.polyfit(x, log_y, 1)
            return float(-poly[0])  # Negative slope = convergence rate
        except:
            return None

    def _calculate_efficiency_score(self, session: OptimizationSession) -> float:
        """Calculate optimization efficiency score (0-100)."""
        base_score = 100.0

        # Penalize excessive iterations
        if session.total_iterations > 1000:
            base_score *= 0.8
        elif session.total_iterations > 2000:
            base_score *= 0.6

        # Penalize long runtime relative to problem size
        expected_time = session.n_parameters * 0.1  # 0.1s per parameter baseline
        if session.duration > expected_time * 10:
            base_score *= 0.5

        # Bonus for success
        if session.success:
            base_score *= 1.2
        else:
            base_score *= 0.3

        return min(100.0, base_score)

    def add_callback(self, callback: Callable[[OptimizationMetrics], None]) -> None:
        """Add monitoring callback function."""
        self.callbacks.append(callback)


class ExperimentTracker:
    """
    Experiment tracking system following MLflow patterns.

    Provides comprehensive experiment management and comparison
    capabilities for optimization research and development.
    """

    def __init__(self, tracking_dir: Optional[Union[str, Path]] = None):
        self.tracking_dir = Path(tracking_dir or "optimization_experiments")
        self.tracking_dir.mkdir(exist_ok=True)

        self.experiments = {}
        self.active_experiment: Optional[str] = None

        # Initialize MLflow if available
        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(f"file://{self.tracking_dir.absolute()}")
                logger.info("MLflow integration enabled")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")

    def start_experiment(self, experiment_name: str, description: str = "") -> str:
        """Start new experiment."""
        experiment_id = str(uuid.uuid4())

        experiment = {
            "id": experiment_id,
            "name": experiment_name,
            "description": description,
            "created_at": time.time(),
            "runs": [],
            "metadata": {},
        }

        self.experiments[experiment_id] = experiment
        self.active_experiment = experiment_id

        # MLflow integration
        if HAS_MLFLOW:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"MLflow experiment creation failed: {e}")

        logger.info(f"Started experiment '{experiment_name}' ({experiment_id})")
        return experiment_id

    def log_run(
        self,
        session: OptimizationSession,
        parameters: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        artifacts: Dict[str, Any] = None,
    ) -> None:
        """Log optimization run to current experiment."""
        if self.active_experiment is None:
            logger.warning("No active experiment for run logging")
            return

        if parameters is None:
            parameters = {}
        if metrics is None:
            metrics = {}
        if artifacts is None:
            artifacts = {}

        run_data = {
            "session": session.to_dict(),
            "parameters": parameters,
            "metrics": metrics,
            "artifacts": artifacts,
            "logged_at": time.time(),
        }

        # Add to experiment
        self.experiments[self.active_experiment]["runs"].append(run_data)

        # MLflow logging
        if HAS_MLFLOW:
            try:
                with mlflow.start_run():
                    # Log parameters
                    for key, value in parameters.items():
                        mlflow.log_param(key, value)

                    # Log metrics
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)

                    # Log session info
                    mlflow.log_metric("final_objective", session.final_objective or 0)
                    mlflow.log_metric("total_iterations", session.total_iterations)
                    mlflow.log_metric("duration", session.duration)
                    mlflow.log_param("strategy", session.strategy)
                    mlflow.log_param("success", session.success)

            except Exception as e:
                logger.warning(f"MLflow run logging failed: {e}")

        # Save to file
        self._save_experiment_data()

    def compare_runs(
        self, experiment_id: Optional[str] = None, metric: str = "final_objective"
    ) -> Dict[str, Any]:
        """Compare runs within an experiment."""
        if experiment_id is None:
            experiment_id = self.active_experiment

        if experiment_id not in self.experiments:
            return {}

        experiment = self.experiments[experiment_id]
        runs = experiment["runs"]

        if not runs:
            return {"message": "No runs to compare"}

        # Extract comparison data
        comparison_data = []
        for run in runs:
            session = run["session"]
            comparison_data.append(
                {
                    "session_id": session["session_id"],
                    "strategy": session["strategy"],
                    "success": session["success"],
                    "final_objective": session.get("final_objective"),
                    "duration": session["duration"],
                    "total_iterations": session["total_iterations"],
                    metric: run["metrics"].get(metric, session.get("final_objective")),
                }
            )

        # Sort by metric
        comparison_data.sort(key=lambda x: x.get(metric, float("inf")))

        # Summary statistics
        successful_runs = [r for r in comparison_data if r["success"]]

        summary = {
            "total_runs": len(comparison_data),
            "successful_runs": len(successful_runs),
            "success_rate": len(successful_runs) / len(comparison_data),
            "best_run": comparison_data[0] if comparison_data else None,
            "all_runs": comparison_data,
            "strategy_performance": self._analyze_strategy_performance(comparison_data),
        }

        return summary

    def _analyze_strategy_performance(
        self, comparison_data: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze performance by optimization strategy."""
        strategy_stats = defaultdict(list)

        for run in comparison_data:
            strategy = run["strategy"]
            if run["success"] and run["final_objective"] is not None:
                strategy_stats[strategy].append(run["final_objective"])

        performance = {}
        for strategy, objectives in strategy_stats.items():
            if objectives:
                performance[strategy] = {
                    "count": len(objectives),
                    "mean_objective": np.mean(objectives),
                    "std_objective": np.std(objectives),
                    "best_objective": min(objectives),
                }

        return performance

    def _save_experiment_data(self) -> None:
        """Save experiment data to file."""
        if self.active_experiment is None:
            return

        experiment = self.experiments[self.active_experiment]
        file_path = self.tracking_dir / f"experiment_{experiment['id']}.json"

        try:
            with open(file_path, "w") as f:
                json.dump(experiment, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save experiment data: {e}")


class OptimizationProfiler:
    """
    Performance profiler for optimization algorithms.

    Provides detailed performance analysis including bottleneck
    identification and resource utilization tracking.
    """

    def __init__(self):
        self.profiling_data = {}
        self.active_profile = None

    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling code sections."""
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            if self.active_profile is None:
                self.active_profile = {}

            if section_name not in self.active_profile:
                self.active_profile[section_name] = []

            self.active_profile[section_name].append(duration)

    def start_profiling(self, profile_id: str) -> None:
        """Start new profiling session."""
        self.active_profile = {}
        self.profiling_data[profile_id] = self.active_profile
        logger.debug(f"Started profiling session: {profile_id}")

    def end_profiling(self) -> Dict[str, Any]:
        """End current profiling session and return results."""
        if self.active_profile is None:
            return {}

        # Analyze profiling data
        analysis = {}
        total_time = 0

        for section_name, times in self.active_profile.items():
            section_total = sum(times)
            analysis[section_name] = {
                "total_time": section_total,
                "call_count": len(times),
                "avg_time": section_total / len(times),
                "min_time": min(times),
                "max_time": max(times),
            }
            total_time += section_total

        # Calculate percentages
        for section_data in analysis.values():
            section_data["percentage"] = (
                (section_data["total_time"] / total_time * 100) if total_time > 0 else 0
            )

        # Sort by total time
        sorted_analysis = dict(
            sorted(analysis.items(), key=lambda x: x[1]["total_time"], reverse=True)
        )

        result = {
            "total_time": total_time,
            "sections": sorted_analysis,
            "bottlenecks": [
                name
                for name, data in sorted_analysis.items()
                if data["percentage"] > 20
            ],  # >20% of total time
        }

        self.active_profile = None
        return result


# Global instances for easy access
global_monitor = PerformanceMonitor()
global_tracker = ExperimentTracker()
global_profiler = OptimizationProfiler()


# Convenience functions for common monitoring tasks


def start_monitoring_session(
    strategy: OptimizationStrategy,
    n_parameters: int,
    problem_difficulty: str = "unknown",
) -> str:
    """Start monitoring session."""
    return global_monitor.start_session(strategy, n_parameters, problem_difficulty)


def log_optimization_metrics(
    iteration: int,
    objective_value: float,
    gradient_norm: Optional[float] = None,
    **kwargs,
) -> None:
    """Log optimization metrics."""
    metrics = OptimizationMetrics(
        iteration=iteration,
        objective_value=objective_value,
        gradient_norm=gradient_norm,
        **kwargs,
    )
    global_monitor.log_metrics(metrics)


def end_monitoring_session(result: OptimizationResult) -> None:
    """End monitoring session."""
    global_monitor.end_session(result)


@contextmanager
def optimization_experiment(experiment_name: str, description: str = ""):
    """Context manager for experiment tracking."""
    experiment_id = global_tracker.start_experiment(experiment_name, description)
    try:
        yield experiment_id
    finally:
        # Experiment automatically saved via individual run logging
        pass


def create_optimization_report(
    session_id: Optional[str] = None, save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Create comprehensive optimization report."""
    summary = global_monitor.get_session_summary(session_id)

    if not summary:
        return {"error": "No session data available"}

    # Add system information
    try:
        import psutil
        import sys

        summary["system_info"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
    except ImportError:
        summary["system_info"] = {"note": "psutil not available for system info"}

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Optimization report saved to {save_path}")

    return summary
