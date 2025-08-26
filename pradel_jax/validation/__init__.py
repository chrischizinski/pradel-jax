"""
RMark Parameter Validation Framework for pradel-jax.

This module provides comprehensive statistical validation of JAX-based Pradel model
results against RMark implementations, following industry standards for numerical
validation and scientific reproducibility.

Key Components:
    - parameter_comparison: Statistical parameter comparison utilities
    - rmark_interface: Multi-environment RMark execution
    - statistical_tests: Industry-standard statistical testing
    - secure_config: Security-first configuration management
    - pipeline: Automated validation orchestration

Usage:
    import pradel_jax.validation as pv

    # Load secure configuration
    config = pv.get_secure_validation_config()

    # Run parameter comparison
    results = pv.compare_model_parameters(jax_result, rmark_result)

    # Execute full validation pipeline
    report = pv.run_validation_pipeline(datasets, models, config)

Security:
    All credentials and sensitive configuration are loaded from environment
    variables. No secrets are stored in code or configuration files committed
    to git. See docs/validation/security-setup.md for setup instructions.
"""

from typing import List, Dict, Any, Optional
import logging

# Core validation components (Phase 1)
from .secure_config import (
    SecureValidationConfig,
    get_secure_validation_config,
    SecurityError,
)

from .parameter_comparison import (
    ParameterComparisonResult,
    ModelComparisonResult,
    compare_parameter_estimates,
    compare_model_results,
)

from .statistical_tests import (
    StatisticalTestResult,
    EquivalenceTestResult,
    test_parameter_equivalence,
    test_confidence_interval_overlap,
    calculate_model_ranking_concordance,
)

# Advanced components (Phase 1 & 2)
try:
    from .rmark_interface import RMarkExecutor, RMarkResult, execute_rmark_analysis

    _RMARK_AVAILABLE = True
except ImportError:
    _RMARK_AVAILABLE = False

try:
    from .advanced_statistics import (
        BootstrapResult,
        BootstrapMethod,
        ConcordanceAnalysisResult,
        bootstrap_parameter_difference,
        comprehensive_concordance_analysis,
        cross_validation_stability_test,
        publication_ready_comparison_summary,
    )

    _ADVANCED_STATS_AVAILABLE = True
except ImportError:
    _ADVANCED_STATS_AVAILABLE = False

# Phase 3 components - Automated Pipeline & Quality Gates
try:
    from .config import (
        ValidationPipelineConfig,
        ValidationCriteria,
        PerformanceConfig,
        ReportingConfig,
        ValidationEnvironment,
        ValidationStatus,
        QualityGateDecision,
        get_validation_pipeline_config,
    )

    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

try:
    from .pipeline import (
        ValidationPipeline,
        ValidationSession,
        ValidationReport,
        run_validation_pipeline,
    )

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

try:
    from .quality_gates import (
        QualityGateEvaluator,
        QualityGateReport,
        ComponentAssessment,
        QualityMetric,
        RiskLevel,
    )

    _QUALITY_GATES_AVAILABLE = True
except ImportError:
    _QUALITY_GATES_AVAILABLE = False

try:
    from .parallel_executor import (
        ParallelValidationExecutor,
        TaskResult,
        ExecutionStats,
        execute_validation_tasks_parallel,
    )

    _PARALLEL_EXECUTION_AVAILABLE = True
except ImportError:
    _PARALLEL_EXECUTION_AVAILABLE = False

try:
    from .error_handling import (
        ValidationErrorHandler,
        ValidationError,
        CriticalValidationError,
        RecoverableValidationError,
        ErrorSeverity,
        ErrorCategory,
        RetryPolicy,
        error_handler_decorator,
        handle_data_processing_errors,
        handle_model_execution_errors,
        handle_rmark_interface_errors,
    )

    _ERROR_HANDLING_AVAILABLE = True
except ImportError:
    _ERROR_HANDLING_AVAILABLE = False

# Version and metadata
__version__ = "1.0.0-alpha"
__author__ = "Pradel-JAX Development Team"

# Configure logging
logger = logging.getLogger(__name__)

# Public API for Phase 1
__all__ = [
    # Core configuration
    "SecureValidationConfig",
    "get_secure_validation_config",
    "SecurityError",
    # Parameter comparison
    "ParameterComparisonResult",
    "ModelComparisonResult",
    "compare_parameter_estimates",
    "compare_model_results",
    # Statistical tests
    "StatisticalTestResult",
    "EquivalenceTestResult",
    "test_parameter_equivalence",
    "test_confidence_interval_overlap",
    "calculate_model_ranking_concordance",
    # Metadata
    "__version__",
    "is_rmark_available",
    "is_pipeline_available",
]

# Conditional exports (implemented phases)
if _RMARK_AVAILABLE:
    __all__.extend(["RMarkExecutor", "RMarkResult", "execute_rmark_analysis"])

if _ADVANCED_STATS_AVAILABLE:
    __all__.extend(
        [
            "BootstrapResult",
            "BootstrapMethod",
            "ConcordanceAnalysisResult",
            "bootstrap_parameter_difference",
            "comprehensive_concordance_analysis",
            "cross_validation_stability_test",
            "publication_ready_comparison_summary",
        ]
    )

# Phase 3 exports
if _CONFIG_AVAILABLE:
    __all__.extend(
        [
            "ValidationPipelineConfig",
            "ValidationCriteria",
            "PerformanceConfig",
            "ReportingConfig",
            "ValidationEnvironment",
            "ValidationStatus",
            "QualityGateDecision",
            "get_validation_pipeline_config",
        ]
    )

if _PIPELINE_AVAILABLE:
    __all__.extend(
        [
            "ValidationPipeline",
            "ValidationSession",
            "ValidationReport",
            "run_validation_pipeline",
        ]
    )

if _QUALITY_GATES_AVAILABLE:
    __all__.extend(
        [
            "QualityGateEvaluator",
            "QualityGateReport",
            "ComponentAssessment",
            "QualityMetric",
            "RiskLevel",
        ]
    )

if _PARALLEL_EXECUTION_AVAILABLE:
    __all__.extend(
        [
            "ParallelValidationExecutor",
            "TaskResult",
            "ExecutionStats",
            "execute_validation_tasks_parallel",
        ]
    )

if _ERROR_HANDLING_AVAILABLE:
    __all__.extend(
        [
            "ValidationErrorHandler",
            "ValidationError",
            "CriticalValidationError",
            "RecoverableValidationError",
            "ErrorSeverity",
            "ErrorCategory",
            "RetryPolicy",
            "error_handler_decorator",
            "handle_data_processing_errors",
            "handle_model_execution_errors",
            "handle_rmark_interface_errors",
        ]
    )


def is_rmark_available() -> bool:
    """Check if RMark execution interface is available."""
    return _RMARK_AVAILABLE


def is_advanced_stats_available() -> bool:
    """Check if advanced statistics are available."""
    return _ADVANCED_STATS_AVAILABLE


def is_pipeline_available() -> bool:
    """Check if validation pipeline is available."""
    return _PIPELINE_AVAILABLE


def is_config_available() -> bool:
    """Check if pipeline configuration is available."""
    return _CONFIG_AVAILABLE


def is_quality_gates_available() -> bool:
    """Check if quality gate evaluation is available."""
    return _QUALITY_GATES_AVAILABLE


def is_parallel_execution_available() -> bool:
    """Check if parallel execution is available."""
    return _PARALLEL_EXECUTION_AVAILABLE


def is_error_handling_available() -> bool:
    """Check if error handling framework is available."""
    return _ERROR_HANDLING_AVAILABLE


def is_phase3_complete() -> bool:
    """Check if Phase 3 components are fully available."""
    return all(
        [
            _CONFIG_AVAILABLE,
            _PIPELINE_AVAILABLE,
            _QUALITY_GATES_AVAILABLE,
            _PARALLEL_EXECUTION_AVAILABLE,
            _ERROR_HANDLING_AVAILABLE,
        ]
    )


def get_validation_info() -> Dict[str, Any]:
    """Get information about validation framework capabilities."""

    # Load configuration to check environment
    try:
        config = get_secure_validation_config()
        has_ssh_config = config.has_ssh_config()
        has_local_r = config.local_r_path is not None
    except Exception:
        has_ssh_config = False
        has_local_r = False

    return {
        "version": __version__,
        "phase_status": {
            "phase_1_core": True,  # Always available
            "phase_2_advanced_stats": _ADVANCED_STATS_AVAILABLE,
            "phase_3_pipeline": is_phase3_complete(),
        },
        "component_availability": {
            "rmark_interface": _RMARK_AVAILABLE,
            "advanced_statistics": _ADVANCED_STATS_AVAILABLE,
            "pipeline_orchestration": _PIPELINE_AVAILABLE,
            "configuration_system": _CONFIG_AVAILABLE,
            "quality_gates": _QUALITY_GATES_AVAILABLE,
            "parallel_execution": _PARALLEL_EXECUTION_AVAILABLE,
            "error_handling": _ERROR_HANDLING_AVAILABLE,
        },
        "environment_detection": {
            "ssh_configured": has_ssh_config,
            "local_r_available": has_local_r,
        },
        "statistical_capabilities": [
            "parameter_equivalence_tost",
            "confidence_interval_overlap",
            "model_ranking_concordance",
            "aic_difference_analysis",
            "bootstrap_confidence_intervals",
            "concordance_correlation_analysis",
            "cross_validation_stability",
        ],
        "pipeline_features": (
            [
                "automated_orchestration",
                "quality_gate_evaluation",
                "parallel_processing",
                "comprehensive_error_handling",
                "configurable_retry_logic",
                "performance_monitoring",
                "publication_ready_reporting",
            ]
            if is_phase3_complete()
            else []
        ),
        "supported_environments": [
            "ssh_windows_rmark",
            "local_r_installation",
            "mock_validation_development",
            "ci_cd_pipeline",
            "cloud_execution",
        ],
        "quality_gate_criteria": (
            [
                "parameter_accuracy_thresholds",
                "model_concordance_analysis",
                "statistical_power_assessment",
                "convergence_reliability",
                "system_stability_monitoring",
            ]
            if _QUALITY_GATES_AVAILABLE
            else []
        ),
    }


# Initialize validation framework
logger.info(f"Pradel-JAX Validation Framework v{__version__} loaded")
logger.info(f"Phase 1 (Core): available")
logger.info(
    f"Phase 2 (Advanced Statistics): {'available' if _ADVANCED_STATS_AVAILABLE else 'not available'}"
)
logger.info(
    f"Phase 3 (Pipeline & Quality Gates): {'complete' if is_phase3_complete() else 'partial'}"
)
logger.info(
    f"  - Configuration system: {'available' if _CONFIG_AVAILABLE else 'not available'}"
)
logger.info(
    f"  - Pipeline orchestration: {'available' if _PIPELINE_AVAILABLE else 'not available'}"
)
logger.info(
    f"  - Quality gates: {'available' if _QUALITY_GATES_AVAILABLE else 'not available'}"
)
logger.info(
    f"  - Parallel execution: {'available' if _PARALLEL_EXECUTION_AVAILABLE else 'not available'}"
)
logger.info(
    f"  - Error handling: {'available' if _ERROR_HANDLING_AVAILABLE else 'not available'}"
)
logger.info(f"RMark interface: {'available' if _RMARK_AVAILABLE else 'not available'}")
