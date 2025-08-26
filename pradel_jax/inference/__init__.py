"""
Statistical inference module for pradel-jax.

Provides robust parameter uncertainty estimation, confidence intervals,
and model diagnostics with sound statistical foundations.
"""

from .uncertainty import (
    compute_hessian_standard_errors,
    compute_fisher_information,
    ParameterUncertainty,
    bootstrap_confidence_intervals,
)

from .diagnostics import (
    ModelDiagnostics,
    compute_model_selection_criteria,
    compute_goodness_of_fit_tests,
    plot_diagnostic_plots,
)

from .regression_tests import (
    PerformanceRegressionTester,
    run_performance_regression_tests,
)

__all__ = [
    "compute_hessian_standard_errors",
    "compute_fisher_information",
    "ParameterUncertainty",
    "bootstrap_confidence_intervals",
    "ModelDiagnostics",
    "compute_model_selection_criteria",
    "compute_goodness_of_fit_tests",
    "plot_diagnostic_plots",
    "PerformanceRegressionTester",
    "run_performance_regression_tests",
]
