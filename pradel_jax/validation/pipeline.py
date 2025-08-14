"""
Automated Validation Pipeline with Quality Gates (Phase 3).

This module provides end-to-end orchestration of the validation pipeline,
integrating all components from Phases 1 and 2 into a comprehensive
automated system with quality gates, parallel processing, and robust
error handling.

Key Features:
    - Automated pipeline orchestration
    - Parallel processing of multiple datasets/models
    - Quality gate evaluation with configurable criteria
    - Comprehensive error handling and recovery
    - Progress monitoring and performance tracking
    - Publication-ready reporting

Usage:
    from pradel_jax.validation import ValidationPipeline, get_validation_pipeline_config
    
    config = get_validation_pipeline_config()
    pipeline = ValidationPipeline(config)
    
    report = pipeline.run_comprehensive_validation(
        datasets=datasets,
        model_specifications=model_specs,
        output_dir=Path("validation_results")
    )
"""

import os
import uuid
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import json

import numpy as np
import pandas as pd

# Import validation components from previous phases
from .config import (
    ValidationPipelineConfig, ValidationStatus, QualityGateDecision,
    ValidationEnvironment, ValidationCriteria
)
from .secure_config import SecureValidationConfig
from .parameter_comparison import compare_parameter_estimates, compare_model_results
from .statistical_tests import test_parameter_equivalence, calculate_model_ranking_concordance
from .rmark_interface import RMarkExecutor, execute_rmark_analysis
from .advanced_statistics import (
    comprehensive_concordance_analysis, 
    cross_validation_stability_test,
    publication_ready_comparison_summary
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationSession:
    """Tracks a complete validation pipeline session."""
    
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Session configuration
    config: ValidationPipelineConfig = None
    datasets: List[Any] = field(default_factory=list)
    model_specifications: List[Any] = field(default_factory=list)
    output_directory: Optional[Path] = None
    
    # Execution status
    status: ValidationStatus = ValidationStatus.PASS
    total_models: int = 0
    completed_models: int = 0
    failed_models: int = 0
    warnings_count: int = 0
    
    # Results storage
    jax_results: List[Any] = field(default_factory=list)
    rmark_results: List[Any] = field(default_factory=list)
    validation_results: List[Any] = field(default_factory=list)
    
    # Performance metrics
    execution_times: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_models == 0:
            return 0.0
        return self.completed_models / self.total_models
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate
    
    def add_error(self, error_message: str) -> None:
        """Add error to session log."""
        timestamp = datetime.now().isoformat()
        self.error_log.append(f"{timestamp}: {error_message}")
        logger.error(error_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "status": self.status.value,
            "total_models": self.total_models,
            "completed_models": self.completed_models,
            "failed_models": self.failed_models,
            "warnings_count": self.warnings_count,
            "success_rate": self.success_rate,
            "error_count": len(self.error_log)
        }


@dataclass
class QualityGateReport:
    """Quality gate evaluation results."""
    
    overall_decision: QualityGateDecision
    pass_rate: float
    critical_failures: List[str]
    warnings: List[str]
    
    # Component assessments
    parameter_assessment: Dict[str, Any]
    model_assessment: Dict[str, Any]
    system_assessment: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    required_actions: List[str]
    
    criteria_used: ValidationCriteria
    
    def is_approved(self) -> bool:
        """Check if quality gates are approved."""
        return self.overall_decision in [
            QualityGateDecision.APPROVED,
            QualityGateDecision.APPROVED_WITH_WARNINGS
        ]


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    session: ValidationSession
    quality_gate_report: QualityGateReport
    
    # Report metadata
    report_id: str
    generation_time: datetime
    report_path: Path
    
    # Summary statistics
    summary_statistics: Dict[str, Any]
    
    # Detailed results
    parameter_results: List[Any]
    model_results: List[Any]
    statistical_test_results: Dict[str, Any]
    
    # Performance data
    performance_metrics: Dict[str, Any]
    
    def is_successful(self) -> bool:
        """Check if validation was successful overall."""
        return (
            self.session.status in [ValidationStatus.PASS, ValidationStatus.WARNING] and
            self.quality_gate_report.is_approved()
        )


class ValidationPipelineError(Exception):
    """Raised when validation pipeline encounters unrecoverable errors."""
    pass


class QualityGateEvaluator:
    """Evaluates validation results against quality gates."""
    
    def __init__(self, criteria: ValidationCriteria):
        self.criteria = criteria
        
    def evaluate_validation_results(
        self,
        session: ValidationSession,
        validation_results: List[Any]
    ) -> QualityGateReport:
        """
        Comprehensive quality gate evaluation.
        
        Args:
            session: Validation session with results
            validation_results: List of validation comparison results
            
        Returns:
            QualityGateReport: Complete quality assessment
        """
        
        logger.info("Evaluating quality gates...")
        
        # Parameter-level evaluation
        param_assessment = self._evaluate_parameter_quality(validation_results)
        
        # Model-level evaluation  
        model_assessment = self._evaluate_model_quality(session)
        
        # System-level evaluation
        system_assessment = self._evaluate_system_quality(session)
        
        # Calculate overall pass rate
        total_tests = len(validation_results)
        passed_tests = sum(1 for r in validation_results if getattr(r, 'status', ValidationStatus.FAIL) == ValidationStatus.PASS)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Collect critical failures and warnings
        critical_failures = []
        warnings = []
        
        # Check critical parameters
        for param_name in self.criteria.critical_parameters:
            param_results = [r for r in validation_results if getattr(r, 'parameter_name', '') == param_name]
            if param_results and not any(getattr(r, 'status', ValidationStatus.FAIL) == ValidationStatus.PASS for r in param_results):
                critical_failures.append(f"Critical parameter {param_name} failed validation")
        
        # Check minimum pass rate
        if pass_rate < self.criteria.min_pass_rate_for_approval:
            critical_failures.append(f"Pass rate {pass_rate:.1%} below minimum {self.criteria.min_pass_rate_for_approval:.1%}")
        
        # Check convergence rate
        if session.success_rate < self.criteria.min_convergence_rate:
            critical_failures.append(f"Convergence rate {session.success_rate:.1%} below minimum {self.criteria.min_convergence_rate:.1%}")
        
        # Determine overall decision
        overall_decision = self._make_overall_decision(
            pass_rate, critical_failures, warnings, param_assessment, model_assessment, system_assessment
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            param_assessment, model_assessment, system_assessment, critical_failures
        )
        
        required_actions = self._generate_required_actions(critical_failures)
        
        return QualityGateReport(
            overall_decision=overall_decision,
            pass_rate=pass_rate,
            critical_failures=critical_failures,
            warnings=warnings,
            parameter_assessment=param_assessment,
            model_assessment=model_assessment,
            system_assessment=system_assessment,
            recommendations=recommendations,
            required_actions=required_actions,
            criteria_used=self.criteria
        )
    
    def _evaluate_parameter_quality(self, validation_results: List[Any]) -> Dict[str, Any]:
        """Evaluate parameter-level quality metrics."""
        
        if not validation_results:
            return {"status": "no_results", "details": "No validation results available"}
        
        # Collect parameter-level statistics
        parameter_stats = {
            "total_parameters": len(validation_results),
            "passed_parameters": 0,
            "failed_parameters": 0,
            "warning_parameters": 0,
            "average_relative_difference": 0.0,
            "max_relative_difference": 0.0,
            "equivalence_test_pass_rate": 0.0
        }
        
        relative_differences = []
        equivalence_passes = 0
        
        for result in validation_results:
            status = getattr(result, 'status', ValidationStatus.FAIL)
            
            if status == ValidationStatus.PASS:
                parameter_stats["passed_parameters"] += 1
            elif status == ValidationStatus.WARNING:
                parameter_stats["warning_parameters"] += 1
            else:
                parameter_stats["failed_parameters"] += 1
            
            # Collect metrics if available
            rel_diff = getattr(result, 'relative_difference_pct', 0.0)
            if rel_diff > 0:
                relative_differences.append(rel_diff)
            
            equiv_conclusion = getattr(result, 'equivalence_conclusion', False)
            if equiv_conclusion:
                equivalence_passes += 1
        
        # Calculate aggregate statistics
        if relative_differences:
            parameter_stats["average_relative_difference"] = np.mean(relative_differences)
            parameter_stats["max_relative_difference"] = np.max(relative_differences)
        
        if len(validation_results) > 0:
            parameter_stats["equivalence_test_pass_rate"] = equivalence_passes / len(validation_results)
        
        # Determine parameter quality status
        if parameter_stats["passed_parameters"] / parameter_stats["total_parameters"] >= self.criteria.min_pass_rate_for_approval:
            status = "excellent" if parameter_stats["equivalence_test_pass_rate"] > 0.95 else "good"
        else:
            status = "poor"
        
        return {
            "status": status,
            "statistics": parameter_stats,
            "details": f"Parameter validation: {parameter_stats['passed_parameters']}/{parameter_stats['total_parameters']} passed"
        }
    
    def _evaluate_model_quality(self, session: ValidationSession) -> Dict[str, Any]:
        """Evaluate model-level quality metrics."""
        
        model_stats = {
            "total_models": session.total_models,
            "converged_models": session.completed_models,
            "failed_models": session.failed_models,
            "convergence_rate": session.success_rate
        }
        
        # Determine model quality status
        if session.success_rate >= self.criteria.min_convergence_rate:
            status = "excellent" if session.success_rate > 0.98 else "good"
        else:
            status = "poor"
        
        return {
            "status": status,
            "statistics": model_stats,
            "details": f"Model convergence: {session.completed_models}/{session.total_models} successful"
        }
    
    def _evaluate_system_quality(self, session: ValidationSession) -> Dict[str, Any]:
        """Evaluate system-level quality metrics."""
        
        # Calculate performance metrics
        total_time = session.duration.total_seconds() if session.duration else 0
        avg_time_per_model = total_time / session.total_models if session.total_models > 0 else 0
        
        system_stats = {
            "total_execution_time": total_time,
            "average_time_per_model": avg_time_per_model,
            "error_count": len(session.error_log),
            "warnings_count": session.warnings_count
        }
        
        # Determine system quality status
        if len(session.error_log) == 0 and session.warnings_count < 5:
            status = "excellent"
        elif len(session.error_log) < 3 and session.warnings_count < 10:
            status = "good"
        else:
            status = "poor"
        
        return {
            "status": status,
            "statistics": system_stats,
            "details": f"System stability: {len(session.error_log)} errors, {session.warnings_count} warnings"
        }
    
    def _make_overall_decision(
        self,
        pass_rate: float,
        critical_failures: List[str],
        warnings: List[str],
        param_assessment: Dict[str, Any],
        model_assessment: Dict[str, Any],
        system_assessment: Dict[str, Any]
    ) -> QualityGateDecision:
        """Make overall quality gate decision."""
        
        # Critical failures always reject
        if critical_failures:
            return QualityGateDecision.REJECTED
        
        # Check component assessments
        poor_components = [
            assessment for assessment in [param_assessment, model_assessment, system_assessment]
            if assessment.get("status") == "poor"
        ]
        
        if poor_components:
            return QualityGateDecision.MANUAL_REVIEW_REQUIRED
        
        # Approve with or without warnings
        if warnings or any(assessment.get("status") == "good" for assessment in [param_assessment, model_assessment, system_assessment]):
            return QualityGateDecision.APPROVED_WITH_WARNINGS
        else:
            return QualityGateDecision.APPROVED
    
    def _generate_recommendations(
        self,
        param_assessment: Dict[str, Any],
        model_assessment: Dict[str, Any],
        system_assessment: Dict[str, Any],
        critical_failures: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        
        recommendations = []
        
        if param_assessment.get("status") == "poor":
            recommendations.append("Review parameter estimation methods for improved accuracy")
            recommendations.append("Consider adjusting validation criteria tolerances")
        
        if model_assessment.get("status") == "poor":
            recommendations.append("Investigate convergence issues in model optimization")
            recommendations.append("Review initial parameter values and bounds")
        
        if system_assessment.get("status") == "poor":
            recommendations.append("Optimize pipeline performance and error handling")
            recommendations.append("Review system resources and configuration")
        
        if critical_failures:
            recommendations.append("Address critical failures before production deployment")
        
        if not recommendations:
            recommendations.append("Validation passed successfully - ready for production use")
        
        return recommendations
    
    def _generate_required_actions(self, critical_failures: List[str]) -> List[str]:
        """Generate required actions for critical failures."""
        
        if not critical_failures:
            return []
        
        required_actions = [
            "Do not deploy to production until critical failures are resolved",
            "Review and address each critical failure individually",
            "Re-run validation after fixes are implemented"
        ]
        
        return required_actions


class ValidationPipeline:
    """
    End-to-end validation pipeline with quality gates.
    
    Orchestrates the complete validation workflow from data preparation
    through statistical testing to quality gate evaluation and reporting.
    """
    
    def __init__(self, config: ValidationPipelineConfig):
        self.config = config
        self.quality_evaluator = QualityGateEvaluator(config.validation_criteria)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.rmark_executor = None
        if self.config.secure_config.has_rmark_config():
            try:
                self.rmark_executor = RMarkExecutor(self.config.secure_config)
                logger.info("RMark executor initialized successfully")
            except Exception as e:
                logger.warning(f"RMark executor initialization failed: {e}")
    
    def _setup_logging(self) -> None:
        """Configure logging for the pipeline."""
        
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure logger
        pipeline_logger = logging.getLogger(__name__)
        pipeline_logger.setLevel(log_level)
        
        # Add handler if not already present
        if not pipeline_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            pipeline_logger.addHandler(handler)
    
    def run_comprehensive_validation(
        self,
        datasets: List[Any],
        model_specifications: List[Any],
        output_dir: Path
    ) -> ValidationReport:
        """
        Execute complete validation pipeline.
        
        Args:
            datasets: List of data contexts to validate
            model_specifications: List of model specifications to test
            output_dir: Directory for output files
            
        Returns:
            ValidationReport: Comprehensive validation results
        """
        
        # Create validation session
        session = ValidationSession(
            session_id=str(uuid.uuid4()),
            start_time=datetime.now(),
            config=self.config,
            datasets=datasets,
            model_specifications=model_specifications,
            output_directory=output_dir,
            total_models=len(datasets) * len(model_specifications)
        )
        
        logger.info(f"Starting validation pipeline session {session.session_id}")
        logger.info(f"Validating {len(datasets)} datasets with {len(model_specifications)} model specifications")
        
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Execute JAX models
            logger.info("Executing JAX model fitting...")
            session.jax_results = self._execute_jax_models(session, datasets, model_specifications)
            
            # Step 2: Execute RMark models (if available)
            logger.info("Executing RMark model fitting...")
            session.rmark_results = self._execute_rmark_models(session, datasets, model_specifications)
            
            # Step 3: Statistical validation
            logger.info("Performing statistical validation...")
            session.validation_results = self._perform_statistical_validation(
                session, session.jax_results, session.rmark_results
            )
            
            # Step 4: Quality gate evaluation
            logger.info("Evaluating quality gates...")
            quality_report = self.quality_evaluator.evaluate_validation_results(
                session, session.validation_results
            )
            
            # Step 5: Generate report
            logger.info("Generating validation report...")
            report = self._generate_validation_report(session, quality_report, output_dir)
            
            # Update session status
            session.status = ValidationStatus.PASS if quality_report.is_approved() else ValidationStatus.FAIL
            
            logger.info(f"Validation pipeline completed: {session.status.value}")
            return report
            
        except Exception as e:
            session.add_error(f"Pipeline execution failed: {str(e)}")
            session.status = ValidationStatus.ERROR
            
            # Generate failure report
            failure_report = self._generate_failure_report(session, e, output_dir)
            raise ValidationPipelineError(f"Pipeline failed: {e}") from e
            
        finally:
            session.end_time = datetime.now()
            session.duration = session.end_time - session.start_time
            
            # Save session metadata
            self._save_session_metadata(session, output_dir)
    
    def _execute_jax_models(
        self,
        session: ValidationSession,
        datasets: List[Any],
        model_specifications: List[Any]
    ) -> List[Any]:
        """Execute JAX model fitting with parallel processing."""
        
        # This is a placeholder - in the real implementation, this would
        # integrate with the actual JAX model fitting framework
        jax_results = []
        
        if self.config.performance.enable_multiprocessing:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.performance.max_parallel_jobs) as executor:
                futures = []
                
                for dataset in datasets:
                    for model_spec in model_specifications:
                        future = executor.submit(self._fit_single_jax_model, dataset, model_spec)
                        futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.performance.single_model_timeout_seconds)
                        jax_results.append(result)
                        session.completed_models += 1
                    except Exception as e:
                        session.add_error(f"JAX model fitting failed: {str(e)}")
                        session.failed_models += 1
        else:
            # Sequential execution
            for dataset in datasets:
                for model_spec in model_specifications:
                    try:
                        result = self._fit_single_jax_model(dataset, model_spec)
                        jax_results.append(result)
                        session.completed_models += 1
                    except Exception as e:
                        session.add_error(f"JAX model fitting failed: {str(e)}")
                        session.failed_models += 1
        
        return jax_results
    
    def _execute_rmark_models(
        self,
        session: ValidationSession,
        datasets: List[Any],
        model_specifications: List[Any]
    ) -> List[Any]:
        """Execute RMark model fitting."""
        
        if not self.rmark_executor:
            logger.warning("RMark executor not available, skipping RMark validation")
            return []
        
        rmark_results = []
        
        for dataset in datasets:
            for model_spec in model_specifications:
                try:
                    result = self.rmark_executor.execute_rmark_analysis(dataset, model_spec)
                    rmark_results.append(result)
                except Exception as e:
                    session.add_error(f"RMark model fitting failed: {str(e)}")
                    # Continue with other models
        
        return rmark_results
    
    def _perform_statistical_validation(
        self,
        session: ValidationSession,
        jax_results: List[Any],
        rmark_results: List[Any]
    ) -> List[Any]:
        """Perform comprehensive statistical validation."""
        
        validation_results = []
        
        # Pair up results for comparison
        paired_results = self._pair_results(jax_results, rmark_results)
        
        for jax_result, rmark_result in paired_results:
            try:
                # Parameter comparison
                param_comparison = compare_parameter_estimates(jax_result, rmark_result)
                validation_results.append(param_comparison)
                
                # Model comparison
                model_comparison = compare_model_results(jax_result, rmark_result)
                validation_results.append(model_comparison)
                
                # Statistical tests
                equiv_test = test_parameter_equivalence(
                    jax_result.parameters,
                    rmark_result.parameters,
                    alpha=self.config.validation_criteria.equivalence_alpha
                )
                validation_results.append(equiv_test)
                
            except Exception as e:
                session.add_error(f"Statistical validation failed: {str(e)}")
        
        return validation_results
    
    def _fit_single_jax_model(self, dataset: Any, model_spec: Any) -> Any:
        """Fit a single JAX model (placeholder implementation)."""
        # This would integrate with the actual JAX model fitting
        import time
        time.sleep(0.1)  # Simulate computation
        
        # Return mock result structure
        return type('JAXResult', (), {
            'dataset_name': getattr(dataset, 'name', 'unknown'),
            'model_formula': str(model_spec),
            'parameters': {'phi_intercept': 0.85, 'p_intercept': 0.7, 'f_intercept': 0.1},
            'standard_errors': {'phi_intercept': 0.05, 'p_intercept': 0.04, 'f_intercept': 0.02},
            'aic': 245.6,
            'log_likelihood': -119.8,
            'success': True,
            'convergence_code': 0
        })()
    
    def _pair_results(self, jax_results: List[Any], rmark_results: List[Any]) -> List[Tuple[Any, Any]]:
        """Pair JAX and RMark results for comparison."""
        
        # Simple pairing by index for now
        # In production, this would match by dataset and model specification
        pairs = []
        
        min_length = min(len(jax_results), len(rmark_results))
        for i in range(min_length):
            pairs.append((jax_results[i], rmark_results[i]))
        
        return pairs
    
    def _generate_validation_report(
        self,
        session: ValidationSession,
        quality_report: QualityGateReport,
        output_dir: Path
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        
        report_id = f"validation_report_{session.session_id}"
        report_path = output_dir / f"{report_id}.html"
        
        # Calculate summary statistics
        summary_stats = {
            "total_models": session.total_models,
            "successful_models": session.completed_models,
            "failed_models": session.failed_models,
            "success_rate": session.success_rate,
            "total_validation_tests": len(session.validation_results),
            "quality_gate_decision": quality_report.overall_decision.value,
            "overall_pass_rate": quality_report.pass_rate
        }
        
        # Performance metrics
        performance_metrics = {
            "total_execution_time": session.duration.total_seconds() if session.duration else 0,
            "average_time_per_model": (session.duration.total_seconds() / session.total_models) if session.duration and session.total_models > 0 else 0,
            "pipeline_efficiency": session.success_rate,
            "memory_usage": session.memory_usage
        }
        
        report = ValidationReport(
            session=session,
            quality_gate_report=quality_report,
            report_id=report_id,
            generation_time=datetime.now(),
            report_path=report_path,
            summary_statistics=summary_stats,
            parameter_results=session.validation_results,
            model_results=session.jax_results + session.rmark_results,
            statistical_test_results={},
            performance_metrics=performance_metrics
        )
        
        # Generate HTML report (simplified version)
        self._write_html_report(report)
        
        # Generate JSON summary if requested
        if self.config.reporting.generate_json_summary:
            json_path = output_dir / f"{report_id}.json"
            self._write_json_summary(report, json_path)
        
        return report
    
    def _generate_failure_report(
        self,
        session: ValidationSession,
        error: Exception,
        output_dir: Path
    ) -> ValidationReport:
        """Generate report for failed validation pipeline."""
        
        # Create minimal failure report
        quality_report = QualityGateReport(
            overall_decision=QualityGateDecision.REJECTED,
            pass_rate=0.0,
            critical_failures=[f"Pipeline execution failed: {str(error)}"],
            warnings=[],
            parameter_assessment={"status": "error"},
            model_assessment={"status": "error"},
            system_assessment={"status": "error"},
            recommendations=["Fix pipeline errors before retrying validation"],
            required_actions=["Review error logs and fix underlying issues"],
            criteria_used=self.config.validation_criteria
        )
        
        report = ValidationReport(
            session=session,
            quality_gate_report=quality_report,
            report_id=f"failure_report_{session.session_id}",
            generation_time=datetime.now(),
            report_path=output_dir / f"failure_report_{session.session_id}.html",
            summary_statistics={"pipeline_status": "failed"},
            parameter_results=[],
            model_results=[],
            statistical_test_results={},
            performance_metrics={}
        )
        
        return report
    
    def _write_html_report(self, report: ValidationReport) -> None:
        """Write HTML validation report (simplified version)."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pradel-JAX Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .status-pass {{ color: green; font-weight: bold; }}
                .status-fail {{ color: red; font-weight: bold; }}
                .status-warning {{ color: orange; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Pradel-JAX Validation Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Generated:</strong> {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Session ID:</strong> {report.session.session_id}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Status:</strong> 
                    <span class="{'status-pass' if report.is_successful() else 'status-fail'}">
                        {report.quality_gate_report.overall_decision.value.replace('_', ' ').title()}
                    </span>
                </p>
                <p><strong>Success Rate:</strong> {report.session.success_rate:.1%}</p>
                <p><strong>Quality Gate Pass Rate:</strong> {report.quality_gate_report.pass_rate:.1%}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Models</td><td>{report.summary_statistics.get('total_models', 0)}</td></tr>
                    <tr><td>Successful Models</td><td>{report.summary_statistics.get('successful_models', 0)}</td></tr>
                    <tr><td>Failed Models</td><td>{report.summary_statistics.get('failed_models', 0)}</td></tr>
                    <tr><td>Execution Time</td><td>{report.performance_metrics.get('total_execution_time', 0):.1f} seconds</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Quality Gate Assessment</h2>
                {'<p class="status-fail">Critical Failures:</p><ul>' + ''.join(f'<li>{failure}</li>' for failure in report.quality_gate_report.critical_failures) + '</ul>' if report.quality_gate_report.critical_failures else '<p class="status-pass">No critical failures</p>'}
                
                {'<p class="status-warning">Warnings:</p><ul>' + ''.join(f'<li>{warning}</li>' for warning in report.quality_gate_report.warnings) + '</ul>' if report.quality_gate_report.warnings else ''}
                
                <p><strong>Recommendations:</strong></p>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in report.quality_gate_report.recommendations)}
                </ul>
            </div>
            
            <div class="section">
                <h2>Configuration Used</h2>
                <p><strong>Environment:</strong> {report.session.config.environment.value}</p>
                <p><strong>Parameter Tolerance:</strong> {report.session.config.validation_criteria.parameter_relative_tolerance_pct:.1f}%</p>
                <p><strong>AIC Difference Threshold:</strong> {report.session.config.validation_criteria.max_aic_difference}</p>
                <p><strong>Minimum Pass Rate:</strong> {report.session.config.validation_criteria.min_pass_rate_for_approval:.1%}</p>
            </div>
        </body>
        </html>
        """
        
        with open(report.report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report written to {report.report_path}")
    
    def _write_json_summary(self, report: ValidationReport, json_path: Path) -> None:
        """Write JSON summary of validation results."""
        
        summary = {
            "report_metadata": {
                "report_id": report.report_id,
                "generation_time": report.generation_time.isoformat(),
                "session_id": report.session.session_id
            },
            "summary_statistics": report.summary_statistics,
            "quality_gate_decision": report.quality_gate_report.overall_decision.value,
            "quality_gate_pass_rate": report.quality_gate_report.pass_rate,
            "critical_failures": report.quality_gate_report.critical_failures,
            "warnings": report.quality_gate_report.warnings,
            "recommendations": report.quality_gate_report.recommendations,
            "performance_metrics": report.performance_metrics,
            "session_summary": report.session.to_dict()
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"JSON summary written to {json_path}")
    
    def _save_session_metadata(self, session: ValidationSession, output_dir: Path) -> None:
        """Save session metadata for future reference."""
        
        metadata_path = output_dir / f"session_{session.session_id}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2, default=str)
        
        logger.debug(f"Session metadata saved to {metadata_path}")


def run_validation_pipeline(
    datasets: List[Any],
    model_specifications: List[Any],
    output_dir: Path,
    config: Optional[ValidationPipelineConfig] = None
) -> ValidationReport:
    """
    Convenience function to run validation pipeline with default configuration.
    
    Args:
        datasets: List of data contexts
        model_specifications: List of model specifications
        output_dir: Output directory
        config: Optional configuration (will use defaults if None)
        
    Returns:
        ValidationReport: Comprehensive validation results
    """
    
    if config is None:
        from .config import get_validation_pipeline_config
        config = get_validation_pipeline_config()
    
    pipeline = ValidationPipeline(config)
    return pipeline.run_comprehensive_validation(datasets, model_specifications, output_dir)