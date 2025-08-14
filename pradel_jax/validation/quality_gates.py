"""
Quality Gate Evaluation Framework (Phase 3).

This module provides comprehensive quality gate evaluation for validation
pipeline results, implementing industry-standard decision criteria and
automated approval/rejection logic for JAX vs RMark parameter validation.

Key Features:
    - Configurable quality criteria with industry standards
    - Multi-level assessment (parameter, model, system)
    - Automated decision making with clear rationale
    - Publication-quality approval criteria
    - Risk-based quality thresholds

Usage:
    evaluator = QualityGateEvaluator(validation_criteria)
    report = evaluator.evaluate_validation_results(session, results)
    
    if report.is_approved():
        print("Validation passed quality gates")
    else:
        print(f"Validation failed: {report.overall_decision}")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

from .config import ValidationCriteria, ValidationStatus, QualityGateDecision

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Quality metrics for assessment."""
    PARAMETER_ACCURACY = "parameter_accuracy"
    MODEL_CONCORDANCE = "model_concordance" 
    STATISTICAL_POWER = "statistical_power"
    CONVERGENCE_RELIABILITY = "convergence_reliability"
    SYSTEM_STABILITY = "system_stability"
    PERFORMANCE_EFFICIENCY = "performance_efficiency"


class RiskLevel(Enum):
    """Risk levels for quality assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityThreshold:
    """Configurable threshold for quality metrics."""
    
    metric: QualityMetric
    excellent_threshold: float
    good_threshold: float
    acceptable_threshold: float
    critical_parameters: List[str] = field(default_factory=list)
    weight: float = 1.0  # Relative importance
    
    def assess_value(self, value: float) -> Tuple[str, RiskLevel]:
        """Assess a metric value against thresholds."""
        
        if value >= self.excellent_threshold:
            return "excellent", RiskLevel.LOW
        elif value >= self.good_threshold:
            return "good", RiskLevel.LOW
        elif value >= self.acceptable_threshold:
            return "acceptable", RiskLevel.MEDIUM
        else:
            return "poor", RiskLevel.HIGH


@dataclass
class ComponentAssessment:
    """Assessment of a quality component."""
    
    component_name: str
    overall_score: float  # 0.0 to 1.0
    status: str  # excellent, good, acceptable, poor
    risk_level: RiskLevel
    
    # Detailed metrics
    metric_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    metric_assessments: Dict[QualityMetric, str] = field(default_factory=dict)
    
    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Supporting evidence
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateReport:
    """Comprehensive quality gate evaluation report."""
    
    # Overall decision
    overall_decision: QualityGateDecision
    overall_score: float  # Weighted average of all components
    pass_rate: float
    
    # Component assessments
    parameter_assessment: ComponentAssessment
    model_assessment: ComponentAssessment
    system_assessment: ComponentAssessment
    
    # Risk analysis
    risk_level: RiskLevel
    critical_failures: List[str]
    warnings: List[str]
    
    # Recommendations and actions
    recommendations: List[str]
    required_actions: List[str]
    
    # Metadata
    criteria_used: ValidationCriteria
    
    # Fields with defaults must come last
    approval_conditions: List[str] = field(default_factory=list)
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    evaluator_version: str = "1.0.0"
    
    def is_approved(self) -> bool:
        """Check if validation is approved for production use."""
        return self.overall_decision in [
            QualityGateDecision.APPROVED,
            QualityGateDecision.APPROVED_WITH_WARNINGS
        ]
    
    def requires_manual_review(self) -> bool:
        """Check if manual review is required."""
        return self.overall_decision == QualityGateDecision.MANUAL_REVIEW_REQUIRED
    
    def is_rejected(self) -> bool:
        """Check if validation is rejected."""
        return self.overall_decision == QualityGateDecision.REJECTED
    
    def get_approval_summary(self) -> str:
        """Get human-readable approval summary."""
        
        if self.is_approved():
            if self.overall_decision == QualityGateDecision.APPROVED:
                return f"APPROVED: Validation passed all quality gates (score: {self.overall_score:.2f})"
            else:
                return f"APPROVED WITH WARNINGS: Validation passed with {len(self.warnings)} warnings (score: {self.overall_score:.2f})"
        elif self.requires_manual_review():
            return f"MANUAL REVIEW REQUIRED: {len(self.critical_failures)} issues need investigation (score: {self.overall_score:.2f})"
        else:
            return f"REJECTED: Validation failed quality gates (score: {self.overall_score:.2f})"


class QualityGateEvaluator:
    """
    Comprehensive quality gate evaluator for validation pipeline results.
    
    Implements multi-level assessment with configurable thresholds and
    industry-standard decision criteria for parameter validation.
    """
    
    def __init__(self, criteria: ValidationCriteria):
        self.criteria = criteria
        self.thresholds = self._initialize_quality_thresholds()
        
    def _initialize_quality_thresholds(self) -> Dict[QualityMetric, QualityThreshold]:
        """Initialize quality thresholds based on validation criteria."""
        
        return {
            QualityMetric.PARAMETER_ACCURACY: QualityThreshold(
                metric=QualityMetric.PARAMETER_ACCURACY,
                excellent_threshold=0.99,  # 99% accuracy
                good_threshold=0.95,       # 95% accuracy  
                acceptable_threshold=0.90, # 90% accuracy
                critical_parameters=self.criteria.critical_parameters,
                weight=2.0  # High importance
            ),
            
            QualityMetric.MODEL_CONCORDANCE: QualityThreshold(
                metric=QualityMetric.MODEL_CONCORDANCE,
                excellent_threshold=0.95,  # 95% model agreement
                good_threshold=0.85,       # 85% model agreement
                acceptable_threshold=self.criteria.min_ranking_concordance,  # 80% from config
                weight=1.5
            ),
            
            QualityMetric.STATISTICAL_POWER: QualityThreshold(
                metric=QualityMetric.STATISTICAL_POWER,
                excellent_threshold=0.95,  # 95% statistical power
                good_threshold=0.90,       # 90% statistical power
                acceptable_threshold=self.criteria.min_statistical_power,  # 80% from config
                weight=1.0
            ),
            
            QualityMetric.CONVERGENCE_RELIABILITY: QualityThreshold(
                metric=QualityMetric.CONVERGENCE_RELIABILITY,
                excellent_threshold=0.98,  # 98% convergence
                good_threshold=self.criteria.min_convergence_rate,  # 95% from config
                acceptable_threshold=0.90, # 90% convergence
                weight=1.8  # High importance for reliability
            ),
            
            QualityMetric.SYSTEM_STABILITY: QualityThreshold(
                metric=QualityMetric.SYSTEM_STABILITY,
                excellent_threshold=0.98,  # 98% stability (few errors/warnings)
                good_threshold=0.95,       # 95% stability
                acceptable_threshold=0.90, # 90% stability
                weight=1.0
            ),
            
            QualityMetric.PERFORMANCE_EFFICIENCY: QualityThreshold(
                metric=QualityMetric.PERFORMANCE_EFFICIENCY,
                excellent_threshold=0.95,  # 95% efficiency
                good_threshold=0.85,       # 85% efficiency
                acceptable_threshold=0.75, # 75% efficiency
                weight=0.5  # Lower importance
            )
        }
    
    def evaluate_validation_results(
        self,
        session: Any,  # ValidationSession
        validation_results: List[Any]
    ) -> QualityGateReport:
        """
        Perform comprehensive quality gate evaluation.
        
        Args:
            session: Validation session with execution data
            validation_results: List of validation comparison results
            
        Returns:
            QualityGateReport: Complete quality assessment with decision
        """
        
        logger.info("Starting comprehensive quality gate evaluation...")
        
        # Assess each quality component
        parameter_assessment = self._assess_parameter_quality(validation_results)
        model_assessment = self._assess_model_quality(session, validation_results)
        system_assessment = self._assess_system_quality(session)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score([
            parameter_assessment, model_assessment, system_assessment
        ])
        
        # Determine overall risk level
        risk_level = self._determine_overall_risk([
            parameter_assessment, model_assessment, system_assessment
        ])
        
        # Collect critical failures and warnings
        critical_failures, warnings = self._collect_issues([
            parameter_assessment, model_assessment, system_assessment
        ])
        
        # Add criteria-specific failures
        criteria_failures = self._check_criteria_compliance(
            session, validation_results, parameter_assessment, model_assessment
        )
        critical_failures.extend(criteria_failures)
        
        # Calculate pass rate
        pass_rate = self._calculate_pass_rate(validation_results)
        
        # Make overall decision
        overall_decision = self._make_overall_decision(
            overall_score, risk_level, critical_failures, warnings,
            parameter_assessment, model_assessment, system_assessment
        )
        
        # Generate recommendations and required actions
        recommendations = self._generate_recommendations([
            parameter_assessment, model_assessment, system_assessment
        ])
        
        required_actions = self._generate_required_actions(
            overall_decision, critical_failures, risk_level
        )
        
        approval_conditions = self._generate_approval_conditions(
            overall_decision, warnings, risk_level
        )
        
        report = QualityGateReport(
            overall_decision=overall_decision,
            overall_score=overall_score,
            pass_rate=pass_rate,
            parameter_assessment=parameter_assessment,
            model_assessment=model_assessment,
            system_assessment=system_assessment,
            risk_level=risk_level,
            critical_failures=critical_failures,
            warnings=warnings,
            recommendations=recommendations,
            required_actions=required_actions,
            approval_conditions=approval_conditions,
            criteria_used=self.criteria
        )
        
        logger.info(f"Quality gate evaluation completed: {overall_decision.value} (score: {overall_score:.3f})")
        return report
    
    def _assess_parameter_quality(self, validation_results: List[Any]) -> ComponentAssessment:
        """Assess parameter-level quality metrics."""
        
        if not validation_results:
            return ComponentAssessment(
                component_name="Parameter Quality",
                overall_score=0.0,
                status="no_data",
                risk_level=RiskLevel.CRITICAL,
                issues=["No validation results available"],
                recommendations=["Ensure parameter comparison is executed"]
            )
        
        # Extract parameter metrics
        parameter_metrics = self._extract_parameter_metrics(validation_results)
        
        # Assess each metric
        metric_scores = {}
        metric_assessments = {}
        issues = []
        recommendations = []
        
        # Parameter accuracy assessment
        accuracy_score = parameter_metrics.get("accuracy_rate", 0.0)
        accuracy_status, accuracy_risk = self.thresholds[QualityMetric.PARAMETER_ACCURACY].assess_value(accuracy_score)
        metric_scores[QualityMetric.PARAMETER_ACCURACY] = accuracy_score
        metric_assessments[QualityMetric.PARAMETER_ACCURACY] = accuracy_status
        
        if accuracy_status == "poor":
            issues.append(f"Parameter accuracy below threshold: {accuracy_score:.1%}")
            recommendations.append("Review parameter estimation methods and numerical precision")
        
        # Statistical power assessment
        power_score = parameter_metrics.get("statistical_power", 0.0)
        power_status, power_risk = self.thresholds[QualityMetric.STATISTICAL_POWER].assess_value(power_score)
        metric_scores[QualityMetric.STATISTICAL_POWER] = power_score
        metric_assessments[QualityMetric.STATISTICAL_POWER] = power_status
        
        if power_status == "poor":
            issues.append(f"Statistical power insufficient: {power_score:.1%}")
            recommendations.append("Increase sample size or adjust significance levels")
        
        # Check critical parameters
        critical_param_issues = self._check_critical_parameters(validation_results)
        issues.extend(critical_param_issues)
        
        # Calculate overall component score
        weighted_score = (
            metric_scores[QualityMetric.PARAMETER_ACCURACY] * self.thresholds[QualityMetric.PARAMETER_ACCURACY].weight +
            metric_scores[QualityMetric.STATISTICAL_POWER] * self.thresholds[QualityMetric.STATISTICAL_POWER].weight
        ) / (self.thresholds[QualityMetric.PARAMETER_ACCURACY].weight + self.thresholds[QualityMetric.STATISTICAL_POWER].weight)
        
        # Determine overall status
        overall_status, risk_level = self._determine_component_status(weighted_score, issues)
        
        return ComponentAssessment(
            component_name="Parameter Quality",
            overall_score=weighted_score,
            status=overall_status,
            risk_level=risk_level,
            metric_scores=metric_scores,
            metric_assessments=metric_assessments,
            issues=issues,
            recommendations=recommendations,
            evidence=parameter_metrics
        )
    
    def _assess_model_quality(self, session: Any, validation_results: List[Any]) -> ComponentAssessment:
        """Assess model-level quality metrics."""
        
        # Extract model metrics
        model_metrics = self._extract_model_metrics(session, validation_results)
        
        metric_scores = {}
        metric_assessments = {}
        issues = []
        recommendations = []
        
        # Model concordance assessment
        concordance_score = model_metrics.get("model_concordance", 0.0)
        concordance_status, concordance_risk = self.thresholds[QualityMetric.MODEL_CONCORDANCE].assess_value(concordance_score)
        metric_scores[QualityMetric.MODEL_CONCORDANCE] = concordance_score
        metric_assessments[QualityMetric.MODEL_CONCORDANCE] = concordance_status
        
        if concordance_status == "poor":
            issues.append(f"Model concordance below threshold: {concordance_score:.1%}")
            recommendations.append("Investigate differences in model implementation or optimization")
        
        # Convergence reliability assessment
        convergence_score = model_metrics.get("convergence_rate", 0.0)
        convergence_status, convergence_risk = self.thresholds[QualityMetric.CONVERGENCE_RELIABILITY].assess_value(convergence_score)
        metric_scores[QualityMetric.CONVERGENCE_RELIABILITY] = convergence_score
        metric_assessments[QualityMetric.CONVERGENCE_RELIABILITY] = convergence_status
        
        if convergence_status == "poor":
            issues.append(f"Convergence rate below threshold: {convergence_score:.1%}")
            recommendations.append("Review optimization algorithms and initial parameter values")
        
        # Calculate weighted score
        weighted_score = (
            concordance_score * self.thresholds[QualityMetric.MODEL_CONCORDANCE].weight +
            convergence_score * self.thresholds[QualityMetric.CONVERGENCE_RELIABILITY].weight
        ) / (self.thresholds[QualityMetric.MODEL_CONCORDANCE].weight + self.thresholds[QualityMetric.CONVERGENCE_RELIABILITY].weight)
        
        overall_status, risk_level = self._determine_component_status(weighted_score, issues)
        
        return ComponentAssessment(
            component_name="Model Quality",
            overall_score=weighted_score,
            status=overall_status,
            risk_level=risk_level,
            metric_scores=metric_scores,
            metric_assessments=metric_assessments,
            issues=issues,
            recommendations=recommendations,
            evidence=model_metrics
        )
    
    def _assess_system_quality(self, session: Any) -> ComponentAssessment:
        """Assess system-level quality metrics."""
        
        # Extract system metrics
        system_metrics = self._extract_system_metrics(session)
        
        metric_scores = {}
        metric_assessments = {}
        issues = []
        recommendations = []
        
        # System stability assessment
        stability_score = system_metrics.get("stability_rate", 0.0)
        stability_status, stability_risk = self.thresholds[QualityMetric.SYSTEM_STABILITY].assess_value(stability_score)
        metric_scores[QualityMetric.SYSTEM_STABILITY] = stability_score
        metric_assessments[QualityMetric.SYSTEM_STABILITY] = stability_status
        
        if stability_status == "poor":
            issues.append(f"System stability below threshold: {stability_score:.1%}")
            recommendations.append("Review error handling and system resources")
        
        # Performance efficiency assessment
        efficiency_score = system_metrics.get("performance_efficiency", 0.0)
        efficiency_status, efficiency_risk = self.thresholds[QualityMetric.PERFORMANCE_EFFICIENCY].assess_value(efficiency_score)
        metric_scores[QualityMetric.PERFORMANCE_EFFICIENCY] = efficiency_score
        metric_assessments[QualityMetric.PERFORMANCE_EFFICIENCY] = efficiency_status
        
        if efficiency_status == "poor":
            issues.append(f"Performance efficiency below threshold: {efficiency_score:.1%}")
            recommendations.append("Optimize pipeline performance and resource usage")
        
        # Calculate weighted score
        weighted_score = (
            stability_score * self.thresholds[QualityMetric.SYSTEM_STABILITY].weight +
            efficiency_score * self.thresholds[QualityMetric.PERFORMANCE_EFFICIENCY].weight
        ) / (self.thresholds[QualityMetric.SYSTEM_STABILITY].weight + self.thresholds[QualityMetric.PERFORMANCE_EFFICIENCY].weight)
        
        overall_status, risk_level = self._determine_component_status(weighted_score, issues)
        
        return ComponentAssessment(
            component_name="System Quality",
            overall_score=weighted_score,
            status=overall_status,
            risk_level=risk_level,
            metric_scores=metric_scores,
            metric_assessments=metric_assessments,
            issues=issues,
            recommendations=recommendations,
            evidence=system_metrics
        )
    
    def _extract_parameter_metrics(self, validation_results: List[Any]) -> Dict[str, Any]:
        """Extract parameter-level metrics from validation results."""
        
        if not validation_results:
            return {"accuracy_rate": 0.0, "statistical_power": 0.0}
        
        # Count passed/failed parameter validations
        passed_count = 0
        total_count = len(validation_results)
        equivalence_passes = 0
        relative_differences = []
        
        for result in validation_results:
            # Check if result has validation status
            status = getattr(result, 'status', ValidationStatus.FAIL)
            if status == ValidationStatus.PASS:
                passed_count += 1
            
            # Check equivalence test results
            equiv_conclusion = getattr(result, 'equivalence_conclusion', False)
            if equiv_conclusion:
                equivalence_passes += 1
            
            # Collect relative differences
            rel_diff = getattr(result, 'relative_difference_pct', 0.0)
            if rel_diff > 0:
                relative_differences.append(rel_diff)
        
        accuracy_rate = passed_count / total_count if total_count > 0 else 0.0
        statistical_power = equivalence_passes / total_count if total_count > 0 else 0.0
        
        return {
            "accuracy_rate": accuracy_rate,
            "statistical_power": statistical_power,
            "total_parameters": total_count,
            "passed_parameters": passed_count,
            "failed_parameters": total_count - passed_count,
            "equivalence_passes": equivalence_passes,
            "average_relative_difference": np.mean(relative_differences) if relative_differences else 0.0,
            "max_relative_difference": np.max(relative_differences) if relative_differences else 0.0
        }
    
    def _extract_model_metrics(self, session: Any, validation_results: List[Any]) -> Dict[str, Any]:
        """Extract model-level metrics from session and validation results."""
        
        # Basic convergence metrics
        convergence_rate = getattr(session, 'success_rate', 0.0)
        total_models = getattr(session, 'total_models', 0)
        completed_models = getattr(session, 'completed_models', 0)
        
        # Model concordance (simplified calculation)
        # In real implementation, this would compare AIC rankings, etc.
        model_concordance = 0.85  # Placeholder - would calculate from actual results
        
        return {
            "convergence_rate": convergence_rate,
            "model_concordance": model_concordance,
            "total_models": total_models,
            "completed_models": completed_models,
            "failed_models": total_models - completed_models
        }
    
    def _extract_system_metrics(self, session: Any) -> Dict[str, Any]:
        """Extract system-level metrics from session."""
        
        # Error and warning counts
        error_count = len(getattr(session, 'error_log', []))
        warnings_count = getattr(session, 'warnings_count', 0)
        total_operations = getattr(session, 'total_models', 1)  # Avoid division by zero
        
        # Calculate stability rate (1 - error rate)
        stability_rate = max(0.0, 1.0 - (error_count + warnings_count * 0.5) / total_operations)
        
        # Performance efficiency (simplified)
        # In real implementation, this would consider execution time, memory usage, etc.
        performance_efficiency = 0.85  # Placeholder
        
        return {
            "stability_rate": stability_rate,
            "performance_efficiency": performance_efficiency,
            "error_count": error_count,
            "warnings_count": warnings_count,
            "total_operations": total_operations
        }
    
    def _check_critical_parameters(self, validation_results: List[Any]) -> List[str]:
        """Check if critical parameters pass validation."""
        
        issues = []
        
        for param_name in self.criteria.critical_parameters:
            # Find results for this parameter
            param_results = [
                r for r in validation_results 
                if getattr(r, 'parameter_name', '') == param_name
            ]
            
            if not param_results:
                issues.append(f"Critical parameter '{param_name}' not found in validation results")
                continue
            
            # Check if any result for this parameter passed
            passed = any(
                getattr(r, 'status', ValidationStatus.FAIL) == ValidationStatus.PASS 
                for r in param_results
            )
            
            if not passed:
                issues.append(f"Critical parameter '{param_name}' failed validation")
        
        return issues
    
    def _check_criteria_compliance(
        self,
        session: Any,
        validation_results: List[Any],
        parameter_assessment: ComponentAssessment,
        model_assessment: ComponentAssessment
    ) -> List[str]:
        """Check compliance with validation criteria."""
        
        failures = []
        
        # Check minimum pass rate
        pass_rate = self._calculate_pass_rate(validation_results)
        if pass_rate < self.criteria.min_pass_rate_for_approval:
            failures.append(
                f"Overall pass rate {pass_rate:.1%} below minimum required {self.criteria.min_pass_rate_for_approval:.1%}"
            )
        
        # Check convergence rate
        convergence_rate = getattr(session, 'success_rate', 0.0)
        if convergence_rate < self.criteria.min_convergence_rate:
            failures.append(
                f"Convergence rate {convergence_rate:.1%} below minimum required {self.criteria.min_convergence_rate:.1%}"
            )
        
        # Check parameter accuracy
        param_accuracy = parameter_assessment.metric_scores.get(QualityMetric.PARAMETER_ACCURACY, 0.0)
        if param_accuracy < 0.9:  # 90% minimum for critical approval
            failures.append(
                f"Parameter accuracy {param_accuracy:.1%} below critical threshold"
            )
        
        return failures
    
    def _calculate_pass_rate(self, validation_results: List[Any]) -> float:
        """Calculate overall validation pass rate."""
        
        if not validation_results:
            return 0.0
        
        passed_count = sum(
            1 for result in validation_results
            if getattr(result, 'status', ValidationStatus.FAIL) == ValidationStatus.PASS
        )
        
        return passed_count / len(validation_results)
    
    def _calculate_overall_score(self, assessments: List[ComponentAssessment]) -> float:
        """Calculate weighted overall score from component assessments."""
        
        # Component weights
        weights = {
            "Parameter Quality": 0.5,  # Highest weight
            "Model Quality": 0.3,
            "System Quality": 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for assessment in assessments:
            weight = weights.get(assessment.component_name, 1.0)
            total_score += assessment.overall_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_overall_risk(self, assessments: List[ComponentAssessment]) -> RiskLevel:
        """Determine overall risk level from component assessments."""
        
        risk_levels = [assessment.risk_level for assessment in assessments]
        
        # If any component is critical, overall is critical
        if RiskLevel.CRITICAL in risk_levels:
            return RiskLevel.CRITICAL
        elif RiskLevel.HIGH in risk_levels:
            return RiskLevel.HIGH
        elif RiskLevel.MEDIUM in risk_levels:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _collect_issues(self, assessments: List[ComponentAssessment]) -> Tuple[List[str], List[str]]:
        """Collect critical failures and warnings from component assessments."""
        
        critical_failures = []
        warnings = []
        
        for assessment in assessments:
            for issue in assessment.issues:
                if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                    critical_failures.append(f"{assessment.component_name}: {issue}")
                else:
                    warnings.append(f"{assessment.component_name}: {issue}")
        
        return critical_failures, warnings
    
    def _determine_component_status(self, score: float, issues: List[str]) -> Tuple[str, RiskLevel]:
        """Determine component status and risk level from score and issues."""
        
        # Check for critical issues first
        if any("critical" in issue.lower() for issue in issues):
            return "poor", RiskLevel.CRITICAL
        
        # Score-based assessment
        if score >= 0.95:
            return "excellent", RiskLevel.LOW
        elif score >= 0.85:
            return "good", RiskLevel.LOW if not issues else RiskLevel.MEDIUM
        elif score >= 0.70:
            return "acceptable", RiskLevel.MEDIUM
        else:
            return "poor", RiskLevel.HIGH
    
    def _make_overall_decision(
        self,
        overall_score: float,
        risk_level: RiskLevel,
        critical_failures: List[str],
        warnings: List[str],
        parameter_assessment: ComponentAssessment,
        model_assessment: ComponentAssessment,
        system_assessment: ComponentAssessment
    ) -> QualityGateDecision:
        """Make final quality gate decision based on all factors."""
        
        # Critical failures always result in rejection
        if critical_failures:
            return QualityGateDecision.REJECTED
        
        # Critical risk level requires manual review
        if risk_level == RiskLevel.CRITICAL:
            return QualityGateDecision.MANUAL_REVIEW_REQUIRED
        
        # Parameter quality is critical - if poor, require manual review
        if parameter_assessment.status == "poor":
            return QualityGateDecision.MANUAL_REVIEW_REQUIRED
        
        # High risk or poor model quality requires manual review
        if risk_level == RiskLevel.HIGH or model_assessment.status == "poor":
            return QualityGateDecision.MANUAL_REVIEW_REQUIRED
        
        # Score-based decisions
        if overall_score >= 0.90:
            if warnings:
                return QualityGateDecision.APPROVED_WITH_WARNINGS
            else:
                return QualityGateDecision.APPROVED
        elif overall_score >= 0.75:
            return QualityGateDecision.APPROVED_WITH_WARNINGS
        else:
            return QualityGateDecision.MANUAL_REVIEW_REQUIRED
    
    def _generate_recommendations(self, assessments: List[ComponentAssessment]) -> List[str]:
        """Generate recommendations from component assessments."""
        
        all_recommendations = []
        
        for assessment in assessments:
            all_recommendations.extend(assessment.recommendations)
        
        # Add general recommendations
        if not all_recommendations:
            all_recommendations.append("Validation completed successfully - system ready for production use")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations
    
    def _generate_required_actions(
        self,
        decision: QualityGateDecision,
        critical_failures: List[str],
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate required actions based on decision and risk level."""
        
        actions = []
        
        if decision == QualityGateDecision.REJECTED:
            actions.extend([
                "Do not deploy to production",
                "Address all critical failures",
                "Re-run complete validation after fixes"
            ])
        
        elif decision == QualityGateDecision.MANUAL_REVIEW_REQUIRED:
            actions.extend([
                "Conduct manual review of identified issues",
                "Obtain stakeholder approval before production deployment",
                "Document review decisions and rationale"
            ])
        
        elif decision == QualityGateDecision.APPROVED_WITH_WARNINGS:
            actions.extend([
                "Document known warnings and their impact",
                "Monitor production deployment closely",
                "Plan improvements for next release cycle"
            ])
        
        if critical_failures:
            actions.append("Review and resolve each critical failure individually")
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            actions.append("Implement additional monitoring and safeguards")
        
        return actions
    
    def _generate_approval_conditions(
        self,
        decision: QualityGateDecision,
        warnings: List[str],
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate conditions for approval."""
        
        conditions = []
        
        if decision == QualityGateDecision.APPROVED_WITH_WARNINGS:
            conditions.append("Warnings acknowledged and documented")
            
            if len(warnings) > 5:
                conditions.append("High warning count requires enhanced monitoring")
        
        if risk_level == RiskLevel.MEDIUM:
            conditions.append("Medium risk level requires periodic review")
        
        if decision in [QualityGateDecision.APPROVED, QualityGateDecision.APPROVED_WITH_WARNINGS]:
            conditions.append("Validation results meet publication quality standards")
        
        return conditions