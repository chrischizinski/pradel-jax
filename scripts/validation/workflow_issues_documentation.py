#!/usr/bin/env python3
"""
Workflow Issues Documentation and Severity Assessment
====================================================

Comprehensive documentation of all identified issues with severity ratings
and actionable solutions.
"""

import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "CRITICAL"      # Blocks core functionality
    HIGH = "HIGH"             # Significant impact on functionality
    MEDIUM = "MEDIUM"         # Moderate impact, workarounds exist
    LOW = "LOW"               # Minor issues, cosmetic or edge cases
    INFO = "INFO"             # Informational, no immediate action needed

@dataclass
class WorkflowIssue:
    """Represents a workflow issue with full documentation."""
    issue_id: str
    title: str
    severity: Severity
    component: str
    description: str
    impact: str
    root_cause: str
    reproduction_steps: List[str]
    affected_datasets: List[str]
    proposed_solution: str
    implementation_effort: str  # LOW, MEDIUM, HIGH
    dependencies: List[str]
    status: str  # IDENTIFIED, IN_PROGRESS, RESOLVED
    discovered_date: str
    notes: Optional[str] = None

class WorkflowIssueDocumenter:
    """Documents and tracks workflow issues with severity assessment."""
    
    def __init__(self):
        self.issues: List[WorkflowIssue] = []
        self.discovery_date = datetime.now().isoformat()
    
    def document_core_issues(self):
        """Document the critical issues identified in the workflow examination."""
        
        # Issue 1: Critical JAX String Handling Bug
        self.add_issue(WorkflowIssue(
            issue_id="WF-001",
            title="JAX String Data Handling Causes Data Loading Failure",
            severity=Severity.CRITICAL,
            component="Data Loading (adapters.py)",
            description=(
                "The data loading process fails when covariate metadata contains string values "
                "that are inadvertently passed to JAX operations. Specifically, categorical "
                "covariate information like 'sex_categories': ['Female', 'Male'] is stored in "
                "the covariates dictionary and later processed by JAX operations like jnp.isnan(), "
                "causing a critical error."
            ),
            impact=(
                "COMPLETE WORKFLOW FAILURE - No datasets can be processed. This blocks all "
                "model fitting, analysis, and validation activities. Affects 100% of real "
                "datasets including dipper, nebraska, and south_dakota datasets."
            ),
            root_cause=(
                "The RMarkFormatAdapter stores categorical metadata (levels, category info) "
                "directly in the covariates dictionary alongside numeric arrays. The downstream "
                "data quality assessment code attempts to run jnp.isnan() on all covariate "
                "values without type checking, causing JAX to fail when encountering strings."
            ),
            reproduction_steps=[
                "1. Load any real dataset using load_data()",
                "2. Data context is created successfully",
                "3. Any downstream code that iterates over data_context.covariates",
                "4. Calls jnp.isnan() on covariate values",
                "5. Fails when encountering string values in categorical metadata"
            ],
            affected_datasets=["dipper", "nebraska", "south_dakota", "all real datasets"],
            proposed_solution=(
                "Separate numeric covariate data from metadata in DataContext structure. "
                "Create separate fields: 'numeric_covariates' for JAX arrays and "
                "'covariate_metadata' for string/categorical information. Update all "
                "downstream code to use appropriate field."
            ),
            implementation_effort="MEDIUM",
            dependencies=["DataContext refactoring", "RMarkFormatAdapter updates", "All downstream consumers"],
            status="IDENTIFIED",
            discovered_date=self.discovery_date,
            notes="This is the blocking issue preventing all workflow analysis"
        ))
        
        # Issue 2: API Function Availability
        self.add_issue(WorkflowIssue(
            issue_id="WF-002", 
            title="High-Level API Functions Not Available at Module Level",
            severity=Severity.HIGH,
            component="Module API (__init__.py, core/api.py)",
            description=(
                "Critical user-facing functions like 'fit_model' and 'create_formula_spec' "
                "are not exposed at the module level, making the package difficult to use. "
                "The core/api.py contains placeholder implementations that raise "
                "NotImplementedError."
            ),
            impact=(
                "Severely impacts usability and user experience. Users cannot easily fit "
                "models or create formula specifications using the documented API. Forces "
                "users to import from internal modules, reducing code maintainability."
            ),
            root_cause=(
                "Incomplete API development - high-level convenience functions were not "
                "implemented or exposed during the modular refactoring phase."
            ),
            reproduction_steps=[
                "1. import pradel_jax as pj",
                "2. Try pj.fit_model() -> AttributeError", 
                "3. Try pj.create_formula_spec() -> AttributeError",
                "4. Check core/api.py -> NotImplementedError placeholders"
            ],
            affected_datasets=["All datasets - workflow cannot be tested"],
            proposed_solution=(
                "Implement high-level API functions and expose them at module level. "
                "Create fit_model(), create_formula_spec(), and other user-facing functions "
                "that combine lower-level components."
            ),
            implementation_effort="HIGH",
            dependencies=["DataContext fixes", "Model fitting orchestration"],
            status="IDENTIFIED",
            discovered_date=self.discovery_date
        ))
        
        # Issue 3: Formula System API Inconsistency
        self.add_issue(WorkflowIssue(
            issue_id="WF-003",
            title="Formula Creation Function Name Inconsistency", 
            severity=Severity.MEDIUM,
            component="Formula System (formulas/__init__.py)",
            description=(
                "The formula system exposes 'create_simple_spec()' but documentation and "
                "examples refer to 'create_formula_spec()'. This creates confusion and "
                "breaks example code."
            ),
            impact=(
                "Breaks documentation examples and user code. Reduces developer experience "
                "and makes the package harder to learn and use."
            ),
            root_cause=(
                "API naming inconsistency between implementation and documentation during "
                "modular refactoring."
            ),
            reproduction_steps=[
                "1. Follow documentation examples",
                "2. Try pj.create_formula_spec() -> AttributeError",
                "3. Check __init__.py -> only create_simple_spec available"
            ],
            affected_datasets=["All datasets - affects formula creation"],
            proposed_solution=(
                "Standardize on either 'create_formula_spec' or 'create_simple_spec' "
                "throughout codebase and documentation. Recommend 'create_formula_spec' "
                "as more descriptive and intuitive."
            ),
            implementation_effort="LOW",
            dependencies=["Documentation updates"],
            status="IDENTIFIED", 
            discovered_date=self.discovery_date
        ))
        
        # Issue 4: Data Quality Assessment Logic Error
        self.add_issue(WorkflowIssue(
            issue_id="WF-004",
            title="Data Quality Assessment Assumes All Covariates are Numeric",
            severity=Severity.HIGH,
            component="Data Quality Assessment",
            description=(
                "The data quality assessment code iterates over all values in the "
                "covariates dictionary and attempts JAX operations without type checking. "
                "This fails when metadata or categorical information is present."
            ),
            impact=(
                "Prevents data quality assessment, which is critical for detecting issues "
                "like sparse data, missing values, and identifiability problems. Blocks "
                "quality gates in the analysis workflow."
            ),
            root_cause=(
                "Implicit assumption that all covariate dictionary values are numeric "
                "JAX arrays, without explicit type checking or validation."
            ),
            reproduction_steps=[
                "1. Load data with categorical covariates",
                "2. Run data quality assessment", 
                "3. Code attempts jnp.isnan() on all covariate values",
                "4. Fails on string/metadata values"
            ],
            affected_datasets=["All datasets with categorical covariates"],
            proposed_solution=(
                "Add type checking in data quality assessment. Only apply numeric "
                "operations to JAX arrays. Handle metadata separately or ignore "
                "during numeric quality checks."
            ),
            implementation_effort="LOW",
            dependencies=["DataContext structure clarification"],
            status="IDENTIFIED",
            discovered_date=self.discovery_date
        ))
        
        # Issue 5: Model Optimization Integration Gap
        self.add_issue(WorkflowIssue(
            issue_id="WF-005",
            title="Gap Between Model Classes and Optimization Framework",
            severity=Severity.MEDIUM,
            component="Model-Optimization Integration",
            description=(
                "The PradelModel class and optimization framework are not fully integrated. "
                "No high-level function combines model setup, optimization, and result "
                "processing into a single user-friendly interface."
            ),
            impact=(
                "Requires users to understand internal architecture to fit models. "
                "Increases complexity and reduces usability. Makes testing and validation "
                "more difficult."
            ),
            root_cause=(
                "Modular architecture was implemented without bridging high-level user "
                "interfaces. Each component works independently but integration layer "
                "is missing."
            ),
            reproduction_steps=[
                "1. Create PradelModel()",
                "2. Create formula specification",
                "3. Build design matrices manually",
                "4. Set up objective function manually",
                "5. Call optimization manually",
                "6. Process results manually"
            ],
            affected_datasets=["All datasets - affects user experience"],
            proposed_solution=(
                "Create high-level fit_model() function that handles the entire pipeline: "
                "model setup, design matrix creation, optimization, and result formatting."
            ),
            implementation_effort="MEDIUM",
            dependencies=["API function implementation", "Result formatting standardization"],
            status="IDENTIFIED",
            discovered_date=self.discovery_date
        ))
        
        # Issue 6: Error Handling and User Feedback
        self.add_issue(WorkflowIssue(
            issue_id="WF-006",
            title="Poor Error Messages and Exception Handling",
            severity=Severity.MEDIUM,
            component="Error Handling (Global)",
            description=(
                "JAX and internal errors bubble up to users without translation into "
                "actionable error messages. The JAX string error is particularly cryptic "
                "and doesn't guide users toward solutions."
            ),
            impact=(
                "Poor user experience, difficult debugging, and increased support burden. "
                "Users cannot easily diagnose and fix problems with their data or code."
            ),
            root_cause=(
                "Lack of comprehensive exception handling and error message translation "
                "throughout the codebase."
            ),
            reproduction_steps=[
                "1. Trigger any workflow error",
                "2. Receive cryptic JAX or internal error message",
                "3. No guidance on root cause or solution"
            ],
            affected_datasets=["All datasets - affects user experience"],
            proposed_solution=(
                "Implement comprehensive try-catch blocks with user-friendly error "
                "translation. Create error codes and solutions guide. Add data validation "
                "with helpful warnings."
            ),
            implementation_effort="MEDIUM", 
            dependencies=["Error message standards", "Documentation"],
            status="IDENTIFIED",
            discovered_date=self.discovery_date
        ))
        
        # Issue 7: Missing Statistical Validation
        self.add_issue(WorkflowIssue(
            issue_id="WF-007",
            title="Incomplete Statistical Inference and Uncertainty Quantification",
            severity=Severity.HIGH,
            component="Statistical Inference",
            description=(
                "The workflow lacks proper uncertainty quantification, confidence intervals, "
                "standard errors, and model diagnostics. Results cannot be properly "
                "interpreted or validated statistically."
            ),
            impact=(
                "Results cannot be trusted or published without uncertainty quantification. "
                "No way to assess parameter precision or model adequacy. Limits scientific "
                "utility of the package."
            ),
            root_cause=(
                "Focus on optimization implementation without corresponding statistical "
                "inference development."
            ),
            reproduction_steps=[
                "1. Fit model successfully",
                "2. Check result object for standard errors -> Not available",
                "3. Check for confidence intervals -> Not available", 
                "4. Check for model diagnostics -> Not available"
            ],
            affected_datasets=["All datasets - affects result interpretation"],
            proposed_solution=(
                "Implement Hessian-based standard errors, bootstrap confidence intervals, "
                "and model adequacy diagnostics. Add statistical testing capabilities."
            ),
            implementation_effort="HIGH",
            dependencies=["Working model fitting pipeline"],
            status="IDENTIFIED",
            discovered_date=self.discovery_date
        ))
        
    def add_issue(self, issue: WorkflowIssue):
        """Add an issue to the documentation."""
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: Severity) -> List[WorkflowIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_critical_path_issues(self) -> List[WorkflowIssue]:
        """Get issues that block the critical path."""
        blocking_issues = []
        for issue in self.issues:
            if issue.severity in [Severity.CRITICAL, Severity.HIGH]:
                if any(keyword in issue.impact.lower() for keyword in 
                       ['complete', 'blocks', 'failure', 'cannot', 'severely']):
                    blocking_issues.append(issue)
        return blocking_issues
    
    def generate_priority_matrix(self) -> Dict[str, List[str]]:
        """Generate priority matrix for issue resolution."""
        return {
            "IMMEDIATE_ACTION": [issue.issue_id for issue in self.get_issues_by_severity(Severity.CRITICAL)],
            "HIGH_PRIORITY": [issue.issue_id for issue in self.get_issues_by_severity(Severity.HIGH)], 
            "MEDIUM_PRIORITY": [issue.issue_id for issue in self.get_issues_by_severity(Severity.MEDIUM)],
            "LOW_PRIORITY": [issue.issue_id for issue in self.get_issues_by_severity(Severity.LOW)],
            "INFORMATION_ONLY": [issue.issue_id for issue in self.get_issues_by_severity(Severity.INFO)]
        }
    
    def generate_implementation_roadmap(self) -> Dict[str, List[str]]:
        """Generate implementation roadmap based on dependencies."""
        roadmap = {
            "PHASE_1_FOUNDATIONS": [],  # Critical fixes that unblock everything
            "PHASE_2_CORE_FEATURES": [], # High-level API implementation
            "PHASE_3_ENHANCEMENTS": [], # Statistical inference and diagnostics
            "PHASE_4_POLISH": []        # Error handling and user experience
        }
        
        # Phase 1: Critical blocking issues
        for issue in self.issues:
            if issue.severity == Severity.CRITICAL:
                roadmap["PHASE_1_FOUNDATIONS"].append(issue.issue_id)
        
        # Phase 2: Core API issues
        api_keywords = ["api", "function", "integration", "fit_model"]
        for issue in self.issues:
            if issue.severity == Severity.HIGH and any(keyword in issue.title.lower() for keyword in api_keywords):
                roadmap["PHASE_2_CORE_FEATURES"].append(issue.issue_id)
        
        # Phase 3: Statistical features
        stats_keywords = ["statistical", "inference", "uncertainty", "validation"]
        for issue in self.issues:
            if any(keyword in issue.title.lower() for keyword in stats_keywords):
                roadmap["PHASE_3_ENHANCEMENTS"].append(issue.issue_id)
        
        # Phase 4: Everything else
        assigned_issues = set()
        for phase_issues in roadmap.values():
            assigned_issues.update(phase_issues)
        
        for issue in self.issues:
            if issue.issue_id not in assigned_issues:
                roadmap["PHASE_4_POLISH"].append(issue.issue_id)
        
        return roadmap
    
    def save_documentation(self, filename: str = None) -> str:
        """Save comprehensive issue documentation."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workflow_issues_documentation_{timestamp}.json"
        
        documentation = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "discovery_date": self.discovery_date,
                "total_issues": len(self.issues),
                "severity_breakdown": {
                    severity.value: len(self.get_issues_by_severity(severity))
                    for severity in Severity
                }
            },
            "issues": [asdict(issue) for issue in self.issues],
            "priority_matrix": self.generate_priority_matrix(),
            "implementation_roadmap": self.generate_implementation_roadmap(),
            "critical_path_summary": {
                "blocking_issues": len(self.get_critical_path_issues()),
                "immediate_action_required": [
                    issue.issue_id for issue in self.get_critical_path_issues()
                ]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        return filename
    
    def print_executive_summary(self):
        """Print executive summary of issues."""
        print("\n" + "="*80)
        print("WORKFLOW ISSUES - EXECUTIVE SUMMARY")
        print("="*80)
        
        severity_counts = {severity: len(self.get_issues_by_severity(severity)) for severity in Severity}
        total_issues = sum(severity_counts.values())
        
        print(f"\nTotal Issues Identified: {total_issues}")
        print("\nSeverity Breakdown:")
        for severity in Severity:
            count = severity_counts[severity]
            if count > 0:
                emoji = {"CRITICAL": "🔥", "HIGH": "⚠️", "MEDIUM": "📋", "LOW": "📝", "INFO": "ℹ️"}
                print(f"  {emoji[severity.value]} {severity.value}: {count}")
        
        critical_path = self.get_critical_path_issues()
        print(f"\nCritical Path Blocking Issues: {len(critical_path)}")
        
        if critical_path:
            print("\nIMMEDIATE ACTION REQUIRED:")
            for issue in critical_path:
                print(f"  🔥 {issue.issue_id}: {issue.title}")
        
        roadmap = self.generate_implementation_roadmap()
        print(f"\nImplementation Phases:")
        for phase, issue_ids in roadmap.items():
            if issue_ids:
                print(f"  {phase.replace('_', ' ')}: {len(issue_ids)} issues")
        
        print("\n" + "="*80)

def main():
    """Generate comprehensive workflow issues documentation."""
    documenter = WorkflowIssueDocumenter()
    documenter.document_core_issues()
    
    # Generate comprehensive documentation
    filename = documenter.save_documentation()
    
    # Print executive summary
    documenter.print_executive_summary()
    
    print(f"\n📋 Comprehensive issue documentation saved to: {filename}")
    print("\nThis document provides:")
    print("  • Detailed issue descriptions with reproduction steps")
    print("  • Severity assessments and impact analysis")
    print("  • Root cause analysis and proposed solutions")
    print("  • Implementation effort estimates and dependencies")
    print("  • Priority matrix and implementation roadmap")
    print("  • Critical path analysis for resolution planning")
    
    return documenter

if __name__ == "__main__":
    main()