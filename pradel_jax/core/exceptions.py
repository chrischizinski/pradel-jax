"""
Exception classes for pradel-jax.

Provides rich error information with actionable suggestions and documentation links.
"""

from typing import List, Optional, Dict, Any


class PradelJaxError(Exception):
    """
    Base exception class for pradel-jax with rich error information.
    
    Provides structured error information including suggestions for resolution
    and links to relevant documentation.
    """
    
    def __init__(
        self, 
        message: str, 
        suggestions: Optional[List[str]] = None,
        documentation_link: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.suggestions = suggestions or []
        self.documentation_link = documentation_link
        self.error_code = error_code
        self.context = context or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return formatted error message with suggestions."""
        message = super().__str__()
        
        if self.error_code:
            message = f"[{self.error_code}] {message}"
        
        if self.suggestions:
            message += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                message += f"\n  {i}. {suggestion}"
        
        if self.documentation_link:
            message += f"\n\nDocumentation: {self.documentation_link}"
        
        return message


class DataFormatError(PradelJaxError):
    """Exception raised for data format issues."""
    
    def __init__(
        self, 
        detected_format: Optional[str] = None,
        expected_formats: Optional[List[str]] = None,
        specific_issue: Optional[str] = None,
        **kwargs
    ):
        if detected_format and expected_formats:
            message = f"Unsupported data format: {detected_format}"
            suggestions = [
                f"Convert your data to one of: {', '.join(expected_formats)}",
                "Use the data format converter: pradel_jax.convert_data()",
                "Check column names and data structure",
                "Verify capture history format (should be 0s and 1s)",
            ]
        elif specific_issue:
            message = f"Data format issue: {specific_issue}"
            suggestions = [
                "Check your data structure and column names",
                "Ensure capture histories contain only 0s and 1s",
                "Verify individual IDs are properly formatted",
                "Check for missing or corrupted data",
            ]
        else:
            message = "Data format validation failed"
            suggestions = [
                "Check the data format documentation",
                "Validate your input data structure",
                "Use pradel_jax.validate_data() for detailed diagnostics",
            ]
        
        # Remove suggestions from kwargs if present to avoid conflict
        kwargs.pop('suggestions', None)
        
        super().__init__(
            message=message,
            suggestions=suggestions,
            documentation_link="https://docs.pradel-jax.org/data-formats",
            error_code="DATA_FORMAT",
            context={"detected_format": detected_format, "expected_formats": expected_formats},
            **kwargs
        )


class ModelSpecificationError(PradelJaxError):
    """Exception raised for model specification issues."""
    
    def __init__(
        self,
        formula: Optional[str] = None,
        parameter: Optional[str] = None,
        available_covariates: Optional[List[str]] = None,
        missing_covariates: Optional[List[str]] = None,
        **kwargs
    ):
        if missing_covariates and available_covariates:
            message = f"Missing covariates in formula '{formula}': {missing_covariates}"
            suggestions = [
                f"Available covariates: {', '.join(sorted(available_covariates))}",
                "Check covariate spelling and case sensitivity",
                "Ensure all required data columns are present",
                "Use pradel_jax.list_covariates() to see available options",
            ]
        elif formula:
            message = f"Invalid formula specification: {formula}"
            suggestions = [
                "Check formula syntax (e.g., 'age + sex', 'age*time')",
                "Ensure all terms reference existing covariates",
                "Use '1' for intercept-only models",
                "See documentation for supported formula syntax",
            ]
        else:
            message = "Model specification error"
            suggestions = [
                "Check your model formula syntax",
                "Verify all covariates exist in your data",
                "Review model specification documentation",
            ]
        
        kwargs.pop('suggestions', None)
        
        super().__init__(
            message=message,
            suggestions=suggestions,
            documentation_link="https://docs.pradel-jax.org/model-specification",
            error_code="MODEL_SPEC",
            context={
                "formula": formula, 
                "parameter": parameter,
                "missing_covariates": missing_covariates,
                "available_covariates": available_covariates,
            },
            **kwargs
        )


class OptimizationError(PradelJaxError):
    """Exception raised for optimization failures."""
    
    def __init__(
        self,
        optimizer: Optional[str] = None,
        reason: Optional[str] = None,
        iterations: Optional[int] = None,
        final_loss: Optional[float] = None,
        **kwargs
    ):
        if optimizer and reason:
            message = f"Optimization failed with {optimizer}: {reason}"
        elif optimizer:
            message = f"Optimization failed with {optimizer}"
        else:
            message = "Optimization failed to converge"
        
        suggestions = [
            "Try a different optimization strategy",
            "Increase maximum iterations",
            "Check for data quality issues",
            "Use multi-start optimization for difficult problems",
            "Simplify the model specification",
            "Check for parameter identifiability issues",
        ]
        
        if iterations and iterations > 500:
            suggestions.insert(0, "Model may be overparameterized")
        
        if final_loss and final_loss > 1e6:
            suggestions.insert(0, "Check for extreme outliers in data")
        
        kwargs.pop('suggestions', None)
        
        super().__init__(
            message=message,
            suggestions=suggestions,
            documentation_link="https://docs.pradel-jax.org/optimization",
            error_code="OPTIMIZATION",
            context={
                "optimizer": optimizer,
                "reason": reason, 
                "iterations": iterations,
                "final_loss": final_loss,
            },
            **kwargs
        )


class ValidationError(PradelJaxError):
    """Exception raised for validation failures."""
    
    def __init__(
        self,
        validation_type: Optional[str] = None,
        failed_checks: Optional[List[str]] = None,
        reference_software: Optional[str] = None,
        **kwargs
    ):
        if validation_type and reference_software:
            message = f"Validation against {reference_software} failed: {validation_type}"
            suggestions = [
                f"Check parameter estimates against {reference_software}",
                "Verify identical data formatting",
                "Check for numerical precision differences",
                "Report validation failures to development team",
            ]
        elif failed_checks:
            message = f"Validation failed: {', '.join(failed_checks)}"
            suggestions = [
                "Check data quality and preprocessing",
                "Verify model specification is appropriate",
                "Consider using more robust optimization",
                "Review validation documentation",
            ]
        else:
            message = "Validation checks failed"
            suggestions = [
                "Run pradel_jax.validate() for detailed diagnostics",
                "Check documentation for validation requirements",
                "Ensure data meets model assumptions",
            ]
        
        kwargs.pop('suggestions', None)
        
        super().__init__(
            message=message,
            suggestions=suggestions,
            documentation_link="https://docs.pradel-jax.org/validation",
            error_code="VALIDATION",
            context={
                "validation_type": validation_type,
                "failed_checks": failed_checks,
                "reference_software": reference_software,
            },
            **kwargs
        )


class ConfigurationError(PradelJaxError):
    """Exception raised for configuration issues."""
    
    def __init__(self, config_key: Optional[str] = None, **kwargs):
        if config_key:
            message = f"Invalid configuration for '{config_key}'"
            suggestions = [
                f"Check the value for configuration key '{config_key}'",
                "Review configuration file syntax",
                "Check environment variable formatting",
                "Use pradel_jax.get_config() to inspect current settings",
            ]
        else:
            message = "Configuration error"
            suggestions = [
                "Check configuration file syntax",
                "Verify all required settings are provided",
                "Review configuration documentation",
            ]
        
        kwargs.pop('suggestions', None)
        
        super().__init__(
            message=message,
            suggestions=suggestions,
            documentation_link="https://docs.pradel-jax.org/configuration",
            error_code="CONFIG",
            context={"config_key": config_key},
            **kwargs
        )


class ConvergenceError(OptimizationError):
    """Exception raised when optimization fails to converge."""
    
    def __init__(self, **kwargs):
        super().__init__(
            reason="Failed to reach convergence criteria",
            suggestions=[
                "Increase maximum iterations",
                "Relax convergence tolerance",
                "Try multi-start optimization",
                "Check for parameter identifiability",
                "Simplify model specification",
                "Examine data for quality issues",
            ],
            **kwargs
        )


class DataQualityError(DataFormatError):
    """Exception raised for data quality issues."""
    
    def __init__(self, quality_issues: Optional[List[str]] = None, **kwargs):
        if quality_issues:
            message = f"Data quality issues detected: {', '.join(quality_issues)}"
            suggestions = [
                "Review data preprocessing options",
                "Check for missing or invalid values",
                "Consider data cleaning strategies",
                "Use pradel_jax.diagnose_data() for detailed analysis",
            ]
        else:
            message = "Data quality validation failed"
            suggestions = [
                "Run data quality diagnostics",
                "Check for missing values and outliers",
                "Verify capture history completeness",
            ]
        
        kwargs.pop('suggestions', None)
        
        super().__init__(
            specific_issue=message,
            suggestions=suggestions,
            error_code="DATA_QUALITY",
            **kwargs
        )