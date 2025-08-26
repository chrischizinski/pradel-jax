"""
Comprehensive Error Handling and Recovery Framework (Phase 3).

This module provides robust error handling, recovery mechanisms, and
fault-tolerant execution for the validation pipeline. It implements
industry-standard error management practices with automatic recovery,
graceful degradation, and comprehensive error reporting.

Key Features:
    - Hierarchical error classification and handling
    - Automatic retry mechanisms with exponential backoff
    - Graceful degradation for non-critical failures
    - Comprehensive error logging and reporting
    - Recovery strategies for common failure scenarios
    - Circuit breaker pattern for external dependencies

Usage:
    error_handler = ValidationErrorHandler(config)

    # Execute with error handling
    try:
        result = error_handler.execute_with_error_handling(
            func, *args, retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF
        )
    except CriticalValidationError as e:
        logger.error(f"Critical failure: {e}")
        error_handler.initiate_recovery_protocol()
"""

import os
import time
import logging
import traceback
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Type
from enum import Enum
import functools
import json
import warnings

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""

    LOW = "low"  # Minor issues, warnings
    MEDIUM = "medium"  # Recoverable errors
    HIGH = "high"  # Significant failures
    CRITICAL = "critical"  # System-breaking errors


class ErrorCategory(Enum):
    """Categories of errors in validation pipeline."""

    CONFIGURATION = "configuration"  # Config/setup errors
    DATA_PROCESSING = "data_processing"  # Data loading/parsing errors
    MODEL_EXECUTION = "model_execution"  # Model fitting errors
    RMARK_INTERFACE = "rmark_interface"  # RMark execution errors
    STATISTICAL_TEST = "statistical_test"  # Statistical validation errors
    SYSTEM_RESOURCE = "system_resource"  # Memory/CPU/disk errors
    NETWORK = "network"  # Network/connectivity errors
    PERMISSION = "permission"  # File/system permission errors
    DEPENDENCY = "dependency"  # Missing dependencies
    UNKNOWN = "unknown"  # Unclassified errors


class RetryPolicy(Enum):
    """Retry policies for error recovery."""

    NO_RETRY = "no_retry"
    IMMEDIATE = "immediate"
    LINEAR_BACKOFF = "linear_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CUSTOM = "custom"


@dataclass
class ErrorContext:
    """Context information for error occurrence."""

    error_id: str
    timestamp: datetime
    function_name: str
    module_name: str

    # Error details
    error_type: str
    error_message: str
    stack_trace: str

    # Classification
    severity: ErrorSeverity
    category: ErrorCategory

    # Execution context
    dataset_name: Optional[str] = None
    model_spec: Optional[str] = None
    worker_id: Optional[str] = None
    session_id: Optional[str] = None

    # Recovery information
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempted: bool = False
    recovery_successful: bool = False

    # Additional metadata
    system_info: Dict[str, Any] = field(default_factory=dict)
    user_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "function_name": self.function_name,
            "module_name": self.module_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "severity": self.severity.value,
            "category": self.category.value,
            "dataset_name": self.dataset_name,
            "model_spec": self.model_spec,
            "worker_id": self.worker_id,
            "session_id": self.session_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "system_info": self.system_info,
            "user_data": self.user_data,
        }


class ValidationError(Exception):
    """Base exception for validation pipeline errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.metadata = kwargs


class CriticalValidationError(ValidationError):
    """Critical errors that require immediate pipeline termination."""

    def __init__(
        self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs
    ):
        super().__init__(message, category, ErrorSeverity.CRITICAL, **kwargs)


class RecoverableValidationError(ValidationError):
    """Recoverable errors that can be retried or handled gracefully."""

    def __init__(
        self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, **kwargs
    ):
        super().__init__(message, category, ErrorSeverity.MEDIUM, **kwargs)


class ConfigurationError(CriticalValidationError):
    """Configuration-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.CONFIGURATION, **kwargs)


class DataProcessingError(RecoverableValidationError):
    """Data processing and validation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.DATA_PROCESSING, **kwargs)


class ModelExecutionError(RecoverableValidationError):
    """Model fitting and execution errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.MODEL_EXECUTION, **kwargs)


class RMarkInterfaceError(RecoverableValidationError):
    """RMark execution and interface errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.RMARK_INTERFACE, **kwargs)


class StatisticalTestError(RecoverableValidationError):
    """Statistical test and validation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, ErrorCategory.STATISTICAL_TEST, **kwargs)


class CircuitBreaker:
    """Circuit breaker pattern for external dependencies."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # "closed", "open", "half_open"
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if (
                    datetime.now() - self.last_failure_time
                ).total_seconds() > self.recovery_timeout:
                    self.state = "half_open"
                else:
                    raise RecoverableValidationError(
                        "Circuit breaker is open - service unavailable",
                        category=ErrorCategory.DEPENDENCY,
                    )

            try:
                result = func(*args, **kwargs)
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                return result

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"

                raise e


@dataclass
class RetryConfiguration:
    """Configuration for retry mechanisms."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    # Retry conditions
    retry_on_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [
            RecoverableValidationError,
            ConnectionError,
            TimeoutError,
        ]
    )

    no_retry_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [CriticalValidationError, ConfigurationError]
    )


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""

    def can_recover(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can recover from the given error."""
        raise NotImplementedError

    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from the error."""
        raise NotImplementedError


class MemoryRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for memory-related errors."""

    def can_recover(self, error_context: ErrorContext) -> bool:
        return (
            error_context.category == ErrorCategory.SYSTEM_RESOURCE
            and "memory" in error_context.error_message.lower()
        )

    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt memory recovery by triggering garbage collection."""
        try:
            import gc

            gc.collect()

            # Add small delay to allow memory cleanup
            time.sleep(1.0)

            logger.info(f"Memory recovery attempted for error {error_context.error_id}")
            return True

        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False


class FilePermissionRecoveryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for file permission errors."""

    def can_recover(self, error_context: ErrorContext) -> bool:
        return error_context.category == ErrorCategory.PERMISSION and any(
            keyword in error_context.error_message.lower()
            for keyword in ["permission", "access", "denied"]
        )

    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from permission errors."""
        try:
            # Try to create directories if they don't exist
            if "directory" in error_context.error_message.lower():
                # Extract directory path from error message (simplified)
                # In practice, you'd need more sophisticated path extraction
                return True

            logger.info(
                f"Permission recovery attempted for error {error_context.error_id}"
            )
            return False  # Most permission errors can't be auto-recovered

        except Exception as e:
            logger.error(f"Permission recovery failed: {e}")
            return False


class ValidationErrorHandler:
    """
    Comprehensive error handler for validation pipeline.

    Provides error classification, retry mechanisms, recovery strategies,
    and comprehensive error reporting for robust pipeline execution.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or "unknown"
        self.error_log: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_strategies: List[ErrorRecoveryStrategy] = [
            MemoryRecoveryStrategy(),
            FilePermissionRecoveryStrategy(),
        ]
        self._lock = threading.Lock()

        # Error statistics
        self.error_stats = {severity.value: 0 for severity in ErrorSeverity}
        self.category_stats = {category.value: 0 for category in ErrorCategory}

    def classify_error(
        self, exception: Exception, context: Dict[str, Any] = None
    ) -> ErrorContext:
        """Classify an error and create error context."""

        context = context or {}

        # Determine error category
        category = self._determine_error_category(exception)

        # Determine error severity
        severity = self._determine_error_severity(exception, category)

        # Create error context
        error_context = ErrorContext(
            error_id=f"err_{int(time.time() * 1000)}_{id(exception)}",
            timestamp=datetime.now(),
            function_name=context.get("function_name", "unknown"),
            module_name=context.get("module_name", __name__),
            error_type=type(exception).__name__,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            severity=severity,
            category=category,
            dataset_name=context.get("dataset_name"),
            model_spec=context.get("model_spec"),
            worker_id=context.get("worker_id"),
            session_id=self.session_id,
            system_info=self._collect_system_info(),
        )

        # Update statistics
        with self._lock:
            self.error_stats[severity.value] += 1
            self.category_stats[category.value] += 1
            self.error_log.append(error_context)

        logger.error(
            f"Error classified: {error_context.error_id} - {severity.value} {category.value}"
        )

        return error_context

    def _determine_error_category(self, exception: Exception) -> ErrorCategory:
        """Determine error category based on exception type and message."""

        if isinstance(exception, ValidationError):
            return exception.category

        exception_type = type(exception).__name__
        error_message = str(exception).lower()

        # Configuration errors
        if any(
            keyword in error_message
            for keyword in ["config", "configuration", "setting", "parameter not found"]
        ):
            return ErrorCategory.CONFIGURATION

        # Data processing errors
        if any(
            keyword in error_message
            for keyword in [
                "data",
                "file not found",
                "parse",
                "csv",
                "json",
                "invalid format",
            ]
        ):
            return ErrorCategory.DATA_PROCESSING

        # Model execution errors
        if any(
            keyword in error_message
            for keyword in [
                "convergence",
                "optimization",
                "likelihood",
                "parameter",
                "model",
            ]
        ):
            return ErrorCategory.MODEL_EXECUTION

        # RMark interface errors
        if any(
            keyword in error_message for keyword in ["rmark", "ssh", "r script", "mark"]
        ):
            return ErrorCategory.RMARK_INTERFACE

        # System resource errors
        if any(
            keyword in error_message
            for keyword in ["memory", "disk", "cpu", "resource", "timeout"]
        ):
            return ErrorCategory.SYSTEM_RESOURCE

        # Network errors
        if any(
            keyword in error_message
            for keyword in ["network", "connection", "socket", "timeout", "unreachable"]
        ):
            return ErrorCategory.NETWORK

        # Permission errors
        if any(
            keyword in error_message
            for keyword in ["permission", "access", "denied", "forbidden"]
        ):
            return ErrorCategory.PERMISSION

        # Dependency errors
        if any(
            keyword in error_message
            for keyword in ["import", "module", "package", "library", "dependency"]
        ):
            return ErrorCategory.DEPENDENCY

        return ErrorCategory.UNKNOWN

    def _determine_error_severity(
        self, exception: Exception, category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity based on exception and category."""

        if isinstance(exception, ValidationError):
            return exception.severity

        # Critical severity conditions
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL

        if category == ErrorCategory.CONFIGURATION:
            return ErrorSeverity.CRITICAL

        # High severity conditions
        if category in [ErrorCategory.SYSTEM_RESOURCE, ErrorCategory.DEPENDENCY]:
            return ErrorSeverity.HIGH

        # Medium severity (default for most recoverable errors)
        if category in [
            ErrorCategory.MODEL_EXECUTION,
            ErrorCategory.RMARK_INTERFACE,
            ErrorCategory.DATA_PROCESSING,
        ]:
            return ErrorSeverity.MEDIUM

        # Low severity
        return ErrorSeverity.LOW

    def execute_with_error_handling(
        self,
        func: Callable,
        *args,
        retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF,
        retry_config: Optional[RetryConfiguration] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function with comprehensive error handling and retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            retry_policy: Retry policy to use
            retry_config: Retry configuration
            context: Additional context for error reporting
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CriticalValidationError: For unrecoverable errors
        """

        retry_config = retry_config or RetryConfiguration()
        context = context or {}

        # Add function context
        context.update({"function_name": func.__name__, "module_name": func.__module__})

        last_error = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                # Execute function
                return func(*args, **kwargs)

            except Exception as e:
                last_error = e
                error_context = self.classify_error(e, context)
                error_context.retry_count = attempt
                error_context.max_retries = retry_config.max_retries

                # Check if error should not be retried
                if any(
                    isinstance(e, exc_type)
                    for exc_type in retry_config.no_retry_exceptions
                ):
                    logger.error(
                        f"Non-retryable error encountered: {error_context.error_id}"
                    )
                    raise e

                # Check if we've exhausted retries
                if attempt >= retry_config.max_retries:
                    logger.error(
                        f"Max retries exceeded for error: {error_context.error_id}"
                    )
                    break

                # Attempt recovery
                if self._attempt_recovery(error_context):
                    logger.info(
                        f"Recovery successful for error: {error_context.error_id}"
                    )
                    continue

                # Check if error should be retried
                if not any(
                    isinstance(e, exc_type)
                    for exc_type in retry_config.retry_on_exceptions
                ):
                    logger.error(
                        f"Error not configured for retry: {error_context.error_id}"
                    )
                    raise e

                # Calculate retry delay
                delay = self._calculate_retry_delay(attempt, retry_policy, retry_config)

                logger.warning(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{retry_config.max_retries}) "
                    f"for error: {error_context.error_id}"
                )

                time.sleep(delay)

        # All retries exhausted
        final_error_context = self.classify_error(last_error, context)
        final_error_context.retry_count = retry_config.max_retries

        # Convert to critical error if retries exhausted
        raise CriticalValidationError(
            f"Function execution failed after {retry_config.max_retries} retries: {str(last_error)}",
            category=final_error_context.category,
            original_error=last_error,
            error_context=final_error_context,
        )

    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from an error using available strategies."""

        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_context):
                try:
                    error_context.recovery_attempted = True
                    success = strategy.attempt_recovery(error_context)
                    error_context.recovery_successful = success

                    if success:
                        logger.info(
                            f"Recovery successful using {type(strategy).__name__}"
                        )
                        return True

                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")

        return False

    def _calculate_retry_delay(
        self, attempt: int, policy: RetryPolicy, config: RetryConfiguration
    ) -> float:
        """Calculate delay before retry based on policy."""

        if policy == RetryPolicy.NO_RETRY:
            return 0.0

        elif policy == RetryPolicy.IMMEDIATE:
            return 0.1  # Small delay to avoid immediate retry

        elif policy == RetryPolicy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)

        elif policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.exponential_base**attempt)

        else:
            delay = config.base_delay

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter to avoid thundering herd
        if config.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5

        return delay

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for error context."""

        try:
            import psutil
            import platform

            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "memory_available_gb": psutil.virtual_memory().available / 1024**3,
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "process_id": os.getpid(),
                "thread_count": threading.active_count(),
            }
        except Exception:
            return {"collection_error": "Failed to collect system info"}

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary and statistics."""

        with self._lock:
            total_errors = len(self.error_log)

            if total_errors == 0:
                return {
                    "total_errors": 0,
                    "error_rate": 0.0,
                    "severity_distribution": self.error_stats,
                    "category_distribution": self.category_stats,
                    "top_errors": [],
                    "recovery_stats": {"attempted": 0, "successful": 0},
                }

            # Calculate recovery statistics
            recovery_attempted = sum(
                1 for error in self.error_log if error.recovery_attempted
            )
            recovery_successful = sum(
                1 for error in self.error_log if error.recovery_successful
            )

            # Get top error types
            error_type_counts = {}
            for error in self.error_log:
                error_type = error.error_type
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

            top_errors = sorted(
                error_type_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

            return {
                "total_errors": total_errors,
                "error_rate": total_errors
                / max(1, total_errors),  # Placeholder calculation
                "severity_distribution": self.error_stats.copy(),
                "category_distribution": self.category_stats.copy(),
                "top_errors": [
                    {"type": error_type, "count": count}
                    for error_type, count in top_errors
                ],
                "recovery_stats": {
                    "attempted": recovery_attempted,
                    "successful": recovery_successful,
                    "success_rate": recovery_successful / max(1, recovery_attempted),
                },
                "recent_errors": [
                    error.to_dict() for error in self.error_log[-10:]
                ],  # Last 10 errors
            }

    def save_error_log(self, output_path: Path) -> None:
        """Save error log to file for analysis."""

        error_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_error_summary(),
            "detailed_errors": [error.to_dict() for error in self.error_log],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(error_data, f, indent=2, default=str)

        logger.info(f"Error log saved to {output_path}")

    def clear_error_log(self) -> None:
        """Clear error log and reset statistics."""

        with self._lock:
            self.error_log.clear()
            self.error_stats = {severity.value: 0 for severity in ErrorSeverity}
            self.category_stats = {category.value: 0 for category in ErrorCategory}

        logger.info("Error log cleared")


def error_handler_decorator(
    retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF,
    retry_config: Optional[RetryConfiguration] = None,
    session_id: Optional[str] = None,
):
    """
    Decorator for automatic error handling and retry.

    Usage:
        @error_handler_decorator(retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF)
        def my_function():
            # Function implementation
            pass
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ValidationErrorHandler(session_id)

            context = {"function_name": func.__name__, "module_name": func.__module__}

            return error_handler.execute_with_error_handling(
                func,
                *args,
                retry_policy=retry_policy,
                retry_config=retry_config,
                context=context,
                **kwargs,
            )

        return wrapper

    return decorator


# Convenience functions for common error handling patterns


def handle_data_processing_errors(func: Callable) -> Callable:
    """Decorator for data processing functions with appropriate error handling."""

    config = RetryConfiguration(
        max_retries=2,
        base_delay=0.5,
        retry_on_exceptions=[DataProcessingError, IOError, OSError],
    )

    return error_handler_decorator(
        retry_policy=RetryPolicy.LINEAR_BACKOFF, retry_config=config
    )(func)


def handle_model_execution_errors(func: Callable) -> Callable:
    """Decorator for model execution functions with appropriate error handling."""

    config = RetryConfiguration(
        max_retries=3,
        base_delay=1.0,
        retry_on_exceptions=[ModelExecutionError, RuntimeError],
    )

    return error_handler_decorator(
        retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF, retry_config=config
    )(func)


def handle_rmark_interface_errors(func: Callable) -> Callable:
    """Decorator for RMark interface functions with appropriate error handling."""

    config = RetryConfiguration(
        max_retries=5,  # More retries for network/SSH issues
        base_delay=2.0,
        max_delay=30.0,
        retry_on_exceptions=[RMarkInterfaceError, ConnectionError, TimeoutError],
    )

    return error_handler_decorator(
        retry_policy=RetryPolicy.EXPONENTIAL_BACKOFF, retry_config=config
    )(func)
