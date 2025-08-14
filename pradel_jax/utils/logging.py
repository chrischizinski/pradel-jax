"""
Logging utilities for pradel-jax.

Provides structured logging with configurable levels, formats, and outputs.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from ..config.settings import get_default_config, LogLevel


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        
        return super().format(record)


class PradelJaxLogger:
    """Custom logger for pradel-jax with rich formatting and context."""
    
    def __init__(self, name: str, config=None):
        self.name = name
        self.config = config or get_default_config()
        self.logger = logging.getLogger(name)
        self._configured = False
    
    def _ensure_configured(self):
        """Ensure logger is configured."""
        if not self._configured:
            self._configure()
            self._configured = True
    
    def _configure(self):
        """Configure the logger based on settings."""
        self.logger.setLevel(getattr(logging, self.config.logging.level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if self.config.logging.console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(self.config.logging.format_string)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.logging.file_logging and self.config.logging.log_file:
            self.config.logging.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.config.logging.log_file)
            file_formatter = logging.Formatter(self.config.logging.format_string)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._ensure_configured()
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._ensure_configured()
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._ensure_configured()
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._ensure_configured()
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._ensure_configured()
        self.logger.critical(self._format_message(message, **kwargs))
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._ensure_configured()
        self.logger.exception(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context."""
        if kwargs:
            context_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {context_str}"
        return message


# Global logger registry
_loggers = {}

def get_logger(name: str = "pradel_jax") -> PradelJaxLogger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to 'pradel_jax')
        
    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        _loggers[name] = PradelJaxLogger(name)
    return _loggers[name]


def setup_logging(
    level: Optional[Union[str, LogLevel]] = None,
    console: Optional[bool] = None,
    file_path: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level
        console: Enable console logging
        file_path: Path for file logging
        format_string: Custom format string
    """
    config = get_default_config()
    
    if level is not None:
        config.logging.level = LogLevel(level) if isinstance(level, str) else level
    
    if console is not None:
        config.logging.console_logging = console
    
    if file_path is not None:
        config.logging.file_logging = True
        config.logging.log_file = Path(file_path)
    
    if format_string is not None:
        config.logging.format_string = format_string
    
    # Reconfigure all existing loggers
    for logger in _loggers.values():
        logger._configured = False


def log_function_call(func):
    """Decorator to log function calls with parameters."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__}", args=args, kwargs=kwargs)
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def log_performance(func):
    """Decorator to log function performance."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed", duration_seconds=duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper