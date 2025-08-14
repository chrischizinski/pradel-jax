"""
RMark Interface for Multi-Environment Execution.

This module provides secure, multi-environment execution of RMark analyses,
supporting SSH, local R, and mock validation modes. All credentials are
loaded from environment variables to maintain security.

Execution Strategies:
    - SSH: Execute RMark on remote Windows machine (home office)
    - Local R: Execute RMark on local machine (work office)  
    - Mock: Generate realistic mock results (development/testing)
    - Cached: Use previously computed results (offline development)

Security Features:
    - Zero credentials in code
    - Environment variable-based configuration
    - Secure temporary file handling
    - Comprehensive audit logging
    - Automatic cleanup of sensitive files
"""

import os
import time
import json
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .secure_config import SecureValidationConfig, get_secure_validation_config, SecurityError
from ..data.adapters import DataContext
from ..formulas.spec import FormulaSpec
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Optional dependencies
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    logger.info("paramiko not available - SSH execution will be disabled")
    PARAMIKO_AVAILABLE = False


class ExecutionMethod(Enum):
    """RMark execution methods."""
    SSH = "ssh"
    LOCAL_R = "local_r"
    MOCK = "mock"
    CACHED = "cached"


class ExecutionStatus(Enum):
    """Execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_AVAILABLE = "not_available"


@dataclass
class RMarkResult:
    """Result from RMark execution."""
    
    # Model identification
    model_formula: str
    execution_method: str
    execution_time: float
    
    # Model results
    converged: bool = False
    aic: float = 0.0
    aicc: float = 0.0
    log_likelihood: float = 0.0
    n_parameters: int = 0
    
    # Parameter estimates
    parameters: Dict[str, float] = field(default_factory=dict)
    std_errors: Dict[str, float] = field(default_factory=dict)
    
    # Execution metadata
    execution_status: ExecutionStatus = ExecutionStatus.FAILED
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    notes: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            'model_formula': self.model_formula,
            'execution_method': self.execution_method,
            'converged': self.converged,
            'aic': self.aic,
            'log_likelihood': self.log_likelihood,
            'n_parameters': self.n_parameters,
            'execution_time': self.execution_time,
            'status': self.execution_status.value
        }


class EnvironmentDetector:
    """Detect available RMark execution environments."""
    
    def __init__(self, config: SecureValidationConfig):
        self.config = config
    
    def detect_available_methods(self) -> List[ExecutionMethod]:
        """Detect which execution methods are available."""
        available = []
        
        # Test SSH availability
        if self._test_ssh_connectivity():
            available.append(ExecutionMethod.SSH)
            logger.info("✅ SSH execution available")
        else:
            logger.info("❌ SSH execution not available")
        
        # Test local R availability
        if self._test_local_r():
            available.append(ExecutionMethod.LOCAL_R)
            logger.info("✅ Local R execution available")
        else:
            logger.info("❌ Local R execution not available")
        
        # Mock is always available
        available.append(ExecutionMethod.MOCK)
        
        # Cached if cache directory exists
        if self._has_cached_results():
            available.append(ExecutionMethod.CACHED)
        
        return available
    
    def _test_ssh_connectivity(self) -> bool:
        """Test SSH connectivity to configured host."""
        if not PARAMIKO_AVAILABLE:
            return False
        
        if not self.config.has_ssh_config():
            return False
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            ssh.connect(
                hostname=self.config.ssh_host,
                username=self.config.ssh_user,
                key_filename=os.path.expanduser(self.config.ssh_key_path),
                timeout=10,
                banner_timeout=10
            )
            ssh.close()
            return True
            
        except Exception as e:
            logger.debug(f"SSH connectivity test failed: {e}")
            return False
    
    def _test_local_r(self) -> bool:
        """Test local R availability."""
        try:
            result = subprocess.run([
                self.config.local_r_path, "--version"
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.debug(f"Local R test failed: {e}")
            return False
    
    def _has_cached_results(self) -> bool:
        """Check if cached results are available."""
        cache_dir = Path("validation_cache")
        return cache_dir.exists() and any(cache_dir.glob("*.json"))
    
    def get_recommended_method(self) -> ExecutionMethod:
        """Get recommended execution method based on environment."""
        available = self.detect_available_methods()
        
        # Preference order based on configuration
        if self.config.preferred_environment == "ssh" and ExecutionMethod.SSH in available:
            return ExecutionMethod.SSH
        elif self.config.preferred_environment == "local_r" and ExecutionMethod.LOCAL_R in available:
            return ExecutionMethod.LOCAL_R
        elif self.config.preferred_environment == "mock":
            return ExecutionMethod.MOCK
        
        # Auto-detection preference order
        if ExecutionMethod.SSH in available:
            return ExecutionMethod.SSH
        elif ExecutionMethod.LOCAL_R in available:
            return ExecutionMethod.LOCAL_R
        else:
            return ExecutionMethod.MOCK


class SSHRMarkExecutor:
    """Execute RMark via SSH on remote Windows machine."""
    
    def __init__(self, config: SecureValidationConfig):
        self.config = config
        self.session_id = config.session_id
        
        if not PARAMIKO_AVAILABLE:
            raise SecurityError("SSH execution requires paramiko package")
        
        if not config.has_ssh_config():
            raise SecurityError("SSH configuration incomplete - check environment variables")
    
    def execute(self, data: DataContext, formula_spec: FormulaSpec) -> RMarkResult:
        """Execute RMark analysis via SSH."""
        logger.info(f"Executing RMark via SSH [{self.session_id}]")
        start_time = time.time()
        
        try:
            # Generate R script
            r_script = self._generate_rmark_script(data, formula_spec)
            
            # Transfer data and execute
            with self._ssh_connection() as ssh:
                # Transfer files
                self._transfer_data_and_script(ssh, data, r_script)
                
                # Execute R script
                raw_output = self._execute_remote_script(ssh)
                
                # Parse results
                result = self._parse_rmark_output(raw_output, formula_spec)
                
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.execution_method = "ssh"
            result.execution_status = ExecutionStatus.SUCCESS
            
            logger.info(f"SSH execution completed in {execution_time:.1f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"SSH execution failed after {execution_time:.1f}s: {e}")
            
            return RMarkResult(
                model_formula=self._format_formula_string(formula_spec),
                execution_method="ssh",
                execution_time=execution_time,
                execution_status=ExecutionStatus.FAILED,
                notes=[f"SSH execution failed: {str(e)}"]
            )
    
    def _ssh_connection(self):
        """Create SSH connection context manager."""
        class SSHContext:
            def __init__(self, config):
                self.config = config
                self.ssh = None
            
            def __enter__(self):
                self.ssh = paramiko.SSHClient()
                self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                self.ssh.connect(
                    hostname=self.config.ssh_host,
                    username=self.config.ssh_user,
                    key_filename=os.path.expanduser(self.config.ssh_key_path),
                    timeout=self.config.ssh_timeout,
                    banner_timeout=30
                )
                return self.ssh
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.ssh:
                    self.ssh.close()
        
        return SSHContext(self.config)
    
    def _generate_rmark_script(self, data: DataContext, formula_spec: FormulaSpec) -> str:
        """Generate R script for RMark analysis."""
        
        # Convert factor columns
        factor_conversions = []
        for name, info in data.covariate_info.items():
            if info.is_categorical:
                factor_conversions.append(f"data${name} <- as.factor(data${name})")
        
        factor_code = "\n    ".join(factor_conversions) if factor_conversions else "# No factor conversions needed"
        
        script = f'''
# RMark Analysis Script (Generated by pradel-jax)
# Session ID: {self.session_id}
library(RMark)
library(jsonlite)
options(warn = -1)  # Suppress warnings

tryCatch({{
    # Load data
    data <- read.csv("C:/temp/validation_data_{self.session_id}.csv", stringsAsFactors = FALSE)
    
    # Convert factors as needed
    {factor_code}
    
    # Process for Pradel model
    processed <- process.data(data, model = "Pradel")
    ddl <- make.design.data(processed)
    
    # Fit model with specified formulas
    model <- mark(
        processed, ddl,
        model.parameters = list(
            Phi = list(formula = ~{formula_spec.phi}),
            p = list(formula = ~{formula_spec.p}),
            f = list(formula = ~{formula_spec.f})
        ),
        delete = TRUE,
        output = FALSE,
        silent = TRUE
    )
    
    # Extract results
    if (!is.null(model) && model$convergence == 0) {{
        
        # Extract parameter estimates
        beta_estimates <- model$results$beta
        param_names <- row.names(beta_estimates)
        param_values <- as.numeric(beta_estimates$estimate)
        param_se <- as.numeric(beta_estimates$se)
        
        # Create parameter dictionaries
        parameters <- setNames(param_values, param_names)
        std_errors <- setNames(param_se, param_names)
        
        results <- list(
            converged = TRUE,
            aic = model$results$AICc,
            aicc = model$results$AICc,
            log_likelihood = model$results$lnl,
            n_parameters = model$results$npar,
            parameters = parameters,
            std_errors = std_errors,
            session_id = "{self.session_id}",
            formula_phi = "{formula_spec.phi}",
            formula_p = "{formula_spec.p}",
            formula_f = "{formula_spec.f}"
        )
        
        # Output as JSON for easy parsing
        cat("RMARK_RESULTS_START\\n")
        cat(toJSON(results, auto_unbox = TRUE, digits = 8))
        cat("\\nRMARK_RESULTS_END\\n")
        
    }} else {{
        # Model failed to converge
        results <- list(
            converged = FALSE,
            error = "Model convergence failed",
            convergence_code = if (!is.null(model)) model$convergence else -1,
            session_id = "{self.session_id}"
        )
        
        cat("RMARK_RESULTS_START\\n")
        cat(toJSON(results, auto_unbox = TRUE))
        cat("\\nRMARK_RESULTS_END\\n")
    }}
    
}}, error = function(e) {{
    # Error handling
    error_result <- list(
        converged = FALSE,
        error = paste("R Error:", e$message),
        session_id = "{self.session_id}"
    )
    
    cat("RMARK_RESULTS_START\\n")
    cat(toJSON(error_result, auto_unbox = TRUE))
    cat("\\nRMARK_RESULTS_END\\n")
}})
'''
        return script
    
    def _transfer_data_and_script(self, ssh, data: DataContext, r_script: str):
        """Transfer data and R script to remote machine."""
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Convert data to DataFrame and save
            import pandas as pd
            
            # Reconstruct data for RMark
            data_dict = {'ch': []}
            
            # Convert capture matrix back to capture histories
            for i in range(data.n_individuals):
                ch = ''.join(str(int(data.capture_matrix[i, j])) for j in range(data.n_occasions))
                data_dict['ch'].append(ch)
            
            # Add covariates
            for name, values in data.covariates.items():
                data_dict[name] = values.tolist()
            
            df = pd.DataFrame(data_dict)
            df.to_csv(f.name, index=False)
            local_data_file = f.name
        
        # Create temporary R script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            f.write(r_script)
            local_script_file = f.name
        
        try:
            # Transfer data file
            sftp = ssh.open_sftp()
            remote_data_path = f"C:/temp/validation_data_{self.session_id}.csv"
            remote_script_path = f"C:/temp/validation_script_{self.session_id}.R"
            
            sftp.put(local_data_file, remote_data_path)
            sftp.put(local_script_file, remote_script_path)
            sftp.close()
            
            logger.debug(f"Transferred files to remote machine: {remote_data_path}, {remote_script_path}")
            
        finally:
            # Cleanup local temporary files
            os.unlink(local_data_file)
            os.unlink(local_script_file)
    
    def _execute_remote_script(self, ssh) -> str:
        """Execute R script on remote machine."""
        
        remote_script_path = f"C:/temp/validation_script_{self.session_id}.R"
        command = f'{self.config.ssh_r_path} {remote_script_path}'
        
        logger.debug(f"Executing remote command: {command}")
        
        stdin, stdout, stderr = ssh.exec_command(command, timeout=self.config.ssh_timeout)
        
        # Read output
        stdout_text = stdout.read().decode('utf-8', errors='replace')
        stderr_text = stderr.read().decode('utf-8', errors='replace')
        exit_status = stdout.channel.recv_exit_status()
        
        # Cleanup remote files
        try:
            ssh.exec_command(f'del "C:/temp/validation_data_{self.session_id}.csv"')
            ssh.exec_command(f'del "C:/temp/validation_script_{self.session_id}.R"')
        except:
            pass  # Cleanup is best-effort
        
        if exit_status != 0:
            raise RuntimeError(f"R script execution failed (exit code {exit_status}): {stderr_text}")
        
        logger.debug(f"Remote execution completed with exit status {exit_status}")
        return stdout_text
    
    def _parse_rmark_output(self, output: str, formula_spec: FormulaSpec) -> RMarkResult:
        """Parse RMark output and extract results."""
        
        # Find JSON results in output
        start_marker = "RMARK_RESULTS_START"
        end_marker = "RMARK_RESULTS_END"
        
        start_idx = output.find(start_marker)
        end_idx = output.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not find RMark results in output")
        
        json_text = output[start_idx + len(start_marker):end_idx].strip()
        
        try:
            results = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse RMark JSON results: {e}")
        
        # Create RMarkResult
        return RMarkResult(
            model_formula=self._format_formula_string(formula_spec),
            converged=results.get('converged', False),
            aic=results.get('aic', 0.0),
            aicc=results.get('aicc', 0.0),
            log_likelihood=results.get('log_likelihood', 0.0),
            n_parameters=results.get('n_parameters', 0),
            parameters=results.get('parameters', {}),
            std_errors=results.get('std_errors', {}),
            raw_output=output,
            notes=[] if results.get('converged') else [results.get('error', 'Unknown error')]
        )
    
    def _format_formula_string(self, formula_spec: FormulaSpec) -> str:
        """Format formula specification as string."""
        return f"phi({formula_spec.phi}), p({formula_spec.p}), f({formula_spec.f})"


class LocalRMarkExecutor:
    """Execute RMark locally."""
    
    def __init__(self, config: SecureValidationConfig):
        self.config = config
        self.session_id = config.session_id
    
    def execute(self, data: DataContext, formula_spec: FormulaSpec) -> RMarkResult:
        """Execute RMark analysis locally."""
        logger.info(f"Executing RMark locally [{self.session_id}]")
        start_time = time.time()
        
        try:
            # Ensure RMark is available
            self._ensure_rmark_available()
            
            # Create temporary working directory
            with tempfile.TemporaryDirectory(prefix=f"rmark_{self.session_id}_") as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save data
                data_file = temp_path / "data.csv"
                self._save_data_for_rmark(data, data_file)
                
                # Generate and save R script
                script_file = temp_path / "analysis.R"
                r_script = self._generate_local_rmark_script(data_file, formula_spec)
                script_file.write_text(r_script)
                
                # Execute R script
                result = subprocess.run([
                    self.config.local_r_path, str(script_file)
                ], capture_output=True, text=True, timeout=self.config.local_r_timeout)
                
                # Parse results
                rmark_result = self._parse_local_output(result, formula_spec)
                
            execution_time = time.time() - start_time
            rmark_result.execution_time = execution_time
            rmark_result.execution_method = "local_r"
            rmark_result.execution_status = ExecutionStatus.SUCCESS
            
            logger.info(f"Local execution completed in {execution_time:.1f}s")
            return rmark_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Local execution failed after {execution_time:.1f}s: {e}")
            
            return RMarkResult(
                model_formula=self._format_formula_string(formula_spec),
                execution_method="local_r",
                execution_time=execution_time,
                execution_status=ExecutionStatus.FAILED,
                notes=[f"Local execution failed: {str(e)}"]
            )
    
    def _ensure_rmark_available(self):
        """Ensure RMark package is installed."""
        if not self.config.auto_install_rmark:
            return
        
        # Test if RMark is available
        test_result = subprocess.run([
            self.config.local_r_path, "-e", "library(RMark); cat('AVAILABLE')"
        ], capture_output=True, text=True, timeout=30)
        
        if "AVAILABLE" in test_result.stdout:
            return
        
        # Install RMark
        logger.info("Installing RMark package...")
        install_result = subprocess.run([
            self.config.local_r_path, "-e",
            'if (!require("RMark", quietly=TRUE)) install.packages("RMark", repos="https://cran.rstudio.com/", dependencies=TRUE)'
        ], capture_output=True, text=True, timeout=600)  # 10 minutes for installation
        
        if install_result.returncode != 0:
            raise RuntimeError(f"Failed to install RMark: {install_result.stderr}")
        
        logger.info("RMark installation completed")
    
    def _save_data_for_rmark(self, data: DataContext, file_path: Path):
        """Save data in RMark format."""
        import pandas as pd
        
        # Convert capture matrix to capture histories
        data_dict = {'ch': []}
        for i in range(data.n_individuals):
            ch = ''.join(str(int(data.capture_matrix[i, j])) for j in range(data.n_occasions))
            data_dict['ch'].append(ch)
        
        # Add covariates
        for name, values in data.covariates.items():
            data_dict[name] = values.tolist()
        
        df = pd.DataFrame(data_dict)
        df.to_csv(file_path, index=False)
    
    def _generate_local_rmark_script(self, data_file: Path, formula_spec: FormulaSpec) -> str:
        """Generate R script for local execution."""
        # Similar to SSH script but with local paths
        return f'''
library(RMark)
library(jsonlite)
options(warn = -1)

tryCatch({{
    data <- read.csv("{data_file}", stringsAsFactors = FALSE)
    
    processed <- process.data(data, model = "Pradel")
    ddl <- make.design.data(processed)
    
    model <- mark(
        processed, ddl,
        model.parameters = list(
            Phi = list(formula = ~{formula_spec.phi}),
            p = list(formula = ~{formula_spec.p}),
            f = list(formula = ~{formula_spec.f})
        ),
        delete = TRUE,
        output = FALSE,
        silent = TRUE
    )
    
    if (!is.null(model) && model$convergence == 0) {{
        beta_estimates <- model$results$beta
        param_names <- row.names(beta_estimates)
        param_values <- as.numeric(beta_estimates$estimate)
        param_se <- as.numeric(beta_estimates$se)
        
        parameters <- setNames(param_values, param_names)
        std_errors <- setNames(param_se, param_names)
        
        results <- list(
            converged = TRUE,
            aic = model$results$AICc,
            log_likelihood = model$results$lnl,
            n_parameters = model$results$npar,
            parameters = parameters,
            std_errors = std_errors
        )
    }} else {{
        results <- list(
            converged = FALSE,
            error = "Model convergence failed"
        )
    }}
    
    cat("RMARK_RESULTS_START\\n")
    cat(toJSON(results, auto_unbox = TRUE, digits = 8))
    cat("\\nRMARK_RESULTS_END\\n")
    
}}, error = function(e) {{
    error_result <- list(
        converged = FALSE,
        error = paste("R Error:", e$message)
    )
    cat("RMARK_RESULTS_START\\n")
    cat(toJSON(error_result, auto_unbox = TRUE))
    cat("\\nRMARK_RESULTS_END\\n")
}})
'''
    
    def _parse_local_output(self, result: subprocess.CompletedProcess, formula_spec: FormulaSpec) -> RMarkResult:
        """Parse local R execution output."""
        if result.returncode != 0:
            raise RuntimeError(f"R execution failed: {result.stderr}")
        
        # Parse output similar to SSH method
        output = result.stdout
        start_marker = "RMARK_RESULTS_START"
        end_marker = "RMARK_RESULTS_END"
        
        start_idx = output.find(start_marker)
        end_idx = output.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not find RMark results in output")
        
        json_text = output[start_idx + len(start_marker):end_idx].strip()
        
        try:
            results = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse RMark JSON results: {e}")
        
        return RMarkResult(
            model_formula=self._format_formula_string(formula_spec),
            converged=results.get('converged', False),
            aic=results.get('aic', 0.0),
            log_likelihood=results.get('log_likelihood', 0.0),
            n_parameters=results.get('n_parameters', 0),
            parameters=results.get('parameters', {}),
            std_errors=results.get('std_errors', {}),
            raw_output=output,
            notes=[] if results.get('converged') else [results.get('error', 'Unknown error')]
        )
    
    def _format_formula_string(self, formula_spec: FormulaSpec) -> str:
        """Format formula specification as string."""
        return f"phi({formula_spec.phi}), p({formula_spec.p}), f({formula_spec.f})"


class MockRMarkExecutor:
    """Generate mock RMark results for development/testing."""
    
    def __init__(self, config: SecureValidationConfig):
        self.config = config
        self.session_id = config.session_id
    
    def execute(self, data: DataContext, formula_spec: FormulaSpec) -> RMarkResult:
        """Generate realistic mock RMark results."""
        logger.info(f"Generating mock RMark results [{self.session_id}]")
        
        import numpy as np
        np.random.seed(42)  # Reproducible mock results
        
        start_time = time.time()
        
        # Mock parameter estimates (realistic values with small noise)
        parameters = {}
        std_errors = {}
        
        # Generate intercept parameters
        parameters['Phi:(Intercept)'] = 0.85 + np.random.normal(0, 0.01)
        parameters['p:(Intercept)'] = 0.65 + np.random.normal(0, 0.01)
        parameters['f:(Intercept)'] = 0.25 + np.random.normal(0, 0.01)
        
        std_errors['Phi:(Intercept)'] = 0.05 + abs(np.random.normal(0, 0.01))
        std_errors['p:(Intercept)'] = 0.04 + abs(np.random.normal(0, 0.01))
        std_errors['f:(Intercept)'] = 0.03 + abs(np.random.normal(0, 0.01))
        
        # Add covariate effects if present in formulas
        for param_type, formula in [('Phi', formula_spec.phi), ('p', formula_spec.p), ('f', formula_spec.f)]:
            if formula != '1':  # Not intercept-only
                # Simple parsing for common covariates
                if 'sex' in formula.lower():
                    parameters[f'{param_type}:sex'] = np.random.normal(0, 0.1)
                    std_errors[f'{param_type}:sex'] = 0.05 + abs(np.random.normal(0, 0.01))
                
                if 'age' in formula.lower():
                    parameters[f'{param_type}:age'] = np.random.normal(0, 0.05)
                    std_errors[f'{param_type}:age'] = 0.03 + abs(np.random.normal(0, 0.01))
        
        # Calculate mock log-likelihood and AIC
        n_params = len(parameters)
        mock_ll = -50.0 + np.random.normal(0, 5)  # Realistic log-likelihood
        mock_aic = -2 * mock_ll + 2 * n_params
        
        execution_time = time.time() - start_time + np.random.uniform(0.1, 0.5)  # Mock execution time
        
        return RMarkResult(
            model_formula=self._format_formula_string(formula_spec),
            execution_method="mock",
            execution_time=execution_time,
            converged=True,
            aic=mock_aic,
            log_likelihood=mock_ll,
            n_parameters=n_params,
            parameters=parameters,
            std_errors=std_errors,
            execution_status=ExecutionStatus.SUCCESS,
            notes=["Mock validation result - no actual RMark execution"]
        )
    
    def _format_formula_string(self, formula_spec: FormulaSpec) -> str:
        """Format formula specification as string."""
        return f"phi({formula_spec.phi}), p({formula_spec.p}), f({formula_spec.f})"


class RMarkExecutor:
    """Main RMark executor with automatic method selection."""
    
    def __init__(self, config: Optional[SecureValidationConfig] = None):
        self.config = config or get_secure_validation_config()
        self.detector = EnvironmentDetector(self.config)
        self.available_methods = self.detector.detect_available_methods()
        
        logger.info(f"RMark executor initialized with methods: {[m.value for m in self.available_methods]}")
    
    def execute_rmark_analysis(
        self,
        data: DataContext,
        formula_spec: FormulaSpec,
        method: Optional[ExecutionMethod] = None
    ) -> RMarkResult:
        """Execute RMark analysis with automatic or specified method."""
        
        # Determine execution method
        if method is None:
            method = self.detector.get_recommended_method()
        
        if method not in self.available_methods:
            logger.warning(f"Requested method {method.value} not available, using mock")
            method = ExecutionMethod.MOCK
        
        logger.info(f"Executing RMark analysis via {method.value}")
        
        # Create appropriate executor
        if method == ExecutionMethod.SSH:
            executor = SSHRMarkExecutor(self.config)
        elif method == ExecutionMethod.LOCAL_R:
            executor = LocalRMarkExecutor(self.config)
        elif method == ExecutionMethod.MOCK:
            executor = MockRMarkExecutor(self.config)
        else:
            raise ValueError(f"Unsupported execution method: {method}")
        
        # Execute and return result
        try:
            return executor.execute(data, formula_spec)
        except Exception as e:
            logger.error(f"RMark execution failed with {method.value}: {e}")
            
            # Fallback to mock if other methods fail
            if method != ExecutionMethod.MOCK:
                logger.info("Falling back to mock execution")
                mock_executor = MockRMarkExecutor(self.config)
                result = mock_executor.execute(data, formula_spec)
                result.notes.append(f"Fallback to mock after {method.value} failure: {str(e)}")
                return result
            else:
                raise


# Convenience function for external use
def execute_rmark_analysis(
    data: DataContext,
    formula_spec: FormulaSpec,
    method: Optional[str] = None,
    config: Optional[SecureValidationConfig] = None
) -> RMarkResult:
    """
    Convenience function to execute RMark analysis.
    
    Args:
        data: Data context with capture histories and covariates
        formula_spec: Model formula specification
        method: Execution method ("ssh", "local_r", "mock", or None for auto)
        config: Optional configuration (uses default if None)
        
    Returns:
        RMark analysis result
    """
    executor = RMarkExecutor(config)
    
    execution_method = None
    if method:
        try:
            execution_method = ExecutionMethod(method.lower())
        except ValueError:
            logger.warning(f"Unknown method '{method}', using auto-detection")
    
    return executor.execute_rmark_analysis(data, formula_spec, execution_method)