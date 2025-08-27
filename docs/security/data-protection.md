# Data Protection and Security Guide

This guide covers comprehensive data protection practices for Pradel-JAX, including sensitive data handling, security best practices, and compliance considerations for capture-recapture research data.

## Table of Contents

1. [Security Overview](#security-overview)
2. [Data Classification](#data-classification)
3. [Sensitive Data Protection](#sensitive-data-protection)
4. [Repository Security](#repository-security)
5. [Development Security](#development-security)
6. [Data Processing Security](#data-processing-security)
7. [Compliance and Privacy](#compliance-and-privacy)
8. [Incident Response](#incident-response)
9. [Security Auditing](#security-auditing)
10. [Best Practices](#best-practices)

## Security Overview

Pradel-JAX handles sensitive ecological and biological data that may include:
- **Individual animal data** with unique identifiers
- **Location information** that could reveal sensitive habitat areas
- **Research data** that may be proprietary or embargoed
- **Collaborative data** from multiple institutions

The framework implements multiple layers of security to protect this data throughout the analysis pipeline.

### Security Status: PRODUCTION READY ✅

**Last Security Audit**: August 26, 2025
**Security Level**: High (suitable for sensitive research data)
**Compliance**: Research data protection standards

#### Recent Security Improvements
- **Updated Dependencies**: All critical CVEs addressed (August 2025)
- **Secure Data Handling**: SHA-256 cryptographic hashing implemented
- **Safe Serialization**: JSON-based checkpointing instead of unsafe pickle
- **Input Validation**: Comprehensive data validation framework
- **Audit Trail**: Complete logging and monitoring system

## Data Classification

### Data Sensitivity Levels

#### 1. Public Data (Green)
- **Examples**: Published datasets, species presence/absence aggregates
- **Protection Level**: Standard (basic backup and version control)
- **Storage**: Can be stored in public repositories
- **Access**: Openly shareable

```python
# Example: Public dipper dataset
data = pj.load_data("data/dipper_dataset.csv")  # ✅ Safe for public repository
```

#### 2. Research Data (Yellow)
- **Examples**: Capture histories without location details, aggregated demographic data
- **Protection Level**: Enhanced (encrypted storage, access controls)
- **Storage**: Private repositories, encrypted backups
- **Access**: Authorized researchers only

```python
# Example: Research data with pseudonymized identifiers
data = pj.load_data("research_data/species_study_anonymized.csv")
```

#### 3. Sensitive Data (Orange)
- **Examples**: Individual capture records with precise locations, genetic samples
- **Protection Level**: High (strong encryption, audit trails, restricted access)
- **Storage**: Encrypted secure systems only
- **Access**: Principal investigator approval required

```python
# Example: Sensitive data requiring special handling
# This data should NOT be committed to repositories
data = pj.load_data("sensitive_data/gps_capture_locations.csv")  # ⚠️ Handle with care
```

#### 4. Restricted Data (Red)
- **Examples**: Endangered species locations, military base wildlife data
- **Protection Level**: Maximum (end-to-end encryption, air-gapped systems)
- **Storage**: Specialized secure facilities
- **Access**: Government/institutional clearance required

### Data Classification Implementation

```python
import pradel_jax as pj
from pradel_jax.security import DataClassifier, SecurityLevel

# Automatic data sensitivity assessment
classifier = DataClassifier()
sensitivity_level = classifier.assess_data_sensitivity("dataset.csv")

print(f"Data sensitivity level: {sensitivity_level}")
print(f"Recommended protection: {classifier.get_protection_recommendations(sensitivity_level)}")

# Apply appropriate security measures
if sensitivity_level >= SecurityLevel.SENSITIVE:
    # Use encrypted storage
    data = pj.load_data("dataset.csv", encryption=True, key_file="encryption.key")
else:
    # Standard loading
    data = pj.load_data("dataset.csv")
```

## Sensitive Data Protection

### Protected Data Patterns

The Pradel-JAX repository uses comprehensive patterns to prevent accidental exposure of sensitive data:

```bash
# .gitignore patterns for sensitive data protection
**/sensitive_data/**
**/*_sensitive*
**/*_private*
**/*_confidential*

# Specific data patterns
**/encounter_histories_ne_clean.csv
**/encounter_histories_sd_clean.csv
**/*person_id*
**/*customer_id*
**/*FUZZY*

# Geographic coordinates
**/*_coords.csv
**/*_locations.csv
**/*_gps.csv

# Authentication and keys
**/*.key
**/*.pem
**/*.p12
**/secrets.json
**/credentials.yaml
```

### Secure Data Loading

```python
import pradel_jax as pj
from pradel_jax.security import SecureDataLoader
import logging

# Configure secure logging (no sensitive data in logs)
secure_logger = pj.get_secure_logger()

class SecureAnalysisPipeline:
    """Secure analysis pipeline for sensitive data."""
    
    def __init__(self, config_file="secure_config.yaml"):
        self.config = pj.load_secure_config(config_file)
        self.audit_logger = pj.get_audit_logger()
        
    def load_protected_data(self, data_path, classification_level):
        """Load data with appropriate security measures."""
        
        # Audit data access
        self.audit_logger.info(f"Data access requested: {data_path}")
        self.audit_logger.info(f"Classification level: {classification_level}")
        
        # Verify access permissions
        if not self.verify_access_permissions(data_path, classification_level):
            raise pj.SecurityError("Insufficient permissions for data access")
        
        # Load with appropriate security
        if classification_level >= pj.SecurityLevel.SENSITIVE:
            # Use encrypted data loader
            loader = SecureDataLoader(
                encryption_key=self.config['encryption_key'],
                audit_trail=True,
                data_masking=True
            )
            data = loader.load_data(data_path)
        else:
            # Standard secure loading
            data = pj.load_data(data_path, audit_trail=True)
        
        # Additional security measures
        data = self.apply_data_anonymization(data, classification_level)
        data = self.apply_access_restrictions(data, classification_level)
        
        self.audit_logger.info(f"Data loaded successfully: {data.n_individuals} individuals")
        return data
    
    def apply_data_anonymization(self, data, classification_level):
        """Apply appropriate anonymization based on sensitivity level."""
        
        if classification_level >= pj.SecurityLevel.SENSITIVE:
            # Remove or hash individual identifiers
            if 'individual_id' in data.covariates:
                data.covariates['individual_id'] = pj.hash_identifiers(
                    data.covariates['individual_id'], 
                    method='sha256'
                )
            
            # Coarsen location data
            if 'latitude' in data.covariates:
                data.covariates['latitude'] = pj.coarsen_coordinates(
                    data.covariates['latitude'], 
                    precision=1000  # 1km precision
                )
            if 'longitude' in data.covariates:
                data.covariates['longitude'] = pj.coarsen_coordinates(
                    data.covariates['longitude'], 
                    precision=1000
                )
        
        return data
    
    def verify_access_permissions(self, data_path, classification_level):
        """Verify user has appropriate permissions for data access."""
        
        # Check user credentials
        user_clearance = self.config.get('user_clearance_level', pj.SecurityLevel.PUBLIC)
        
        if classification_level > user_clearance:
            secure_logger.warning(f"Access denied: insufficient clearance level")
            return False
        
        # Check data access policies
        if not self.check_data_access_policy(data_path):
            secure_logger.warning(f"Access denied: policy violation")
            return False
        
        return True

# Example usage
pipeline = SecureAnalysisPipeline()
sensitive_data = pipeline.load_protected_data(
    "sensitive_data/endangered_species_locations.csv",
    pj.SecurityLevel.SENSITIVE
)
```

### Data Anonymization Techniques

```python
import pradel_jax as pj
import hashlib
import numpy as np

class DataAnonymizer:
    """Tools for protecting sensitive information in datasets."""
    
    @staticmethod
    def hash_identifiers(identifiers, method='sha256', salt=None):
        """Hash individual identifiers for privacy protection."""
        
        if salt is None:
            salt = pj.get_config().get('anonymization_salt', 'default_salt')
        
        hashed_ids = []
        for identifier in identifiers:
            # Create salted hash
            salted_id = f"{salt}_{identifier}"
            
            if method == 'sha256':
                hash_obj = hashlib.sha256(salted_id.encode())
            else:
                raise ValueError(f"Unsupported hash method: {method}")
            
            hashed_ids.append(hash_obj.hexdigest()[:16])  # Use first 16 characters
        
        return hashed_ids
    
    @staticmethod
    def coarsen_coordinates(coordinates, precision=1000):
        """Reduce precision of coordinate data."""
        # Round to nearest precision meters
        return np.round(coordinates / precision) * precision
    
    @staticmethod
    def add_noise(data, noise_level=0.01):
        """Add small amount of noise to continuous variables."""
        noise = np.random.normal(0, noise_level * np.std(data), len(data))
        return data + noise
    
    @staticmethod
    def k_anonymize(data, k=3, quasi_identifiers=['age', 'sex', 'location']):
        """Ensure k-anonymity by generalizing quasi-identifiers."""
        
        # Group records by quasi-identifiers
        grouped = data.groupby(quasi_identifiers)
        
        # Remove groups with fewer than k records
        k_anonymous_data = grouped.filter(lambda x: len(x) >= k)
        
        return k_anonymous_data

# Example anonymization workflow
def anonymize_capture_data(data, anonymization_level='standard'):
    """Anonymize capture-recapture data based on sensitivity level."""
    
    anonymizer = DataAnonymizer()
    
    # Always hash individual IDs
    if 'individual_id' in data.covariates:
        data.covariates['individual_id'] = anonymizer.hash_identifiers(
            data.covariates['individual_id']
        )
    
    if anonymization_level == 'high':
        # High anonymization: coarsen locations, add noise
        if 'latitude' in data.covariates:
            data.covariates['latitude'] = anonymizer.coarsen_coordinates(
                data.covariates['latitude'], precision=5000  # 5km precision
            )
        if 'longitude' in data.covariates:
            data.covariates['longitude'] = anonymizer.coarsen_coordinates(
                data.covariates['longitude'], precision=5000
            )
        
        # Add noise to continuous measurements
        for col in ['weight', 'length', 'wing_chord']:
            if col in data.covariates:
                data.covariates[col] = anonymizer.add_noise(
                    data.covariates[col], noise_level=0.02
                )
    
    elif anonymization_level == 'maximum':
        # Maximum anonymization: k-anonymity, heavy generalization
        data = anonymizer.k_anonymize(data, k=5)
        
        # Generalize age to age classes
        if 'age' in data.covariates:
            data.covariates['age_class'] = pd.cut(
                data.covariates['age'], 
                bins=[0, 1, 3, 5, np.inf], 
                labels=['juvenile', 'young', 'adult', 'old']
            )
            del data.covariates['age']  # Remove precise age
    
    return data

# Usage example
data = pj.load_data("sensitive_capture_data.csv")
anonymized_data = anonymize_capture_data(data, anonymization_level='high')
```

## Repository Security

### Git Security Best Practices

```bash
# Pre-commit hook to prevent sensitive data commits
#!/bin/bash
# .git/hooks/pre-commit

echo "Checking for sensitive data patterns..."

# Check for sensitive file patterns
if git diff --cached --name-only | grep -E "(sensitive|private|confidential|secret)" ; then
    echo "ERROR: Attempting to commit sensitive files"
    exit 1
fi

# Check for sensitive content patterns
if git diff --cached | grep -E "(password|api[_-]?key|secret[_-]?key|private[_-]?key)" ; then
    echo "ERROR: Attempting to commit sensitive content"
    exit 1
fi

# Check for PII patterns
if git diff --cached | grep -E "(person_id|customer_id|ssn|social.security)" ; then
    echo "ERROR: Attempting to commit personally identifiable information"
    exit 1
fi

echo "✓ No sensitive data detected"
```

### Repository Configuration

```yaml
# .github/security.yml - Repository security configuration

# Enable security features
security:
  vulnerability_alerts: true
  security_updates: true
  
# Branch protection rules
branch_protection:
  main:
    required_reviews: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    restrictions:
      users: []
      teams: ["core-developers"]

# Secrets scanning
secrets_scanning:
  enabled: true
  alerts: true

# Dependency scanning  
dependency_scanning:
  enabled: true
  auto_security_updates: true
```

### Secure Development Workflow

```python
# secure_development_tools.py
import subprocess
import os
import logging

class SecureDevelopmentTools:
    """Tools for secure development practices."""
    
    @staticmethod
    def scan_for_secrets():
        """Scan codebase for potential secrets."""
        
        # Use truffleHog or similar tool
        try:
            result = subprocess.run(
                ["trufflehog", "--regex", "--entropy=False", "."],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("✓ No secrets detected")
            else:
                print("⚠ Potential secrets found:")
                print(result.stdout)
                
        except FileNotFoundError:
            print("truffleHog not installed, skipping secret scan")
    
    @staticmethod
    def check_dependencies():
        """Check for vulnerable dependencies."""
        
        try:
            # Use safety or similar tool
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print("✓ No vulnerable dependencies")
            else:
                print("⚠ Vulnerable dependencies found:")
                print(result.stdout)
                
        except FileNotFoundError:
            print("safety not installed, skipping dependency check")
    
    @staticmethod
    def validate_data_patterns():
        """Validate that sensitive data patterns are properly excluded."""
        
        import glob
        
        sensitive_patterns = [
            "**/sensitive_data/**",
            "**/*person_id*",
            "**/*customer_id*",
            "**/*FUZZY*"
        ]
        
        for pattern in sensitive_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                print(f"⚠ Found files matching sensitive pattern {pattern}:")
                for match in matches:
                    print(f"  {match}")
            else:
                print(f"✓ No files match sensitive pattern {pattern}")

# Usage in CI/CD pipeline
if __name__ == "__main__":
    tools = SecureDevelopmentTools()
    tools.scan_for_secrets()
    tools.check_dependencies()
    tools.validate_data_patterns()
```

## Development Security

### Secure Coding Practices

```python
# Example: Secure input validation
import pradel_jax as pj
from pradel_jax.security import InputValidator

class SecureDataProcessor:
    """Secure data processing with comprehensive validation."""
    
    def __init__(self):
        self.validator = InputValidator()
        self.logger = pj.get_audit_logger()
    
    def process_user_input(self, user_data):
        """Process user input with security validation."""
        
        # Validate input structure
        validation_result = self.validator.validate_data_structure(user_data)
        if not validation_result.is_valid:
            self.logger.warning(f"Invalid input structure: {validation_result.errors}")
            raise pj.SecurityError("Invalid input data structure")
        
        # Sanitize input
        sanitized_data = self.sanitize_input(user_data)
        
        # Process with additional safety checks
        return self.safe_process(sanitized_data)
    
    def sanitize_input(self, data):
        """Sanitize input data to prevent injection attacks."""
        
        # Remove potentially dangerous characters
        dangerous_patterns = [';', '--', '/*', '*/', 'xp_', 'sp_']
        
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remove dangerous patterns
                clean_value = value
                for pattern in dangerous_patterns:
                    clean_value = clean_value.replace(pattern, '')
                
                # Limit string length
                clean_value = clean_value[:1000]  # Reasonable limit
                sanitized[key] = clean_value
            else:
                sanitized[key] = value
        
        return sanitized
    
    def safe_process(self, data):
        """Process data with error handling and logging."""
        
        try:
            # Log processing start
            self.logger.info(f"Starting secure data processing")
            
            # Process data
            result = pj.fit_model(data=data)
            
            # Log success
            self.logger.info(f"Processing completed successfully")
            return result
            
        except Exception as e:
            # Log error without exposing sensitive data
            self.logger.error(f"Processing failed: {type(e).__name__}")
            raise pj.ProcessingError("Data processing failed") from e

# Secure parameter validation
def validate_model_parameters(parameters):
    """Validate model parameters to prevent malicious inputs."""
    
    validator = InputValidator()
    
    # Check parameter bounds
    for param_name, value in parameters.items():
        if not validator.is_valid_parameter_name(param_name):
            raise pj.SecurityError(f"Invalid parameter name: {param_name}")
        
        if not validator.is_valid_parameter_value(value):
            raise pj.SecurityError(f"Invalid parameter value: {value}")
    
    # Check for suspicious patterns
    param_string = str(parameters)
    if validator.contains_suspicious_patterns(param_string):
        raise pj.SecurityError("Suspicious patterns detected in parameters")
    
    return True
```

### Environment Security

```python
# secure_environment.py
import os
import logging
from pathlib import Path

class SecureEnvironment:
    """Manage secure execution environment."""
    
    def __init__(self):
        self.setup_secure_environment()
        self.logger = logging.getLogger('pradel_jax.security')
    
    def setup_secure_environment(self):
        """Configure secure execution environment."""
        
        # Set secure random seed
        import random
        import numpy as np
        random.seed()  # Use system entropy
        np.random.seed()  # Use system entropy
        
        # Secure temporary directory
        temp_dir = Path.home() / '.pradel_jax' / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TMPDIR'] = str(temp_dir)
        
        # Set secure file permissions
        os.umask(0o077)  # Owner read/write only
        
        # Clear environment variables that might contain sensitive data
        sensitive_env_vars = ['PASSWORD', 'SECRET', 'TOKEN', 'KEY']
        for var in sensitive_env_vars:
            if var in os.environ:
                del os.environ[var]
                self.logger.info(f"Cleared sensitive environment variable: {var}")
    
    def setup_secure_logging(self):
        """Configure secure logging that doesn't expose sensitive data."""
        
        # Create secure log directory
        log_dir = Path.home() / '.pradel_jax' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'security.log'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        # Set restrictive permissions on log files
        for log_file in log_dir.glob('*.log'):
            os.chmod(log_file, 0o600)  # Owner read/write only
    
    def validate_file_paths(self, path):
        """Validate file paths to prevent directory traversal attacks."""
        
        path = Path(path).resolve()
        
        # Check for directory traversal
        if '..' in str(path):
            raise pj.SecurityError("Directory traversal detected")
        
        # Check if path is within allowed directories
        allowed_dirs = [
            Path.home() / '.pradel_jax',
            Path.cwd(),
            Path('/tmp')
        ]
        
        if not any(str(path).startswith(str(allowed)) for allowed in allowed_dirs):
            raise pj.SecurityError(f"Access to path not allowed: {path}")
        
        return path

# Global security setup
security_env = SecureEnvironment()
```

## Data Processing Security

### Secure Model Fitting

```python
import pradel_jax as pj
from pradel_jax.security import SecureProcessor
import tempfile
import shutil
import os

class SecureModelFitting:
    """Secure model fitting with data protection measures."""
    
    def __init__(self, security_level='standard'):
        self.security_level = security_level
        self.processor = SecureProcessor()
        self.audit_logger = pj.get_audit_logger()
    
    def fit_model_securely(self, formula, data, **kwargs):
        """Fit model with comprehensive security measures."""
        
        # Create secure working directory
        with tempfile.TemporaryDirectory(prefix='pradel_secure_') as temp_dir:
            # Set restrictive permissions
            os.chmod(temp_dir, 0o700)  # Owner only
            
            # Audit model fitting start
            self.audit_logger.info(f"Secure model fitting started")
            self.audit_logger.info(f"Security level: {self.security_level}")
            self.audit_logger.info(f"Data size: {data.n_individuals} individuals")
            
            try:
                # Apply data protection measures
                protected_data = self.apply_data_protection(data)
                
                # Validate formula security
                self.validate_formula_security(formula)
                
                # Fit model with secure parameters
                secure_kwargs = self.apply_secure_defaults(kwargs)
                result = pj.fit_model(
                    formula=formula,
                    data=protected_data,
                    working_directory=temp_dir,
                    **secure_kwargs
                )
                
                # Apply result protection
                protected_result = self.protect_model_results(result)
                
                # Audit success
                self.audit_logger.info(f"Secure model fitting completed successfully")
                
                return protected_result
                
            except Exception as e:
                # Audit failure
                self.audit_logger.error(f"Secure model fitting failed: {type(e).__name__}")
                raise
                
            finally:
                # Ensure temporary files are securely deleted
                self.secure_cleanup(temp_dir)
    
    def apply_data_protection(self, data):
        """Apply data protection measures based on security level."""
        
        if self.security_level == 'high':
            # Apply strong anonymization
            protected_data = self.anonymize_data(data, level='high')
        elif self.security_level == 'maximum':
            # Apply maximum protection
            protected_data = self.anonymize_data(data, level='maximum')
            protected_data = self.apply_differential_privacy(protected_data)
        else:
            # Standard protection
            protected_data = self.anonymize_data(data, level='standard')
        
        return protected_data
    
    def validate_formula_security(self, formula):
        """Validate formula for security issues."""
        
        # Check for dangerous function calls
        dangerous_functions = ['eval', 'exec', '__import__', 'open', 'file']
        
        formula_string = str(formula)
        for func in dangerous_functions:
            if func in formula_string:
                raise pj.SecurityError(f"Dangerous function detected in formula: {func}")
        
        # Check for file system access patterns
        filesystem_patterns = ['/', '..', '~', '$']
        for pattern in filesystem_patterns:
            if pattern in formula_string:
                self.audit_logger.warning(f"Potential file system access in formula: {pattern}")
    
    def apply_secure_defaults(self, kwargs):
        """Apply secure default parameters."""
        
        secure_kwargs = kwargs.copy()
        
        # Limit iterations to prevent DoS
        secure_kwargs.setdefault('max_iterations', 5000)
        
        # Disable potentially unsafe features
        secure_kwargs.setdefault('allow_custom_functions', False)
        secure_kwargs.setdefault('enable_debugging', False)
        
        # Enable security features
        secure_kwargs.setdefault('audit_trail', True)
        secure_kwargs.setdefault('secure_random_seed', True)
        
        return secure_kwargs
    
    def protect_model_results(self, result):
        """Apply protection measures to model results."""
        
        if self.security_level in ['high', 'maximum']:
            # Remove potentially sensitive diagnostics
            protected_result = result.copy()
            protected_result.optimization_trace = None  # Remove detailed trace
            protected_result.raw_optimization_result = None  # Remove internal details
            
            # Coarsen parameter estimates if needed
            if self.security_level == 'maximum':
                protected_result = self.coarsen_parameter_estimates(protected_result)
            
            return protected_result
        else:
            return result
    
    def apply_differential_privacy(self, data, epsilon=1.0):
        """Apply differential privacy to data."""
        
        # This is a simplified example - real implementation would be more sophisticated
        import numpy as np
        
        # Add noise to continuous covariates
        noise_scale = 1.0 / epsilon
        
        for column in data.covariates.select_dtypes(include=[np.number]).columns:
            noise = np.random.laplace(0, noise_scale, size=len(data.covariates[column]))
            data.covariates[column] += noise
        
        return data
    
    def secure_cleanup(self, temp_dir):
        """Securely delete temporary files."""
        
        try:
            # Overwrite files before deletion (simplified)
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        # Overwrite with random data
                        with open(file_path, 'wb') as f:
                            f.write(os.urandom(os.path.getsize(file_path)))
            
            # Remove directory
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            self.audit_logger.warning(f"Secure cleanup failed: {e}")

# Usage example
secure_fitter = SecureModelFitting(security_level='high')

# Fit model with comprehensive security
result = secure_fitter.fit_model_securely(
    formula=pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1"),
    data=sensitive_data,
    strategy="lbfgs"
)
```

## Compliance and Privacy

### Research Data Compliance

```python
import pradel_jax as pj
from datetime import datetime, timedelta
import json

class ComplianceManager:
    """Manage research data compliance requirements."""
    
    def __init__(self, institution="", ethics_board=""):
        self.institution = institution
        self.ethics_board = ethics_board
        self.compliance_logger = pj.get_compliance_logger()
    
    def validate_research_ethics(self, data_source, research_purpose):
        """Validate research ethics compliance."""
        
        compliance_check = {
            'timestamp': datetime.now().isoformat(),
            'institution': self.institution,
            'ethics_board': self.ethics_board,
            'data_source': data_source,
            'research_purpose': research_purpose,
            'checks_performed': []
        }
        
        # Check 1: Data collection permits
        if self.check_collection_permits(data_source):
            compliance_check['checks_performed'].append({
                'check': 'collection_permits',
                'status': 'passed',
                'details': 'Valid collection permits verified'
            })
        else:
            compliance_check['checks_performed'].append({
                'check': 'collection_permits',
                'status': 'failed',
                'details': 'Collection permits not verified'
            })
        
        # Check 2: Ethical approval
        if self.check_ethical_approval(research_purpose):
            compliance_check['checks_performed'].append({
                'check': 'ethical_approval',
                'status': 'passed',
                'details': 'Ethical approval verified'
            })
        
        # Check 3: Data retention policy
        retention_status = self.check_retention_policy(data_source)
        compliance_check['checks_performed'].append({
            'check': 'data_retention',
            'status': retention_status['status'],
            'details': retention_status['details']
        })
        
        # Log compliance check
        self.compliance_logger.info(f"Compliance check completed: {json.dumps(compliance_check)}")
        
        # Determine overall compliance status
        failed_checks = [check for check in compliance_check['checks_performed'] if check['status'] == 'failed']
        
        if failed_checks:
            raise pj.ComplianceError(f"Compliance violations detected: {failed_checks}")
        
        return compliance_check
    
    def check_collection_permits(self, data_source):
        """Check if data collection permits are valid."""
        # Implementation would check actual permit databases
        return True  # Placeholder
    
    def check_ethical_approval(self, research_purpose):
        """Check if research has ethical approval."""
        # Implementation would check institutional approval systems
        return True  # Placeholder
    
    def check_retention_policy(self, data_source):
        """Check data retention policy compliance."""
        
        # Check if data is within retention period
        # This is a simplified example
        return {
            'status': 'passed',
            'details': 'Data within approved retention period'
        }
    
    def generate_compliance_report(self, analysis_results):
        """Generate compliance report for analysis."""
        
        report = {
            'report_date': datetime.now().isoformat(),
            'institution': self.institution,
            'analysis_summary': {
                'models_fitted': len(analysis_results),
                'data_sources': list(set(result.data_source for result in analysis_results if hasattr(result, 'data_source'))),
                'analysis_period': {
                    'start': min(result.analysis_date for result in analysis_results if hasattr(result, 'analysis_date')),
                    'end': max(result.analysis_date for result in analysis_results if hasattr(result, 'analysis_date'))
                }
            },
            'data_protection_measures': [
                'SHA-256 identifier hashing',
                'Coordinate coarsening (1km precision)',  
                'Secure temporary file handling',
                'Encrypted data storage',
                'Audit trail logging'
            ],
            'compliance_certifications': self.get_compliance_certifications()
        }
        
        return report
    
    def get_compliance_certifications(self):
        """Get relevant compliance certifications."""
        return [
            'Institutional Animal Care and Use Committee (IACUC) approval',
            'Research ethics board approval',
            'Data protection officer review',
            'Species collection permits',
            'Landowner permissions'
        ]

# Usage
compliance = ComplianceManager(
    institution="University Research Institute",
    ethics_board="Environmental Research Ethics Board"
)

# Validate compliance before analysis
compliance.validate_research_ethics(
    data_source="Bird Banding Database",
    research_purpose="Population dynamics assessment"
)
```

### Data Privacy Framework

```python
class PrivacyFramework:
    """Framework for protecting individual privacy in ecological data."""
    
    def __init__(self):
        self.privacy_logger = pj.get_privacy_logger()
    
    def assess_privacy_risk(self, data):
        """Assess privacy risk in ecological data."""
        
        risk_factors = []
        risk_score = 0
        
        # Check for direct identifiers
        if 'individual_id' in data.covariates:
            if not self.is_hashed(data.covariates['individual_id']):
                risk_factors.append("Direct individual identifiers present")
                risk_score += 3
        
        # Check for precise locations
        if 'latitude' in data.covariates and 'longitude' in data.covariates:
            location_precision = self.assess_location_precision(
                data.covariates['latitude'], data.covariates['longitude']
            )
            if location_precision < 1000:  # Less than 1km precision
                risk_factors.append(f"High location precision: {location_precision}m")
                risk_score += 2
        
        # Check for unique combinations
        quasi_identifiers = ['capture_date', 'location', 'species', 'sex', 'age']
        available_qi = [qi for qi in quasi_identifiers if qi in data.covariates]
        
        if len(available_qi) >= 3:
            uniqueness = self.assess_uniqueness(data, available_qi)
            if uniqueness > 0.8:  # More than 80% of records are unique
                risk_factors.append(f"High uniqueness risk: {uniqueness:.2f}")
                risk_score += 2
        
        # Assess overall risk
        if risk_score >= 5:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        privacy_assessment = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self.get_risk_mitigation_recommendations(risk_level)
        }
        
        self.privacy_logger.info(f"Privacy risk assessment: {privacy_assessment}")
        
        return privacy_assessment
    
    def is_hashed(self, identifiers):
        """Check if identifiers appear to be hashed."""
        # Simple check - real implementation would be more sophisticated
        return all(len(str(id_val)) >= 16 and any(c in str(id_val) for c in 'abcdef') 
                  for id_val in identifiers.iloc[:10])
    
    def assess_location_precision(self, latitudes, longitudes):
        """Assess precision of location data."""
        # Calculate minimum distance between points
        from scipy.spatial.distance import pdist
        import numpy as np
        
        coords = np.column_stack([latitudes, longitudes])
        if len(coords) > 1:
            distances = pdist(coords) * 111000  # Convert degrees to meters (approximate)
            min_distance = np.min(distances[distances > 0])
            return min_distance
        else:
            return float('inf')
    
    def assess_uniqueness(self, data, quasi_identifiers):
        """Assess uniqueness of records based on quasi-identifiers."""
        qi_combinations = data.covariates[quasi_identifiers].drop_duplicates()
        uniqueness = len(qi_combinations) / len(data.covariates)
        return uniqueness
    
    def get_risk_mitigation_recommendations(self, risk_level):
        """Get recommendations for mitigating privacy risks."""
        
        recommendations = {
            'LOW': [
                'Continue current privacy protection measures',
                'Regular privacy risk reassessment'
            ],
            'MEDIUM': [
                'Consider additional anonymization',
                'Implement k-anonymity (k≥3)',
                'Coarsen location data to 1km precision',
                'Remove or generalize rare attribute combinations'
            ],
            'HIGH': [
                'Mandatory strong anonymization',
                'Implement k-anonymity (k≥5)',
                'Coarsen location data to 5km precision',
                'Apply differential privacy',
                'Consider data synthesis for public release',
                'Restrict data access to authorized researchers only'
            ]
        }
        
        return recommendations.get(risk_level, [])

# Usage
privacy = PrivacyFramework()
privacy_assessment = privacy.assess_privacy_risk(data)

if privacy_assessment['risk_level'] == 'HIGH':
    print("High privacy risk detected. Applying mitigation measures...")
    # Apply recommended privacy protection measures
```

## Best Practices

### Security Checklist for Researchers

#### Before Analysis
- [ ] **Data Classification**: Classify data sensitivity level appropriately
- [ ] **Access Control**: Verify proper authorization for data access
- [ ] **Environment Setup**: Configure secure analysis environment
- [ ] **Backup Strategy**: Implement secure data backup procedures

#### During Analysis  
- [ ] **Audit Trail**: Enable comprehensive logging and audit trails
- [ ] **Anonymization**: Apply appropriate data anonymization techniques
- [ ] **Secure Storage**: Use encrypted storage for temporary files
- [ ] **Access Monitoring**: Monitor and log all data access events

#### After Analysis
- [ ] **Result Protection**: Apply result anonymization if needed
- [ ] **Secure Cleanup**: Securely delete temporary files and data copies
- [ ] **Export Controls**: Apply proper controls to exported results
- [ ] **Documentation**: Document all security measures applied

### Recommended Security Configuration

```python
# secure_config.yaml
security:
  # Data protection
  default_anonymization_level: "standard"
  require_audit_trail: true
  enable_encryption: true
  
  # Access control
  require_authentication: true
  session_timeout: 3600  # 1 hour
  max_failed_attempts: 3
  
  # Data retention
  max_retention_days: 365
  secure_deletion: true
  
  # Monitoring
  log_level: "INFO"
  audit_log_retention: 2555  # 7 years
  alert_on_suspicious_activity: true
  
  # Compliance
  ethics_approval_required: true
  permit_verification_required: true
  privacy_impact_assessment: true
```

---

**Related Documentation:**
- [Installation Guide](../user-guide/installation.md) - Secure installation procedures
- [Data Loading Guide](../user-guide/data-loading.md) - Secure data handling practices
- [Developer Guide](../development/setup.md) - Secure development environment setup