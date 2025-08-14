# Secure RMark Validation Setup

**SECURITY FIRST**: This guide ensures no sensitive information is committed to the repository.

---

## 🔐 **Security Architecture**

### **Principle: Zero Secrets in Repository**
- **NO** SSH credentials, hostnames, or usernames in git
- **NO** hardcoded connection details in code
- **ALL** sensitive config loaded from environment variables
- **COMPREHENSIVE** .gitignore protection for credentials

### **Protected Information**
- SSH hostnames, usernames, key paths
- R installation paths with personal info
- Temporary files that might contain sensitive data
- Validation cache with potentially identifying information

---

## ⚙️ **Secure Configuration Setup**

### **Step 1: Environment Variables (Recommended)**

Create a secure environment file (NEVER committed to git):

```bash
# Create secure environment file
touch ~/.pradel_validation_env
chmod 600 ~/.pradel_validation_env  # Read/write for owner only

# Edit with your secure editor
nano ~/.pradel_validation_env
```

Add your configuration:
```bash
# ~/.pradel_validation_env
# SSH Configuration (Home Office)
export RMARK_SSH_HOST="192.168.86.21"  # Your Windows machine IP
export RMARK_SSH_USER="chris"           # Your username
export RMARK_SSH_KEY_PATH="~/.ssh/id_rsa"  # SSH key location
export RMARK_R_PATH='"C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe"'

# Additional security
export RMARK_VALIDATION_TOKEN="$(uuidgen)"  # Unique session token
export RMARK_TEMP_DIR="/tmp/pradel_validation_$$"  # Process-specific temp
```

Load environment before validation:
```bash
# Add to your ~/.bashrc or ~/.zshrc for automatic loading
if [ -f ~/.pradel_validation_env ]; then
    source ~/.pradel_validation_env
fi

# Or load manually when needed
source ~/.pradel_validation_env
```

### **Step 2: Create Personal Config (Local Only)**

```bash
# Copy template and create personal config
cp config/validation_config_template.yaml config/validation_config.yaml

# This file is automatically excluded from git by .gitignore
```

### **Step 3: Verify Security**

```bash
# Verify your secrets are protected
git status  # Should NOT show any credential files

# Test git ignore patterns
git check-ignore config/validation_config.yaml  # Should return the filename
git check-ignore ~/.ssh/id_rsa  # Should return the filename (if in repo)

# Verify environment loading
echo $RMARK_SSH_HOST  # Should show your host (only after source command)
```

---

## 🏗️ **Secure Implementation Architecture**

### **Configuration Loading (Security-First)**

```python
# pradel_jax/validation/secure_config.py
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import yaml
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class SecureValidationConfig:
    """Security-focused validation configuration."""
    
    # SSH configuration (loaded from environment)
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None  
    ssh_key_path: Optional[str] = None
    ssh_r_path: Optional[str] = None
    ssh_port: int = 22
    
    # Local R configuration
    local_r_path: str = "Rscript"
    local_temp_dir: str = "/tmp/pradel_validation"
    
    # Security settings
    max_execution_time: int = 1800  # 30 minutes max
    cleanup_temp_files: bool = True
    enable_audit_logging: bool = True
    
    # Validation criteria
    parameter_tolerance: float = 0.001
    aic_tolerance: float = 2.0
    
    def __post_init__(self):
        """Load secure configuration from environment."""
        # Load SSH configuration from environment variables
        self.ssh_host = os.getenv("RMARK_SSH_HOST")
        self.ssh_user = os.getenv("RMARK_SSH_USER") 
        self.ssh_key_path = os.getenv("RMARK_SSH_KEY_PATH")
        self.ssh_r_path = os.getenv("RMARK_R_PATH")
        
        # Validate critical configuration
        if self.ssh_host and not self.ssh_user:
            raise SecurityError("SSH host specified but no username (set RMARK_SSH_USER)")
        
        # Ensure temp directory is process-specific for security
        if self.cleanup_temp_files:
            import uuid
            self.local_temp_dir = f"/tmp/pradel_validation_{uuid.uuid4().hex[:8]}"
        
        # Log configuration status (without exposing secrets)
        self._log_config_status()
    
    def _log_config_status(self):
        """Log configuration status without exposing secrets."""
        ssh_available = bool(self.ssh_host and self.ssh_user)
        logger.info(f"Validation config loaded: SSH={'enabled' if ssh_available else 'disabled'}")
        
    def has_ssh_config(self) -> bool:
        """Check if SSH configuration is available."""
        return all([self.ssh_host, self.ssh_user, self.ssh_key_path])
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get connection summary without exposing credentials."""
        return {
            "ssh_available": self.has_ssh_config(),
            "ssh_host_configured": bool(self.ssh_host),
            "ssh_user_configured": bool(self.ssh_user),
            "local_r_path": self.local_r_path,
            "security_enabled": True
        }

class SecureConfigLoader:
    """Secure configuration loader with multiple fallbacks."""
    
    def __init__(self):
        self.config_paths = [
            Path("config/validation_config.yaml"),
            Path("~/.config/pradel/validation.yaml").expanduser(),
            Path("config/validation_config_template.yaml")  # Fallback template
        ]
    
    def load_config(self) -> SecureValidationConfig:
        """Load configuration with security validation."""
        
        # Try loading from config files
        config_data = {}
        for config_path in self.config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        file_config = yaml.safe_load(f)
                    
                    # Merge configuration
                    if file_config:
                        config_data.update(file_config.get('validation', {}))
                    
                    logger.info(f"Loaded config from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Create secure config (automatically loads from environment)
        secure_config = SecureValidationConfig()
        
        # Apply file-based settings (non-sensitive only)
        if 'criteria' in config_data:
            criteria = config_data['criteria']
            secure_config.parameter_tolerance = criteria.get('parameter_absolute_tolerance', 0.001)
            secure_config.aic_tolerance = criteria.get('max_aic_difference', 2.0)
        
        if 'security' in config_data:
            security = config_data['security']
            secure_config.max_execution_time = security.get('max_execution_time_minutes', 30) * 60
            secure_config.cleanup_temp_files = security.get('cleanup_temp_files_on_success', True)
        
        # Validate security configuration
        self._validate_security(secure_config)
        
        return secure_config
    
    def _validate_security(self, config: SecureValidationConfig):
        """Validate security configuration."""
        
        # Check for potential security issues
        security_issues = []
        
        # Warn if using default/template paths that might be in git
        if config.ssh_key_path and 'template' in str(config.ssh_key_path).lower():
            security_issues.append("SSH key path appears to reference template file")
        
        # Warn if temp directory is not process-specific
        if config.local_temp_dir == "/tmp/pradel_validation":
            security_issues.append("Temp directory not process-specific (potential security risk)")
        
        # Log security issues
        for issue in security_issues:
            logger.warning(f"Security issue detected: {issue}")
        
        if security_issues:
            logger.warning("Consider reviewing security configuration")

# Singleton pattern for secure config
_secure_config: Optional[SecureValidationConfig] = None

def get_secure_config() -> SecureValidationConfig:
    """Get singleton secure configuration."""
    global _secure_config
    if _secure_config is None:
        loader = SecureConfigLoader()
        _secure_config = loader.load_config()
    return _secure_config

class SecurityError(Exception):
    """Security-related configuration error."""
    pass
```

### **Secure Execution Context**

```python
# pradel_jax/validation/secure_execution.py
import tempfile
import shutil
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Generator
import uuid

logger = logging.getLogger(__name__)

class SecureExecutionContext:
    """Secure execution context with automatic cleanup."""
    
    def __init__(self, config: SecureValidationConfig):
        self.config = config
        self.session_id = uuid.uuid4().hex[:8]
        self.temp_base = None
        self.audit_log = []
        
    @contextmanager
    def secure_temp_directory(self) -> Generator[Path, None, None]:
        """Create secure temporary directory with automatic cleanup."""
        
        try:
            # Create process-specific temp directory
            self.temp_base = Path(tempfile.mkdtemp(
                prefix=f"pradel_validation_{self.session_id}_"
            ))
            
            # Set restrictive permissions (owner only)
            self.temp_base.chmod(0o700)
            
            self.audit_log.append(f"Created secure temp directory: {self.temp_base}")
            logger.info(f"Secure temp directory: {self.temp_base}")
            
            yield self.temp_base
            
        finally:
            # Guaranteed cleanup
            if self.temp_base and self.temp_base.exists():
                shutil.rmtree(self.temp_base, ignore_errors=True)
                self.audit_log.append(f"Cleaned up temp directory: {self.temp_base}")
                logger.info(f"Cleaned up temp directory: {self.temp_base}")
    
    def log_security_event(self, event: str, details: Optional[str] = None):
        """Log security-relevant events."""
        log_entry = f"[{self.session_id}] {event}"
        if details:
            log_entry += f": {details}"
        
        self.audit_log.append(log_entry)
        logger.info(f"Security event: {log_entry}")
        
        # Write to audit log if enabled
        if self.config.enable_audit_logging:
            self._write_audit_log(log_entry)
    
    def _write_audit_log(self, entry: str):
        """Write to security audit log."""
        try:
            audit_file = Path("logs/validation_security_audit.log")
            audit_file.parent.mkdir(exist_ok=True)
            
            with open(audit_file, "a") as f:
                f.write(f"{entry}\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")
```

### **Usage Example (Secure)**

```python
# Usage in validation pipeline
def run_secure_validation():
    """Run validation with security-first approach."""
    
    # Load secure configuration (no secrets in code)
    config = get_secure_config()
    
    # Log connection status without exposing secrets
    connection_info = config.get_connection_summary()
    logger.info(f"Validation environment: {connection_info}")
    
    # Create secure execution context
    security_context = SecureExecutionContext(config)
    
    with security_context.secure_temp_directory() as temp_dir:
        # All validation work happens in secure temp directory
        security_context.log_security_event("Validation started")
        
        try:
            # Execute validation (implementation details...)
            results = execute_validation_pipeline(config, temp_dir)
            
            security_context.log_security_event("Validation completed successfully")
            return results
            
        except Exception as e:
            security_context.log_security_event(f"Validation failed: {str(e)}")
            raise
        
        finally:
            # Automatic cleanup handled by context manager
            security_context.log_security_event("Cleanup completed")
```

---

## ✅ **Security Checklist**

### **Before Implementation**
- [ ] Environment variables configured in `~/.pradel_validation_env`
- [ ] Personal config file created from template (`validation_config.yaml`)
- [ ] Verified .gitignore protects all credential files
- [ ] SSH keys secured with proper permissions (`chmod 600`)

### **During Development**  
- [ ] No hardcoded credentials in any code files
- [ ] All sensitive config loaded from environment variables
- [ ] Temporary files use process-specific paths
- [ ] Audit logging enabled for security events

### **Before Committing**
- [ ] Run `git status` - no credential files listed
- [ ] Run `git check-ignore config/validation_config.yaml` - should return filename
- [ ] Search code for any IP addresses: `grep -r "192.168" . --exclude-dir=.git`
- [ ] Search code for any usernames: `grep -ri "chris" . --exclude-dir=.git`

### **Production Deployment**
- [ ] Environment variables set on deployment system
- [ ] Audit logs monitored for security events
- [ ] Temporary directories cleaned up after execution
- [ ] SSH keys rotated regularly

---

## 🚨 **Emergency Security Procedures**

### **If Credentials Are Accidentally Committed**
1. **Immediately rotate all affected credentials**
2. **Remove from git history**: `git filter-branch` or BFG Repo-Cleaner
3. **Force push to remote**: `git push --force-with-lease`
4. **Notify all team members** to pull clean repository
5. **Update .gitignore** to prevent recurrence

### **Security Incident Response**
1. **Document the incident** in security audit log
2. **Assess scope of exposure** (what credentials, how long)
3. **Rotate all potentially compromised credentials**
4. **Review and update security procedures**
5. **Test new security measures**

---

This security-first approach ensures that your SSH credentials and connection details are never at risk of being committed to the repository, while still enabling full validation functionality across your different office environments.

**Key Security Features:**
- ✅ **Zero secrets in git** - all credentials via environment variables
- ✅ **Comprehensive .gitignore** protection for all credential types  
- ✅ **Secure temp directories** with process-specific paths and cleanup
- ✅ **Audit logging** for security event tracking
- ✅ **Configuration validation** with security checks
- ✅ **Emergency procedures** for credential compromise