# Secure Environment Setup Guide

**SECURITY FIRST**: Complete setup guide with zero secrets in repository

---

## 🔐 **Quick Security Setup**

### **Step 1: Create Secure Environment File**

```bash
# Create secure environment file (NEVER committed to git)
touch ~/.pradel_validation_env
chmod 600 ~/.pradel_validation_env  # Owner read/write only

# Add your specific configuration
cat << 'EOF' > ~/.pradel_validation_env
# RMark Validation Environment (SECURE - NOT IN GIT)

# SSH Configuration (Home Office)
export RMARK_SSH_HOST="YOUR_WINDOWS_IP"      # Replace with your IP
export RMARK_SSH_USER="YOUR_USERNAME"        # Replace with your username  
export RMARK_SSH_KEY_PATH="~/.ssh/id_rsa"    # Your SSH key path
export RMARK_R_PATH='"C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe"'  # Windows R path

# Security Settings
export RMARK_MAX_EXECUTION_TIME="1800"       # 30 minutes timeout
export RMARK_CLEANUP_FILES="true"            # Clean temporary files
export RMARK_AUDIT_LOGGING="true"           # Enable security logging

# Session Identification
export RMARK_SESSION_ID="$(date +%Y%m%d_%H%M%S)_$$"  # Unique session ID
EOF

# Load environment automatically
echo 'source ~/.pradel_validation_env' >> ~/.bashrc  # or ~/.zshrc
source ~/.pradel_validation_env
```

### **Step 2: Create Personal Configuration**

```bash
# Copy template and customize (excluded from git)
cd /path/to/pradel-jax
cp config/validation_config_template.yaml config/validation_config.yaml

# Verify it's protected by .gitignore
git check-ignore config/validation_config.yaml  # Should return the filename
```

### **Step 3: Security Verification**

```bash
# Verify no secrets in git
git status  # Should show no credential files
grep -r "192.168" . --exclude-dir=.git  # Should find no hardcoded IPs
grep -ri "your_username" . --exclude-dir=.git  # Should find no usernames

# Test environment loading
echo "SSH Host: $RMARK_SSH_HOST"  # Should show your configuration
echo "SSH User: $RMARK_SSH_USER"  # Should show your username
```

---

## 🏗️ **Environment-Specific Configuration**

### **Home Office Setup (SSH Access)**

If you have SSH access to your Windows machine:

```bash
# Set SSH-specific variables
export RMARK_ENVIRONMENT="home_office"
export RMARK_PRIMARY_METHOD="ssh"

# Test SSH connectivity
ssh -o ConnectTimeout=5 $RMARK_SSH_USER@$RMARK_SSH_HOST exit
if [ $? -eq 0 ]; then
    echo "✅ SSH connectivity verified"
else
    echo "❌ SSH connectivity failed - check network and credentials"
fi

# Test R availability on Windows machine
ssh $RMARK_SSH_USER@$RMARK_SSH_HOST "$RMARK_R_PATH --version"
```

### **Work Office Setup (Local R)**

If you need to use local R installation:

```bash
# Set local R variables
export RMARK_ENVIRONMENT="work_office" 
export RMARK_PRIMARY_METHOD="local_r"
export RMARK_LOCAL_R_PATH="Rscript"  # or full path to R

# Test local R
Rscript --version
if [ $? -eq 0 ]; then
    echo "✅ Local R found"
else
    echo "❌ R not found - install R or set correct path"
fi

# Install/test RMark
Rscript -e "
if (!require('RMark', quietly=TRUE)) {
    cat('Installing RMark...\n')
    install.packages('RMark', repos='https://cran.rstudio.com/')
}
library(RMark)
cat('✅ RMark available\n')
"
```

### **Limited Access Setup (Development Mode)**

For development without external dependencies:

```bash
# Set development mode
export RMARK_ENVIRONMENT="development"
export RMARK_PRIMARY_METHOD="mock"
export RMARK_USE_CACHED_RESULTS="true"

echo "⚡ Development mode - using mock validation"
```

---

## 🧪 **Testing Your Setup**

### **Connectivity Test Script**

```bash
#!/bin/bash
# test_validation_environment.sh

echo "🔍 Testing RMark Validation Environment..."

# Check environment variables
if [ -z "$RMARK_SSH_HOST" ]; then
    echo "❌ RMARK_SSH_HOST not set"
    exit 1
else
    echo "✅ Environment variables loaded"
fi

# Test SSH (if configured)
if [ "$RMARK_PRIMARY_METHOD" = "ssh" ]; then
    echo "Testing SSH connectivity..."
    timeout 10s ssh -o ConnectTimeout=5 $RMARK_SSH_USER@$RMARK_SSH_HOST exit
    if [ $? -eq 0 ]; then
        echo "✅ SSH connection successful"
        
        # Test R on remote machine
        echo "Testing remote R installation..."
        ssh $RMARK_SSH_USER@$RMARK_SSH_HOST "$RMARK_R_PATH --version" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Remote R accessible"
        else
            echo "⚠️ Remote R path may be incorrect"
        fi
    else
        echo "❌ SSH connection failed"
    fi
fi

# Test local R (if configured)
if [ "$RMARK_PRIMARY_METHOD" = "local_r" ]; then
    echo "Testing local R installation..."
    Rscript --version >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Local R found"
        
        # Test RMark
        Rscript -e "library(RMark); cat('✅ RMark available')" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "⚠️ RMark not installed - will auto-install when needed"
        fi
    else
        echo "❌ Local R not found"
    fi
fi

echo "🎉 Environment test completed!"
```

Make it executable and run:
```bash
chmod +x test_validation_environment.sh
./test_validation_environment.sh
```

---

## 📋 **Configuration Examples**

### **Minimal Working Configuration**

For `config/validation_config.yaml`:
```yaml
validation:
  preferred_environment: "auto"
  
  criteria:
    parameter_absolute_tolerance: 0.001
    parameter_relative_tolerance_pct: 5.0
    max_aic_difference: 2.0
    
  output:
    base_output_dir: "./validation_results"
    generate_html_report: true
    cleanup_temp_files: true
    
security:
  max_execution_time_minutes: 30
  cleanup_temp_files_on_success: true
  enable_audit_logging: true
```

### **Development Configuration**

For development/testing:
```yaml
validation:
  preferred_environment: "mock"
  
development:
  enable_mock_validation: true
  use_cached_results: true
  verbose_logging: true
  save_intermediate_files: true
```

---

## 🚨 **Security Best Practices**

### **Do's ✅**
- ✅ Store all credentials in environment variables
- ✅ Use process-specific temporary directories
- ✅ Enable audit logging for security events
- ✅ Regularly rotate SSH keys
- ✅ Verify .gitignore protection before commits

### **Don'ts ❌**
- ❌ Never hardcode IP addresses or usernames in code
- ❌ Never commit credential files to git
- ❌ Never use shared temporary directories
- ❌ Never store passwords (use SSH keys only)
- ❌ Never disable security logging in production

### **Regular Security Checks**

```bash
# Run before every commit
git status | grep -i "credential\|secret\|ssh\|config\.yaml"
if [ $? -eq 0 ]; then
    echo "⚠️ Potential credential files detected - review before commit"
else
    echo "✅ No credential files detected"
fi

# Check for hardcoded credentials in code
grep -r "192\.168\|chris@\|password\|secret" . --exclude-dir=.git --exclude-dir=logs
```

---

## 🔧 **Implementation Integration**

This secure setup integrates with the validation framework through:

1. **Environment Detection**: Automatically detects available methods
2. **Secure Configuration Loading**: Loads credentials from environment only
3. **Audit Logging**: Tracks all security-relevant events
4. **Automatic Cleanup**: Removes temporary files and processes
5. **Fallback Mechanisms**: Gracefully handles missing credentials

**Next Steps**: Once your environment is configured, you can proceed with implementing the validation framework components, knowing that all sensitive information is properly protected.

---

**🎯 Your secure environment is ready for RMark validation implementation!**