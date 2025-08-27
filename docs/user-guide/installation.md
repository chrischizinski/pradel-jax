# Pradel-JAX Installation Guide

Complete installation instructions for all environments and use cases.

## 🚀 Quick Installation (Recommended)

Choose the setup that matches your needs:

### Basic Setup (Most Users)
Perfect for getting started with standard analysis:

```bash
# Clone repository and auto-setup (creates virtual environment)
git clone https://github.com/chrischizinski/pradel-jax.git
cd pradel-jax
./quickstart.sh

# Activate environment
source pradel_env/bin/activate

# Verify installation
python -m pytest tests/integration/test_optimization_minimal.py -v
```

### Performance Demo Setup (Advanced Users)
Includes large-scale demonstrations and GPU benchmarking:

```bash
# Clone repository and setup with performance demos
git clone https://github.com/chrischizinski/pradel-jax.git
cd pradel-jax
./quickstart_parallel.sh

# Activate environment
source pradel_env/bin/activate

# Performance features are now demonstrated
```

**What's included in each quickstart:**

| Feature | `quickstart.sh` | `quickstart_parallel.sh` |
|---------|----------------|---------------------------|
| ✅ Basic installation | ✅ | ✅ |
| ✅ Core functionality demo | ✅ | ✅ |
| ✅ Integration tests | ✅ | ✅ |
| ⚡ Large-scale benchmarks | ❌ | ✅ |
| 🖥️ GPU acceleration demo | ❌ | ✅ |
| 📊 Performance monitoring | ❌ | ✅ |

## 📋 System Requirements

**Minimum Requirements:**
- Python 3.8+ (Python 3.9+ recommended)
- 4 GB RAM
- 2 GB free disk space

**Recommended Configuration:**
- Python 3.10+
- 8 GB+ RAM
- Multi-core CPU
- GPU (optional, for large datasets)

## Installation Options

### 1. Basic Installation (Recommended)
Installs core optimization framework with reliable SciPy optimizers:

```bash
pip install -r requirements.txt
```

**Included capabilities:**
- ✅ Intelligent strategy selection
- ✅ SciPy optimizers (L-BFGS-B, SLSQP, BFGS)
- ✅ JAX optimizers (Adam with Optax)
- ✅ Bayesian optimization (scikit-optimize)
- ✅ Hyperparameter tuning (Optuna)
- ✅ Experiment tracking (MLflow)
- ✅ Performance monitoring
- ✅ Full framework functionality

### 2. Development Installation (Advanced)
Includes enhanced features for development and research:

```bash
pip install -r requirements-dev.txt
```

**Additional capabilities:**
- 🔬 Advanced visualizations (Plotly, Bokeh)
- 📊 Alternative experiment tracking (Weights & Biases, TensorBoard)
- ⚡ Enhanced profiling and monitoring
- 🧪 Multi-objective optimization (PyMOO)
- 🔄 Evolutionary algorithms (DEAP, Nevergrad)
- 🏗️ Development tools (Black, MyPy, Sphinx)

### 3. Minimal Installation (Core Only)
For environments with strict dependency constraints:

```bash
pip install jax jaxlib optax scipy numpy pandas
```

**Limited capabilities:**
- ✅ Basic optimization with SciPy and JAX
- ✅ Strategy selection (simplified)
- ❌ No global optimization
- ❌ Limited monitoring
- ❌ No experiment tracking

## Platform-Specific Notes

### GPU Support (Optional)
For GPU-accelerated optimization:

```bash
# CUDA 11.x
pip install jax[cuda11_pip]

# CUDA 12.x  
pip install jax[cuda12_pip]

# Apple Silicon (M1/M2)
# JAX automatically uses Metal acceleration
pip install jax  # Metal support included by default
```

### Windows
Some optional packages may require additional setup:
```bash
# Install Microsoft C++ Build Tools if needed
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install -r requirements.txt
```

### macOS
```bash
# Ensure Xcode command line tools are installed
xcode-select --install

pip install -r requirements.txt
```

### Linux
```bash
# Most packages should install without issues
pip install -r requirements.txt

# For GPU support (optional)
pip install jax[cuda11_pip]  # or cuda12_pip
```

## Dependency Overview

### Core Optimization Dependencies

| Package | Purpose | Impact if Missing |
|---------|---------|-------------------|
| `jax` | Automatic differentiation | ❌ **Critical**: Core functionality broken |
| `optax` | Modern JAX optimizers | ⚠️ **Degraded**: Falls back to basic Adam |
| `scipy` | Proven optimization algorithms | ❌ **Critical**: Primary optimizers unavailable |
| `numpy` | Numerical computing | ❌ **Critical**: Framework won't function |

### Advanced Optimization Dependencies

| Package | Purpose | Impact if Missing |
|---------|---------|-------------------|
| `scikit-optimize` | Bayesian optimization | ⚠️ **Limited**: BayesianOptimizer disabled |
| `optuna` | Hyperparameter tuning | ⚠️ **Limited**: OptunaOptimizer disabled |
| `mlflow` | Experiment tracking | ⚠️ **Limited**: No MLflow integration |
| `psutil` | System monitoring | ⚠️ **Limited**: No resource monitoring |

### Utility Dependencies

| Package | Purpose | Impact if Missing |
|---------|---------|-------------------|
| `matplotlib` | Basic plotting | ⚠️ **Limited**: No built-in visualizations |
| `seaborn` | Statistical plots | ⚠️ **Limited**: Reduced plotting options |
| `pandas` | Data manipulation | ⚠️ **Limited**: Some data processing features disabled |

## Verification

Test your installation:

```bash
# Basic functionality test
python -c "
from pradel_jax.optimization import optimize_model, OptimizationStrategy
import numpy as np

print('Testing optimization framework...')

class TestContext:
    n_parameters = 2
    n_individuals = 100  
    n_occasions = 5
    capture_matrix = np.random.binomial(1, 0.3, (100, 5))
    def get_condition_estimate(self): return 1e5

def simple_objective(x): return np.sum((x - 1)**2)

response = optimize_model(
    objective_function=simple_objective,
    initial_parameters=np.array([0., 0.]),
    context=TestContext()
)

if response.success:
    print(f'✅ SUCCESS: Framework working correctly!')
    print(f'   Strategy: {response.strategy_used}')
    print(f'   Final error: {np.linalg.norm(response.result.x - 1.0):.6f}')
else:
    print('❌ FAILED: Framework not working properly')
"
```

Expected output:
```
Testing optimization framework...
✅ SUCCESS: Framework working correctly!
   Strategy: scipy_lbfgs
   Final error: 0.000000
```

## Troubleshooting

### Common Issues

1. **JAX installation problems**:
   ```bash
   # Try upgrading pip first
   pip install --upgrade pip
   pip install jax jaxlib
   ```

2. **Optax not available warning**:
   ```bash
   # Install optax explicitly
   pip install optax>=0.1.7
   ```

3. **SciPy compilation errors**:
   ```bash
   # Install from conda-forge (alternative)
   conda install -c conda-forge scipy
   ```

4. **Memory issues during installation**:
   ```bash
   # Install packages one at a time
   pip install --no-cache-dir -r requirements.txt
   ```

### Dependency Conflicts

If you encounter version conflicts:

```bash
# Create clean virtual environment
python -m venv pradel_env
source pradel_env/bin/activate  # Linux/macOS
# pradel_env\Scripts\activate  # Windows

# Install fresh dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Optional Features Not Working

The framework gracefully handles missing optional dependencies:

- **Missing Optax**: JAX optimizers use basic fallback implementation
- **Missing scikit-optimize**: BayesianOptimizer unavailable but framework works
- **Missing MLflow**: Experiment tracking disabled but optimization works
- **Missing Optuna**: OptunaOptimizer unavailable but other strategies work

## Environment Setup

### Recommended Virtual Environment Setup

```bash
# Create environment
python -m venv pradel_env
source pradel_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from pradel_jax.optimization import optimize_model; print('Ready!')"
```

### For Development

```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
isort .
```

## Performance Optimization

For best performance:

1. **Install with optimized BLAS**:
   ```bash
   pip install scipy  # Uses optimized BLAS if available
   ```

2. **GPU Support** (if available):
   ```bash
   pip install jax[cuda11_pip]  # NVIDIA GPUs
   ```

3. **Parallel Processing**:
   ```bash
   export JAX_ENABLE_X64=True  # Higher precision
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8  # GPU memory limit
   ```

The optimization framework is designed to work well with the default installation while providing enhanced features when additional dependencies are available.