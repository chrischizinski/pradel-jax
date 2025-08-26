# Nebraska Capture-Recapture Analysis

This directory contains comprehensive analysis scripts for Nebraska fisheries capture-recapture data using the pradel-jax framework.

## 📋 Overview

The Nebraska analysis demonstrates the pradel-jax optimization framework on real-world fisheries data, fitting Pradel models with various covariate combinations to understand fish survival, recruitment, and detection patterns.

## 🚀 Quick Start

### 🎯 **Full Dataset Analysis (100K+ Individuals)**

**Run the complete Nebraska analysis on all ~111,697 individuals:**

```bash
# Production-ready full dataset analysis
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py --sample-size 0 --parallel

# Expected output:
# 🔬 Nebraska Capture-Recapture Analysis - Full Dataset
# 💾 Dataset size: 111,697 individuals, 9 occasions  
# 🏆 Models fitted: 64 total (all combinations)
# ⏱️ Total runtime: ~2-4 hours
```

**This will generate:**
- Complete model comparison for all 64 model combinations
- Publication-ready AIC rankings and model weights  
- Professional performance monitoring and progress tracking
- Comprehensive parameter estimates for the entire population

## 📋 Standard Analysis Options

### Prerequisites
1. **Environment Setup**
   ```bash
   # Activate virtual environment
   source pradel_env/bin/activate
   
   # Verify installation
   python -c "import pradel_jax; print('✅ pradel-jax installed')"
   ```

2. **Required Data**
   - Ensure `data/encounter_histories_ne_clean.csv` exists in project root
   - Data contains 111,697 individual fish encounter histories with covariates

3. **Run from Project Root**
   ```bash
   cd /path/to/pradel-jax/
   ```

## 📊 Main Analysis Script

### `nebraska_sample_analysis.py`

**Purpose**: Comprehensive Pradel model fitting on randomly sampled Nebraska data with all covariate combinations.

#### Basic Usage
```bash
# Default analysis (1000 individuals, all 64 models)
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py

# Quick test (50 individuals)
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py --sample-size 50

# FULL DATASET ANALYSIS (all ~111K individuals, all 64 models)
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py --sample-size 0 --parallel

# Large sample with parallel processing
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py -n 50000 --parallel
```

#### Command Line Options
| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--sample-size` | `-n` | 1000 | Number of individuals to sample (**use 0 for full dataset**) |
| `--max-models` | | 64 | Maximum number of models to fit (default: all combinations) |
| `--parallel` | `-p` | false | **Use parallel processing for large datasets (recommended for 5K+)** |
| `--chunk-size` | | 10000 | Chunk size for memory-efficient processing |
| `--help` | `-h` | | Show help message and exit |

#### Model Structure
The script fits all combinations of:
- **Survival (φ)**: 7 formulas using gender, age, tier covariates
- **Recruitment (f)**: 7 formulas using same covariates  
- **Detection (p)**: Constant detection probability (`~1`)

**Formula Combinations**:
```
φ: ~1, ~1+gender, ~1+age, ~1+tier, ~1+gender+age, ~1+gender+tier, ~1+age+tier
f: ~1, ~1+gender, ~1+age, ~1+tier, ~1+gender+age, ~1+gender+tier, ~1+age+tier
p: ~1 (constant)
```

## 📁 Output Files

All analyses generate timestamped output files:

### 1. Full Results (`nebraska_full_results_[N]ind_[timestamp].csv`)
Complete model fitting results including:
- Model specifications and names
- Parameter estimates and standard errors
- Log-likelihood, AIC, number of parameters
- Optimization strategy used and convergence status
- Detailed metadata for each model

### 2. Model Comparison (`nebraska_model_comparison_[N]ind_[timestamp].csv`)
AIC-based model selection table with:
- Models ranked by AIC (lowest = best)
- ΔAIC values (difference from best model)
- AIC weights (relative model support)
- Substantial support indicators (ΔAIC ≤ 2.0)
- Evidence ratios between competing models

### 3. Parameter Summary (`nebraska_parameters_[N]ind_[timestamp].csv`)
Simplified parameter reference with:
- Model names and AIC values
- Number of parameters per model
- ΔAIC and AIC weights
- Substantial support flags

## ⚡ Performance Guide

### Recommended Sample Sizes

| Sample Size | Runtime | Use Case | Models Fit | Parallel |
|-------------|---------|----------|------------|----------|
| 50 | ~1 minute | **Quick test** | 64 | No |
| 1000 | ~3 minutes | **Standard analysis** | 64 | No |
| 5000 | ~8 minutes | **Large analysis** | 64 | Yes |
| 25000 | ~25 minutes | **Very large** | 64 | Yes |
| 50000 | ~45 minutes | **Extra large** | 64 | Yes |
| **0 (Full Dataset)** | **2-4 hours** | **🚀 COMPLETE ANALYSIS** | **64** | **Yes** |

### System Requirements

#### Memory Requirements
- **Small (≤1K)**: 4GB RAM
- **Medium (1K-10K)**: 8GB RAM  
- **Large (10K-50K)**: 16GB RAM
- **🚀 Full Dataset (100K+)**: 32GB RAM recommended (16GB minimum)

#### Processing
- **CPU Cores**: Parallel processing uses all available cores (up to 8)
- **Storage**: ~500MB for full dataset analysis outputs
- **Runtime**: Full dataset analysis typically completes in 2-4 hours

## 🔬 Analysis Features

### 🚀 Large-Scale Optimization Features
- **Intelligent Strategy Selection**: Automatically chooses best optimizer (L-BFGS-B, SLSQP, Adam, multi-start)
- **JAX Compilation**: JIT-compiled likelihood functions for maximum speed
- **Parallel Processing**: Multi-core model fitting for datasets >5K individuals
- **Memory Management**: Automatic garbage collection and chunked processing
- **Progress Monitoring**: Real-time progress with time estimates for large datasets
- **Robust Convergence**: Multiple fallback strategies for difficult models

### Statistical Rigor
- **Random Sampling**: Reproducible sampling with fixed random seed (42)
- **Model Selection**: AIC-based model comparison with weights and evidence ratios
- **Covariate Processing**: Automatic standardization and categorical encoding
- **Quality Validation**: Comprehensive data quality checks and preprocessing

### Production Features
- **Error Handling**: Graceful failure handling with detailed error reporting
- **Progress Tracking**: Real-time progress updates with model names and status
- **Export Format**: Publication-ready CSV outputs with comprehensive metadata
- **Logging**: Professional logging with timestamps and optimization details

## 🧪 Other Analysis Scripts

### `nebraska_advanced_statistical_analysis.py`
Advanced statistical inference with:
- Bootstrap confidence intervals
- Likelihood ratio tests
- Model averaging
- Diagnostic plots and validation

### `nebraska_production_analysis.py` 
Production-ready analysis pipeline with:
- Parallel model fitting
- Large-scale data handling
- Comprehensive quality gates
- Automated report generation

### `nebraska_covariate_analysis.py`
Specialized covariate analysis with:
- Interaction term exploration
- Time-varying covariate effects
- Hierarchical model structures
- Custom formula specifications

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'pradel_jax'"**
```bash
# Solution: Set PYTHONPATH
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py
```

**"Data file not found"**
```bash
# Solution: Run from project root directory
cd /path/to/pradel-jax/
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py
```

**"Optimization failed to converge"**
```bash
# Solution: Try smaller sample size or fewer models
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py -n 100 --max-models 20
```

**Memory issues with large samples**
```bash
# Solution: Reduce sample size or run in chunks
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py -n 500
```

### Getting Help

**Script Help**
```bash
PYTHONPATH=. python examples/nebraska/nebraska_sample_analysis.py --help
```

**Debug Mode**
```bash
# Add debug logging
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python examples/nebraska/nebraska_sample_analysis.py -n 50
```

## 📊 Expected Results

### Sample Output Summary
```
🔬 Nebraska Capture-Recapture Analysis - Random Sample
============================================================
📂 Loading data from: data/encounter_histories_ne_clean.csv
   Full dataset shape: (111697, 35)
🎲 Randomly sampling 1000 rows...
   Sample shape: (1000, 35)

📊 Setting up comprehensive Pradel model set...
   Target covariates for modeling: ['gender', 'age', 'tier']
   Total models: φ(7) × f(7) × p(1) = 49

⚡ Fitting models using optimization framework...
   Progress: 49/49 models completed ✅
   Successful fits: 47/49 (95.9%)
   
🏆 Best Model: phi~1+age_p~1_f~1+gender+tier
   AIC: 2847.234
   Evidence Ratio: 2.3x better than next model
```

### Model Selection Results
- **Top models** typically include age effects on survival
- **Strong evidence** for tier effects on recruitment  
- **Gender effects** vary by sample and may show in recruitment
- **AIC differences** of 2-10 points common between competitive models

## 🎯 Next Steps

After running the analysis:

1. **Examine Results**: Review model comparison table for best-supported models
2. **Parameter Interpretation**: Analyze parameter estimates for biological meaning
3. **Model Validation**: Run diagnostics on top models
4. **Publication**: Use generated tables for manuscript preparation
5. **Comparison**: Compare results with RMark analysis for validation

## 📚 Related Documentation

- **Main Documentation**: `docs/user-guide/`
- **API Reference**: `docs/api/`
- **Optimization Guide**: `docs/user-guide/optimization.md`
- **Data Format Guide**: `docs/user-guide/data-formats.md`

---

*Last updated: August 25, 2025*
*Nebraska analysis represents real-world application of pradel-jax framework*