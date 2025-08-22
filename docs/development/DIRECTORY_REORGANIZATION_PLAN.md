# Directory Reorganization Plan

## 🎯 Target Structure

```
pradel-jax/
├── 📦 pradel_jax/              # Main package (already organized)
├── 📚 docs/                    # All documentation (mostly organized)
├── 🧪 tests/                   # All test files (mostly organized)
├── 📊 outputs/                 # All result files and reports
├── 🔧 scripts/                 # Utility and development scripts  
├── 💾 data/                    # Input datasets (already clean)
├── 📁 archive/                 # Historical files (already exists)
├── ⚙️  config/                 # Configuration files
└── 📋 Root files               # Only essential project files

```

## 🚨 Files That Need Moving

### Root Directory Cleanup (move to docs/reports/)
```
ADVANCED_STATISTICAL_FEATURES_SUMMARY.md           → docs/reports/
BALANCED_IMPROVEMENT_STRATEGY.md                   → docs/reports/
COMPREHENSIVE_REMEDIATION_PLAN.md                  → docs/reports/
CRITICAL_ARCHITECTURAL_PLAN.md                     → docs/reports/  
ESTABLISHED_TOOLS_STRATEGY.md                      → docs/reports/
IMMEDIATE_ACTIONS_CHECKLIST.md                     → docs/reports/
PARALLEL_OPTIMIZATION_FIX_SUMMARY.md               → docs/reports/
PRADEL_LIKELIHOOD_CORRECTION_REPORT.md             → docs/reports/
VALIDATION_REPORT_FOR_SUBMISSION.md                → docs/reports/
```

### Root Scripts (move to scripts/)
```
analyze_mathematical_formulation.py                → scripts/analysis/
check_likelihood_differentiation.py                → scripts/analysis/
comprehensive_validation_audit.py                  → scripts/validation/
debug_direct_optimization.py                       → scripts/debugging/
debug_f_parameter.py                                → scripts/debugging/
debug_f_parameter_issue.py                         → scripts/debugging/
debug_gradient_computation.py                      → scripts/debugging/
debug_likelihood_step_by_step.py                   → scripts/debugging/
debug_optimization_failure.py                      → scripts/debugging/
fix_gradient_computation.py                        → scripts/fixes/
fix_optimization_bounds.py                         → scripts/fixes/
fix_pradel_likelihood.py                           → scripts/fixes/
mathematical_validation_audit.py                   → scripts/analysis/
minimal_working_test.py                            → scripts/validation/
```

### Root Test Files (move to tests/)
```
test_*.py files                                     → tests/unit/ or tests/integration/
*_test.py files                                     → tests/unit/ or tests/integration/
```

### Benchmark and Performance Files (move to outputs/)
```
gpu_acceleration_benchmark_report_*.md              → outputs/benchmarks/
large_scale_scalability_report_*.md                → outputs/benchmarks/
*_benchmark_results_*.csv                          → outputs/benchmarks/
*_benchmark_results_*.json                         → outputs/benchmarks/
jax_adam_scalability_test_*.csv                    → outputs/benchmarks/
production_validation_results/                     → outputs/validation/
```

### Nebraska and Data Processing Scripts (reorganize)
```
nebraska_*.py                                       → scripts/analysis/nebraska/
*validation*.py                                     → scripts/validation/
```

### Example Scripts (move to examples/)
```
gpu_acceleration_benchmark.py                      → examples/benchmarks/
large_scale_scalability_demonstration.py          → examples/benchmarks/
production_readiness_validation.py                → examples/validation/
```

## 📁 New Directory Structure

### outputs/ (NEW)
```
outputs/
├── benchmarks/           # Performance benchmark results
├── validation/          # Validation test results  
├── analysis/           # Analysis outputs
├── models/            # Model fitting results
└── reports/          # Generated reports
```

### scripts/ (ENHANCED)
```scripts/
├── analysis/           # Data analysis scripts
│   ├── nebraska/      # Nebraska-specific analysis
│   └── mathematical/  # Mathematical validation
├── debugging/         # Debug and diagnostic scripts
├── validation/        # Validation test scripts
├── fixes/            # Bug fix scripts
├── benchmarks/       # Performance testing
└── utilities/        # General utilities
```

## ⚠️ Files to Keep in Root
```
README.md              # Main project documentation
CLAUDE.md             # Development guidance
pyproject.toml        # Package configuration  
requirements*.txt     # Dependencies
pytest.ini           # Test configuration
quickstart.sh        # Setup script
.gitignore           # Version control
```

## 🚨 Critical Actions
1. Create new directory structure
2. Move files systematically
3. Update import paths where necessary
4. Update documentation references
5. Test that everything still works
6. Commit reorganization

This reorganization will make the project much more navigable and professional.