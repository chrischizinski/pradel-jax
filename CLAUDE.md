---
# Codacy Rules
Configuration for AI behavior when interacting with Codacy's MCP Server

## After ANY successful file edit
- YOU MUST run the `codacy_cli_analyze` tool for each edited file
- If any issues are found, propose and apply fixes immediately
- This is a critical requirement, not optional

## After ANY dependency installation
- IMMEDIATELY run `codacy_cli_analyze` with tool set to "trivy"
- Check for security vulnerabilities in new packages
- Stop all operations if vulnerabilities found
- Fix security issues before continuing

## Repository setup
- Use `git remote -v` to determine provider/organization/repository
- Provider mapping: GitHub = "gh", Bitbucket = "bb", GitLab = "gl"
- If repository not found in Codacy, offer to run `codacy_setup_repository`
---

# Pradel-JAX Project Guide

This file provides guidance to Claude Code when working with the Pradel-JAX repository.

## Current Project Status (August 2025)

### MAJOR MILESTONE COMPLETED: Optimization Framework Integration
The Pradel-JAX optimization framework is **fully integrated and production-ready**:
- Complete JAX-based Pradel model implementation
- Intelligent optimization strategy selection (L-BFGS-B, SLSQP, Adam, multi-start)
- Industry-standard performance monitoring and experiment tracking
- Comprehensive formula system with R-style syntax
- Robust error handling and validation framework
- Full integration test suite (all tests passing)
- Repository securely published on GitHub with data protection

## Repository Structure
```
pradel-jax/
├── docs/              # Documentation hub
├── tests/             # Test suite (integration, unit, benchmarks)
├── pradel_jax/        # Main package
│   ├── optimization/  # Optimization framework
│   ├── models/        # Pradel model implementation
│   ├── formulas/      # R-style formula system
│   └── data/          # Data handling
├── data/              # Safe datasets only
└── examples/          # Usage demonstrations
```

## Quick Setup
```bash
./quickstart.sh
source pradel_env/bin/activate
python -m pytest tests/integration/test_optimization_minimal.py -v
```

## Basic Model Fitting
```python
import pradel_jax as pj

data_context = pj.load_data("data/dipper_dataset.csv")
formula_spec = pj.create_formula_spec(
    phi="~1 + sex",
    p="~1 + sex",
    f="~1"
)
result = pj.fit_model(
    model=pj.PradelModel(),
    formula=formula_spec,
    data=data_context
)
```

## Running Tests
```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/unit/ -v               # Unit tests
python -m pytest tests/integration/ -v        # Integration tests
python -m pytest tests/ --cov=pradel_jax      # With coverage
```

## Security
- Sensitive NE/SD data protected via .gitignore
- Never commit files with person_id, customer_id, FUZZY identifiers
- Only dipper_dataset.csv and synthetic data allowed in repo

## Development Standards
- PEP 8 style, Black formatting
- Type hints for public APIs
- Google-style docstrings
- >90% test coverage target

## Troubleshooting
- **Optimization failed**: Try `strategy="multi_start"`
- **Formula error**: Check `data_context.covariates.keys()`
- **JAX warnings**: Ignore TPU warnings on CPU systems
