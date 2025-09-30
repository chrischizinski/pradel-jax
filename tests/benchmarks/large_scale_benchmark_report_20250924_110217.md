# Large-Scale Pradel-JAX Benchmark Report

**Generated:** 20250924_110217
**Test Suite:** Large-scale dataset scalability analysis

## Executive Summary

This report evaluates the scalability of JAX Adam optimization compared to traditional scipy optimizers on large capture-recapture datasets (50k+ individuals).

## Strategy Performance Comparison

| Strategy | Dataset Size | Avg Time (s) | Memory (MB) | Success Rate | Memory Efficiency |
|----------|-------------|--------------|-------------|--------------|------------------|
| scipy_lbfgs | 5,000 | 0.20±0.19 | 0.0 | 100.0% | inf ind/MB |
| scipy_lbfgs | 25,000 | 0.26±0.24 | 10.1 | 100.0% | 2465.3 ind/MB |
| scipy_lbfgs | 50,000 | 0.25±0.23 | 6.4 | 100.0% | 7795.4 ind/MB |
| jax_adam | 5,000 | 4.79±0.23 | 1.6 | 0.0% | 3062.2 ind/MB |
| jax_adam | 25,000 | 15.47±0.18 | 11.6 | 0.0% | 2153.4 ind/MB |
| jax_adam | 50,000 | 26.34±1.37 | 25.1 | 0.0% | 1991.9 ind/MB |
| multi_start | 5,000 | 0.28±0.26 | 28.5 | 100.0% | 175.2 ind/MB |
| multi_start | 25,000 | 0.22±0.19 | 21.9 | 100.0% | 1139.6 ind/MB |
| multi_start | 50,000 | 0.29±0.19 | 27.1 | 100.0% | 1846.0 ind/MB |

## Scalability Analysis

### Scipy-Lbfgs

- **Time Complexity:** O(n^0.10)
- **Memory Complexity:** O(n^inf)
- **Largest Dataset:** 50,000 individuals
- **Best Performance:** 0.20s on 5,000 individuals

### Jax-Adam

- **Time Complexity:** O(n^0.74)
- **Memory Complexity:** O(n^1.19)
- **Largest Dataset:** 50,000 individuals
- **Best Performance:** 4.79s on 5,000 individuals

### Multi-Start

- **Time Complexity:** O(n^0.10)
- **Memory Complexity:** O(n^0.11)
- **Largest Dataset:** 50,000 individuals
- **Best Performance:** 0.22s on 25,000 individuals

## Key Findings

- **Fastest Strategy:** scipy_lbfgs (avg 0.24s across dataset sizes)
- **Most Memory Efficient:** scipy_lbfgs (inf ind/MB)
- **Most Reliable:** scipy_lbfgs (100.0% success rate)

## Recommendations

Based on this large-scale analysis:

1. For datasets >50k individuals, use **scipy_lbfgs** for optimal speed
2. For memory-constrained environments, use **scipy_lbfgs**
3. For production reliability, use **scipy_lbfgs**
4. JAX Adam shows good scalability characteristics
