# Pradel-JAX Performance Benchmark Report
Generated: 2025-08-15 08:48:06
Timestamp: 20250815_084720

## Executive Summary

- **Total benchmark modules:** 3
- **Successful modules:** 2/3 (66.7%)
- **Total execution time:** 45.6s

## Benchmark Results by Module

### performance_quick
- **Status:** ✅ PASSED
- **Duration:** 33.9s
- **Return Code:** 0

### memory_quick
- **Status:** ✅ PASSED
- **Duration:** 6.6s
- **Return Code:** 0

### convergence_quick
- **Status:** ❌ FAILED
- **Duration:** 5.1s
- **Return Code:** 1

**Error Output:**
```
/Users/cchizinski2/.pyenv/versions/3.11.7/lib/python3.11/site-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))

```

## Performance Highlights

### performance_quick
```
=== Simple Model Strategy Performance ===
```

### memory_quick
```
=== Memory Usage by Strategy ===
```

## Recommendations

❌ Multiple benchmark failures detected. Investigation required.

## File Outputs

The following files were generated during benchmarking:

- `benchmark_results_20250815_084720.json`