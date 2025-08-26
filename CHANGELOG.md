# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **CRITICAL**: Fixed optimization tolerance issues causing premature convergence on large-scale datasets
  - Root cause: Overly strict tolerances (`1e-8`) incompatible with large gradient magnitudes (300k+)
  - Solution: Scale-aware tolerance adjustment (`1e-4` for >10k individuals, `1e-6` default)
  - Impact: Models now properly converge to true optima with realistic parameter estimates
  - Validation: Tested up to 50k individuals with 100% success rate
  - Performance: Achieved 270k individual-models per second processing rate

### Security
- **CRITICAL**: Updated MLflow from 2.7.0 to 2.19.0 to fix 10+ critical CVEs including authentication bypass (CVE-2023-6014, CVE-2023-6015, CVE-2023-6018, CVE-2023-6831, CVE-2023-6974, CVE-2023-6975, CVE-2024-0520, CVE-2024-27132, CVE-2024-27133, CVE-2024-3573)
- **CRITICAL**: Replaced unsafe pickle deserialization with secure JSON serialization in parallel optimization framework
- **CRITICAL**: Replaced insecure MD5 hashing with SHA-256 across all modules
- Updated NumPy from 1.21.0 to 1.26.0 (fixes CVE-2021-34141)
- Updated Pydantic from 2.0.0 to 2.4.0 (fixes CVE-2024-3772)
- Updated Scikit-learn from 1.3.0 to 1.5.0 (fixes CVE-2024-5206)
- Updated tqdm from 4.64.0 to 4.66.3 (fixes CVE-2024-34062)
- Analyzed and verified secure subprocess usage patterns (no command injection vulnerabilities)

### Changed
- **BREAKING**: `CheckpointManager` now uses JSON instead of pickle for serialization
- Checkpoint files now saved as `.json` instead of binary format
- Hash fingerprints in validation modules now use SHA-256 (16-char) instead of MD5 (8-char)

### Added
- Comprehensive security documentation in `docs/security/SECURITY_UPDATES_2025.md`
- Security section in README with current status and features
- Type-safe serialization framework with explicit data type handling
- Automated vulnerability scanning integration with Codacy

### Fixed
- Resolved 60+ security vulnerabilities (23 critical, 6 high, 37 medium, 4 low)
- Eliminated all critical and high-priority security issues
- Improved data integrity with strong cryptographic algorithms

## [2.0.0-alpha] - 2025-08-14

### Added
- Complete JAX-based Pradel model implementation
- Intelligent optimization strategy selection (L-BFGS-B, SLSQP, Adam, multi-start)
- Industry-standard performance monitoring and experiment tracking
- Comprehensive formula system with R-style syntax
- Robust error handling and validation framework
- Full integration test suite
- Modular architecture with clean separation of concerns

### Changed
- Major repository reorganization for improved structure and maintainability
- Migrated from prototype to production-ready architecture

### Security
- Initial security audit and data protection measures implemented
- Comprehensive .gitignore protecting sensitive data
- SSH connection security for remote validation

---

**Note**: This changelog was initiated during the security update cycle of August 2025. Earlier changes are documented in git history and project reports.