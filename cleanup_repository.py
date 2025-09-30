#!/usr/bin/env python3
"""
Repository Cleanup Script for Pradel-JAX
========================================
Moves temporary/debug files to archive/ directory before GitHub push.
Creates detailed report of all changes.

Usage:
    python cleanup_repository.py          # Dry run (shows what would happen)
    python cleanup_repository.py --execute  # Actually perform cleanup
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json


class RepositoryCleanup:
    """Manages cleanup of temporary files in Pradel-JAX repository."""

    def __init__(self, repo_root="."):
        self.repo_root = Path(repo_root)
        self.archive_dir = self.repo_root / "archive" / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "root_files_moved": [],
            "root_files_kept": [],
            "subdirs_moved": [],
            "subdirs_kept": [],
            "data_files_removed": [],
            "test_artifacts_removed": [],
            "total_size_freed_mb": 0
        }

    def get_file_size_mb(self, path):
        """Get size of file or directory in MB."""
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total / (1024 * 1024)
        return 0

    def should_keep_root_file(self, filename):
        """Determine if a root-level file should be kept."""
        # Essential project files
        essential_files = {
            'README.md', 'CHANGELOG.md', 'LICENSE', 'LICENSE.md',
            'setup.py', 'setup.cfg', 'pyproject.toml',
            'requirements.txt', 'requirements-dev.txt',
            '.gitignore', '.gitattributes',
            'conftest.py',  # Pytest root configuration
            'pytest.ini',
            'MANIFEST.in',
            '.python-version',
            'Makefile'
        }

        # Files that should stay (even if modified)
        keep_patterns = [
            'requirements',  # Any requirements file
            'setup',         # Setup files
        ]

        if filename in essential_files:
            return True

        for pattern in keep_patterns:
            if pattern in filename.lower():
                return True

        return False

    def should_move_root_file(self, filename):
        """Determine if a root-level file should be moved to archive."""
        # Patterns for temporary/debug/analysis files
        move_patterns = [
            'debug_', 'investigate_', 'test_', 'fixed_', 'verify_',
            'comprehensive_', 'alternative_', 'compare_', 'final_',
            'calculate_', 'reconcile_', 'examine_', 'data_pattern',
            'data_interpretation', 'detailed_parameter', 'convergence_',
            'confidence_interval', 'lambda_', 'phi_f_', 'reproduce_',
            'quick_', 'simple_', 'no_checkpoint', 'opencr_', 'pradel_',
            'nebraska_', 'south_dakota_', 'run_constant', 'run_full',
            'direct_alternative', 'deep_results', 'robust_boundary',
            'corrected_boundary'
        ]

        # Analysis/investigation markdown files
        move_md_patterns = [
            'ALTERNATIVE_', 'COMPLETE_', 'CORRECTED_', 'FINAL_',
            'GRADIENT_', 'NE_VS_', 'SD_CONFIDENCE', 'USING_SECOND',
            'WHY_WIDE', 'AGENTS.md'
        ]

        # Check Python files
        if filename.endswith('.py'):
            for pattern in move_patterns:
                if filename.startswith(pattern):
                    return True

        # Check markdown files
        if filename.endswith('.md'):
            for pattern in move_md_patterns:
                if filename.startswith(pattern):
                    return True

        # All CSV/JSON result files with timestamps or state prefixes
        if filename.endswith(('.csv', '.json')):
            if any(x in filename for x in ['_1759', '_2025', 'ne_', 'sd_', 'nebraska', 'south_dakota', 'subset_', 'jax_adam']):
                return True

        # Patch files
        if filename.endswith('.patch'):
            return True

        return False

    def should_move_directory(self, dirname):
        """Determine if a directory should be moved to archive."""
        move_dirs = {
            '.mplconfig',
            'checkpoints',
            'test_checkpoints',
            'optimization_experiments',
            'boundary_analysis_results',
            'comprehensive_model_results',
            'corrected_boundary_results',
            'robust_boundary_results',
        }
        return dirname in move_dirs

    def should_keep_directory(self, dirname):
        """Directories that must be kept."""
        keep_dirs = {
            'pradel_jax',     # Main package
            'tests',          # Test suite
            'docs',           # Documentation
            'examples',       # Example scripts
            'data',           # Data directory (will clean inside)
            'scripts',        # Utility scripts (will clean inside)
            'config',         # Configuration
            'logs',           # Logs directory (will clean inside)
            'archive',        # Archive directory
            '.git',           # Git directory
            '.github',        # GitHub workflows
            '.venv',          # Virtual environment
            'pradel_env',     # Virtual environment
            '.pytest_cache',  # Pytest cache
            '__pycache__',    # Python cache
            'pradel_jax.egg-info',  # Package info
            '.codacy',        # Codacy config
            '.claude',        # Claude config
            '.benchmarks',    # Pytest benchmarks
        }
        return dirname in keep_dirs

    def clean_data_directory(self, dry_run=True):
        """Remove sensitive/large data files, keep safe ones."""
        data_dir = self.repo_root / "data"
        if not data_dir.exists():
            return

        # Files to remove (sensitive state data)
        remove_patterns = [
            '20250903_sd_hip_tier_data.csv',
            '20250904_ne_hip_tier_data.csv',
            '.DS_Store'
        ]

        for pattern in remove_patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                size_mb = self.get_file_size_mb(file_path)
                self.report['data_files_removed'].append({
                    'file': str(file_path.relative_to(self.repo_root)),
                    'size_mb': round(size_mb, 2)
                })
                self.report['total_size_freed_mb'] += size_mb

                if not dry_run:
                    archive_path = self.archive_dir / "data" / pattern
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(archive_path))
                    print(f"  Moved: {file_path.relative_to(self.repo_root)} ({size_mb:.1f} MB)")

    def clean_test_artifacts(self, dry_run=True):
        """Remove test result artifacts, keep test code."""
        patterns = [
            'tests/integration/*results*.json',
            'tests/integration/*test_*.json',
            'tests/benchmarks/*.json',
            'tests/benchmarks/*.csv',
            'tests/benchmarks/results/',
        ]

        for pattern in patterns:
            for file_path in self.repo_root.glob(pattern):
                if file_path.is_file():
                    size_mb = self.get_file_size_mb(file_path)
                    self.report['test_artifacts_removed'].append({
                        'file': str(file_path.relative_to(self.repo_root)),
                        'size_mb': round(size_mb, 2)
                    })
                    self.report['total_size_freed_mb'] += size_mb

                    if not dry_run:
                        archive_path = self.archive_dir / file_path.relative_to(self.repo_root)
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file_path), str(archive_path))
                elif file_path.is_dir():
                    size_mb = self.get_file_size_mb(file_path)
                    self.report['test_artifacts_removed'].append({
                        'directory': str(file_path.relative_to(self.repo_root)),
                        'size_mb': round(size_mb, 2)
                    })
                    self.report['total_size_freed_mb'] += size_mb

                    if not dry_run:
                        archive_path = self.archive_dir / file_path.relative_to(self.repo_root)
                        shutil.move(str(file_path), str(archive_path))

    def clean_root_directory(self, dry_run=True):
        """Clean root-level files."""
        for item in self.repo_root.iterdir():
            if item.name.startswith('.'):
                continue

            if item.is_file():
                if self.should_keep_root_file(item.name):
                    self.report['root_files_kept'].append(item.name)
                elif self.should_move_root_file(item.name):
                    size_mb = self.get_file_size_mb(item)
                    self.report['root_files_moved'].append({
                        'file': item.name,
                        'size_mb': round(size_mb, 2)
                    })
                    self.report['total_size_freed_mb'] += size_mb

                    if not dry_run:
                        archive_path = self.archive_dir / "root" / item.name
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(item), str(archive_path))
                        print(f"  Moved: {item.name}")
                else:
                    # Unknown file - keep but report
                    self.report['root_files_kept'].append(f"{item.name} (unknown)")

            elif item.is_dir():
                if self.should_keep_directory(item.name):
                    self.report['subdirs_kept'].append(item.name)
                elif self.should_move_directory(item.name):
                    size_mb = self.get_file_size_mb(item)
                    self.report['subdirs_moved'].append({
                        'directory': item.name,
                        'size_mb': round(size_mb, 2)
                    })
                    self.report['total_size_freed_mb'] += size_mb

                    if not dry_run:
                        archive_path = self.archive_dir / "directories" / item.name
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(item), str(archive_path))
                        print(f"  Moved directory: {item.name} ({size_mb:.1f} MB)")
                else:
                    self.report['subdirs_kept'].append(f"{item.name} (unknown)")

    def generate_report(self):
        """Generate human-readable cleanup report."""
        report_lines = [
            "=" * 80,
            "PRADEL-JAX REPOSITORY CLEANUP REPORT",
            "=" * 80,
            f"Timestamp: {self.report['timestamp']}",
            f"Archive Location: {self.archive_dir}",
            "",
            "=" * 80,
            "ROOT FILES",
            "=" * 80,
            "",
            f"Files Moved: {len(self.report['root_files_moved'])}",
        ]

        if self.report['root_files_moved']:
            for item in sorted(self.report['root_files_moved'], key=lambda x: x['size_mb'], reverse=True)[:20]:
                report_lines.append(f"  - {item['file']} ({item['size_mb']:.2f} MB)")
            if len(self.report['root_files_moved']) > 20:
                report_lines.append(f"  ... and {len(self.report['root_files_moved']) - 20} more")

        report_lines.extend([
            "",
            f"Files Kept: {len(self.report['root_files_kept'])}",
            f"  Essential: {', '.join(sorted([f for f in self.report['root_files_kept'] if '(unknown)' not in f])[:10])}",
            "",
            "=" * 80,
            "DIRECTORIES",
            "=" * 80,
            "",
            f"Directories Moved: {len(self.report['subdirs_moved'])}",
        ])

        if self.report['subdirs_moved']:
            for item in self.report['subdirs_moved']:
                report_lines.append(f"  - {item['directory']:<40} {item['size_mb']:>8.1f} MB")

        report_lines.extend([
            "",
            f"Directories Kept: {len(self.report['subdirs_kept'])}",
            f"  {', '.join(sorted([d for d in self.report['subdirs_kept'] if '(unknown)' not in d]))}",
            "",
            "=" * 80,
            "DATA FILES",
            "=" * 80,
            "",
            f"Data Files Removed: {len(self.report['data_files_removed'])}",
        ])

        if self.report['data_files_removed']:
            for item in self.report['data_files_removed']:
                report_lines.append(f"  - {item['file']:<50} {item['size_mb']:>8.2f} MB")

        report_lines.extend([
            "",
            "=" * 80,
            "TEST ARTIFACTS",
            "=" * 80,
            "",
            f"Test Artifacts Removed: {len(self.report['test_artifacts_removed'])}",
        ])

        if self.report['test_artifacts_removed']:
            for item in self.report['test_artifacts_removed'][:10]:
                if 'file' in item:
                    report_lines.append(f"  - {item['file']}")
                else:
                    report_lines.append(f"  - {item['directory']} (directory)")
            if len(self.report['test_artifacts_removed']) > 10:
                report_lines.append(f"  ... and {len(self.report['test_artifacts_removed']) - 10} more")

        report_lines.extend([
            "",
            "=" * 80,
            "SUMMARY",
            "=" * 80,
            "",
            f"Total Files/Directories Moved: {len(self.report['root_files_moved']) + len(self.report['subdirs_moved']) + len(self.report['data_files_removed']) + len(self.report['test_artifacts_removed'])}",
            f"Total Space Freed: {self.report['total_size_freed_mb']:.1f} MB ({self.report['total_size_freed_mb']/1024:.2f} GB)",
            "",
            "All moved items are preserved in:",
            f"  {self.archive_dir}",
            "",
            "=" * 80,
        ])

        return "\n".join(report_lines)

    def run(self, dry_run=True):
        """Execute cleanup process."""
        print("\n" + "=" * 80)
        if dry_run:
            print("DRY RUN MODE - No files will be moved")
            print("Run with --execute flag to perform actual cleanup")
        else:
            print("EXECUTING CLEANUP")
            self.archive_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80 + "\n")

        print("Cleaning root directory...")
        self.clean_root_directory(dry_run)

        print("\nCleaning data directory...")
        self.clean_data_directory(dry_run)

        print("\nCleaning test artifacts...")
        self.clean_test_artifacts(dry_run)

        # Generate and save report
        report_text = self.generate_report()
        print("\n" + report_text)

        if not dry_run:
            report_file = self.repo_root / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_file.write_text(report_text)

            json_file = self.archive_dir / "cleanup_report.json"
            json_file.write_text(json.dumps(self.report, indent=2))

            print(f"\nReports saved:")
            print(f"  - {report_file}")
            print(f"  - {json_file}")


if __name__ == "__main__":
    import sys

    dry_run = "--execute" not in sys.argv

    cleanup = RepositoryCleanup()
    cleanup.run(dry_run=dry_run)

    if dry_run:
        print("\n" + "=" * 80)
        print("To execute cleanup, run:")
        print("  python cleanup_repository.py --execute")
        print("=" * 80)