#!/usr/bin/env python3
"""
Comprehensive benchmark suite runner for Pradel-JAX optimization framework.
Executes all performance, memory, and convergence benchmarks with reporting.
"""

import subprocess
import sys
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse


class BenchmarkSuiteRunner:
    """Orchestrates comprehensive benchmark execution and reporting."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def run_benchmark_module(self, module_name: str, test_pattern: str = None) -> Dict[str, Any]:
        """Run a specific benchmark module and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {module_name} benchmarks...")
        print(f"{'='*60}")
        
        # Construct pytest command
        module_path = Path(__file__).parent / f"{module_name}.py"
        cmd = [
            sys.executable, "-m", "pytest", 
            str(module_path),
            "-v", "-s", "--tb=short"
        ]
        
        if test_pattern:
            cmd.extend(["-k", test_pattern])
        
        # Run benchmark
        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800  # 30 minute timeout
            )
            elapsed = time.perf_counter() - start_time
            
            return {
                'module': module_name,
                'success': result.returncode == 0,
                'duration': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start_time
            return {
                'module': module_name,
                'success': False,
                'duration': elapsed,
                'stdout': '',
                'stderr': 'Benchmark timed out after 30 minutes',
                'return_code': -1
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return {
                'module': module_name,
                'success': False,
                'duration': elapsed,
                'stdout': '',
                'stderr': str(e),
                'return_code': -2
            }
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run optimization performance benchmarks."""
        return self.run_benchmark_module("test_optimization_performance")
    
    def run_memory_benchmarks(self) -> Dict[str, Any]:
        """Run memory performance benchmarks."""
        return self.run_benchmark_module("test_memory_performance")
    
    def run_convergence_benchmarks(self) -> Dict[str, Any]:
        """Run convergence analysis benchmarks."""
        return self.run_benchmark_module("test_convergence_analysis")
    
    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive benchmark tests."""
        comprehensive_results = {}
        
        # Run comprehensive tests from each module
        modules_and_tests = [
            ("test_optimization_performance", "test_comprehensive_benchmark_suite"),
            ("test_memory_performance", "test_comprehensive_memory_benchmark"),
            ("test_convergence_analysis", "test_comprehensive_convergence_analysis")
        ]
        
        for module, test_pattern in modules_and_tests:
            result = self.run_benchmark_module(module, test_pattern)
            comprehensive_results[f"{module}_{test_pattern}"] = result
        
        return comprehensive_results
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        report_lines = [
            f"# Pradel-JAX Performance Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Timestamp: {self.timestamp}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Overall success rate
        total_modules = len(self.results)
        successful_modules = sum(1 for r in self.results.values() if r['success'])
        success_rate = successful_modules / total_modules if total_modules > 0 else 0
        
        report_lines.extend([
            f"- **Total benchmark modules:** {total_modules}",
            f"- **Successful modules:** {successful_modules}/{total_modules} ({success_rate:.1%})",
            f"- **Total execution time:** {sum(r['duration'] for r in self.results.values()):.1f}s",
            ""
        ])
        
        # Module-by-module results
        report_lines.extend([
            "## Benchmark Results by Module",
            ""
        ])
        
        for module_name, result in self.results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            report_lines.extend([
                f"### {module_name}",
                f"- **Status:** {status}",
                f"- **Duration:** {result['duration']:.1f}s",
                f"- **Return Code:** {result['return_code']}",
                ""
            ])
            
            if not result['success'] and result['stderr']:
                report_lines.extend([
                    "**Error Output:**",
                    "```",
                    result['stderr'][:1000] + ("..." if len(result['stderr']) > 1000 else ""),
                    "```",
                    ""
                ])
        
        # Performance highlights from stdout
        report_lines.extend([
            "## Performance Highlights",
            ""
        ])
        
        for module_name, result in self.results.items():
            if result['success'] and result['stdout']:
                # Extract key performance metrics from stdout
                stdout_lines = result['stdout'].split('\n')
                performance_lines = [
                    line for line in stdout_lines 
                    if any(keyword in line.lower() for keyword in 
                          ['strategy performance', 'memory usage', 'convergence', 'benchmark suite'])
                ]
                
                if performance_lines:
                    report_lines.extend([
                        f"### {module_name}",
                        "```"
                    ])
                    report_lines.extend(performance_lines[:10])  # Limit output
                    report_lines.extend([
                        "```",
                        ""
                    ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if success_rate == 1.0:
            report_lines.append("✅ All benchmarks passed successfully. The optimization framework is performing well.")
        elif success_rate >= 0.8:
            report_lines.append("⚠️ Most benchmarks passed. Review failed modules for potential issues.")
        else:
            report_lines.append("❌ Multiple benchmark failures detected. Investigation required.")
        
        report_lines.extend([
            "",
            "## File Outputs",
            "",
            "The following files were generated during benchmarking:",
            ""
        ])
        
        # List generated files
        benchmark_files = list(self.output_dir.glob(f"*{self.timestamp}*"))
        for file_path in sorted(benchmark_files):
            report_lines.append(f"- `{file_path.name}`")
        
        return "\n".join(report_lines)
    
    def save_results(self):
        """Save benchmark results and generate reports."""
        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate and save summary report
        summary_report = self.generate_summary_report()
        report_file = self.output_dir / f"benchmark_report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(summary_report)
        
        # Create summary CSV
        summary_data = []
        for module_name, result in self.results.items():
            summary_data.append({
                'module': module_name,
                'success': result['success'],
                'duration_seconds': result['duration'],
                'return_code': result['return_code'],
                'timestamp': self.timestamp
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / f"benchmark_summary_{self.timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUITE COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to:")
        print(f"  📊 {results_file}")
        print(f"  📝 {report_file}")
        print(f"  📈 {summary_file}")
        print(f"\nSummary:")
        print(f"  Total modules: {len(self.results)}")
        print(f"  Successful: {sum(1 for r in self.results.values() if r['success'])}")
        print(f"  Failed: {sum(1 for r in self.results.values() if not r['success'])}")
        print(f"  Total time: {sum(r['duration'] for r in self.results.values()):.1f}s")


def main():
    """Main benchmark suite execution."""
    parser = argparse.ArgumentParser(description="Run Pradel-JAX benchmark suite")
    parser.add_argument("--suite", choices=["quick", "comprehensive", "performance", "memory", "convergence"], 
                       default="quick", help="Benchmark suite to run")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per module in seconds")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkSuiteRunner(output_dir=args.output_dir)
    
    print(f"🚀 Starting Pradel-JAX Benchmark Suite")
    print(f"Suite: {args.suite}")
    print(f"Output: {runner.output_dir}")
    print(f"Timestamp: {runner.timestamp}")
    
    # Run selected benchmark suite
    if args.suite == "quick":
        print("\n📊 Running quick benchmark suite...")
        runner.results['performance_quick'] = runner.run_benchmark_module(
            "test_optimization_performance", "test_strategy_comparison_simple_model"
        )
        runner.results['memory_quick'] = runner.run_benchmark_module(
            "test_memory_performance", "test_memory_usage_by_strategy"
        )
        runner.results['convergence_quick'] = runner.run_benchmark_module(
            "test_convergence_analysis", "test_lbfgs_convergence_stability"
        )
        
    elif args.suite == "comprehensive":
        print("\n📊 Running comprehensive benchmark suite...")
        runner.results.update(runner.run_comprehensive_benchmarks())
        
    elif args.suite == "performance":
        print("\n📊 Running performance benchmarks...")
        runner.results['performance_full'] = runner.run_performance_benchmarks()
        
    elif args.suite == "memory":
        print("\n📊 Running memory benchmarks...")
        runner.results['memory_full'] = runner.run_memory_benchmarks()
        
    elif args.suite == "convergence":
        print("\n📊 Running convergence benchmarks...")
        runner.results['convergence_full'] = runner.run_convergence_benchmarks()
    
    # Save results and generate reports
    runner.save_results()
    
    # Exit with appropriate code
    failed_modules = sum(1 for r in runner.results.values() if not r['success'])
    sys.exit(failed_modules)


if __name__ == "__main__":
    main()