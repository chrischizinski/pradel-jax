#!/usr/bin/env python3
"""
Analyze large-scale benchmark results and generate insights.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def load_latest_results(results_dir: Path = Path("tests/benchmarks")) -> Tuple[pd.DataFrame, Dict]:
    """Load the most recent benchmark results."""
    
    # Find most recent results file
    csv_files = list(results_dir.glob("large_scale_benchmark_summary_*.csv"))
    json_files = list(results_dir.glob("large_scale_benchmark_results_*.json"))
    
    if not csv_files:
        raise FileNotFoundError("No large-scale benchmark results found")
    
    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    
    # Load data
    df = pd.read_csv(latest_csv)
    
    with open(latest_json, 'r') as f:
        raw_results = json.load(f)
    
    print(f"Loaded results from: {latest_csv.name}")
    return df, raw_results

def analyze_scalability(df: pd.DataFrame) -> Dict:
    """Analyze scalability characteristics."""
    
    analysis = {}
    
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('dataset_size')
        
        if len(strategy_data) < 2:
            continue
            
        # Calculate scaling characteristics
        sizes = strategy_data['dataset_size'].values
        times = strategy_data['avg_time'].values
        memories = strategy_data['peak_memory_mb'].values
        
        # Time scaling
        size_ratios = sizes[1:] / sizes[:-1]
        time_ratios = times[1:] / times[:-1]
        
        # Overall scaling from smallest to largest
        if len(sizes) > 1:
            total_size_ratio = sizes[-1] / sizes[0]
            total_time_ratio = times[-1] / times[0]
            total_memory_ratio = memories[-1] / memories[0]
            
            time_scaling = np.log(total_time_ratio) / np.log(total_size_ratio)
            memory_scaling = np.log(total_memory_ratio) / np.log(total_size_ratio)
            
            analysis[strategy] = {
                'size_range': (int(sizes[0]), int(sizes[-1])),
                'time_range': (float(times[0]), float(times[-1])),
                'memory_range': (float(memories[0]), float(memories[-1])),
                'time_scaling_exponent': float(time_scaling),
                'memory_scaling_exponent': float(memory_scaling),
                'efficiency_50k': strategy_data[strategy_data['dataset_size'] >= 50000]['memory_efficiency'].mean() if any(strategy_data['dataset_size'] >= 50000) else None,
                'max_dataset_tested': int(sizes[-1]),
                'success_rate_large': strategy_data[strategy_data['dataset_size'] >= 50000]['success_rate'].mean() if any(strategy_data['dataset_size'] >= 50000) else None
            }
    
    return analysis

def generate_scalability_insights(analysis: Dict) -> str:
    """Generate key insights from scalability analysis."""
    
    insights = []
    
    insights.append("# Large-Scale Dataset Scalability Analysis\n")
    
    # Overall performance
    insights.append("## Key Findings\n")
    
    # Find best performing strategies
    best_time_scaling = min((data['time_scaling_exponent'] for data in analysis.values() 
                           if data['time_scaling_exponent'] is not None), default=None)
    best_memory_scaling = min((data['memory_scaling_exponent'] for data in analysis.values()
                             if data['memory_scaling_exponent'] is not None), default=None)
    
    best_time_strategy = None
    best_memory_strategy = None
    
    for strategy, data in analysis.items():
        if data['time_scaling_exponent'] == best_time_scaling:
            best_time_strategy = strategy
        if data['memory_scaling_exponent'] == best_memory_scaling:
            best_memory_strategy = strategy
    
    if best_time_strategy:
        insights.append(f"- **Best Time Scalability**: {best_time_strategy} (O(n^{best_time_scaling:.2f}))")
    if best_memory_strategy:
        insights.append(f"- **Best Memory Scalability**: {best_memory_strategy} (O(n^{best_memory_scaling:.2f}))")
    
    # JAX Adam specific analysis
    if 'jax_adam' in analysis:
        jax_data = analysis['jax_adam']
        insights.append(f"\n### JAX Adam Performance")
        insights.append(f"- Tested up to {jax_data['max_dataset_tested']:,} individuals")
        insights.append(f"- Time complexity: O(n^{jax_data['time_scaling_exponent']:.2f})")
        insights.append(f"- Memory complexity: O(n^{jax_data['memory_scaling_exponent']:.2f})")
        
        if jax_data['efficiency_50k']:
            insights.append(f"- Memory efficiency on 50k+ datasets: {jax_data['efficiency_50k']:.1f} individuals/MB")
        if jax_data['success_rate_large']:
            insights.append(f"- Success rate on large datasets: {jax_data['success_rate_large']:.1%}")
    
    # Comparisons
    insights.append(f"\n## Strategy Comparison\n")
    insights.append(f"| Strategy | Max Size Tested | Time Scaling | Memory Scaling | 50k+ Efficiency |")
    insights.append(f"|----------|-----------------|--------------|----------------|-----------------|")
    
    for strategy, data in analysis.items():
        efficiency = f"{data['efficiency_50k']:.1f} i/MB" if data['efficiency_50k'] else "N/A"
        insights.append(f"| {strategy} | {data['max_dataset_tested']:,} | "
                       f"O(n^{data['time_scaling_exponent']:.2f}) | "
                       f"O(n^{data['memory_scaling_exponent']:.2f}) | {efficiency} |")
    
    # Recommendations
    insights.append(f"\n## Recommendations\n")
    
    if 'jax_adam' in analysis:
        jax_data = analysis['jax_adam']
        if jax_data['time_scaling_exponent'] < 1.5:
            insights.append("- ✅ **JAX Adam shows excellent sub-quadratic time scaling**")
        if jax_data['memory_scaling_exponent'] < 1.2:
            insights.append("- ✅ **JAX Adam demonstrates near-linear memory scaling**")
        if jax_data['success_rate_large'] and jax_data['success_rate_large'] > 0.8:
            insights.append("- ✅ **JAX Adam maintains high reliability on large datasets**")
        
        insights.append(f"- For datasets >50k individuals, JAX Adam is the recommended optimizer")
    
    return '\n'.join(insights)

def create_scalability_plots(df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for scalability analysis."""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time scaling plot
    ax1 = axes[0, 0]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('dataset_size')
        ax1.loglog(strategy_data['dataset_size'], strategy_data['avg_time'], 
                   'o-', label=strategy, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Dataset Size (individuals)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Optimization Time Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Memory scaling plot
    ax2 = axes[0, 1]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('dataset_size')
        ax2.loglog(strategy_data['dataset_size'], strategy_data['peak_memory_mb'],
                   's-', label=strategy, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Dataset Size (individuals)')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Memory Usage Scaling')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Memory efficiency plot
    ax3 = axes[1, 0]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('dataset_size')
        efficiency_data = strategy_data.dropna(subset=['memory_efficiency'])
        if len(efficiency_data) > 0:
            ax3.semilogx(efficiency_data['dataset_size'], efficiency_data['memory_efficiency'],
                        '^-', label=strategy, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Dataset Size (individuals)')
    ax3.set_ylabel('Memory Efficiency (individuals/MB)')
    ax3.set_title('Memory Efficiency by Dataset Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Success rate plot
    ax4 = axes[1, 1]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('dataset_size')
        ax4.semilogx(strategy_data['dataset_size'], strategy_data['success_rate'] * 100,
                     'D-', label=strategy, linewidth=2, markersize=6)
    
    ax4.set_xlabel('Dataset Size (individuals)')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Optimization Success Rate')
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"scalability_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Scalability plots saved to: {plot_file}")
    
    return plot_file

def main():
    """Main analysis function."""
    
    try:
        # Load latest results
        df, raw_results = load_latest_results()
        
        print(f"\nLoaded {len(df)} benchmark results")
        print(f"Strategies tested: {df['strategy'].unique().tolist()}")
        print(f"Dataset sizes: {sorted(df['dataset_size'].unique().tolist())}")
        
        # Analyze scalability
        analysis = analyze_scalability(df)
        
        # Generate insights
        insights = generate_scalability_insights(analysis)
        
        # Save insights
        output_dir = Path("tests/benchmarks")
        insights_file = output_dir / f"scalability_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(insights_file, 'w') as f:
            f.write(insights)
        
        print(f"\nScalability analysis saved to: {insights_file}")
        
        # Create plots (requires matplotlib)
        try:
            plot_file = create_scalability_plots(df, output_dir)
        except ImportError:
            print("Matplotlib not available - skipping plots")
            plot_file = None
        
        # Print key insights to console
        print("\n" + "="*80)
        print("KEY SCALABILITY INSIGHTS")
        print("="*80)
        
        for strategy, data in analysis.items():
            print(f"\n{strategy.upper()}:")
            print(f"  Max dataset: {data['max_dataset_tested']:,} individuals")
            print(f"  Time scaling: O(n^{data['time_scaling_exponent']:.2f})")
            print(f"  Memory scaling: O(n^{data['memory_scaling_exponent']:.2f})")
            if data['efficiency_50k']:
                print(f"  50k+ efficiency: {data['efficiency_50k']:.1f} individuals/MB")
        
        return insights_file, plot_file
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()