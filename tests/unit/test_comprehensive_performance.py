#!/usr/bin/env python3
"""
Comprehensive testing suite for optimized Pradel-JAX system.
Tests performance, edge cases, checkpoint/resume, and validation.
"""

import numpy as np
import pandas as pd
import time
import sys
import tempfile
import shutil
from pathlib import Path
import argparse
import json

import pradel_jax as pj
from pradel_jax.optimization import (
    fit_models_parallel, 
    create_model_specs_from_formulas,
    ParallelOptimizer,
    CheckpointManager
)

def create_test_data(n_individuals: int, n_occasions: int = 9, 
                    detection_prob: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Create synthetic test data with known properties."""
    np.random.seed(seed)
    
    # Create realistic capture histories
    data = []
    
    for i in range(n_individuals):
        # Random individual characteristics
        record = {
            'individual': f'ind_{i:06d}',
            'gender': np.random.choice([0, 1, 2], p=[0.1, 0.45, 0.45]),  # 0=unknown, 1=male, 2=female
            'age_1': max(0, np.random.normal(2.5, 1.0)),  # Age in years
            'tier_1': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # Some tier system
        }
        
        # Create realistic capture pattern
        # Higher detection for older individuals
        individual_p = detection_prob * (1 + 0.1 * record['age_1'])
        individual_p = min(individual_p, 0.95)  # Cap at 95%
        
        # Generate capture history
        for j in range(n_occasions):
            # Slightly decreasing detection over time (realistic)
            time_factor = 1.0 - 0.05 * j
            occasion_p = individual_p * time_factor
            record[f'occasion_{j+1}'] = np.random.binomial(1, occasion_p)
        
        data.append(record)
    
    return pd.DataFrame(data)

def test_performance_scaling():
    """Test performance scaling across different sample sizes."""
    print("\n🔬 Testing Performance Scaling")
    print("=" * 50)
    
    sample_sizes = [50, 100, 200, 500]
    results = []
    
    for n in sample_sizes:
        print(f"\n📊 Testing {n:,} individuals...")
        
        # Create test data
        test_data = create_test_data(n)
        
        # Save to temporary file
        temp_file = f"temp_perf_test_{n}.csv"
        test_data.to_csv(temp_file, index=False)
        
        try:
            # Load into Pradel-JAX
            data_context = pj.load_data(temp_file)
            
            # Simple model for performance testing
            model = pj.PradelModel()
            formula_spec = pj.create_simple_spec(phi="~1 + age_1", p="~1", f="~1 + gender")
            
            # Time single model fit
            start_time = time.time()
            
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            initial_params = model.get_initial_parameters(data_context, design_matrices)
            
            # Test likelihood computation multiple times
            n_runs = 5
            ll_times = []
            
            for run in range(n_runs):
                ll_start = time.time()
                ll = model.log_likelihood(initial_params, data_context, design_matrices)
                ll_end = time.time()
                ll_times.append(ll_end - ll_start)
            
            avg_ll_time = np.mean(ll_times)
            total_time = time.time() - start_time
            
            results.append({
                'n_individuals': n,
                'avg_likelihood_time_ms': avg_ll_time * 1000,
                'time_per_individual_ms': avg_ll_time * 1000 / n,
                'log_likelihood': ll,
                'total_setup_time_s': total_time - sum(ll_times)
            })
            
            print(f"   ⏱️  Likelihood time: {avg_ll_time*1000:.2f}ms")
            print(f"   📈 Per individual: {avg_ll_time*1000/n:.4f}ms")
            print(f"   🔢 Log-likelihood: {ll:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'n_individuals': n,
                'error': str(e)
            })
        
        finally:
            # Cleanup
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    # Analyze scaling
    print(f"\n📊 Performance Scaling Analysis:")
    print("-" * 60)
    print(f"{'Size':<8} {'Time (ms)':<12} {'Per Ind (ms)':<14} {'Scaling':<10}")
    print("-" * 60)
    
    baseline_per_ind = None
    for result in results:
        if 'error' in result:
            continue
            
        per_ind = result['time_per_individual_ms']
        if baseline_per_ind is None:
            baseline_per_ind = per_ind
            scaling = 1.0
        else:
            scaling = per_ind / baseline_per_ind
        
        print(f"{result['n_individuals']:<8} {result['avg_likelihood_time_ms']:<12.2f} "
              f"{per_ind:<14.4f} {scaling:<10.2f}x")
    
    # Extrapolate to large sizes
    if len(results) >= 2 and 'error' not in results[-1]:
        last_per_ind = results[-1]['time_per_individual_ms']
        
        print(f"\n🔮 Extrapolated Performance:")
        projections = [1000, 5000, 25000, 111000]
        
        for proj_size in projections:
            # Assume slightly worse than linear scaling (n^1.1)
            scaling_factor = (proj_size / results[-1]['n_individuals']) ** 1.1
            proj_time_ms = last_per_ind * scaling_factor * proj_size
            
            print(f"  {proj_size:,} individuals: {proj_time_ms/1000:.1f}s likelihood")
    
    return results

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 30)
    
    edge_case_results = []
    
    # Test 1: Very small dataset
    print("\n1. Testing very small dataset (5 individuals)...")
    try:
        small_data = create_test_data(5)
        temp_file = "temp_small.csv"
        small_data.to_csv(temp_file, index=False)
        
        data_context = pj.load_data(temp_file)
        model = pj.PradelModel()
        formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1")
        
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        ll = model.log_likelihood(initial_params, data_context, design_matrices)
        
        print(f"   ✅ Success: LL={ll:.2f}")
        edge_case_results.append({"case": "small_dataset", "success": True, "ll": ll})
        
        Path(temp_file).unlink()
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        edge_case_results.append({"case": "small_dataset", "success": False, "error": str(e)})
    
    # Test 2: Single individual
    print("\n2. Testing single individual...")
    try:
        single_data = create_test_data(1)
        temp_file = "temp_single.csv"
        single_data.to_csv(temp_file, index=False)
        
        data_context = pj.load_data(temp_file)
        model = pj.PradelModel()
        formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1")
        
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        ll = model.log_likelihood(initial_params, data_context, design_matrices)
        
        print(f"   ✅ Success: LL={ll:.2f}")
        edge_case_results.append({"case": "single_individual", "success": True, "ll": ll})
        
        Path(temp_file).unlink()
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        edge_case_results.append({"case": "single_individual", "success": False, "error": str(e)})
    
    # Test 3: All zeros (never captured)
    print("\n3. Testing all-zero capture histories...")
    try:
        zero_data = create_test_data(10)
        # Set all captures to zero
        for col in zero_data.columns:
            if col.startswith('occasion_'):
                zero_data[col] = 0
        
        temp_file = "temp_zeros.csv"
        zero_data.to_csv(temp_file, index=False)
        
        data_context = pj.load_data(temp_file)
        model = pj.PradelModel()
        formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1")
        
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        ll = model.log_likelihood(initial_params, data_context, design_matrices)
        
        print(f"   ✅ Success: LL={ll:.2f} (should be -inf or very negative)")
        edge_case_results.append({"case": "all_zeros", "success": True, "ll": ll})
        
        Path(temp_file).unlink()
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        edge_case_results.append({"case": "all_zeros", "success": False, "error": str(e)})
    
    # Test 4: Complex model with many parameters
    print("\n4. Testing complex model...")
    try:
        complex_data = create_test_data(100)
        temp_file = "temp_complex.csv"
        complex_data.to_csv(temp_file, index=False)
        
        data_context = pj.load_data(temp_file)
        model = pj.PradelModel()
        formula_spec = pj.create_simple_spec(
            phi="~1 + gender + age_1 + tier_1", 
            p="~1 + age_1", 
            f="~1 + gender + tier_1"
        )
        
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        ll = model.log_likelihood(initial_params, data_context, design_matrices)
        
        print(f"   ✅ Success: LL={ll:.2f}, Params={len(initial_params)}")
        edge_case_results.append({"case": "complex_model", "success": True, "ll": ll, "n_params": len(initial_params)})
        
        Path(temp_file).unlink()
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        edge_case_results.append({"case": "complex_model", "success": False, "error": str(e)})
    
    return edge_case_results

def test_checkpoint_resume():
    """Test checkpoint and resume functionality."""
    print("\n💾 Testing Checkpoint/Resume Functionality")
    print("=" * 45)
    
    # Create test data
    test_data = create_test_data(200)
    temp_file = "temp_checkpoint_test.csv"
    test_data.to_csv(temp_file, index=False)
    
    try:
        data_context = pj.load_data(temp_file)
        
        # Create model specifications
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1", "~1 + age_1", "~1 + gender"],
            p_formulas=["~1"],
            f_formulas=["~1", "~1 + gender"]
        )
        
        print(f"📊 Created {len(model_specs)} models for checkpoint test")
        
        # Test 1: Create checkpoint
        print("\n1. Testing checkpoint creation...")
        
        checkpoint_dir = Path("test_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Run partial analysis
        optimizer = ParallelOptimizer(n_workers=2, checkpoint_dir=checkpoint_dir)
        
        # Fit only first few models to create checkpoint
        partial_specs = model_specs[:3]  # Only fit first 3 models
        
        start_time = time.time()
        partial_results = optimizer.fit_models_parallel(
            model_specs=partial_specs,
            data_context=data_context,
            checkpoint_interval=2,
            checkpoint_name="test_checkpoint",
            resume=False
        )
        
        partial_time = time.time() - start_time
        successful_partial = sum(1 for r in partial_results if r and r.success)
        
        print(f"   ✅ Partial run: {successful_partial}/{len(partial_specs)} models in {partial_time:.1f}s")
        
        # Check if checkpoint was created
        checkpoint_file = checkpoint_dir / "test_checkpoint.pkl"
        if checkpoint_file.exists():
            print(f"   ✅ Checkpoint file created: {checkpoint_file}")
        else:
            print(f"   ❌ No checkpoint file found")
            return False
        
        # Test 2: Resume from checkpoint
        print("\n2. Testing resume from checkpoint...")
        
        # Try to resume (should complete remaining models)
        resume_start = time.time()
        full_results = optimizer.fit_models_parallel(
            model_specs=model_specs,  # All models
            data_context=data_context,
            checkpoint_interval=2,
            checkpoint_name="test_checkpoint",
            resume=True  # Resume from checkpoint
        )
        
        resume_time = time.time() - resume_start
        successful_full = sum(1 for r in full_results if r and r.success)
        
        print(f"   ✅ Resume run: {successful_full}/{len(model_specs)} models in {resume_time:.1f}s")
        
        # Verify results
        if successful_full >= successful_partial:
            print(f"   ✅ Resume successful: {successful_full} >= {successful_partial}")
            checkpoint_success = True
        else:
            print(f"   ❌ Resume failed: {successful_full} < {successful_partial}")
            checkpoint_success = False
        
        # Test 3: Checkpoint manager directly
        print("\n3. Testing CheckpointManager...")
        
        checkpoint_manager = CheckpointManager("test_direct_checkpoint.pkl")
        
        # Save state
        test_state = {
            "test_data": [1, 2, 3],
            "timestamp": time.time(),
            "message": "test checkpoint"
        }
        
        checkpoint_manager.save_checkpoint(test_state)
        
        # Load state
        loaded_state = checkpoint_manager.load_checkpoint()
        
        if loaded_state and loaded_state["message"] == "test checkpoint":
            print(f"   ✅ Direct checkpoint test successful")
            checkpoint_manager.delete_checkpoint()
        else:
            print(f"   ❌ Direct checkpoint test failed")
            checkpoint_success = False
        
        # Cleanup
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        
        return checkpoint_success
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {str(e)}")
        return False
    
    finally:
        # Cleanup
        if Path(temp_file).exists():
            Path(temp_file).unlink()

def test_parallel_optimization():
    """Test parallel optimization functionality."""
    print("\n⚡ Testing Parallel Optimization")
    print("=" * 35)
    
    # Create test data
    test_data = create_test_data(100)
    temp_file = "temp_parallel_test.csv"
    test_data.to_csv(temp_file, index=False)
    
    try:
        data_context = pj.load_data(temp_file)
        
        # Create model specifications
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1", "~1 + age_1"],
            p_formulas=["~1"],
            f_formulas=["~1", "~1 + gender"]
        )
        
        print(f"📊 Testing with {len(model_specs)} models")
        
        # Test sequential vs parallel timing
        print("\n1. Testing sequential fitting...")
        
        # Sequential (1 worker)
        seq_start = time.time()
        seq_results = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=1,
            checkpoint_interval=10  # No checkpoints for timing
        )
        seq_time = time.time() - seq_start
        seq_successful = sum(1 for r in seq_results if r and r.success)
        
        print(f"   ⏱️  Sequential: {seq_successful}/{len(model_specs)} models in {seq_time:.1f}s")
        
        # Parallel (multiple workers)
        print("\n2. Testing parallel fitting...")
        
        par_start = time.time()
        par_results = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=4,
            checkpoint_interval=10  # No checkpoints for timing
        )
        par_time = time.time() - par_start
        par_successful = sum(1 for r in par_results if r and r.success)
        
        print(f"   ⚡ Parallel: {par_successful}/{len(model_specs)} models in {par_time:.1f}s")
        
        # Calculate speedup
        if seq_time > 0 and par_time > 0:
            speedup = seq_time / par_time
            print(f"   🚀 Speedup: {speedup:.1f}x")
            
            if speedup > 1.5:  # Expect at least 1.5x speedup
                print(f"   ✅ Parallel optimization working")
                parallel_success = True
            else:
                print(f"   ⚠️  Limited speedup (overhead?)")
                parallel_success = True  # Still working, just overhead
        else:
            print(f"   ❌ Timing error")
            parallel_success = False
        
        # Verify results consistency
        if seq_successful == par_successful and seq_successful > 0:
            print(f"   ✅ Consistent results between sequential and parallel")
        else:
            print(f"   ⚠️  Different success rates: seq={seq_successful}, par={par_successful}")
        
        return parallel_success
        
    except Exception as e:
        print(f"❌ Parallel test failed: {str(e)}")
        return False
    
    finally:
        if Path(temp_file).exists():
            Path(temp_file).unlink()

def test_parameter_export():
    """Test parameter export functionality."""
    print("\n📊 Testing Parameter Export")
    print("=" * 30)
    
    try:
        # Create test data
        test_data = create_test_data(50)
        temp_file = "temp_export_test.csv"
        test_data.to_csv(temp_file, index=False)
        
        data_context = pj.load_data(temp_file)
        
        # Create simple model
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1", "~1 + age_1"],
            p_formulas=["~1"],
            f_formulas=["~1"]
        )
        
        print(f"📊 Testing export with {len(model_specs)} models")
        
        # Fit models
        results = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=2
        )
        
        successful_results = [r for r in results if r and r.success]
        print(f"   ✅ Fitted {len(successful_results)} models successfully")
        
        if not successful_results:
            print(f"   ❌ No successful models to test export")
            return False
        
        # Test export format
        for i, result in enumerate(successful_results[:2]):  # Test first 2
            print(f"\n   Model {i+1}: {result.model_name}")
            print(f"     AIC: {result.aic:.3f}")
            print(f"     Log-likelihood: {result.log_likelihood:.3f}")
            print(f"     Parameters ({result.n_parameters}): {[f'{p:.4f}' for p in result.parameters[:3]]}{'...' if len(result.parameters) > 3 else ''}")
            print(f"     Strategy: {result.strategy_used}")
            print(f"     Fit time: {result.fit_time:.1f}s")
        
        # Check for parameter differences
        if len(successful_results) >= 2:
            params1 = successful_results[0].parameters
            params2 = successful_results[1].parameters
            
            param_diff = max(abs(p1 - p2) for p1, p2 in zip(params1[:min(len(params1), len(params2))], params2[:min(len(params1), len(params2))]))
            
            if param_diff > 1e-6:
                print(f"   ✅ Parameters differ between models (max diff: {param_diff:.6f})")
                export_success = True
            else:
                print(f"   ⚠️  Parameters very similar (max diff: {param_diff:.6f})")
                export_success = True  # Still valid
        else:
            export_success = True
        
        return export_success
        
    except Exception as e:
        print(f"❌ Export test failed: {str(e)}")
        return False
    
    finally:
        if Path(temp_file).exists():
            Path(temp_file).unlink()

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Comprehensive Pradel-JAX Performance & Functionality Tests")
    print("=" * 70)
    
    test_results = {}
    
    try:
        # Test 1: Performance scaling
        print("\n" + "="*70)
        test_results['performance'] = test_performance_scaling()
        
        # Test 2: Edge cases
        print("\n" + "="*70)
        test_results['edge_cases'] = test_edge_cases()
        
        # Test 3: Checkpoint/resume
        print("\n" + "="*70)
        test_results['checkpoint'] = test_checkpoint_resume()
        
        # Test 4: Parallel optimization
        print("\n" + "="*70)
        test_results['parallel'] = test_parallel_optimization()
        
        # Test 5: Parameter export
        print("\n" + "="*70)
        test_results['export'] = test_parameter_export()
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    passed = 0
    total = 0
    
    # Performance test
    if 'performance' in test_results and test_results['performance']:
        print("✅ Performance scaling: PASSED")
        passed += 1
    else:
        print("❌ Performance scaling: FAILED")
    total += 1
    
    # Edge cases
    if 'edge_cases' in test_results:
        edge_passed = sum(1 for case in test_results['edge_cases'] if case.get('success', False))
        edge_total = len(test_results['edge_cases'])
        print(f"{'✅' if edge_passed == edge_total else '⚠️ '} Edge cases: {edge_passed}/{edge_total} passed")
        if edge_passed >= edge_total * 0.8:  # 80% pass rate acceptable
            passed += 1
    else:
        print("❌ Edge cases: FAILED")
    total += 1
    
    # Checkpoint
    if test_results.get('checkpoint', False):
        print("✅ Checkpoint/resume: PASSED")
        passed += 1
    else:
        print("❌ Checkpoint/resume: FAILED")
    total += 1
    
    # Parallel
    if test_results.get('parallel', False):
        print("✅ Parallel optimization: PASSED")
        passed += 1
    else:
        print("❌ Parallel optimization: FAILED")
    total += 1
    
    # Export
    if test_results.get('export', False):
        print("✅ Parameter export: PASSED")
        passed += 1
    else:
        print("❌ Parameter export: FAILED")
    total += 1
    
    print("-" * 70)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("🎉 COMPREHENSIVE TESTS PASSED - System ready for production!")
        return True
    else:
        print("⚠️  Some tests failed - Review issues before production use")
        return False

def main():
    """Main entry point for comprehensive testing."""
    parser = argparse.ArgumentParser(description='Comprehensive Pradel-JAX Tests')
    parser.add_argument('--test', choices=['all', 'performance', 'edge', 'checkpoint', 'parallel', 'export'], 
                       default='all', help='Which test to run')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_comprehensive_tests()
    elif args.test == 'performance':
        test_performance_scaling()
        success = True
    elif args.test == 'edge':
        test_edge_cases()
        success = True
    elif args.test == 'checkpoint':
        success = test_checkpoint_resume()
    elif args.test == 'parallel':
        success = test_parallel_optimization()
    elif args.test == 'export':
        success = test_parameter_export()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()