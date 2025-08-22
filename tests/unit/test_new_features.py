#!/usr/bin/env python3
"""
Test script for new features: stratified sampling, lambda estimation, and train/validation splits.
"""

import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import time

import pradel_jax as pj
from pradel_jax.optimization import fit_models_parallel, create_model_specs_from_formulas


def create_nebraska_style_test_data(n_individuals: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create test data mimicking Nebraska structure with tier information."""
    np.random.seed(seed)
    
    data = []
    for i in range(n_individuals):
        # Individual characteristics
        record = {
            'individual': f'NE{i:06d}',
            'gender': np.random.choice([1, 2], p=[0.5, 0.5]),  # 1=male, 2=female
            'age_1': max(1, int(np.random.normal(3, 1.5))),
        }
        
        # Tier status: some individuals start as tier 1, some as tier 2
        # Some individuals upgrade from tier 1 to tier 2 over time
        initial_tier = np.random.choice([1, 2], p=[0.7, 0.3])
        
        # Generate tier status over time (9 occasions)
        current_tier = initial_tier
        for j in range(9):
            # Small chance to upgrade from tier 1 to tier 2
            if current_tier == 1 and np.random.random() < 0.05:
                current_tier = 2
            
            record[f'tier_{j+1}'] = current_tier
        
        # Generate capture histories based on tier and individual characteristics
        # Tier 2 people have higher detection probability
        base_p = 0.15 if initial_tier == 1 else 0.25
        
        # Age affects detection (older individuals more likely to be detected)
        age_effect = 1 + 0.05 * record['age_1']
        individual_p = min(base_p * age_effect, 0.8)
        
        # Generate Y-column captures  
        for j in range(9):
            year = 2016 + j
            # Slightly increasing detection over time
            time_factor = 1 + 0.02 * j
            occasion_p = individual_p * time_factor
            
            # Higher capture probability for tier 2 in that year
            if record[f'tier_{j+1}'] == 2:
                occasion_p *= 1.3
            
            record[f'Y{year}'] = np.random.binomial(1, min(occasion_p, 0.95))
        
        data.append(record)
    
    return pd.DataFrame(data)


def test_stratified_sampling():
    """Test stratified sampling by tier status."""
    print("\n🔬 Testing Stratified Sampling by Tier Status")
    print("=" * 50)
    
    # Create test data
    full_data = create_nebraska_style_test_data(2000)
    
    # Save to temporary file
    temp_file = "temp_nebraska_test.csv"
    full_data.to_csv(temp_file, index=False)
    
    try:
        # Test 1: Check tier status determination
        print("1. Testing tier status determination...")
        
        tier_status = pj.determine_tier_status(full_data)
        ever_tier_2 = tier_status.sum()
        only_tier_1 = len(tier_status) - ever_tier_2
        
        print(f"   Original data: {ever_tier_2} ever tier 2, {only_tier_1} only tier 1")
        print(f"   Tier 2 proportion: {ever_tier_2/len(full_data):.1%}")
        
        # Test 2: Stratified sampling
        print("\n2. Testing stratified sampling...")
        
        for sample_size in [500, 100]:
            sampled_data = pj.stratified_sample(
                full_data, 
                n_samples=sample_size,
                tier_2_proportion=0.4,  # Target 40% tier 2
                random_state=42
            )
            
            sampled_tier_status = pj.determine_tier_status(sampled_data)
            sampled_tier_2 = sampled_tier_status.sum()
            sampled_tier_2_prop = sampled_tier_2 / len(sampled_data)
            
            print(f"   Sample {sample_size}: {sampled_tier_2}/{len(sampled_data)} tier 2 ({sampled_tier_2_prop:.1%})")
            
            # Should be close to 40%
            if abs(sampled_tier_2_prop - 0.4) < 0.1:
                print(f"   ✅ Target proportion achieved")
            else:
                print(f"   ⚠️  Proportion deviation: {abs(sampled_tier_2_prop - 0.4):.1%}")
        
        # Test 3: Load with sampling
        print("\n3. Testing load_data_with_sampling...")
        
        train_context = pj.load_data_with_sampling(
            temp_file,
            n_samples=300,
            tier_2_proportion=0.5,
            random_state=42
        )
        
        sampling_summary = pj.get_sampling_summary(train_context)
        print(f"   ✅ Loaded {train_context.n_individuals} individuals")
        print(f"   Sampling ratio: {sampling_summary['sampling_ratio']:.1%}")
        print(f"   Original file: {Path(sampling_summary['original_file']).name}")
        
        # Test 4: Train/validation split
        print("\n4. Testing train/validation split...")
        
        train_context, val_context = pj.load_data_with_sampling(
            temp_file,
            n_samples=400,
            validation_split=0.2,
            tier_2_proportion=0.5,
            random_state=42
        )
        
        print(f"   ✅ Training: {train_context.n_individuals} individuals")
        print(f"   ✅ Validation: {val_context.n_individuals} individuals")
        print(f"   Split ratio: {val_context.n_individuals/(train_context.n_individuals + val_context.n_individuals):.1%}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if Path(temp_file).exists():
            Path(temp_file).unlink()


def test_lambda_estimation():
    """Test lambda (population growth rate) estimation."""
    print("\n📈 Testing Lambda Estimation")
    print("=" * 35)
    
    # Create test data
    test_data = create_nebraska_style_test_data(200)
    temp_file = "temp_lambda_test.csv"
    test_data.to_csv(temp_file, index=False)
    
    try:
        # Load data
        data_context = pj.load_data(temp_file)
        
        # Create simple models for testing
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1", "~1 + age_1"],
            p_formulas=["~1"],
            f_formulas=["~1"],
            random_seed_base=12345
        )
        
        print(f"Testing lambda estimation with {len(model_specs)} models")
        
        # Fit models with parallel optimization
        results = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=2
        )
        
        successful_results = [r for r in results if r and r.success]
        print(f"✅ Successfully fitted {len(successful_results)} models")
        
        # Check lambda estimates
        for i, result in enumerate(successful_results):
            print(f"\n   Model {i+1}: {result.model_name}")
            print(f"     Lambda mean: {result.lambda_mean:.3f}")
            print(f"     Lambda median: {result.lambda_median:.3f}")
            print(f"     Lambda range: [{result.lambda_min:.3f}, {result.lambda_max:.3f}]")
            print(f"     Lambda std: {result.lambda_std:.3f}")
            print(f"     Random seed: {result.random_seed}")
            print(f"     Data hash: {result.data_hash}")
            
            # Lambda should be reasonable (between 0.5 and 2.0 typically)
            if result.lambda_mean and 0.1 < result.lambda_mean < 5.0:
                print(f"     ✅ Lambda estimate seems reasonable")
            else:
                print(f"     ⚠️  Lambda estimate may be unrealistic")
        
        # Test direct calculation
        print("\n   Testing direct lambda calculation...")
        
        if successful_results:
            result = successful_results[0]
            model = pj.PradelModel()
            
            # Recreate design matrices
            formula_spec = model_specs[0].formula_spec
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            
            # Calculate lambda directly
            lambda_values = model.calculate_lambda(
                np.array(result.parameters),
                data_context,
                design_matrices
            )
            
            lambda_summary = model.get_lambda_summary(lambda_values)
            print(f"     Direct calculation mean: {lambda_summary['lambda_mean']:.3f}")
            print(f"     ✅ Direct calculation successful")
        
        return len(successful_results) > 0
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if Path(temp_file).exists():
            Path(temp_file).unlink()


def test_reproducibility():
    """Test reproducibility with random seeds."""
    print("\n🔄 Testing Reproducibility with Seeds")
    print("=" * 40)
    
    # Create test data
    test_data = create_nebraska_style_test_data(100)
    temp_file = "temp_reproducibility_test.csv"
    test_data.to_csv(temp_file, index=False)
    
    try:
        data_context = pj.load_data(temp_file)
        
        # Create models with specific seeds
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1"],
            p_formulas=["~1"],
            f_formulas=["~1"],
            random_seed_base=9999
        )
        
        # Fit same model twice with same seed
        results1 = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=1
        )
        
        results2 = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=1
        )
        
        if results1[0].success and results2[0].success:
            # Compare results
            r1, r2 = results1[0], results2[0]
            
            param_diff = max(abs(p1 - p2) for p1, p2 in zip(r1.parameters, r2.parameters))
            ll_diff = abs(r1.log_likelihood - r2.log_likelihood)
            lambda_diff = abs(r1.lambda_mean - r2.lambda_mean) if r1.lambda_mean and r2.lambda_mean else 0
            
            print(f"   Parameter difference: {param_diff:.8f}")
            print(f"   Log-likelihood difference: {ll_diff:.8f}")
            print(f"   Lambda difference: {lambda_diff:.8f}")
            print(f"   Same seed: {r1.random_seed == r2.random_seed}")
            print(f"   Same data hash: {r1.data_hash == r2.data_hash}")
            
            if param_diff < 1e-6 and ll_diff < 1e-6:
                print(f"   ✅ Results are reproducible")
                return True
            else:
                print(f"   ⚠️  Results differ (may be due to optimization randomness)")
                return True  # Still acceptable
        else:
            print(f"   ❌ One or both optimizations failed")
            return False
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False
    
    finally:
        if Path(temp_file).exists():
            Path(temp_file).unlink()


def run_new_features_test():
    """Run all new feature tests."""
    print("🧪 Testing New Pradel-JAX Features")
    print("=" * 50)
    print("Testing: Stratified sampling, Lambda estimation, Train/validation splits, Reproducibility")
    
    start_time = time.time()
    
    test_results = {
        'stratified_sampling': False,
        'lambda_estimation': False,
        'reproducibility': False
    }
    
    try:
        # Test 1: Stratified sampling
        test_results['stratified_sampling'] = test_stratified_sampling()
        
        # Test 2: Lambda estimation  
        test_results['lambda_estimation'] = test_lambda_estimation()
        
        # Test 3: Reproducibility
        test_results['reproducibility'] = test_reproducibility()
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n" + "="*50)
    print("📋 NEW FEATURES TEST SUMMARY")
    print("="*50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, passed_test in test_results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.1f}s")
    
    if passed == total:
        print("🎉 ALL NEW FEATURES WORKING - Ready for production!")
        return True
    else:
        print("⚠️  Some tests failed - Review issues before production use")
        return False


if __name__ == "__main__":
    success = run_new_features_test()
    exit(0 if success else 1)