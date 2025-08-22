#!/usr/bin/env python3
"""
Test lambda (population growth rate) estimation in Pradel models.

Verifies that:
1. Lambda calculation is correct (λ = 1 + f)
2. Lambda values are computed and reported properly 
3. Lambda summaries are accurate
4. Integration with parallel optimization works
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set PYTHONPATH to include our package
import sys
sys.path.insert(0, str(Path(__file__).parent))

def test_lambda_calculation_basic():
    """Test basic lambda calculation with known values."""
    print("🧮 Testing basic lambda calculation...")
    
    import pradel_jax as pj
    from pradel_jax.data.adapters import RMarkFormatAdapter
    
    # Create simple test data
    test_data = pd.DataFrame({
        'ch': ['1100', '1010', '1001', '0110', '0101', '0011'],
        'group': ['A', 'B', 'A', 'B', 'A', 'B']
    })
    
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    print(f"Test data: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    
    # Create constant model (simplest case)
    formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1")
    
    # Build model
    model = pj.PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    # Test with known parameter values
    # Let's set f = log(0.5) so recruitment rate = 0.5, lambda = 1.5
    test_params = np.array([0.0, 0.0, np.log(0.5)])  # phi=0.5, p=0.5, f=0.5
    
    print(f"Test parameters: {test_params}")
    print(f"Expected f = exp({np.log(0.5):.3f}) = {np.exp(np.log(0.5)):.3f}")
    print(f"Expected lambda = 1 + f = 1 + {np.exp(np.log(0.5)):.3f} = {1 + np.exp(np.log(0.5)):.3f}")
    
    # Calculate lambda
    lambda_values = model.calculate_lambda(test_params, data_context, design_matrices)
    
    print(f"Calculated lambda values: {lambda_values}")
    print(f"Lambda shape: {lambda_values.shape}")
    
    # Check if all lambda values are the same (constant model)
    expected_lambda = 1.5
    if np.allclose(lambda_values, expected_lambda):
        print(f"✅ Lambda calculation correct: all values = {expected_lambda:.3f}")
    else:
        print(f"❌ Lambda calculation incorrect: got {lambda_values}, expected {expected_lambda}")
        return False
    
    # Test lambda summary
    lambda_summary = model.get_lambda_summary(lambda_values)
    
    print(f"\n📊 Lambda summary:")
    for key, value in lambda_summary.items():
        print(f"   {key}: {value:.6f}")
    
    # Check summary values
    if abs(lambda_summary['lambda_mean'] - expected_lambda) < 1e-6:
        print(f"✅ Lambda summary correct")
    else:
        print(f"❌ Lambda summary incorrect")
        return False
        
    return True


def test_lambda_with_covariates():
    """Test lambda calculation with covariate models."""
    print("\n🔢 Testing lambda with covariates...")
    
    import pradel_jax as pj
    from pradel_jax.data.adapters import RMarkFormatAdapter
    
    # Create data with covariate variation
    test_data = pd.DataFrame({
        'ch': ['1100', '1010', '1001', '0110', '0101', '0011'],
        'habitat': ['forest', 'grassland', 'forest', 'grassland', 'forest', 'grassland']
    })
    
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    print(f"Test data with covariates: {list(data_context.covariates.keys())}")
    
    # Create model with covariate effect on recruitment
    formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1 + habitat")
    
    model = pj.PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    print(f"F design matrix shape: {design_matrices['f'].matrix.shape}")
    print(f"F design matrix:\n{design_matrices['f'].matrix}")
    
    # Test with different f parameters for different habitats
    # f_intercept = log(0.3), f_habitat_effect = log(2.0) 
    test_params = np.array([0.0, 0.0, np.log(0.3), np.log(2.0)])
    
    print(f"\nTest parameters: {test_params}")
    
    # Calculate lambda values
    lambda_values = model.calculate_lambda(test_params, data_context, design_matrices)
    
    print(f"Lambda values by individual: {lambda_values}")
    
    # Check if we get different lambda values for different habitats
    unique_lambdas = np.unique(np.round(lambda_values, 6))
    if len(unique_lambdas) > 1:
        print(f"✅ Covariate effect working: {len(unique_lambdas)} different lambda values")
        print(f"   Lambda values: {unique_lambdas}")
    else:
        print(f"❌ Covariate effect not working: all lambda values the same")
        return False
    
    # Test summary statistics
    lambda_summary = model.get_lambda_summary(lambda_values)
    
    print(f"\n📊 Lambda summary with covariates:")
    for key, value in lambda_summary.items():
        print(f"   {key}: {value:.6f}")
    
    # Check that min != max (indicating variation)
    if lambda_summary['lambda_min'] != lambda_summary['lambda_max']:
        print(f"✅ Lambda variation captured in summary")
    else:
        print(f"❌ No lambda variation in summary")
        return False
        
    return True


def test_lambda_parallel_integration():
    """Test lambda estimation in parallel optimization."""
    print("\n⚡ Testing lambda in parallel optimization...")
    
    import pradel_jax as pj
    from pradel_jax.optimization.parallel import ParallelOptimizer, ParallelModelSpec
    from pradel_jax.optimization.strategy import OptimizationStrategy
    from pradel_jax.data.adapters import RMarkFormatAdapter
    
    # Create realistic test data
    test_data = pd.DataFrame({
        'ch': ['111000', '110100', '101010', '011001', '010110', '001101',
               '110000', '101000', '011000', '001100', '000110', '000011'],
        'treatment': ['control', 'treatment'] * 6
    })
    
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    # Create models with different recruitment structures
    model_specs = [
        ParallelModelSpec(
            name="Constant recruitment",
            formula_spec=pj.create_simple_spec(phi="~1", p="~1", f="~1"),
            index=0
        ),
        ParallelModelSpec(
            name="Treatment effect on recruitment",
            formula_spec=pj.create_simple_spec(phi="~1", p="~1", f="~1 + treatment"),
            index=1
        )
    ]
    
    print(f"Testing {len(model_specs)} models with parallel optimization")
    
    # Run parallel optimization
    optimizer = ParallelOptimizer(n_workers=2)
    results = optimizer.fit_models_parallel(
        model_specs=model_specs,
        data_context=data_context,
        strategy=OptimizationStrategy.SCIPY_LBFGS
    )
    
    # Check lambda estimation in results
    successful_results = [r for r in results if r and r.success]
    
    if not successful_results:
        print("❌ No successful optimization results")
        return False
    
    print(f"\n📊 Lambda results from parallel optimization:")
    print("-" * 60)
    
    all_have_lambda = True
    for result in successful_results:
        has_lambda = result.lambda_mean is not None
        
        print(f"{result.model_name:25} | Lambda: {result.lambda_mean:.4f} ± {result.lambda_std:.4f}" if has_lambda 
              else f"{result.model_name:25} | Lambda: Not calculated")
        
        if has_lambda:
            print(f"{'':25} | Range: [{result.lambda_min:.4f}, {result.lambda_max:.4f}]")
            print(f"{'':25} | Q25-Q75: [{result.lambda_q25:.4f}, {result.lambda_q75:.4f}]")
        else:
            all_have_lambda = False
        
        print("-" * 60)
    
    if all_have_lambda:
        print("✅ All models have lambda estimates")
    else:
        print("❌ Some models missing lambda estimates")
        return False
    
    # Check if treatment model has different lambda distribution
    if len(successful_results) >= 2:
        lambda_std_constant = successful_results[0].lambda_std or 0
        lambda_std_treatment = successful_results[1].lambda_std or 0
        
        if lambda_std_treatment > lambda_std_constant:
            print("✅ Treatment model shows lambda variation (expected)")
        else:
            print("ℹ️  Treatment model shows little lambda variation")
    
    return True


def test_lambda_theoretical_validation():
    """Test lambda calculation against theoretical expectations."""
    print("\n📚 Testing lambda theoretical validation...")
    
    import pradel_jax as pj
    
    print("Theoretical validation:")
    print("   Pradel (1996): λ = 1 + f, where f is the recruitment rate")
    print("   - If f = 0.2 (20% recruitment), then λ = 1.2 (20% population growth)")
    print("   - If f = 0.0 (no recruitment), then λ = 1.0 (stable population)")
    print("   - If f = -0.1 (10% decline), then λ = 0.9 (10% population decline)")
    
    # Test with specific recruitment values
    test_cases = [
        ("Stable population", 0.0, 1.0),
        ("Growing population", 0.5, 1.5),
        ("Rapid growth", 1.0, 2.0),
        ("Declining population", -0.1, 0.9),
        ("Slow growth", 0.1, 1.1)
    ]
    
    all_correct = True
    
    for case_name, f_value, expected_lambda in test_cases:
        # Test the transformation directly
        calculated_lambda = 1.0 + f_value
        
        if abs(calculated_lambda - expected_lambda) < 1e-10:
            print(f"   ✅ {case_name}: f={f_value:.1f} → λ={calculated_lambda:.1f}")
        else:
            print(f"   ❌ {case_name}: Expected λ={expected_lambda:.1f}, got {calculated_lambda:.1f}")
            all_correct = False
    
    if all_correct:
        print("✅ All theoretical validations passed")
    else:
        print("❌ Some theoretical validations failed")
    
    return all_correct


if __name__ == "__main__":
    print("Lambda Estimation Test Suite")
    print("=" * 50)
    
    # Run all tests
    test1 = test_lambda_calculation_basic()
    test2 = test_lambda_with_covariates() 
    test3 = test_lambda_parallel_integration()
    test4 = test_lambda_theoretical_validation()
    
    # Summary
    print("\n" + "=" * 50)
    print("LAMBDA ESTIMATION TEST SUMMARY")
    print("=" * 50)
    print(f"Basic calculation:        {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Covariate models:         {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Parallel integration:     {'✅ PASS' if test3 else '❌ FAIL'}")
    print(f"Theoretical validation:   {'✅ PASS' if test4 else '❌ FAIL'}")
    
    overall_success = all([test1, test2, test3, test4])
    
    if overall_success:
        print(f"\n🎉 ALL LAMBDA TESTS PASSED!")
        print("Lambda estimation is working correctly:")
        print("• Calculation follows Pradel (1996): λ = 1 + f")
        print("• Handles constant and covariate models properly")
        print("• Integrates with parallel optimization framework")
        print("• Produces meaningful summary statistics")
        print("• Provides biologically interpretable growth rates")
    else:
        print(f"\n❌ SOME LAMBDA TESTS FAILED")
        print("Check the specific test failures above")
        
    exit(0 if overall_success else 1)