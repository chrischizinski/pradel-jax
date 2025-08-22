#!/usr/bin/env python3
"""
Simple parameter recovery test using working optimization approach.

Tests f parameter estimation with controlled, simple synthetic data.
"""

import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
from pradel_jax.optimization.orchestrator import optimize_model
import tempfile
import os

def generate_simple_data(n_individuals=200, n_occasions=6, phi=0.75, p=0.6, f=0.2, seed=42):
    """Generate simple synthetic data with known parameters."""
    np.random.seed(seed)
    
    encounter_histories = np.zeros((n_individuals, n_occasions), dtype=int)
    
    # Simple approach: all individuals present from start, recruit new ones each period
    for i in range(n_individuals):
        # Some start from beginning, others enter later (recruitment)
        if i < int(n_individuals * 0.7):  # 70% start from beginning
            start_time = 0
        else:
            # Recruit randomly in later periods
            start_time = np.random.choice(range(1, n_occasions))
        
        alive = True
        for t in range(start_time, n_occasions):
            if alive:
                # Detection
                if np.random.binomial(1, p):
                    encounter_histories[i, t] = 1
                
                # Survival to next period
                if t < n_occasions - 1:
                    alive = np.random.binomial(1, phi)
    
    return encounter_histories

def test_parameter_recovery():
    """Test parameter recovery with simple approach."""
    print("Simple Parameter Recovery Test")
    print("=" * 40)
    
    # Test cases with different true parameter values
    test_cases = [
        {'phi': 0.70, 'p': 0.60, 'f': 0.10, 'name': 'Low survival, low recruitment'},
        {'phi': 0.80, 'p': 0.65, 'f': 0.20, 'name': 'High survival, medium recruitment'},
        {'phi': 0.75, 'p': 0.70, 'f': 0.15, 'name': 'Medium parameters'},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        true_phi = test_case['phi']
        true_p = test_case['p']
        true_f = test_case['f']
        name = test_case['name']
        
        print(f"\nTest {i+1}: {name}")
        print(f"True parameters: phi={true_phi:.3f}, p={true_p:.3f}, f={true_f:.3f}")
        print("-" * 60)
        
        # Generate data
        encounter_data = generate_simple_data(
            n_individuals=300,
            n_occasions=8,
            phi=true_phi,
            p=true_p,
            f=true_f,
            seed=42 + i
        )
        
        n_captured = np.sum(np.sum(encounter_data, axis=1) > 0)
        print(f"Generated: {len(encounter_data)} individuals, {n_captured} captured")
        
        # Create data context
        df_data = []
        for j, history in enumerate(encounter_data):
            ch = ''.join(map(str, history))
            df_data.append({'individual_id': j, 'ch': ch})
        
        df = pd.DataFrame(df_data)
        
        # Save and load through temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            data_context = pj.load_data(temp_file.name)
        finally:
            os.unlink(temp_file.name)
        
        # Create model
        parser = FormulaParser()
        formula_spec = FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )
        
        model = PradelModel()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        # Test optimization (try up to 3 times with different starting points)
        best_result = None
        best_ll = -np.inf
        
        for attempt in range(3):
            try:
                def objective(params):
                    return -model.log_likelihood(params, data_context, design_matrices)
                
                initial_parameters = model.get_initial_parameters(data_context, design_matrices)
                if attempt > 0:
                    # Add small perturbation for additional attempts
                    noise = np.random.normal(0, 0.05, size=initial_parameters.shape)
                    initial_parameters = initial_parameters + noise
                
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_parameters,
                    context=data_context,
                    bounds=bounds
                )
                
                if result.success and hasattr(result, 'fun'):
                    current_ll = -result.fun
                    if current_ll > best_ll:
                        best_ll = current_ll
                        best_result = result
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result and best_result.success:
            # Extract estimates
            phi_est = inv_logit(best_result.x[0])
            p_est = inv_logit(best_result.x[1])
            f_est = exp_link(best_result.x[2])
            
            # Calculate relative errors
            phi_error = abs(phi_est - true_phi) / true_phi * 100
            p_error = abs(p_est - true_p) / true_p * 100
            f_error = abs(f_est - true_f) / true_f * 100
            
            print(f"✅ OPTIMIZATION SUCCESSFUL")
            print(f"   Estimated: phi={phi_est:.3f}, p={p_est:.3f}, f={f_est:.3f}")
            print(f"   Errors:    phi={phi_error:5.1f}%, p={p_error:5.1f}%, f={f_error:5.1f}%")
            print(f"   Log-likelihood: {best_ll:.2f}")
            
            results.append({
                'test_name': name,
                'true_phi': true_phi, 'true_p': true_p, 'true_f': true_f,
                'est_phi': phi_est, 'est_p': p_est, 'est_f': f_est,
                'error_phi': phi_error, 'error_p': p_error, 'error_f': f_error,
                'log_likelihood': best_ll,
                'success': True
            })
        else:
            print(f"❌ OPTIMIZATION FAILED")
            results.append({
                'test_name': name,
                'success': False
            })
    
    # Summary
    print(f"\n" + "=" * 60)
    print("PARAMETER RECOVERY SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    
    if successful:
        print(f"Successful tests: {len(successful)}/{len(results)}")
        print(f"\nParameter estimation errors:")
        print(f"{'Test':<30} {'φ Error':<10} {'p Error':<10} {'f Error':<10}")
        print("-" * 65)
        
        for r in successful:
            print(f"{r['test_name'][:28]:<30} {r['error_phi']:6.1f}%   {r['error_p']:6.1f}%   {r['error_f']:6.1f}%")
        
        # Calculate average errors
        avg_phi_error = np.mean([r['error_phi'] for r in successful])
        avg_p_error = np.mean([r['error_p'] for r in successful])
        avg_f_error = np.mean([r['error_f'] for r in successful])
        
        print("-" * 65)
        print(f"{'AVERAGE':<30} {avg_phi_error:6.1f}%   {avg_p_error:6.1f}%   {avg_f_error:6.1f}%")
        
        # Assessment
        print(f"\nASSESSMENT:")
        if avg_f_error < 20:
            print(f"✅ F parameter recovery: EXCELLENT (avg error {avg_f_error:.1f}%)")
        elif avg_f_error < 50:
            print(f"⚠️  F parameter recovery: ACCEPTABLE (avg error {avg_f_error:.1f}%)")
        else:
            print(f"❌ F parameter recovery: POOR (avg error {avg_f_error:.1f}%)")
        
        print(f"\nDetailed f parameter results:")
        for r in successful:
            print(f"  {r['test_name'][:25]:<25}: true={r['true_f']:.3f}, est={r['est_f']:.3f}, error={r['error_f']:5.1f}%")
        
    else:
        print(f"❌ NO SUCCESSFUL PARAMETER RECOVERY TESTS")
    
    return successful

if __name__ == "__main__":
    results = test_parameter_recovery()