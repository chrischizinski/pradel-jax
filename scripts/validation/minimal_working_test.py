#!/usr/bin/env python3
"""
Minimal working test to establish baseline parameter recovery capability.

Based on the successful test from debug_optimization_failure.py
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

def generate_working_data(n_individuals=100, n_occasions=5, phi=0.75, p=0.6, f=0.15, seed=42):
    """Generate data using the exact approach that worked in debug test."""
    np.random.seed(seed)
    
    encounter_data = np.zeros((n_individuals, n_occasions), dtype=int)
    
    for i in range(n_individuals):
        alive = True
        for t in range(n_occasions):
            if alive:
                # Detection
                if np.random.binomial(1, p):
                    encounter_data[i, t] = 1
                
                # Survival to next period
                if t < n_occasions - 1:
                    alive = np.random.binomial(1, phi)
    
    return encounter_data

def test_multiple_parameter_sets():
    """Test parameter recovery on multiple realistic parameter sets."""
    print("Minimal Working Parameter Recovery Test")
    print("=" * 50)
    
    # Start with parameters known to work, then vary them
    test_cases = [
        {'phi': 0.75, 'p': 0.60, 'f': 0.15, 'n': 100, 'name': 'Baseline (known working)'},
        {'phi': 0.70, 'p': 0.60, 'f': 0.15, 'n': 100, 'name': 'Lower survival'},
        {'phi': 0.80, 'p': 0.60, 'f': 0.15, 'n': 100, 'name': 'Higher survival'},
        {'phi': 0.75, 'p': 0.50, 'f': 0.15, 'n': 100, 'name': 'Lower detection'},
        {'phi': 0.75, 'p': 0.70, 'f': 0.15, 'n': 100, 'name': 'Higher detection'},
        {'phi': 0.75, 'p': 0.60, 'f': 0.10, 'n': 100, 'name': 'Lower recruitment'},
        {'phi': 0.75, 'p': 0.60, 'f': 0.20, 'n': 100, 'name': 'Higher recruitment'},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        true_phi = test_case['phi']
        true_p = test_case['p']
        true_f = test_case['f']
        n_individuals = test_case['n']
        name = test_case['name']
        
        print(f"\nTest {i+1}: {name}")
        print(f"True: phi={true_phi:.3f}, p={true_p:.3f}, f={true_f:.3f}")
        print("-" * 45)
        
        # Generate data
        encounter_data = generate_working_data(
            n_individuals=n_individuals,
            n_occasions=5,
            phi=true_phi,
            p=true_p,
            f=true_f,
            seed=42 + i
        )
        
        n_captured = np.sum(np.sum(encounter_data, axis=1) > 0)
        print(f"Data: {n_individuals} individuals, {n_captured} captured")
        
        # Create data context
        df_data = []
        for j, history in enumerate(encounter_data):
            ch = ''.join(map(str, history))
            df_data.append({'individual_id': j, 'ch': ch})
        
        df = pd.DataFrame(df_data)
        
        # Save and load
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
        
        # Test with detailed error reporting
        try:
            def objective(params):
                ll = model.log_likelihood(params, data_context, design_matrices)
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10
                return -ll
            
            initial_parameters = model.get_initial_parameters(data_context, design_matrices)
            bounds = model.get_parameter_bounds(data_context, design_matrices)
            
            # Test objective function first
            initial_obj = objective(initial_parameters)
            print(f"Initial objective: {initial_obj:.4f}")
            
            if initial_obj >= 1e10:
                print(f"❌ FAILED: Invalid initial objective")
                results.append({'name': name, 'success': False, 'reason': 'Invalid initial objective'})
                continue
            
            result = optimize_model(
                objective_function=objective,
                initial_parameters=initial_parameters,
                context=data_context,
                bounds=bounds
            )
            
            if result.success and hasattr(result, 'x'):
                # Extract estimates
                phi_est = inv_logit(result.x[0])
                p_est = inv_logit(result.x[1])
                f_est = exp_link(result.x[2])
                
                # Calculate errors
                phi_error = abs(phi_est - true_phi) / true_phi * 100
                p_error = abs(p_est - true_p) / true_p * 100
                f_error = abs(f_est - true_f) / true_f * 100
                
                print(f"✅ SUCCESS")
                print(f"Est:  phi={phi_est:.3f}, p={p_est:.3f}, f={f_est:.3f}")
                print(f"Err:  phi={phi_error:5.1f}%, p={p_error:5.1f}%, f={f_error:5.1f}%")
                
                results.append({
                    'name': name,
                    'success': True,
                    'true_phi': true_phi, 'true_p': true_p, 'true_f': true_f,
                    'est_phi': phi_est, 'est_p': p_est, 'est_f': f_est,
                    'error_phi': phi_error, 'error_p': p_error, 'error_f': f_error
                })
            else:
                print(f"❌ FAILED: Optimization unsuccessful")
                reason = getattr(result, 'message', 'Unknown optimization failure')
                results.append({'name': name, 'success': False, 'reason': reason})
                
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append({'name': name, 'success': False, 'reason': str(e)})
    
    # Summary
    print(f"\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        print(f"\nParameter estimation errors:")
        for r in successful:
            print(f"  {r['name'][:25]:<25}: φ={r['error_phi']:5.1f}%, p={r['error_p']:5.1f}%, f={r['error_f']:5.1f}%")
        
        # Focus on f parameter
        f_errors = [r['error_f'] for r in successful]
        avg_f_error = np.mean(f_errors)
        
        print(f"\nF parameter analysis:")
        print(f"  Average error: {avg_f_error:.1f}%")
        print(f"  Range: {min(f_errors):.1f}% - {max(f_errors):.1f}%")
        
        if avg_f_error < 25:
            print(f"  ✅ F parameter recovery: GOOD")
        elif avg_f_error < 75:
            print(f"  ⚠️  F parameter recovery: MODERATE")
        else:
            print(f"  ❌ F parameter recovery: POOR")
    
    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  {r['name']}: {r.get('reason', 'Unknown')}")
    
    return successful

if __name__ == "__main__":
    results = test_multiple_parameter_sets()