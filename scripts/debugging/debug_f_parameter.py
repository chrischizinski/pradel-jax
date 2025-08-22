#!/usr/bin/env python3
"""
Debug f parameter estimation issues.

Focused investigation of why f parameter shows 100% error in parameter recovery.
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

def generate_controlled_data(n_individuals=500, n_occasions=8, phi=0.8, p=0.7, f=0.2, seed=123):
    """Generate synthetic data with strong recruitment signal."""
    np.random.seed(seed)
    
    encounter_histories = np.zeros((n_individuals, n_occasions), dtype=int)
    
    # Track when individuals first enter the population
    entry_times = []
    
    for i in range(n_individuals):
        # Decide when this individual enters (based on recruitment)
        # For Pradel model, individuals can enter at any time with prob related to f
        # Create entry probabilities that decrease over time
        entry_probs = np.array([0.5])  # Higher prob of being present from start
        for t in range(1, n_occasions):
            # Prob of entering at time t (recruitment)
            entry_probs = np.append(entry_probs, f * (1 - f)**(t-1))
        entry_probs = entry_probs / np.sum(entry_probs)  # Normalize
        
        entry_time = np.random.choice(range(n_occasions), p=entry_probs)
        entry_times.append(entry_time)
        
        alive = True
        for t in range(entry_time, n_occasions):
            if alive:
                # Detection
                if np.random.binomial(1, p):
                    encounter_histories[i, t] = 1
                
                # Survival to next period
                if t < n_occasions - 1:
                    alive = np.random.binomial(1, phi)
    
    print(f"Entry time distribution: {np.bincount(entry_times, minlength=n_occasions)}")
    return encounter_histories

def test_f_parameter_recovery():
    """Test parameter recovery with focus on f parameter."""
    print("F Parameter Recovery Investigation")
    print("=" * 50)
    
    # Test multiple scenarios with different f values
    test_cases = [
        (0.1, "Low recruitment"),
        (0.2, "Medium recruitment"), 
        (0.3, "High recruitment"),
    ]
    
    results = []
    
    for true_f, description in test_cases:
        print(f"\nTesting {description}: f = {true_f}")
        print("-" * 40)
        
        # Fixed phi and p for comparison
        true_phi = 0.75
        true_p = 0.65
        
        # Generate data with strong recruitment signal
        encounter_data = generate_controlled_data(
            n_individuals=800,
            n_occasions=10,
            phi=true_phi,
            p=true_p,
            f=true_f,
            seed=42 + int(true_f * 100)  # Different seed for each case
        )
        
        # Count basic statistics
        n_captured = np.sum(np.sum(encounter_data, axis=1) > 0)
        n_never_captured = len(encounter_data) - n_captured
        
        # Look at capture patterns that indicate recruitment
        first_captures = []
        for history in encounter_data:
            if np.sum(history) > 0:
                first_capture = np.argmax(history)
                first_captures.append(first_capture)
        
        print(f"  Generated: {len(encounter_data)} individuals")
        print(f"  Captured: {n_captured}, Never captured: {n_never_captured}")
        print(f"  First capture distribution: {np.bincount(first_captures, minlength=10)}")
        
        # Create data context
        df_data = []
        for i, history in enumerate(encounter_data):
            ch = ''.join(map(str, history))
            df_data.append({'individual_id': i, 'ch': ch})
        
        df = pd.DataFrame(df_data)
        
        # Save and load
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            data_context = pj.load_data(temp_file.name)
        finally:
            os.unlink(temp_file.name)
        
        # Create model and fit
        parser = FormulaParser()
        formula_spec = FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )
        
        model = PradelModel()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        # Try multiple optimization attempts
        best_result = None
        best_ll = -np.inf
        
        for attempt in range(5):
            print(f"  Optimization attempt {attempt + 1}/5...")
            
            # Define objective
            def objective(params):
                return -model.log_likelihood(params, data_context, design_matrices)
            
            # Get starting parameters with some randomness
            initial_parameters = model.get_initial_parameters(data_context, design_matrices)
            if attempt > 0:
                # Add small random perturbation
                noise = np.random.normal(0, 0.1, size=initial_parameters.shape)
                initial_parameters = initial_parameters + noise
            
            bounds = model.get_parameter_bounds(data_context, design_matrices)
            
            try:
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
                print(f"    Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result and best_result.success:
            # Extract estimates
            phi_est = inv_logit(best_result.x[0])
            p_est = inv_logit(best_result.x[1])
            f_est = exp_link(best_result.x[2])
            
            # Calculate errors
            phi_error = abs(phi_est - true_phi) / true_phi * 100
            p_error = abs(p_est - true_p) / true_p * 100
            f_error = abs(f_est - true_f) / true_f * 100
            
            print(f"  SUCCESS!")
            print(f"  True:      phi={true_phi:.3f}, p={true_p:.3f}, f={true_f:.3f}")
            print(f"  Estimated: phi={phi_est:.3f}, p={p_est:.3f}, f={f_est:.3f}")
            print(f"  Errors:    phi={phi_error:.1f}%, p={p_error:.1f}%, f={f_error:.1f}%")
            print(f"  Log-likelihood: {best_ll:.2f}")
            
            results.append({
                'true_f': true_f,
                'description': description,
                'phi_est': phi_est,
                'p_est': p_est,
                'f_est': f_est,
                'phi_error': phi_error,
                'p_error': p_error,
                'f_error': f_error,
                'log_likelihood': best_ll,
                'success': True
            })
        else:
            print(f"  FAILED: No successful optimization")
            results.append({
                'true_f': true_f,
                'description': description,
                'success': False
            })
    
    # Summary
    print(f"\n" + "=" * 50)
    print("SUMMARY OF F PARAMETER RECOVERY")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"Successful fits: {len(successful_results)}/{len(results)}")
        print(f"\nF parameter errors:")
        for r in successful_results:
            print(f"  {r['description']:20s}: {r['f_error']:6.1f}% (est={r['f_est']:.3f}, true={r['true_f']:.3f})")
        
        avg_f_error = np.mean([r['f_error'] for r in successful_results])
        print(f"\nAverage f parameter error: {avg_f_error:.1f}%")
        
        if avg_f_error > 50:
            print(f"⚠️  HIGH F PARAMETER ERROR - Investigation needed")
            print(f"Possible causes:")
            print(f"  1. Model identifiability issues")
            print(f"  2. Insufficient recruitment signal in data")
            print(f"  3. Parameter bounds too restrictive")
            print(f"  4. Local optima issues")
        elif avg_f_error > 20:
            print(f"⚠️  MODERATE F PARAMETER ERROR - Some issues remain")
        else:
            print(f"✅ F PARAMETER RECOVERY GOOD")
    else:
        print(f"❌ NO SUCCESSFUL OPTIMIZATIONS")
    
    return successful_results

if __name__ == "__main__":
    results = test_f_parameter_recovery()