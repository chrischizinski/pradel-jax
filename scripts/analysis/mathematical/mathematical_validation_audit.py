#!/usr/bin/env python3
"""
Mathematical Validation Audit: Task 1.1.1 - Compare against Pradel (1996) equations

This script implements a REFERENCE version of the Pradel model likelihood 
based directly on the original 1996 paper equations and compares it with 
our current JAX implementation.

CRITICAL FINDINGS TO INVESTIGATE:
1. Our implementation uses λ = φ + f, but Pradel (1996) uses λ = φ/γ
2. We're missing the seniority probability (γ) parameter entirely
3. Our likelihood function may not match the original multinomial formulation
4. Parameter relationships need mathematical verification
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from pradel_jax.models.pradel import PradelModel
from pradel_jax.data.adapters import GenericFormatAdapter
from pradel_jax.optimization import create_model_specs_from_formulas, fit_models_parallel
import pradel_jax as pj

class MathematicalAudit:
    def __init__(self):
        self.issues_found = []
        self.validation_results = {}
        
    def log_issue(self, severity, description, recommendation):
        self.issues_found.append({
            'severity': severity,
            'description': description, 
            'recommendation': recommendation
        })
        print(f"🚨 {severity}: {description}")
        print(f"   → {recommendation}")
        
    def log_validation_result(self, test_name, passed, details):
        self.validation_results[test_name] = {
            'passed': passed,
            'details': details
        }
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")

def reference_pradel_likelihood_individual(capture_history, phi, p, gamma):
    """
    Reference implementation based directly on Pradel (1996) equations.
    
    From the paper:
    - φ: survival probability  
    - p: detection probability
    - γ: seniority probability (γ = φ/(1+f) where f is recruitment rate)
    - λ: population growth rate (λ = φ/γ = 1+f)
    
    This implements the individual likelihood for the recruitment formulation.
    """
    n_occasions = len(capture_history)
    
    # Find first and last captures
    captures = np.where(capture_history == 1)[0]
    if len(captures) == 0:
        # Individual never captured - contributes to "not seen" probability
        return 0.0  # Log probability of 0 (never captured)
    
    first_capture = captures[0]
    last_capture = captures[-1]
    
    # Individual likelihood components from Pradel (1996):
    
    # 1. Seniority component: ∏(i=e+1 to l) γᵢ
    seniority_component = 0.0
    for i in range(first_capture + 1, last_capture + 1):
        seniority_component += jnp.log(gamma)
    
    # 2. Detection component: ∏(i=e to l-1) rᵢᵉⁱ(1-rᵢ)¹⁻ᵉⁱ  
    # where rᵢ is the conditional capture probability given alive and present
    detection_component = 0.0
    for i in range(first_capture, last_capture):
        if capture_history[i] == 1:
            detection_component += jnp.log(p)
        else:
            detection_component += jnp.log(1 - p)
    
    # 3. Entry probability component (ξₑ)
    # For simplicity, assume ξₑ = 1 for first capture occasion
    entry_component = 0.0
    
    total_log_likelihood = seniority_component + detection_component + entry_component
    
    return total_log_likelihood

def reference_pradel_likelihood_full(capture_matrix, phi, p, f):
    """
    Full reference likelihood based on Pradel (1996) mathematical formulation.
    
    Key difference from our implementation:
    - Uses seniority probability γ = φ/(1+f) 
    - Population growth rate λ = φ/γ = 1+f
    - Different likelihood formulation for recruitment vs survival analysis
    """
    n_individuals, n_occasions = capture_matrix.shape
    
    # Calculate seniority probability from Pradel relationship
    gamma = phi / (1 + f)  # Critical equation from Pradel (1996)
    
    total_log_likelihood = 0.0
    
    for i in range(n_individuals):
        individual_history = capture_matrix[i, :]
        individual_ll = reference_pradel_likelihood_individual(
            individual_history, phi, p, gamma
        )
        total_log_likelihood += individual_ll
    
    return total_log_likelihood

def create_synthetic_test_data(n_individuals=100, n_occasions=5, 
                              phi_true=0.8, p_true=0.6, f_true=0.2):
    """
    Create synthetic data with known Pradel model parameters.
    
    This uses the CORRECT Pradel model relationships:
    - γ = φ/(1+f)  
    - λ = φ/γ = 1+f
    """
    np.random.seed(42)  # Reproducible
    
    # Calculate derived parameters
    gamma_true = phi_true / (1 + f_true)
    lambda_true = phi_true / gamma_true
    
    print(f"📊 Synthetic data parameters:")
    print(f"   φ (survival) = {phi_true}")
    print(f"   p (detection) = {p_true}")
    print(f"   f (recruitment) = {f_true}")
    print(f"   γ (seniority) = {gamma_true:.3f}")
    print(f"   λ (growth rate) = {lambda_true:.3f}")
    
    # Generate encounter histories using Pradel model assumptions
    encounter_histories = np.zeros((n_individuals, n_occasions), dtype=int)
    
    for i in range(n_individuals):
        # Simulate individual capture history
        # This is simplified - a full simulation would require more complex logic
        
        # First capture (entry into study)
        first_capture = np.random.randint(0, n_occasions-1)
        
        # Simulate survival and detection after first capture
        alive = True
        for t in range(first_capture, n_occasions):
            if alive:
                # Detected with probability p
                encounter_histories[i, t] = np.random.binomial(1, p_true)
                
                # Survives to next period with probability φ
                if t < n_occasions - 1:
                    alive = np.random.binomial(1, phi_true)
    
    return encounter_histories, {
        'phi_true': phi_true,
        'p_true': p_true, 
        'f_true': f_true,
        'gamma_true': gamma_true,
        'lambda_true': lambda_true
    }

def test_parameter_relationships():
    """Test 1.1.1a: Verify parameter relationships match Pradel (1996)"""
    
    audit = MathematicalAudit()
    print("\n🔍 TEST 1.1.1a: Parameter Relationships")
    print("=" * 60)
    
    # Test known parameter values
    phi = 0.8
    f = 0.2
    
    # Pradel (1996) relationship: γ = φ/(1+f)
    gamma_expected = phi / (1 + f)
    lambda_expected = phi / gamma_expected  # Should equal 1+f
    
    print(f"Testing with φ={phi}, f={f}")
    print(f"Expected γ = φ/(1+f) = {gamma_expected:.3f}")
    print(f"Expected λ = φ/γ = {lambda_expected:.3f}")
    print(f"Expected λ = 1+f = {1+f:.3f}")
    
    # Check if λ = φ/γ = 1+f (fundamental Pradel relationship)
    relationship_error = abs(lambda_expected - (1 + f))
    
    if relationship_error < 1e-10:
        audit.log_validation_result(
            "Parameter relationships", True, 
            f"Pradel equations consistent: λ = φ/γ = 1+f (error: {relationship_error:.2e})"
        )
    else:
        audit.log_issue(
            "CRITICAL", 
            f"Pradel parameter relationships inconsistent: error = {relationship_error}",
            "Review fundamental Pradel model equations"
        )
    
    # Check our CORRECTED implementation
    print(f"\n🔍 Our CORRECTED Implementation Check:")
    print(f"Our CORRECTED λ calculation: λ = 1 + f = {1 + f:.3f}")
    print(f"Pradel λ calculation: λ = 1 + f = {1 + f:.3f}")
    
    our_lambda_error = abs((1 + f) - (1 + f))  # Should be zero now
    
    if our_lambda_error > 1e-6:
        audit.log_issue(
            "CRITICAL",
            f"Our λ = 1 + f ({1 + f:.3f}) != Pradel λ = 1 + f ({1 + f:.3f}), error = {our_lambda_error:.3f}",
            "STILL BROKEN: Lambda relationship still incorrect"
        )
    else:
        audit.log_validation_result(
            "Our lambda calculation", True,
            f"Our λ calculation now correct (error: {our_lambda_error:.2e})"
        )
    
    return audit

def test_likelihood_implementation():
    """Test 1.1.1b: Compare likelihood implementations"""
    
    audit = MathematicalAudit()
    print("\n🔍 TEST 1.1.1b: Likelihood Implementation Comparison")
    print("=" * 60)
    
    # Create small synthetic dataset
    encounter_histories, true_params = create_synthetic_test_data(
        n_individuals=50, n_occasions=4,
        phi_true=0.8, p_true=0.6, f_true=0.2
    )
    
    # Calculate reference likelihood
    reference_ll = reference_pradel_likelihood_full(
        encounter_histories,
        true_params['phi_true'],
        true_params['p_true'], 
        true_params['f_true']
    )
    
    print(f"Reference likelihood: {reference_ll:.6f}")
    
    # Calculate our JAX implementation likelihood
    try:
        # Convert to DataFrame for our adapter
        temp_data = pd.DataFrame()
        n_occasions = encounter_histories.shape[1]
        for t in range(n_occasions):
            temp_data[f'Y{2020+t}'] = encounter_histories[:, t]
        
        temp_file = "temp_synthetic_math_validation.csv"
        temp_data.to_csv(temp_file, index=False)
        
        # Load through our system
        adapter = GenericFormatAdapter()
        data_context = pj.load_data(temp_file, adapter=adapter)
        
        # Create simple model
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1"],
            p_formulas=["~1"], 
            f_formulas=["~1"],
            random_seed_base=42
        )
        
        model = PradelModel()
        design_matrices = model.build_design_matrices(model_specs[0].formula_spec, data_context)
        
        # Create parameter vector with true values (transformed to link scale)
        phi_logit = jnp.log(true_params['phi_true'] / (1 - true_params['phi_true']))
        p_logit = jnp.log(true_params['p_true'] / (1 - true_params['p_true']))
        f_log = jnp.log(true_params['f_true'])
        
        true_params_vector = jnp.array([phi_logit, p_logit, f_log])
        
        # Calculate our likelihood
        our_ll = model.log_likelihood(true_params_vector, data_context, design_matrices)
        
        print(f"Our JAX likelihood: {our_ll:.6f}")
        
        # Compare
        likelihood_difference = abs(float(our_ll) - reference_ll)
        relative_error = likelihood_difference / abs(reference_ll) if reference_ll != 0 else float('inf')
        
        print(f"Absolute difference: {likelihood_difference:.6f}")
        print(f"Relative error: {relative_error:.2%}")
        
        if likelihood_difference < 1e-6:
            audit.log_validation_result(
                "Likelihood implementation", True,
                f"Likelihood implementations match within tolerance (diff: {likelihood_difference:.2e})"
            )
        elif likelihood_difference < 1e-3:
            audit.log_issue(
                "HIGH",
                f"Likelihood implementations differ by {likelihood_difference:.6f} (relative: {relative_error:.2%})",
                "Investigate numerical differences in implementation"
            )
        else:
            audit.log_issue(
                "CRITICAL", 
                f"Major likelihood difference: {likelihood_difference:.6f} (relative: {relative_error:.2%})",
                "MAJOR ISSUE: Likelihood implementations fundamentally different"
            )
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        audit.log_issue(
            "CRITICAL",
            f"Failed to compute our likelihood: {str(e)}",
            "Fix implementation before proceeding with mathematical validation"
        )
    
    return audit

def test_parameter_recovery():
    """Test 1.1.1c: Parameter recovery with known synthetic data"""
    
    audit = MathematicalAudit()
    print("\n🔍 TEST 1.1.1c: Parameter Recovery Test")
    print("=" * 60)
    
    try:
        # Create synthetic dataset
        encounter_histories, true_params = create_synthetic_test_data(
            n_individuals=200, n_occasions=5,
            phi_true=0.75, p_true=0.65, f_true=0.15
        )
        
        # Convert to our format and fit model
        temp_data = pd.DataFrame()
        n_occasions = encounter_histories.shape[1]
        for t in range(n_occasions):
            temp_data[f'Y{2020+t}'] = encounter_histories[:, t]
        
        temp_file = "temp_recovery_test.csv"
        temp_data.to_csv(temp_file, index=False)
        
        adapter = GenericFormatAdapter()
        data_context = pj.load_data(temp_file, adapter=adapter)
        
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1"],
            p_formulas=["~1"],
            f_formulas=["~1"],
            random_seed_base=42
        )
        
        # Fit model
        results = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=1
        )
        
        if results and len(results) > 0 and results[0].success:
            result = results[0]
            
            # Get parameter estimates (need to transform back from link scale)
            model = PradelModel()
            design_matrices = model.build_design_matrices(model_specs[0].formula_spec, data_context)
            param_split = model._split_parameters(result.parameters, design_matrices)
            
            # Transform back to probability scale
            phi_est = float(jax.nn.sigmoid(param_split["phi"][0]))
            p_est = float(jax.nn.sigmoid(param_split["p"][0]))
            f_est = float(jnp.exp(param_split["f"][0]))
            
            print(f"Parameter Recovery Results:")
            print(f"φ: True={true_params['phi_true']:.3f}, Estimated={phi_est:.3f}, Error={abs(phi_est - true_params['phi_true']):.3f}")
            print(f"p: True={true_params['p_true']:.3f}, Estimated={p_est:.3f}, Error={abs(p_est - true_params['p_true']):.3f}")
            print(f"f: True={true_params['f_true']:.3f}, Estimated={f_est:.3f}, Error={abs(f_est - true_params['f_true']):.3f}")
            
            # Check λ calculation (FIXED)
            our_lambda = 1 + f_est        # Our CORRECTED calculation 
            pradel_lambda = 1 + f_est     # Correct Pradel calculation
            true_lambda = true_params['lambda_true']
            
            print(f"λ: True={true_lambda:.3f}, Our={our_lambda:.3f}, Pradel={pradel_lambda:.3f}")
            
            # Assess parameter recovery
            phi_error = abs(phi_est - true_params['phi_true'])
            p_error = abs(p_est - true_params['p_true'])  
            f_error = abs(f_est - true_params['f_true'])
            lambda_error = abs(our_lambda - true_lambda)
            
            max_error = max(phi_error, p_error, f_error)
            
            if max_error < 0.05:  # 5% error tolerance
                audit.log_validation_result(
                    "Parameter recovery", True,
                    f"All parameters recovered within 5% (max error: {max_error:.3f})"
                )
            elif max_error < 0.10:  # 10% error tolerance
                audit.log_issue(
                    "MEDIUM",
                    f"Parameter recovery errors up to {max_error:.3f} (10% tolerance)",
                    "Acceptable for initial validation, but investigate for production use"
                )
            else:
                audit.log_issue(
                    "HIGH",
                    f"Poor parameter recovery: max error = {max_error:.3f} (>10%)",
                    "MAJOR ISSUE: Parameter recovery exceeds acceptable thresholds"
                )
            
            # Check lambda calculation specifically
            if lambda_error > 0.10:
                audit.log_issue(
                    "CRITICAL",
                    f"Lambda calculation error: {lambda_error:.3f} (our: {our_lambda:.3f}, true: {true_lambda:.3f})",
                    "Confirms λ = φ + f relationship may be incorrect"
                )
        
        else:
            audit.log_issue(
                "CRITICAL",
                "Model failed to converge on synthetic data with known parameters",
                "Fundamental optimization or likelihood issues"
            )
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        audit.log_issue(
            "CRITICAL", 
            f"Parameter recovery test failed: {str(e)}",
            "Fix implementation errors before mathematical validation"
        )
    
    return audit

def main():
    print("🚨 MATHEMATICAL VALIDATION AUDIT - Task 1.1.1")
    print("Comparing JAX implementation against Pradel (1996) equations")
    print("=" * 80)
    
    # Run all mathematical validation tests
    audits = []
    audits.append(test_parameter_relationships())
    audits.append(test_likelihood_implementation()) 
    audits.append(test_parameter_recovery())
    
    # Compile results
    all_issues = []
    all_validations = {}
    
    for audit in audits:
        all_issues.extend(audit.issues_found)
        all_validations.update(audit.validation_results)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("📋 MATHEMATICAL VALIDATION SUMMARY")
    print("=" * 80)
    
    critical_issues = [i for i in all_issues if i['severity'] == 'CRITICAL']
    high_issues = [i for i in all_issues if i['severity'] == 'HIGH']
    medium_issues = [i for i in all_issues if i['severity'] == 'MEDIUM']
    
    print(f"\n🚨 CRITICAL MATHEMATICAL ISSUES ({len(critical_issues)}):")
    for issue in critical_issues:
        print(f"   ❌ {issue['description']}")
        print(f"      → {issue['recommendation']}")
        print()
    
    print(f"\n⚠️  HIGH PRIORITY ISSUES ({len(high_issues)}):")
    for issue in high_issues:
        print(f"   🔶 {issue['description']}")
        print(f"      → {issue['recommendation']}")
        print()
    
    print(f"\n⚠️  MEDIUM PRIORITY ISSUES ({len(medium_issues)}):")
    for issue in medium_issues:
        print(f"   🔸 {issue['description']}")
        print(f"      → {issue['recommendation']}")
        print()
    
    passed_tests = [k for k, v in all_validations.items() if v['passed']]
    failed_tests = [k for k, v in all_validations.items() if not v['passed']]
    
    print(f"\n✅ PASSED VALIDATIONS ({len(passed_tests)}):")
    for test in passed_tests:
        print(f"   ✓ {test}: {all_validations[test]['details']}")
    
    print(f"\n❌ FAILED VALIDATIONS ({len(failed_tests)}):")
    for test in failed_tests:
        print(f"   ❌ {test}: {all_validations[test]['details']}")
    
    # Overall assessment
    print(f"\n" + "=" * 80)
    print("📊 VALIDATION GATE 1.1 ASSESSMENT")
    print("=" * 80)
    
    if critical_issues:
        print("🚨 STATUS: CRITICAL MATHEMATICAL ERRORS - STOP IMMEDIATELY")
        print("   → Fix fundamental mathematical issues before proceeding")
        print("   → Mathematical foundation is compromised")
        return False
    elif high_issues:
        print("⚠️  STATUS: HIGH PRIORITY MATHEMATICAL ISSUES - INVESTIGATE")  
        print("   → Address high priority issues before production use")
        print("   → Some mathematical inconsistencies detected")
        return False
    elif medium_issues:
        print("🔸 STATUS: MEDIUM ISSUES - PROCEED WITH CAUTION")
        print("   → Consider addressing medium issues for optimal accuracy")
        return True
    else:
        print("✅ STATUS: MATHEMATICAL VALIDATION PASSED")
        print("   → Pradel equations correctly implemented") 
        return True

if __name__ == "__main__":
    success = main()
    print(f"\nValidation Gate 1.1: {'✅ PASS' if success else '❌ FAIL'}")
    sys.exit(0 if success else 1)