#!/usr/bin/env python3
"""
Comprehensive Validation Audit for Pradel-JAX

This script identifies potential silent failure modes that could compromise
statistical results without obvious error messages.

CRITICAL VALIDATION AREAS:
1. Data preprocessing and covariate handling
2. Model identifiability and parameter bounds
3. Optimization convergence validation
4. Statistical accuracy against known benchmarks
5. Edge cases and boundary conditions
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import create_model_specs_from_formulas, fit_models_parallel
from pradel_jax.data.adapters import GenericFormatAdapter, RMarkFormatAdapter
import warnings
import sys
from pathlib import Path

class ValidationAudit:
    def __init__(self):
        self.issues = []
        self.passed_tests = []
        self.warnings = []
        
    def log_issue(self, severity, category, description, recommendation):
        self.issues.append({
            'severity': severity,  # CRITICAL, HIGH, MEDIUM, LOW
            'category': category,
            'description': description,
            'recommendation': recommendation
        })
        
    def log_pass(self, test_name, description):
        self.passed_tests.append({
            'test': test_name,
            'description': description
        })
        
    def log_warning(self, category, description):
        self.warnings.append({
            'category': category,
            'description': description
        })

def test_covariate_preprocessing():
    """Test 1: Covariate preprocessing validation"""
    audit = ValidationAudit()
    
    print("🔍 TEST 1: Covariate Preprocessing Validation")
    print("=" * 60)
    
    try:
        # Load Nebraska data
        data_file = "data/encounter_histories_ne_clean.csv"
        if not Path(data_file).exists():
            audit.log_issue("CRITICAL", "Data Access", 
                          "Cannot access Nebraska dataset for validation",
                          "Ensure data file exists at expected location")
            return audit
            
        data = pd.read_csv(data_file)
        sample = data.sample(n=200, random_state=42)
        
        # Test 1.1: Missing value handling
        missing_cols = sample.columns[sample.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                missing_pct = sample[col].isnull().mean() * 100
                if missing_pct > 50:
                    audit.log_issue("HIGH", "Missing Data",
                                  f"Column '{col}' has {missing_pct:.1f}% missing values",
                                  "High missingness can lead to biased results")
                elif missing_pct > 10:
                    audit.log_warning("Missing Data", 
                                    f"Column '{col}' has {missing_pct:.1f}% missing values")
        
        # Test 1.2: Categorical variable encoding
        categorical_cols = ['gender', 'tier', 'tier_history']
        for col in categorical_cols:
            if col in sample.columns:
                unique_vals = sample[col].unique()
                if len(unique_vals) > 20:
                    audit.log_issue("MEDIUM", "Categorical Encoding",
                                  f"Column '{col}' has {len(unique_vals)} unique values",
                                  "Too many categories can cause convergence issues")
                
                # Check for numeric coding of categories
                if sample[col].dtype in ['float64', 'int64']:
                    if not all(val in [0, 1] for val in unique_vals if not pd.isna(val)):
                        audit.log_issue("HIGH", "Categorical Encoding",
                                      f"Column '{col}' uses numeric codes instead of meaningful labels",
                                      "Convert to meaningful categorical labels before modeling")
        
        # Test 1.3: Continuous variable scaling
        continuous_cols = ['age']
        for col in continuous_cols:
            if col in sample.columns:
                col_std = sample[col].std()
                col_mean = abs(sample[col].mean())
                if col_std > 100 or col_mean > 100:
                    audit.log_issue("MEDIUM", "Variable Scaling",
                                  f"Column '{col}' has large scale (mean={col_mean:.1f}, std={col_std:.1f})",
                                  "Consider standardizing continuous variables")
        
        audit.log_pass("Covariate Preprocessing", "Basic preprocessing validation completed")
        
    except Exception as e:
        audit.log_issue("CRITICAL", "Test Failure", 
                      f"Covariate preprocessing test failed: {str(e)}",
                      "Fix test environment and data access")
    
    return audit

def test_model_identifiability():
    """Test 2: Model identifiability and parameter bounds"""
    audit = ValidationAudit()
    
    print("\n🔍 TEST 2: Model Identifiability")
    print("=" * 60)
    
    try:
        # Create simple synthetic data for controlled testing
        np.random.seed(42)
        n_ind, n_occ = 100, 5
        
        # Create encounter histories with known structure
        true_phi = 0.8  # Survival probability
        true_p = 0.6    # Detection probability
        
        encounter_histories = np.zeros((n_ind, n_occ), dtype=int)
        alive = np.ones(n_ind, dtype=bool)
        
        for i in range(n_ind):
            for t in range(n_occ):
                if alive[i]:
                    # Detected with probability p
                    encounter_histories[i, t] = np.random.binomial(1, true_p)
                    # Survives to next period with probability phi
                    if t < n_occ - 1:
                        alive[i] = np.random.binomial(1, true_phi)
        
        # Test 2.1: Parameter recovery with known data
        # Create DataFrame in expected format
        synthetic_data = pd.DataFrame()
        for t in range(n_occ):
            synthetic_data[f'Y{2020+t}'] = encounter_histories[:, t]
        
        temp_file = "temp_synthetic_validation.csv"
        synthetic_data.to_csv(temp_file, index=False)
        
        try:
            adapter = GenericFormatAdapter()
            data_context = pj.load_data(temp_file, adapter=adapter)
            
            # Fit simple model
            model_specs = create_model_specs_from_formulas(
                phi_formulas=["~1"],
                p_formulas=["~1"],
                f_formulas=["~1"],
                random_seed_base=42
            )
            
            results = fit_models_parallel(
                model_specs=model_specs,
                data_context=data_context,
                n_workers=1
            )
            
            if results and len(results) > 0 and results[0].success:
                # Check parameter recovery
                params = results[0].parameters
                estimated_phi = 1 / (1 + np.exp(-params[0]))  # Logit transform
                estimated_p = 1 / (1 + np.exp(-params[1]))
                
                phi_error = abs(estimated_phi - true_phi)
                p_error = abs(estimated_p - true_p)
                
                if phi_error > 0.15:  # Allow 15% error for small sample
                    audit.log_issue("HIGH", "Parameter Recovery",
                                  f"Survival parameter recovery poor: true={true_phi:.3f}, est={estimated_phi:.3f}",
                                  "Check model formulation and optimization")
                
                if p_error > 0.15:
                    audit.log_issue("HIGH", "Parameter Recovery",
                                  f"Detection parameter recovery poor: true={true_p:.3f}, est={estimated_p:.3f}",
                                  "Check model formulation and optimization")
                
                audit.log_pass("Parameter Recovery", 
                             f"Synthetic data test: φ error={phi_error:.3f}, p error={p_error:.3f}")
                
            else:
                audit.log_issue("CRITICAL", "Model Fitting",
                              "Failed to fit simple model on synthetic data",
                              "Check basic model implementation")
        
        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
    except Exception as e:
        audit.log_issue("CRITICAL", "Test Failure",
                      f"Model identifiability test failed: {str(e)}",
                      "Fix synthetic data generation and model fitting")
    
    return audit

def test_optimization_reliability():
    """Test 3: Optimization convergence and reliability"""
    audit = ValidationAudit()
    
    print("\n🔍 TEST 3: Optimization Reliability")
    print("=" * 60)
    
    try:
        # Load real data for convergence testing
        data_file = "data/encounter_histories_ne_clean.csv"
        if Path(data_file).exists():
            data = pd.read_csv(data_file)
            sample = data.sample(n=100, random_state=42)
            
            # Apply proper preprocessing
            if 'gender' in sample.columns:
                sample['gender'] = sample['gender'].fillna(1.0).map({1.0: 'Male', 2.0: 'Female'})
            
            if 'age' in sample.columns:
                sample['age'] = (sample['age'] - sample['age'].mean()) / sample['age'].std()
            
            temp_file = "temp_optimization_test.csv"
            sample.to_csv(temp_file, index=False)
            
            try:
                adapter = GenericFormatAdapter()
                data_context = pj.load_data(temp_file, adapter=adapter)
                
                # Test multiple random seeds for consistency
                model_specs_list = []
                for seed in [42, 123, 456]:
                    specs = create_model_specs_from_formulas(
                        phi_formulas=["~1", "~1 + gender"] if 'gender' in sample.columns else ["~1"],
                        p_formulas=["~1"],
                        f_formulas=["~1"],
                        random_seed_base=seed
                    )
                    model_specs_list.extend(specs)
                
                results = fit_models_parallel(
                    model_specs=model_specs_list,
                    data_context=data_context,
                    n_workers=1
                )
                
                # Check convergence rates
                successful = [r for r in results if r and r.success]
                convergence_rate = len(successful) / len(results)
                
                if convergence_rate < 0.8:
                    audit.log_issue("HIGH", "Convergence",
                                  f"Low convergence rate: {convergence_rate:.1%}",
                                  "Check optimization settings and data quality")
                
                # Check for consistent results across seeds
                if len(successful) >= 2:
                    same_model_results = []
                    for i in range(0, len(successful), len([42, 123, 456])):
                        batch = successful[i:i+len([42, 123, 456])]
                        if len(batch) > 1:
                            same_model_results.append(batch)
                    
                    for batch in same_model_results:
                        if len(batch) >= 2:
                            likelihoods = [r.log_likelihood for r in batch]
                            if max(likelihoods) - min(likelihoods) > 0.1:
                                audit.log_issue("MEDIUM", "Consistency",
                                              f"Results vary across seeds: {likelihoods}",
                                              "Consider multi-start optimization")
                
                audit.log_pass("Optimization Reliability",
                             f"Convergence rate: {convergence_rate:.1%}")
                
            finally:
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
    except Exception as e:
        audit.log_issue("CRITICAL", "Test Failure",
                      f"Optimization reliability test failed: {str(e)}",
                      "Check optimization framework")
    
    return audit

def test_edge_cases():
    """Test 4: Edge cases and boundary conditions"""
    audit = ValidationAudit()
    
    print("\n🔍 TEST 4: Edge Cases and Boundary Conditions")
    print("=" * 60)
    
    try:
        # Test 4.1: All zeros (never detected)
        n_ind, n_occ = 50, 4
        zeros_data = pd.DataFrame()
        for t in range(n_occ):
            zeros_data[f'Y{2020+t}'] = np.zeros(n_ind, dtype=int)
        
        temp_file = "temp_zeros_test.csv"
        zeros_data.to_csv(temp_file, index=False)
        
        try:
            adapter = GenericFormatAdapter()
            data_context = pj.load_data(temp_file, adapter=adapter)
            
            model_specs = create_model_specs_from_formulas(
                phi_formulas=["~1"],
                p_formulas=["~1"],
                f_formulas=["~1"],
                random_seed_base=42
            )
            
            results = fit_models_parallel(model_specs, data_context, n_workers=1)
            
            if not results or not results[0].success:
                audit.log_issue("HIGH", "Edge Cases",
                              "Model fails with all-zero encounter histories",
                              "Add handling for extreme data cases")
            else:
                # Check if detection probability is near zero
                params = results[0].parameters
                est_p = 1 / (1 + np.exp(-params[1]))
                if est_p > 0.1:  # Should be very low
                    audit.log_issue("MEDIUM", "Edge Cases",
                                  f"Detection probability unrealistic with all-zero data: {est_p:.3f}",
                                  "Check likelihood formulation for edge cases")
                    
        except Exception as e:
            audit.log_issue("HIGH", "Edge Cases",
                          f"All-zeros test failed: {str(e)}",
                          "Add robust handling for extreme data")
        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Test 4.2: Perfect detection (all ones after first detection)
        perfect_data = pd.DataFrame()
        for i in range(n_ind):
            first_detect = np.random.randint(0, n_occ)
            row = np.zeros(n_occ)
            row[first_detect:] = 1  # Perfect detection after first
            for t in range(n_occ):
                perfect_data.loc[i, f'Y{2020+t}'] = row[t]
        
        temp_file = "temp_perfect_test.csv"
        perfect_data.to_csv(temp_file, index=False)
        
        try:
            data_context = pj.load_data(temp_file, adapter=adapter)
            results = fit_models_parallel(model_specs, data_context, n_workers=1)
            
            if results and results[0].success:
                params = results[0].parameters
                est_p = 1 / (1 + np.exp(-params[1]))
                if est_p < 0.9:  # Should be very high
                    audit.log_issue("MEDIUM", "Edge Cases",
                                  f"Detection probability too low with perfect detection data: {est_p:.3f}",
                                  "Check model behavior at boundaries")
            
        except Exception as e:
            audit.log_warning("Edge Cases", f"Perfect detection test encountered: {str(e)}")
        finally:
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        audit.log_pass("Edge Cases", "Edge case testing completed")
        
    except Exception as e:
        audit.log_issue("CRITICAL", "Test Failure",
                      f"Edge cases test failed: {str(e)}",
                      "Fix edge case testing framework")
    
    return audit

def main():
    print("🚨 COMPREHENSIVE PRADEL-JAX VALIDATION AUDIT")
    print("=" * 80)
    print("Identifying potential silent failure modes that could compromise results...")
    print()
    
    # Run all validation tests
    audits = []
    audits.append(test_covariate_preprocessing())
    audits.append(test_model_identifiability())
    audits.append(test_optimization_reliability())
    audits.append(test_edge_cases())
    
    # Compile results
    all_issues = []
    all_passes = []
    all_warnings = []
    
    for audit in audits:
        all_issues.extend(audit.issues)
        all_passes.extend(audit.passed_tests)
        all_warnings.extend(audit.warnings)
    
    # Generate report
    print("\n" + "=" * 80)
    print("📋 VALIDATION AUDIT REPORT")
    print("=" * 80)
    
    # Critical issues
    critical_issues = [i for i in all_issues if i['severity'] == 'CRITICAL']
    high_issues = [i for i in all_issues if i['severity'] == 'HIGH']
    medium_issues = [i for i in all_issues if i['severity'] == 'MEDIUM']
    low_issues = [i for i in all_issues if i['severity'] == 'LOW']
    
    print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
    for issue in critical_issues:
        print(f"   ❌ {issue['category']}: {issue['description']}")
        print(f"      → {issue['recommendation']}")
        print()
    
    print(f"\n⚠️  HIGH PRIORITY ISSUES ({len(high_issues)}):")
    for issue in high_issues:
        print(f"   🔶 {issue['category']}: {issue['description']}")
        print(f"      → {issue['recommendation']}")
        print()
    
    print(f"\n⚠️  MEDIUM PRIORITY ISSUES ({len(medium_issues)}):")
    for issue in medium_issues:
        print(f"   🔸 {issue['category']}: {issue['description']}")
        print(f"      → {issue['recommendation']}")
        print()
    
    print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
    for warning in all_warnings:
        print(f"   ⚠️  {warning['category']}: {warning['description']}")
    
    print(f"\n✅ PASSED TESTS ({len(all_passes)}):")
    for test in all_passes:
        print(f"   ✓ {test['test']}: {test['description']}")
    
    # Overall assessment
    print(f"\n" + "=" * 80)
    print("📊 OVERALL ASSESSMENT")
    print("=" * 80)
    
    if critical_issues:
        print("🚨 STATUS: CRITICAL ISSUES DETECTED - DO NOT USE FOR PRODUCTION")
        print("   → Resolve critical issues before using for analysis")
    elif high_issues:
        print("⚠️  STATUS: HIGH PRIORITY ISSUES - USE WITH CAUTION")
        print("   → Address high priority issues for reliable results")
    elif medium_issues:
        print("🔸 STATUS: MEDIUM PRIORITY ISSUES - GENERALLY SAFE")
        print("   → Consider addressing medium priority issues for optimal results")
    else:
        print("✅ STATUS: VALIDATION PASSED - SAFE FOR PRODUCTION USE")
    
    print(f"\nSUMMARY:")
    print(f"   Critical: {len(critical_issues)}")
    print(f"   High:     {len(high_issues)}")
    print(f"   Medium:   {len(medium_issues)}")
    print(f"   Low:      {len(low_issues)}")
    print(f"   Warnings: {len(all_warnings)}")
    print(f"   Passed:   {len(all_passes)}")
    
    return len(critical_issues) == 0 and len(high_issues) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)