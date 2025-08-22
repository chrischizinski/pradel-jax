#!/usr/bin/env python3
"""
Test and demonstrate capture-recapture model validation best practices.

This script shows the proper statistical workflow for model validation in 
capture-recapture studies following ecological literature best practices.
"""

import numpy as np
import pandas as pd
import time

import pradel_jax as pj
from pradel_jax.optimization import create_model_specs_from_formulas, fit_models_parallel
from pradel_jax.validation.capture_recapture_validation import (
    comprehensive_model_validation,
    perform_model_averaging,
    bootstrap_model_parameters,
    goodness_of_fit_test
)


def create_realistic_capture_data(n_individuals: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create realistic capture-recapture data for validation testing."""
    np.random.seed(seed)
    
    data = []
    for i in range(n_individuals):
        # Individual characteristics affecting survival and detection
        record = {
            'individual': f'ID_{i:05d}',
            'gender': np.random.choice([1, 2], p=[0.5, 0.5]),
            'age_1': max(1, int(np.random.normal(3, 1.2))),
            'tier_1': np.random.choice([1, 2], p=[0.7, 0.3])
        }
        
        # True parameters (vary by individual characteristics)
        base_survival = 0.7
        base_detection = 0.4
        base_recruitment = 0.1
        
        # Age effect
        age_effect = 1 + 0.05 * record['age_1']
        survival_prob = min(base_survival * age_effect, 0.95)
        detection_prob = min(base_detection * age_effect, 0.9)
        
        # Gender effect  
        if record['gender'] == 2:  # Female
            survival_prob *= 1.1
            detection_prob *= 0.9
        
        # Tier effect
        if record['tier_1'] == 2:
            detection_prob *= 1.3
        
        # Generate capture history (9 occasions)
        alive = True
        for j in range(9):
            year = 2016 + j
            
            if alive:
                # Survival check (except first occasion)
                if j > 0:
                    alive = np.random.random() < survival_prob
                
                # Detection check if alive
                if alive:
                    detected = np.random.random() < detection_prob
                    record[f'Y{year}'] = 1 if detected else 0
                else:
                    record[f'Y{year}'] = 0
            else:
                record[f'Y{year}'] = 0
        
        data.append(record)
    
    return pd.DataFrame(data)


def demonstrate_model_averaging_workflow():
    """Demonstrate proper model averaging workflow for capture-recapture."""
    print("\n🔬 Demonstrating Capture-Recapture Model Validation Best Practices")
    print("=" * 70)
    
    # Create realistic test data
    print("Creating realistic capture-recapture data...")
    test_data = create_realistic_capture_data(300)
    
    # Save and load through normal pipeline
    temp_file = "temp_validation_demo.csv"
    test_data.to_csv(temp_file, index=False)
    
    try:
        # Load data
        data_context = pj.load_data(temp_file)
        print(f"✅ Loaded {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
        
        # Create model set following capture-recapture best practices
        # Start simple and add complexity systematically
        model_specs = create_model_specs_from_formulas(
            phi_formulas=[
                "~1",                    # Constant survival
                "~1 + age_1",           # Age effect on survival  
                "~1 + gender",          # Gender effect
                "~1 + age_1 + gender"   # Additive effects
            ],
            p_formulas=[
                "~1",                    # Constant detection
                "~1 + tier_1"           # Tier effect on detection
            ],
            f_formulas=["~1"],          # Keep recruitment simple
            random_seed_base=12345
        )
        
        print(f"✅ Created {len(model_specs)} models for comparison")
        
        # Demonstrate comprehensive validation
        print("\n📊 Running Comprehensive Model Validation...")
        validation_results = comprehensive_model_validation(
            model_specs=model_specs,
            data_context=data_context,
            n_bootstrap=50,  # Reduced for demo
            include_gof_tests=True
        )
        
        # Print results following capture-recapture reporting standards
        print("\n" + "="*70)
        print("📋 VALIDATION RESULTS SUMMARY")
        print("="*70)
        
        summary = validation_results['validation_summary']
        print(f"Models fitted successfully: {summary['models_fitted']}")
        print(f"Models failed: {summary['models_failed']}")
        print(f"Validation methods completed: {', '.join(summary['validation_methods_completed'])}")
        
        # Model averaging results (key for capture-recapture)
        if 'model_averaging' in validation_results:
            avg_result = validation_results['model_averaging']
            print(f"\n🎯 MODEL AVERAGING RESULTS (Burnham & Anderson approach)")
            print("-" * 50)
            print(f"Model selection uncertainty: {avg_result.selection_summary['interpretation']}")
            print(f"Weighted lambda estimate: {avg_result.weighted_lambda_mean:.3f}")
            print(f"Lambda uncertainty (SD): {avg_result.weighted_lambda_std:.3f}")
            
            print(f"\n📊 Individual Model Results:")
            for i, model_info in enumerate(avg_result.individual_models):
                support = "✓" if model_info['substantial_support'] else " "
                print(f"  {support} {model_info['model_name']}: "
                      f"AIC={model_info['aic']:.1f}, "
                      f"ΔAIC={model_info['delta_aic']:.1f}, "
                      f"Weight={model_info['weight']:.3f}")
        
        # Bootstrap results
        if 'bootstrap_validation' in validation_results:
            boot_result = validation_results['bootstrap_validation']
            print(f"\n🔢 BOOTSTRAP VALIDATION")
            print("-" * 30)
            print(f"Original estimate: {boot_result.original_estimate:.3f}")
            print(f"Bootstrap mean: {boot_result.bootstrap_mean:.3f}")
            print(f"Bias estimate: {boot_result.bias_estimate:.4f}")
            print(f"95% CI: [{boot_result.confidence_interval_95[0]:.3f}, {boot_result.confidence_interval_95[1]:.3f}]")
            print(f"Bootstrap samples: {boot_result.n_bootstrap_samples}")
        
        # Goodness of fit
        if 'goodness_of_fit' in validation_results:
            gof_results = validation_results['goodness_of_fit']
            print(f"\n🎯 GOODNESS-OF-FIT TESTS")
            print("-" * 30)
            for gof in gof_results:
                fit_status = "✅ GOOD" if gof.model_fits_data else "⚠️  POOR"
                print(f"{fit_status}: {gof.diagnostic_message}")
        
        # Interpretation and recommendations
        print(f"\n💡 INTERPRETATION & RECOMMENDATIONS")
        print("-" * 40)
        
        if 'model_averaging' in validation_results:
            avg_result = validation_results['model_averaging']
            n_substantial = avg_result.selection_summary['n_substantial_support']
            best_weight = avg_result.selection_summary['best_model_weight']
            
            if best_weight > 0.9:
                print("✅ Clear best model identified (weight > 0.9)")
                print("   Recommendation: Use single best model")
            elif n_substantial > 1:
                print("📊 Multiple models have substantial support (ΔAIC ≤ 2)")
                print("   Recommendation: Use model averaging for inference")
                print(f"   {n_substantial} models should be averaged")
            else:
                print("⚠️  Model selection uncertainty is high")
                print("   Recommendation: Consider additional models or larger sample size")
        
        print(f"\n🔬 STATISTICAL VALIDITY ASSESSMENT")
        print("-" * 40)
        
        # Check for common issues
        issues_found = []
        
        if 'goodness_of_fit' in validation_results:
            poor_fit_models = sum(1 for gof in validation_results['goodness_of_fit'] 
                                 if not gof.model_fits_data)
            if poor_fit_models > 0:
                issues_found.append(f"{poor_fit_models} models show poor fit")
        
        if 'bootstrap_validation' in validation_results:
            boot_result = validation_results['bootstrap_validation']
            if abs(boot_result.bias_estimate / boot_result.original_estimate) > 0.1:
                issues_found.append("Substantial bias detected in estimates")
        
        if not issues_found:
            print("✅ No major statistical issues detected")
            print("   Models appear statistically valid for inference")
        else:
            print("⚠️  Potential issues detected:")
            for issue in issues_found:
                print(f"   - {issue}")
        
        print(f"\n📖 CAPTURE-RECAPTURE BEST PRACTICES FOLLOWED:")
        print("-" * 50)
        print("✅ Multiple models compared (avoid single model bias)")
        print("✅ Information-theoretic approach (AIC-based)")
        print("✅ Model averaging when appropriate")
        print("✅ Bootstrap confidence intervals")
        print("✅ Goodness-of-fit assessment")
        print("✅ Systematic model complexity progression")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Validation demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def demonstrate_model_selection_workflow():
    """Demonstrate model selection for prediction on new data."""
    print("\n🎯 Demonstrating Model Selection for Prediction")
    print("=" * 50)
    
    # Create training and validation datasets
    print("Creating training and validation datasets...")
    
    # Training data (larger, for model fitting)
    train_data = create_realistic_capture_data(400, seed=42)
    
    # Validation data (smaller, for prediction assessment)
    val_data = create_realistic_capture_data(100, seed=123)
    
    # Save datasets
    train_file = "temp_train_data.csv"
    val_file = "temp_val_data.csv"
    
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    
    try:
        # Load data contexts
        train_context = pj.load_data(train_file)
        val_context = pj.load_data(val_file)
        
        print(f"✅ Training: {train_context.n_individuals} individuals")
        print(f"✅ Validation: {val_context.n_individuals} individuals")
        
        # Create candidate models
        model_specs = create_model_specs_from_formulas(
            phi_formulas=["~1", "~1 + age_1", "~1 + gender"],
            p_formulas=["~1", "~1 + tier_1"],
            f_formulas=["~1"],
            random_seed_base=9999
        )
        
        print(f"✅ Created {len(model_specs)} candidate models")
        
        # Fit models on training data
        print("\n📊 Fitting models on training data...")
        train_results = fit_models_parallel(
            model_specs=model_specs,
            data_context=train_context,
            n_workers=2
        )
        
        successful_models = [r for r in train_results if r and r.success]
        print(f"✅ {len(successful_models)} models fitted successfully")
        
        # Perform model averaging for final prediction
        print("\n🎯 Performing model averaging for prediction...")
        averaging_result = perform_model_averaging(
            train_results,
            model_specs
        )
        
        # Make predictions on validation data using best model
        print("\n📈 Making predictions on validation data...")
        
        # Find best model (highest AIC weight)
        best_model_idx = averaging_result.substantial_support_models[0]
        best_model_spec = model_specs[best_model_idx]
        best_train_result = successful_models[best_model_idx]
        
        print(f"Best model: {best_model_spec.name}")
        print(f"AIC weight: {averaging_result.model_weights[best_model_idx]:.3f}")
        
        # Calculate predictions on validation data
        model = pj.PradelModel()
        val_design_matrices = model.build_design_matrices(best_model_spec.formula_spec, val_context)
        
        val_predictions = model.predict(
            np.array(best_train_result.parameters),
            val_context,
            val_design_matrices,
            return_individual_predictions=False
        )
        
        # Calculate validation metrics
        val_metrics = model.calculate_validation_metrics(
            np.array(best_train_result.parameters),
            train_context,
            val_context,
            model.build_design_matrices(best_model_spec.formula_spec, train_context),
            val_design_matrices
        )
        
        print(f"\n📋 PREDICTION RESULTS")
        print("-" * 30)
        print(f"Training log-likelihood: {val_metrics['train_log_likelihood']:.2f}")
        print(f"Validation log-likelihood: {val_metrics['val_log_likelihood']:.2f}")
        print(f"Overfitting ratio: {val_metrics['overfitting_ratio']:.3f}")
        
        print(f"\nValidation predictions:")
        print(f"  Lambda mean: {val_predictions.get('lambda_mean', 'N/A'):.3f}")
        print(f"  Phi mean: {val_predictions.get('phi_mean', 'N/A'):.3f}")
        print(f"  P mean: {val_predictions.get('p_mean', 'N/A'):.3f}")
        print(f"  F mean: {val_predictions.get('f_mean', 'N/A'):.3f}")
        
        # Assessment
        if abs(val_metrics['overfitting_ratio']) < 0.05:
            print(f"\n✅ Model generalizes well to new data (low overfitting)")
        elif abs(val_metrics['overfitting_ratio']) < 0.15:
            print(f"\n⚠️  Moderate overfitting detected")
        else:
            print(f"\n❌ High overfitting - model may not generalize well")
        
        print(f"\n💡 WORKFLOW SUMMARY:")
        print("-" * 25)
        print("1. ✅ Split data into training/validation")
        print("2. ✅ Fit multiple models on training data")
        print("3. ✅ Use model averaging for model selection")
        print("4. ✅ Predict on validation data using best model")
        print("5. ✅ Assess overfitting and generalization")
        
        return {
            'averaging_result': averaging_result,
            'validation_metrics': val_metrics,
            'validation_predictions': val_predictions
        }
        
    except Exception as e:
        print(f"❌ Model selection demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        import os
        for f in [train_file, val_file]:
            if os.path.exists(f):
                os.unlink(f)


def main():
    """Run all validation demonstrations."""
    print("🧪 Capture-Recapture Model Validation Demonstrations")
    print("="*70)
    print("Implementing statistical best practices from ecological literature")
    
    start_time = time.time()
    
    # Demo 1: Comprehensive validation
    demo1_results = demonstrate_model_averaging_workflow()
    
    # Demo 2: Model selection for prediction
    demo2_results = demonstrate_model_selection_workflow()
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*70)
    print("🎉 VALIDATION DEMONSTRATIONS COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.1f}s")
    
    if demo1_results and demo2_results:
        print("✅ All validation workflows completed successfully")
        print("\n📚 Key Principles Demonstrated:")
        print("  • Model averaging when no single best model")
        print("  • Information-theoretic model selection (AIC)")
        print("  • Bootstrap confidence intervals")
        print("  • Goodness-of-fit assessment")
        print("  • Overfitting detection with holdout validation")
        print("  • Following capture-recapture statistical standards")
        
        return True
    else:
        print("⚠️  Some validation workflows failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)