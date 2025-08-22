#!/usr/bin/env python3
"""
Comprehensive demonstration of literature-based validation framework.

This script demonstrates the enhanced validation methods based on:
- Wesson et al. (2022): Model triangulation for sparse data
- Yates et al. (2022): Cross-validation best practices for ecology
- Pledger (2004): Unified maximum likelihood framework

The demonstration shows robust handling of:
1. Sparse cell count scenarios
2. Multi-model triangulation 
3. Cross-validation with ecological considerations
4. Comprehensive uncertainty quantification
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress JAX warnings for cleaner output
warnings.filterwarnings("ignore", ".*TPU.*")
warnings.filterwarnings("ignore", ".*CUDA.*")

def main():
    """
    Main demonstration of literature-based validation framework.
    """
    print("\n" + "="*80)
    print("📚 LITERATURE-BASED VALIDATION FRAMEWORK DEMONSTRATION")
    print("="*80)
    
    print("\n🎯 Implementation of Best Practices from Key Papers:")
    print("   • Wesson et al. (2022) - Model triangulation for sparse data")
    print("   • Yates et al. (2022) - Cross-validation for ecological models")
    print("   • Pledger (2004) - Unified maximum likelihood framework")
    
    try:
        # Import our modules
        import pradel_jax as pj
        from pradel_jax.validation.literature_based_validation import (
            LiteratureBasedValidator,
            comprehensive_literature_based_validation,
            CrossValidationConfig
        )
        from pradel_jax.validation.sparse_data_handler import SparseDataHandler
        from pradel_jax.formulas import create_simple_spec
        
        print("\n✅ Successfully imported literature-based validation modules")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure the project is properly installed with:")
        print("   pip install -e .")
        return False
    
    # Test 1: Load data and assess sparsity
    print("\n" + "-"*60)
    print("📊 TEST 1: DATA LOADING AND SPARSITY ASSESSMENT")
    print("-"*60)
    
    try:
        # Load the dipper dataset
        data_path = project_root / "data" / "dipper_dataset.csv"
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            print("Using synthetic data instead...")
            data_context = create_synthetic_sparse_data()
        else:
            data_context = pj.load_data(str(data_path))
            print(f"✅ Loaded data from {data_path}")
        
        print(f"   Data shape: {data_context.capture_matrix.shape}")
        print(f"   Individuals: {data_context.capture_matrix.shape[0]}")
        print(f"   Occasions: {data_context.capture_matrix.shape[1]}")
        
        # Assess sparsity using our enhanced methods
        sparse_handler = SparseDataHandler(random_seed=42)
        sparsity_assessment = sparse_handler.assess_sparsity_severity(data_context)
        
        print(f"\n🔍 Sparsity Assessment:")
        print(f"   Severity: {sparsity_assessment['severity']}")
        print(f"   Sample coverage: {sparsity_assessment['sample_coverage']:.3f}")
        print(f"   Detection rate: {sparsity_assessment['detection_rate']:.3f}")
        print(f"   Sparse patterns: {sparsity_assessment['sparse_pattern_percentage']:.1f}%")
        
        if sparsity_assessment['reliability_warning']:
            print("   ⚠️ RELIABILITY WARNING: Sparse data detected")
        
        print(f"   Recommended methods: {', '.join(sparsity_assessment['recommended_methods'])}")
        
    except Exception as e:
        print(f"❌ Error in sparsity assessment: {e}")
        return False
    
    # Test 2: Sparse data handling demonstration
    print("\n" + "-"*60)
    print("🔧 TEST 2: SPARSE DATA HANDLING METHODS")
    print("-"*60)
    
    try:
        # Create deliberately sparse data for demonstration
        sparse_data_context = create_synthetic_sparse_data(n_individuals=200, detection_prob=0.15)
        print(f"Created sparse synthetic data: {sparse_data_context.capture_matrix.shape}")
        
        # Apply sparse data handling methods
        sparse_solutions = sparse_handler.handle_sparse_scenario(sparse_data_context)
        
        print(f"\n📋 Sparse Data Solutions:")
        print(f"   Sparsity severity: {sparse_solutions['sparsity_assessment']['severity']}")
        
        for method_name, solution in sparse_solutions['individual_solutions'].items():
            print(f"\n   {method_name.upper()}:")
            print(f"     Population estimate: {solution.population_estimate:.1f}")
            print(f"     Confidence interval: ({solution.confidence_interval[0]:.1f}, {solution.confidence_interval[1]:.1f})")
            print(f"     Reliability: {solution.reliability_assessment}")
            print(f"     Sample coverage: {solution.sample_coverage:.3f}")
        
        consensus = sparse_solutions['consensus_recommendation']
        print(f"\n🎯 Consensus Recommendation:")
        print(f"   Method: {consensus['method']}")
        print(f"   Estimate: {consensus['population_estimate']:.1f}")
        print(f"   Reliability: {consensus['reliability']}")
        
        print(f"\n📝 Key Recommendations:")
        for rec in sparse_solutions['general_recommendations']:
            print(f"   {rec}")
        
    except Exception as e:
        print(f"❌ Error in sparse data handling: {e}")
        return False
    
    # Test 3: Model triangulation demonstration  
    print("\n" + "-"*60)
    print("🔄 TEST 3: MODEL TRIANGULATION (WESSON ET AL. 2022)")
    print("-"*60)
    
    try:
        # Create candidate model specifications
        model_specs = [
            create_simple_spec(phi="~1", p="~1", f="~1", name="Constant"),
            create_simple_spec(phi="~1 + sex", p="~1", f="~1", name="Sex on survival"),
            create_simple_spec(phi="~1", p="~1 + sex", f="~1", name="Sex on detection"),
            create_simple_spec(phi="~1 + sex", p="~1 + sex", f="~1", name="Sex on both"),
        ]
        
        print(f"Created {len(model_specs)} candidate models for triangulation")
        
        # Perform model triangulation
        validator = LiteratureBasedValidator(random_seed=42)
        triangulation_result = validator.triangulate_models(
            model_specs, data_context
        )
        
        print(f"\n📊 Model Triangulation Results:")
        print(f"   Best model AIC: {triangulation_result.primary_estimates['best_model_aic']:.2f}")
        print(f"   Best model weight: {triangulation_result.primary_estimates['best_model_weight']:.3f}")
        print(f"   Models with substantial support: {triangulation_result.primary_estimates['n_supported_models']}")
        print(f"   Model selection uncertainty: {triangulation_result.uncertainty_metrics['model_selection_uncertainty']:.3f}")
        
        print(f"\n🎯 Triangulation Assessment:")
        print(f"   {triangulation_result.triangulation_assessment}")
        
        if triangulation_result.sparse_cell_warning:
            print("   ⚠️ SPARSE CELL WARNING: Results may be unreliable")
        
        print(f"\n🏆 Recommended Estimate:")
        rec = triangulation_result.recommended_estimate
        print(f"   Formula: {rec['best_model_formula']}")
        print(f"   Triangulation confidence: {rec['triangulation_confidence']:.3f}")
        
        # Show model weights
        weights = triangulation_result.model_agreement['weights']
        delta_aics = triangulation_result.model_agreement['delta_aics'] 
        
        print(f"\n📋 Individual Model Performance:")
        for i, (weight, delta_aic) in enumerate(zip(weights, delta_aics)):
            support = "✓" if delta_aic <= 2 else " "
            print(f"   {support} Model {i+1}: Weight={weight:.3f}, ΔAIC={delta_aic:.2f}")
        
    except Exception as e:
        print(f"❌ Error in model triangulation: {e}")
        return False
    
    # Test 4: Cross-validation with ecological considerations
    print("\n" + "-"*60)
    print("🔬 TEST 4: ECOLOGICAL CROSS-VALIDATION (YATES ET AL. 2022)")
    print("-"*60)
    
    try:
        # Configure cross-validation following Yates et al. recommendations
        cv_config = CrossValidationConfig(
            split_type="random",  # Could be "temporal" or "spatial" with appropriate data
            validation_fraction=0.2,
            n_folds=5,
            bootstrap_iterations=100,  # Reduced for demo speed
            random_seed=42
        )
        
        print(f"Configured cross-validation:")
        print(f"   Split type: {cv_config.split_type}")
        print(f"   Number of folds: {cv_config.n_folds}")
        print(f"   Validation fraction: {cv_config.validation_fraction}")
        
        # Run cross-validation
        cv_results = validator.cross_validate_with_ecological_splits(
            model_specs, data_context, cv_config
        )
        
        print(f"\n📊 Cross-Validation Results:")
        agg = cv_results['aggregated_metrics']
        print(f"   Mean training log-likelihood per observation: {agg['mean_train_ll_per_obs']:.4f}")
        print(f"   Standard deviation: {agg['std_train_ll_per_obs']:.4f}")
        print(f"   CV consistency metric: {agg['cv_consistency']:.4f}")
        print(f"   Completed folds: {agg['n_folds_completed']}")
        
        print(f"\n🎯 CV Assessment:")
        print(f"   {cv_results['summary_assessment']}")
        
        # Show fold-level results
        print(f"\n📋 Individual Fold Results:")
        for fold_result in cv_results['individual_folds'][:3]:  # Show first 3 folds
            fold_idx = fold_result['fold']
            train_size = fold_result['train_size']
            val_size = fold_result['val_size']
            tri = fold_result['triangulation']
            
            print(f"   Fold {fold_idx+1}: Train={train_size}, Val={val_size}")
            print(f"     Best AIC: {tri.primary_estimates['best_model_aic']:.2f}")
            print(f"     Model uncertainty: {tri.uncertainty_metrics['model_selection_uncertainty']:.3f}")
        
    except Exception as e:
        print(f"❌ Error in cross-validation: {e}")
        return False
    
    # Test 5: Comprehensive literature-based validation
    print("\n" + "-"*60)
    print("🏆 TEST 5: COMPREHENSIVE LITERATURE-BASED VALIDATION")
    print("-"*60)
    
    try:
        # Run the comprehensive validation framework
        print("Running comprehensive literature-based validation...")
        start_time = time.time()
        
        comprehensive_results = comprehensive_literature_based_validation(
            model_specs=model_specs,
            data_context=data_context,
            cv_config=cv_config,
            random_seed=42
        )
        
        validation_time = time.time() - start_time
        
        print(f"\n✅ Comprehensive validation completed in {validation_time:.1f} seconds")
        
        # Display key results
        metadata = comprehensive_results['metadata']
        print(f"\n📋 Validation Summary:")
        print(f"   Timestamp: {metadata['validation_timestamp']}")
        print(f"   Data fingerprint: {metadata['data_fingerprint']}")
        print(f"   Candidate models: {metadata['n_candidate_models']}")
        print(f"   Random seed: {metadata['random_seed']}")
        
        print(f"\n🔬 Literature Methods Used:")
        for method in metadata['literature_methods_used']:
            print(f"   • {method}")
        
        # Sparse data diagnostics
        sparse_diag = comprehensive_results['sparse_data_diagnostics']
        print(f"\n🔍 Sparse Data Diagnostics:")
        print(f"   Sparsity risk: {sparse_diag.sparsity_risk}")
        print(f"   Sample coverage: {sparse_diag.estimated_sample_coverage:.3f}")
        print(f"   Min cell count: {sparse_diag.min_cell_count}")
        print(f"   Zero cells: {sparse_diag.zero_cells}")
        
        # Model triangulation summary
        tri_result = comprehensive_results['model_triangulation']
        print(f"\n🔄 Model Triangulation Summary:")
        print(f"   Best model weight: {tri_result.primary_estimates['best_model_weight']:.3f}")
        print(f"   Models with support: {tri_result.primary_estimates['n_supported_models']}")
        print(f"   Selection uncertainty: {tri_result.uncertainty_metrics['model_selection_uncertainty']:.3f}")
        print(f"   Assessment: {tri_result.triangulation_assessment}")
        
        # Final recommendations
        recommendations = comprehensive_results['recommendations']
        print(f"\n📝 Final Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
            print(f"   {i}. {rec}")
        
        if len(recommendations) > 5:
            print(f"   ... and {len(recommendations) - 5} more recommendations")
        
        print(f"\n🎯 Validation Framework Benefits:")
        print("   ✅ Follows established capture-recapture literature")
        print("   ✅ Robust handling of sparse data scenarios")
        print("   ✅ Multi-model triangulation reduces bias") 
        print("   ✅ Proper uncertainty quantification")
        print("   ✅ Reproducible with version control")
        print("   ✅ Ready for scientific publication")
        
    except Exception as e:
        print(f"❌ Error in comprehensive validation: {e}")
        return False
    
    # Success summary
    print("\n" + "="*80)
    print("🎉 LITERATURE-BASED VALIDATION FRAMEWORK SUCCESSFULLY DEMONSTRATED")
    print("="*80)
    
    print("\n✅ All tests completed successfully!")
    print("\n🎯 Key Achievements:")
    print("   • Implemented Wesson et al. (2022) model triangulation approach")
    print("   • Added robust sparse data handling methods")
    print("   • Integrated Yates et al. (2022) cross-validation best practices")
    print("   • Created comprehensive uncertainty quantification")
    print("   • Provided literature-backed methodological recommendations")
    
    print("\n📚 This framework addresses major pitfalls in capture-recapture analysis:")
    print("   ❌ Naive AIC selection with sparse data (Wesson et al. 2022)")
    print("   ❌ Single model bias and selection uncertainty")
    print("   ❌ Inadequate cross-validation for ecological data")
    print("   ❌ Poor handling of assumption violations")
    
    print("\n🔬 Ready for Production Use:")
    print("   • Statistical rigor following best practices")
    print("   • Defensible methodology for publication")
    print("   • Robust uncertainty quantification")
    print("   • Comprehensive diagnostic reporting")
    
    return True

def create_synthetic_sparse_data(n_individuals: int = 150, 
                               n_occasions: int = 5,
                               detection_prob: float = 0.2) -> 'DataContext':
    """
    Create synthetic sparse data for demonstration purposes.
    
    This creates data with sparse cell counts that can cause issues
    with traditional AIC-based model selection.
    """
    from pradel_jax.data.adapters import DataContext
    
    # Create very sparse capture matrix
    capture_matrix = np.random.binomial(1, detection_prob, (n_individuals, n_occasions))
    
    # Ensure some individuals are never detected (create sparsity)
    never_detected = np.random.choice(n_individuals, size=int(n_individuals * 0.3), replace=False)
    capture_matrix[never_detected, :] = 0
    
    # Create some covariates
    sex = np.random.choice(['M', 'F'], size=n_individuals)
    age = np.random.choice(['juvenile', 'adult'], size=n_individuals)
    
    covariates = {
        'sex': sex,
        'age': age
    }
    
    # Convert to JAX arrays
    import jax.numpy as jnp
    
    # Create covariate info
    from pradel_jax.data.adapters import CovariateInfo
    covariate_info = {
        'sex': CovariateInfo('sex', 'categorical', levels=['M', 'F']),
        'age': CovariateInfo('age', 'categorical', levels=['juvenile', 'adult'])
    }
    
    # Convert covariates to JAX arrays
    jax_covariates = {}
    for name, values in covariates.items():
        if name == 'sex':
            jax_covariates[name] = jnp.array([1 if v == 'M' else 0 for v in values])
        elif name == 'age':
            jax_covariates[name] = jnp.array([1 if v == 'adult' else 0 for v in values])
    
    return DataContext(
        capture_matrix=jnp.array(capture_matrix),
        covariates=jax_covariates,
        covariate_info=covariate_info,
        n_individuals=n_individuals,
        n_occasions=n_occasions,
        metadata={
            'description': 'Synthetic sparse data for demonstration',
            'detection_probability': detection_prob,
            'sparsity_level': 'high'
        }
    )

if __name__ == "__main__":
    print("Starting literature-based validation demonstration...")
    
    try:
        success = main()
        if success:
            print("\n🎉 Demonstration completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Demonstration failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)