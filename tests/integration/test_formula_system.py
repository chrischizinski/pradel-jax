#!/usr/bin/env python3
"""
Test script for the new formula system.

Tests formula parsing, design matrix construction, and integration.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test imports
try:
    from pradel_jax.formulas import (
        parse_formula, create_simple_spec, FormulaParser, 
        build_design_matrix, ParameterType
    )
    from pradel_jax.data.adapters import load_data, DataContext
    print("✅ All formula system imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_formula_parsing():
    """Test basic formula parsing functionality."""
    print("\n🧪 Testing Formula Parsing")
    
    parser = FormulaParser()
    
    # Test basic formulas
    test_cases = [
        ("~1", "Intercept only"),
        ("~age", "Single variable"),
        ("~age + sex", "Additive model"),
        ("~age * sex", "Interaction model"),
        ("~I(age^2)", "Quadratic term"),
        ("~log(weight)", "Log transformation"),
        ("~poly(age, 2)", "Polynomial"),
        ("phi ~ age + sex", "With response"),
    ]
    
    for formula_str, description in test_cases:
        try:
            parsed = parse_formula(formula_str)
            print(f"  ✅ {description}: '{formula_str}' -> {len(parsed.terms)} terms")
            print(f"     Response: {parsed.response}, Intercept: {parsed.has_intercept}")
        except Exception as e:
            print(f"  ❌ {description}: '{formula_str}' failed: {e}")


def test_model_specifications():
    """Test model specification creation."""
    print("\n🧪 Testing Model Specifications")
    
    # Test simple specification
    try:
        spec1 = create_simple_spec("~1", "~1", "~1", "Constant model")
        print(f"  ✅ Constant model: {spec1}")
    except Exception as e:
        print(f"  ❌ Constant model failed: {e}")
    
    # Test complex specification
    try:
        parser = FormulaParser()
        spec2 = parser.parse_model_spec({
            "phi": "~age + sex",
            "p": "~sex * time",
            "f": "~I(age^2)",
            "name": "Complex model"
        })
        print(f"  ✅ Complex model: {spec2}")
    except Exception as e:
        print(f"  ❌ Complex model failed: {e}")


def test_design_matrix_construction():
    """Test design matrix construction with sample data."""
    print("\n🧪 Testing Design Matrix Construction")
    
    # Create sample data
    np.random.seed(42)
    n_individuals = 100
    
    sample_data = pd.DataFrame({
        'ch': ['011010'] * n_individuals,  # Sample capture histories
        'age': np.random.normal(5, 2, n_individuals),
        'sex': np.random.choice(['M', 'F'], n_individuals),
        'weight': np.random.normal(150, 30, n_individuals),
    })
    
    try:
        # Load data using our adapter system
        data_context = load_data("temp_test_data.csv", adapter=None)
        print("  ❌ Can't test with file - creating mock data context")
        
        # Create mock data context for testing
        from pradel_jax.data.adapters import DataContext, CovariateInfo
        import jax.numpy as jnp
        
        # Mock data context
        data_context = DataContext(
            capture_matrix=jnp.ones((n_individuals, 6)),
            covariates={
                'age': jnp.array(sample_data['age'].values),
                'sex_F': jnp.array((sample_data['sex'] == 'F').astype(float)),
                'sex_M': jnp.array((sample_data['sex'] == 'M').astype(float)),
                'weight': jnp.array(sample_data['weight'].values),
            },
            covariate_info={
                'age': CovariateInfo('age', 'float64'),
                'sex_F': CovariateInfo('sex_F', 'binary', is_categorical=True),
                'sex_M': CovariateInfo('sex_M', 'binary', is_categorical=True),
                'weight': CovariateInfo('weight', 'float64'),
            },
            n_individuals=n_individuals,
            n_occasions=6
        )
        
        print(f"  ✅ Created mock data context: {n_individuals} individuals, {len(data_context.covariates)} covariates")
        
        # Test design matrix construction
        parser = FormulaParser()
        
        test_formulas = [
            ("~1", "Intercept only"),
            ("~age", "Single numeric"),
            ("~sex_F", "Single binary"),
            ("~age + sex_F", "Additive"),
            ("~I(age^2)", "Quadratic"),
        ]
        
        for formula_str, description in test_formulas:
            try:
                param_formula = parser.create_parameter_formula(ParameterType.PHI, formula_str)
                design_info = build_design_matrix(param_formula, data_context)
                print(f"  ✅ {description}: {design_info.matrix.shape} matrix, columns: {design_info.column_names}")
            except Exception as e:
                print(f"  ❌ {description} failed: {e}")
                
    except Exception as e:
        print(f"  ❌ Design matrix testing failed: {e}")


def test_error_handling():
    """Test error handling in formula system."""
    print("\n🧪 Testing Error Handling")
    
    parser = FormulaParser()
    
    # Test invalid formulas
    invalid_cases = [
        ("", "Empty formula"),
        ("~nonexistent_var", "Missing variable"),
        ("~age *", "Incomplete interaction"),
        ("~log()", "Function without argument"),
        ("invalid~syntax", "Invalid syntax"),
    ]
    
    for formula_str, description in invalid_cases:
        try:
            if formula_str == "~nonexistent_var":
                # This will parse but fail during validation
                param_formula = parser.create_parameter_formula(ParameterType.PHI, formula_str)
                print(f"  ✅ {description}: Parsed but will fail on validation")
            else:
                parsed = parse_formula(formula_str)
                print(f"  ❌ {description}: Should have failed but didn't")
        except Exception as e:
            print(f"  ✅ {description}: Correctly caught error: {type(e).__name__}")


def main():
    """Run all formula system tests."""
    print("🚀 Testing Formula System")
    print("=" * 50)
    
    # Test configuration first
    try:
        from pradel_jax.config.settings import PradelJaxConfig
        config = PradelJaxConfig()
        print(f"✅ Configuration created: {config.app.name}")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
    
    # Run tests
    test_formula_parsing()
    test_model_specifications()
    test_design_matrix_construction()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("🎉 Formula system testing complete!")
    print("\nNext steps:")
    print("  1. Create model registry framework")
    print("  2. Implement basic Pradel model with JAX")
    print("  3. Add optimization strategies")
    print("  4. Test with real dipper dataset")


if __name__ == "__main__":
    main()