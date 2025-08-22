#!/usr/bin/env python3
"""
Test script to verify DataContext serialization for parallel optimization.

This test creates a DataContext, serializes it, deserializes it, and verifies
that all components are preserved correctly.
"""

import numpy as np
import pandas as pd
import pickle
import tempfile
from pathlib import Path

# Set PYTHONPATH to include our package
import sys
sys.path.insert(0, str(Path(__file__).parent))

def test_datacontext_serialization():
    """Test DataContext serialization/deserialization."""
    print("Testing DataContext serialization...")
    
    # Import pradel-jax components
    import pradel_jax as pj
    from pradel_jax.data.adapters import DataContext, CovariateInfo
    
    # Create test data
    test_data = pd.DataFrame({
        'ch': ['0110', '1010', '1110', '0101'],
        'sex': ['M', 'F', 'M', 'F'],
        'age': [1.2, 2.1, 1.8, 2.5]
    })
    
    print(f"Created test data with {len(test_data)} individuals")
    
    # Process data into DataContext using adapter directly
    from pradel_jax.data.adapters import RMarkFormatAdapter
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    print(f"DataContext created:")
    print(f"  - Capture matrix shape: {data_context.capture_matrix.shape}")
    print(f"  - Covariates: {list(data_context.covariates.keys())}")
    print(f"  - Individuals: {data_context.n_individuals}")
    print(f"  - Occasions: {data_context.n_occasions}")
    
    # Test serialization to dictionary
    print("\n1. Testing to_dict() serialization...")
    try:
        data_dict = data_context.to_dict()
        print("✅ to_dict() successful")
        
        # Check that all arrays are numpy (not JAX)
        assert isinstance(data_dict['capture_matrix'], np.ndarray)
        print("✅ capture_matrix converted to numpy")
        
        for name, value in data_dict['covariates'].items():
            if hasattr(value, 'dtype'):  # Check if it's an array-like object
                assert isinstance(value, np.ndarray)
        print("✅ All covariate arrays converted to numpy")
        
        # Check covariate info
        assert isinstance(data_dict['covariate_info'], dict)
        for name, info_dict in data_dict['covariate_info'].items():
            assert isinstance(info_dict, dict)
            assert 'name' in info_dict
            assert 'dtype' in info_dict
        print("✅ CovariateInfo objects converted to dicts")
        
    except Exception as e:
        print(f"❌ to_dict() failed: {e}")
        return False
    
    # Test deserialization from dictionary
    print("\n2. Testing from_dict() deserialization...")
    try:
        restored_context = DataContext.from_dict(data_dict)
        print("✅ from_dict() successful")
        
        # Verify data integrity
        assert restored_context.n_individuals == data_context.n_individuals
        assert restored_context.n_occasions == data_context.n_occasions
        print("✅ Basic properties preserved")
        
        # Check capture matrix
        import jax.numpy as jnp
        assert isinstance(restored_context.capture_matrix, jnp.ndarray)
        np.testing.assert_array_equal(
            np.array(restored_context.capture_matrix),
            np.array(data_context.capture_matrix)
        )
        print("✅ Capture matrix preserved and converted back to JAX")
        
        # Check covariates
        assert set(restored_context.covariates.keys()) == set(data_context.covariates.keys())
        for name in restored_context.covariates:
            if hasattr(restored_context.covariates[name], 'dtype'):  # Array-like
                assert isinstance(restored_context.covariates[name], jnp.ndarray)
                np.testing.assert_array_equal(
                    np.array(restored_context.covariates[name]),
                    np.array(data_context.covariates[name])
                )
        print("✅ Covariates preserved and converted back to JAX")
        
        # Check covariate info
        assert set(restored_context.covariate_info.keys()) == set(data_context.covariate_info.keys())
        for name in restored_context.covariate_info:
            original_info = data_context.covariate_info[name]
            restored_info = restored_context.covariate_info[name]
            assert isinstance(restored_info, CovariateInfo)
            assert restored_info.name == original_info.name
            assert restored_info.dtype == original_info.dtype
            assert restored_info.is_time_varying == original_info.is_time_varying
            assert restored_info.is_categorical == original_info.is_categorical
        print("✅ CovariateInfo objects restored correctly")
        
    except Exception as e:
        print(f"❌ from_dict() failed: {e}")
        return False
    
    # Test pickle serialization (what multiprocessing uses)
    print("\n3. Testing pickle serialization...")
    try:
        with tempfile.NamedTemporaryFile() as tmp_file:
            # Serialize the dictionary (not the DataContext directly)
            with open(tmp_file.name, 'wb') as f:
                pickle.dump(data_dict, f)
            print("✅ Dictionary pickled successfully")
            
            # Deserialize and reconstruct
            with open(tmp_file.name, 'rb') as f:
                loaded_dict = pickle.load(f)
            
            final_context = DataContext.from_dict(loaded_dict)
            print("✅ Dictionary unpickled and DataContext reconstructed")
            
            # Final verification
            np.testing.assert_array_equal(
                np.array(final_context.capture_matrix),
                np.array(data_context.capture_matrix)
            )
            print("✅ Final data integrity check passed")
            
    except Exception as e:
        print(f"❌ Pickle serialization failed: {e}")
        return False
    
    print("\n🎉 All serialization tests passed!")
    return True


def test_parallel_worker_simulation():
    """Simulate what happens in a parallel worker process."""
    print("\n" + "="*50)
    print("Testing parallel worker simulation...")
    
    # Import components
    import pradel_jax as pj
    from pradel_jax.optimization.parallel import ParallelModelSpec
    
    # Create test data and context
    test_data = pd.DataFrame({
        'ch': ['0110', '1010', '1110', '0101', '1100', '0011'],
        'sex': ['M', 'F', 'M', 'F', 'M', 'F'],
        'age': [1.2, 2.1, 1.8, 2.5, 1.5, 2.0]
    })
    
    # Process data into DataContext using adapter directly
    from pradel_jax.data.adapters import RMarkFormatAdapter
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    # Create model spec
    formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1")
    model_spec = ParallelModelSpec(
        name="test_model",
        formula_spec=formula_spec,
        index=0
    )
    
    print(f"Created model spec: {model_spec.name}")
    
    # Simulate what happens in worker
    try:
        print("\n1. Serializing DataContext...")
        data_dict = data_context.to_dict()
        print("✅ DataContext serialized")
        
        print("2. Simulating process boundary (pickle round-trip)...")
        serialized_data = pickle.dumps(data_dict)
        deserialized_dict = pickle.loads(serialized_data)
        print("✅ Data survived process boundary")
        
        print("3. Reconstructing DataContext in 'worker'...")
        worker_context = pj.data.adapters.DataContext.from_dict(deserialized_dict)
        print("✅ DataContext reconstructed in worker")
        
        print("4. Building design matrices...")
        model = pj.PradelModel()
        design_matrices = model.build_design_matrices(formula_spec, worker_context)
        print("✅ Design matrices built successfully")
        
        print("5. Testing model operations...")
        initial_params = model.get_initial_parameters(worker_context, design_matrices)
        print(f"✅ Initial parameters: {len(initial_params)} parameters")
        
        # Test likelihood computation
        ll = model.log_likelihood(initial_params, worker_context, design_matrices)
        print(f"✅ Log-likelihood computed: {ll:.3f}")
        
    except Exception as e:
        print(f"❌ Worker simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🚀 Parallel worker simulation passed!")
    return True


if __name__ == "__main__":
    print("DataContext Serialization Test Suite")
    print("="*50)
    
    # Test 1: Basic serialization
    success1 = test_datacontext_serialization()
    
    # Test 2: Parallel worker simulation  
    success2 = test_parallel_worker_simulation()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Basic serialization: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Worker simulation:   {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED - DataContext serialization is working!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Need to fix serialization issues")
        exit(1)