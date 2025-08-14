#!/usr/bin/env python3
"""
Test the new pradel-jax architecture with dipper data.

This example demonstrates the improved data handling and error reporting.
"""

import sys
sys.path.insert(0, '.')

import pradel_jax as pj
from pradel_jax.utils.logging import setup_logging

def main():
    """Test the new architecture."""
    # Setup logging
    setup_logging(level="INFO", console=True)
    logger = pj.get_config().logging
    
    print("🧪 Testing new pradel-jax architecture")
    print("=" * 50)
    
    # Test 1: Load dipper data
    print("\n1. Testing data loading...")
    try:
        data_context = pj.load_data("data/test_datasets/dipper_dataset.csv")
        print(f"✅ Successfully loaded data:")
        print(f"   - {data_context.n_individuals} individuals")
        print(f"   - {data_context.n_occasions} occasions") 
        print(f"   - {len(data_context.covariates)} covariates")
        print(f"   - Covariates: {list(data_context.covariates.keys())}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Test 2: Test configuration system
    print("\n2. Testing configuration system...")
    config = pj.get_config()
    print(f"✅ Configuration loaded:")
    print(f"   - Default format: {config.data.default_format}")
    print(f"   - Optimization strategy: {config.optimization.default_strategy}")
    print(f"   - Log level: {config.logging.level}")
    
    # Test 3: Test error handling
    print("\n3. Testing error handling...")
    try:
        pj.load_data("nonexistent_file.csv")
    except pj.DataFormatError as e:
        print(f"✅ Error handling works:")
        print(f"   Error: {str(e).split(chr(10))[0]}")  # First line only
        print(f"   Suggestions provided: {len(e.suggestions)}")
    
    # Test 4: Test validation
    print("\n4. Testing data validation...")
    from pradel_jax.utils.validation import validate_capture_matrix
    try:
        validate_capture_matrix(data_context.capture_matrix)
        print("✅ Capture matrix validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Architecture test completed!")
    print("\nNext steps:")
    print("- Implement formula system")
    print("- Create model registry")
    print("- Add optimization framework")


if __name__ == "__main__":
    main()