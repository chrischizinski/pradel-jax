#!/usr/bin/env python3
"""
Fix the Pradel likelihood gradient computation issue.
"""

import sys
sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tempfile
import os

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser  
from pradel_jax.formulas.spec import ParameterType, FormulaSpec

def test_current_issue():
    """Test to confirm the current gradient computation issue."""
    print("Testing Current Gradient Issue")
    print("=" * 40)
    
    # Generate simple data
    np.random.seed(42)
    encounter_data = np.zeros((20, 5), dtype=int)
    
    for i in range(20):
        alive = True
        for t in range(5):
            if alive:
                if np.random.binomial(1, 0.6):
                    encounter_data[i, t] = 1
                if t < 4:
                    alive = np.random.binomial(1, 0.75)
    
    # Create data context
    df_data = []
    for i, history in enumerate(encounter_data):
        ch = ''.join(map(str, history))
        df_data.append({'individual_id': i, 'ch': ch})
    
    df = pd.DataFrame(df_data)
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
    
    print(f"ISSUE: vectorized function only uses phi[0], p[0], f[0]")
    print(f"This breaks gradients for intercept parameters!")
    
    return data_context, design_matrices

if __name__ == "__main__":
    data_context, design_matrices = test_current_issue()
