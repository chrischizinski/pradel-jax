"""
Thin Python bridge for fitting a single Pradel mark-recapture model.

Called from R via reticulate::source_python(). Kept deliberately dumb: it
takes a prepared capture-history CSV (already built by the R side) and one
candidate formula set, fits it with pradel-jax, and returns plain
Python types (no jax/numpy arrays) so reticulate can convert the result
without needing numpy<->R type bridging.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pradel_jax as pj  # noqa: E402


def fit_pradel_model(csv_path: str, phi_formula: str, p_formula: str, f_formula: str) -> dict:
    """Fit one Pradel model and return a JSON-safe results dict.

    The prepared CSV must already contain: individual_id, ch, and any
    covariates referenced by phi_formula/p_formula/f_formula.
    """
    t0 = time.time()
    try:
        data_context = pj.load_data(csv_path)
        model = pj.PradelModel()
        formula_spec = pj.create_formula_spec(phi=phi_formula, p=p_formula, f=f_formula)
        result = pj.fit_model(model=model, formula=formula_spec, data=data_context)
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        parameter_names = model.get_parameter_names(design_matrices)

        parameters = [float(x) for x in result.parameters]
        parameter_se = (
            [float(x) for x in result.parameter_se]
            if result.parameter_se is not None
            else None
        )

        return dict(
            success=bool(result.success),
            log_likelihood=float(result.log_likelihood),
            aic=float(result.aic),
            n_parameters=int(result.n_parameters),
            fit_time=time.time() - t0,
            optimizer_used=str(result.optimizer_used),
            parameter_names=list(parameter_names),
            parameters=parameters,
            parameter_se=parameter_se,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - surface any fit failure to R, don't crash the loop
        return dict(
            success=False,
            log_likelihood=None,
            aic=None,
            n_parameters=None,
            fit_time=time.time() - t0,
            optimizer_used=None,
            parameter_names=None,
            parameters=None,
            parameter_se=None,
            error=str(exc),
        )
