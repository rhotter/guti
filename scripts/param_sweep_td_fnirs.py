"""Sweep TD-fNIRS over sensor count to produce SVD variants for the web export.

Mirrors scripts/param_sweep.py (CW fNIRS): same sensor counts and grid so the
two fNIRS modalities are directly comparable in the web charts. Saved variants
are picked up by scripts/export_svd_json.py (modality key "td_fnirs").

Note: TD multiplies the Jacobian rows by n_time_gates, so the larger sensor
counts are sizeable SVDs (slow on CPU, fine on GPU).
"""

from copy import deepcopy

from guti.parameters import Parameters
from guti.modalities.td_fnirs.modality import TDfNIRSAnalytical
from guti.scaling_utils import show_sweep_results

param_name = "num_sensors"
param_values = [50, 100, 200, 400, 600, 800]

default_params = Parameters(grid_resolution_mm=6)

for i, value in enumerate(param_values):
    print(
        f"-------------------------------- Running {param_name} = {value} "
        f"({i+1}/{len(param_values)}) --------------------------------"
    )
    params = deepcopy(default_params)
    setattr(params, param_name, value)
    modality = TDfNIRSAnalytical(params=params)
    modality.run(save_results=True)

# After running the sweep, visualize the results:
show_sweep_results(
    modality_name=modality.name,
    param_key=param_name,
    constant_params=default_params,
)
