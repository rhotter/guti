from guti.parameters import Parameters
from guti.modalities.fnirs_analytical.modality import fNIRSAnalytical
from guti.scaling_utils import show_sweep_results
from copy import deepcopy

# param_name = "grid_resolution_mm"
param_name = "num_sensors"
param_values = [600, 800]

default_params = Parameters(grid_resolution_mm=6)

for i, value in enumerate(param_values):
    print(f"-------------------------------- Running {param_name} = {value} ({i+1}/{len(param_values)}) --------------------------------")
    params = deepcopy(default_params)
    setattr(params, param_name, value)
    modality = fNIRSAnalytical(params=params)
    modality.run(save_results=True)

# After running the sweep, visualize the results:
show_sweep_results(
    modality_name=modality.name,
    param_key=param_name,
    constant_params=default_params
)