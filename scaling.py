# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

from guti.scaling_utils import (
    plot_parameter_sweep_spectra,
    plot_first_singular_value_vs_parameter,
    plot_bitrate_vs_parameter
)
from guti.parameters import Parameters

modality_name = "eeg_analytical_2"
param_key = "num_brain_grid_points"
constant_params = Parameters(num_sensors=110*110)
# param_key = "num_sensors"
# constant_params = Parameters(num_brain_grid_points=34*34*34)

# %%
plot_parameter_sweep_spectra(
    modality_name=modality_name,
    param_key=param_key,
    constant_params=constant_params,
    ylim=(1e-5, 10)
)

# %%
plot_first_singular_value_vs_parameter(
    modality_name=modality_name,
    param_key=param_key,
    constant_params=constant_params
)

# %%
plot_bitrate_vs_parameter(
    modality_name=modality_name,
    param_key=param_key,
    constant_params=constant_params
)

# %%
