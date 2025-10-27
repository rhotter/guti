# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

from guti.scaling_utils import (
    plot_parameter_sweep_spectra,
    plot_first_singular_value_vs_parameter,
    plot_bitrate_vs_parameter,
    plot_bitrate_vs_snr
)
from guti.parameters import Parameters
import numpy as np

modality_name = "us_free_field_analytical_n_sources_sweep_50khz"
param_key = "num_sensors"
constant_params = Parameters(n_sensors = 3028)

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
    constant_params=constant_params,
    snr=2000
)

# %%
plot_bitrate_vs_snr(
    modality_name=modality_name,
    param_key=param_key,
    param_value=5.0,
    snr_values=np.logspace(-1, 6, 50),
    constant_params=constant_params
)