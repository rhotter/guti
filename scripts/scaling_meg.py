# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

from guti.scaling_utils import (
    plot_parameter_sweep_spectra,
    plot_first_singular_value_vs_parameter,
    plot_bitrate_vs_parameter,
)
from guti.parameters import Parameters

# %% [markdown]
# ## MEG OPM - Sweep num_sensors (fixed source_spacing=5.0mm)

# %%
modality_name = "meg_opm"
param_key = "num_sensors"
constant_params = Parameters(source_spacing_mm=5.0)

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

# %% [markdown]
# ## MEG OPM - Sweep source_spacing_mm (fixed num_sensors=500)

# %%
modality_name = "meg_opm"
param_key = "source_spacing_mm"
constant_params = Parameters(num_sensors=500)

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
