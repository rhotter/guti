# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

# %%
from guti.data_utils import list_svd_variants
import matplotlib.pyplot as plt
import numpy as np

# modality_name = "fnirs_analytical_cw"
modality_name = "us_analytical"

# %%
# subdir = "grid_sweep"
# param_key = "grid_resolution_mm"
# subdir = "us_free_field_analytical"
# subdir = "us_free_field_analytical_n_sources_sweep"
subdir = "us_free_field_analytical_n_sources_sweep_300khz"
param_key = "num_brain_grid_points"

variants = list_svd_variants(modality_name, subdir=subdir)
for k, v in variants.items():
    print(f"  {k}: {v['params']}")

# %%
# Set the parameter key you want to analyze

plt.figure(figsize=(10, 6))
sorted_variants = sorted(
    variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
)

# Create colormap based on parameter values
param_values = [getattr(v["params"], param_key) for k,v in sorted_variants]
min_val, max_val = min(param_values), max(param_values)
colors = plt.cm.viridis((np.array(param_values) - min_val) / (max_val - min_val))

for (k, v), color in zip(sorted_variants, colors):
    params = v["params"]
    s = v["s"]
    param_value = getattr(params, param_key)

    plt.plot(
        np.arange(1, len(s) + 1),
        s / s[0],
        label=f"{param_key}={param_value}",
        color=color
    )
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Singular Value Index")
plt.ylabel("Singular Value")
plt.title(f"{param_key} Scaling - {modality_name}")
plt.ylim(1e-5, 1)
plt.show()

# %%

from guti.core import get_bitrate, noise_floor_heuristic

# Plot bitrate vs parameter value
plt.figure(figsize=(10, 6))

param_values = []
bitrates = []

n_sensors = 8000


for k, v in sorted_variants:
    params = v["params"]
    s = v["s"]
    param_value = getattr(params, param_key)
    noise_level = noise_floor_heuristic(s, heuristic="power", n_detectors=n_sensors)
    
    # Get time resolution if available, otherwise use default 1.0
    # time_res = params.time_resolution if params.time_resolution is not None else 1.0
    time_res = 1.0
    
    bitrate = get_bitrate(s, noise_level, time_resolution=time_res, n_detectors=n_sensors)
    
    param_values.append(param_value)
    bitrates.append(bitrate)

plt.plot(param_values, bitrates, 'o-')
plt.xlabel(param_key)
plt.ylabel('Bitrate (bits/s)')
plt.title(f'Bitrate vs {param_key} - {modality_name}')
plt.grid(True)
plt.show()

# %%
