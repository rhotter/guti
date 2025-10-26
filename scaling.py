# %%
from guti.notebook_utils import enable_autoreload
enable_autoreload()

from guti.data_utils import list_svd_variants
from guti.parameters import Parameters
import matplotlib.pyplot as plt
import numpy as np


modality_name = "us_free_field_analytical_n_sources_sweep_50khz"
# modality_name = "us_free_field_analytical_n_sources_sweep_100khz"

n_sensors = 3028
# n_sensors = 1904
# n_sensors = 1000

# %%
# Specify which parameter to vary along the x-axis
# param_key = "num_brain_grid_points"
param_key = "num_sensors"

# Specify parameters to hold constant (None = don't filter on that parameter)
# Example: constant_params = Parameters(num_sensors=8000, time_resolution=1.0)
constant_params = Parameters(
    # num_sensors=n_sensors,
)  # No filtering by default

variants = list_svd_variants(modality_name, constant_params=constant_params)
for k, v in variants.items():
    print(f"  {k}: {v['params']}")

# %%
# Set the parameter key you want to analyze

# plt.figure(figsize=(10, 6))

# sorted_variants = sorted(
#     variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
# )

# For each unique value of param_key, find the variant with the largest num_brain_grid_points
from collections import defaultdict

variants_by_param = defaultdict(list)
for k, v in variants.items():
    param_value = getattr(v["params"], param_key)
    variants_by_param[param_value].append((k, v))

# For each param_key value, select the variant with max num_brain_grid_points
sorted_variants = []
for param_value in sorted(variants_by_param.keys()):
    variant_list = variants_by_param[param_value]
    best_variant = max(variant_list, key=lambda x: x[1]["params"].num_brain_grid_points)
    sorted_variants.append(best_variant)



# # Create colormap based on parameter values
# param_values = [getattr(v["params"], param_key) for k,v in sorted_variants]
# min_val, max_val = min(param_values), max(param_values)
# colors = plt.cm.viridis((np.array(param_values) - min_val) / (max_val - min_val))

# for (k, v), color in zip(sorted_variants, colors):
#     params = v["params"]
#     s = v["s"]
#     param_value = getattr(params, param_key)
#     s_normalized = s / param_value**0.5
    
#     assert param_key == "num_brain_grid_points"

#     plt.plot(
#         np.arange(1, len(s) + 1),
#         s_normalized,
#         label=f"{param_key}={param_value}",
#         color=color
#     )
# plt.legend()
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Singular Value Index")
# plt.ylabel("Singular Value")
# plt.title(f"{param_key} Scaling - {modality_name}")
# plt.ylim(1e-5, 1)
# plt.savefig(f"scaling_{param_key}_{modality_name}.png")

# %%

from guti.core import get_bitrate, noise_floor_heuristic

# Plot bitrate vs parameter value
plt.figure(figsize=(10, 6))

param_values = []
bitrates = []

for k, v in sorted_variants:
    params = v["params"]
    s = v["s"]
    param_value = getattr(params, param_key)
    s_normalized = s / param_value**0.5
    noise_level = noise_floor_heuristic(s_normalized, heuristic="power")
    
    # Get time resolution if available, otherwise use default 1.0
    # time_res = params.time_resolution if params.time_resolution is not None else 1.0
    time_res = 1.0
    
    bitrate = get_bitrate(s_normalized, noise_level, time_resolution=time_res, n_sources=param_value, n_detectors=n_sensors)
    print(bitrate)
    
    param_values.append(param_value)
    bitrates.append(bitrate)

plt.plot(param_values, bitrates, 'o-')
plt.xlabel(param_key)
plt.ylabel('Bitrate (bits/s)')
plt.title(f'Bitrate vs {param_key} - {modality_name}')
plt.grid(True)
# plt.show()
plt.savefig(f"bitrate_{param_key}_{modality_name}.png")

# %%
