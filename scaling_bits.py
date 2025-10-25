# %%
try:
    import IPython

    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")
except:
    pass

# %%
from guti.data_utils import list_svd_variants
from guti.core import get_bitrate, noise_floor_heuristic
import matplotlib.pyplot as plt

modality_name = "fnirs_analytical_cw"
subdir = "grid_sweep"
param_key = "num_brain_grid_points"

variants = list_svd_variants(modality_name, subdir=subdir)
for k, v in variants.items():
    print(f"  {k}: {v['params']}")

# %%

plt.figure(figsize=(10, 6))
sorted_variants = sorted(
    variants.items(), key=lambda x: getattr(x[1]["params"], param_key)
)

# Collect data for single plot
x_values = []
y_values = []

for k, v in sorted_variants:
    params = v["params"]
    s = v["s"]
    param_value = getattr(params, param_key)
    
    # Calculate noise floor using heuristic
    noise_floor = noise_floor_heuristic(s)
    
    # Calculate bitrate
    bits = get_bitrate(s, noise_floor)

    # Store values for plotting
    x_values.append(param_value)
    y_values.append(bits)

plt.plot(x_values, y_values, 'o-', markersize=8, linewidth=2)
plt.xlabel(f"{param_key}")
plt.ylabel("Bits")
plt.title(f"Bits vs {param_key} - {modality_name}")
plt.xscale("log")
plt.tight_layout()
plt.show()

# %%
