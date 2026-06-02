# %% [markdown]
# # 3D Tomography Experiments

# %%
%load_ext autoreload
%autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Medium

# %%
from problems.hemisphere_3d import hemisphere_3d

voxel_size_mm = 2.0
medium, sensors, brain_mask = hemisphere_3d(noptodes=300, voxel_size_mm=voxel_size_mm)

# %% [markdown]
# ## Forward

# %%
from solver import Solver, jacobian

solver = Solver(medium, sensors, tstart=0, tend=5e-9, tstep=5e-9)

# %%
nx, ny, nz = medium.shape
ndet, nsrc = sensors.ndet, sensors.nsrc
nt = solver.nt

# %%
# initialize arrays
J = np.zeros((nx, ny, nz, nt, ndet, nsrc))

# %%
from tqdm.notebook import tqdm

nphoton = 1e5

# run forward for all sources (for both the ground truth and background)
for src_idx in tqdm(range(nsrc), desc="Processing sources"):
    solver.medium = medium
    res_bg, cfg_bg = solver.forward(
        src_idx,
        random_seed=1,
        nphoton=nphoton,
        voxel_size_mm=voxel_size_mm,
        max_src_det_distance_mm=50,
    )
    J[..., src_idx] = jacobian(res_bg, cfg_bg)

# %% [markdown]
# ## SVD

# %%
# keep only the jacobian elements from the brain
J_brain = J[brain_mask]
# %%
J_cpu = J_brain.reshape((brain_mask.sum() * nt, ndet * nsrc))
J_cpu.shape

# %%
from guti.svd import compute_svd_gpu

s = compute_svd_gpu(J_cpu)

# %%
from guti.svd import plot_svd

plot_svd(s)

# %% [markdown]
# ## Save the singular value spectrum

# %%
from guti.data_utils import save_svd

save_svd(s, "fnirs_cw")

# %%
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Add source positions
fig.add_trace(
    go.Scatter3d(
        x=sensors.src_pos[:, 0],
        y=sensors.src_pos[:, 1],
        z=sensors.src_pos[:, 2],
        mode="markers",
        marker=dict(size=2, symbol="circle", color="red"),
        name="Sources",
    )
)

# Add detector positions
fig.add_trace(
    go.Scatter3d(
        x=sensors.det_pos[:, 0],
        y=sensors.det_pos[:, 1],
        z=sensors.det_pos[:, 2],
        mode="markers",
        marker=dict(size=2, symbol="circle", color="blue"),
        name="Detectors",
    )
)

# Add source direction vectors
for i in range(len(sensors.src_pos)):
    # Calculate end point of vector
    end_point = (
        sensors.src_pos[i] + sensors.src_dirs[i] * 5
    )  # Scale vector for visibility

    fig.add_trace(
        go.Scatter3d(
            x=[sensors.src_pos[i, 0], end_point[0]],
            y=[sensors.src_pos[i, 1], end_point[1]],
            z=[sensors.src_pos[i, 2], end_point[2]],
            mode="lines",
            line=dict(color="red", width=2),
            showlegend=False,
        )
    )

# Update layout
fig.update_layout(
    title="Sensor Positions",
    scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
    showlegend=True,
)

fig.show()


