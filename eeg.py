# %%
import numpy as np
import modalities.meg.meg as meg
import matplotlib.pyplot as plt
from pathlib import Path

# %% [markdown]
# # EEG Lead Field (Jacobian) Visualization
# This notebook demonstrates how to visualize the EEG lead field distribution for specific electrodes using MNE-Python.

# %%
# Set up the directory for storing MNE sample data
subjects_dir = meg.datasets.sample.data_path() / "subjects"

# %% [markdown]
# ## Create sample EEG data with standard 10-20 montage

# %%
# Create sample data with 10-20 electrode positions
ch_names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
data = np.random.RandomState(0).randn(len(ch_names), 1000)
info = meg.create_info(ch_names, 1000.0, "eeg")
raw = meg.io.RawArray(data, info)

# %% [markdown]
# ## Set up the head model and source space

# %%
# Download fsaverage files
fs_dir = meg.datasets.fetch_fsaverage(verbose=True)
subjects_dir = Path(fs_dir).parent

# Set up the source space
src = meg.setup_source_space(
    "fsaverage", spacing="oct6", subjects_dir=subjects_dir, add_dist=False
)

# Get the BEM solution
conductivity = (0.3, 0.006, 0.3)  # for three layers
model = meg.make_bem_model(
    "fsaverage", ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = meg.make_bem_solution(model)

# %% [markdown]
# ## Set up the montage and compute the forward solution

# %%
# Setup the montage
montage = meg.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Compute transformation matrix
fiducials = "estimated"  # get fiducials from the standard montage
trans = "fsaverage"  # use fsaverage transformation

# Compute forward solution
fwd = meg.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0
)
# %%
# Convert to fixed orientation
fwd_fixed = meg.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)

# %% [markdown]
# ## Visualize the lead field for a specific electrode

# %%
# Get the lead field matrix
leadfield = fwd_fixed["sol"]["data"]

# Select a specific electrode (e.g., Cz which is often used as reference)
electrode_idx = ch_names.index("Cz")

# Get the lead field for this electrode
electrode_leadfield = leadfield[electrode_idx]

# Split the leadfield into left and right hemispheres
n_sources_lh = len(fwd_fixed["src"][0]["vertno"])
n_sources_rh = len(fwd_fixed["src"][1]["vertno"])

leadfield_lh = electrode_leadfield[:n_sources_lh]
leadfield_rh = electrode_leadfield[n_sources_lh:]

# %% [markdown]
# ## Plot the lead field distribution on the brain

# %%
# Create source estimate object for visualization
vertices = [fwd_fixed["src"][0]["vertno"], fwd_fixed["src"][1]["vertno"]]

# %% [markdown]
# The visualization above shows how the electrical potential measured at the Cz electrode
# is influenced by different source locations in the brain. Brighter colors indicate
# regions where neural activity has a stronger influence on the measurement at Cz.

# %%

import pyvista as pv
import numpy as np
from pathlib import Path

# Create PyVista plotter
plotter = pv.Plotter()

# Load the fsaverage brain surface
surf_path = Path(subjects_dir) / "fsaverage" / "surf"
lh_path = surf_path / "lh.pial"
rh_path = surf_path / "rh.pial"


# Read surfaces using PyVista
def read_surface(filepath):
    coords, faces = meg.read_surface(str(filepath))
    mesh = pv.PolyData()
    mesh.points = coords
    mesh.faces = np.hstack((3 * np.ones((len(faces), 1)), faces)).astype(np.int32)
    return mesh


# Load both hemispheres
lh_mesh = read_surface(lh_path)
rh_mesh = read_surface(rh_path)

# Map the leadfield data to the surfaces
lh_mesh.point_data["leadfield"] = np.zeros(lh_mesh.n_points)
rh_mesh.point_data["leadfield"] = np.zeros(rh_mesh.n_points)

# Map data to vertices
for i, vertex in enumerate(vertices[0]):
    lh_mesh.point_data["leadfield"][vertex] = np.abs(leadfield_lh[i])
for i, vertex in enumerate(vertices[1]):
    rh_mesh.point_data["leadfield"][vertex] = np.abs(leadfield_rh[i])

# Add meshes to plotter with custom coloring
max_value = max(np.max(np.abs(leadfield_lh)), np.max(np.abs(leadfield_rh)))
plotter.add_mesh(
    lh_mesh,
    scalars="leadfield",
    cmap="plasma",
    clim=[0, max_value],  # Adjust the upper limit
    show_scalar_bar=True,
)
plotter.add_mesh(
    rh_mesh,
    scalars="leadfield",
    cmap="plasma",
    clim=[0, max_value],  # Adjust the upper limit
    show_scalar_bar=True,
)

# Set camera position for better view
plotter.camera_position = "xz"
plotter.camera.zoom(1.5)

# Add title
plotter.add_text("Lead Field Visualization", font_size=20)

# Show the plot
plotter.show()
# %%
plt.plot(leadfield_lh)
plt.show()

# %%
import numpy as np
import modalities.meg.meg as meg
import pyvista as pv
from scipy.spatial import cKDTree

# -- Suppose you already have:
#    lh_mesh, rh_mesh         # high-res PyVista meshes for L/R hemispheres
#    vertices = [vert_lh, vert_rh]  # source-space vertex indices for L/R
#    leadfield_lh, leadfield_rh     # lead-field values at each source vertex

# For convenience, collect the "source-space" coordinates from the pial mesh
# i.e. the actual 3D coords for just those source vertices
lh_source_coords = lh_mesh.points[vertices[0]]
rh_source_coords = rh_mesh.points[vertices[1]]

# Build KD trees for L and R hemisphere source coords
tree_lh = cKDTree(lh_source_coords)
tree_rh = cKDTree(rh_source_coords)

# We'll do abs(...) just to keep everything positive for plotting
lh_data = np.abs(leadfield_lh)
rh_data = np.abs(leadfield_rh)

# Now for EVERY vertex in the high-res pial mesh, find nearest source vertex
dist_lh, idx_lh = tree_lh.query(lh_mesh.points)
dist_rh, idx_rh = tree_rh.query(rh_mesh.points)

# Map each pial vertex to lead-field value of its nearest source vertex
lh_mesh.point_data["leadfield"] = lh_data[idx_lh]
rh_mesh.point_data["leadfield"] = rh_data[idx_rh]

# --- Now do the actual PyVista plotting ---
plotter = pv.Plotter()
max_value = max(lh_data.max(), rh_data.max())

plotter.add_mesh(
    lh_mesh,
    scalars="leadfield",
    cmap="plasma",
    clim=[0, max_value],
    show_scalar_bar=True,
)
plotter.add_mesh(
    rh_mesh,
    scalars="leadfield",
    cmap="plasma",
    clim=[0, max_value],
    show_scalar_bar=True,
)

plotter.add_text("Lead Field Visualization", font_size=20)
plotter.show()

# %%
import mne.viz

meg.viz.set_3d_backend("pyvistaqt")

# %%
# Create an stc with one time point, so data has shape=(n_vertices_total, n_times)
data_stc = np.concatenate([leadfield_lh, leadfield_rh])
data_stc = data_stc[:, np.newaxis]  # shape => (n_sources, 1)

stc = meg.SourceEstimate(
    data_stc,
    vertices=vertices,  # [lh_vertno, rh_vertno]
    tmin=0.0,
    tstep=1.0,  # dummy
    subject="fsaverage",  # make sure to match your subject
)

# Now plot with MNE's built-in 3D viewer
brain = stc.plot(
    subject="fsaverage",
    surface="pial",
    subjects_dir=subjects_dir,
    hemi="both",
    time_viewer=False,
    views=["lat"],
    size=(800, 800),
    colormap="plasma",
    clim=dict(kind="value", lims=[0, 0.5 * data_stc.max(), data_stc.max()]),
    smoothing_steps=5,  # Smooth the data for better visualization
    transparent=False,  # Make the surface fully opaque
)

# # Use PyVista to export to HTML
# plotter = brain.plotter

# # Set opacity for all actors in the scene
# # Ensure scalar bar actor is handled correctly
# for actor in plotter.renderer.scalar_bars.values():
#     actor.SetVisibility(True)
# plotter.export_html("brain_visualization.html")


# %%
