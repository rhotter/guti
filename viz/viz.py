# %%
import os
import numpy as np
import mne
import mne.viz
from pathlib import Path

# %% [markdown]
# # EEG Lead Field (Jacobian) Visualization
# This notebook demonstrates how to visualize the EEG lead field distribution for specific electrodes using MNE-Python.

# %%
# Set up the directory for storing MNE sample data
subjects_dir = mne.datasets.sample.data_path() / "subjects"

# %% [markdown]
# ## Create sample EEG data with standard 10-20 montage

# %%
# Visualization mode: 'leadfield' (default), 'blob' (Gaussian), 'fourier' (high-frequency)
MODE = "fourier"  # options: 'leadfield', 'blob', 'fourier'
FAST_MODE = False  # drastically reduces resolution and rendering cost

# Parameters for synthetic modes
BLOB_SIGMA_M = 0.02  # Gaussian sigma in meters
FOURIER_K_RAD_PER_M = 250.0  # wave number magnitude (higher => higher frequency)
FOURIER_DIRECTION = np.array([1.0, 0.0, 0.0])  # direction of the plane wave

# Control whether to open the interactive UI window or just save images
OPEN_UI = True  # set to False to render offscreen and only save figures

# Create sample data with 10-20 electrode positions
ch_names = "Fz Cz Pz Oz Fp1 Fp2 F3 F4 F7 F8 C3 C4 T7 T8 P3 P4 P7 P8 O1 O2".split()
data = np.random.RandomState(0).randn(len(ch_names), 1000)
info = mne.create_info(ch_names, 1000.0, "eeg")
raw = mne.io.RawArray(data, info)

# %% [markdown]
# ## Set up the head model and source space

# %%
# Download fsaverage files
fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
subjects_dir = Path(fs_dir).parent

# Set up the source space
if MODE in ("blob", "fourier"):
    # For synthetic modes, denser mesh yields smoother-looking patterns, still cheap (no BEM)
    spacing = "ico4" if FAST_MODE else "oct5"
else:
    spacing = "ico3" if FAST_MODE else "oct5"
src = mne.setup_source_space(
    "fsaverage", spacing=spacing, subjects_dir=subjects_dir, add_dist=False
)

# Get the BEM solution (only needed for leadfield mode)
if MODE == "leadfield":
    conductivity = (0.3, 0.006, 0.3)  # for three layers
    bem_ico = 2 if FAST_MODE else 3
    model = mne.make_bem_model(
        "fsaverage", ico=bem_ico, conductivity=conductivity, subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)

# %% [markdown]
# ## Set up the montage and compute the forward solution

# %%
# Setup the montage (only needed for leadfield mode)
if MODE == "leadfield":
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

if MODE == "leadfield":
    # Compute transformation matrix
    fiducials = "estimated"  # get fiducials from the standard montage
    trans = "fsaverage"  # use fsaverage transformation

    # Compute forward solution
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0
    )
    # Convert to fixed orientation
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True)

# %% [markdown]
# ## Visualize the lead field for a specific electrode

# %%
if MODE == "leadfield":
    # Get the lead field matrix
    leadfield = fwd_fixed["sol"]["data"]

    # Select a specific electrode (e.g., Cz)
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
src_used = fwd_fixed["src"] if MODE == "leadfield" else src
vertices = [src_used[0]["vertno"], src_used[1]["vertno"]]

# Build data for synthetic modes if selected
if MODE in ("blob", "fourier"):
    # Coordinates (in meters) for vertices actually used
    coords_lh = src[0]["rr"][vertices[0]]
    coords_rh = src[1]["rr"][vertices[1]]
    coords = np.vstack([coords_lh, coords_rh])

    if MODE == "blob":
        # Center at a reproducible vertex (middle of left hemisphere list)
        center_idx = len(vertices[0]) // 2
        center = coords_lh[center_idx]
        d2 = np.sum((coords - center) ** 2, axis=1)
        data_values = np.exp(-d2 / (2.0 * (BLOB_SIGMA_M ** 2)))
    else:  # MODE == 'fourier'
        k_vec = FOURIER_DIRECTION / (np.linalg.norm(FOURIER_DIRECTION) + 1e-12)
        k_vec = k_vec * FOURIER_K_RAD_PER_M
        phase = coords @ k_vec
        # Shift to [0, 1] for compatibility with positive clim
        data_values = 0.5 * (np.sin(phase) + 1.0)

# %% [markdown]
# The visualization above shows how the electrical potential measured at the Cz electrode
# is influenced by different source locations in the brain. Brighter colors indicate
# regions where neural activity has a stronger influence on the measurement at Cz.

if OPEN_UI:
    mne.viz.set_3d_backend("pyvistaqt")
else:
    # Offscreen rendering using the non-Qt backend
    os.environ["PYVISTA_OFF_SCREEN"] = "true"
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except Exception:
        pass
    mne.viz.set_3d_backend("pyvista")

# %%
# Create an stc with one time point, so data has shape=(n_vertices_total, n_times)
if MODE == "leadfield":
    data_stc = np.concatenate([leadfield_lh, leadfield_rh])
else:
    data_stc = data_values
data_stc = data_stc[:, np.newaxis]  # shape => (n_sources, 1)

stc = mne.SourceEstimate(
    data_stc,
    vertices=vertices,  # [lh_vertno, rh_vertno]
    tmin=0.0,
    tstep=1.0,  # dummy
    subject="fsaverage",  # make sure to match your subject
)

# Now plot with MNE's built-in 3D viewer
if MODE in ("blob", "fourier"):
    SMOOTHING_STEPS = 8 if FAST_MODE else 10
else:
    SMOOTHING_STEPS = 0 if FAST_MODE else 3
WIN_SIZE = (400, 400) if FAST_MODE else (600, 600)

brain = stc.plot(
    subject="fsaverage",
    surface=("inflated" if MODE in ("blob", "fourier") else "pial"),
    subjects_dir=subjects_dir,
    hemi="both",
    time_viewer=False,
    views=["lat"],
    size=WIN_SIZE,
    colormap="plasma",
    clim=dict(kind="value", lims=[0, 0.5 * data_stc.max(), data_stc.max()]),
    smoothing_steps=SMOOTHING_STEPS,
    transparent=False,
)

# %%
# In headless mode, force a render once before screenshots to avoid black images
if not OPEN_UI:
    try:
        brain._renderer.plotter.show(auto_close=False)
    except Exception:
        pass

# Save snapshots from multiple views in the current directory
for view in ["lat", "med", "dor"]:
    brain.show_view(view)
    brain.save_image(f"leadfield_{view}.png")

# Also save an oblique angled view
angled_view = dict(azimuth=120, elevation=30, distance=450)
brain.show_view(angled_view)
brain.save_image("leadfield_angle.png")

# Close the viewer in headless mode
if not OPEN_UI:
    try:
        brain.close()
    except Exception:
        pass
