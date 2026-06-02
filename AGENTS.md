# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, etc.) when working with code in this repository.

## Project Overview

This is the "Grand unified theory of imaging" project (GUTI) - a research codebase that implements various medical imaging modalities and their mathematical foundations. The project consists of:

1. **Python Package (`guti/`)**: Core imaging simulation library
2. **Web Interface (`web/`)**: Next.js + MDX site of interactive visualizations and writeups

## Architecture

### Unified Modality Interface

All modalities share a common workflow defined in `guti/base_modality.py`:

- **`guti/base_modality.py`**: `ImagingModality` abstract base class. Subclasses implement
  `modality_name()`, `setup_geometry()`, `compute_forward_model()` (returns the Jacobian /
  sensitivity matrix), and `_get_default_modality_params()`. The base class drives the standard
  pipeline: setup geometry → compute forward model → SVD analysis → save results.
- **`guti/parameters.py`**: `Parameters` dataclass holding shared simulation parameters. Modality
  defaults are overridden by any non-None fields supplied at construction.
- **`run_modality.py`** (repo root): CLI driver to run any modality with custom parameters, e.g.
  ```bash
  python run_modality.py blur_1d --num_brain_grid_points 256
  python run_modality.py cw_fnirs --num_sensors 400
  python run_modality.py td_fnirs --scaled-up --no-save
  ```
  It auto-discovers modalities (directories under `guti/modalities/` that contain a `modality.py`
  and do not start with `_`).

### Core Python Package Structure

- **`guti/core.py`**: Core constants and geometry utilities for brain/skull/scalp
  - Physical constants: `BRAIN_RADIUS=80mm`, `CSF_RADIUS=81mm`, `SKULL_RADIUS=86mm`, `SCALP_RADIUS=92mm`
  - Conductivity values for different tissues
  - Geometry generation functions for sources, sensors, grids, and BEM meshes

- **`guti/modalities/`**: Imaging modality implementations (each with a `modality.py`)
  - `blur_1d/`: 1D blurring toy modality
  - `cw_fnirs/`: Continuous-wave functional near-infrared spectroscopy
  - `td_fnirs/`: Time-domain functional near-infrared spectroscopy
  - `eeg/`: Electroencephalography (uses OpenMEEG; BEM meshes in `eeg/_openmeeg_model/`)
  - `meg/`: Magnetoencephalography
  - `us/`: Ultrasound imaging (analytical implementation in `us/analytical.py`)
  - `_legacy/`: Retired / superseded implementations kept for reference
    (`ct`, `eit`, `fnirs`, `eeg_analytical`, `eeg_jax`, `meg_openmeeg`, `openmeeg`, `us_jwave`)

- **Other modules**:
  - `guti/svd.py`: Singular value decomposition analysis
  - `guti/slq.py`: Stochastic Lanczos quadrature for spectral estimation
  - `guti/linop.py`: Linear operator helpers
  - `guti/capacity.py`: Channel-capacity / information-rate computations
  - `guti/noise_models.py`: Noise models
  - `guti/hrf.py`: Hemodynamic response function utilities
  - `guti/scaling_utils.py`: Scaling / asymptotic analysis helpers
  - `guti/data_utils.py`: Data handling utilities
  - `guti/viz.py`, `guti/tri_view.py`: Visualization utilities
  - `guti/notebook_utils.py`, `guti/utils.py`: Misc helpers

### Scripts and Tests

- **`scripts/`**: Analysis, sweep, plotting, and benchmark drivers (e.g. `export_svd_json.py`,
  `compute_information_maps.py`, parameter sweeps, Modal GPU SVD benchmarks).
- **`tests/`**: Pytest suite (`test_capacity.py`, `test_information_maps.py`,
  `test_modality_params.py`, `test_slq.py`). `pyproject.toml` sets `pythonpath = ["scripts"]`
  so tests can import scripts by bare module name.

### Web Interface Structure

- Built with **Next.js 15** (App Router) + **MDX**, with **KaTeX** for math and **Recharts** for charts.
- Package name `lxm.house`; uses **pnpm**.
- Interactive visualizations / writeups for signal-processing and imaging concepts.

## Development Commands

### Python Package
```bash
# Setup virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Run a modality
python run_modality.py <modality_name> [--param value ...]

# Run tests
pytest
```

### Web Interface
```bash
cd web/
pnpm install         # Install dependencies
pnpm dev             # Start development server (http://localhost:3000)
pnpm build           # Build for production
pnpm start           # Run production build
pnpm lint            # Run Next.js lint
```

### OpenMEEG Integration
EEG uses OpenMEEG (boundary element method) via its in-process Python API:
- BEM meshes and geometry live in `guti/modalities/eeg/_openmeeg_model/`
  (`.tri` surfaces, `sphere_head.geom`)
- OpenMEEG is installed as a Python dependency (see `guti/modalities/eeg/requirements.txt`)

## Key Concepts

### Coordinate System
- Hemisphere-based geometry with origin at brain center
- Brain center at `(BRAIN_RADIUS, BRAIN_RADIUS, 0)` = `(80, 80, 0)` mm
- Z-axis points upward (positive hemisphere)

### Modality-Specific Notes
- Each modality in `guti/modalities/` has its own `requirements.txt` for modality-specific deps
- The ultrasound modality has an analytical implementation (`us/analytical.py`)
- Results are stored in `results/` with SVD spectrum analysis

### Data Flow
1. Generate source/sensor positions using `guti/core.py` functions (via `setup_geometry()`)
2. Compute forward models for specific modalities (`compute_forward_model()`)
3. Perform SVD analysis using `guti/svd.py` (and `guti/slq.py` for large spectra)
4. Visualize results using `guti/viz.py` or the web interface

## File Formats
- `.tri`: Triangle mesh files (BrainVisa format)
- `.geom`: OpenMEEG geometry description
- `.cond`: Conductivity specifications
- `.npz`: Numpy compressed arrays for results storage
