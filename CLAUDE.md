# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the "Grand unified theory of imaging" project (GUTI) - a research codebase that implements various medical imaging modalities and their mathematical foundations. The project consists of:

1. **Python Package (`guti/`)**: Core imaging simulation library
2. **Web Interface (`web/`)**: Interactive React visualizations for signal processing concepts

## Architecture

### Core Python Package Structure

- **`guti/core.py`**: Core constants and utilities for brain/skull/scalp geometry
  - Contains physical constants (BRAIN_RADIUS=80mm, SKULL_RADIUS=86mm, SCALP_RADIUS=92mm)
  - Conductivity values for different tissues
  - Geometry generation functions for sources, sensors, and BEM models

- **`guti/modalities/`**: Imaging modality implementations
  - `eeg/`: Electroencephalography
  - `meg/`: Magnetoencephalography  
  - `fnirs/`: Functional near-infrared spectroscopy
  - `fnirs_analytical/`: Analytical fNIRS solutions
  - `ct/`: Computed tomography
  - `eit/`: Electrical impedance tomography
  - `us/`: Ultrasound imaging
  - `bem_model/`: Boundary element method files and meshes

- **`guti/data_utils.py`**: Data handling utilities
- **`guti/svd.py`**: Singular value decomposition analysis
- **`guti/viz.py`**: Visualization utilities

### Web Interface Structure

- Built with React + Vite
- Interactive visualizations for signal processing concepts:
  - Linear model visualization
  - Fourier transform demonstrations
  - Eigenvalue analysis
  - Orthogonality and noise visualization
  - SNR and channel coding concepts

## Development Commands

### Python Package
```bash
# Setup virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Web Interface
```bash
cd web/
npm install           # Install dependencies
npm run dev          # Start development server (http://localhost:5173)
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### OpenMEEG Integration
The project uses OpenMEEG for boundary element method calculations:
- `guti/modalities/compute_leadfields.sh`: Shell script for computing leadfield matrices
- Requires OpenMEEG to be installed and available in PATH
- Generates EEG, MEG, and other modality leadfields

## Key Concepts

### Coordinate System
- Hemisphere-based geometry with origin at brain center
- Brain center at (BRAIN_RADIUS, BRAIN_RADIUS, 0) = (80, 80, 0) mm
- Z-axis points upward (positive hemisphere)

### Modality-Specific Notes
- Each modality in `guti/modalities/` may have its own requirements.txt
- Some modalities (fnirs, fnirs_analytical, us) have analytical implementations
- Results are stored in `results/` with SVD spectrum analysis

### Data Flow
1. Generate source/sensor positions using `guti/core.py` functions
2. Compute forward models for specific modalities
3. Perform SVD analysis using `guti/svd.py`
4. Visualize results using `guti/viz.py` or web interface

## File Formats
- `.tri`: Triangle mesh files (BrainVisa format)
- `.geom`: OpenMEEG geometry description
- `.cond`: Conductivity specifications
- `.npz`: Numpy compressed arrays for results storage