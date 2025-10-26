import os
import numpy as np
from pathlib import Path
import warnings
from typing import Literal

np.random.seed(239)

BRAIN_RADIUS = 80  # mm
SKULL_RADIUS = 86
SCALP_RADIUS = 92

AIR_CONDUCTIVITY = 0
SCALP_CONDUCTIVITY = 1
BRAIN_CONDUCTIVITY = 1
SKULL_CONDUCTIVITY = 0.03

N_SOURCES_DEFAULT = 100
N_SENSORS_DEFAULT = 100


def get_sensor_positions(
    n_sensors: int = N_SENSORS_DEFAULT,
    offset: float = 0,
    start_n: int = 0,
    end_n: int | None = None,
) -> np.ndarray:
    """
    Get sensor positions uniformly on the surface of a hemisphere.
    """
    # Deterministic uniform sampling on a hemisphere using a spherical Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(n_sensors)
    # z coordinates uniformly spaced in [0,1)
    z = (indices + 0.5) / n_sensors
    # polar angle
    theta = np.arccos(z)
    # azimuthal angle using golden angle
    phi = golden_angle * indices
    # convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    # unit hemisphere points
    positions = np.stack([x, y, z], axis=1)
    # scale to SCALP_RADIUS and translate to center at (BRAIN_RADIUS, BRAIN_RADIUS, 0)
    positions = positions * (SCALP_RADIUS + offset) + np.array(
        [SCALP_RADIUS, SCALP_RADIUS, 0]
    )
    return positions[start_n:end_n]


def get_grid_positions(
    grid_spacing_mm: float = 5.0, radius: float = BRAIN_RADIUS
) -> np.ndarray:
    """Generate positions using a uniform 3D grid within the hemisphere.

    Parameters
    ----------
    grid_spacing_mm : float
        Spacing between grid points in mm

    Returns
    -------
    positions : ndarray of shape (n_points, 3)
        Grid positions inside the hemisphere in mm
    """
    # Create grid coordinates
    # Grid extends from 0 to 2*radius in x and y, and 0 to radius in z
    # to account for brain, skull, and scalp layers
    x_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
    y_coords = np.arange(0, 2 * radius + grid_spacing_mm, grid_spacing_mm)
    z_coords = np.arange(0, radius + grid_spacing_mm, grid_spacing_mm)

    # Create meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # Flatten to get all grid points
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Filter points that are inside the hemisphere
    # Center of hemisphere is at (radius, radius, 0)
    center = np.array([radius, radius, 0])
    distances = np.linalg.norm(grid_points - center, axis=1)

    # Keep points inside the hemisphere (distance <= radius and z >= 0)
    inside_hemisphere = (distances <= radius) & (grid_points[:, 2] >= 0)
    hemisphere_points = grid_points[inside_hemisphere]

    print(f"Using {len(hemisphere_points)} grid points")

    return hemisphere_points


def get_voxel_mask(resolution: float = 1, offset: float = 0) -> np.ndarray:
    """
    Create a voxel mask for the brain.
    The mask is a 3D array of size (nx, ny, nz)
    The mask is 1 for the brain, 2 for the skull, 3 for the scalp and 0 for the rest
    """
    radius = SCALP_RADIUS + offset
    nx = int(2 * radius / resolution)
    ny = int(2 * radius / resolution)
    nz = int(radius / resolution)
    mask = np.zeros((nx, ny, nz))

    # Create coordinate grids
    x = np.linspace(-radius, radius, nx)
    y = np.linspace(-radius, radius, ny)
    z = np.linspace(0, radius, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Calculate distances from origin for all points at once
    distances = np.sqrt(X**2 + Y**2 + Z**2)

    # Set mask values based on distances
    mask[distances <= BRAIN_RADIUS] = 1
    mask[(distances > BRAIN_RADIUS) & (distances <= SKULL_RADIUS)] = 2
    mask[(distances > SKULL_RADIUS) & (distances <= radius)] = 3

    return mask


# ---- FEM mesh functions ----


def create_sphere(radius, n_phi=8, n_theta=8, resolution=None, center=None):
    """Create a sphere mesh.

    Parameters
    ----------
    radius : float
        Radius of the sphere in meters
    n_phi : int, optional
        Number of points in the azimuthal direction (ignored if resolution is provided)
    n_theta : int, optional
        Number of points in the polar direction (ignored if resolution is provided)
    resolution : float, optional
        Desired grid spacing in meters. If provided, n_phi and n_theta are calculated
        automatically to achieve approximately uniform grid spacing.
    center : ndarray, optional
        Center of the sphere as (x, y, z). If None, defaults to (0, 0, 0).

    Returns
    -------
    vertices : ndarray
        Vertex coordinates
    triangles : ndarray
        Triangle indices (0-based)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    if resolution is not None:
        # Calculate n_phi and n_theta based on desired resolution
        # For uniform spacing, we want approximately equal arc lengths
        # Arc length = radius * angle, so angle = resolution / radius

        # Calculate n_phi based on circumference at equator
        n_phi = max(8, int(2 * np.pi * radius / resolution))

        # Calculate n_theta based on meridian length
        # For uniform spacing, we want similar arc lengths in both directions
        n_theta = max(8, int(np.pi * radius / resolution))

        # Ensure n_phi is even for better triangulation
        if n_phi % 2 != 0:
            n_phi += 1

    # Generate grid of points in spherical coordinates
    # For a full sphere, theta goes from 0 to π
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    # Create vertices
    vertices = []

    # Add the north pole point at the top of the sphere
    vertices.append([0, 0, radius])

    # Add vertices for the middle of the sphere
    for t in theta[1:-1]:  # Skip the first and last theta (poles)
        for p in phi[:-1]:  # Skip the last phi (duplicate of phi=0)
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            vertices.append([x, y, z])

    # Add the south pole point at the bottom of the sphere
    vertices.append([0, 0, -radius])

    vertices = np.array(vertices)

    # Create triangles
    triangles = []

    # Number of unique phi points
    n_phi_actual = n_phi - 1

    # Create triangles connecting the pole to the first ring
    for i in range(n_phi_actual):
        v1 = 0  # Pole vertex
        v2 = i + 1
        v3 = (i + 1) % n_phi_actual + 1
        triangles.append([v1, v2, v3])

    # Create triangles for the middle of the sphere
    for i in range(
        n_theta - 3
    ):  # -3 because we have two poles and handle them separately
        row_start = 1 + i * n_phi_actual
        next_row_start = 1 + (i + 1) * n_phi_actual

        for j in range(n_phi_actual):
            v1 = row_start + j
            v2 = row_start + (j + 1) % n_phi_actual
            v3 = next_row_start + j
            v4 = next_row_start + (j + 1) % n_phi_actual

            # Add two triangles for each quad
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])

    # Create triangles connecting the last row to the south pole
    south_pole_index = len(vertices) - 1
    last_row_start = 1 + (n_theta - 3) * n_phi_actual
    for i in range(n_phi_actual):
        v1 = last_row_start + i
        v2 = last_row_start + (i + 1) % n_phi_actual
        v3 = south_pole_index
        triangles.append([v1, v3, v2])  # Note reversed order for proper orientation

    triangles = np.array(triangles)

    # Translate vertices to the specified center
    vertices = vertices + center

    print(
        f"Created sphere with {len(vertices)} vertices and {len(triangles)} triangles"
    )

    return vertices, triangles


def create_hemisphere(radius, n_phi=8, n_theta=8, resolution=None, center=None):
    """Create a hemisphere mesh (upper half-sphere with flat base).

    Parameters
    ----------
    radius : float
        Radius of the hemisphere in meters
    n_phi : int, optional
        Number of points in the azimuthal direction (ignored if resolution is provided)
    n_theta : int, optional
        Number of points in the polar direction (ignored if resolution is provided)
    resolution : float, optional
        Desired grid spacing in meters. If provided, n_phi and n_theta are calculated
        automatically to achieve approximately uniform grid spacing.
    center : ndarray, optional
        Center of the hemisphere base as (x, y, z). If None, defaults to (0, 0, 0).
        The hemisphere extends upward (positive z direction) from this center.

    Returns
    -------
    vertices : ndarray
        Vertex coordinates
    triangles : ndarray
        Triangle indices (0-based)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    if resolution is not None:
        # Calculate n_phi and n_theta based on desired resolution
        # For uniform spacing, we want approximately equal arc lengths
        # Arc length = radius * angle, so angle = resolution / radius

        # Calculate n_phi based on circumference at equator
        n_phi = max(8, int(2 * np.pi * radius / resolution))

        # Calculate n_theta based on meridian length (half for hemisphere)
        # For uniform spacing, we want similar arc lengths in both directions
        n_theta = max(4, int(np.pi * radius / (2 * resolution)))

        # Ensure n_phi is even for better triangulation
        if n_phi % 2 != 0:
            n_phi += 1

    # Generate grid of points in spherical coordinates
    # For a hemisphere, theta goes from 0 to π/2
    theta = np.linspace(0, np.pi / 2, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    # Create vertices
    vertices = []

    # Add the north pole point at the top of the hemisphere
    vertices.append([0, 0, radius])

    # Add vertices for the curved surface
    for t in theta[1:-1]:  # Skip the first (pole) and last (equator)
        for p in phi[:-1]:  # Skip the last phi (duplicate of phi=0)
            x = radius * np.sin(t) * np.cos(p)
            y = radius * np.sin(t) * np.sin(p)
            z = radius * np.cos(t)
            vertices.append([x, y, z])

    # Add vertices for the equator (z=0 plane)
    equator_start_idx = len(vertices)
    for p in phi[:-1]:
        x = radius * np.cos(p)
        y = radius * np.sin(p)
        z = 0
        vertices.append([x, y, z])

    # Add center point for the base
    base_center_idx = len(vertices)
    vertices.append([0, 0, 0])

    vertices = np.array(vertices)

    # Create triangles
    triangles = []

    # Number of unique phi points
    n_phi_actual = n_phi - 1

    # Create triangles connecting the pole to the first ring
    for i in range(n_phi_actual):
        v1 = 0  # North pole vertex
        v2 = i + 1
        v3 = (i + 1) % n_phi_actual + 1
        triangles.append([v1, v2, v3])

    # Create triangles for the curved surface (excluding equator)
    for i in range(n_theta - 3):  # -3 because we have pole, skip equator
        row_start = 1 + i * n_phi_actual
        next_row_start = 1 + (i + 1) * n_phi_actual

        for j in range(n_phi_actual):
            v1 = row_start + j
            v2 = row_start + (j + 1) % n_phi_actual
            v3 = next_row_start + j
            v4 = next_row_start + (j + 1) % n_phi_actual

            # Add two triangles for each quad
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])

    # Connect the last curved ring to the equator ring
    if n_theta > 2:  # Only if we have intermediate rings
        last_curved_row_start = 1 + (n_theta - 3) * n_phi_actual
        for i in range(n_phi_actual):
            v1 = last_curved_row_start + i
            v2 = last_curved_row_start + (i + 1) % n_phi_actual
            v3 = equator_start_idx + i
            v4 = equator_start_idx + (i + 1) % n_phi_actual

            # Add two triangles for each quad
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])

    # Create triangles for the flat base (fan from center)
    for i in range(n_phi_actual):
        v1 = base_center_idx  # Center of base
        v2 = equator_start_idx + (i + 1) % n_phi_actual
        v3 = equator_start_idx + i
        triangles.append(
            [v1, v2, v3]
        )  # Note: reversed order for downward-pointing normal

    triangles = np.array(triangles)

    # Translate vertices to the specified center
    vertices = vertices + center

    print(
        f"Created hemisphere with {len(vertices)} vertices and {len(triangles)} triangles"
    )

    return vertices, triangles


def write_tri(filename, vertices, triangles, center=None):
    """Write a .tri file following the BrainVisa format.

    Parameters
    ----------
    filename : str
        Path to the output file
    vertices : array
        Vertex coordinates (N, 3)
    triangles : array
        Triangle indices (M, 3), must use 0-based indexing
    center : ndarray, optional
        Center of the sphere as (x, y, z). Used to compute normals correctly.
        If None, defaults to (0, 0, 0).
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    with open(filename, "w") as f:
        # Write number of vertices
        f.write(f"- {len(vertices)}\n")

        # Write vertices with normals (normals = normalized vertex positions for a sphere)
        for v in vertices:
            # Calculate normal relative to sphere center
            relative_pos = v - center
            n = relative_pos / np.linalg.norm(relative_pos)
            f.write(
                f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n"
            )

        # Write number of triangles (repeated three times as per format)
        f.write(f"- {len(triangles)} {len(triangles)} {len(triangles)}\n")

        # Write triangles with 0-based indexing
        for t in triangles:
            f.write(f"{t[0]} {t[1]} {t[2]}\n")


def get_random_orientations(n_sources: int) -> np.ndarray:
    """Generate random unit vectors for dipole orientations.

    Parameters
    ----------
    n_sources : int
        Number of dipole orientations to generate

    Returns
    -------
    orientations : ndarray of shape (n_sources, 3)
        Random unit vectors representing dipole orientations
    """
    # Generate random vectors
    orientations = np.random.randn(n_sources, 3)
    # Normalize to unit vectors
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    orientations = orientations / norms
    return orientations


def get_cortical_positions(n_sources=N_SOURCES_DEFAULT, radius=BRAIN_RADIUS):
    """Generate positions uniformly distributed on the cortical surface (brain hemisphere).

    For EIT modeling, sources should be positioned on the cortical surface rather
    than in the brain volume interior.

    Parameters
    ----------
    n_sources : int
        Number of source positions to generate
    radius : float
        Radius of the brain surface (cortex)

    Returns
    -------
    positions : ndarray
        Array of (x, y, z) positions on the hemisphere surface
    """
    positions = []

    # Generate uniform points on hemisphere using rejection sampling
    # This ensures uniform distribution on the curved surface
    while len(positions) < n_sources:
        # Generate random points in a cube
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = np.random.uniform(0, radius)  # Only upper hemisphere (z >= 0)

        # Check if point is on or near the sphere surface
        distance = np.sqrt(x**2 + y**2 + z**2)
        if distance <= radius:  # Inside or on the sphere
            # Project onto sphere surface
            if distance > 0:  # Avoid division by zero
                scale = radius / distance
                pos = [x * scale, y * scale, z * scale]
                positions.append(pos)

    # Translate to match the coordinate system (centered at (radius, radius, 0))
    positions = np.array(positions[:n_sources])
    positions = positions + np.array([radius, radius, 0])

    return positions


def create_bem_model(vertices=None, triangles=None):
    """Create a 3-layer spherical model with cortical sources."""
    # Ensure model directory exists
    os.makedirs("bem_model/", exist_ok=True)

    # Define the center of the spheres to match source/sensor coordinate system
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

    # Create the three spherical meshes
    for name, radius in [
        ("brain", BRAIN_RADIUS),
        ("skull", SKULL_RADIUS),
        ("scalp", SCALP_RADIUS),
    ]:
        # if vertices is None or triangles is None:
        # vertices, triangles = create_sphere(radius, n_phi=16, n_theta=10)
        vertices, triangles = create_sphere(radius, resolution=20, center=center)
        write_tri(f"bem_model/{name}_sphere.tri", vertices, triangles, center=center)

    # Create the geometry file (format 1.1)
    with open("bem_model/sphere_head.geom", "w") as f:
        f.write("# Domain Description 1.1\n\n")
        f.write("Interfaces 3\n\n")
        f.write('Interface Brain: "brain_sphere.tri"\n')
        f.write('Interface Skull: "skull_sphere.tri"\n')
        f.write('Interface Scalp: "scalp_sphere.tri"\n\n')
        f.write("Domains 4\n\n")
        f.write("Domain Brain: -Brain\n")
        f.write("Domain Skull: -Skull +Brain\n")
        f.write("Domain Scalp: -Scalp +Skull\n")
        f.write("Domain Air: +Scalp\n")

    # Create the conductivity file
    with open("bem_model/sphere_head.cond", "w") as f:
        f.write("# Properties Description 1.0 (Conductivities)\n\n")
        f.write(f"Air         {AIR_CONDUCTIVITY}\n")
        f.write(f"Scalp       {SCALP_CONDUCTIVITY}\n")
        f.write(f"Brain       {BRAIN_CONDUCTIVITY}\n")
        f.write(f"Skull       {SKULL_CONDUCTIVITY}\n")

    # Generate brain volume dipole positions for EEG/MEG (interior sources)
    brain_positions = get_grid_positions(grid_spacing_mm=10.0)
    n_brain_sources = len(brain_positions)
    brain_orientations = get_random_orientations(n_brain_sources)

    # Write brain volume dipoles to file (for EEG, MEG, ECoG)
    with open("bem_model/dipole_locations.txt", "w") as f:
        for pos, ori in zip(brain_positions, brain_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate cortical dipole positions for EIT (surface sources)
    cortical_positions = get_cortical_positions(n_sources=N_SOURCES_DEFAULT)
    n_cortical_sources = len(cortical_positions)
    cortical_orientations = get_random_orientations(n_cortical_sources)

    # Write cortical dipoles to separate file (for EIT)
    with open("bem_model/eit_dipole_locations.txt", "w") as f:
        for pos, ori in zip(cortical_positions, cortical_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate EEG sensor positions (scalp surface, positions only)
    eeg_sensor_positions = get_sensor_positions(N_SENSORS_DEFAULT)
    with open("bem_model/sensor_locations.txt", "w") as f:
        for pos in eeg_sensor_positions:
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    # Generate MEG sensor positions (further out, with orientations)
    # MEG sensors need to be positioned outside the head with radial orientations
    meg_sensor_positions = get_sensor_positions(
        N_SENSORS_DEFAULT, offset=20
    )  # 20mm further out

    # Calculate radial orientations (pointing inward toward center of head)
    head_center = np.array([SCALP_RADIUS, SCALP_RADIUS, 0])

    with open("bem_model/meg_sensor_locations.txt", "w") as f:
        for pos in meg_sensor_positions:
            # Calculate inward-pointing radial orientation
            direction = head_center - pos
            orientation = direction / np.linalg.norm(direction)
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{orientation[0]:.6f}\t{orientation[1]:.6f}\t{orientation[2]:.6f}\n"
            )

    print(
        f"EEG/MEG: Using {n_brain_sources} brain volume dipoles and {N_SENSORS_DEFAULT} sensors"
    )
    print(
        f"EIT: Using {n_cortical_sources} cortical dipoles and {N_SENSORS_DEFAULT} sensors"
    )


def create_bem_model_hemisphere():
    """Create a 3-layer hemisphere model with cortical sources."""
    # Ensure model directory exists
    os.makedirs("bem_model/", exist_ok=True)

    # Define the center of the hemispheres to match source/sensor coordinate system
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

    # Create the three hemisphere meshes
    for name, radius in [
        ("brain", BRAIN_RADIUS),
        ("skull", SKULL_RADIUS),
        ("scalp", SCALP_RADIUS),
    ]:
        vertices, triangles = create_hemisphere(radius, resolution=20, center=center)
        write_tri(f"bem_model/{name}_sphere.tri", vertices, triangles, center=center)

    # Create the geometry file (format 1.1)
    with open("bem_model/sphere_head.geom", "w") as f:
        f.write("# Domain Description 1.1\n\n")
        f.write("Interfaces 3\n\n")
        f.write('Interface Brain: "brain_sphere.tri"\n')
        f.write('Interface Skull: "skull_sphere.tri"\n')
        f.write('Interface Scalp: "scalp_sphere.tri"\n\n')
        f.write("Domains 4\n\n")
        f.write("Domain Brain: -Brain\n")
        f.write("Domain Skull: -Skull +Brain\n")
        f.write("Domain Scalp: -Scalp +Skull\n")
        f.write("Domain Air: +Scalp\n")

    # Create the conductivity file
    with open("bem_model/sphere_head.cond", "w") as f:
        f.write("# Properties Description 1.0 (Conductivities)\n\n")
        f.write(f"Air         {AIR_CONDUCTIVITY}\n")
        f.write(f"Scalp       {SCALP_CONDUCTIVITY}\n")
        f.write(f"Brain       {BRAIN_CONDUCTIVITY}\n")
        f.write(f"Skull       {SKULL_CONDUCTIVITY}\n")

    # Generate brain volume dipole positions for EEG/MEG (interior sources)
    brain_positions = get_grid_positions(grid_spacing_mm=10.0)
    n_brain_sources = len(brain_positions)
    brain_orientations = get_random_orientations(n_brain_sources)

    # Write brain volume dipoles to file (for EEG, MEG, ECoG)
    with open("bem_model/dipole_locations.txt", "w") as f:
        for pos, ori in zip(brain_positions, brain_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate cortical dipole positions for EIT (surface sources)
    cortical_positions = get_cortical_positions(n_sources=N_SOURCES_DEFAULT)
    n_cortical_sources = len(cortical_positions)
    cortical_orientations = get_random_orientations(n_cortical_sources)

    # Write cortical dipoles to separate file (for EIT)
    with open("bem_model/eit_dipole_locations.txt", "w") as f:
        for pos, ori in zip(cortical_positions, cortical_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate EEG sensor positions (scalp surface, positions only)
    eeg_sensor_positions = get_sensor_positions(N_SENSORS_DEFAULT)
    with open("bem_model/sensor_locations.txt", "w") as f:
        for pos in eeg_sensor_positions:
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    # Generate MEG sensor positions (further out, with orientations)
    # MEG sensors need to be positioned outside the head with radial orientations
    meg_sensor_positions = get_sensor_positions(
        N_SENSORS_DEFAULT, offset=20
    )  # 20mm further out

    # Calculate radial orientations (pointing inward toward center of head)
    head_center = np.array([SCALP_RADIUS, SCALP_RADIUS, 0])

    with open("bem_model/meg_sensor_locations.txt", "w") as f:
        for pos in meg_sensor_positions:
            # Calculate inward-pointing radial orientation
            direction = head_center - pos
            orientation = direction / np.linalg.norm(direction)
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{orientation[0]:.6f}\t{orientation[1]:.6f}\t{orientation[2]:.6f}\n"
            )

    print(
        f"Hemisphere model - EEG/MEG: Using {n_brain_sources} brain volume dipoles and {N_SENSORS_DEFAULT} sensors"
    )
    print(
        f"Hemisphere model - EIT: Using {n_cortical_sources} cortical dipoles and {N_SENSORS_DEFAULT} sensors"
    )


def _create_sphere_meshes(output_dir, grid_resolution, center=None):
    """Create 3-layer spherical meshes (brain, skull, scalp).

    Parameters
    ----------
    output_dir : str
        Directory to write mesh files
    grid_resolution : float
        Grid resolution in mm (mesh resolution)
    center : ndarray, optional
        Center of the spheres as (x, y, z). If None, defaults to (0, 0, 0).
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    for name, radius in [
        ("brain", BRAIN_RADIUS),
        ("skull", SKULL_RADIUS),
        ("scalp", SCALP_RADIUS),
    ]:
        vertices, triangles = create_sphere(
            radius, resolution=grid_resolution, center=center
        )
        write_tri(f"{output_dir}/{name}_sphere.tri", vertices, triangles, center=center)


def _write_geometry_file(output_dir):
    """Write OpenMEEG geometry file.

    Parameters
    ----------
    output_dir : str
        Directory to write geometry file
    """
    with open(f"{output_dir}/sphere_head.geom", "w") as f:
        f.write("# Domain Description 1.1\n\n")
        f.write("Interfaces 3\n\n")
        f.write('Interface Brain: "brain_sphere.tri"\n')
        f.write('Interface Skull: "skull_sphere.tri"\n')
        f.write('Interface Scalp: "scalp_sphere.tri"\n\n')
        f.write("Domains 4\n\n")
        f.write("Domain Brain: -Brain\n")
        f.write("Domain Skull: -Skull +Brain\n")
        f.write("Domain Scalp: -Scalp +Skull\n")
        f.write("Domain Air: +Scalp\n")


def _write_conductivity_file(output_dir):
    """Write OpenMEEG conductivity file.

    Parameters
    ----------
    output_dir : str
        Directory to write conductivity file
    """
    with open(f"{output_dir}/sphere_head.cond", "w") as f:
        f.write("# Properties Description 1.0 (Conductivities)\n\n")
        f.write(f"Air         {AIR_CONDUCTIVITY}\n")
        f.write(f"Scalp       {SCALP_CONDUCTIVITY}\n")
        f.write(f"Brain       {BRAIN_CONDUCTIVITY}\n")
        f.write(f"Skull       {SKULL_CONDUCTIVITY}\n")


def create_eeg_bem_model(
    source_spacing_mm=10.0, n_sensors=N_SENSORS_DEFAULT, grid_resolution=20
):
    """Create BEM model specifically for EEG with configurable parameters.

    Parameters
    ----------
    source_spacing_mm : float
        Spacing between dipole sources in mm (smaller = more sources)
    n_sensors : int
        Number of scalp EEG sensors
    grid_resolution : float
        Grid resolution in mm for spherical meshes (mesh resolution)
    """
    output_dir = "bem_model/eeg"
    os.makedirs(output_dir, exist_ok=True)

    # Define the center of the spheres to match source/sensor coordinate system
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

    # Create meshes, geometry, and conductivity files
    _create_sphere_meshes(output_dir, grid_resolution, center=center)
    _write_geometry_file(output_dir)
    _write_conductivity_file(output_dir)

    # Generate brain volume dipole positions
    brain_positions = get_grid_positions(grid_spacing_mm=source_spacing_mm)
    brain_orientations = get_random_orientations(len(brain_positions))

    # Write brain volume dipoles to file
    with open(f"{output_dir}/dipole_locations.txt", "w") as f:
        for pos, ori in zip(brain_positions, brain_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate EEG sensor positions (scalp surface)
    eeg_sensor_positions = get_sensor_positions(n_sensors)
    with open(f"{output_dir}/sensor_locations.txt", "w") as f:
        for pos in eeg_sensor_positions:
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    print(
        f"EEG BEM model: {len(brain_positions)} sources ({source_spacing_mm}mm spacing), {n_sensors} sensors, {grid_resolution}mm resolution"
    )


def create_meg_bem_model(
    source_spacing_mm=10.0,
    n_sensors=N_SENSORS_DEFAULT,
    grid_resolution=20,
    sensor_offset=20,
):
    """Create BEM model specifically for MEG with configurable parameters.

    Parameters
    ----------
    source_spacing_mm : float
        Spacing between dipole sources in mm (smaller = more sources)
    n_sensors : int
        Number of MEG sensors
    grid_resolution : float
        Grid resolution in mm for spherical meshes (mesh resolution)
    sensor_offset : float
        Distance in mm that MEG sensors are placed outside the scalp
        (5mm for OPMs, 20mm for SQUIDs)
    """
    output_dir = "bem_model/meg"
    os.makedirs(output_dir, exist_ok=True)

    # Define the center of the spheres to match source/sensor coordinate system
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

    # Create meshes, geometry, and conductivity files
    _create_sphere_meshes(output_dir, grid_resolution, center=center)
    _write_geometry_file(output_dir)
    _write_conductivity_file(output_dir)

    # Generate brain volume dipole positions
    brain_positions = get_grid_positions(grid_spacing_mm=source_spacing_mm)
    brain_orientations = get_random_orientations(len(brain_positions))

    # Write brain volume dipoles to file
    with open(f"{output_dir}/dipole_locations.txt", "w") as f:
        for pos, ori in zip(brain_positions, brain_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate MEG sensor positions (outside head with radial orientations)
    meg_sensor_positions = get_sensor_positions(n_sensors, offset=sensor_offset)
    head_center = np.array([SCALP_RADIUS, SCALP_RADIUS, 0])

    with open(f"{output_dir}/meg_sensor_locations.txt", "w") as f:
        for pos in meg_sensor_positions:
            # Calculate inward-pointing radial orientation
            direction = head_center - pos
            orientation = direction / np.linalg.norm(direction)
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{orientation[0]:.6f}\t{orientation[1]:.6f}\t{orientation[2]:.6f}\n"
            )

    sensor_type = (
        "OPMs"
        if sensor_offset == 5
        else "SQUIDs" if sensor_offset == 20 else f"{sensor_offset}mm offset"
    )
    print(
        f"MEG BEM model: {len(brain_positions)} sources ({source_spacing_mm}mm spacing), {n_sensors} sensors ({sensor_type}), {grid_resolution}mm resolution"
    )


def create_eit_bem_model(
    n_sources=N_SOURCES_DEFAULT, n_sensors=N_SENSORS_DEFAULT, grid_resolution=20
):
    """Create BEM model specifically for EIT with configurable parameters.

    Parameters
    ----------
    n_sources : int
        Number of cortical surface dipole sources
    n_sensors : int
        Number of scalp EIT electrodes
    grid_resolution : float
        Grid resolution in mm for spherical meshes (mesh resolution)
    """
    output_dir = "bem_model/eit"
    os.makedirs(output_dir, exist_ok=True)

    # Define the center of the spheres to match source/sensor coordinate system
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

    # Create meshes, geometry, and conductivity files
    _create_sphere_meshes(output_dir, grid_resolution, center=center)
    _write_geometry_file(output_dir)
    _write_conductivity_file(output_dir)

    # Generate cortical dipole positions (on brain surface)
    cortical_positions = get_cortical_positions(n_sources=n_sources)
    cortical_orientations = get_random_orientations(len(cortical_positions))

    # Write cortical dipoles to file
    with open(f"{output_dir}/eit_dipole_locations.txt", "w") as f:
        for pos, ori in zip(cortical_positions, cortical_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate EIT sensor positions (scalp surface, same as EEG)
    eit_sensor_positions = get_sensor_positions(n_sensors)
    with open(f"{output_dir}/sensor_locations.txt", "w") as f:
        for pos in eit_sensor_positions:
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    print(
        f"EIT BEM model: {len(cortical_positions)} cortical sources, {n_sensors} sensors, {grid_resolution}mm resolution"
    )


# ---- Bitrate calculations ----


def get_bitrate(
    svd_spectrum: np.ndarray,
    noise_full_brain: float,
    time_resolution: float = 1.0,
    n_detectors: int | None = None,
) -> float:
    return (1 / time_resolution) * np.sum(
        np.log2(
            1
            + svd_spectrum
            / (noise_full_brain / np.sqrt(n_detectors or len(svd_spectrum)))
        )
    )


def noise_floor_heuristic(
    svd_spectrum: np.ndarray,
    n_detectors: int | None = None,
    heuristic: Literal["power", "first"] = "power",
    factor: float = 10.0,
) -> float:
    n_detectors = n_detectors or len(svd_spectrum)
    if heuristic == "power":
        total_power = np.sum(np.abs(svd_spectrum) ** 2)
        return np.sqrt(total_power / n_detectors) / factor
    elif heuristic == "first":
        return svd_spectrum[0] / factor


# %%
