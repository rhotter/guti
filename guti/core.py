import os
import numpy as np
from pathlib import Path
import warnings
from typing import Literal

np.random.seed(239)

BRAIN_RADIUS = 80  # mm
CSF_RADIUS = 81
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
        [BRAIN_RADIUS, BRAIN_RADIUS, 0]
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


def _fibonacci_sphere_points(n_points, radius=1.0):
    """Generate uniformly distributed points on a sphere using Fibonacci spiral.

    Parameters
    ----------
    n_points : int
        Number of points to generate
    radius : float
        Radius of the sphere

    Returns
    -------
    points : ndarray of shape (n_points, 3)
        Uniformly distributed points on sphere surface
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 degrees

    # Generate points using Fibonacci spiral
    indices = np.arange(n_points)
    z = 1 - (2 * indices + 1) / n_points  # z goes from 1 to -1
    radius_at_z = np.sqrt(1 - z * z)  # radius at height z

    theta = golden_angle * indices

    x = radius_at_z * np.cos(theta)
    y = radius_at_z * np.sin(theta)

    points = np.column_stack([x, y, z]) * radius
    return points


def _delaunay_triangulation_sphere(points):
    """Create Delaunay triangulation for points on a sphere using scipy.

    Parameters
    ----------
    points : ndarray of shape (n_points, 3)
        Points on sphere surface

    Returns
    -------
    triangles : ndarray of shape (n_triangles, 3)
        Triangle indices (0-based)
    """
    from scipy.spatial import ConvexHull

    # For a sphere, convex hull gives us the correct triangulation
    hull = ConvexHull(points)
    return hull.simplices


def create_sphere(radius, n_phi=8, n_theta=8, resolution=None, center=None):
    """Create a sphere mesh with uniform Fibonacci spiral point distribution.

    Parameters
    ----------
    radius : float
        Radius of the sphere in meters
    n_phi : int, optional
        Approximate number of points (ignored if resolution is provided).
        Actual number will be adjusted for Fibonacci distribution.
    n_theta : int, optional
        Ignored (kept for backward compatibility)
    resolution : float, optional
        Desired grid spacing in meters. If provided, number of points is calculated
        to achieve approximately uniform spacing using surface area estimation.
    center : ndarray, optional
        Center of the sphere as (x, y, z). If None, defaults to (0, 0, 0).

    Returns
    -------
    vertices : ndarray
        Vertex coordinates
    triangles : ndarray
        Triangle indices (0-based)

    Notes
    -----
    Uses Fibonacci spiral for uniform point distribution on sphere surface.
    This provides much better uniformity than latitude-longitude grids.
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    if resolution is not None:
        # Calculate number of points needed for desired resolution
        # Surface area of sphere: A = 4πr²
        # Area per point ≈ resolution²
        # n_points ≈ 4πr² / resolution²
        surface_area = 4 * np.pi * radius**2
        area_per_point = resolution**2
        n_points = max(8, int(surface_area / area_per_point))
    else:
        # Use n_phi as rough guide for number of points
        n_points = max(8, n_phi * (n_theta if n_theta else n_phi // 2))

    # Generate uniformly distributed points using Fibonacci spiral
    vertices = _fibonacci_sphere_points(n_points, radius)

    # Create Delaunay triangulation
    triangles = _delaunay_triangulation_sphere(vertices)

    # Translate vertices to the specified center
    vertices = vertices + center

    print(
        f"Created sphere with {len(vertices)} vertices and {len(triangles)} triangles (Fibonacci distribution)"
    )

    return vertices, triangles


def create_hemisphere(radius, n_phi=8, n_theta=8, resolution=None, center=None):
    """Create a hemisphere mesh with uniform Fibonacci point distribution on curved surface.

    Parameters
    ----------
    radius : float
        Radius of the hemisphere in meters
    n_phi : int, optional
        Approximate number of points on curved surface (ignored if resolution is provided)
    n_theta : int, optional
        Ignored (kept for backward compatibility)
    resolution : float, optional
        Desired grid spacing in meters. If provided, number of points is calculated
        to achieve approximately uniform spacing.
    center : ndarray, optional
        Center of the hemisphere base as (x, y, z). If None, defaults to (0, 0, 0).
        The hemisphere extends upward (positive z direction) from this center.

    Returns
    -------
    vertices : ndarray
        Vertex coordinates
    triangles : ndarray
        Triangle indices (0-based)

    Notes
    -----
    Uses Fibonacci spiral for uniform point distribution on the curved hemisphere surface.
    The flat base is added separately with a circular boundary and center point.
    """
    from scipy.spatial import ConvexHull, Delaunay

    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    if resolution is not None:
        # Calculate number of points for curved surface
        # Surface area of hemisphere: A = 2πr²
        curved_surface_area = 2 * np.pi * radius**2
        area_per_point = resolution**2
        n_curved_points = max(8, int(curved_surface_area / area_per_point))

        # Number of points on base circle perimeter
        circumference = 2 * np.pi * radius
        n_base_points = max(8, int(circumference / resolution))
    else:
        n_curved_points = max(8, n_phi * (n_theta if n_theta else n_phi // 2) // 2)
        n_base_points = max(8, n_phi)

    # Generate Fibonacci points on full sphere, then filter for hemisphere
    n_full_sphere = n_curved_points * 2  # Generate more to get enough in hemisphere
    full_sphere_points = _fibonacci_sphere_points(n_full_sphere, radius)

    # Keep only points with z >= 0 (upper hemisphere)
    hemisphere_points = full_sphere_points[full_sphere_points[:, 2] >= 0]

    # If we don't have enough points, adjust
    if len(hemisphere_points) < n_curved_points:
        n_full_sphere = int(n_curved_points * 2.5)
        full_sphere_points = _fibonacci_sphere_points(n_full_sphere, radius)
        hemisphere_points = full_sphere_points[full_sphere_points[:, 2] >= 0]

    # Take approximately the desired number of points
    hemisphere_points = hemisphere_points[:n_curved_points]

    # Generate points on the base circle (z=0)
    theta_base = np.linspace(0, 2 * np.pi, n_base_points, endpoint=False)
    base_circle_points = np.column_stack([
        radius * np.cos(theta_base),
        radius * np.sin(theta_base),
        np.zeros(n_base_points)
    ])

    # Add center point of base
    base_center = np.array([[0.0, 0.0, 0.0]])

    # Combine all points
    vertices = np.vstack([hemisphere_points, base_circle_points, base_center])

    # Create triangulation
    # For the curved surface, use convex hull of hemisphere points only
    curved_hull = ConvexHull(hemisphere_points)
    curved_triangles = curved_hull.simplices

    # For the base, create triangles from center to circle
    n_curved = len(hemisphere_points)
    n_circle = len(base_circle_points)
    base_center_idx = len(vertices) - 1

    base_triangles = []
    for i in range(n_circle):
        # Triangle from center to two consecutive points on circle
        v1 = base_center_idx
        v2 = n_curved + i
        v3 = n_curved + (i + 1) % n_circle
        base_triangles.append([v1, v3, v2])  # Reversed for downward normal

    # Combine triangles
    triangles = np.vstack([curved_triangles, np.array(base_triangles)])

    # Translate vertices to the specified center
    vertices = vertices + center

    print(
        f"Created hemisphere with {len(vertices)} vertices and {len(triangles)} triangles (Fibonacci distribution)"
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


def get_random_orientations(n_sources: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors for dipole orientations (reproducible).

    Parameters
    ----------
    n_sources : int
        Number of dipole orientations to generate
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    orientations : ndarray of shape (n_sources, 3)
        Random unit vectors representing dipole orientations
    """
    # Use local RNG for reproducibility (doesn't affect global state)
    rng = np.random.RandomState(seed)
    # Generate random vectors
    orientations = rng.randn(n_sources, 3)
    # Normalize to unit vectors
    norms = np.linalg.norm(orientations, axis=1, keepdims=True)
    orientations = orientations / norms
    return orientations


def expand_positions_with_orientations(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand positions to have 3 orthogonal dipoles (x, y, z) at each location.

    Parameters
    ----------
    positions : ndarray of shape (n_positions, 3)
        Spatial positions for dipole sources

    Returns
    -------
    expanded_positions : ndarray of shape (3*n_positions, 3)
        Each position repeated 3 times
    expanded_orientations : ndarray of shape (3*n_positions, 3)
        Orthogonal orientations [1,0,0], [0,1,0], [0,0,1] tiled for each position

    Notes
    -----
    This creates a complete orthogonal basis at each spatial location, allowing
    any arbitrary dipole orientation to be represented as a linear combination.
    The output structure is: [pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z, ...]
    """
    n_positions = len(positions)

    # Create 3 orthogonal dipoles at each position
    orthogonal_basis = np.array([
        [1.0, 0.0, 0.0],  # x-direction
        [0.0, 1.0, 0.0],  # y-direction
        [0.0, 0.0, 1.0],  # z-direction
    ])

    # Expand positions: each position appears 3 times (once per orientation)
    expanded_positions = np.repeat(positions, 3, axis=0)

    # Tile orientations: [x,y,z, x,y,z, x,y,z, ...]
    expanded_orientations = np.tile(orthogonal_basis, (n_positions, 1))

    return expanded_positions, expanded_orientations


def create_radial_dipoles(
    n_radial_lines: int = 100,
    n_dipoles_per_line: int = 10,
    radius: float = BRAIN_RADIUS,
    center: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create dipoles uniformly distributed along radial lines in hemisphere.

    Parameters
    ----------
    n_radial_lines : int
        Number of radial lines from center to brain surface
    n_dipoles_per_line : int
        Number of dipoles along each radial line
    radius : float
        Radius of the brain hemisphere (default: BRAIN_RADIUS)
    center : ndarray, optional
        Center of the hemisphere as (x, y, z). If None, defaults to (radius, radius, 0).

    Returns
    -------
    positions : ndarray of shape (n_radial_lines * n_dipoles_per_line, 3)
        Dipole positions along radial lines
    orientations : ndarray of shape (n_radial_lines * n_dipoles_per_line, 3)
        Radial orientations (pointing outward from center)

    Notes
    -----
    This creates a physiologically realistic dipole distribution where:
    - Dipoles are arranged along radial lines from center to brain surface
    - All dipoles point radially outward (perpendicular to cortical surface)
    - Lines are uniformly distributed on hemisphere using Fibonacci spiral
    - Dipoles are uniformly spaced along each line from near-center to surface
    """
    if center is None:
        center = np.array([radius, radius, 0.0])

    # Generate uniform directions on hemisphere using Fibonacci spiral
    # These define the radial lines
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_radial_lines)
    z = (indices + 0.5) / n_radial_lines  # z from 0 to 1 for hemisphere
    theta = np.arccos(z)  # polar angle
    phi = golden_angle * indices  # azimuthal angle

    # Convert to unit vectors (radial directions)
    radial_directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    # Create dipoles along each radial line
    positions = []
    orientations = []

    # Radial distances from center to surface (don't go all the way to center)
    # Start at small radius (e.g., 10% of brain radius) to avoid singularity at center
    # Use r^3 spacing for uniform volume distribution in hemisphere
    min_radius_fraction = 0.1
    r_min = min_radius_fraction * radius
    r_max = radius
    r_cubed = np.linspace(r_min**3, r_max**3, n_dipoles_per_line)
    radial_distances = np.cbrt(r_cubed)

    for direction in radial_directions:
        for distance in radial_distances:
            # Position along the radial line
            pos = center + distance * direction
            positions.append(pos)

            # Orientation points radially outward
            orientations.append(direction)

    positions = np.array(positions)
    orientations = np.array(orientations)

    print(
        f"Created {n_radial_lines} radial lines × {n_dipoles_per_line} dipoles/line = {len(positions)} total dipoles"
    )

    return positions, orientations


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


def create_bem_model(mesh_resolution=20, source_spacing_mm=5.0, n_cortical_sources=N_SOURCES_DEFAULT, n_sensors=N_SENSORS_DEFAULT, meg_sensor_offset=20, output_dir="bem_model/"):
    """Create a 3-layer spherical model with cortical sources."""
    # Ensure model directory exists
    os.makedirs(output_dir, exist_ok=True)

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
        vertices, triangles = create_sphere(radius, resolution=mesh_resolution, center=center)
        write_tri(f"{output_dir}/{name}_sphere.tri", vertices, triangles, center=center)

    # Create the geometry file (format 1.1)
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

    # Create the conductivity file
    with open(f"{output_dir}/sphere_head.cond", "w") as f:
        f.write("# Properties Description 1.0 (Conductivities)\n\n")
        f.write(f"Air         {AIR_CONDUCTIVITY}\n")
        f.write(f"Scalp       {SCALP_CONDUCTIVITY}\n")
        f.write(f"Brain       {BRAIN_CONDUCTIVITY}\n")
        f.write(f"Skull       {SKULL_CONDUCTIVITY}\n")

    # Generate brain volume dipole positions for EEG/MEG (interior sources)
    brain_positions = get_grid_positions(grid_spacing_mm=source_spacing_mm)
    brain_positions_expanded, brain_orientations_expanded = expand_positions_with_orientations(brain_positions)

    # Write brain volume dipoles to file (for EEG, MEG, ECoG)
    with open(f"{output_dir}/dipole_locations.txt", "w") as f:
        for pos, ori in zip(brain_positions_expanded, brain_orientations_expanded):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate cortical dipole positions for EIT (surface sources)
    cortical_positions = get_cortical_positions(n_sources=n_cortical_sources)
    n_cortical_sources = len(cortical_positions)
    cortical_orientations = get_random_orientations(n_cortical_sources)

    # Write cortical dipoles to separate file (for EIT)
    with open(f"{output_dir}/eit_dipole_locations.txt", "w") as f:
        for pos, ori in zip(cortical_positions, cortical_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate EEG sensor positions (scalp surface, positions only)
    eeg_sensor_positions = get_sensor_positions(n_sensors)
    with open(f"{output_dir}/sensor_locations.txt", "w") as f:
        for pos in eeg_sensor_positions:
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    # Generate MEG sensor positions (further out, with orientations)
    # MEG sensors need to be positioned outside the head with radial orientations
    meg_sensor_positions = get_sensor_positions(
        n_sensors, offset=meg_sensor_offset
    )  # 20mm further out

    # Calculate radial orientations (pointing inward toward center of head)
    head_center = np.array([SCALP_RADIUS, SCALP_RADIUS, 0])

    with open(f"{output_dir}/meg_sensor_locations.txt", "w") as f:
        for pos in meg_sensor_positions:
            # Calculate inward-pointing radial orientation
            direction = head_center - pos
            orientation = direction / np.linalg.norm(direction)
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{orientation[0]:.6f}\t{orientation[1]:.6f}\t{orientation[2]:.6f}\n"
            )

    print(
        f"EEG/MEG: Using {len(brain_positions)} positions × 3 orientations = {len(brain_positions_expanded)} dipoles and {N_SENSORS_DEFAULT} sensors"
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
    brain_positions_expanded, brain_orientations_expanded = expand_positions_with_orientations(brain_positions)

    # Write brain volume dipoles to file (for EEG, MEG, ECoG)
    with open("bem_model/dipole_locations.txt", "w") as f:
        for pos, ori in zip(brain_positions_expanded, brain_orientations_expanded):
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
        f"Hemisphere model - EEG/MEG: Using {len(brain_positions)} positions × 3 orientations = {len(brain_positions_expanded)} dipoles and {N_SENSORS_DEFAULT} sensors"
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
    source_spacing_mm=10.0,
    n_sensors=N_SENSORS_DEFAULT,
    grid_resolution=20,
    output_dir="bem_model/eeg",
    n_radial_lines=None,
    n_dipoles_per_line=None,
    use_radial_orientations=False,
    source_radius_margin_mm=0.0,
):
    """Create BEM model specifically for EEG with configurable parameters.

    Parameters
    ----------
    source_spacing_mm : float
        Spacing between dipole sources in mm (smaller = more sources).
        Only used if n_radial_lines is None (grid-based method).
    n_sensors : int
        Number of scalp EEG sensors
    grid_resolution : float
        Grid resolution in mm for spherical meshes (mesh resolution)
    output_dir : str
        Directory to write BEM model files
    n_radial_lines : int, optional
        Number of radial lines from center to brain surface.
        If specified, uses radial dipole distribution instead of grid.
    n_dipoles_per_line : int, optional
        Number of dipoles along each radial line.
        Required if n_radial_lines is specified.
    use_radial_orientations : bool, optional
        If True, creates a single radial dipole at each position pointing outward
        (normal to surface), instead of 3 orthogonal dipoles. Works with both
        radial lines method and grid-based method.
    source_radius_margin_mm : float, optional
        Exclude grid sources closer than this distance to the brain boundary.
        OpenMEEG rejects dipoles exactly on an interface, so clean sweeps can set
        a tiny positive margin without changing existing default behavior.

    Notes
    -----
    Four dipole distribution methods are supported:

    1. **Radial lines with radial orientations**: Single radial dipole at each point
       - Set n_radial_lines, n_dipoles_per_line, and use_radial_orientations=True
       - Physiologically realistic (radial orientation like pyramidal neurons)
       - Total dipoles = n_radial_lines × n_dipoles_per_line

    2. **Radial lines with orthogonal basis**: 3 orthogonal dipoles at each point
       - Set n_radial_lines, n_dipoles_per_line, and use_radial_orientations=False
       - Captures complete orientation space at each location along radial lines
       - Total dipoles = 3 × n_radial_lines × n_dipoles_per_line

    3. **Grid method with orthogonal basis**: 3 orthogonal dipoles at each grid point
       - Set source_spacing_mm (leave n_radial_lines=None, use_radial_orientations=False)
       - Captures complete orientation space at each location
       - Total dipoles = 3 × number_of_grid_points

    4. **Grid method with radial orientations**: Single radial dipole at each grid point
       - Set source_spacing_mm and use_radial_orientations=True
       - Dipoles point radially outward (normal to brain surface)
       - Total dipoles = number_of_grid_points
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the center of the spheres to match source/sensor coordinate system
    center = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])

    # Create meshes, geometry, and conductivity files
    _create_sphere_meshes(output_dir, grid_resolution, center=center)
    _write_geometry_file(output_dir)
    _write_conductivity_file(output_dir)

    # Generate brain volume dipole positions and orientations
    if n_radial_lines is not None:
        # Use radial lines distribution
        if n_dipoles_per_line is None:
            raise ValueError("n_dipoles_per_line must be specified when using n_radial_lines")

        # First create radial dipoles (positions along radial lines with radial orientations)
        radial_positions, radial_orientations = create_radial_dipoles(
            n_radial_lines=n_radial_lines,
            n_dipoles_per_line=n_dipoles_per_line,
            radius=BRAIN_RADIUS,
            center=center,
        )

        if use_radial_orientations:
            # Use single radial dipole at each position
            dipole_positions = radial_positions
            dipole_orientations = radial_orientations
        else:
            # Expand to 3 orthogonal dipoles at each radial position
            dipole_positions, dipole_orientations = expand_positions_with_orientations(radial_positions)
    else:
        # Use grid-based distribution
        brain_positions = get_grid_positions(grid_spacing_mm=source_spacing_mm)
        if source_radius_margin_mm > 0:
            distances = np.linalg.norm(brain_positions - center, axis=1)
            brain_positions = brain_positions[
                distances < BRAIN_RADIUS - source_radius_margin_mm
            ]

        if use_radial_orientations:
            # Single radial dipole at each position pointing outward
            dipole_positions = brain_positions
            # Calculate radial orientations (normalized direction from center)
            dipole_orientations = brain_positions - center
            dipole_orientations = dipole_orientations / np.linalg.norm(dipole_orientations, axis=1, keepdims=True)
        else:
            # 3 orthogonal dipoles per position
            dipole_positions, dipole_orientations = expand_positions_with_orientations(brain_positions)

    # Write brain volume dipoles to file
    with open(f"{output_dir}/dipole_locations.txt", "w") as f:
        for pos, ori in zip(dipole_positions, dipole_orientations):
            f.write(
                f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t{ori[0]:.6f}\t{ori[1]:.6f}\t{ori[2]:.6f}\n"
            )

    # Generate EEG sensor positions (scalp surface)
    eeg_sensor_positions = get_sensor_positions(n_sensors)
    with open(f"{output_dir}/sensor_locations.txt", "w") as f:
        for pos in eeg_sensor_positions:
            f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    if n_radial_lines is not None:
        orientation_str = "radial" if use_radial_orientations else "orthogonal basis"
        method_str = f"{n_radial_lines} radial lines × {n_dipoles_per_line} dipoles/line, {orientation_str}"
    else:
        orientation_str = "radial" if use_radial_orientations else "orthogonal basis"
        method_str = f"grid-based ({source_spacing_mm}mm spacing, {orientation_str})"

    print(
        f"EEG BEM model: {len(dipole_positions)} dipoles ({method_str}), {n_sensors} sensors, {grid_resolution}mm resolution"
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
    expanded_positions, expanded_orientations = expand_positions_with_orientations(brain_positions)

    # Write brain volume dipoles to file
    with open(f"{output_dir}/dipole_locations.txt", "w") as f:
        for pos, ori in zip(expanded_positions, expanded_orientations):
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
        f"MEG BEM model: {len(brain_positions)} positions × 3 orientations = {len(expanded_positions)} dipoles ({source_spacing_mm}mm spacing), {n_sensors} sensors ({sensor_type}), {grid_resolution}mm resolution"
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
    s: np.ndarray, # svd spectrum
    noise: float, # noise level
    time_resolution: float = 1.0,
) -> float:
    return (1 / (2*time_resolution)) * np.sum(
        np.log2(1 + (s/noise)**2)
    )


def get_bitrate_temporal_filter(
    s: np.ndarray,
    noise: float,
    freqs: np.ndarray,
    H_magnitude: np.ndarray,
) -> float:
    """Capacity in bits/s for a spatial spectrum followed by a temporal filter."""
    if len(freqs) < 2:
        return 0.0
    df = float(freqs[1] - freqs[0])
    sigma_eff = np.outer(np.asarray(s), np.asarray(H_magnitude)).ravel()
    return df * float(np.sum(np.log2(1 + (sigma_eff / noise) ** 2)))


def noise_floor_from_total_snr(
    s: np.ndarray,
    total_snr: float,
) -> float:
    """
    Convert a target total output SNR into an equivalent flat detector noise level.

    For iid unit-variance inputs, total output SNR is
    sqrt(sum_i s_i^2) / noise, so the matching flat noise floor is
    sqrt(sum_i s_i^2) / total_snr.
    """
    total_power = np.sum(np.abs(s) ** 2)
    return np.sqrt(total_power) / total_snr

def water_filling_spectrum(
    s: np.ndarray, # svd spectrum
    snr: float, # noise level
) -> np.ndarray:
    """
    P_i is either 0 or mu - (noise/s)**2, where mu is the water level.
    """
    # Binary search for water level mu
    # We want: sum_k s_k^2 * max(0, mu_tilde - (1/s_k)^2) = snr^2

    # Precompute (noise/s_k)^2 for all k
    reciprocal_s_squared = (1 / s) ** 2

    # Sort in ascending order for water-filling
    sorted_indices = np.argsort(reciprocal_s_squared)
    reciprocal_s_squared_sorted = reciprocal_s_squared[sorted_indices]
    s_sorted = s[sorted_indices]
    s_sq_sorted = s_sorted ** 2

    # Binary search bounds
    mu_tilde_min = 0.0
    mu_tilde_max = reciprocal_s_squared_sorted[-1] + snr**2 / s_sq_sorted.min()

    tolerance = 1e-10
    max_iterations = 1000

    for _ in range(max_iterations):
        mu_tilde = (mu_tilde_min + mu_tilde_max) / 2

        # Compute total output power for this mu
        power_allocation_tilde = np.maximum(0, mu_tilde - reciprocal_s_squared_sorted)
        output_snr_squared = np.sum(s_sq_sorted * power_allocation_tilde)

        if abs(output_snr_squared - snr**2) < tolerance:
            break

        if output_snr_squared < snr**2:
            mu_tilde_min = mu_tilde
        else:
            mu_tilde_max = mu_tilde

    # Compute final power allocation and unsort
    power_allocation_tilde = np.maximum(0, mu_tilde - reciprocal_s_squared_sorted)
    P_tilde = np.zeros_like(s)
    P_tilde[sorted_indices] = power_allocation_tilde

    return P_tilde

def total_iid_input_power(
    s: np.ndarray, # svd spectrum
    total_output_power: float,
) -> np.ndarray:
    return total_output_power / (s**2).sum()


def get_bitrate_channel_capacity(
    s: np.ndarray, # svd spectrum
    snr_at_reference_nsensors: float, # snr level
    nsensors_reference: int | None = None, # reference number of sensors
    n_sensors: int | None = None, # number of sensors
    time_resolution: float = 1.0,
    sensor_count_snr_exponent: float = 0.5,
) -> float:
    if nsensors_reference is None or n_sensors is None:
        snr = snr_at_reference_nsensors
    else:
        snr = snr_at_reference_nsensors * (
            nsensors_reference / n_sensors
        ) ** sensor_count_snr_exponent
    optimal_input_power_spectrum_over_noise = water_filling_spectrum(s, snr)
    channel_capacity = (1 / (2 * time_resolution)) * np.sum(
        np.log2(1 + optimal_input_power_spectrum_over_noise*s**2)
    )
    return channel_capacity


def get_bitrate_channel_capacity_temporal(*args, **kwargs):
    """Deprecated. Use :func:`guti.capacity.get_bitrate_temporal_filter`.

    The temporal spatial-filter bitrate now lives in ``guti.capacity`` under the
    project-wide equal-power (i.i.d. input) convention, alongside the rest of the
    capacity/bitrate utilities. This older water-filling variant has been removed
    to keep a single source of truth; import the new function directly instead.
    """
    raise NotImplementedError(
        "get_bitrate_channel_capacity_temporal has been removed. Use "
        "guti.capacity.get_bitrate_temporal_filter (equal-power convention)."
    )


def noise_floor_heuristic(
    s: np.ndarray, # svd spectrum
    n_detectors: int | None = None, # TODO: include this, right now it's implicitly included in total_power
    heuristic: Literal["power", "first"] = "power",
    snr: float = 10.0,
) -> float:
    # n_detectors = n_detectors or 1
    if heuristic == "power":
        return noise_floor_from_total_snr(s, snr)
    elif heuristic == "first":
        return s[0] / snr


# %%
