#!/usr/bin/env python
import argparse
import numpy as np
import pyvista as pv
from pathlib import Path

def read_brainvisa_tri(path: str):
    """Parse BrainVisa .tri with per-vertex normals (0-based indices)."""
    pts, norms, faces = [], [], []
    with open(path) as f:
        # --- vertex section header: "- N"
        for line in f:
            line = line.strip()
            if line.startswith("-"):
                parts = line.split()
                if len(parts) >= 2:
                    npts = int(parts[1])
                    break
        else:
            raise ValueError("No vertex section header ('- N') found")

        # vertices: x y z nx ny nz
        for _ in range(npts):
            toks = f.readline().strip().split()
            if len(toks) < 6:
                raise ValueError("Each vertex line must have 6 numbers: x y z nx ny nz")
            x, y, z, nx, ny, nz = map(float, toks[:6])
            pts.append((x, y, z))
            norms.append((nx, ny, nz))

        # --- triangle section header: "- N N N"
        for line in f:
            line = line.strip()
            if line.startswith("-"):
                parts = line.split()
                if len(parts) >= 2:
                    ntri = int(parts[1])
                    break
        else:
            raise ValueError("No triangle section header ('- N N N') found")

        for _ in range(ntri):
            i, j, k = map(int, f.readline().strip().split()[:3])
            faces.append((i, j, k))  # BrainVisa .tri uses 0-based indices

    pts = np.asarray(pts, float)
    norms = np.asarray(norms, float)
    tri = np.asarray(faces, int)
    return pts, norms, tri

def read_dipoles(path: str):
    """Parse dipoles text: one per line -> x y z nx ny nz."""
    pos, vec = [], []
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith(("#", "%", "//")):
                continue
            toks = s.split()
            if len(toks) < 6:
                raise ValueError(f"Bad dipole line (need 6 floats): {raw!r}")
            x, y, z, nx, ny, nz = map(float, toks[:6])
            pos.append((x, y, z))
            vec.append((nx, ny, nz))
    if not pos:
        raise ValueError("No dipoles parsed")
    return np.asarray(pos, float), np.asarray(vec, float)

def read_sensors(path: str):
    """Parse sensor positions: one per line -> x y z (ignore extra tokens)."""
    pts = []
    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith(("#", "%", "//")):
                continue
            toks = s.split()
            if len(toks) < 3:
                raise ValueError(f"Bad sensor line (need at least 3 floats): {raw!r}")
            x, y, z = map(float, toks[:3])
            pts.append((x, y, z))
    if not pts:
        raise ValueError("No sensors parsed")
    return np.asarray(pts, float)

def read_geom_file(geom_path: str):
    """Parse OpenMEEG .geom file to extract interface file names.

    Returns
    -------
    dict
        Dictionary mapping interface names to .tri file paths
    """
    import re
    geom_dir = Path(geom_path).parent
    interfaces = {}

    with open(geom_path) as f:
        for line in f:
            line = line.strip()
            # Match lines like: Interface Brain: "brain_sphere.tri"
            match = re.match(r'Interface\s+(\w+):\s+"([^"]+)"', line)
            if match:
                name, filename = match.groups()
                interfaces[name] = str(geom_dir / filename)

    return interfaces

def visualize_bem_layers(
    geom_path: str,
    dipole_path: str = None,
    sensor_path: str = None,
    show_edges: bool = True,
    normalize_dipoles: bool = True,
    dipole_length: float = None,
    dipole_size: float = 10.0,
    sensor_size: float = 12.0,
    screenshot: str = None,
    layer_opacity: dict = None,
    layer_colors: dict = None,
):
    """Visualize all BEM layers from a .geom file with dipoles and sensors.

    Parameters
    ----------
    geom_path : str
        Path to .geom file
    dipole_path : str, optional
        Path to dipole file (x y z nx ny nz per line)
    sensor_path : str, optional
        Path to sensor file (x y z per line)
    show_edges : bool, default=True
        Show triangle edges
    normalize_dipoles : bool, default=True
        Normalize dipole orientations
    dipole_length : float, optional
        Arrow length (default: 5% of mesh diagonal)
    dipole_size : float, default=10.0
        Point size for dipole locations (pixels)
    sensor_size : float, default=12.0
        Point size for sensors (pixels)
    screenshot : str, optional
        Path to save screenshot
    layer_opacity : dict, optional
        Opacity for each layer (e.g., {"Brain": 0.3, "Skull": 0.2, "Scalp": 0.8})
    layer_colors : dict, optional
        Color for each layer (e.g., {"Brain": "pink", "Skull": "white", "Scalp": "beige"})
    """
    # Default opacity values
    if layer_opacity is None:
        layer_opacity = {"Brain": 0.3, "Skull": 0.2, "Scalp": 0.8}

    # Default colors for each layer
    if layer_colors is None:
        layer_colors = {
            "Brain": "pink",
            "Skull": "white",
            "Scalp": "beige",
        }

    # Read .geom file to get interface meshes
    interfaces = read_geom_file(geom_path)

    # Create plotter
    p = pv.Plotter()

    # Load and add each mesh layer
    for name, tri_path in interfaces.items():
        pts, norms, tri = read_brainvisa_tri(tri_path)
        faces = np.hstack([np.full((tri.shape[0], 1), 3, int), tri]).ravel()
        mesh = pv.PolyData(pts, faces)

        if norms.shape == pts.shape:
            mesh["Normals"] = norms
        else:
            mesh.compute_normals(inplace=True, auto_orient_normals=True)

        # Add mesh with appropriate opacity and color
        p.add_mesh(
            mesh,
            smooth_shading=True,
            show_edges=show_edges,
            lighting=True,
            specular=0.2,
            opacity=layer_opacity.get(name, 0.5),
            color=layer_colors.get(name, "gray"),
            label=name,
        )

    p.add_axes()
    p.show_bounds(grid='front', location='outer')
    p.add_camera_orientation_widget()
    p.add_legend()

    # Dipoles
    if dipole_path and Path(dipole_path).exists():
        dip_pos, dip_vec = read_dipoles(dipole_path)

        vec = dip_vec.copy()
        if normalize_dipoles:
            lens = np.linalg.norm(vec, axis=1)
            nz = lens > 0
            vec[nz] = vec[nz] / lens[nz, None]

        # Default arrow length = 5% of first mesh bounding-box diagonal
        if dipole_length is None:
            first_mesh_path = list(interfaces.values())[0]
            pts, _, _ = read_brainvisa_tri(first_mesh_path)
            bounds = [pts[:, i].min() for i in range(3)] + [pts[:, i].max() for i in range(3)]
            diag = np.sqrt(sum((bounds[i+3] - bounds[i])**2 for i in range(3)))
            factor = 0.05 * diag if np.isfinite(diag) and diag > 0 else 1.0
        else:
            factor = float(dipole_length)

        # Points (small spheres at dipole locations)
        p.add_points(dip_pos, render_points_as_spheres=True, point_size=dipole_size, color="yellow")

        # Arrows (glyphs)
        dip_pd = pv.PolyData(dip_pos)
        dip_pd["vectors"] = vec
        arrows = dip_pd.glyph(orient="vectors", scale=False, factor=factor)
        p.add_mesh(arrows, color="yellow")

    # Sensors (red circles)
    if sensor_path and Path(sensor_path).exists():
        s_pos = read_sensors(sensor_path)
        p.add_points(
            s_pos,
            render_points_as_spheres=True,
            point_size=sensor_size,
            color="red",
            name="sensors",
        )

    # Show / screenshot
    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        p.show(screenshot=screenshot)
    else:
        p.show()

def visualize_bem_model(
    tri_path: str,
    dipole_path: str = None,
    sensor_path: str = None,
    show_edges: bool = True,
    normalize_dipoles: bool = True,
    dipole_length: float = None,
    dipole_size: float = 10.0,
    sensor_size: float = 12.0,
    screenshot: str = None,
):
    """Visualize BEM model with dipoles and sensors.

    Parameters
    ----------
    tri_path : str
        Path to .tri mesh file
    dipole_path : str, optional
        Path to dipole file (x y z nx ny nz per line)
    sensor_path : str, optional
        Path to sensor file (x y z per line)
    show_edges : bool, default=True
        Show triangle edges
    normalize_dipoles : bool, default=True
        Normalize dipole orientations
    dipole_length : float, optional
        Arrow length (default: 5% of mesh diagonal)
    dipole_size : float, default=10.0
        Point size for dipole locations (pixels)
    sensor_size : float, default=12.0
        Point size for sensors (pixels)
    screenshot : str, optional
        Path to save screenshot
    """
    # Load mesh
    pts, norms, tri = read_brainvisa_tri(tri_path)
    faces = np.hstack([np.full((tri.shape[0], 1), 3, int), tri]).ravel()
    mesh = pv.PolyData(pts, faces)

    if norms.shape == pts.shape:
        mesh["Normals"] = norms
    else:
        mesh.compute_normals(inplace=True, auto_orient_normals=True)

    # Plotter
    p = pv.Plotter()
    p.add_mesh(mesh, smooth_shading=True, show_edges=show_edges, lighting=True, specular=0.2)
    p.add_axes()
    p.show_bounds(grid='front', location='outer')
    p.add_camera_orientation_widget()

    # Dipoles
    if dipole_path:
        dip_pos, dip_vec = read_dipoles(dipole_path)

        vec = dip_vec.copy()
        if normalize_dipoles:
            lens = np.linalg.norm(vec, axis=1)
            nz = lens > 0
            vec[nz] = vec[nz] / lens[nz, None]

        # Default arrow length = 5% of mesh bounding-box diagonal
        if dipole_length is None:
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            diag = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
            factor = 0.05 * diag if np.isfinite(diag) and diag > 0 else 1.0
        else:
            factor = float(dipole_length)

        # Points (small spheres at dipole locations)
        p.add_points(dip_pos, render_points_as_spheres=True, point_size=dipole_size, color="white")

        # Arrows (glyphs)
        dip_pd = pv.PolyData(dip_pos)
        dip_pd["vectors"] = vec
        arrows = dip_pd.glyph(orient="vectors", scale=False, factor=factor)  # constant length
        p.add_mesh(arrows, color="white")

    # Sensors (red circles)
    if sensor_path:
        s_pos = read_sensors(sensor_path)
        # Spherical point markers -> appear as red circles in the viewer
        p.add_points(
            s_pos,
            render_points_as_spheres=True,
            point_size=sensor_size,
            color="red",
            name="sensors",
        )

    # Show / screenshot
    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        p.show(screenshot=screenshot)
    else:
        p.show()

def main():
    ap = argparse.ArgumentParser(
        description="Interactive viewer for BrainVisa .tri meshes or .geom BEM models (+ dipoles, + sensors)."
    )

    # Mesh input - either single .tri or .geom file
    mesh_group = ap.add_mutually_exclusive_group(required=True)
    mesh_group.add_argument("--tri", help=".tri file path (single mesh)")
    mesh_group.add_argument("--geom", help=".geom file path (multi-layer BEM model)")

    ap.add_argument("--edges", action="store_true", help="show triangle edges")
    ap.add_argument("--screenshot", help="save screenshot to this path")

    # Dipoles
    ap.add_argument("--dip", help="dipole file (.dip/.txt): each line x y z nx ny nz")
    ap.add_argument("--no-normalize", action="store_true", help="do NOT normalize dipole orientations")
    ap.add_argument("--dip-len", type=float, default=None, help="arrow length (same units as mesh). Default = 5% of mesh diagonal")
    ap.add_argument("--dip-size", type=float, default=10.0, help="point size for dipole locations (pixels)")

    # Sensors
    ap.add_argument("--sensors", help="sensor positions file (.txt): each line x y z")
    ap.add_argument("--sensor-size", type=float, default=12.0, help="point size for sensors (pixels)")

    # Layer customization (for .geom only)
    ap.add_argument("--layer-opacity", help='layer opacity as JSON (e.g., \'{"Brain": 0.3, "Skull": 0.2, "Scalp": 0.8}\')')
    ap.add_argument("--layer-colors", help='layer colors as JSON (e.g., \'{"Brain": "pink", "Skull": "white", "Scalp": "beige"}\')')

    args = ap.parse_args()

    # Parse layer customization options (for .geom mode)
    layer_opacity = None
    layer_colors = None
    if args.layer_opacity:
        import json
        layer_opacity = json.loads(args.layer_opacity)
    if args.layer_colors:
        import json
        layer_colors = json.loads(args.layer_colors)

    # Two modes: single .tri or multi-layer .geom
    if args.geom:
        # Multi-layer BEM model mode
        visualize_bem_layers(
            geom_path=args.geom,
            dipole_path=args.dip,
            sensor_path=args.sensors,
            show_edges=args.edges,
            normalize_dipoles=not args.no_normalize,
            dipole_length=args.dip_len,
            dipole_size=args.dip_size,
            sensor_size=args.sensor_size,
            screenshot=args.screenshot,
            layer_opacity=layer_opacity,
            layer_colors=layer_colors,
        )
    else:
        # Single .tri mesh mode
        visualize_bem_model(
            tri_path=args.tri,
            dipole_path=args.dip,
            sensor_path=args.sensors,
            show_edges=args.edges,
            normalize_dipoles=not args.no_normalize,
            dipole_length=args.dip_len,
            dipole_size=args.dip_size,
            sensor_size=args.sensor_size,
            screenshot=args.screenshot,
        )

if __name__ == "__main__":
    main()