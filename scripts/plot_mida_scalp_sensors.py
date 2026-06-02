"""Plot MIDA scalp sensors in GUTI coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from guti.core import SCALP_RADIUS, get_grid_positions, get_sensor_positions
from guti.mida_geometry import (
    GUTI_HEAD_CENTER_MM,
    load_mida_surface_triangles,
    mida_eeg_bem_layers,
    mida_scalp_triangles,
)


LAYER_COLORS = {
    "Brain": "#2563eb",
    "CSF": "#06b6d4",
    "Dura": "#a855f7",
    "SkullInnerTable": "#f59e0b",
    "SkullDiploe": "#f97316",
    "SkullOuterTable": "#dc2626",
    "SubcutaneousAdipose": "#22c55e",
    "Scalp": "#64748b",
}


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def _legacy_hemisphere_lines() -> list[np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    phi = np.linspace(0.0, 0.5 * np.pi, 24)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x = GUTI_HEAD_CENTER_MM[0] + SCALP_RADIUS * np.sin(phi_grid) * np.cos(theta_grid)
    y = GUTI_HEAD_CENTER_MM[1] + SCALP_RADIUS * np.sin(phi_grid) * np.sin(theta_grid)
    z = GUTI_HEAD_CENTER_MM[2] + SCALP_RADIUS * np.cos(phi_grid)
    lines = []
    for row in range(0, x.shape[0], 4):
        lines.append(np.column_stack([x[row], y[row], z[row]]))
    for col in range(0, x.shape[1], 6):
        lines.append(np.column_stack([x[:, col], y[:, col], z[:, col]]))
    return lines


def _plot_legacy_hemisphere(ax) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    phi = np.linspace(0.0, 0.5 * np.pi, 24)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x = GUTI_HEAD_CENTER_MM[0] + SCALP_RADIUS * np.sin(phi_grid) * np.cos(theta_grid)
    y = GUTI_HEAD_CENTER_MM[1] + SCALP_RADIUS * np.sin(phi_grid) * np.sin(theta_grid)
    z = GUTI_HEAD_CENTER_MM[2] + SCALP_RADIUS * np.cos(phi_grid)
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=4,
        cstride=6,
        color="#6b7280",
        linewidth=0.35,
        alpha=0.22,
    )


def _rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    rz = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev), -np.sin(elev)],
            [0.0, np.sin(elev), np.cos(elev)],
        ]
    )
    return rx @ rz


def _project_points(
    points: np.ndarray,
    *,
    center: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotated = (points - center) @ rotation.T
    xy = np.empty((len(points), 2), dtype=np.float64)
    xy[:, 0] = width / 2.0 + rotated[:, 0] * scale
    xy[:, 1] = height / 2.0 - rotated[:, 1] * scale
    return xy, rotated[:, 2]


def _plot_with_pillow(
    *,
    sensors: np.ndarray,
    scalp: np.ndarray,
    sources: np.ndarray,
    layers: list[tuple[str, np.ndarray, str]],
    n_sensors: int,
    output_path: Path,
    region: str,
    view: tuple[float, float],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 1250
    margin = 120
    all_points = np.vstack([scalp, sensors, sources, *(points for _, points, _ in layers)])
    center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
    rotation = _rotation_matrix(view[0], view[1])
    rotated = (all_points - center) @ rotation.T
    max_span = max(float(np.ptp(rotated[:, 0])), float(np.ptp(rotated[:, 1])), 1.0)
    scale = (min(width, height) - 2 * margin) / max_span

    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image, "RGBA")

    def draw_points(points: np.ndarray, radius: float, color: tuple[int, int, int, int]) -> None:
        xy, depth = _project_points(
            points,
            center=center,
            rotation=rotation,
            scale=scale,
            width=width,
            height=height,
        )
        order = np.argsort(depth)
        for x, y in xy[order]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    for line in _legacy_hemisphere_lines():
        xy, _ = _project_points(
            line,
            center=center,
            rotation=rotation,
            scale=scale,
            width=width,
            height=height,
        )
        draw.line([tuple(p) for p in xy], fill=(107, 114, 128, 55), width=1)

    for _, points, color in layers:
        rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        draw_points(points, 1.0, (*rgb, 48))

    draw_points(sources, 1.6, (37, 99, 235, 36))

    xy, depth = _project_points(
        sensors,
        center=center,
        rotation=rotation,
        scale=scale,
        width=width,
        height=height,
    )
    for x, y in xy[np.argsort(depth)]:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 255, 255, 235))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(220, 38, 38, 240))

    font = ImageFont.load_default()
    draw.text(
        (36, 28),
        f"MIDA layered head + scalp sensors ({n_sensors} sensors, {region} region)",
        fill=(15, 23, 42, 255),
        font=font,
    )
    draw.text(
        (36, 54),
        "colored: MIDA BEM layers   red: sensors   blue: brain source grid   wire: old hemisphere",
        fill=(71, 85, 105, 255),
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _plot_with_matplotlib(
    *,
    sensors: np.ndarray,
    scalp: np.ndarray,
    sources: np.ndarray,
    layers: list[tuple[str, np.ndarray, str]],
    n_sensors: int,
    output_path: Path,
    region: str,
    view: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.5, 8.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    for name, points, color in layers:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=0.45,
            color=color,
            alpha=0.11 if name != "Scalp" else 0.08,
            linewidths=0,
            label=name,
        )
    ax.scatter(
        sources[:, 0],
        sources[:, 1],
        sources[:, 2],
        s=2.0,
        color="#2563eb",
        alpha=0.12,
        linewidths=0,
        label="Brain source grid",
    )
    ax.scatter(
        sensors[:, 0],
        sensors[:, 1],
        sensors[:, 2],
        s=18,
        color="#dc2626",
        alpha=0.96,
        edgecolor="white",
        linewidth=0.25,
        label="Sensors",
    )
    _plot_legacy_hemisphere(ax)

    all_points = np.vstack([scalp, sensors, sources, *(points for _, points, _ in layers)])
    _set_equal_axes(ax, all_points)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.set_title(f"MIDA layered head + scalp sensors ({n_sensors} sensors, {region} region)")
    ax.legend(loc="upper left", markerscale=2.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_geometry(
    *,
    n_sensors: int,
    output_path: Path,
    region: str,
    method: str,
    sensor_offset_mm: float,
    max_scalp_points: int,
    max_layer_points: int,
    max_source_points: int,
    view: tuple[float, float],
) -> None:
    sensors = get_sensor_positions(
        n_sensors,
        offset=sensor_offset_mm,
        scalp_region=region,
        scalp_sampling=method,
    )
    scalp = mida_scalp_triangles(region=region, coordinate_frame="guti").reshape(-1, 3)
    scalp = _sample_points(scalp, max_scalp_points)
    layers = []
    for layer in mida_eeg_bem_layers():
        layer_points = load_mida_surface_triangles(
            layer.surface_name,
            coordinate_frame="guti",
        ).reshape(-1, 3)
        layers.append(
            (
                layer.domain_name,
                _sample_points(layer_points, max_layer_points),
                LAYER_COLORS.get(layer.domain_name, "#64748b"),
            )
        )
    sources = get_grid_positions(grid_spacing_mm=10.0)
    sources = _sample_points(sources, max_source_points)

    try:
        _plot_with_matplotlib(
            sensors=sensors,
            scalp=scalp,
            sources=sources,
            layers=layers,
            n_sensors=n_sensors,
            output_path=output_path,
            region=f"{region}, {method}",
            view=view,
        )
    except ModuleNotFoundError:
        _plot_with_pillow(
            sensors=sensors,
            scalp=scalp,
            sources=sources,
            layers=layers,
            n_sensors=n_sensors,
            output_path=output_path,
            region=f"{region}, {method}",
            view=view,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sensors", type=int, default=256)
    parser.add_argument(
        "--region",
        choices=["superior", "cranial", "full"],
        default="superior",
    )
    parser.add_argument(
        "--method",
        choices=["area", "projected_fibonacci", "max_distance"],
        default="projected_fibonacci",
    )
    parser.add_argument("--sensor-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-scalp-points", type=int, default=45000)
    parser.add_argument("--max-layer-points", type=int, default=9000)
    parser.add_argument("--max-source-points", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/mida_scalp_sensors_3d.png"),
    )
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-54.0)
    args = parser.parse_args()

    plot_geometry(
        n_sensors=args.n_sensors,
        output_path=args.output,
        region=args.region,
        method=args.method,
        sensor_offset_mm=args.sensor_offset_mm,
        max_scalp_points=args.max_scalp_points,
        max_layer_points=args.max_layer_points,
        max_source_points=args.max_source_points,
        view=(args.elev, args.azim),
    )


if __name__ == "__main__":
    main()
