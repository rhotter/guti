#!/usr/bin/env python3
"""Plot one MEG Johnson-noise correlation row on an interpolated 3D surface."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-config")

from guti.core import SCALP_RADIUS, get_sensor_positions
from guti.modalities.meg.johnson_noise import HEAD_CENTER_MM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_npz",
        type=Path,
        help="NPZ written by scripts/run_meg_johnson_covariance.py.",
    )
    parser.add_argument(
        "--selected-index",
        type=int,
        default=None,
        help=(
            "Sensor/channel index whose correlation row will be plotted. "
            "Default: choose a high, off-center sensor."
        ),
    )
    parser.add_argument(
        "--surface-samples",
        type=int,
        default=20_000,
        help="Dense hemisphere samples for interpolation. Default: 20000.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1e-4,
        help="RBF smoothing used for interpolation. Default: 1e-4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: input NPZ directory.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix. Default: input NPZ stem plus selected index.",
    )
    return parser


def unit_vectors(positions_mm: np.ndarray) -> np.ndarray:
    radial = np.asarray(positions_mm, dtype=float) - HEAD_CENTER_MM[None, :]
    return radial / np.linalg.norm(radial, axis=1, keepdims=True)


def default_selected_index(sensor_positions_mm: np.ndarray) -> int:
    """Pick a sensor that is visible and not at the pole/equator."""

    target = np.array([0.45, -0.35, 0.82], dtype=float)
    target /= np.linalg.norm(target)
    return int(np.argmax(unit_vectors(sensor_positions_mm) @ target))


def infer_sensor_offset_mm(sensor_positions_mm: np.ndarray, metadata: dict[str, object]) -> float:
    offset = metadata.get("sensor_offset_mm")
    if offset is not None:
        return float(offset)
    radii = np.linalg.norm(sensor_positions_mm - HEAD_CENTER_MM[None, :], axis=1)
    return float(np.median(radii) - SCALP_RADIUS)


def interpolate_to_surface(
    sensor_positions_mm: np.ndarray,
    sensor_values: np.ndarray,
    surface_positions_mm: np.ndarray,
    *,
    smoothing: float,
) -> np.ndarray:
    from scipy.interpolate import RBFInterpolator

    rbf = RBFInterpolator(
        unit_vectors(sensor_positions_mm),
        np.asarray(sensor_values, dtype=float),
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
    )
    return np.clip(rbf(unit_vectors(surface_positions_mm)), -1.0, 1.0)


def color_limits(sensor_values: np.ndarray, surface_values: np.ndarray) -> tuple[float, float]:
    low = float(np.min([np.min(sensor_values), np.min(surface_values)]))
    high = float(np.max([np.max(sensor_values), np.max(surface_values)]))
    vmin = max(-1.0, min(-0.05, low))
    vmax = min(1.0, max(0.05, high))
    return vmin, vmax


def plot_surface_3d(
    positions_mm: np.ndarray,
    values: np.ndarray,
    *,
    selected_position_mm: np.ndarray,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rel = positions_mm - HEAD_CENTER_MM[None, :]
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    tri = mtri.Triangulation(x, y)
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    colors = plt.get_cmap("coolwarm")(norm(values))
    vertices = np.stack(
        [x[tri.triangles], y[tri.triangles], z[tri.triangles]],
        axis=-1,
    )
    facecolors = colors[tri.triangles].mean(axis=1)

    fig = plt.figure(figsize=(9.0, 7.4), dpi=190)
    ax = fig.add_subplot(111, projection="3d")
    surface = Poly3DCollection(
        vertices,
        facecolors=facecolors,
        linewidths=0.0,
        antialiased=True,
    )
    ax.add_collection3d(surface)

    selected = selected_position_mm - HEAD_CENTER_MM
    selected_marker = selected * 1.12
    ax.plot(
        [selected[0], selected_marker[0]],
        [selected[1], selected_marker[1]],
        [selected[2], selected_marker[2]],
        color="black",
        linewidth=2.2,
        zorder=30,
    )
    ax.scatter(
        [selected_marker[0]],
        [selected_marker[1]],
        [selected_marker[2]],
        c="black",
        marker="*",
        s=160,
        depthshade=False,
        label="selected sensor",
    )
    radius = float(np.max(np.linalg.norm(rel, axis=1)))
    pad = 0.08 * radius
    ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))
    ax.set_ylim(float(np.min(y) - pad), float(np.max(y) + pad))
    ax.set_zlim(float(np.min(z)), float(np.max(z) + pad))
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=24, azim=-55)
    ax.set_axis_off()
    ax.set_title(title)

    mappable = ScalarMappable(norm=norm, cmap="coolwarm")
    mappable.set_array(values)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("correlation with selected sensor")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_topdown(
    positions_mm: np.ndarray,
    values: np.ndarray,
    *,
    selected_position_mm: np.ndarray,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize, TwoSlopeNorm

    rel = positions_mm - HEAD_CENTER_MM[None, :]
    x, y = rel[:, 0], rel[:, 1]
    tri = mtri.Triangulation(x, y)
    if vmin < 0.0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=190)
    collection = ax.tricontourf(tri, values, levels=80, cmap="coolwarm", norm=norm)
    selected = selected_position_mm - HEAD_CENTER_MM
    ax.scatter([selected[0]], [selected[1]], c="black", marker="*", s=110)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    cbar = fig.colorbar(collection, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("correlation with selected sensor")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    input_npz: Path,
    selected_index: int,
    sensor_positions_mm: np.ndarray,
    selected_correlations: np.ndarray,
    surface_correlations: np.ndarray,
    surface_samples: int,
    smoothing: float,
    artifacts: list[Path],
    metadata: dict[str, object],
) -> None:
    selected = sensor_positions_mm[selected_index]
    rel = selected - HEAD_CENTER_MM
    radius = float(np.linalg.norm(rel))
    polar_deg = math.degrees(math.acos(float(rel[2] / radius)))
    azimuth_deg = math.degrees(math.atan2(float(rel[1]), float(rel[0])))
    offdiag = np.delete(selected_correlations, selected_index)
    lines = [
        "# MEG Johnson Selected-Sensor Correlation Surface",
        "",
        f"- Input covariance: `{input_npz}`",
        f"- Selected sensor index: {selected_index}",
        f"- Sensor components: `{metadata.get('sensor_components', 'unknown')}`",
        f"- Sensors in covariance: {sensor_positions_mm.shape[0]}",
        f"- Interpolated surface samples: {surface_samples}",
        f"- RBF smoothing: {smoothing:g}",
        f"- Selected polar angle from top: {polar_deg:.2f} deg",
        f"- Selected azimuth: {azimuth_deg:.2f} deg",
        "",
        "## Correlation Row Stats",
        "",
        f"- min sensor correlation: {float(np.min(selected_correlations)):.5g}",
        f"- max sensor correlation: {float(np.max(selected_correlations)):.5g}",
        f"- mean off-diagonal correlation: {float(np.mean(offdiag)):.5g}",
        f"- mean absolute off-diagonal correlation: {float(np.mean(np.abs(offdiag))):.5g}",
        f"- min interpolated surface correlation: {float(np.min(surface_correlations)):.5g}",
        f"- max interpolated surface correlation: {float(np.max(surface_correlations)):.5g}",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- [{artifact.name}]({artifact.name})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.surface_samples <= 0:
        raise SystemExit("--surface-samples must be positive")
    if args.smoothing < 0.0 or not math.isfinite(args.smoothing):
        raise SystemExit("--smoothing must be finite and nonnegative")

    data = np.load(args.input_npz, allow_pickle=True)
    sensor_positions = np.asarray(data["sensor_positions_mm"], dtype=float)
    correlation = np.asarray(data["johnson_correlation"], dtype=float)
    metadata_json = str(data["metadata_json"]) if "metadata_json" in data else "{}"
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        metadata = {}

    if sensor_positions.ndim != 2 or sensor_positions.shape[1] != 3:
        raise SystemExit("sensor_positions_mm must have shape (n_sensors, 3)")
    if correlation.shape != (sensor_positions.shape[0], sensor_positions.shape[0]):
        raise SystemExit(
            "This surface plot expects one scalar channel per sensor. "
            "Run the covariance script with --sensor-components radial."
        )

    selected_index = (
        default_selected_index(sensor_positions)
        if args.selected_index is None
        else int(args.selected_index)
    )
    if not (0 <= selected_index < sensor_positions.shape[0]):
        raise SystemExit("--selected-index must be in [0, n_sensors)")

    sensor_offset_mm = infer_sensor_offset_mm(sensor_positions, metadata)
    surface_positions = get_sensor_positions(
        args.surface_samples,
        offset=sensor_offset_mm,
    )
    selected_correlations = np.asarray(correlation[selected_index], dtype=float)
    surface_correlations = interpolate_to_surface(
        sensor_positions,
        selected_correlations,
        surface_positions,
        smoothing=args.smoothing,
    )
    vmin, vmax = color_limits(selected_correlations, surface_correlations)

    output_dir = args.output_dir if args.output_dir is not None else args.input_npz.parent
    prefix = args.prefix or f"{args.input_npz.stem}_selected{selected_index}"
    surface_3d_path = output_dir / f"{prefix}_correlation_surface_3d.png"
    topdown_path = output_dir / f"{prefix}_correlation_topdown.png"
    arrays_path = output_dir / f"{prefix}_surface_arrays.npz"
    summary_path = output_dir / f"{prefix}_surface_summary.md"

    title = f"MEG Johnson correlation to sensor {selected_index}"
    plot_surface_3d(
        surface_positions,
        surface_correlations,
        selected_position_mm=sensor_positions[selected_index],
        output_path=surface_3d_path,
        title=title,
        vmin=vmin,
        vmax=vmax,
    )
    plot_topdown(
        surface_positions,
        surface_correlations,
        selected_position_mm=sensor_positions[selected_index],
        output_path=topdown_path,
        title=title,
        vmin=vmin,
        vmax=vmax,
    )
    np.savez_compressed(
        arrays_path,
        sensor_positions_mm=sensor_positions,
        surface_positions_mm=surface_positions,
        selected_sensor_correlation=selected_correlations,
        selected_surface_correlation=surface_correlations,
        selected_index=np.array(selected_index),
        input_npz=np.array(str(args.input_npz)),
        metadata_json=np.array(json.dumps(metadata, indent=2, sort_keys=True)),
    )
    artifacts = [surface_3d_path, topdown_path, arrays_path]
    write_summary(
        summary_path,
        input_npz=args.input_npz,
        selected_index=selected_index,
        sensor_positions_mm=sensor_positions,
        selected_correlations=selected_correlations,
        surface_correlations=surface_correlations,
        surface_samples=args.surface_samples,
        smoothing=args.smoothing,
        artifacts=artifacts,
        metadata=metadata,
    )
    artifacts.append(summary_path)

    print(f"selected sensor index: {selected_index}")
    print(
        "selected row corr "
        f"min={float(np.min(selected_correlations)):.4g}, "
        f"mean offdiag={float(np.mean(np.delete(selected_correlations, selected_index))):.4g}, "
        f"mean abs offdiag={float(np.mean(np.abs(np.delete(selected_correlations, selected_index)))):.4g}"
    )
    for artifact in artifacts:
        print(f"wrote {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
