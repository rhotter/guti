"""Plot ultrasound-only MIDA acoustic-window receiver apertures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from guti.core import get_grid_positions
from guti.mida_geometry import (
    mida_scalp_triangles,
    mida_us_acoustic_windows,
    sample_mida_us_acoustic_window_positions_by_window,
)


WINDOW_COLORS = {
    "left_temporal": "#dc2626",
    "right_temporal": "#f97316",
    "left_occipital": "#2563eb",
    "right_occipital": "#0891b2",
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


def plot_us_windows(
    *,
    n_sensors: int,
    output_path: Path,
    max_scalp_points: int,
    max_source_points: int,
    elev: float,
    azim: float,
) -> None:
    import matplotlib.pyplot as plt

    scalp = mida_scalp_triangles(region="full", coordinate_frame="guti").reshape(-1, 3)
    scalp = _sample_points(scalp, max_scalp_points)
    sources = _sample_points(get_grid_positions(grid_spacing_mm=10.0), max_source_points)
    sensors_by_window = sample_mida_us_acoustic_window_positions_by_window(n_sensors)
    windows = mida_us_acoustic_windows()

    fig = plt.figure(figsize=(9.5, 8.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        scalp[:, 0],
        scalp[:, 1],
        scalp[:, 2],
        s=0.35,
        color="#94a3b8",
        alpha=0.08,
        linewidths=0,
        label="MIDA scalp",
    )
    ax.scatter(
        sources[:, 0],
        sources[:, 1],
        sources[:, 2],
        s=1.3,
        color="#64748b",
        alpha=0.11,
        linewidths=0,
        label="Brain source grid",
    )

    all_points = [scalp, sources]
    for name, sensors in sensors_by_window.items():
        color = WINDOW_COLORS.get(name, "#111827")
        ax.scatter(
            sensors[:, 0],
            sensors[:, 1],
            sensors[:, 2],
            s=16,
            color=color,
            alpha=0.98,
            edgecolor="white",
            linewidth=0.25,
            label=name.replace("_", " "),
        )
        all_points.append(sensors)

    centers = np.array([window.center_mm for window in windows], dtype=np.float64)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        s=48,
        marker="x",
        color="#111827",
        linewidths=1.5,
        label="aperture centers",
    )
    all_points.append(centers)

    _set_equal_axes(ax, np.vstack(all_points))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_zlabel("z mm")
    ax.set_title(f"MIDA ultrasound temporal/occipital windows ({n_sensors} receivers)")
    ax.legend(loc="upper left", markerscale=2.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sensors", type=int, default=1000)
    parser.add_argument("--max-scalp-points", type=int, default=45000)
    parser.add_argument("--max-source-points", type=int, default=5000)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-52.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/mida_us_windows_3d.png"),
    )
    args = parser.parse_args()

    plot_us_windows(
        n_sensors=args.n_sensors,
        output_path=args.output,
        max_scalp_points=args.max_scalp_points,
        max_source_points=args.max_source_points,
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    main()
