"""Scalp electrode resistance for a layered spherical EEG head model.

This estimates the tissue volume-conduction contribution between two scalp
electrodes. It does not include electrode, gel, or skin contact impedance.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from guti.core import (
    BRAIN_CONDUCTIVITY,
    BRAIN_RADIUS,
    CSF_RADIUS,
    SCALP_CONDUCTIVITY,
    SCALP_RADIUS,
    SKULL_CONDUCTIVITY,
    SKULL_RADIUS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

# GUTI stores EEG conductivities as ratios. Anchor those ratios to a typical
# brain/scalp absolute conductivity so the result is reported in ohms.
BRAIN_SCALP_CONDUCTIVITY_S_PER_M = 0.33
CSF_TO_BRAIN_CONDUCTIVITY_RATIO = 5.0

DEFAULT_SPACING_CM = 8.0
DEFAULT_AREA_MIN_CM2 = 0.25
DEFAULT_AREA_MAX_CM2 = 25.0
DEFAULT_AREA_POINTS = 50
DEFAULT_LMAX = 5000


@dataclass(frozen=True)
class Layer:
    name: str
    outer_radius_m: float
    conductivity_s_per_m: float

    @property
    def outer_radius_mm(self) -> float:
        return self.outer_radius_m * 1000.0


def default_layers() -> tuple[Layer, ...]:
    """Return GUTI's four EEG layers from inner to outer."""
    scale = BRAIN_SCALP_CONDUCTIVITY_S_PER_M
    return (
        Layer("brain", BRAIN_RADIUS / 1000.0, BRAIN_CONDUCTIVITY * scale),
        Layer("csf", CSF_RADIUS / 1000.0, CSF_TO_BRAIN_CONDUCTIVITY_RATIO * scale),
        Layer("skull", SKULL_RADIUS / 1000.0, SKULL_CONDUCTIVITY * scale),
        Layer("scalp", SCALP_RADIUS / 1000.0, SCALP_CONDUCTIVITY * scale),
    )


def legendre_values(x: float, n_max: int) -> np.ndarray:
    """Return P_0(x)..P_n_max(x) using the Legendre recurrence."""
    p = np.empty(n_max + 1, dtype=np.float64)
    p[0] = 1.0
    if n_max >= 1:
        p[1] = x
    for n in range(1, n_max):
        p[n + 1] = ((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1)
    return p


def surface_impedance_by_degree(layers: tuple[Layer, ...], lmax: int) -> np.ndarray:
    """Return V(R) / q(R) for each spherical-harmonic degree.

    q is the outward radial current-density harmonic at the scalp surface. The
    zero-degree term is undefined because the injected net current is zero.
    """
    transfer = np.full(lmax + 1, np.nan, dtype=np.float64)
    inner = layers[0]

    for ell in range(1, lmax + 1):
        # Regular solid-sphere core: V = A r^ell, q / V = sigma*ell/r.
        admittance = inner.conductivity_s_per_m * ell / inner.outer_radius_m
        previous_radius = inner.outer_radius_m

        for layer in layers[1:]:
            radius = layer.outer_radius_m
            sigma = layer.conductivity_s_per_m
            lam = previous_radius / radius

            # Shell basis at x = r/radius:
            # V = C*x^ell + D*x^-(ell+1)
            # q = (sigma/radius) * dV/dx
            a = lam**ell
            b = lam ** (-(ell + 1))
            c = (sigma / radius) * ell * lam ** (ell - 1)
            d = -(sigma / radius) * (ell + 1) * lam ** (-(ell + 2))

            ratio = (c - admittance * a) / (admittance * b - d)
            admittance = (sigma / radius) * (
                ell - (ell + 1) * ratio
            ) / (1.0 + ratio)
            previous_radius = radius

        transfer[ell] = 1.0 / admittance

    return transfer


def normalized_cap_axisymmetric_coeffs(
    cap_half_angle_rad: float, lmax: int
) -> np.ndarray:
    """Return m=0 harmonic coefficients for a unit-integral circular cap."""
    cos_alpha = math.cos(cap_half_angle_rad)
    p = legendre_values(cos_alpha, lmax + 1)
    coeffs = np.zeros(lmax + 1, dtype=np.float64)
    denom = 1.0 - cos_alpha

    ell = np.arange(1, lmax + 1, dtype=np.float64)
    integral = (p[:-2] - p[2:]) / (2.0 * ell + 1.0)
    coeffs[1:] = (
        np.sqrt((2.0 * ell + 1.0) / (4.0 * math.pi)) * integral / denom
    )
    return coeffs


def resistance_for_area(
    area_cm2: float,
    transfer: np.ndarray,
    spacing_cm: float = DEFAULT_SPACING_CM,
    scalp_radius_m: float = SCALP_RADIUS / 1000.0,
) -> tuple[float, float]:
    """Return resistance in ohms and circular-cap half-angle in radians."""
    area_m2 = area_cm2 * 1e-4
    cap_half_angle = math.acos(1.0 - area_m2 / (2.0 * math.pi * scalp_radius_m**2))
    center_angle = (spacing_cm / 100.0) / scalp_radius_m

    cap_coeffs = normalized_cap_axisymmetric_coeffs(cap_half_angle, len(transfer) - 1)
    p_sep = legendre_values(math.cos(center_angle), len(transfer) - 1)

    ell_slice = slice(1, None)
    diff_power = 2.0 * cap_coeffs[ell_slice] ** 2 * (1.0 - p_sep[ell_slice])
    resistance = np.sum(transfer[ell_slice] * diff_power) / scalp_radius_m**2
    return float(resistance), cap_half_angle


def sweep_resistance(
    area_min_cm2: float = DEFAULT_AREA_MIN_CM2,
    area_max_cm2: float = DEFAULT_AREA_MAX_CM2,
    area_points: int = DEFAULT_AREA_POINTS,
    spacing_cm: float = DEFAULT_SPACING_CM,
    lmax: int = DEFAULT_LMAX,
    layers: tuple[Layer, ...] | None = None,
) -> list[dict[str, float]]:
    """Sweep circular electrode area and return resistance rows."""
    if layers is None:
        layers = default_layers()

    transfer = surface_impedance_by_degree(layers, lmax)
    areas = np.geomspace(area_min_cm2, area_max_cm2, area_points)

    rows: list[dict[str, float]] = []
    for area_cm2 in areas:
        resistance, cap_half_angle = resistance_for_area(
            float(area_cm2),
            transfer,
            spacing_cm=spacing_cm,
            scalp_radius_m=layers[-1].outer_radius_m,
        )
        rows.append(
            {
                "area_cm2": float(area_cm2),
                "equivalent_flat_radius_mm": math.sqrt(area_cm2 * 1e-4 / math.pi)
                * 1000.0,
                "cap_half_angle_deg": math.degrees(cap_half_angle),
                "resistance_ohm": resistance,
            }
        )
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float]], path: Path, layers: tuple[Layer, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = np.array([row["area_cm2"] for row in rows])
    resistance = np.array([row["resistance_ohm"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=180)
    ax.plot(areas, resistance, color="#16718f", linewidth=2.6)
    ax.scatter(areas[::7], resistance[::7], color="#16718f", s=18, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(areas.min() * 0.92, areas.max() * 1.08)
    ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 25])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "25"])
    ax.set_xlabel("Circular electrode area (cm^2, log scale)")
    ax.set_ylabel("Effective head volume resistance (ohm)")
    ax.set_title("Layered spherical EEG scalp resistance")
    ax.grid(True, which="both", color="#d6dadd", linewidth=0.8)

    layer_text = "\n".join(
        [
            "GUTI layer radii/conductivity",
            *[
                f"{layer.name}: {layer.outer_radius_mm:g} mm, "
                f"{layer.conductivity_s_per_m:.4g} S/m"
                for layer in layers
            ],
        ]
    )
    ax.text(
        0.98,
        0.96,
        layer_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#b7c0c5", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary(
    rows: list[dict[str, float]],
    path: Path,
    layers: tuple[Layer, ...],
    spacing_cm: float,
    lmax: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def closest(area: float) -> dict[str, float]:
        return min(rows, key=lambda row: abs(math.log(row["area_cm2"] / area)))

    selected = [closest(area) for area in [0.25, 0.5, 1, 2, 5, 10, 25]]
    log_area = np.log(np.array([row["area_cm2"] for row in rows]))
    log_resistance = np.log(np.array([row["resistance_ohm"] for row in rows]))
    slope, intercept = np.polyfit(log_area, log_resistance, 1)

    layer_lines = []
    previous_radius = 0.0
    for layer in layers:
        thickness = layer.outer_radius_m * 1000.0 - previous_radius
        layer_lines.append(
            f"- {layer.name}: outer radius {layer.outer_radius_mm:g} mm, "
            f"thickness {thickness:g} mm, "
            f"conductivity {layer.conductivity_s_per_m:.6g} S/m."
        )
        previous_radius = layer.outer_radius_m * 1000.0

    lines = [
        "EEG scalp-to-scalp head volume resistance simulation",
        "",
        "Model:",
        "- Four concentric spherical layers using GUTI's radius constants.",
        *layer_lines,
        f"- Two identical circular scalp electrodes, {spacing_cm:g} cm center-to-center arc spacing.",
        "- Uniform current density over each electrode.",
        "- Contact/electrode/gel/skin-interface impedance is not included.",
        f"- Spherical harmonic truncation: l <= {lmax}.",
        "",
        "Selected results:",
        "area_cm2,equivalent_flat_radius_mm,resistance_ohm",
    ]
    for row in selected:
        lines.append(
            f"{row['area_cm2']:.4g},"
            f"{row['equivalent_flat_radius_mm']:.3f},"
            f"{row['resistance_ohm']:.2f}"
        )
    lines.extend(
        [
            "",
            "Area scaling:",
            f"- Global log-log fit: R ~= {math.exp(intercept):.1f} * A^{slope:.3f} ohm, with A in cm^2.",
            "- The head-volume part falls more weakly than 1/A because current spreads in 3D tissue.",
            "",
            "Interpretation:",
            "These values are a lower bound on measured EEG impedance because contact",
            "impedance at the electrode/gel/skin interface is omitted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-cm", type=float, default=DEFAULT_SPACING_CM)
    parser.add_argument("--area-min-cm2", type=float, default=DEFAULT_AREA_MIN_CM2)
    parser.add_argument("--area-max-cm2", type=float, default=DEFAULT_AREA_MAX_CM2)
    parser.add_argument("--area-points", type=int, default=DEFAULT_AREA_POINTS)
    parser.add_argument("--lmax", type=int, default=DEFAULT_LMAX)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "eeg_scalp_resistance_vs_area.csv",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=REPO_ROOT / "plots" / "eeg_scalp_resistance_vs_area.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results" / "eeg_scalp_resistance_summary.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = default_layers()
    rows = sweep_resistance(
        area_min_cm2=args.area_min_cm2,
        area_max_cm2=args.area_max_cm2,
        area_points=args.area_points,
        spacing_cm=args.spacing_cm,
        lmax=args.lmax,
        layers=layers,
    )
    write_csv(rows, args.csv)
    plot_rows(rows, args.plot, layers)
    write_summary(rows, args.summary, layers, args.spacing_cm, args.lmax)

    for area in [0.25, 1.0, 5.0, 25.0]:
        row = min(rows, key=lambda r: abs(math.log(r["area_cm2"] / area)))
        print(f"{row['area_cm2']:6.3f} cm^2 -> {row['resistance_ohm']:8.2f} ohm")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.plot}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
