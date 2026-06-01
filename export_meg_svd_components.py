"""Export first MEG SVD modes for the web SVD-component explorer.

The artifact stores a small analytical Sarvas forward model with full
left/right singular vectors. It is intentionally lower resolution than the
main bitrate sweeps so the website can render source and detector patterns
directly in the browser.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from guti.core import BRAIN_RADIUS, SCALP_RADIUS, get_grid_positions, get_sensor_positions
from recompute_meg_variants import sarvas_formula


CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])
OUT_PATH = Path("web/public/data/meg_svd_components.json")


def compute_forward_matrix(
    n_sensors: int,
    spacing_mm: float,
    offset_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=spacing_mm)
    matrix = np.zeros((3 * len(sensors), 3 * len(sources)), dtype=np.float64)

    for sensor_idx, sensor in enumerate(sensors):
        row = slice(3 * sensor_idx, 3 * sensor_idx + 3)
        for source_idx, source in enumerate(sources):
            col = slice(3 * source_idx, 3 * source_idx + 3)
            matrix[row, col] = sarvas_formula(sensor, source)

    return matrix, sensors, sources


def unit_radial(points: np.ndarray) -> np.ndarray:
    rel = points - CENTER
    norm = np.linalg.norm(rel, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return rel / norm


def normalize(values: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(values))) if len(values) else 0.0
    return values / max_abs if max_abs > 0 else values


def node_payload(
    positions: np.ndarray,
    vectors: np.ndarray,
    vector_keys: tuple[str, str, str],
) -> list[dict[str, float]]:
    radial = unit_radial(positions)
    amp = normalize(np.linalg.norm(vectors, axis=1))
    signed = normalize(np.sum(vectors * radial, axis=1))

    nodes = []
    for pos, vec, node_amp, node_signed in zip(positions, vectors, amp, signed):
        rel = pos - CENTER
        nodes.append(
            {
                "x": round(float(rel[0]), 4),
                "y": round(float(rel[1]), 4),
                "z": round(float(rel[2]), 4),
                "amp": round(float(node_amp), 6),
                "signed": round(float(node_signed), 6),
                vector_keys[0]: round(float(vec[0]), 9),
                vector_keys[1]: round(float(vec[1]), 9),
                vector_keys[2]: round(float(vec[2]), 9),
            }
        )
    return nodes


def mode_payload(
    left_vectors: np.ndarray,
    singular_values: np.ndarray,
    right_vectors_t: np.ndarray,
    sources: np.ndarray,
    sensors: np.ndarray,
    mode_idx: int,
) -> dict:
    source_vectors = right_vectors_t[mode_idx].reshape(len(sources), 3)
    detector_vectors = left_vectors[:, mode_idx].reshape(len(sensors), 3)

    return {
        "component": mode_idx + 1,
        "singularValue": float(singular_values[mode_idx]),
        "relativeGain": float(singular_values[mode_idx] / singular_values[0]),
        "sources": node_payload(sources, source_vectors, ("vx", "vy", "vz")),
        "detectors": node_payload(sensors, detector_vectors, ("bx", "by", "bz")),
    }


def build_dataset(
    key: str,
    label: str,
    n_sensors: int,
    spacing_mm: float,
    offset_mm: float,
    n_modes: int,
) -> dict:
    print(
        f"Computing {key}: N={n_sensors}, spacing={spacing_mm} mm, "
        f"offset={offset_mm} mm"
    )
    matrix, sensors, sources = compute_forward_matrix(n_sensors, spacing_mm, offset_mm)
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    return {
        "key": key,
        "label": label,
        "nSensors": n_sensors,
        "nSources": len(sources),
        "sourceSpacingMm": spacing_mm,
        "sensorOffsetMm": offset_mm,
        "brainRadiusMm": BRAIN_RADIUS,
        "scalpRadiusMm": SCALP_RADIUS,
        "singularValues": [float(value) for value in singular_values[:n_modes]],
        "modes": [
            mode_payload(left_vectors, singular_values, right_vectors_t, sources, sensors, idx)
            for idx in range(n_modes)
        ],
    }


def main() -> None:
    n_modes = 8
    payload = {
        "description": (
            "First MEG SVD components computed from the analytical Sarvas "
            "forward matrix. Source vectors are right singular vectors; "
            "detector vectors are left singular vectors."
        ),
        "coordinateUnits": "mm relative to brain center",
        "datasets": [
            build_dataset("opm", "OPM-like MEG", 180, 12.0, 7.0, n_modes),
            build_dataset("squid", "SQUID-like MEG", 180, 12.0, 25.0, n_modes),
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
