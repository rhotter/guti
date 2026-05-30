import os
import numpy as np

from guti.core import get_sensor_positions, get_grid_positions, BRAIN_RADIUS
from guti.data_utils import list_svd_variants, save_svd

SPHERE_CENTER = np.array([BRAIN_RADIUS, BRAIN_RADIUS, 0.0])


def sarvas_formula(r, r0, center=SPHERE_CENTER):
    """
    Correct Sarvas lead-field matrix for a spherical conductor.
    Returns M such that B = M @ q.
    """
    mu0 = 4 * np.pi * 1e-7
    r = (np.asarray(r) - center) * 1e-3
    r0 = (np.asarray(r0) - center) * 1e-3

    a_vec = r - r0
    a = np.linalg.norm(a_vec)
    r_norm = np.linalg.norm(r)

    if a < 1e-12 or r_norm < 1e-12:
        return np.zeros((3, 3))

    F = a * (a * r_norm + r_norm**2 - np.dot(r0, r))
    if abs(F) < 1e-20:
        return np.zeros((3, 3))

    a_dot_r = np.dot(a_vec, r)
    nabla_F = (
        (a**2 / r_norm + a_dot_r / a + 2 * a + 2 * r_norm) * r
        - (a + 2 * r_norm + a_dot_r / a) * r0
    )

    r0_cross = np.array(
        [
            [0.0, -r0[2], r0[1]],
            [r0[2], 0.0, -r0[0]],
            [-r0[1], r0[0], 0.0],
        ]
    )
    r0xr = r0_cross @ r
    M = (mu0 / (4 * np.pi)) * (-F * r0_cross - np.outer(nabla_F, r0xr)) / (F**2)
    return M


def compute_forward_matrix(n_sensors, grid_spacing_mm, offset_mm):
    sensors = get_sensor_positions(n_sensors, offset=offset_mm)
    sources = get_grid_positions(grid_spacing_mm=grid_spacing_mm)
    n_sources = len(sources)
    A = np.zeros((3 * n_sensors, 3 * n_sources))
    print(
        f"Computing A: sensors={n_sensors}, sources={n_sources}, "
        f"spacing={grid_spacing_mm}mm, offset={offset_mm}mm"
    )
    for i, sensor in enumerate(sensors):
        for j, source in enumerate(sources):
            M = sarvas_formula(sensor, source)
            A[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = M
    return A


def compute_svd(n_sensors, grid_spacing_mm, offset_mm):
    A = compute_forward_matrix(n_sensors, grid_spacing_mm, offset_mm)
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    return s


def recompute_modality(modality_name, variants_root=None):
    variants = list_svd_variants(modality_name) if variants_root is None else {}
    if variants_root is not None:
        # Load variants from an explicit directory (e.g., backup)
        from pathlib import Path
        from guti.parameters import Parameters

        root = Path(variants_root)
        search_dir = root / modality_name
        if not search_dir.exists():
            # Fallback: pick newest directory matching "{modality}_*"
            candidates = sorted(
                root.glob(f"{modality_name}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                search_dir = candidates[0]
        for filename in search_dir.glob("*.npz"):
            try:
                data = np.load(filename, allow_pickle=True)
                params_dict = data["parameters"].item()
                params = Parameters.from_dict(params_dict)
                variants[filename.stem] = dict(s=data["singular_values"], params=params)
            except Exception:
                continue
    for params_hash, entry in variants.items():
        params = entry["params"]
        if params.source_spacing_mm == 3.0:
            print(f"Skipping {modality_name} {params_hash} (3.0 mm spacing)")
            continue
        # Skip if already computed in current variants dir
        from pathlib import Path
        target_dir = Path("results/variants") / modality_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{params.get_hash()}.npz"
        if target_path.exists():
            print(f"Skipping {modality_name} {params.get_hash()} (already computed)")
            continue
        s = compute_svd(
            n_sensors=params.num_sensors,
            grid_spacing_mm=params.source_spacing_mm,
            offset_mm=params.sensor_offset_mm,
        )
        save_svd(s, modality_name, params)
        print(f"Saved {modality_name} {params_hash} with {params}")


if __name__ == "__main__":
    variants_root = os.environ.get("MEG_VARIANTS_SRC")
    for modality in ["meg_opm", "meg_squid"]:
        recompute_modality(modality, variants_root=variants_root)
