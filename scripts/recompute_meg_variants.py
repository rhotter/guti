import os

import numpy as np

from guti.data_utils import list_svd_variants, save_svd
# Canonical Sarvas forward model now lives in the meg package; re-exported here
# so existing imports (e.g. compute_information_maps) keep working.
from guti.modalities.meg.meg import (
    sarvas_formula,
    compute_forward_matrix,
    compute_svd,
)


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
