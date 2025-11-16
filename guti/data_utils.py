import numpy as np
import os
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray
from guti.parameters import Parameters

# Get the absolute path to the results directory at the root level
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
VARIANTS_DIR = os.path.join(RESULTS_DIR, "variants")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VARIANTS_DIR, exist_ok=True)


def save_svd(
    s: NDArray,
    modality_name: str,
    params: Optional[Parameters] = None,
    default_run: bool = False,
) -> None:
    """
    Save the singular value spectrum and optional parameters to a file.

    Parameters
    ----------
    s : ndarray
        Singular values from SVD
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    params : Parameters, optional
        Parameters object with the following structure:
        Parameters(
            num_sensors: int,
            grid_resolution: float,
            num_brain_grid_points: int,
            time_resolution: float,
            comment: str,
            noise_full_brain: float
        )
    """
    if default_run:
        # Save as default configuration in main results directory
        filepath = os.path.join(RESULTS_DIR, f"{modality_name}_svd_spectrum.npz")
        np.savez(filepath, singular_values=s)
        print(f"Saved default SVD spectrum to {filepath}")
    else:
        if params is None:
            raise ValueError("Params must be provided for non-default runs")
        # Save as variant in variants directory with hash
        params_hash = params.get_hash()
        # Directory: variants/[modality_name]/
        target_dir = os.path.join(VARIANTS_DIR, modality_name)
        os.makedirs(target_dir, exist_ok=True)

        filepath = os.path.join(target_dir, f"{params_hash}.npz")
        structured_params = asdict(params)
        np.savez(filepath, singular_values=s, parameters=structured_params)  # type: ignore
    return filepath


def load_svd(modality_name: str) -> Tuple[NDArray, Optional[Parameters]]:
    """
    Load the singular value spectrum from a file.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'

    Returns
    -------
    tuple
        (singular_values, parameters) where parameters is a Parameters object
        or None if no parameters were saved
    """
    filepath = os.path.join(RESULTS_DIR, f"{modality_name}_svd_spectrum.npz")
    data = np.load(filepath, allow_pickle=True)
    try:
        params_dict = data["parameters"].item()  # Use .item() to get the dictionary
        print(data["singular_values"].shape)
        return data["singular_values"], Parameters.from_dict(params_dict)
    except KeyError:
        return data["singular_values"], None


def load_all_svds() -> Dict[str, Tuple[NDArray, Optional[Parameters]]]:
    """
    Load all SVD spectrums from the results folder.

    Returns
    -------
    dict
        Dictionary mapping modality names to tuples of (singular_values, Parameters)
        where Parameters is a Parameters object or None if no parameters were saved
    """
    results = {}
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("_svd_spectrum.npz"):
            modality_name = filename.replace("_svd_spectrum.npz", "")
            # Use label if exists, otherwise use modality_name
            results[modality_name] = load_svd(modality_name)
    return results


def load_svd_variant(
    modality_name: str, params_hash: str, subdir: Optional[str] = None
) -> Tuple[NDArray, Optional[Parameters]]:
    """
    Load a specific parameter variant of the SVD spectrum.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    params_hash : str
        Hash of the parameter configuration
    subdir : str, optional
        Subdirectory within variants/[modality_name]/ where the file is located

    Returns
    -------
    tuple
        (singular_values, parameters) where parameters is a Parameters object
    """
    if subdir is not None:
        filepath = os.path.join(
            VARIANTS_DIR, modality_name, subdir, f"{params_hash}.npz"
        )
    else:
        filepath = os.path.join(VARIANTS_DIR, modality_name, f"{params_hash}.npz")
    data = np.load(filepath, allow_pickle=True)
    params_dict = data["parameters"].item()
    return data["singular_values"], Parameters.from_dict(params_dict)


def list_svd_variants(
    modality_name: str,
    constant_params: Optional[Parameters] = None,
    sort_by: Optional[str] = None,
):
    """
    List all available parameter variants for a modality.

    Parameters
    ----------
    modality_name : str
        Name of modality, e.g. 'fnirs_cw' or 'eeg'
    constant_params : Parameters, optional
        Parameters object specifying which parameters to hold constant.
        Only variants matching the non-None fields will be returned.
        Example: Parameters(num_sensors=8000) will only return variants
        with num_sensors=8000, ignoring other parameters.
    sort_by : str, optional
        Parameter key to sort variants by. If specified, only variants that have
        this parameter (non-None) will be returned, sorted by this parameter value.
        Returns a list of tuples instead of a dict.
        Example: "grid_resolution_mm" will return variants sorted by grid_resolution_mm.

    Returns
    -------
    dict or list
        If sort_by is None: Dictionary mapping parameter hashes to variant dicts
        If sort_by is specified: List of tuples (hash, variant_dict) sorted by the parameter
    """
    variants = {}

    search_dir = os.path.join(VARIANTS_DIR, modality_name)

    if not os.path.exists(search_dir):
        return [] if sort_by is not None else variants

    for filename in os.listdir(search_dir):
        if filename.endswith(".npz"):
            # Extract hash from filename (hash is the filename without .npz)
            hash_part = filename[:-4]  # Remove .npz suffix
            if len(hash_part) == 8:  # Our hashes are 8 characters
                try:
                    s, params = load_svd_variant(modality_name, hash_part)
                    print(f"Loaded variant {hash_part} with params {params}")
                    if params is not None:
                        variants[hash_part] = dict(s=s, params=params)
                except FileNotFoundError:
                    continue
                except EOFError:
                    continue

    # Filter by constant parameters and sort_by parameter if specified
    if constant_params is not None or sort_by is not None:
        constant_dict = constant_params.to_dict() if constant_params is not None else {}

        filtered_variants = {}
        for k, v in variants.items():
            params = v["params"]
            # Check if all specified constant params match
            constant_match = all(
                getattr(params, key, None) == value
                for key, value in constant_dict.items()
            )
            # Check if sort_by param exists (are not None)
            sort_match = (
                getattr(params, sort_by, None) is not None
                if sort_by is not None
                else True
            )

            if constant_match and sort_match:
                filtered_variants[k] = v
        variants = filtered_variants

    # Sort if requested
    if sort_by is not None:
        return sorted(variants.items(), key=lambda x: getattr(x[1]["params"], sort_by))

    return variants


def add_param_to_svd_variants(
    modality_name: str,
    param_key: str,
    param_value: Any,
    target_subdir: str,
    source_subdir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load all SVD variants for a modality, add/update a parameter, and save to a subfolder.

    Parameters
    ----------
    modality_name : str
        Modality name, e.g., 'fnirs_cw' or 'eeg'
    param_key : str
        Parameter key to add/update in the saved parameters dict
    param_value : Any
        Value to assign to the specified parameter key
    target_subdir : str
        Subdirectory under variants/[modality_name]/ to write updated files into
    source_subdir : str, optional
        If provided, read variants from variants/[modality_name]/[source_subdir] instead of the root

    Returns
    -------
    dict
        Mapping from source file path to saved destination file path
    """
    # Resolve input and output directories
    source_dir = (
        os.path.join(VARIANTS_DIR, modality_name, source_subdir)
        if source_subdir is not None
        else os.path.join(VARIANTS_DIR, modality_name)
    )
    dest_dir = os.path.join(VARIANTS_DIR, target_subdir)

    if not os.path.exists(source_dir):
        print(f"No variants directory found at {source_dir}")
        return {}

    os.makedirs(dest_dir, exist_ok=True)

    saved_mapping: Dict[str, str] = {}

    for filename in os.listdir(source_dir):
        if not filename.endswith(".npz"):
            continue

        src_path = os.path.join(source_dir, filename)
        try:
            data = np.load(src_path, allow_pickle=True)
        except EOFError:
            # Corrupted file; skip
            continue

        try:
            s = data["singular_values"]
        except KeyError:
            # Not an SVD variant file; skip
            # Ensure file is closed before continue
            try:
                data.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            continue

        # Extract existing parameters dict if present
        params_dict: Dict[str, Any]
        if "parameters" in data:
            try:
                params_obj = data["parameters"].item()
                params_dict = dict(params_obj) if isinstance(params_obj, dict) else {}
            except Exception:
                params_dict = {}
        else:
            params_dict = {}

        # Add or update the specified parameter
        params_dict[param_key] = param_value

        print(params_dict)
        print(f"Added {param_key} = {param_value} to {filename}")

        # Save updated variant into destination subdirectory, preserving filename
        dest_path = os.path.join(dest_dir, filename)
        np.savez(dest_path, singular_values=s, parameters=params_dict)  # type: ignore
        saved_mapping[src_path] = dest_path

        # Attempt to close npz handle
        try:
            data.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    if not saved_mapping:
        print(f"No variant files were processed in {source_dir}")

    return saved_mapping
