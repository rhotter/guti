#!/usr/bin/env python
"""
Run any GUTI modality with custom parameters.

Usage:
    python run_modality.py <modality_name> [--param value ...]

Example:
    python run_modality.py 1d_blurring --num_brain_grid_points 256
    python run_modality.py eeg --num_sensors 64
    python run_modality.py us --num_voxels 128
"""

import argparse
import importlib
import sys
from pathlib import Path
from guti.parameters import Parameters


def get_available_modalities():
    """Get list of available modalities."""
    modalities = []
    modalities_dir = Path(__file__).parent / "guti" / "modalities"
    for item in sorted(modalities_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("_"):
            modality_file = item / "modality.py"
            if modality_file.exists():
                modalities.append(item.name)
    return modalities


def get_parameters_help():
    """Generate help text for available Parameters fields."""
    help_lines = ["\nAvailable modalities:"]
    for modality in get_available_modalities():
        help_lines.append(f"  {modality}")

    help_lines.append("\nAvailable parameters from Parameters class:")
    for field_name in Parameters.__dataclass_fields__:
        help_lines.append(f"  --{field_name}")

    return "\n".join(help_lines)


def main():
    # Build epilog with parameter help
    epilog = "Example: python run_modality.py 1d_blurring --num_brain_grid_points 256"
    epilog += get_parameters_help()

    parser = argparse.ArgumentParser(
        description="Run a GUTI imaging modality with custom parameters",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("modality_name", help="Modality name (e.g., 1d_blurring, eeg, us)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    parser.add_argument("--default-run", action="store_true", help="Run as default configuration")
    args, unknown = parser.parse_known_args()

    # Parse remaining args as --key value pairs
    params_dict = {}
    for i in range(0, len(unknown), 2):
        if unknown[i].startswith("--") and i + 1 < len(unknown):
            key = unknown[i][2:]
            value = unknown[i + 1]
            # Try to convert to number
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            params_dict[key] = value

    # Construct module path: guti.modalities.<modality_name>.modality
    modality_module = f"guti.modalities.{args.modality_name}.modality"

    try:
        # Import the modality module
        mod = importlib.import_module(modality_module)
    except ModuleNotFoundError:
        print(f"Error: Could not find modality '{args.modality_name}'")
        print(f"Tried to import: {modality_module}")
        print("\nAvailable modalities:")

        # List available modalities
        modalities_dir = Path(__file__).parent / "guti" / "modalities"
        for item in sorted(modalities_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("_"):
                modality_file = item / "modality.py"
                if modality_file.exists():
                    print(f"  - {item.name}")
        sys.exit(1)

    # Find the modality class (look for subclass of ImagingModality)
    modality_class = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, 'run') and name != 'ImagingModality':
            modality_class = obj
            break

    if modality_class is None:
        print(f"Error: Could not find modality class in {modality_module}")
        sys.exit(1)

    # Create and run modality
    params = Parameters(**params_dict) if params_dict else None
    modality = modality_class(params)

    singular_values = modality.run(save_results=not args.no_save, default_run=args.default_run)

if __name__ == "__main__":
    main()
