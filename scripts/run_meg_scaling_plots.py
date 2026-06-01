import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Disable interactive show() to avoid blocking.
plt.show = lambda *args, **kwargs: None

from guti.scaling_utils import (
    plot_parameter_sweep_spectra,
    plot_first_singular_value_vs_parameter,
    plot_bitrate_vs_parameter,
)
from guti.parameters import Parameters


def run_plots(modality_name: str, sweep: str):
    if sweep == "num_sensors":
        param_key = "num_sensors"
        constant_params = Parameters(source_spacing_mm=5.0)
        suffix = "num_sensors"
    elif sweep == "source_spacing_mm":
        param_key = "source_spacing_mm"
        constant_params = Parameters(num_sensors=500)
        suffix = "source_spacing"
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    plot_parameter_sweep_spectra(
        modality_name=modality_name,
        param_key=param_key,
        constant_params=constant_params,
        ylim=(1e-5, 10),
    )
    shutil.move(
        "plots/spectra.png",
        f"plots/{modality_name}_{suffix}_spectra.png",
    )

    plot_first_singular_value_vs_parameter(
        modality_name=modality_name,
        param_key=param_key,
        constant_params=constant_params,
    )
    shutil.move(
        "plots/first_sv.png",
        f"plots/{modality_name}_{suffix}_first_sv.png",
    )

    plot_bitrate_vs_parameter(
        modality_name=modality_name,
        param_key=param_key,
        constant_params=constant_params,
        snr=2000,
    )
    shutil.move(
        "plots/bitrate.png",
        f"plots/{modality_name}_{suffix}_bitrate.png",
    )


if __name__ == "__main__":
    for modality in ["meg_opm", "meg_squid"]:
        run_plots(modality, "num_sensors")
        run_plots(modality, "source_spacing_mm")
