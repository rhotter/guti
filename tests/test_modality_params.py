from guti.parameters import Parameters
from guti.modalities.blur_1d.modality import Blur1D
from guti.modalities.cw_fnirs.modality import CWfNIRS
from guti.modalities.us.modality import USModality


def test_partial_params_override_cw_fnirs_defaults():
    modality = CWfNIRS(Parameters(num_sensors=12))

    assert modality.params.num_sensors == 12
    assert modality.params.grid_resolution_mm == 6.0
    assert modality.params.max_dist == 50.0


def test_partial_params_override_us_defaults():
    modality = USModality(Parameters(bitrate_method="svd"))

    assert modality.params.bitrate_method == "svd"
    assert modality.params.num_sensors == 100
    assert modality.params.source_spacing_mm == 10.0
    assert modality.params.noise_full_brain == 1e-7
    assert modality.params.slq_num_lanczos == 40


def test_partial_params_override_blur_defaults():
    modality = Blur1D(Parameters(input_dim=16))

    assert modality.params.input_dim == 16
    assert modality.params.output_dim == 128
