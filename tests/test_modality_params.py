import pytest

from guti.parameters import Parameters
from guti.modalities.blur_1d.modality import Blur1D
from guti.modalities.cw_fnirs.modality import CWfNIRS
from guti.modalities.eeg.modality import EEGModality
from guti.modalities.meg.meg import OPM_OFFSET_MM, SQUID_OFFSET_MM
from guti.modalities.meg.modality import MEGModality
from guti.modalities.td_fnirs.modality import TDfNIRSAnalytical
from guti.modalities.us.modality import USModality
from guti.noise_models import canonicalize_modality_name


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


def test_runnable_modality_names_match_folders_except_meg_variants():
    assert Blur1D().name == "blur_1d"
    assert CWfNIRS().name == "cw_fnirs"
    assert EEGModality().name == "eeg"
    assert TDfNIRSAnalytical().name == "td_fnirs"
    assert USModality().name == "us"

    assert MEGModality(Parameters(sensor_offset_mm=OPM_OFFSET_MM)).name == "meg_opm"
    assert MEGModality(Parameters(sensor_offset_mm=SQUID_OFFSET_MM)).name == "meg_squid"


def test_folder_name_noise_model_aliases_keep_meg_explicit():
    assert canonicalize_modality_name("eeg") == "eeg_openmeeg"
    assert canonicalize_modality_name("td_fnirs") == "td_fnirs_analytical"
    assert canonicalize_modality_name("us") == "us_analytical"
    assert canonicalize_modality_name("meg_opm") == "meg_opm"
    assert canonicalize_modality_name("meg_squid") == "meg_squid"

    with pytest.raises(KeyError):
        canonicalize_modality_name("meg")
