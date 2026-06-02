"""Ultrasound (free-field) imaging modality — matrix-free, SLQ bitrate.

US is the canonical example of a modality that should NOT materialize its
forward matrix: at production scale (~150k sources x 10k sensors x time gates)
the operator is hundreds of GB. Instead it uses the matrix-free pipeline:

    forward_operator()  -> ChunkedForwardOperator   (builds receiver-row blocks
                                                      on demand, never the whole A)
    bitrate_method="slq" -> guti.slq.bitrate_slq     (trace-log via stochastic
                                                      Lanczos quadrature, no SVD)

So ``run()`` dispatches to the SLQ path in the base class. Geometry and
free-field physics come from :mod:`guti.modalities.us.utils` (the single shared
definition), so this in-process, single-machine entry point and the production
GPU CLI (``analytical.py`` / ``scripts/run_modal_us_analytical_sweep.py``) build
the exact same operator.

Requires torch; imported lazily so the class can be inspected without it.
"""

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.modalities.us.utils import (
    DEFAULT_CENTER_FREQ_HZ as _DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_RECEIVER_BATCH as _RECEIVER_BATCH,
    free_field_source_spacing_mm as _source_spacing_for_count,
    create_free_field_sources,
    create_free_field_receivers,
    build_free_field_operator,
)


class USModality(ImagingModality):
    @property
    def name(self) -> str:
        return "us"

    @property
    def noise_model_name(self) -> str:
        return "us_analytical"

    def _get_default_modality_params(self) -> Parameters:
        # Small, single-machine configuration.
        return Parameters(
            num_sensors=100,
            source_spacing_mm=10.0,
            frequency_hz=_DEFAULT_CENTER_FREQ_HZ,
            temporal_sampling=5,
            bitrate_method="slq",
            noise_full_brain=1e-7,
            slq_num_probes=16,
            slq_num_lanczos=40,
        )

    @classmethod
    def scaled_up_params(cls) -> Parameters:
        """Production-scale US configuration (asymptotic bitrate).

        ~150k sources x 10k sensors at 50 kHz. The dense operator is hundreds of
        GB, so the bitrate is computed matrix-free via SLQ. At this scale run it
        on GPUs through ``run_modal_us_analytical_sweep.py`` (which calls the
        same shared SLQ estimator); the in-process build here will be far too
        slow/large on a single CPU.
        """
        return Parameters(
            num_sensors=10000,
            source_spacing_mm=_source_spacing_for_count(150000),
            frequency_hz=_DEFAULT_CENTER_FREQ_HZ,
            temporal_sampling=5,
            bitrate_method="slq",
            noise_full_brain=1e-7,
            slq_num_probes=128,
            slq_num_lanczos=128,
            comment="production scale; run via run_modal_us_analytical_sweep.py on GPUs",
        )

    def setup_geometry(self) -> None:
        # Shared helpers return positions in meters (propagation physics is in m).
        self.sources = create_free_field_sources(
            source_spacing_mm=self.params.source_spacing_mm
        )
        self.sensors = create_free_field_receivers(self.params.num_sensors)
        self.params.num_brain_grid_points = len(self.sources)

    def forward_operator(self):
        import torch

        operator, meta = build_free_field_operator(
            self.sources,
            self.sensors,
            center_frequency=self.params.frequency_hz or _DEFAULT_CENTER_FREQ_HZ,
            temporal_sampling=self.params.temporal_sampling or 1,
            receiver_batch=_RECEIVER_BATCH,
            backend="torch",
            device="cpu",
            dtype=torch.float32,
        )
        self.params.time_resolution = meta["time_resolution"]
        return operator

    def compute_forward_model(self):
        """Dense fallback (small configs only) for the SVD path / debugging.

        Builds the full matrix by concatenating every row block — only viable
        for tiny problems. The production path uses ``forward_operator`` +
        ``bitrate_method="slq"`` and never calls this.
        """
        import torch

        op = self.forward_operator()
        blocks = [op.block_builder(i) for i in range(op.num_blocks)]
        return torch.cat(blocks, dim=0)


if __name__ == "__main__":
    modality = USModality()
    singular_values = modality.run()
