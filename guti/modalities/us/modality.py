"""Ultrasound (free-field) imaging modality — matrix-free, SLQ bitrate.

US is the canonical example of a modality that should NOT materialize its
forward matrix: at production scale (~150k sources x 10k sensors x time gates)
the operator is hundreds of GB. Instead it uses the matrix-free pipeline:

    forward_operator()  -> ChunkedForwardOperator   (builds receiver-row blocks
                                                      on demand, never the whole A)
    bitrate_method="slq" -> guti.slq.bitrate_slq     (trace-log via stochastic
                                                      Lanczos quadrature, no SVD)

So ``run()`` dispatches to the SLQ path in the base class. The geometry and
free-field propagation mirror the production Modal pipeline
(``modal_us_analytical.py`` / ``run_modal_us_analytical_sweep.py``), which
remains the heavy multi-GPU runner for large sweeps — this class is the
in-process, single-machine entry point sharing the same physics and the same
shared SLQ estimator.

Requires torch (and jax, pulled in by ``us.utils``); imported lazily so the
class can be inspected without them. The propagation build has not been
executed in this environment (no torch/jax/GPU).
"""

import numpy as np

from guti.base_modality import ImagingModality
from guti.parameters import Parameters
from guti.core import get_grid_positions, get_sensor_positions, BRAIN_RADIUS

# Free-field constants (mirror run_us_simulation in modal_us_analytical.py).
_MIN_SPEED = 1500.0          # m/s
_PPW = 24                    # points per wavelength -> voxel size
_TIME_DURATION_S = 120e-6
_DEFAULT_CENTER_FREQ_HZ = 50e3
_RECEIVER_BATCH = 256        # receivers per row-block


def _source_spacing_for_count(n_sources: int) -> float:
    """Grid spacing (mm) giving ~n_sources points in the brain hemisphere.

    Matches ``create_sources_real`` in the Modal pipeline.
    """
    volume = (2.0 / 3.0) * np.pi * BRAIN_RADIUS**3
    return (volume / n_sources) ** (1.0 / 3.0)


class USModality(ImagingModality):
    @property
    def name(self) -> str:
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
        self.sources = get_grid_positions(self.params.source_spacing_mm)
        self.sensors = get_sensor_positions(self.params.num_sensors)
        self.params.num_brain_grid_points = len(self.sources)

    def forward_operator(self):
        import torch
        from guti.linop import ChunkedForwardOperator
        from guti.modalities.us.utils import simulate_free_field_propagation

        center_freq = self.params.frequency_hz or _DEFAULT_CENTER_FREQ_HZ
        ts = self.params.temporal_sampling or 1

        time_step = 1e-1 / center_freq
        time_axis = np.arange(0, _TIME_DURATION_S, time_step)
        nt = time_axis.shape[0] // ts + 1  # matches Modal row accounting

        n_sources = len(self.sources)
        source_signals = np.tile(
            np.sin(2 * np.pi * time_axis * center_freq), (n_sources, 1)
        )
        dx_m = _MIN_SPEED / (_PPW * center_freq)
        voxel_size = np.array([dx_m, dx_m, dx_m])

        sources_t = torch.as_tensor(self.sources)
        signals_t = torch.as_tensor(source_signals)
        voxel_t = torch.as_tensor(voxel_size)

        n_recv = len(self.sensors)
        starts = list(range(0, n_recv, _RECEIVER_BATCH))
        block_row_counts = [
            (min(s + _RECEIVER_BATCH, n_recv) - s) * nt for s in starts
        ]

        self.params.time_resolution = time_step

        def block_builder(i, _starts=starts):
            s = _starts[i]
            e = min(s + _RECEIVER_BATCH, n_recv)
            receivers_t = torch.as_tensor(self.sensors[s:e])
            pf = simulate_free_field_propagation(
                sources_t,
                receivers_t,
                signals_t,
                time_step,
                center_freq,
                voxel_t,
                compute_time_series=True,
                temporal_sampling=ts,
            )
            # (batch, nt, n_sources) -> (batch*nt, n_sources)
            return pf.permute(0, 2, 1).reshape(-1, n_sources).float()

        return ChunkedForwardOperator(
            n_sources,
            block_builder,
            block_row_counts,
            backend="torch",
            dtype=torch.float32,
        )

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
