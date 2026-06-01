"""Stochastic Lanczos Quadrature (SLQ) bitrate estimator.

This is a general, modality-agnostic alternative to the SVD bitrate pipeline.
Given a forward operator ``A`` (a :class:`guti.linop.ForwardOperator`) and a
detector noise level, it estimates the Gaussian-channel bitrate

    C = 1/(2 * time_resolution) * Σ_i log₂(1 + σ_i² / noise²)
      = 1/(2 * time_resolution) * (1/ln2) * tr ln(I + (1/noise²) A Aᵀ)

WITHOUT computing the SVD or ever materializing ``A``. The trace-log is
estimated by stochastic trace estimation (Hutchinson) with each quadratic form
evaluated via Lanczos quadrature. The only thing touched matrix-side is
``A @ x`` / ``Aᵀ @ x``, so a matrix-free :class:`ChunkedForwardOperator` works
just as well as a dense one.

This convention matches :func:`guti.core.get_bitrate` (equal/iid input power),
so on a small dense operator ``bitrate_slq`` agrees with
``get_bitrate(svdvals(A), noise)`` — see ``tests/test_slq.py``. (It is NOT the
water-filling capacity, which would require the spectrum SLQ avoids.)

Backends: works for NumPy and PyTorch operators. Matvecs and probes stay in the
operator's backend; the tiny per-probe tridiagonal eigendecomposition is done
in NumPy.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np

from guti.linop import ForwardOperator, as_operator, transpose


def _scalar(x) -> float:
    """Extract a Python float from a (1,1) NumPy/torch result."""
    return float(x.reshape(()) if hasattr(x, "reshape") else x)


def _dot(a, b) -> float:
    return _scalar(transpose(a) @ b)


def _norm(v) -> float:
    return math.sqrt(max(_dot(v, v), 0.0))


def _lanczos(B_mv, z, num_steps: int):
    """Lanczos on ``B`` started at ``z`` (with full reorthogonalization).

    Returns (alphas, betas) as Python-float lists defining the symmetric
    tridiagonal ``T`` (alphas on the diagonal, betas on the off-diagonal).
    ``B_mv(v)`` applies the (implicit) SPD matrix ``B`` to a ``(d, 1)`` vector.
    """
    beta = _norm(z)
    if beta == 0.0:
        return [0.0], []
    q = z / beta
    Q = [q]
    alphas: list[float] = []
    betas: list[float] = []
    q_prev = None
    beta_prev = 0.0
    for k in range(num_steps):
        w = B_mv(q)
        if k > 0:
            w = w - beta_prev * q_prev
        a = _dot(q, w)
        alphas.append(a)
        w = w - a * q
        # Full reorthogonalization against all previous Lanczos vectors.
        for qi in Q:
            w = w - _dot(qi, w) * qi
        b = _norm(w)
        if b < 1e-12:
            break
        betas.append(b)
        q_prev = q
        beta_prev = b
        q = w / b
        Q.append(q)
    return alphas, betas


def _quadrature(alphas, betas, f) -> float:
    """Gauss quadrature weight·f(node) sum from a Lanczos tridiagonal."""
    t = len(alphas)
    T = np.zeros((t, t), dtype=np.float64)
    for i in range(t):
        T[i, i] = alphas[i]
    for i in range(len(betas)):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    theta, Y = np.linalg.eigh(T)
    weights = Y[0, :] ** 2
    return float(np.sum(weights * f(theta)))


def bitrate_slq(
    A,
    noise_std: float,
    *,
    num_probes: int = 16,
    num_lanczos: int = 40,
    side: Literal["auto", "left", "right"] = "auto",
    probe: Literal["rademacher", "basis"] = "rademacher",
    time_resolution: float = 1.0,
    n_detectors: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Estimate the equal-power bitrate of ``A`` via stochastic Lanczos quadrature.

    Parameters
    ----------
    A : ForwardOperator | array
        The forward operator (or a dense array, which is wrapped).
    noise_std : float
        Detector noise standard deviation.
    num_probes : int
        Number of stochastic probe vectors (Hutchinson samples). Ignored when
        ``probe="basis"`` (which uses all ``d`` basis vectors for an exact,
        deterministic trace — only practical for small ``d``, used in tests).
    num_lanczos : int
        Lanczos steps per probe.
    side : {"auto","left","right"}
        Whether to work with ``B = A Aᵀ`` ("left", ``d = m``) or ``B = Aᵀ A``
        ("right", ``d = n``). "auto" picks the smaller dimension.
    probe : {"rademacher","basis"}
        Probe distribution. "basis" gives an exact trace (deterministic) and is
        intended for testing on small operators.
    time_resolution : float
        Same meaning as in :func:`guti.core.get_bitrate`.
    n_detectors : int, optional
        If given, the noise variance is divided by ``n_detectors`` (matches the
        ultrasound SLQ convention of per-detector noise scaling).
    rng : numpy.random.Generator, optional
        Source of randomness for reproducible probes.

    Returns
    -------
    float
        Estimated bitrate in bits per sample-period.
    """
    op: ForwardOperator = as_operator(A)
    m, n = op.shape
    if side == "auto":
        left = m <= n
    else:
        left = side == "left"
    d = m if left else n

    noise_var = noise_std ** 2
    if n_detectors is not None:
        noise_var = noise_var / n_detectors

    def B_mv(v):
        # Apply B = A Aᵀ (left) or Aᵀ A (right) to a (d, 1) vector.
        if left:
            return op.matvec(op.rmatvec(v))
        return op.rmatvec(op.matvec(v))

    def f(theta):
        # f(λ) = ln(1 + λ / noise_var); clamp tiny negatives from roundoff.
        return np.log1p(np.clip(theta, 0.0, None) / noise_var)

    if probe == "basis":
        # Exact trace: sum of e_iᵀ f(B) e_i over the standard basis.
        total = 0.0
        for i in range(d):
            z = op.zeros((d, 1))
            z[i] = 1.0
            alphas, betas = _lanczos(B_mv, z, num_lanczos)
            total += _quadrature(alphas, betas, f)  # ||e_i||² = 1
        trace_est = total
    else:
        if rng is None:
            rng = np.random.default_rng()
        total = 0.0
        for _ in range(num_probes):
            z = op.rademacher((d, 1), rng)
            znorm2 = _dot(z, z)  # = d for Rademacher
            alphas, betas = _lanczos(B_mv, z, num_lanczos)
            total += znorm2 * _quadrature(alphas, betas, f)
        trace_est = total / num_probes

    return (0.5 / math.log(2.0)) * trace_est / time_resolution
