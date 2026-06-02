"""NumPy-only tests for the linear-operator + SLQ bitrate framework.

These run without torch/GPU and pin the SLQ math against the exact spectrum.
"""

import numpy as np

from guti.core import get_bitrate
from guti.linop import DenseForwardOperator, ChunkedForwardOperator, as_operator
from guti.slq import bitrate_slq


def _random_matrix(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))


def test_dense_operator_matvec_rmatvec():
    A = _random_matrix(20, 12)
    op = DenseForwardOperator(A)
    X = _random_matrix(12, 3, seed=1)
    Y = _random_matrix(20, 4, seed=2)
    assert np.allclose(op.matvec(X), A @ X)
    assert np.allclose(op.rmatvec(Y), A.T @ Y)
    assert np.allclose(op.to_dense(), A)
    assert op.shape == (20, 12)


def test_chunked_operator_matches_dense():
    A = _random_matrix(17, 9, seed=3)
    # Build via uneven row blocks to exercise offset bookkeeping.
    blocks = [A[0:5], A[5:11], A[11:17]]
    op = ChunkedForwardOperator(
        n=9,
        block_builder=lambda i: blocks[i],
        block_row_counts=[5, 6, 6],
        dtype=A.dtype,
    )
    X = _random_matrix(9, 2, seed=4)
    Y = _random_matrix(17, 2, seed=5)
    assert op.shape == (17, 9)
    assert np.allclose(op.matvec(X), A @ X)
    assert np.allclose(op.rmatvec(Y), A.T @ Y)

    # from_row_builder convenience path
    op2 = ChunkedForwardOperator.from_row_builder(
        17, 9, lambda s, e: A[s:e], row_chunk=4, dtype=A.dtype
    )
    assert np.allclose(op2.matvec(X), A @ X)
    assert np.allclose(op2.rmatvec(Y), A.T @ Y)


def test_slq_basis_matches_exact_bitrate():
    """With basis probes + enough Lanczos steps, SLQ is exact (= get_bitrate)."""
    A = _random_matrix(14, 10, seed=6)
    noise = 0.3
    s = np.linalg.svd(A, compute_uv=False)
    exact = get_bitrate(s, noise)
    est = bitrate_slq(
        A, noise, probe="basis", num_lanczos=10, side="right"
    )  # d = n = 10
    assert np.isclose(est, exact, rtol=1e-6, atol=1e-6), (est, exact)


def test_slq_rademacher_approximates_exact_bitrate():
    A = _random_matrix(30, 18, seed=7)
    noise = 0.5
    s = np.linalg.svd(A, compute_uv=False)
    exact = get_bitrate(s, noise)
    est = bitrate_slq(
        A,
        noise,
        probe="rademacher",
        num_probes=200,
        num_lanczos=18,
        rng=np.random.default_rng(123),
    )
    # Stochastic estimate: expect within a few percent.
    assert abs(est - exact) / exact < 0.05, (est, exact)


def test_as_operator_passthrough():
    A = _random_matrix(4, 4, seed=8)
    op = DenseForwardOperator(A)
    assert as_operator(op) is op
    assert isinstance(as_operator(A), DenseForwardOperator)


if __name__ == "__main__":
    test_dense_operator_matvec_rmatvec()
    test_chunked_operator_matches_dense()
    test_slq_basis_matches_exact_bitrate()
    test_slq_rademacher_approximates_exact_bitrate()
    test_as_operator_passthrough()
    print("all SLQ/linop tests passed")
