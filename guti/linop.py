"""Matrix-free linear-operator abstraction shared by the bitrate estimators.

A forward model ``A`` maps sources (length ``n``) to measurements (length
``m``). Some modalities can hold ``A`` densely; others (e.g. scaled-up
ultrasound) cannot — ``A`` would be hundreds of GB. Both, however, only ever
need to apply ``A`` and ``Aᵀ`` to blocks of vectors. This module captures that
common interface so the SVD and SLQ bitrate estimators can consume either kind
of operator identically.

Backends: arrays may be NumPy or PyTorch. Operations are written to work for
both via duck typing; tiny dense linear algebra (e.g. a Lanczos tridiagonal
eigendecomposition) is done in NumPy by the consumers, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def is_torch(x) -> bool:
    """True if ``x`` is a torch tensor (without importing torch unnecessarily)."""
    return type(x).__module__.split(".")[0] == "torch"


def transpose(M):
    """2D transpose that works for both NumPy arrays and torch tensors."""
    if is_torch(M):
        return M.transpose(0, 1)
    return M.T


class ForwardOperator(ABC):
    """Abstract ``m x n`` linear operator exposing ``matvec`` / ``rmatvec``.

    Subclasses implement how ``A @ X`` and ``Aᵀ @ X`` are applied. ``X`` is
    always 2D with shape ``(n, k)`` for ``matvec`` and ``(m, k)`` for
    ``rmatvec`` (``k`` = number of probe/Krylov columns).
    """

    def __init__(self, shape, *, backend: str = "numpy", dtype=None, device=None):
        self.shape: Tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.backend = backend
        self.dtype = dtype
        self.device = device

    @property
    def m(self) -> int:
        return self.shape[0]

    @property
    def n(self) -> int:
        return self.shape[1]

    @abstractmethod
    def matvec(self, X):
        """Apply ``A @ X``: ``(n, k) -> (m, k)``."""

    @abstractmethod
    def rmatvec(self, X):
        """Apply ``Aᵀ @ X``: ``(m, k) -> (n, k)``."""

    # -- backend helpers (used by estimators) --------------------------------

    def zeros(self, shape):
        if self.backend == "torch":
            import torch

            return torch.zeros(shape, dtype=self.dtype, device=self.device)
        return np.zeros(shape, dtype=self.dtype)

    def rademacher(self, shape, rng: Optional[np.random.Generator] = None):
        """Random ±1 tensor in this operator's backend."""
        if self.backend == "torch":
            import torch

            g = None
            if rng is not None:
                # Derive a torch seed from the numpy generator for reproducibility.
                g = torch.Generator(device=self.device)
                g.manual_seed(int(rng.integers(0, 2**63 - 1)))
            r = torch.randint(0, 2, shape, generator=g, device=self.device)
            return (r * 2 - 1).to(self.dtype or torch.float64)
        r = rng if rng is not None else np.random.default_rng()
        return (r.integers(0, 2, size=shape).astype(np.float64) * 2 - 1)

    def to_dense(self):
        """Materialize the dense matrix via ``matvec`` on the identity.

        WARNING: allocates the full ``m x n`` matrix; only for small operators
        / debugging. Subclasses backed by a real array should override.
        """
        eye = self.zeros((self.n, self.n))
        if self.backend == "torch":
            import torch

            eye = torch.eye(self.n, dtype=self.dtype, device=self.device)
        else:
            eye = np.eye(self.n, dtype=self.dtype)
        return self.matvec(eye)


class DenseForwardOperator(ForwardOperator):
    """Wraps an in-memory dense matrix (NumPy array or torch tensor)."""

    def __init__(self, A):
        backend = "torch" if is_torch(A) else "numpy"
        device = getattr(A, "device", None)
        super().__init__(A.shape, backend=backend, dtype=A.dtype, device=device)
        self.A = A

    def matvec(self, X):
        return self.A @ X

    def rmatvec(self, X):
        return transpose(self.A) @ X

    def to_dense(self):
        return self.A


class ChunkedForwardOperator(ForwardOperator):
    """Matrix-free operator assembled from horizontal row blocks on demand.

    The matrix is partitioned into ``len(block_row_counts)`` blocks of rows.
    ``block_builder(i)`` returns block ``i`` as an array of shape
    ``(block_row_counts[i], n)``. Blocks are built, used, and discarded inside
    each ``matvec`` / ``rmatvec``, so the full matrix is never materialized.

    This generalizes the ultrasound ``compute_chunk_matrix`` pattern: there
    each block is one batch of receivers (contributing ``nt`` rows each).
    """

    def __init__(
        self,
        n: int,
        block_builder: Callable[[int], object],
        block_row_counts: Sequence[int],
        *,
        backend: str = "numpy",
        dtype=None,
        device=None,
    ):
        m = int(sum(block_row_counts))
        super().__init__((m, n), backend=backend, dtype=dtype, device=device)
        self.block_builder = block_builder
        self.block_row_counts = [int(c) for c in block_row_counts]
        # Cumulative row offsets per block.
        self._offsets = [0]
        for c in self.block_row_counts:
            self._offsets.append(self._offsets[-1] + c)

    @classmethod
    def from_row_builder(
        cls,
        m: int,
        n: int,
        build_rows: Callable[[int, int], object],
        *,
        row_chunk: int = 4096,
        backend: str = "numpy",
        dtype=None,
        device=None,
    ) -> "ChunkedForwardOperator":
        """Build from a ``build_rows(start, end) -> (end-start, n)`` callable."""
        starts = list(range(0, m, row_chunk))
        counts = [min(s + row_chunk, m) - s for s in starts]

        def block_builder(i, _starts=starts, _row_chunk=row_chunk):
            s = _starts[i]
            e = min(s + _row_chunk, m)
            return build_rows(s, e)

        return cls(
            n, block_builder, counts, backend=backend, dtype=dtype, device=device
        )

    @property
    def num_blocks(self) -> int:
        return len(self.block_row_counts)

    def matvec(self, X):
        out = self.zeros((self.m, X.shape[1]))
        for i in range(self.num_blocks):
            s, e = self._offsets[i], self._offsets[i + 1]
            out[s:e] = self.block_builder(i) @ X
        return out

    def rmatvec(self, X):
        out = self.zeros((self.n, X.shape[1]))
        for i in range(self.num_blocks):
            s, e = self._offsets[i], self._offsets[i + 1]
            out = out + transpose(self.block_builder(i)) @ X[s:e]
        return out


def as_operator(A) -> ForwardOperator:
    """Coerce a dense array or an existing ForwardOperator into a ForwardOperator."""
    if isinstance(A, ForwardOperator):
        return A
    return DenseForwardOperator(A)
