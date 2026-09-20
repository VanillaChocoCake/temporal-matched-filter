"""Multi-kernel matched-filter bank with upper-half-mean aggregation.

Pointwise upper-half-mean response — at each output bin, compute K
kernel convolutions and return the mean of the top ⌈K/2⌉ values. For
the default K=4 bank a hardcoded inlined numba kernel
with a 6-comparison top-2 selection is used; generic K falls back to
``np.convolve`` + ``np.partition``.

Module-level shape:

- ``make_gaussian_kernels(widths, scale=3)``: build L2-normalized Gaussian
  kernels from σ values. Returns a list of float64 arrays.
- ``matched_filter_response(psd, kernels, out=None)``: public dispatcher.
- ``prewarm_matched_filter(kernels, n_bins=64)``: compile/cache the
  K=4 hot path before realtime use.

Structure: numpy fallback + @njit inner kernel + dispatcher; ``_NUMBA_OK`` flag for graceful
degradation when numba is absent.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import windows

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    njit = None  # type: ignore[assignment]


def make_gaussian_kernels(widths: Sequence[float],
                          scale: float = 3.0) -> list[np.ndarray]:
    """Build L2-normalized Gaussian kernels from σ values (NOT FWHM).

    Parameters
    ----------
    widths : Sequence[float]
        Kernel σ values in bins (e.g. ``(1, 2, 4, 8)``).
    scale : float, default 3.0
        Kernel half-width = ``scale·σ`` (default 3 → 6σ full width
        captures >99% mass).

    Returns
    -------
    list[np.ndarray]
        One float64, C-contiguous, L2-normalised kernel per width.
    """
    out: list[np.ndarray] = []
    for sigma in widths:
        h = int(scale * sigma)
        k = windows.gaussian(2 * h + 1, std=sigma).astype(np.float64)
        k /= np.linalg.norm(k)
        out.append(np.ascontiguousarray(k))
    return out


def _matched_filter_response_np(psd: np.ndarray,
                                 kernels: Sequence[np.ndarray]
                                 ) -> np.ndarray:
    """Pure-numpy reference. Used as fallback when numba is unavailable
    and as the bit-exactness reference for the JIT path."""
    responses = np.stack(
        [np.convolve(psd, k, mode="same") for k in kernels], axis=0
    )
    split = len(kernels) // 2
    return np.partition(responses, split, axis=0)[split:].mean(axis=0)


if _NUMBA_OK:
    @njit(fastmath=True, cache=True)
    def _mf_response4_nb(psd, k0, k1, k2, k3, out):
        """K=4 hardcoded same-mode convolution + 6-comparison top-2 mean.

        Bit-equivalent to ``_matched_filter_response_np`` for K=4 (verified
        to ~1 ulp). Replaces 4× ``np.convolve`` + ``np.stack`` +
        ``np.partition`` with a single per-bin pass."""
        N = psd.shape[0]
        for j in range(N):
            acc0 = 0.0
            h0 = k0.shape[0] // 2
            for kk in range(k0.shape[0]):
                src = j + kk - h0
                if 0 <= src < N:
                    acc0 += psd[src] * k0[kk]
            acc1 = 0.0
            h1 = k1.shape[0] // 2
            for kk in range(k1.shape[0]):
                src = j + kk - h1
                if 0 <= src < N:
                    acc1 += psd[src] * k1[kk]
            acc2 = 0.0
            h2 = k2.shape[0] // 2
            for kk in range(k2.shape[0]):
                src = j + kk - h2
                if 0 <= src < N:
                    acc2 += psd[src] * k2[kk]
            acc3 = 0.0
            h3 = k3.shape[0] // 2
            for kk in range(k3.shape[0]):
                src = j + kk - h3
                if 0 <= src < N:
                    acc3 += psd[src] * k3[kk]
            top1 = acc0
            top2 = acc1
            if top2 > top1:
                tmp = top1; top1 = top2; top2 = tmp
            if acc2 >= top1:
                top2 = top1; top1 = acc2
            elif acc2 > top2:
                top2 = acc2
            if acc3 >= top1:
                top2 = top1; top1 = acc3
            elif acc3 > top2:
                top2 = acc3
            out[j] = 0.5 * (top1 + top2)
        return out


def matched_filter_response(psd: np.ndarray,
                            kernels: Sequence[np.ndarray],
                            out: np.ndarray | None = None) -> np.ndarray:
    """Pointwise upper-half-mean matched-filter response over K kernels.

    For K=4 with numba available, dispatches to ``_mf_response4_nb`` (the
    fast path). Otherwise falls back to the numpy
    reference (np.convolve + np.partition).

    Parameters
    ----------
    psd : np.ndarray
        1-D input signal (float64 expected; will be cast if not).
    kernels : Sequence[np.ndarray]
        List of L2-normalised kernels; typically from
        ``make_gaussian_kernels``. K=4 hits the fast path.
    out : np.ndarray, optional
        Pre-allocated output buffer (float64). Allocated if None.

    Returns
    -------
    np.ndarray
        Per-bin upper-half-mean response over the kernel bank.
    """
    if (
        _NUMBA_OK
        and len(kernels) == 4
        and psd.shape[0] >= max(k.shape[0] for k in kernels)
    ):
        if out is None:
            out = np.empty(psd.shape[0], dtype=np.float64)
        return _mf_response4_nb(
            psd, kernels[0], kernels[1], kernels[2], kernels[3], out,
        )
    resp = _matched_filter_response_np(psd, kernels)
    if out is not None:
        out[:] = resp
        return out
    return resp


def prewarm_matched_filter(kernels: Sequence[np.ndarray],
                            n_bins: int = 64) -> None:
    """Trigger numba JIT compile/cache-load for the K=4 hot path.

    Call once during estimator construction to pay the ~100 ms cold
    JIT cost up-front rather than on the first realtime frame.
    No-op when numba is unavailable or K≠4.
    """
    if _NUMBA_OK and len(kernels) == 4:
        z = np.zeros(n_bins, dtype=np.float64)
        _mf_response4_nb(
            z, kernels[0], kernels[1], kernels[2], kernels[3],
            np.empty_like(z),
        )
