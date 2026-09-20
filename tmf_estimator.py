"""Temporal matched-filter (TMF) tune estimator: matched-filter bank + motion-compensated EMA +
wide-centroid sub-bin readout.

Deterministic Schottky-tune estimator. Operates on the PSD mapped onto the
folded tune axis; no learnable parameters; designed for quasi-real-time
operation with at most one frame of lag.

Pipeline (input → output):

    mapped_psd  →  [M1] Postprocess (low-q suppress + Savitzky-Golay)
                →  [M2] Motion-comp EMA (shift accumulator + adaptive decay)
                →  [M3] Matched-filter bank (K=4 Gaussian; upper-half-mean)
                →  [M4] Readout (local-window argmax + jump gate
                       + wide MAD-gated centroid sub-bin)
                →  q_out = clip(μ, 0, N-1) · bin_step

Reusable DSP primitives live in tune_pipeline.dsp.*:
- tune_pipeline.dsp.savgol — cached-coef SG (bit-exact vs scipy mode='interp')
- tune_pipeline.dsp.matched_filter — K-kernel MF bank with upper-half-mean

The fused mc-EMA shift+update kernel stays in this file — it encodes the
estimator-specific blend rule.

Public API:
    TMFEstimator — main class; ``.process(frame)`` → MeasurementResult.
    TMFConfig    — frozen dataclass of all tunable parameters.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from tune_pipeline.frames import FrontendFrame, MeasurementResult, QualityFlag
from tune_pipeline.dsp.matched_filter import (
    make_gaussian_kernels,
    matched_filter_response,
    prewarm_matched_filter,
)
from tune_pipeline.dsp.savgol import (
    make_savgol_interp_coeffs,
    prewarm_savgol_interp,
    savgol_filter_interp,
)

import numba as _numba

# Shift-magnitude epsilon: below this, treat as no shift (skip interp).
_SHIFT_EPS = 1e-6


@_numba.njit(fastmath=True, cache=True)
def _ema_shift_update_njit(ema_psd, psd_clean, shift_bins, decay, out):
    """Fused mc-EMA shift + EMA decay update, single per-bin pass.

    Equivalent to ``shift(ema_psd, v_bins) → ema_psd; np.multiply(ema_psd,
    decay) + (1-decay)*psd_clean`` in one numba pass, without the second
    full-array pass and the ``(1-decay)*psd_clean`` intermediate
    allocation. Output is written to ``out``; caller swaps buffers.
    """
    N = ema_psd.shape[0]
    keep = decay
    add = 1.0 - decay
    if abs(shift_bins) < _SHIFT_EPS:
        for j in range(N):
            out[j] = keep * ema_psd[j] + add * psd_clean[j]
    else:
        for j in range(N):
            src = j - shift_bins
            i_lo = int(np.floor(src))
            i_hi = i_lo + 1
            if i_lo >= 0 and i_hi < N:
                frac = src - i_lo
                shifted = ema_psd[i_lo] * (1.0 - frac) + ema_psd[i_hi] * frac
            else:
                shifted = 0.0
            out[j] = keep * shifted + add * psd_clean[j]
    return out


@dataclass(frozen=True)
class TMFConfig:
    # Each parameter is tagged PHYSICAL / DERIVED / HEURISTIC / EMPIRICAL
    # according to how its value is obtained.
    q_suppress_below: float = 0.03    # PHYSICAL: margin below the minimum operating tune ~0.05
    initial_q: float = 0.333          # EMPIRICAL: cold-start prior in [0, 0.5]
    mapwin_tune: float = 1e-2         # tune-unit SG window W_SG, converted to the nearest odd bin count on the operating grid (21 bins at L=1024)
    mapping_window_size: Optional[int] = None  # optional explicit bin override for the SG window; None -> derive from mapwin_tune
    # Kernel σ in TUNE units (NOT FWHM, NOT bins). Converted to bins per-L in
    # TMFState.new (σ_bins = σ_tune / bin_step, bin_step = 0.5/L) so the MF bank
    # is q-resolution-adaptive. At L=1024 → ≈(1,2,4,8) bins.
    mf_widths_tune: tuple[float, ...] = (0.5e-3, 1e-3, 2e-3, 4e-3)
    sg_polyorder: int = 2             # DERIVED: Taylor 2nd-order curvature
    ema_decay: float = 0.50           # EMPIRICAL: N_eff=2 frames (1/(1-β))
    half_win_tune: float = 1e-2       # tune-unit local-search half-width w_q, converted to bins on the operating grid; v_limit = 2*w_q = 0.02 tune/frame
    mc_v_hist_len: int = 3            # HEURISTIC/EMPIRICAL: v̂ window length
    # --- Sub-bin readout and pooling depth ---
    # (1) WIDE-window MAD-gated weighted centroid sub-bin readout. On a fast trajectory
    # the tune sweeps WITHIN one acquisition, so the per-frame peak is smeared over
    # ~10-23 bins; a wide (±0.05 tune) centroid recovers the smeared MEAN (= true tune)
    # instead of locking the biased peak top. The gate factor and the window follow the
    # standard weighted-centroid readout (no additional free parameters).
    centroid_half_tune: float = 0.05  # half-width of the wide centroid window, in tune units
    # (2) One-sided motion-adaptive pool depth β = min(ema_decay, σ₁/(|v̂|+σ₁)) — shortens
    # the pool on fast trajectories to remove the EMA temporal lag (the wide centroid
    # handles the within-frame smear; this handles the cross-frame lag); never deeper
    # than N_eff = 2 frames. PSD pooling is kept (it is the robust low-SNR denoiser —
    # it averages BEFORE peak localisation). β = σ₁/(|v̂|+σ₁) > 0 self-limits to
    # ~1/(v_limit+1)=0.018 (|v̂| clipped at v_limit), so a very fast peak is read
    # essentially per-frame and no explicit lower floor is needed.
    adaptive_decay_enabled: bool = True  # DERIVED form + capped at ema_decay; 0 free params


@dataclass
class TMFState:
    ema_psd: np.ndarray              # EMA'd PSD buffer
    ema_scratch: np.ndarray          # ping-pong buffer for fused shift+update
    q_prev: float
    frame_idx: int
    kernels: list
    q_prev_prev: float = 0.0   # for motion-comp v̂ estimate
    v_hist: Deque[float] = field(default_factory=deque)
    frev_prev: float = 4e6     # revolution-frequency fallback carried for the preprocessing stage
    mf_widths_bins: tuple = ()  # cfg.mf_widths_tune converted to bins at this frame's L (set in .new)
    # Per-frame scratch buffers (caller-owned; eliminate hot-path allocation).
    sg_tmp: Optional[np.ndarray] = None
    sg_out: Optional[np.ndarray] = None
    mf_out: Optional[np.ndarray] = None
    mad_scratch: Optional[np.ndarray] = None

    @classmethod
    def new(cls, cfg: TMFConfig, tune_unit: np.ndarray) -> "TMFState":
        # MF-bank σ are configured in TUNE units; convert to BINS at this frame's
        # q-resolution so the bank is L-adaptive: σ_bins = σ_tune / bin_step.
        bin_step = float(tune_unit[1] - tune_unit[0])
        mf_widths_bins = tuple(w / bin_step for w in cfg.mf_widths_tune)
        return cls(
            ema_psd=np.zeros_like(tune_unit),
            ema_scratch=np.empty_like(tune_unit),
            q_prev=cfg.initial_q,
            frame_idx=0,
            kernels=make_gaussian_kernels(mf_widths_bins),
            mf_widths_bins=mf_widths_bins,
            q_prev_prev=cfg.initial_q,
            v_hist=deque(maxlen=cfg.mc_v_hist_len),
            # float64 explicitly: shared DSP out/tmp contracts require it,
            # independent of tune_unit's dtype.
            sg_tmp=np.empty(tune_unit.shape[0], dtype=np.float64),
            sg_out=np.empty(tune_unit.shape[0], dtype=np.float64),
            mf_out=np.empty(tune_unit.shape[0], dtype=np.float64),
            mad_scratch=np.empty(tune_unit.shape[0], dtype=np.float64),
        )

    def clear(self, cfg: TMFConfig) -> None:
        self.ema_psd[:] = 0.0
        self.q_prev = cfg.initial_q
        self.q_prev_prev = cfg.initial_q
        self.v_hist = deque(maxlen=cfg.mc_v_hist_len)
        self.frame_idx = 0
        self.frev_prev = 4e6
        # Scratch buffers (sg_*/mf_out/mad_scratch) need no reset:
        # fully overwritten before read each frame.


@dataclass
class TMFHistory:
    q_predicted: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    failed_to_detect: Deque[bool] = field(default_factory=lambda: deque(maxlen=2048))

    def clear(self) -> None:
        self.q_predicted.clear()
        self.failed_to_detect.clear()


class TMFEstimator:
    """Classical tune estimator: motion-compensated EMA, matched-filter bank,
    MAD-gated centroid sub-bin readout.

    Consumes ``frame.mapped_psd`` (the tune-axis PSD, 1024 bins at the
    deployed operating point) and emits ``MeasurementResult(q, failed, source)``.
    """

    def __init__(self, cfg: TMFConfig, tune_unit: np.ndarray,
                 state: Optional[TMFState] = None,
                 history: Optional[TMFHistory] = None):
        self.cfg = cfg
        self.tune_unit = tune_unit
        self.state = state if state is not None else TMFState.new(cfg, tune_unit)
        self.history = history if history is not None else TMFHistory()
        self._init_static_dsp()
        self._prewarm_jit()

    def _init_static_dsp(self) -> None:
        """Precompute frame-invariant masks and SG coefficients.

        Done once per estimator instance so ``process()`` only does
        per-frame work. ``_suppress_base`` is the low-q DC-region mask
        (q < ``cfg.q_suppress_below``); ``_sg_coeffs`` is the cached
        ``SavgolInterpCoefficients`` bundle for the SG window and
        ``cfg.sg_polyorder``.
        """
        cfg = self.cfg
        tune_unit = self.tune_unit
        self._suppress_base = np.asarray(
            tune_unit < cfg.q_suppress_below, dtype=np.bool_)
        bin_step = float(tune_unit[1] - tune_unit[0])
        # SG window in bins (nearest odd count on this grid); 21 bins at L=1024.
        sg_bins = (cfg.mapping_window_size if cfg.mapping_window_size is not None
                   else int(round(cfg.mapwin_tune / bin_step)) | 1)
        self._sg_coeffs = make_savgol_interp_coeffs(
            sg_bins, cfg.sg_polyorder)
        # Wide-centroid window half-width in bins (= centroid_half_tune / bin_step).
        self._cw = int(round(cfg.centroid_half_tune / bin_step))
        # Local-search half-width in bins (= half_win_tune / bin_step); 20 bins at
        # L=1024. v_limit = 2*half_win then equals 2*w_q = 0.02 tune/frame.
        self._half_win = int(round(cfg.half_win_tune / bin_step))

    def _prewarm_jit(self) -> None:
        """Compile/cache numba kernels under sample inputs.

        Without this, the first real-time frame pays the cold JIT compile
        (~100 ms) or cache-load (~10 ms) cost. Pre-warm failure is non-fatal
        (kernels JIT-compile on the first real call instead).
        """
        cfg = self.cfg
        try:
            # Shared DSP kernels (tune_pipeline.dsp)
            prewarm_savgol_interp(
                self.tune_unit.shape[0], self._suppress_base, self._sg_coeffs)
            prewarm_matched_filter(self.state.kernels)
            # Estimator-specific kernel (local): fused mc-EMA shift + update.
            _ema_shift_update_njit(
                np.zeros(8), np.zeros(8), 0.5, cfg.ema_decay, np.empty(8))
            _ema_shift_update_njit(
                np.zeros(8), np.zeros(8), 0.0, cfg.ema_decay, np.empty(8))
        except Exception:
            # Pre-warm failure is non-fatal — the JIT will compile
            # on the first real call instead.
            pass

    def postprocess(self, mapped_psd: np.ndarray,
                    valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Postprocess: low-q suppress + SG poly fit. No clip, no baseline anchor.

        SG preserves the polynomial shape of betatron sidebands across
        width, so the downstream multi-width matched-filter bank can
        exploit shape diversity. ``valid_mask`` is OR'd into the
        suppression set so smoothing cannot leak signal back into bins
        the preprocessing stage declared invalid.
        """
        # Cached-coefficient SG via tune_pipeline.dsp.savgol; caller-owned
        # tmp/out scratch buffers avoid per-frame allocation.
        return savgol_filter_interp(
            mapped_psd, self._suppress_base, self._sg_coeffs, valid_mask,
            tmp=self.state.sg_tmp, out=self.state.sg_out)

    def _centroid_subbin(self, sig: np.ndarray, center_idx: int) -> float:
        """Wide-window MAD-gated weighted centroid around ``center_idx``.
        Window ±``_cw`` bins (= ±0.05 tune);
        keep bins above median + 2·MAD of the window; weight by (value − threshold);
        return Σ idx·w / Σ w. Falls back to the integer peak if fewer than three bins
        pass the gate. Unlike a log-parabolic vertex fit, the mass-based centroid over a
        WIDE window recovers the centroid of a within-frame-smeared (fast-trajectory)
        peak instead of locking its biased top."""
        N = sig.shape[0]
        cw = self._cw
        # Noise floor from the wide reference window (robust median/MAD).
        rlo = max(0, center_idx - cw)
        rhi = min(N, center_idx + cw + 1)
        ref = sig[rlo:rhi]
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        thr = med + 2.0 * mad
        # SNR-adaptive window in two stages:
        #  (cap) the CONTIGUOUS run of bins above the BASELINE (median) around the peak
        #        = the sideband's signal region. At high SNR this spans the full
        #        within-frame-smeared sideband (wide); at low SNR the noise crosses the
        #        median within a couple of bins so the run is short (narrow) — rejecting
        #        the spurious far noise spikes a fixed wide window would admit. Using the
        #        baseline (not median+2·MAD) as the cap bridges the dips of a bumpy
        #        smeared peak that strict 2·MAD-contiguity would cut short.
        #  (gate) inside the cap, the median+2·MAD gate + (value−thr) weight.
        lo = center_idx
        while lo - 1 >= rlo and sig[lo - 1] > med:
            lo -= 1
        hi = center_idx
        while hi + 1 < rhi and sig[hi + 1] > med:
            hi += 1
        seg = sig[lo:hi + 1]
        gate = seg > thr
        if int(np.count_nonzero(gate)) >= 3:
            w = seg[gate] - thr
            wsum = float(w.sum())
            if wsum > 1e-12:
                idx = np.nonzero(gate)[0] + lo
                return float(np.dot(idx, w) / wsum)
        return float(center_idx)

    def process(self, frame: FrontendFrame) -> MeasurementResult:
        cfg, st, tune_unit = self.cfg, self.state, self.tune_unit
        bin_step = tune_unit[1] - tune_unit[0]
        N = len(tune_unit)

        mapped_psd = frame.mapped_psd

        if int(frame.quality_flags) & int(QualityFlag.FEW_PEAKS):
            last_q = (self.history.q_predicted[-1]
                      if len(self.history.q_predicted) > 0 else cfg.initial_q)
            self._record(q_pred=last_q, failed=True)
            st.frame_idx += 1
            return MeasurementResult(q=last_q, failed=True, source="tmf")

        psd_clean = self.postprocess(mapped_psd, frame.valid_mask)

        # Motion-compensated coherent EMA: shift the accumulator to align
        # with where the peak is expected NOW, using a 3-frame arithmetic
        # mean of v̂. The shift + decay update is fused into one numba
        # kernel below (no intermediate (1-decay)*psd_clean array).
        v_bins = 0.0
        if st.frame_idx >= 2:
            # v_raw clip at ±2·half_win_bins is the tracker's algorithmic
            # self-consistency bound (DERIVED): both q_prev and q_prev_prev
            # were located by local-argmax within their own ±half_win
            # search windows, so |q_prev - q_prev_prev| ≤ 2·half_win in
            # any non-spurious frame pair. A |v_raw| exceeding this bound
            # proves at least one of the two estimates was a spurious
            # lock — clip prevents the bad value from corrupting v_hist.
            # (The factor 2 is a geometric half-to-full identity, not a
            #  free factor.)
            v_raw = (st.q_prev - st.q_prev_prev) / bin_step
            v_limit = 2.0 * self._half_win
            if v_raw < -v_limit:
                v_raw = -v_limit
            elif v_raw > v_limit:
                v_raw = v_limit
            st.v_hist.append(v_raw)
            # v_hist is a 3-element deque — native sum/len skips numpy.
            v_bins = sum(st.v_hist, 0.0) / len(st.v_hist)

        # One-sided motion-adaptive pool depth β = min(ema_decay, σ₁/(|v̂|+σ₁)),
        # σ₁ = narrowest kernel (rationale: see TMFConfig.adaptive_decay_enabled).
        decay = cfg.ema_decay
        if cfg.adaptive_decay_enabled and st.frame_idx >= 2:
            sigma1 = float(st.mf_widths_bins[0])
            beta = sigma1 / (abs(v_bins) + sigma1)
            if beta > cfg.ema_decay:
                beta = cfg.ema_decay
            decay = beta

        # Fused shift + EMA update: ema_psd[j] = decay * shift(ema_psd)[j]
        # + (1-decay) * psd_clean[j], single pass, no intermediate alloc.
        _ema_shift_update_njit(
            st.ema_psd, psd_clean, v_bins, decay, st.ema_scratch)
        st.ema_psd, st.ema_scratch = st.ema_scratch, st.ema_psd

        mf_resp = matched_filter_response(st.ema_psd, st.kernels, out=st.mf_out)

        # Runtime noise estimator: median absolute deviation of the
        # matched-filter response. MAD × 1.4826 gives an unbiased σ
        # estimate for Gaussian noise; the estimate is robust to the peak
        # (and the surrounding signal) through the 50% breakdown point of
        # the MAD. Used by the jump-gate Shewhart test.
        mf_median = float(np.median(mf_resp))
        np.subtract(mf_resp, mf_median, out=st.mad_scratch)
        np.abs(st.mad_scratch, out=st.mad_scratch)
        sigma_n = 1.4826 * float(np.median(st.mad_scratch))

        # Readout: local-window argmax + jump-to-global fallback.
        # Jump-gate: Shewhart absolute test — accept the global peak only
        # if it exceeds the local peak by ≥ 2·σ_n (2σ Gaussian rule).
        # Anchored to the runtime MAD noise estimate; no dependence on the
        # spectrum length L.
        i_global = int(np.argmax(mf_resp))
        i_prev = max(0, min(N - 1, round(st.q_prev / bin_step)))
        lo = max(0, i_prev - self._half_win)
        hi = min(N, i_prev + self._half_win + 1)
        local = mf_resp[lo:hi]
        fit_idx = lo + int(np.argmax(local))
        if (i_global < lo or i_global >= hi) and \
                mf_resp[i_global] > local.max() + 2.0 * sigma_n:
            fit_idx = i_global
        # Detection via the matched filter (fit_idx, robust at low SNR); sub-bin via a
        # WIDE MAD-gated centroid on the pooled raw PSD (sharp, not MF-broadened) to
        # recover the within-frame-smeared peak's mean while the EMA pool denoises at
        # low SNR.
        mu = self._centroid_subbin(st.ema_psd, fit_idx)

        # tune_unit is np.linspace(0, 0.5, N, endpoint=False), so
        # tune_unit[i] = i * bin_step exactly. The general linear-interp
        # collapses to a single multiply on the equispaced grid; the
        # min(q, 1-q) reflection is a provable no-op since
        # mu_c <= N-1 ⇒ q_raw = mu_c·bin_step < 0.5.
        q_raw = float(np.clip(mu, 0.0, N - 1) * bin_step)

        st.q_prev_prev, st.q_prev = st.q_prev, q_raw
        st.frame_idx += 1

        self._record(q_pred=q_raw, failed=False)
        return MeasurementResult(q=q_raw, failed=False, source="tmf")

    def _record(self, *, q_pred: float, failed: bool) -> None:
        """Append one frame's q estimate + fail flag to history."""
        h = self.history
        h.q_predicted.append(q_pred)
        h.failed_to_detect.append(failed)
