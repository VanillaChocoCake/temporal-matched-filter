# temporal-matched-filter

Core implementation of the classical betatron-tune estimator described in

> P. Sun, M. Zhang, R. Yuan, D. Li and J. Dong, *Robust betatron-tune measurement from Schottky spectra: complementary classical and deep-learning paradigms*, JINST **21** (2026) P08005, [doi:10.1088/1748-0221/21/08/P08005](https://doi.org/10.1088/1748-0221/21/08/P08005)

(section "Classical estimator: motion-compensated PSD-domain temporal context"). The deep-learning estimator of the same paper is in [fftconv-bayes-tracker](https://github.com/VanillaChocoCake/fftconv-bayes-tracker).

## Method

The temporal matched-filter (TMF) estimator reads the betatron tune from Schottky spectra folded onto the tune axis q ∈ [0, 0.5). It has no learnable parameters and adds at most one frame of lag. For every frame it runs four stages:

1. **Postprocess.** Low-tune mask followed by Savitzky-Golay smoothing, with the window fixed in tune units.
2. **Motion-compensated EMA.** The pooled PSD is shifted by the estimated tune velocity before each exponential update, and the pooling depth shortens as the tune moves faster.
3. **Matched-filter bank.** Four Gaussian kernels with widths fixed in tune units, aggregated by the mean of the upper half of the responses.
4. **Readout.** Local-window argmax with a Shewhart-gated jump to the global peak, then a wide MAD-gated centroid that gives the sub-bin tune.

All windows, kernel widths and velocity limits are specified in tune units and converted to bins on the operating grid, so one configuration applies to any spectrum length L.

## Files

| File | Content |
|---|---|
| `tmf_estimator.py` | `TMFEstimator` (`process(frame) -> MeasurementResult`), `TMFConfig`, and the fused shift-and-update EMA kernel |
| `tune_pipeline/dsp/matched_filter.py` | Gaussian kernel bank and the upper-half-mean matched-filter response |
| `tune_pipeline/dsp/savgol.py` | Savitzky-Golay filter with cached coefficients, equivalent to SciPy `mode="interp"` |
| `tune_pipeline/frames.py` | `FrontendFrame` and `MeasurementResult` data contracts |

## Scope

This repository contains the estimator only. The spectral preprocessing that produces a `FrontendFrame` (revolution-frequency estimation, folding and soft-binning of the PSD onto the tune grid; paper section "Upstream preprocessing pipeline"), the benchmarks and the datasets are not included.

## Dependencies

NumPy, SciPy, Numba.

## Citation

```bibtex
@article{sun2026robust,
  title     = {Robust betatron-tune measurement from Schottky spectra: complementary classical and deep-learning paradigms},
  author    = {Sun, Peihan and Zhang, Manzhou and Yuan, Renxian and Li, Deming and Dong, Jian},
  journal   = {Journal of Instrumentation},
  volume    = {21},
  number    = {08},
  pages     = {P08005},
  year      = {2026},
  publisher = {IOP Publishing}
}
```

## License

MIT, see [LICENSE](LICENSE).
