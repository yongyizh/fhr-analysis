"""
detect_v9.py — ``detect_v4`` with a *local* (per-block) peak-height floor.

Relationship to the other detectors
------------------------------------
- ``detect_v2`` : Hilbert envelope + per-window additive MAD floor, point peaks.
- ``detect_v3`` : AGC-normalised envelope + autocorr period prior + Viterbi DP.
- ``detect_v4`` : region/FWHM detection with a **global** percentile height floor.
- ``detect_v9`` : (this) ``detect_v4`` unchanged except that the height floor is
  measured per ``floor_window_s`` block instead of once over the whole signal.

Why the floor moved
-------------------
``detect_v4`` gates peaks at ``percentile(env_smooth, peak_height_pct)`` taken over the
entire recording. That single number is set by the loudest stretch of the signal, so on a
recording whose level drifts — probe reseat, position change, a noisy passage — the floor
sits above the envelope of every quiet stretch and those stretches lose *all* their beats,
while the loud stretch keeps even its noise. The failure is silent: the detector simply
returns nothing there.

Measuring the percentile inside a window instead gates each beat against its own stretch of
signal. The blocks are non-overlapping, so the floor is a step function that changes only at
block edges; the trade-off is that the gate jumps mid-signal, so two beats either side of an
edge are judged against different levels. A 4 s block is ~10 fetal beats: long enough for the
percentile to be a stable estimate of the local baseline, short enough to track drift.

Everything else — analytic envelope, Gaussian smoothing, FWHM region growth, merge, duration
gate, region midpoints, amplitude-aware IBI rejection — is ``detect_v4`` verbatim.

It is SELF-CONTAINED and detection-only, like the other detectors in this package.

Return contract matches ``detect_v2``/``detect_v3``/``detect_v4``: ``v9_beat_detector`` takes
``(X: Audio, bpm_range, out, tag)`` and returns ``{peaks, times}``. Use it by passing it to
``hr.fiber_beats`` / ``hr.sot_beats`` in the pipeline, e.g. ``fiber_beats(v9_beat_detector,
out)``.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks, hilbert

from analyze.data import Audio

# Block length the height floor is measured over. See the module docstring.
DEFAULT_FLOOR_WINDOW_S = 4.0

# Multiplier on the block percentile. The percentile alone is an order statistic, so it
# cannot go below the block's minimum however small ``peak_height_pct`` gets -- scaling is
# the only way further down. Below the point where the floor passes under the block minimum
# the height gate stops rejecting anything, and 0.0 disables it outright.
DEFAULT_FLOOR_SCALE = 1.0


# ---------------------------------------------------------------------------
# Envelope (cfg.envelope_method: 'analytic' | 'rms' | 'peak')
# ---------------------------------------------------------------------------

def _envelope(sig: np.ndarray, fs: float, method: str,
              rms_win_s: float, peak_dist_s: float) -> np.ndarray:
    """Raw detection envelope. Port of the ``switch cfg.envelope_method`` block.

    'analytic' = ``abs(hilbert(sig))`` (default), 'rms' = sliding-window RMS,
    'peak' = upper envelope interpolated over local maxima.
    """
    sig = np.asarray(sig, float)
    m = (method or "analytic").lower()
    # 'precomputed': the input already IS an envelope (A(t) from analyze.demodulate, or a
    # model's beat activity), so rectifying it again only distorts it. See detect_v7.
    if m == "precomputed":
        return np.abs(sig)
    if m == "rms":
        n = max(2, int(round(rms_win_s * fs)))
        return np.sqrt(np.maximum(uniform_filter1d(sig ** 2, size=n, mode="reflect"), 0.0))
    if m == "peak":
        dist = max(1, int(round(peak_dist_s * fs)))
        idx, _ = find_peaks(sig, distance=dist)
        if len(idx) < 2:
            return np.abs(sig)
        return np.interp(np.arange(len(sig)), idx, sig[idx])
    # 'analytic' (default)
    return np.abs(hilbert(sig))


# ---------------------------------------------------------------------------
# Local height floor (the only change from detect_v4)
# ---------------------------------------------------------------------------

def _block_percentile(x: np.ndarray, fs: float, pct: float, win_s: float) -> np.ndarray:
    """Percentile of ``x`` within each non-overlapping ``win_s`` block, held flat across the
    block, as a per-sample array the same length as ``x``.

    ``scipy.signal.find_peaks`` accepts an array ``height``, applied element-wise, so this
    drops straight in where ``detect_v4`` passes a scalar.

    A block shorter than half a window at the tail is folded into the previous block rather
    than taking its percentile from a handful of samples. A signal shorter than one block
    degenerates to the global floor, i.e. exactly ``detect_v4``.
    """
    n = len(x)
    win = max(1, int(round(win_s * fs)))
    edges = list(range(0, n, win))
    if len(edges) > 1 and n - edges[-1] < win // 2:
        edges.pop()

    out = np.empty(n, dtype=float)
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else n
        out[lo:hi] = np.percentile(x[lo:hi], pct)
    return out


# ---------------------------------------------------------------------------
# Beat-REGION detection (port of detect_beats, MATLAB lines 666-723)
# ---------------------------------------------------------------------------

def _detect_beat_regions(
        t: np.ndarray,
        sig: np.ndarray,
        bpm_max: float,
        *,
        envelope_method: str = "analytic",
        env_rms_win_s: float = 0.020,
        env_peak_dist_s: float = 0.005,
        envelope_sigma_s: float = 0.020,
        gauss_truncate: float = 4.0,
        peak_height_pct: float = 65.0,
        floor_window_s: float = DEFAULT_FLOOR_WINDOW_S,
        floor_scale: float = DEFAULT_FLOOR_SCALE,
        half_max_limit_s: float = 0.120,
        min_beat_dur_s: float = 0.040,
        max_beat_dur_s: float = 0.300,
        merge_gap_s: float = 0.040,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect beat regions ``[start, end]`` (seconds), and return the smoothed envelope
    used for downstream amplitude lookups plus the per-sample height floor.

    Steps: envelope -> Gaussian smooth -> peak pick (min separation from ``bpm_max``,
    height = ``peak_height_pct`` percentile *of the surrounding ``floor_window_s`` block*)
    -> expand each peak to its half-maximum extent (capped at ``half_max_limit_s``) ->
    merge regions within ``merge_gap_s`` -> keep regions whose duration is in
    ``[min_beat_dur_s, max_beat_dur_s]``.
    """
    t = np.asarray(t, float)
    sig = np.asarray(sig, float)
    fs = 1.0 / float(np.median(np.diff(t)))

    env = _envelope(sig, fs, envelope_method, env_rms_win_s, env_peak_dist_s)
    sigma_samp = envelope_sigma_s * fs
    env_smooth = (gaussian_filter1d(env, sigma=sigma_samp, truncate=gauss_truncate,
                                    mode="reflect") if sigma_samp > 0 else env)

    n = len(env_smooth)
    if n < 2:
        return np.zeros((0, 2), float), env_smooth, np.zeros(n, float)

    min_sep = max(1, int(np.floor(fs * 60.0 / bpm_max)))
    min_sep = min(min_sep, n - 1)
    # v4 used one percentile over the whole signal here; v9 measures it per block so the
    # gate follows the local level (numpy 'linear' interpolation either way), then scales it
    # so the floor can be pushed below the block minimum, which no percentile can reach.
    height_th = floor_scale * _block_percentile(env_smooth, fs, peak_height_pct, floor_window_s)

    pk_idx, _ = find_peaks(env_smooth, distance=min_sep, height=height_th)
    if len(pk_idx) == 0:
        return np.zeros((0, 2), float), env_smooth, height_th

    half_samp = int(np.floor(half_max_limit_s * fs))
    raw = np.zeros((len(pk_idx), 2), float)
    for kk, pidx in enumerate(pk_idx):
        hv = 0.5 * env_smooth[pidx]
        li = pidx
        while li > 0 and env_smooth[li] > hv and (pidx - li) < half_samp:
            li -= 1
        ri = pidx
        while ri < n - 1 and env_smooth[ri] > hv and (ri - pidx) < half_samp:
            ri += 1
        raw[kk] = (t[li], t[ri])

    # sort by start time, then merge regions closer than merge_gap_s
    raw = raw[np.argsort(raw[:, 0])]
    merged = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    merged = np.asarray(merged, float) if merged else np.zeros((0, 2), float)

    if len(merged) == 0:
        return merged, env_smooth, height_th
    dur = merged[:, 1] - merged[:, 0]
    keep = (dur >= min_beat_dur_s) & (dur <= max_beat_dur_s)
    return merged[keep], env_smooth, height_th


def _to_centers(regions: np.ndarray) -> np.ndarray:
    """Region midpoints. Port of ``to_centers``."""
    if regions is None or len(regions) == 0:
        return np.zeros(0, float)
    return np.mean(regions, axis=1)


# ---------------------------------------------------------------------------
# Amplitude-aware IBI rejection (port of ibi_reject, MATLAB lines 735-769)
# ---------------------------------------------------------------------------

def _ibi_reject(centers: np.ndarray, min_ibi: float,
                t_sig: np.ndarray, env_smooth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively drop the *weaker-amplitude* beat of the closest sub-``min_ibi``
    pair until all inter-beat intervals are >= ``min_ibi`` (ties drop the earlier).
    Returns ``(clean_centers, keep_mask)``.
    """
    centers = np.asarray(centers, float)
    t_sig = np.asarray(t_sig, float)
    env_smooth = np.asarray(env_smooth, float)
    n = len(centers)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return centers[keep], keep

    def amp_at(c: float) -> float:
        return float(env_smooth[int(np.argmin(np.abs(t_sig - c)))])

    while True:
        idx_keep = np.flatnonzero(keep)
        c = centers[idx_keep]
        if len(c) < 2:
            break
        ibis = np.diff(c)
        bad = np.flatnonzero(ibis < min_ibi)
        if len(bad) == 0:
            break
        worst = bad[int(np.argmin(ibis[bad]))]
        i0, i1 = idx_keep[worst], idx_keep[worst + 1]
        if amp_at(centers[i0]) <= amp_at(centers[i1]):
            keep[i0] = False  # tie -> drop the earlier beat
        else:
            keep[i1] = False

    return centers[keep], keep


# ---------------------------------------------------------------------------
# Public: signal-level detector
# ---------------------------------------------------------------------------

def v9_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (90.0, 180.0),
        out=None,
        tag: str = "",
        *,
        envelope_method: str = "analytic",
        env_rms_win_s: float = 0.020,
        env_peak_dist_s: float = 0.005,
        envelope_sigma_s: float = 0.020,
        gauss_truncate: float = 4.0,
        peak_height_pct: float = 20.0,
        floor_window_s: float = DEFAULT_FLOOR_WINDOW_S,
        floor_scale: float = DEFAULT_FLOOR_SCALE,
        half_max_limit_s: float = 0.120,
        min_beat_dur_s: float = 0.040,
        max_beat_dur_s: float = 0.300,
        merge_gap_s: float = 0.040,
        ibi_reject: bool = True,
        return_debug: bool = False,
) -> dict:
    """Detect beats on a band-limited acoustic signal via the region/FWHM method, gating
    peaks against a height floor measured per ``floor_window_s`` block.

    ``X`` is expected to be already band-limited to the cardiac band (e.g. the
    190-210 Hz separated/ANC output, or a maternal-band chest signal). Beat times
    are region midpoints after amplitude-aware IBI rejection. Returns the same
    dict shape as ``detect_v2``/``detect_v3``/``detect_v4``, plus ``regions``/``env``/
    ``floor`` when ``return_debug=True``.

    Setting ``floor_window_s`` longer than the signal reproduces ``detect_v4`` exactly.
    """
    t = np.asarray(X.time, float)
    bpm_max = float(bpm_range[1])

    regions, env_smooth, floor = _detect_beat_regions(
        t, X.data, bpm_max,
        envelope_method=envelope_method,
        env_rms_win_s=env_rms_win_s,
        env_peak_dist_s=env_peak_dist_s,
        envelope_sigma_s=envelope_sigma_s,
        gauss_truncate=gauss_truncate,
        peak_height_pct=peak_height_pct,
        floor_window_s=floor_window_s,
        floor_scale=floor_scale,
        half_max_limit_s=half_max_limit_s,
        min_beat_dur_s=min_beat_dur_s,
        max_beat_dur_s=max_beat_dur_s,
        merge_gap_s=merge_gap_s,
    )

    centers = _to_centers(regions)
    if ibi_reject and len(centers):
        min_ibi = 60.0 / bpm_max
        centers, _ = _ibi_reject(centers, min_ibi, t, env_smooth)

    if len(centers):
        peaks = np.clip(np.searchsorted(t, centers), 0, len(t) - 1)
    else:
        peaks = np.array([], dtype=int)
    result = {"peaks": peaks, "times": centers}
    if return_debug:
        result.update({"regions": regions, "env": env_smooth, "floor": floor})
    return result


def v9_envelope_aware_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (90.0, 180.0),
        out=None,
        tag: str = "",
        **kwargs,
) -> dict:
    """``v9_beat_detector`` for an ``X`` that is *already* an envelope.

    Same smoothing, per-block floor, FWHM regions and IBI rejection; only the envelope
    stage is skipped. It exists as its own function because ``hr.fiber_beats`` and the
    ``beat_app`` detector registry address detectors by name with a fixed signature,
    leaving nowhere to pass a keyword through. The stock default is untouched.
    """
    kwargs.pop("envelope_method", None)
    return v9_beat_detector(X, bpm_range, out, tag, envelope_method="precomputed", **kwargs)
