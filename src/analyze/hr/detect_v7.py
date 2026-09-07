"""
detect_v7.py — duration-dependent HMM (Springer/Schmidt-style) heart-sound
segmenter for fetal acoustic (SOT/mic) signals.

Why this exists
---------------
``detect_v2``/``v5`` pick the tallest envelope lobe and flip between S1 and S2.
``detect_v6`` period-locks (one sound per cycle) but tracks greedily and cannot
reliably decide *which* sound is S1, so on already-clean windows it adds jitter.

This detector instead models the whole cardiac cycle as a 4-state hidden
semi-Markov model and decodes the single globally-most-likely state sequence:

    S1  ->  systole  ->  S2  ->  diastole  ->  S1  -> ...   (fixed cyclic order)

with **explicit Gaussian duration priors** per state (the mean systole/diastole
lengths come from the autocorrelation-estimated heart rate and systolic interval).
This is the Schmidt (2010) / Springer (2016) approach used as the reference for
PCG segmentation. Two properties matter here:

* The fixed S1->systole->S2->diastole ordering plus the duration priors encode the
  fPCG rule that *diastole is longer than systole*. The Viterbi therefore labels
  each acoustic event S1-vs-S2 from the whole sequence, not from local amplitude —
  it cannot flip within a cycle, and it phase-aligns globally.
* Explicit state durations bridge dropped beats (the duration model "coasts"
  through a missing sound) instead of collapsing the rhythm.

Unlike the published Springer model, the emission model here is **unsupervised**
(we have no S1/S2 reference labels): S1/S2 share a "sound-present" likelihood and
systole/diastole a "silence" likelihood, so the S1-vs-S2 disambiguation is carried
entirely by the duration model. An optional mild amplitude asymmetry (``amp_asym``)
nudges S1 onto the louder sound.

Contract matches the other detectors: ``v7_beat_detector(X, bpm_range, out,
energy_range=0.5, tag="", ...)`` -> ``{"peaks", "times"}`` (beats = S1 onsets), a
drop-in for ``v2_beat_detector``.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from analyze.data import Audio
from analyze.hr.detect_v3 import _moving_avg, _suppress_transients
from analyze.hr.detect_v5 import _shannon_energy_envelope

# State indices for the fixed cyclic cardiac cycle.
S1, SYS, S2, DIA = 0, 1, 2, 3
STATE_NAMES = ["S1", "systole", "S2", "diastole"]
NEG = -1e18


# ---------------------------------------------------------------------------
# Features: robustly-normalised envelope on a coarse grid
# ---------------------------------------------------------------------------

def _features(
        X: Audio,
        feat_fs: float,
        *,
        suppress_transients: bool = True,
        transient_k: float = 4.0,
        smooth_s: float = 0.020,
        envelope_method: str = "shannon",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(grid_times, a)`` where ``a`` in [0, 1] is a soundness envelope on
    a ``feat_fs`` grid: high during a heart sound (S1/S2), low during the
    silent systole/diastole gaps. Robustly scaled (10th-95th pct) so the emission
    model is amplitude-invariant across windows."""
    x = np.asarray(X.data, float)
    hz = float(X.hz)
    if suppress_transients:
        x = _suppress_transients(x, hz, k=transient_k)
    # 'precomputed' is for an X that is ALREADY an envelope (e.g. A(t) from
    # analyze.demodulate, or a model's beat activity). Shannon energy is -x^2 log(x^2):
    # built to rectify an OSCILLATING waveform, and non-monotonic -- it peaks at moderate
    # amplitude and pushes both small and large values down. Run over something already
    # enveloped it therefore suppresses the strongest beats, which is the opposite of what
    # the decoder needs. abs() is the identity here, since an envelope is already >= 0.
    if (envelope_method or "shannon").lower() == "precomputed":
        env = np.abs(x)
    else:
        env = _shannon_energy_envelope(x)
    env = _moving_avg(env, max(1, int(round(smooth_s * hz))))

    t0, t1 = float(X.time[0]), float(X.time[-1])
    grid = np.arange(t0, t1, 1.0 / feat_fs)
    if len(grid) < 4:
        return grid, np.zeros_like(grid)
    eg = np.interp(grid, X.time, env)
    lo, hi = np.percentile(eg, [10, 95])
    a = np.clip((eg - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return grid, a


# ---------------------------------------------------------------------------
# Heart-rate + systolic-interval estimation (autocorrelation, Schmidt-style)
# ---------------------------------------------------------------------------

def _hr_systole(a: np.ndarray, fs: float, min_rr_s: float, max_rr_s: float
                ) -> Tuple[float, float]:
    """Estimate ``(RR, systole)`` in samples from the envelope autocorrelation.

    ``RR`` = lag of the dominant autocorrelation peak inside the physiological
    cardiac-period band; ``systole`` = lag of the dominant peak in the
    ``[0.30 RR, 0.45 RR]`` band and **clamped to it**. The fetal systolic time
    interval is a fairly stable ~0.30-0.45 fraction of the cycle, and the envelope
    autocorrelation often lacks a clean systole sub-peak (it then grabs a band
    edge), so bounding to physiology keeps the duration priors sane. Falls back to
    0.38 RR if the band is degenerate.
    """
    n = len(a)
    if n < 8:
        rr = max(1.0, 0.5 * (min_rr_s + max_rr_s) * fs)
        return rr, 0.38 * rr
    a0 = a - float(np.mean(a))
    ac = np.correlate(a0, a0, mode="full")[n - 1:]
    lo, hi = int(round(min_rr_s * fs)), int(round(max_rr_s * fs))
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        rr = 0.5 * (min_rr_s + max_rr_s) * fs
    else:
        rr = lo + int(np.argmax(ac[lo:hi + 1]))
    slo, shi = int(round(0.30 * rr)), int(round(0.45 * rr))
    shi = min(shi, len(ac) - 1)
    if shi <= slo:
        sys = 0.38 * rr
    else:
        sys = slo + int(np.argmax(ac[slo:shi + 1]))
    return float(rr), float(sys)


# ---------------------------------------------------------------------------
# Emission + duration models
# ---------------------------------------------------------------------------

def _log_emission(a: np.ndarray, amp_asym: float = 0.0) -> np.ndarray:
    """Per-sample log-emission for the 4 states, shape ``(T, 4)``.

    S1/S2 favour "sound present" (high ``a``); systole/diastole favour "silence"
    (low ``a``). With ``amp_asym > 0`` the S1 sound state additionally prefers
    louder samples than S2, nudging the (otherwise duration-only) phase onto the
    physiological S1."""
    eps = 1e-3
    sound = a
    silence = 1.0 - a
    p = np.empty((len(a), 4), dtype=float)
    p[:, S1] = sound * (1.0 + amp_asym * a)
    p[:, SYS] = silence
    p[:, S2] = sound * (1.0 + amp_asym * (1.0 - a))
    p[:, DIA] = silence
    return np.log(p + eps)


def _dur_logpdf(mean: float, std: float, fs: float, hard_min_s: float = 0.02
                ) -> Tuple[int, int, np.ndarray]:
    """Gaussian duration log-pdf over an integer sample range ``[dmin, dmax]``."""
    mean = max(mean, hard_min_s * fs)
    std = max(std, 0.012 * fs)
    dmin = max(1, int(round(mean - 2.5 * std)))
    dmax = max(dmin + 1, int(round(mean + 2.5 * std)))
    d = np.arange(dmax + 1, dtype=float)
    lp = -0.5 * ((d - mean) / std) ** 2 - np.log(std)
    lp[:dmin] = NEG
    return dmin, dmax, lp


def _build_durations(rr: float, sys: float, fs: float
                     ) -> List[Tuple[int, int, np.ndarray]]:
    """Duration priors for [S1, systole, S2, diastole] (samples), derived from the
    estimated RR and systolic interval. Systole = S1->S2 interval minus the S1
    sound; diastole = the remaining (longer) gap."""
    m_s1, m_s2 = 0.09 * fs, 0.07 * fs
    m_sys = max(0.03 * fs, sys - m_s1)              # gap between S1 and S2 onsets
    m_dia = max(0.05 * fs, rr - sys - m_s2)          # long gap to next S1
    return [
        _dur_logpdf(m_s1, 0.020 * fs, fs),
        _dur_logpdf(m_sys, 0.030 * fs, fs),
        _dur_logpdf(m_s2, 0.020 * fs, fs),
        _dur_logpdf(m_dia, 0.045 * fs, fs),
    ]


# ---------------------------------------------------------------------------
# Explicit-duration (semi-Markov) Viterbi over the fixed cyclic cycle
# ---------------------------------------------------------------------------

def _viterbi_hsmm(log_e: np.ndarray,
                  durations: List[Tuple[int, int, np.ndarray]]
                  ) -> np.ndarray:
    """Decode the most-likely S1->systole->S2->diastole tiling of ``[0, T)``.

    Segments must tile the window exactly and follow the fixed cyclic order
    ``i -> (i+1) % 4``; each segment pays its Gaussian duration log-pdf plus the
    summed log-emission over its span (O(1) via a cumulative sum). The first
    segment may start in any state (window starts mid-cycle). Returns a per-sample
    state label array of length ``T``.
    """
    T, Sn = log_e.shape
    csum = np.vstack([np.zeros(Sn), np.cumsum(log_e, axis=0)])  # (T+1, Sn)

    # delta[e, j] = best score of a tiling of [0, e) whose last segment is state j.
    delta = np.full((T + 1, Sn), NEG)
    back_start = np.full((T + 1, Sn), -1, dtype=int)  # segment start index
    back_prev = np.full((T + 1, Sn), -1, dtype=int)   # predecessor state (-1 = window start)

    for j in range(Sn):
        dmin, dmax, lp = durations[j]
        for e in range(dmin, min(dmax, T) + 1):
            sc = lp[e] + (csum[e, j] - csum[0, j])   # segment [0, e), any start state
            if sc > delta[e, j]:
                delta[e, j] = sc
                back_start[e, j] = 0
                back_prev[e, j] = -1

    for e in range(1, T + 1):
        for j in range(Sn):
            i = (j - 1) % Sn                          # only i -> j allowed
            dmin, dmax, lp = durations[j]
            best, bs = delta[e, j], back_start[e, j]
            bp = back_prev[e, j]
            s_lo = max(1, e - dmax)
            s_hi = e - dmin
            for s in range(s_lo, s_hi + 1):
                prev = delta[s, i]
                if prev <= NEG / 2:
                    continue
                sc = prev + lp[e - s] + (csum[e, j] - csum[s, j])
                if sc > best:
                    best, bs, bp = sc, s, i
            if bs != back_start[e, j] or bp != back_prev[e, j] or best != delta[e, j]:
                delta[e, j] = best
                back_start[e, j] = bs
                back_prev[e, j] = bp

    labels = np.zeros(T, dtype=int)
    e = T
    j = int(np.argmax(delta[T]))
    while e > 0 and j >= 0:
        s = back_start[e, j]
        labels[s:e] = j
        j_prev = back_prev[e, j]
        e, j = s, j_prev
    return labels


# ---------------------------------------------------------------------------
# Public: signal-level detector
# ---------------------------------------------------------------------------

def v7_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (90.0, 210.0),
        out=None,
        energy_range: float = 0.5,   # signature-compat (unused)
        tag: str = "",
        *,
        use_floor: bool = True,      # signature-compat (unused)
        feat_fs: float = 100.0,
        suppress_transients: bool = True,
        transient_k: float = 4.0,
        amp_asym: float = 0.0,
        return_debug: bool = False,
        envelope_method: str = "shannon",   # 'shannon' | 'precomputed'; see _features
) -> dict:
    """Segment the cardiac cycle with a duration-dependent HMM; beats are S1 onsets.

    ``bpm_range`` bounds the physiological FHR. The heart rate and systolic
    interval are estimated from the envelope autocorrelation to set the duration
    priors, then the S1/systole/S2/diastole sequence is Viterbi-decoded and each
    S1 segment's onset is a beat. Returns ``{"peaks": indices into X.time,
    "times": beat times}``.
    """
    min_rr_s = 60.0 / float(bpm_range[1])
    max_rr_s = 60.0 / float(bpm_range[0])

    grid, a = _features(X, feat_fs, suppress_transients=suppress_transients,
                        transient_k=transient_k, envelope_method=envelope_method)

    empty = {"peaks": np.array([], dtype=int), "times": np.array([], dtype=float)}

    def _finish(beat_times, labels=None, rr=None, sys=None):
        if out is not None:
            _plot_stages(X, grid, a, labels, beat_times, rr, sys, feat_fs, out, tag)
        if len(beat_times) == 0:
            res = dict(empty)
        else:
            peaks = np.clip(np.searchsorted(X.time, beat_times), 0, len(X.time) - 1)
            res = {"peaks": peaks, "times": np.asarray(beat_times, float)}
        if return_debug:
            res.update({"grid": grid, "a": a, "labels": labels, "rr": rr, "sys": sys})
        return res

    if len(grid) < 8 or float(np.max(a)) <= 0:
        return _finish(np.array([], dtype=float))

    rr, sys = _hr_systole(a, feat_fs, min_rr_s, max_rr_s)
    durations = _build_durations(rr, sys, feat_fs)
    log_e = _log_emission(a, amp_asym=amp_asym)
    labels = _viterbi_hsmm(log_e, durations)

    # S1 onsets = rising edges into state S1.
    is_s1 = (labels == S1).astype(int)
    onsets = np.flatnonzero(np.diff(np.concatenate([[0], is_s1])) == 1)
    beat_times = grid[onsets] if len(onsets) else np.array([], dtype=float)
    return _finish(beat_times, labels=labels, rr=rr, sys=sys)


def v7_envelope_aware_beat_detector(
        X: Audio,
        bpm_range: Tuple[float, float] = (90.0, 210.0),
        out=None,
        energy_range: float = 0.5,   # signature-compat (unused)
        tag: str = "",
        **kwargs,
) -> dict:
    """``v7_beat_detector`` for an ``X`` that is *already* an envelope.

    Same transient suppression, smoothing, normalisation, priors and HSMM decode; only the
    Shannon-energy stage is skipped. It exists as its own function because ``hr.fiber_beats``
    and the ``beat_app`` detector registry both address detectors by name with a fixed
    signature, leaving nowhere to pass a keyword through. The stock default is untouched.

    Use it on the demodulated amplitude A(t) (analyze.demodulate) or on a model's beat
    activity -- anything non-negative and already smooth.
    """
    kwargs.pop("envelope_method", None)   # this wrapper's whole purpose; not overridable
    return v7_beat_detector(X, bpm_range, out, energy_range, tag,
                            envelope_method="precomputed", **kwargs)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_STATE_COLORS = {S1: "tab:green", SYS: "0.85", S2: "tab:orange", DIA: "0.95"}


def _plot_stages(X, grid, a, labels, beat_times, rr, sys, feat_fs, out, tag):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)

    axes[0].plot(grid, a, lw=0.7, color="tab:purple")
    if labels is not None and len(labels) == len(grid):
        for st in (S1, S2):  # shade the two sound states
            mask = labels == st
            axes[0].fill_between(grid, 0, 1, where=mask, color=_STATE_COLORS[st],
                                 alpha=0.35, step="mid",
                                 label=STATE_NAMES[st])
    axes[0].legend(loc="upper right", fontsize=8)
    rr_bpm = 60.0 * feat_fs / rr if rr else float("nan")
    axes[0].set_title(f"{tag} HSMM states on envelope  |  RR~{rr_bpm:.0f} bpm  "
                      f"systole~{1000*sys/feat_fs:.0f} ms")

    axes[1].plot(X.time, X.data, lw=0.5, color="tab:blue", rasterized=True)
    if len(beat_times):
        axes[1].vlines(beat_times, float(np.min(X.data)), float(np.max(X.data)),
                       linestyles="--", color="tab:red", lw=0.9)
    axes[1].set_title(f"S1 beats on waveform (n={len(beat_times)})")
    axes[1].set_xlabel("Time (s)")

    fig.savefig(out / f"detect_v7_{tag}.png", dpi=150)
    plt.close(fig)
