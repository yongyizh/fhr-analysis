"""Validate a fiber HR against the NST ground truth.

The fiber signal can come from any of model_runner.py's modes -- FUNet or TSLNet beat
activity, a NeoSSNet heart waveform, or a plain band-pass -- so nothing here is named for
a particular model: titles and legends are built from run_meta.json.

Reads fiber_activity.npy (the model output) + nst.npy (from
process_fiber_5channel_funet_pt_validate.py) and computes HR three ways, each a
2-curve plot (fiber vs NST) with Pearson r annotated and the model in the title:

  1. IBI          — beats -> 60/IBI (MA, outlier-rejected)
  2. counting     — beats -> sliding COUNT_WIN / COUNT_STEP event-rate HR
  3. autocorr     — envelope autocorrelation HR, AC_WIN / AC_STEP

Both beat trains are produced by model_runner.py and read from funet_beats.npy /
nst_beats.npy -- this script detects nothing. The autocorrelation method is the exception
that still touches waveforms: it runs on the model output directly (already an envelope)
and on the RMS envelope of the band-passed mic in nst.npy.

The NST clock drifts wherever the recording dropped samples. model_runner.py owns that
correction and records it in run_meta.json; the beats arrive already corrected. The shift
is rebuilt here for one purpose only -- remapping the autocorrelation, which reads the
nominal-clock waveform rather than the beats. There is no switch for it here: the upstream
run decided, and disagreeing would put the three HR methods on two different clocks.

Output (into OUT_DIR): hr_ibi.png, hr_counting.png, hr_autocorr.png,
bland_altman.png, hr_scatter.png (IBI agreement scatter + line of equality)

HR is computed here from those beat times and nowhere else: inst_hr (60/IBI, clipped to
BPM_RANGE, averaged over IBI_MA beats) and agreement (r/MAE on a shared AGREEMENT_GRID_HZ
grid). At IBI_MA=20 both reproduce src/analyze/main.py's number exactly.
counting and autocorrelation have no main.py counterpart -- they run on
the same beats but are this script's own methods. The old
yz signal_utils.moving_average_v2 / hr_autocorr._autocorr_bpm helpers are inlined below
verbatim.

OUT_DIR must match model_runner.py's.

Run:  poetry run python yz/hr_plotter.py
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import correlate, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))       # local yz helpers

from analyze.constants import FETAL_BPM_RANGE   # noqa: E402
from analyze.drift import correct_drift         # noqa: E402
from analyze.util import moving_average_v2      # noqa: E402

# ---------------------------------------------------------------------------
# SELECT HERE  (OUT_DIR must match process_fiber_5channel_funet_pt_validate.py)
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "out_yz" / "pt12" / "funet" / "v47"
# Beat detection happens entirely in model_runner.py -- both the fiber beats and the NST
# beats arrive here already detected and already drift-corrected. This script only turns
# them into heart rate and plots it.
IBI_MA = 20                      # beats averaged into the IBI HR trace. 20 is the value
                                 # src/analyze/main.py hardcodes -- keep it
                                 # there to reproduce main.py's r, change it to trade noise
                                 # against responsiveness.
BPM_RANGE = FETAL_BPM_RANGE      # the autocorrelation lag-search band, and the fixed band
                                 # the IBI trace used to be clipped to (see IBI_GATE below).
                                 # model_runner.py has its own copy that governs beat
                                 # DETECTION; this one does not.
IBI_GATE = "band"          # how the IBI trace throws out non-physiological 60/IBI values:
                                 #   "continuity"  keep a point only if it is within CONT_TOL of
                                 #                 the mean of the CONT_N previous KEPT points --
                                 #                 no absolute band at all, so a fetus outside
                                 #                 FETAL_BPM_RANGE is still followed
                                 #   "band"        the old fixed BPM_RANGE clip
                                 #   "both"        band first, then continuity
CONT_TOL = 0.20                  # continuity tolerance, as a FRACTION of the running baseline.
                                 # 0.20 rejects a beat-to-beat jump of >20%; a real acceleration
                                 # ramps over many beats, so the baseline follows it and only
                                 # step changes (a missed beat doubles an interval, an inserted
                                 # one halves it) get cut.
CONT_N = 5                       # how many previously KEPT points the baseline averages
CONT_RESYNC = 20                 # after this many consecutive rejections the baseline is reseeded
                                 # from the next CONT_N values. Without it a genuine step change
                                 # (or a detector dropout that lands the trace somewhere new)
                                 # would freeze the baseline and reject the rest of the record.
                                 # 0 disables the reseed.
YLIM = (120, 180)               # HR y-axis
COUNT_WIN, COUNT_STEP = 10.0, 1.0
COUNT_MA = 20
AC_WIN, AC_STEP = 10.0, 1.0
AC_SMOOTH_S = 0.02              # RMS-envelope smoothing for the NST autocorrelation
AGREEMENT_GRID_HZ = 4.0         # both HR traces are resampled onto a grid this dense before
                                # r/MAE. 4 Hz is what main.py uses.


# ---------------------------------------------------------------------------

FIBER_COLOR, NST_COLOR = "tab:red", "tab:blue"


def _source(meta):
    """How the fiber signal was produced, for titles: the model name plus the fibers it
    actually read. model_runner.py records only the consumed fibers in ``fibers`` (a
    single-channel mode like bandpass or neossnet reads one, whatever FIBERS held), with
    the full loaded set kept separately in ``fibers_loaded``."""
    used = meta.get("fibers", [])
    signal = meta.get("signal")
    bits = [meta.get("model", "?")]
    if used:
        bits.append(f"fiber{'s' if len(used) > 1 else ''} {'+'.join(used)}")
    if signal:
        bits.append(str(signal))
    return f"{bits[0]} ({', '.join(bits[1:])})" if len(bits) > 1 else bits[0]


# --- helpers inlined verbatim from the old yz signal_utils / hr_autocorr ----
def moving_average_v2(y, window):
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or y.size == 0:
        return y.copy()
    kernel = np.ones(w)
    counts = np.convolve(np.ones_like(y), kernel, mode="same")  # points per window
    sums = np.convolve(y, kernel, mode="same")
    return sums / counts


def continuity_keep(bpm, tol=None, n=None, resync=None):
    """Boolean mask over ``bpm``: keep a point only if it sits within ``tol`` (a fraction)
    of the mean of the ``n`` previous KEPT points.

    Heart rate is continuous -- it ramps, it does not jump -- so a 60/IBI value far from
    where the trace has just been is a beat-detection error rather than a physiological
    event: a missed beat doubles an interval (halves the bpm), an inserted one halves it.
    That is a statement about the LOCAL trace, so unlike a fixed band it neither passes a
    doubled interval that happens to land inside the band nor rejects a fetus whose real
    rate sits outside it.

    Rejected points are not fed into the baseline -- otherwise the values being compared
    against are dragged by the very outliers being removed. The baseline is seeded with
    the MEDIAN of the first ``n`` values rather than the first value, so one bad beat at
    the start cannot lock the filter onto the wrong rate. After ``resync`` consecutive
    rejections it is reseeded the same way (see CONT_RESYNC).
    """
    tol = CONT_TOL if tol is None else tol
    n = CONT_N if n is None else n
    resync = CONT_RESYNC if resync is None else resync
    bpm = np.asarray(bpm, dtype=float)
    if bpm.size == 0:
        return np.zeros(0, dtype=bool)

    def seed(i):
        w = bpm[i:i + max(1, n)]
        w = w[np.isfinite(w)]
        return deque([float(np.median(w))] if w.size else [], maxlen=n)

    hist = seed(0)
    keep = np.zeros(bpm.size, dtype=bool)
    misses = 0
    for i, v in enumerate(bpm):
        if not hist:                       # everything so far was non-finite
            hist = seed(i)
        base = sum(hist) / len(hist)
        if np.isfinite(v) and abs(v - base) <= tol * base:
            keep[i] = True
            hist.append(v)
            misses = 0
        else:
            misses += 1
            if resync and misses >= resync:
                hist = seed(i)             # the trace has moved somewhere new; follow it
                misses = 0
    return keep


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak location via a 3-point parabola around index ``i``."""
    if 1 <= i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        denom = a - 2.0 * b + c
        if denom != 0.0:
            return i + 0.5 * (a - c) / denom
    return float(i)


def _autocorr_bpm(env_win, fs, bpm_range):
    """HR (bpm) and confidence from one window's envelope via autocorrelation.

    Returns (bpm, confidence). The period is the lag of the tallest
    autocorrelation peak inside the physiological lag band [60/bpm_hi, 60/bpm_lo];
    confidence is that peak's normalised height in [0, 1]. (nan, 0) if none.
    """
    e = np.asarray(env_win, dtype=float)
    e = e - np.mean(e)
    n = e.size
    if n < 4 or np.allclose(e, 0.0):
        return float("nan"), 0.0

    # Unbiased, lag-0-normalised autocorrelation (positive lags only).
    r = correlate(e, e, mode="full", method="fft")[n - 1:]
    counts = np.arange(n, 0, -1)          # samples averaged at each lag: N, N-1, ..., 1
    r = r / counts
    r0 = r[0]
    if r0 <= 0:
        return float("nan"), 0.0
    r = r / r0

    lag_min = max(1, int(np.floor(fs * 60.0 / bpm_range[1])))     # fastest HR -> shortest lag
    lag_max = min(n - 1, int(np.ceil(fs * 60.0 / bpm_range[0])))  # slowest HR -> longest lag
    if lag_max <= lag_min:
        return float("nan"), 0.0

    seg = r[lag_min:lag_max + 1]
    peaks, _ = find_peaks(seg)
    if peaks.size:
        best = peaks[int(np.argmax(seg[peaks]))]     # tallest local peak in the band
    else:
        best = int(np.argmax(seg))                   # fallback: band maximum
    conf = float(seg[best])

    lag = lag_min + _parabolic_peak(seg, best)       # sub-sample lag
    if lag <= 0:
        return float("nan"), 0.0
    bpm = 60.0 * fs / lag
    return float(bpm), max(0.0, conf)


# --- beat extraction -------------------------------------------------------
def inst_hr(beats, band, ma=None, gate=None):
    """Instantaneous HR from beat times: 60/IBI at each beat, outlier-gated, then averaged
    over ``ma`` beats. This is the whole HR computation -- beats come in from
    model_runner.py, heart rate comes out.

    ``gate`` (default IBI_GATE) picks how the outliers go:
      "continuity"  continuity_keep -- within CONT_TOL of the running baseline
      "band"        the fixed ``band`` clip (main.py's rule)
      "both"        band first, then continuity on what survives

    At gate="band", ma=20 it is identical to what src/analyze/main.py plots (main.py
    hardcodes 20 and the band); the other gates deliberately are not.
    """
    ma = IBI_MA if ma is None else ma      # read at call time so IBI_MA stays live
    gate = IBI_GATE if gate is None else gate
    if gate not in ("continuity", "band", "both"):
        raise ValueError(f"gate must be 'continuity', 'band' or 'both', got {gate!r}")
    beats = np.sort(np.asarray(beats, dtype=float))
    if beats.size < 2:
        return np.array([]), np.array([])
    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    t = beats[1:]
    if gate in ("band", "both"):
        keep = (bpm >= band[0]) & (bpm <= band[1])
        bpm, t = bpm[keep], t[keep]
    if gate in ("continuity", "both"):
        keep = continuity_keep(bpm)
        bpm, t = bpm[keep], t[keep]
    # the moving average runs on the SURVIVORS, so a rejected beat is skipped over rather
    # than smeared into its neighbours
    return t, moving_average_v2(bpm, ma)


def agreement(ta, ya, tb, yb, grid_hz=AGREEMENT_GRID_HZ):
    """Pearson r and MAE between two HR traces.

    Each trace is sampled at its own beat times, so they have to be interpolated onto one
    common grid -- spanning only where both exist -- before they can be compared. r says
    whether they move together, MAE how far apart the numbers are; a constant offset scores
    well on the first and badly on the second, so both are returned.
    """
    ta, ya, tb, yb = (np.asarray(v, dtype=float) for v in (ta, ya, tb, yb))
    if ta.size < 2 or tb.size < 2:
        return float("nan"), float("nan")
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if hi <= lo:
        return float("nan"), float("nan")
    grid = np.arange(lo, hi, 1.0 / grid_hz)
    ra, rb = np.interp(grid, ta, ya), np.interp(grid, tb, yb)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(ra, rb)[0, 1]), float(np.mean(np.abs(ra - rb)))


# --- HR methods ------------------------------------------------------------
def counting_hr(beats, centers, win=COUNT_WIN, ma=COUNT_MA):
    b = np.sort(np.asarray(beats, dtype=float))
    half = win / 2.0
    hr = np.array([np.sum((b >= c - half) & (b <= c + half)) / win * 60.0 for c in centers])
    return moving_average_v2(hr, ma) if ma else hr


def _rms_envelope(x, fs, smooth_s=AC_SMOOTH_S):
    n = max(1, int(round(fs * smooth_s)))
    return np.sqrt(np.maximum(np.convolve(x * x, np.ones(n) / n, mode="same"), 0.0))


def autocorr_hr(env, t, fs, centers, win=AC_WIN, bpm_range=BPM_RANGE):
    """Autocorrelation HR on an ALREADY-envelope signal `env` (the model output, or
    an RMS envelope for the raw mic)."""
    hr = np.full(centers.size, np.nan)
    half = win / 2.0
    for i, c in enumerate(centers):
        s = int(round((c - half - t[0]) * fs))
        e = int(round((c + half - t[0]) * fs))
        if 0 <= s and e <= env.size and e - s >= 4:
            hr[i], _ = _autocorr_bpm(env[s:e], fs, bpm_range)
    return hr


# --- correlation / plotting ------------------------------------------------
def interp_to_grid(t, y, grid, gap=3.0):
    if t.size < 2:
        return np.full(grid.shape, np.nan)
    yi = np.interp(grid, t, y)
    yi[(grid < t[0]) | (grid > t[-1])] = np.nan
    idx = np.clip(np.searchsorted(t, grid), 1, t.size - 1)
    yi[(t[idx] - t[idx - 1]) > gap] = np.nan
    return yi


def pearson(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def median(y):
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    return float(np.median(y)) if y.size else float("nan")


def break_gaps(t, y, gap=1.5):
    if t.size == 0:
        return t, y
    tt, yy = [t[0]], [y[0]]
    for i in range(1, t.size):
        if t[i] - t[i - 1] > gap:
            tt.append(0.5 * (t[i] + t[i - 1])); yy.append(np.nan)
        tt.append(t[i]); yy.append(y[i])
    return np.asarray(tt), np.asarray(yy)


def bland_altman(pairs, meta, out_png):
    """Bland-Altman agreement of fiber vs NST HR, one panel per HR method.

    `pairs` maps method name -> (fiber_hr, nst_hr), both sampled on the SAME common
    time grid (NaNs where either is missing). Each panel plots, per matched time
    point, the mean of the two HRs (x) against their difference fiber-NST (y), with
    the bias (mean difference) and the 95% limits of agreement (bias +/- 1.96 SD).
    """
    items = list(pairs.items())
    fig, axes = plt.subplots(1, len(items), figsize=(6 * len(items), 4.5),
                             squeeze=False, constrained_layout=True)
    stats = {}
    for ax, (name, (hf, hn)) in zip(axes[0], items):
        hf, hn = np.asarray(hf, float), np.asarray(hn, float)
        m = ~np.isnan(hf) & ~np.isnan(hn)
        f, n = hf[m], hn[m]
        if f.size < 3:
            ax.set_title(f"{name}: too few paired points"); stats[name] = (np.nan, np.nan, int(f.size)); continue
        mean = 0.5 * (f + n)
        diff = f - n                                   # fiber - NST
        bias = float(diff.mean()); sd = float(diff.std(ddof=1))
        hi, lo = bias + 1.96 * sd, bias - 1.96 * sd
        ax.scatter(mean, diff, s=9, alpha=0.45, color="tab:purple", edgecolors="none")
        ax.axhline(bias, color="k", lw=1.3)
        ax.axhline(hi, color="tab:red", ls="--", lw=1.0)
        ax.axhline(lo, color="tab:red", ls="--", lw=1.0)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        xr = ax.get_xlim()[1]
        for y, lab in [(bias, f"bias {bias:+.1f}"), (hi, f"+1.96SD {hi:+.1f}"), (lo, f"-1.96SD {lo:+.1f}")]:
            ax.text(xr, y, " " + lab, va="center", ha="left", fontsize=8,
                    color="k" if lab.startswith("bias") else "tab:red")
        ax.set_title(f"{name}  (n={f.size}, bias {bias:+.1f}, SD {sd:.1f} bpm)", fontsize=10)
        ax.set_xlabel("mean of fiber & NST HR (bpm)"); ax.set_ylabel("fiber − NST HR (bpm)")
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        stats[name] = (bias, sd, int(f.size))
    corrected = " (NST drift-corrected)" if meta.get("nst_drift_correction") else ""
    fig.suptitle(f"{meta.get('patient','')} — {_source(meta)} vs NST — "
                 f"Bland–Altman{corrected}", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    for name, (bias, sd, n) in stats.items():
        print(f"  BA {name}: bias {bias:+.1f} bpm, SD {sd:.1f}, LoA [{bias-1.96*sd:+.1f}, {bias+1.96*sd:+.1f}], n={n}")
    print(f"  -> {out_png.name}")
    return stats


def hr_scatter(gf, gn, meta, out_png):
    """Agreement scatter of fiber vs NST HR (IBI): NST (ground truth) on x, fiber on y,
    open circles, with the red line of equality (y = x). Uses the SAME paired grid
    points as the IBI Bland-Altman panel."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6), constrained_layout=True)
    hf, hn = np.asarray(gf, float), np.asarray(gn, float)
    m = ~np.isnan(hf) & ~np.isnan(hn)
    f, n = hf[m], hn[m]                                    # f = fiber, n = NST
    if f.size < 3:
        ax.set_title(f"IBI scatter: too few paired points (n={f.size})")
        fig.savefig(out_png, dpi=150); plt.close(fig)
        print(f"  scatter IBI: too few paired points (n={f.size})  ->  {out_png.name}")
        return
    r, _ = pearson(hf, hn)
    lo, hi = min(f.min(), n.min()), max(f.max(), n.max())
    pad = 0.03 * (hi - lo) + 1.0
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "-", color="red", lw=1.0, zorder=1)      # line of equality
    ax.scatter(n, f, s=26, facecolors="none", edgecolors="black", linewidths=0.8, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
    ax.set_xlabel("beat-by-beat FHR NST — IBI [BPM]")
    ax.set_ylabel(f"beat-by-beat FHR {meta['model']} — IBI [BPM]")
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    ax.set_title(f"{meta.get('patient','')} — {_source(meta)} vs NST — "
                 f"IBI agreement scatter\n(n={f.size}, Pearson r = {r:+.3f})", fontsize=9)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  scatter IBI: Pearson r = {r:+.3f} (n={f.size})  ->  {out_png.name}")


def plot_pair(method, meta, window, tf, hf, tn, hn, r, npts, out_png, irregular):
    fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    fplot = break_gaps(tf, hf) if irregular else (tf, hf)
    nplot = break_gaps(tn, hn) if irregular else (tn, hn)
    ax.plot(*nplot, "-", color=NST_COLOR, lw=1.6, label=f"NST (ground truth) — median {median(hn):.1f} bpm")
    ax.plot(*fplot, "-", color=FIBER_COLOR, lw=1.1, alpha=0.9,
            label=f"{meta['model']} — median {median(hf):.1f} bpm")
    ax.set_xlim(*window); ax.set_ylim(*YLIM)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Fetal HR (bpm)")
    patient = meta.get("patient", "")
    corrected = "   |   NST drift-corrected" if meta.get("nst_drift_correction") else ""
    ax.set_title(f"{patient} — {_source(meta)} vs NST — {method} HR   |   "
                 f"Pearson r = {r:+.3f}{corrected}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  {method}: Pearson r = {r:+.3f} (n={npts})  ->  {out_png.name}")


def main():
    meta = json.loads((OUT_DIR / "run_meta.json").read_text()) if (OUT_DIR / "run_meta.json").exists() \
        else {"model": "?", "fibers": [], "window": [None, None]}
    window = tuple(meta["window"]) if meta.get("window") and None not in meta["window"] else None

    fa = np.load(OUT_DIR / "fiber_activity.npy")
    na = np.load(OUT_DIR / "nst.npy")
    tf, act = fa[:, 0], fa[:, 1]
    tn, xn = na[:, 0], na[:, 1]
    fs_f = int(round(1.0 / np.median(np.diff(tf))))
    fs_n = int(round(1.0 / np.median(np.diff(tn))))
    if window is None:
        window = (max(tf[0], tn[0]), min(tf[-1], tn[-1]))
    print(f"{_source(meta)} @ {fs_f} Hz vs NST @ {fs_n} Hz, window {window}")

    # Beats come from the process script when it wrote them: they are main.py-equivalent
    # (v2 on the activity, v7 on load_sot's 44.1 kHz double-band-passed mic) and already
    # drift-corrected, so re-detecting from the 2 kHz nst.npy here would silently swap in a
    # different NST signal and undo the match. Fall back to local detection only if absent.
    bf_file, bn_file = OUT_DIR / "funet_beats.npy", OUT_DIR / "nst_beats.npy"
    if not (bf_file.exists() and bn_file.exists()):
        raise FileNotFoundError(
            f"{bf_file.name} / {bn_file.name} missing from {OUT_DIR}. Beat detection lives in "
            f"model_runner.py -- run it first. (There used to be a fallback that re-detected "
            f"here from the 2 kHz nst.npy; it silently produced a different NST beat train "
            f"than the 44.1 kHz one main.py uses, so it was removed rather than kept as a trap.)")
    beats_f, beats_n = np.load(bf_file), np.load(bn_file)
    print(f"  beats: from model_runner.py (main.py-equivalent"
          f"{', drift-corrected' if meta.get('nst_drift_correction') else ''})")
    print(f"  beats: fiber={beats_f.size}, NST={beats_n.size}")

    # --- NST clock drift ----------------------------------------------------
    # model_runner.py owns this decision: it applies the dropout-log correction to the
    # beats and records what it did in run_meta.json. Nothing is re-decided here, so the
    # beats used by the IBI and counting methods are already on the corrected clock.
    #
    # The one loose end is autocorrelation, which reads the WAVEFORM (nst.npy) rather than
    # the beats, and nst.npy is deliberately left on the nominal clock. Its time axis
    # cannot simply be shifted -- autocorr_hr indexes the envelope as (centre - t[0]) * fs,
    # so a non-uniform axis is silently ignored (t[0] sits before the first dropout and
    # never moves). Its RESULT is remapped onto the corrected clock further down instead.
    drift_applied = bool(meta.get("nst_drift_correction"))
    drift_log = meta.get("nst_drift_log")
    nst_drift_shift = None
    if drift_applied:
        if not drift_log or not Path(drift_log).exists():
            raise FileNotFoundError(
                f"run_meta.json says the NST was drift-corrected, but its dropout log "
                f"({drift_log!r}) is missing. The beats are corrected and the "
                f"autocorrelation cannot be put on the same clock without it.")
        nst_drift_shift = lambda g, _l=Path(drift_log): (            # noqa: E731
            correct_drift(np.asarray(g, dtype=float), _l) - np.asarray(g, dtype=float))

    grid = np.arange(window[0], window[1] + 1e-9, 1.0)

    ba_pairs = {}    # method -> (fiber_hr, nst_hr) on the common grid, for Bland-Altman

    # 1) IBI — main.py's trace (MA IBI_MA, clipped to BPM_RANGE) and agreement (4 Hz
    #    grid), so this panel reproduces run_funet_pipeline's number exactly. interp_to_grid
    #    still puts both traces on the shared 1 s grid for Bland-Altman, which needs paired
    #    samples for Bland-Altman, which needs paired values.
    tfi, hfi = inst_hr(beats_f, BPM_RANGE)
    tni, hni = inst_hr(beats_n, BPM_RANGE)
    gfi, gni = interp_to_grid(tfi, hfi, grid), interp_to_grid(tni, hni, grid)
    r, _mae = agreement(tfi, hfi, tni, hni)
    n = int(np.sum(~np.isnan(gfi) & ~np.isnan(gni)))   # paired 1 s samples, as for Bland-Altman
    plot_pair("IBI", meta, window, tfi, hfi, tni, hni, r, n, OUT_DIR / "hr_ibi.png", irregular=True)
    ba_pairs["IBI"] = (gfi, gni)

    # 2) counting
    hfc = counting_hr(beats_f, grid)
    hnc = counting_hr(beats_n, grid)
    r, n = pearson(hfc, hnc)
    plot_pair("counting", meta, window, grid, hfc, grid, hnc, r, n, OUT_DIR / "hr_counting.png", irregular=False)
    ba_pairs["counting"] = (hfc, hnc)

    # 3) autocorrelation (model output directly; NST via RMS envelope)
    hfa = autocorr_hr(np.asarray(act, float), tf, fs_f, grid)
    hna = autocorr_hr(_rms_envelope(xn, fs_n), tn, fs_n, grid)
    # remap NST HR onto the corrected (fiber) clock -- autocorr_hr ran on the nominal axis
    if nst_drift_shift is not None:
        hna = np.interp(grid, grid + nst_drift_shift(grid), hna, left=np.nan, right=np.nan)
    r, n = pearson(hfa, hna)
    plot_pair("autocorrelation", meta, window, grid, hfa, grid, hna, r, n, OUT_DIR / "hr_autocorr.png", irregular=False)
    ba_pairs["autocorrelation"] = (hfa, hna)

    # Bland-Altman agreement (fiber vs NST) for all three HR methods.
    bland_altman(ba_pairs, meta, OUT_DIR / "bland_altman.png")

    # Agreement scatter (line of equality) for the IBI method — same data as the IBI B-A panel.
    hr_scatter(gfi, gni, meta, OUT_DIR / "hr_scatter.png")


if __name__ == "__main__":
    main()
