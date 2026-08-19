"""Validate the 5-fiber FUNet HR against the NST ground truth.

Reads fiber_activity.npy (FUNet beat-activity) + nst.npy (from
process_fiber_5channel_funet_pt_validate.py) and computes HR three ways, each a
2-curve plot (FUNet vs NST) with Pearson r annotated and the model in the title:

  1. IBI          — beats -> 60/IBI (MA, outlier-rejected)
  2. counting     — beats -> sliding COUNT_WIN / COUNT_STEP event-rate HR
  3. autocorr     — envelope autocorrelation HR, AC_WIN / AC_STEP

FUNet beats come from peak-picking the activity envelope; its autocorrelation runs
on the activity directly (already an envelope). NST beats come from the v7 detector;
its autocorrelation runs on the RMS envelope of the band-passed mic.

Output (into OUT_DIR): hr_ibi.png, hr_counting.png, hr_autocorr.png,
bland_altman.png, hr_scatter.png (IBI agreement scatter + line of equality)

Ported to the restructured repo: Audio comes from the installed analyze package,
the NST detector runs through beat_app.detectors.run_detector (old "v7" is now
"v7_beat_detector"; debug scratch output is gone from that API), and the old
yz signal_utils.moving_average_v2 / hr_autocorr._autocorr_bpm helpers are inlined
below verbatim. lag_align is the same module, copied alongside this script.

Run:  poetry run python yz/hr_fiber_5channel_funet_pt_validate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))       # local yz helpers
from lag_align import estimate_drift_lag, make_shift_fn, plot_lag, shift_beats  # noqa: E402

from analyze.data import Audio                  # noqa: E402
from beat_app.detectors import run_detector     # noqa: E402

# ---------------------------------------------------------------------------
# SELECT HERE  (OUT_DIR must match process_fiber_5channel_funet_pt_validate.py)
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "out_yz" / "pt8_funet-v36_whole"
NST_DETECTOR = "v9_beat_detector"  # beat detector for the NST ground truth
BPM_RANGE = (110.0, 200.0)       # fetal HR bounds (beat spacing + autocorr lag band)
YLIM = (120, 190)               # HR y-axis
IBI_MA = 10
IBI_REJECT = (110, 200)          # IBI outlier-rejection range (None to keep all)
COUNT_WIN, COUNT_STEP = 10.0, 1.0
COUNT_MA = None
AC_WIN, AC_STEP = 10.0, 1.0
AC_SMOOTH_S = 0.02              # RMS-envelope smoothing for the NST autocorrelation
FUNET_HEIGHT_K = 0.5            # activity peak height threshold = mean + k*std

# --- NST drift correction (mic drops samples -> accumulating fiber<->NST lag) ---
LAG_APPLY = False               # SWITCH: estimate drift lag and shift NST onto the fiber clock
LAG_SEG_LEN = 8.0              # cross-correlation segment length (s)          [adjustable]
LAG_SEG_STEP = 8.0            # cross-correlation segment step (s)            [adjustable]
LAG_MAX_STEP = 5           # per-segment search half-width around the running lag (s; < beat period)
LAG_INIT_GUESS = None        # None = auto-anchor via beat-RATE xcorr (finds lags of many seconds);
                             #        set a number to force the starting lag instead
LAG_MAX_TOTAL = 30.0         # max TOTAL lag searched (s) — raise if the drift exceeds this
LAG_ANCHOR_HALFWIN = 0.5      # first-segment search half-width around LAG_INIT_GUESS (s)
LAG_PULSE_WIDTH = 0.04        # sigmoid pulse half-width (s)
LAG_PULSE_STEEP = 0.01        # sigmoid edge steepness (s)
LAG_MONOTONIC = True          # enforce lag non-decreasing with time (drift accumulates)
LAG_GAP_TOL = 0.02            # an IBI stretched >this by a lag jump is shown as a GAP, not an HR value
# ---------------------------------------------------------------------------

FIBER_COLOR, NST_COLOR = "tab:red", "tab:blue"


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
def funet_beats(activity, t, fs, bpm_range=BPM_RANGE, height_k=FUNET_HEIGHT_K):
    """Peak-pick the FUNet activity envelope into beat times (repo funet_beats)."""
    sig = np.asarray(activity, dtype=float)
    distance = max(1, int(round(60.0 / bpm_range[1] * fs)))
    height = sig.mean() + height_k * sig.std()
    peaks, _ = find_peaks(sig, distance=distance, height=height)
    return t[peaks]


def detector_beats(x, t, fs, name=NST_DETECTOR):
    audio = Audio(np.asarray(t, dtype=float), fs, np.asarray(x, dtype=float))
    return run_detector(name, audio, BPM_RANGE)


# --- HR methods ------------------------------------------------------------
def hr_ibi(beats, reject=IBI_REJECT, ma=IBI_MA, pair_bad=None):
    """HR = 60/IBI at beat-pair midpoints, smoothed.

    `pair_bad` (from lag_align.shift_beats) marks intervals opened up by a jump in
    the drift correction: those are excluded from the HR/smoothing and an explicit
    NaN break is spliced in at their midpoints, so the curve SHOWS A GAP there
    rather than a fake 60/IBI value across the gap.
    """
    b = np.sort(np.asarray(beats, dtype=float))
    if b.size < 2:
        return np.array([]), np.array([])
    t = 0.5 * (b[:-1] + b[1:])
    hr = 60.0 / (b[1:] - b[:-1])
    keep = np.ones(hr.size, dtype=bool)
    if reject is not None:
        keep &= (hr >= reject[0]) & (hr <= reject[1])
    if pair_bad is not None:
        keep &= ~np.asarray(pair_bad, dtype=bool)
    t_out, hr_out = t[keep], moving_average_v2(hr[keep], ma)
    if pair_bad is not None and np.any(pair_bad):
        t_gap = t[np.asarray(pair_bad, dtype=bool)]        # explicit breaks at the gaps
        t_out = np.concatenate([t_out, t_gap])
        hr_out = np.concatenate([hr_out, np.full(t_gap.size, np.nan)])
        order = np.argsort(t_out)
        t_out, hr_out = t_out[order], hr_out[order]
    return t_out, hr_out


def counting_hr(beats, centers, win=COUNT_WIN, ma=COUNT_MA):
    b = np.sort(np.asarray(beats, dtype=float))
    half = win / 2.0
    hr = np.array([np.sum((b >= c - half) & (b <= c + half)) / win * 60.0 for c in centers])
    return moving_average_v2(hr, ma) if ma else hr


def _rms_envelope(x, fs, smooth_s=AC_SMOOTH_S):
    n = max(1, int(round(fs * smooth_s)))
    return np.sqrt(np.maximum(np.convolve(x * x, np.ones(n) / n, mode="same"), 0.0))


def autocorr_hr(env, t, fs, centers, win=AC_WIN, bpm_range=BPM_RANGE):
    """Autocorrelation HR on an ALREADY-envelope signal `env` (FUNet activity, or
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
    """Bland-Altman agreement of FUNet vs NST HR, one panel per HR method.

    `pairs` maps method name -> (fiber_hr, nst_hr), both sampled on the SAME common
    time grid (NaNs where either is missing). Each panel plots, per matched time
    point, the mean of the two HRs (x) against their difference FUNet-NST (y), with
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
        diff = f - n                                   # FUNet - NST
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
        ax.set_xlabel("mean of FUNet & NST HR (bpm)"); ax.set_ylabel("FUNet − NST HR (bpm)")
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        stats[name] = (bias, sd, int(f.size))
    fibers = "+".join(meta.get("fibers", []))
    corrected = " (NST drift-corrected)" if LAG_APPLY else ""
    fig.suptitle(f"{meta.get('patient','')} FUNet ({meta['model']}, fibers {fibers}) vs NST — "
                 f"Bland–Altman{corrected}", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    for name, (bias, sd, n) in stats.items():
        print(f"  BA {name}: bias {bias:+.1f} bpm, SD {sd:.1f}, LoA [{bias-1.96*sd:+.1f}, {bias+1.96*sd:+.1f}], n={n}")
    print(f"  -> {out_png.name}")
    return stats


def hr_scatter(gf, gn, meta, out_png):
    """Agreement scatter of FUNet vs NST HR (IBI): NST (ground truth) on x, FUNet on y,
    open circles, with the red line of equality (y = x). Uses the SAME paired grid
    points as the IBI Bland-Altman panel."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6), constrained_layout=True)
    hf, hn = np.asarray(gf, float), np.asarray(gn, float)
    m = ~np.isnan(hf) & ~np.isnan(hn)
    f, n = hf[m], hn[m]                                    # f = FUNet, n = NST
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
    ax.set_ylabel(f"beat-by-beat FHR FUNet {meta['model']} — IBI [BPM]")
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    fibers = "+".join(meta.get("fibers", []))
    ax.set_title(f"{meta.get('patient','')} FUNet ({meta['model']}, fibers {fibers}) vs NST — "
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
            label=f"FUNet {meta['model']} — median {median(hf):.1f} bpm")
    ax.set_xlim(*window); ax.set_ylim(*YLIM)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Fetal HR (bpm)")
    fibers = "+".join(meta.get("fibers", []))
    patient = meta.get("patient", "")
    corrected = "   |   NST drift-corrected" if LAG_APPLY else ""
    ax.set_title(f"{patient} FUNet ({meta['model']}, fibers {fibers}) vs NST — {method} HR   |   "
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
    print(f"FUNet {meta['model']} activity @ {fs_f} Hz vs NST @ {fs_n} Hz, window {window}")

    beats_f = funet_beats(act, tf, fs_f)          # FUNet: peak-pick the activity
    beats_n = detector_beats(xn, tn, fs_n)        # NST: v7 detector
    print(f"  beats: FUNet={beats_f.size}, NST={beats_n.size}")

    # --- NST drift correction: estimate the accumulating fiber<->NST lag from the two
    #     beat trains (cross-corr of sigmoid pulse trains) and shift NST onto the fiber
    #     clock. Fiber stays fixed; NST beats + autocorr time axis move by +lag(t). ---
    nst_shift, nst_pair_bad = None, None
    if LAG_APPLY and beats_f.size >= 2 and beats_n.size >= 2:
        seg_c, lag_a, lag_m, valid = estimate_drift_lag(
            beats_f, beats_n, window,
            seg_len_s=LAG_SEG_LEN, seg_step_s=LAG_SEG_STEP, max_lag_step_s=LAG_MAX_STEP,
            pulse_width_s=LAG_PULSE_WIDTH, pulse_steep_s=LAG_PULSE_STEEP,
            monotonic=LAG_MONOTONIC, initial_lag_guess=LAG_INIT_GUESS,
            max_lag_s=LAG_MAX_TOTAL,
            anchor_halfwin_s=LAG_ANCHOR_HALFWIN)
        plot_lag(seg_c, lag_a, lag_m, valid, OUT_DIR / "lag_vs_time.png",
                 title=f"{meta.get('patient','')} fiber↔NST drift lag ({meta['model']})")
        nst_shift = make_shift_fn(seg_c, lag_a)
        # move NST beats onto the fiber clock; intervals broken open by a lag jump are
        # flagged so the IBI curve shows a gap there instead of a fake 60/IBI value
        beats_n, nst_pair_bad = shift_beats(beats_n, nst_shift, LAG_GAP_TOL)
        print(f"  NST drift lag: {lag_a[0]:+.2f}s -> {lag_a[-1]:+.2f}s over the window "
              f"({int(np.sum(nst_pair_bad))} IBI gaps from lag jumps) -> lag_vs_time.png")

    grid = np.arange(window[0], window[1] + 1e-9, 1.0)

    ba_pairs = {}    # method -> (fiber_hr, nst_hr) on the common grid, for Bland-Altman

    # 1) IBI
    tfi, hfi = hr_ibi(beats_f)
    tni, hni = hr_ibi(beats_n, pair_bad=nst_pair_bad)
    gfi, gni = interp_to_grid(tfi, hfi, grid), interp_to_grid(tni, hni, grid)
    r, n = pearson(gfi, gni)
    plot_pair("IBI", meta, window, tfi, hfi, tni, hni, r, n, OUT_DIR / "hr_ibi.png", irregular=True)
    ba_pairs["IBI"] = (gfi, gni)

    # 2) counting
    hfc = counting_hr(beats_f, grid)
    hnc = counting_hr(beats_n, grid)
    r, n = pearson(hfc, hnc)
    plot_pair("counting", meta, window, grid, hfc, grid, hnc, r, n, OUT_DIR / "hr_counting.png", irregular=False)
    ba_pairs["counting"] = (hfc, hnc)

    # 3) autocorrelation (FUNet activity directly; NST via RMS envelope)
    hfa = autocorr_hr(np.asarray(act, float), tf, fs_f, grid)
    hna = autocorr_hr(_rms_envelope(xn, fs_n), tn, fs_n, grid)
    if nst_shift is not None:                      # remap NST HR onto the corrected (fiber) clock
        hna = np.interp(grid, grid + nst_shift(grid), hna, left=np.nan, right=np.nan)
    r, n = pearson(hfa, hna)
    plot_pair("autocorrelation", meta, window, grid, hfa, grid, hna, r, n, OUT_DIR / "hr_autocorr.png", irregular=False)
    ba_pairs["autocorrelation"] = (hfa, hna)

    # Bland-Altman agreement (FUNet vs NST) for all three HR methods.
    bland_altman(ba_pairs, meta, OUT_DIR / "bland_altman.png")

    # Agreement scatter (line of equality) for the IBI method — same data as the IBI B-A panel.
    hr_scatter(gfi, gni, meta, OUT_DIR / "hr_scatter.png")


if __name__ == "__main__":
    main()
