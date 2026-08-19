"""Prepare the 5-fiber FUNet beat-activity + the NST ground truth for HR validation.

FUNet (funet-vN) takes N stacked fibers and outputs a per-sample beat-activity
(fetal-beat probability) envelope. This windows the selected fibers, runs FUNet,
and writes the activity plus the band-passed NST (microphone) to .npy for
hr_fiber_5channel_funet_pt_validate.py.

  fibers -> window -> stack (C, T) -> FUNet -> beat activity
         -> fiber_activity.npy  ([time, activity], fiber rate)
  NST    -> window -> resample -> bandpass NST_BAND (cheby1) -> nst.npy ([time, nst])

It also extracts the beat timing right here -- FUNet activity peak-picked, NST via
the NST_DETECTOR beat detector -- and writes two check figures:

  raw_window.png      every selected fiber (raw) + the bandpassed NST, stacked
  beats_hr_check.html interactive (zoom/pan, x-linked panels): NST envelope +
                      FUNet activity with every beat a vertical line, above both
                      beat trains' 60/IBI BPM (moving-averaged over IBI_MA
                      beats); Pearson r in the title. Zooming in the browser
                      replaces a fixed zoom crop.

Writes run_meta.json (model / fibers / window / band) for the plotting script's titles.

Ported to the restructured repo: FUNet is the installed `funet` package (no more
sys.path into lib/funet/src, and no bare `config`/`data`/`model` module collisions),
checkpoints are resolved through the rtmon.models registry (same
lib/funet/models/<version>/ folders), hop_length moved from cfg.data to cfg.model,
and the model sample rate lives in common.audio.

DATA_DIR is a raw recording folder holding ps4000.npy / ps3000a.npy /
microphone.wav (e.g. ~/Downloads/session-02); the mic clock is assumed to start
at recording t=0 like the fibers'.

Select MODEL_VERSION / FIBERS / WINDOW / NST_BAND / OUT_DIR below, then:
    poetry run python yz/process_fiber_5channel_funet_pt_validate.py
"""
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read as wavread
from scipy.signal import resample_poly, cheby1, find_peaks, sosfiltfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))       # local yz helpers
from interactive_plot import (  # noqa: E402
    save_interactive, save_interactive_beats_hr, save_interactive_multi)

from analyze.data import Audio           # noqa: E402
from beat_app.detectors import run_detector        # noqa: E402
from common.audio import SAMPLE_RATE     # noqa: E402  (FUNet model sample rate, Hz)
from funet.config import load_config     # noqa: E402
from funet.inference import load_funet, run_funet  # noqa: E402
from rtmon.models import find as find_model        # noqa: E402

# ---------------------------------------------------------------------------
# SELECT HERE
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Users/yongyi/Downloads/Fiber_research/Banner patient data/"
                "Banner_test_20250520/patient8-session1")
MODEL_VERSION = "funet-v36"                 # registry name (folder under lib/funet/models/)
INFERENCE_WINDOW_S = None                    # FUNet inference chunk length (s, integer). The spectrogram is
                                             # processed in chunks this long, each normalised on its own, so
                                             # this also governs how much the activity shifts when the window
                                             # START moves. None -> the model's trained crop_len (~7 s for
                                             # v24). Straying far from the trained value can hurt accuracy
                                             # (GroupNorm then sees a different time extent than it trained on).
# FIBERS = ["1B", "2A", "2B", "2C", "2D"]      # fibers to stack, in TRAINING order (count = model channels)
FIBERS = ["1B", "2A", "2B"]
WINDOW = (0.0, 1400.0)                      # analysis window (s); must sit within mic coverage
NST_BAND = (190.0, 220.0)                    # NST (microphone) bandpass — selectable
OUT_DIR = Path(__file__).resolve().parent / "out_yz" / "pt8_funet-v36_whole"

NST_PAD = 30.0                               # extra s of NST saved on each side, so a large drift
                                             # shift has NST data to pull in (lag can be many s)
NST_TARGET_FS = 2000

# Beat-timing check (beats_hr_check.html): keep in sync with the HR script.
NST_DETECTOR = "v7_beat_detector"            # beat detector for the NST beat timing
BPM_RANGE = (110.0, 200.0)                    # fetal HR bounds (beat spacing gate)
FUNET_HEIGHT_K = 0.5                         # activity peak height threshold = mean + k*std
NST_ENV_SMOOTH_S = 0.02                      # NST RMS envelope smoothing for the check plot
IBI_MA = 10                                  # moving-average width (beats) for the BPM panel; 1 = raw
SAVE_INTERACTIVE = True                      # also write activity_check.html (zoom/pan/drag in a browser)
INTERACTIVE_MAX_POINTS = 200000              # per-trace point budget for the HTML (peak-preserving decimation)
# ---------------------------------------------------------------------------

FIBER_MAP = {"1A": ("ps4000.npy", 1), "1B": ("ps4000.npy", 2),
             "2A": ("ps3000a.npy", 1), "2B": ("ps3000a.npy", 2),
             "2C": ("ps3000a.npy", 3), "2D": ("ps3000a.npy", 4)}


def bandpass(x, fs, lo, hi, order=3, rp=1):
    sos = cheby1(order, rp=rp, Wn=[lo, hi], fs=fs, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def rms_envelope(x, fs, smooth_s):
    n = max(1, int(round(fs * smooth_s)))
    return np.sqrt(np.maximum(np.convolve(x * x, np.ones(n) / n, mode="same"), 0.0))


def moving_average_v2(y, window):
    """Same edge-aware moving average as the HR script's (old yz signal_utils)."""
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or y.size == 0:
        return y.copy()
    kernel = np.ones(w)
    counts = np.convolve(np.ones_like(y), kernel, mode="same")  # points per window
    sums = np.convolve(y, kernel, mode="same")
    return sums / counts


def ibi_bpm(beats, ma=1):
    """60/IBI at beat-pair midpoints, moving-averaged over ``ma`` beats (1 = raw)."""
    b = np.sort(np.asarray(beats, float))
    if b.size < 2:
        return np.array([]), np.array([])
    return 0.5 * (b[:-1] + b[1:]), moving_average_v2(60.0 / np.diff(b), ma)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = find_model("funet", MODEL_VERSION)
    if entry is None:
        raise KeyError(f"no funet model named {MODEL_VERSION!r} in the registry")
    cfg = load_config(entry.config)
    if len(FIBERS) != cfg.model.channels:
        raise ValueError(f"selected {len(FIBERS)} fibers but {MODEL_VERSION} expects "
                         f"{cfg.model.channels} channels")

    # FUNet inference chunk window: run_funet derives it from cfg.train.crop_len, so override
    # that to set it from here. Report the actual (frame-/divisor-quantised) window it resolves to.
    if INFERENCE_WINDOW_S is not None:
        cfg.train.crop_len = int(INFERENCE_WINDOW_S)
    _div = 2 ** len(cfg.model.dilations)
    _win = max(_div, ((cfg.train.crop_len * SAMPLE_RATE) // cfg.model.hop_length) // _div * _div)
    print(f"FUNet inference chunk window: {_win} frames = "
          f"{_win * cfg.model.hop_length / SAMPLE_RATE:.3f} s (crop_len={cfg.train.crop_len}s)")

    # --- Load + window the selected fibers, stack (C, T) ---
    cache = {}
    def windowed_file(fname):
        if fname not in cache:
            a = np.load(DATA_DIR / fname)
            t = a[:, 0]
            m = (t >= WINDOW[0]) & (t <= WINDOW[1])
            cache[fname] = (t[m], a[m], int(round(1.0 / np.median(np.diff(t)))))
        return cache[fname]

    cols, t_ref, fs_fib = [], None, None
    for name in FIBERS:
        fname, col = FIBER_MAP[name]
        tw, aw, fs = windowed_file(fname)
        cols.append(aw[:, col])
        if t_ref is None:
            t_ref, fs_fib = tw, fs
    L = min([len(c) for c in cols] + [len(t_ref)])
    x = np.stack([c[:L].astype(np.float32) for c in cols])     # (C, L)
    t_fib = np.asarray(t_ref[:L], dtype=float)
    print(f"fibers {FIBERS}: stacked {x.shape} @ {fs_fib} Hz, window {WINDOW}")

    # --- FUNet beat activity ---
    device = torch.device("cpu")
    model = load_funet(cfg, entry.checkpoint, device)
    activity = np.asarray(run_funet(x, fs_fib, model, cfg, device), dtype=float)[:L]
    np.save(OUT_DIR / "fiber_activity.npy", np.column_stack([t_fib, activity]))
    print(f"{MODEL_VERSION}: beat activity {activity.shape}, "
          f"range [{activity.min():.4g}, {activity.max():.4g}] -> fiber_activity.npy")

    # --- NST (microphone): window -> resample -> bandpass ---
    fs_mic, mic = wavread(DATA_DIR / "microphone.wav")
    mic = np.asarray(mic, dtype=float)
    if mic.ndim > 1:
        mic = mic[:, 0]
    t_mic = np.arange(mic.size) / fs_mic
    mm = (t_mic >= WINDOW[0] - NST_PAD) & (t_mic <= WINDOW[1] + NST_PAD)
    if not mm.any():
        raise ValueError(f"WINDOW {WINDOW} outside mic coverage (0-{t_mic[-1]:.0f}s).")
    seg, t0 = mic[mm], t_mic[mm][0]
    g = gcd(int(fs_mic), NST_TARGET_FS)
    seg_rs = resample_poly(seg, NST_TARGET_FS // g, int(fs_mic) // g)
    t_nst = t0 + np.arange(seg_rs.size) / NST_TARGET_FS
    nst = np.asarray(bandpass(seg_rs, NST_TARGET_FS, NST_BAND[0], NST_BAND[1]), dtype=float)
    np.save(OUT_DIR / "nst.npy", np.column_stack([t_nst, nst]))
    print(f"NST (microphone.wav): {seg_rs.size} samp @ {NST_TARGET_FS} Hz, band {NST_BAND} -> nst.npy")

    # --- Metadata for the plotting script ---
    meta = dict(model=MODEL_VERSION, fibers=FIBERS, window=list(WINDOW),
                nst_band=list(NST_BAND), fiber_fs=fs_fib, nst_fs=NST_TARGET_FS,
                patient=DATA_DIR.name)
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # --- Beat timing: FUNet activity peak-picked vs the NST_DETECTOR ---------
    distance = max(1, int(round(60.0 / BPM_RANGE[1] * fs_fib)))
    height = activity.mean() + FUNET_HEIGHT_K * activity.std()
    peaks, _ = find_peaks(activity, distance=distance, height=height)
    beats_f = t_fib[peaks]
    beats_n = run_detector(NST_DETECTOR, Audio(t_nst, NST_TARGET_FS, nst), BPM_RANGE)
    beats_n = beats_n[(beats_n >= WINDOW[0]) & (beats_n <= WINDOW[1])]   # drop the NST_PAD margins
    print(f"beats: FUNet={beats_f.size}, NST({NST_DETECTOR})={beats_n.size}")

    # --- raw_window.png: the selected fibers (raw) + bandpassed NST, stacked ---
    nm = (t_nst >= WINDOW[0]) & (t_nst <= WINDOW[1])
    fig, axes = plt.subplots(len(FIBERS) + 1, 1, figsize=(15, 1.6 * (len(FIBERS) + 1)),
                             sharex=True, constrained_layout=True)
    for ax, name, row in zip(axes, FIBERS, x):
        ax.plot(t_fib[::10], row[::10], lw=0.3, color="0.4")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[-1].plot(t_nst[nm][::5], nst[nm][::5], lw=0.3, color="tab:red")
    axes[-1].set_ylabel("NST", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(*WINDOW)
    fig.suptitle(f"{DATA_DIR.name} raw fibers {'+'.join(FIBERS)} + NST "
                 f"({NST_BAND[0]:g}-{NST_BAND[1]:g} Hz) — {WINDOW[0]:.0f}-{WINDOW[1]:.0f}s "
                 f"(decimated)", fontsize=10)
    fig.savefig(OUT_DIR / "raw_window.png", dpi=150)
    plt.close(fig)

    # --- beats_hr_check.html: beats over the signals + both trains' 60/IBI BPM
    #     (moving-averaged over IBI_MA beats), as an interactive x-linked figure
    #     over the whole window -- browser zoom replaces the old fixed ZOOM crop. ---
    env_n = rms_envelope(nst, NST_TARGET_FS, NST_ENV_SMOOTH_S)
    tbn, bn = ibi_bpm(beats_n, IBI_MA)
    tbf, bf = ibi_bpm(beats_f, IBI_MA)
    grid = np.arange(WINDOW[0], WINDOW[1] + 1e-9, 1.0)
    def _on_grid(t, y):
        if t.size < 2:
            return np.full(grid.shape, np.nan)
        yi = np.interp(grid, t, y)
        yi[(grid < t[0]) | (grid > t[-1])] = np.nan
        return yi
    ga, gb = _on_grid(tbn, bn), _on_grid(tbf, bf)
    mv = ~np.isnan(ga) & ~np.isnan(gb)
    r = float(np.corrcoef(ga[mv], gb[mv])[0, 1]) if mv.sum() >= 3 else float("nan")
    smooth_label = f"MA {IBI_MA}" if IBI_MA > 1 else "unsmoothed"
    print(f"beat-timing check: 60/IBI ({smooth_label}) Pearson r = {r:.3f} "
          f"(paired on a 1 s grid)")
    wrote_beats_html = False
    try:
        save_interactive_beats_hr(
            str(OUT_DIR / "beats_hr_check.html"),
            f"{DATA_DIR.name} FUNet ({MODEL_VERSION}) vs NST ({NST_DETECTOR}) — "
            f"beats + 60/IBI BPM ({smooth_label}) — r = {r:.3f} — "
            f"{beats_n.size} NST / {beats_f.size} FUNet beats",
            t_nst[nm], env_n[nm], "NST envelope",
            t_fib, activity, "FUNet activity",
            beats_n, beats_f,
            tbn, bn, "NST BPM", tbf, bf, "FUNet BPM",
            max_points=INTERACTIVE_MAX_POINTS,
            bpm_subtitle=f"BPM (60/IBI, {smooth_label})")
        wrote_beats_html = True
        print("  -> beats_hr_check.html")
    except ImportError:
        print("  [interactive] plotly not installed — skipping beats_hr_check.html "
              "(poetry run pip install plotly)")

    # --- Check plot: FUNet activity over the window ---
    fig, (ax0) = plt.subplots(1, 1, figsize=(15, 6), constrained_layout=True)
    ax0.plot(t_fib, activity, lw=0.5, color="tab:green")
    ax0.set_title(f"FUNet ({MODEL_VERSION}) beat activity — {WINDOW[0]:.0f}-{WINDOW[1]:.0f}s", fontsize=10)
    ax0.set_xlim(*WINDOW); ax0.set_xlabel("Time (s)"); ax0.set_ylabel("Activity")
    fig.savefig(OUT_DIR / "activity_check.png", dpi=150)
    plt.close(fig)

    outputs = ("fiber_activity.npy, nst.npy, run_meta.json, raw_window.png, "
               "activity_check.png")
    if wrote_beats_html:
        outputs += ", beats_hr_check.html"
    if SAVE_INTERACTIVE:
        try:
            save_interactive(str(OUT_DIR / "activity_check.html"),
                             f"{DATA_DIR.name} FUNet ({MODEL_VERSION}) activity vs NST — "
                             f"{WINDOW[0]:.0f}-{WINDOW[1]:.0f}s",
                             t_fib, activity, "FUNet activity",
                             t_nst, nst, "NST (bandpassed)",
                             y_a_title="FUNet activity", y_b_title="NST amplitude",
                             max_points=INTERACTIVE_MAX_POINTS)
            outputs += ", activity_check.html"
            # raw model input: the windowed fibers + NST ground truth, one x-linked
            # panel each (zoom/pan/drag). NST is clipped to WINDOW so it lines up with
            # the fibers (nst.npy is saved with NST_PAD padding for the HR drift shift).
            nm = (t_nst >= WINDOW[0]) & (t_nst <= WINDOW[1])
            fiber_panels = [(f"fiber {name}", t_fib, x[i]) for i, name in enumerate(FIBERS)]
            save_interactive_multi(
                str(OUT_DIR / "raw_fibers_check.html"),
                f"{DATA_DIR.name} raw fibers {'+'.join(FIBERS)} (FUNet input) + NST — "
                f"{WINDOW[0]:.0f}-{WINDOW[1]:.0f}s",
                fiber_panels + [("NST (bandpassed mic)", t_nst[nm], nst[nm])],
                max_points=INTERACTIVE_MAX_POINTS)
            outputs += ", raw_fibers_check.html"
        except ImportError:
            print("  [interactive] plotly not installed — skipping activity_check.html "
                  "(poetry run pip install plotly)")
    print(f"Wrote {OUT_DIR}/  ({outputs})")


if __name__ == "__main__":
    main()
