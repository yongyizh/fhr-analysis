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

NST_DRIFT_CORRECTION (below) toggles the same NST clock-drift correction the main
pipeline applies in src/analyze/main.py: the mic drops samples, so its clock falls
behind real time, and analyze.drift.correct_drift inserts the lost seconds back in as
spacers -- shifting every NST beat time (and, so the check plots stay coherent, the NST
waveform's time axis) later by the drift accrued up to that point. The fibers never
drift, so nothing on the FUNet side moves. Off -> the NST stays on its nominal clock.

Writes run_meta.json (model / fibers / window / band / drift state) for the plotting
script's titles, and nst_beats.npy (the NST beat times actually used, corrected or not).

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
from dataclasses import replace
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
    save_interactive, save_interactive_beats_hr, save_interactive_multi,
    save_interactive_overlay)

from analyze.constants import (  # noqa: E402
    NST_DRIFT_LOG_FILE, FETAL_BPM_RANGE, FETAL_ACOUSTIC_BAND_HZ,
    FETAL_ACOUSTIC_BAND_NARROW_HZ)
from analyze.data import Audio, windowed  # noqa: E402
from analyze.drift import correct_drift  # noqa: E402
from analyze.demodulate import demodulate_audio  # noqa: E402
from analyze.filters import bp_filter    # noqa: E402
from analyze.neossnet import _run_neossnet_chunked  # noqa: E402
from analyze.hr import fHROutput, sot_beats           # noqa: E402
from analyze.plot_hr import plot_hr_comparison, _inst_hr_v2, _pairwise_r  # noqa: E402
from analyze.sot import load_sot                      # noqa: E402
from beat_app.detectors import discover_detectors, run_detector  # noqa: E402
from common.audio import SAMPLE_RATE     # noqa: E402  (FUNet model sample rate, Hz)
from funet.config import load_config     # noqa: E402
from funet.inference import load_funet, run_funet  # noqa: E402
from palnet.config import load_config as load_palnet_config  # noqa: E402
from palnet.inference import load_palnet, run_palnet         # noqa: E402
from rtmon.models import discover_all              # noqa: E402
from tslnet.config import load_config as load_tslnet_config  # noqa: E402
from tslnet.inference import load_tslnet, run_tslnet         # noqa: E402

# ---------------------------------------------------------------------------
# SELECT HERE
# ---------------------------------------------------------------------------
DATA_DIR = Path("/Users/yongyi/Downloads/Fiber_research/Banner patient data/"
                "Banner_test_20251220/PT12_2")
MODEL_VERSION = "demodulation"                  # what produces the signal the beats come off.
# Any version directory under lib/{funet,tslnet,palnet,tune-ssnet}/models/ -- the family is
# looked up from the name, so this one line switches models:
#   "funet-v24"        stacked FIBERS -> beat-activity envelope   (main.py: v2 beats)
#   "tslnet-v9"        stacked FIBERS -> beat-activity envelope   (main.py: v7 beats)
#   "palnet-v4"        stacked FIBERS -> beat-activity envelope   (main.py: v7 beats)
#   "tuned-model-v13"  NeoSSNet on SINGLE_FIBER -> heart waveform (main.py: v7 beats,
#                      band-passed NEOSSNET_PRE_BAND before, NEOSSNET_POST_BAND after)
#   "bandpass"         no model at all: SINGLE_FIBER band-passed to BANDPASS_BAND
#                      (main.py's run_raw_bandpass; v7 beats)
#   "demodulation"     no model either: SINGLE_FIBER read as a CARRIER the heartbeat
#                      amplitude-modulates -> A(t) = |u(t)|  (see analyze/demodulate.py).
#                      Emits an ENVELOPE, not a waveform, so unlike "bandpass" the beat is
#                      the bump itself rather than a burst of oscillation.
# FIBERS must match the checkpoint's channel count for funet/tslnet/palnet; the
# single-channel modes ignore it except that SINGLE_FIBER has to be in the list.
BANDPASS_ONLY = "bandpass"                   # the model-free MODEL_VERSION value
SINGLE_FIBER = "1B"                          # fiber the single-channel modes use (main.py: 1B)
BANDPASS_BAND = (200,230)       # (190, 220) — "bandpass" mode band, adjustable
DEMODULATION_ONLY = "demodulation"           # the other model-free MODEL_VERSION value
DEMOD_BAND = (200.0, 230.0)     # "demodulation" carrier band: narrower than BANDPASS_BAND on
                                # purpose -- the phase fit assumes ONE dominant tone, and a
                                # wide band with two comparable ones gives a meaningless
                                # amplitude-weighted average (see estimate_carrier_hz).
DEMOD_BASEBAND_HZ = 25.0        # baseband low-pass; only has to pass a 1.8-3 Hz beat
DEMOD_BANDPASS_ORDER = 4
NEOSSNET_PRE_BAND = (200,245)         # (190, 220) before NeoSSNet   (200,245) for pt13_2
NEOSSNET_POST_BAND = (200,245) # (190, 210) after NeoSSNet
INFERENCE_WINDOW_S = None                    # FUNet inference chunk length (s, integer). The spectrogram is
                                             # processed in chunks this long, each normalised on its own, so
                                             # this also governs how much the activity shifts when the window
                                             # START moves. None -> the model's trained crop_len (~7 s for
                                             # v24). Straying far from the trained value can hurt accuracy
                                             # (GroupNorm then sees a different time extent than it trained on).
FIBERS = ["1B", "2A", "2B", "2C", "2D"]      # fibers to stack, in TRAINING order (count = model channels)
# FIBERS = ["1B", "2B", "2C"]
# FIBERS = ["1B"]
# FIBERS = ["1B", "2A", "2B"]           # palnet-v4: channels=3, trained on 1B/2A/2B in this order
WINDOW = (0.0, 660.0)                      # analysis window (s); must sit within mic coverage
NST_BAND = (190.0, 220.0)                    # NST (microphone) bandpass — selectable
OUT_DIR = Path(__file__).resolve().parent / "out_yz" / "pt12" / "demodulation (non-envelope)" 

NST_PAD = 30.0                               # extra s of NST saved on each side, so a large drift
                                             # shift has NST data to pull in (lag can be many s)
NST_TARGET_FS = 2000

# --- NST clock-drift correction -------------------------------------------
# The NST clock falls behind real time wherever the recording dropped samples, so its
# timestamps need that lost time inserted back in (the "spacers") before its HR can be
# compared against the fibers', which never drifted. correct_drift shifts each timestamp
# later by the cumulative seconds_lost accrued by that point -- the identical helper
# src/analyze/main.py's pipelines use via plot_hr_corrected, so the two agree beat for beat.
NST_DRIFT_CORRECTION = True                  # SWITCH: False -> leave the NST on its nominal clock
NST_DRIFT_LOG = None                         # dropout CSV; None -> DATA_DIR / nst_dropouts.csv

# --- Beat detectors -------------------------------------------------------
# Two independent knobs, one per side. Names come from the beat_app registry
# (discover_detectors()): "v1_beat_detector" ... "v9_beat_detector". Both are checked at
# the top of main(), so a typo fails immediately instead of part-way through inference.
# Whatever is set here is what produces funet_beats.npy / nst_beats.npy, so it governs
# every HR number downstream -- not just the beats_hr_check.html panel.
NST_DETECTOR = "v7_beat_detector"            # detector for the NST (microphone) beat timing.
                                             # main.py uses v7 here.
FIBER_DETECTOR = None            # detector for the MODEL OUTPUT -- the funet/tslnet/palnet
                                             # activity, the NeoSSNet waveform, or the plain
                                             # band-passed fiber, whichever MODEL_VERSION
                                             # selects. Set a name to pin it, e.g.
                                             # "v9_beat_detector" to match the NST side.
                                             # None = main.py's per-family default:
                                             #   funet -> v2_beat_detector
                                             #   tslnet / palnet / ssnet / bandpass -> v7
BPM_RANGE = FETAL_BPM_RANGE                  # the band handed to BOTH detectors, as main.py's
                                             # fiber_beats and sot_beats do. It is a detection
                                             # parameter, not just a filter: it sets v2's minimum
                                             # peak spacing (60/hi seconds) and v7's HMM duration
                                             # priors, so narrowing it changes which beats exist.
FUNET_HEIGHT_K = 0.5                         # activity peak height threshold = mean + k*std
NST_ENV_SMOOTH_S = 0.02                      # NST RMS envelope smoothing for the check plot
IBI_MA = 10                                  # moving-average width (beats) for the BPM panel; 1 = raw
SAVE_INTERACTIVE = True                      # also write activity_check.html (zoom/pan/drag in a browser)
ACTIVITY_CHECK_LAYOUT = "panels"             # how activity_check.html shows the model output against the
                                             # NST. They are different signals on different scales, and the
                                             # old dual-y overlay drew them on top of each other -- in
                                             # bandpass mode, where both are ~200 Hz waveforms of similar
                                             # shape, the two are impossible to tell apart. Options:
                                             #   "panels"  one x-linked panel each, own y-axis (nothing overlaps)
                                             #   "offset"  one panel, each trace normalised then shifted up
                                             #             by ACTIVITY_CHECK_OFFSET so they sit in bands
                                             #   "overlay" the old shared panel + secondary y-axis
ACTIVITY_CHECK_OFFSET = 6.0                  # "offset" layout only: vertical gap between the two traces, in
                                             # units of their own robust (99th-percentile) amplitude. Both
                                             # signals here are waveforms whose peaks run to ~2-3x that, so
                                             # the bands only clear each other well above the overlay
                                             # writer's own default of 2.
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
    # Clamp to the array length. np.convolve(mode="same") returns max(M, N) elements, so a
    # window wider than the trace yields MORE smoothed values than there are timestamps --
    # silently desyncing t from bpm on a sparse beat trace (few beats + the hardcoded
    # 20-point HR window is exactly that case) and turning any correlation into noise.
    w = min(w, y.size)
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



# --- model dispatch --------------------------------------------------------
# The beat detector each family gets in src/analyze/main.py: run_funet_pipeline uses
# fiber_beats(v2_beat_detector); run_tslnet_pipeline, run_palnet_pipeline,
# run_neossnet_pipeline and run_raw_bandpass all use v7.
_FAMILY_DETECTOR = {"funet": "v2_beat_detector", "tslnet": "v7_beat_detector",
                    "palnet": "v7_beat_detector", "ssnet": "v7_beat_detector",
                    "bandpass": "v7_beat_detector",
                    # A(t) is already an envelope, so v7's Shannon-energy stage is doing
                    # less work here than it does on a waveform. Override FIBER_DETECTOR to
                    # try others -- the choice is orthogonal to producing the signal.
                    # NOT plain v7: A(t) is already an envelope, and v7's Shannon-energy
                    # stage (-x^2 log x^2) is non-monotonic, so run over an envelope it
                    # SUPPRESSES the strongest beats. Measured cost of getting this wrong on
                    # pt13_2: r = 0.036 instead of 0.24.
                    "demodulation": "v7_envelope_aware_beat_detector"}

# Families that stack every fiber in FIBERS and emit a beat-activity envelope; the rest
# are single-channel and read SINGLE_FIBER only.
_MULTI_FIBER = ("funet", "tslnet", "palnet")


def resolve_model(version):
    """``(family, entry)`` for ``version``. ``entry`` is None for the model-free bandpass
    mode. The family is looked up by scanning the registry, so MODEL_VERSION alone selects
    both which directory the checkpoint comes from and which inference path runs."""
    if version == BANDPASS_ONLY:
        return "bandpass", None
    if version == DEMODULATION_ONLY:
        return "demodulation", None
    for family, entries in discover_all().items():
        for e in entries:
            if e.version == version:
                return family, e
    have = {f: [e.version for e in es] for f, es in discover_all().items()}
    raise KeyError(f"no model named {version!r} in the registry (or {BANDPASS_ONLY!r} / "
                   f"{DEMODULATION_ONLY!r}). "
                   f"Available: {have}")


def model_signal(family, entry, x, fs, fiber_names):
    """Run ``family``'s inference over the stacked fibers ``x`` (C, T) and return the 1-D
    signal the beat detector should read, plus a short label for plot titles.

    funet/tslnet consume every channel and emit a beat-activity envelope; ssnet (NeoSSNet)
    and bandpass are single-channel and emit a WAVEFORM, which is why they get v7 rather
    than v2 -- the same split main.py makes between run_funet_pipeline and the others.
    """
    if family in ("bandpass", "ssnet", "demodulation"):
        if SINGLE_FIBER not in fiber_names:
            raise ValueError(f"SINGLE_FIBER {SINGLE_FIBER!r} is not in FIBERS {fiber_names}")
        row = np.asarray(x[fiber_names.index(SINGLE_FIBER)], dtype=float)

    if family == "bandpass":
        # main.py run_raw_bandpass: bp(*FETAL_ACOUSTIC_BAND_HZ, "butter") on the fiber.
        sig = bp_filter(Audio(np.arange(row.size) / fs, fs, row),
                        BANDPASS_BAND[0], BANDPASS_BAND[1], filter_type="butter").data
        return np.asarray(sig, dtype=float), f"bandpass {BANDPASS_BAND[0]:g}-{BANDPASS_BAND[1]:g} Hz"

    if family == "demodulation":
        # The band as a CARRIER rather than as acoustics: bandpass -> Hilbert -> two-pass
        # mix-down -> A(t) = |u(t)|. The carrier is measured from the analytic phase, not
        # configured, so the printout below is the one number worth checking -- a final
        # residual that is not ~0 means the fit never locked and A(t) is not an envelope of
        # anything. Returns an envelope on the fiber's own time axis.
        t = np.arange(row.size) / fs
        bandpassed = bp_filter(Audio(t, fs, row), DEMOD_BAND[0], DEMOD_BAND[1],
                               order=DEMOD_BANDPASS_ORDER, filter_type="butter")
        amplitude, initial, residual, corrected, final = demodulate_audio(
            bandpassed, baseband_hz=DEMOD_BASEBAND_HZ)
        print(f"demodulation: fiber {SINGLE_FIBER} @ {fs:g} Hz, "
              f"carrier band {DEMOD_BAND[0]:g}-{DEMOD_BAND[1]:g} Hz, "
              f"baseband LPF {DEMOD_BASEBAND_HZ:g} Hz")
        print(f"demodulation: carrier {initial:.6f} Hz -> {corrected:.6f} Hz "
              f"(residual {residual:+.6f} Hz, final {final:+.3e} Hz)")
        return (np.asarray(amplitude.data, dtype=float),
                f"demodulated envelope ({DEMOD_BAND[0]:g}-{DEMOD_BAND[1]:g} Hz carrier)")

    if family == "ssnet":
        # main.py run_neossnet_pipeline: band-pass, separate, then narrow to the fetal band.
        # _run_neossnet_chunked is main.py's own splitter (30 s chunks) -- long inputs are
        # otherwise one enormous forward pass.
        t = np.arange(row.size) / fs
        pre = bp_filter(Audio(t, fs, row), *NEOSSNET_PRE_BAND, filter_type="butter").data
        heart, _lung = _run_neossnet_chunked(pre, fs, entry.checkpoint, entry.config)
        post = bp_filter(Audio(t, fs, np.asarray(heart, dtype=float)),
                         *NEOSSNET_POST_BAND, filter_type="butter").data
        return np.asarray(post, dtype=float), "NeoSSNet heart"

    device = torch.device("cpu")
    if family == "funet":
        cfg = load_config(entry.config)
        _check_channels(cfg, fiber_names)
        _report_funet_window(cfg)
        model = load_funet(cfg, entry.checkpoint, device)
        return np.asarray(run_funet(x, fs, model, cfg, device), dtype=float), "beat activity"

    if family == "tslnet":
        cfg = load_tslnet_config(entry.config)
        _check_channels(cfg, fiber_names)
        model = load_tslnet(cfg, entry.checkpoint, device)
        return np.asarray(run_tslnet(x, fs, model, cfg, device), dtype=float), "beat activity"

    if family == "palnet":
        # main.py run_palnet_pipeline: same contract as FUNet's -- stacked fibers in,
        # per-sample beat activity out -- only the trunk differs (frozen AudioSet instead
        # of a learned U-Net), so palnet.inference mirrors funet.inference signature for
        # signature. The checkpoint is head-only; the trunk comes from the HF cache.
        cfg = load_palnet_config(entry.config)
        _check_channels(cfg, fiber_names)
        model = load_palnet(cfg, entry.checkpoint, device)
        return np.asarray(run_palnet(x, fs, model, cfg, device), dtype=float), "beat activity"

    raise KeyError(f"no inference path for family {family!r}")


def _check_channels(cfg, fiber_names):
    if len(fiber_names) != cfg.model.channels:
        raise ValueError(f"selected {len(fiber_names)} fibers {fiber_names} but "
                         f"{MODEL_VERSION} expects {cfg.model.channels} channels")


def _report_funet_window(cfg):
    """FUNet's inference chunk window comes from cfg.train.crop_len; INFERENCE_WINDOW_S
    overrides it. Report the frame-/divisor-quantised value actually used."""
    if INFERENCE_WINDOW_S is not None:
        cfg.train.crop_len = int(INFERENCE_WINDOW_S)
    div = 2 ** len(cfg.model.dilations)
    win = max(div, ((cfg.train.crop_len * SAMPLE_RATE) // cfg.model.hop_length) // div * div)
    print(f"FUNet inference chunk window: {win} frames = "
          f"{win * cfg.model.hop_length / SAMPLE_RATE:.3f} s (crop_len={cfg.train.crop_len}s)")


def main():
    if ACTIVITY_CHECK_LAYOUT not in ("panels", "offset", "overlay"):
        raise ValueError(f"ACTIVITY_CHECK_LAYOUT must be 'panels', 'offset' or 'overlay', "
                         f"got {ACTIVITY_CHECK_LAYOUT!r}")
    known = sorted(discover_detectors())
    for label, name in (("FIBER_DETECTOR", FIBER_DETECTOR), ("NST_DETECTOR", NST_DETECTOR)):
        if name is not None and name not in known:
            raise ValueError(f"{label} = {name!r} is not a registered beat detector. "
                             f"Available: {known}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    family, entry = resolve_model(MODEL_VERSION)
    detector_name = FIBER_DETECTOR or _FAMILY_DETECTOR[family]
    # funet/tslnet/palnet stack every fiber in FIBERS; bandpass and neossnet read
    # SINGLE_FIBER only. Recording the difference keeps downstream titles from claiming
    # fibers that nothing consumed.
    fibers_used = list(FIBERS) if family in _MULTI_FIBER else [SINGLE_FIBER]
    print(f"model {MODEL_VERSION!r}: family={family}, "
          f"input fibers {'+'.join(fibers_used)}"
          + (f" (of {'+'.join(FIBERS)} loaded)" if fibers_used != list(FIBERS) else ""))
    print(f"beat detectors: fiber {detector_name}"
          f"{'' if FIBER_DETECTOR else ' (family default)'} / NST {NST_DETECTOR}, "
          f"both over BPM_RANGE {tuple(BPM_RANGE)}")

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

    # --- Model output: beat activity (funet/tslnet) or a waveform (neossnet/bandpass) ---
    activity, signal_label = model_signal(family, entry, x, fs_fib, FIBERS)
    activity = activity[:L]
    np.save(OUT_DIR / "fiber_activity.npy", np.column_stack([t_fib, activity]))
    print(f"{MODEL_VERSION}: {signal_label} {activity.shape}, "
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

    # --- Beat timing, exactly as src/analyze/main.py's run_funet_pipeline does it ----
    # Fiber beats: the family's detector (see _FAMILY_DETECTOR) over BPM_RANGE, which is
    # what main.py's fiber_beats does to the same signal.
    # NST beats:   analyze's own load_sot -> windowed -> sot_beats chain. That keeps the mic
    #              at its native 44.1 kHz and band-passes it TWICE (cheby1 inside load_sot,
    #              then again inside sot_beats). The 2 kHz nst.npy written above is a
    #              DIFFERENT signal and yields different beats from the same detector, so it
    #              is no longer the beat source -- it is kept only as the waveform the HR
    #              script's autocorrelation method needs, main.py having no autocorr method
    #              to be equivalent to.
    if detector_name:
        beats_f = run_detector(detector_name, Audio(t_fib, fs_fib, activity), BPM_RANGE)
    else:
        distance = max(1, int(round(60.0 / BPM_RANGE[1] * fs_fib)))
        height = activity.mean() + FUNET_HEIGHT_K * activity.std()
        peaks, _ = find_peaks(activity, distance=distance, height=height)
        beats_f = t_fib[peaks]

    # sot_beats takes the detector FUNCTION, so resolve NST_DETECTOR's name through the
    # same registry run_detector uses and the switch keeps working.
    sot = sot_beats(discover_detectors()[NST_DETECTOR], OUT_DIR, fetal_bpm=BPM_RANGE)(
        windowed(WINDOW[0], WINDOW[1])(load_sot()(str(DATA_DIR))))
    beats_n = np.asarray(sot.mic_beats, dtype=float)
    print(f"beats: {MODEL_VERSION}({detector_name or 'peak-pick'})={beats_f.size}, "
          f"NST({NST_DETECTOR}, via load_sot @ {sot.mic.hz} Hz)={beats_n.size}")

    # --- NST clock-drift correction: insert the dropped-sample spacers -------
    # Runs after the detector, and on the SOT beats only -- exactly what plot_hr_corrected
    # does (replace(sot, mic_beats=correct_drift(...))). t_nst itself stays nominal because
    # it is what nst.npy is written with, and the HR script's autocorrelation reads that
    # file; the check plots get their own corrected copy (t_nst_plot) below.
    drift_log = Path(NST_DRIFT_LOG) if NST_DRIFT_LOG else DATA_DIR / NST_DRIFT_LOG_FILE
    if NST_DRIFT_CORRECTION:
        if not drift_log.exists():
            raise FileNotFoundError(
                f"NST_DRIFT_CORRECTION is on but there is no dropout log at {drift_log}. "
                f"Set NST_DRIFT_CORRECTION = False to run on the nominal clock instead.")
        corrected = correct_drift(beats_n, drift_log)
        shift = corrected - beats_n
        beats_n = corrected
        sot = replace(sot, mic_beats=beats_n)
        drift_total = float(shift.max()) if shift.size else 0.0
        print(f"NST drift correction: ON ({drift_log.name}) — beats shifted "
              f"+{shift.min() if shift.size else 0:.3f}..{drift_total:.3f}s")
    else:
        drift_total = 0.0
        print(f"NST drift correction: OFF "
              f"(log {'present' if drift_log.exists() else 'absent'} at {drift_log.name})")
    drift_label = "drift-corrected" if NST_DRIFT_CORRECTION else "nominal clock"

    # NST time axis for the CHECK PLOTS only. correct_drift moved the NST beats onto the
    # corrected clock just above; the waveform has to move with them or every beat line is
    # drawn up to nst_drift_total_s (19.8 s on PT13_2) away from the beat it marks, and the
    # NST no longer lines up with the fiber either. nst.npy keeps the nominal axis: the HR
    # script's autocorrelation reads that file and remaps its own RESULT instead.
    t_nst_plot = correct_drift(t_nst, drift_log) if NST_DRIFT_CORRECTION else t_nst

    # Both beat trains, main.py-equivalent and already drift-corrected, for the HR script.
    np.save(OUT_DIR / "nst_beats.npy", beats_n)
    np.save(OUT_DIR / "funet_beats.npy", beats_f)

    # --- Metadata for the plotting script ---
    meta = dict(model=MODEL_VERSION, fibers=fibers_used, fibers_loaded=list(FIBERS),
                window=list(WINDOW),
                nst_band=list(NST_BAND), fiber_fs=fs_fib, nst_fs=NST_TARGET_FS,
                patient=DATA_DIR.name, bpm_range=list(BPM_RANGE),
                family=family, fiber_detector=detector_name, nst_detector=NST_DETECTOR,
                signal=signal_label,
                beats_are_mainpy_equivalent=True,
                nst_drift_correction=bool(NST_DRIFT_CORRECTION),
                nst_drift_log=str(drift_log),   # recorded either way, so the HR script can find it
                nst_drift_total_s=drift_total)
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # --- hr_comparison_mainpy.png: main.py's own figure, from main.py's own plotter ---
    # No maternal panel: no chest fiber is loaded here, so fiber_beats' maternal half has
    # no counterpart. The fetal panel and its R/MAE box are identical to main.py's.
    plot_hr_comparison(
        fHROutput(fetal_source=Audio(t_fib, fs_fib, activity), fetal_beats=beats_f,
                  maternal_source=None, maternal_beats=None),
        sot, OUT_DIR, filename="hr_comparison_mainpy.png")

    # --- raw_window.png: the selected fibers (raw) + bandpassed NST, stacked ---
    nm = (t_nst_plot >= WINDOW[0]) & (t_nst_plot <= WINDOW[1])
    fig, axes = plt.subplots(len(FIBERS) + 1, 1, figsize=(15, 1.6 * (len(FIBERS) + 1)),
                             sharex=True, constrained_layout=True)
    for ax, name, row in zip(axes, FIBERS, x):
        ax.plot(t_fib[::10], row[::10], lw=0.3, color="0.4")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[-1].plot(t_nst_plot[nm][::5], nst[nm][::5], lw=0.3, color="tab:red")
    axes[-1].set_ylabel("NST", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(*WINDOW)
    fig.suptitle(f"{DATA_DIR.name} raw fibers {'+'.join(FIBERS)} "
                 f"({MODEL_VERSION} reads {'+'.join(fibers_used)}) + NST "
                 f"({NST_BAND[0]:g}-{NST_BAND[1]:g} Hz, {drift_label}) — "
                 f"{WINDOW[0]:.0f}-{WINDOW[1]:.0f}s (decimated)", fontsize=10)
    fig.savefig(OUT_DIR / "raw_window.png", dpi=150)
    plt.close(fig)

    # --- beats_hr_check.html: beats over the signals + both trains' 60/IBI BPM
    #     (moving-averaged over IBI_MA beats), as an interactive x-linked figure
    #     over the whole window -- browser zoom replaces the old fixed ZOOM crop. ---
    env_n = rms_envelope(nst, NST_TARGET_FS, NST_ENV_SMOOTH_S)
    # main.py's own HR trace + agreement functions (moving-average over 20 beats, band
    # clipped to BPM_RANGE, paired on a 4 Hz grid) rather than the local ibi_bpm/1 s grid,
    # so the r printed here is the same number plot_hr_comparison puts in the figure.
    tbn, bn = _inst_hr_v2(beats_n, BPM_RANGE)
    tbf, bf = _inst_hr_v2(beats_f, BPM_RANGE)
    _stat = _pairwise_r([("Fiber fetal", tbf, bf), ("Mic (SOT)", tbn, bn)])
    r, mae = next(iter(_stat.values()), (float("nan"), float("nan")))
    smooth_label = "MA 20"
    print(f"beat-timing check: 60/IBI ({smooth_label}) Pearson r = {r:.3f}, MAE = {mae:.1f} bpm "
          f"(main.py conventions, NST {drift_label})")
    wrote_beats_html = False
    try:
        save_interactive_beats_hr(
            str(OUT_DIR / "beats_hr_check.html"),
            f"{DATA_DIR.name} {MODEL_VERSION} ({detector_name or 'peak-pick'}) vs "
            f"NST ({NST_DETECTOR}, {drift_label}) — "
            f"beats + 60/IBI BPM ({smooth_label}) — r = {r:.3f} — "
            f"{beats_n.size} NST / {beats_f.size} FUNet beats",
            t_nst_plot[nm], env_n[nm], "NST envelope",
            t_fib, activity, signal_label,
            beats_n, beats_f,
            tbn, bn, "NST BPM", tbf, bf, f"{MODEL_VERSION} BPM",
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
    ax0.set_title(f"{MODEL_VERSION} {signal_label} — {WINDOW[0]:.0f}-{WINDOW[1]:.0f}s", fontsize=10)
    ax0.set_xlim(*WINDOW); ax0.set_xlabel("Time (s)"); ax0.set_ylabel(signal_label)
    fig.savefig(OUT_DIR / "activity_check.png", dpi=150)
    plt.close(fig)

    outputs = ("fiber_activity.npy, nst.npy, nst_beats.npy, funet_beats.npy, "
               "run_meta.json, raw_window.png, activity_check.png, hr_comparison_mainpy.png")
    if wrote_beats_html:
        outputs += ", beats_hr_check.html"
    if SAVE_INTERACTIVE:
        try:
            # NST clipped to WINDOW so it lines up with the fiber (nst.npy keeps its
            # NST_PAD padding for the HR script's drift shift).
            nm = (t_nst_plot >= WINDOW[0]) & (t_nst_plot <= WINDOW[1])
            act_title = (f"{DATA_DIR.name} {MODEL_VERSION} {signal_label} vs NST "
                         f"({drift_label}) — {WINDOW[0]:.0f}-{WINDOW[1]:.0f}s — beats: "
                         f"{MODEL_VERSION} {detector_name or 'peak-pick'} / "
                         f"NST {NST_DETECTOR}")
            def in_window(b):
                """Beat times inside WINDOW. Drift correction pushes the last NST beats
                past WINDOW[1] (44 of them on PT13_2, which ends ~19.8 s late); drawing
                those would stretch the shared x-axis well past where either waveform
                stops. Display only -- nst_beats.npy keeps the full train."""
                b = np.asarray(b, dtype=float)
                return b[(b >= WINDOW[0]) & (b <= WINDOW[1])]

            # 4th element = the beat train drawn over that trace as dashed vertical lines:
            # each signal gets the beats its own detector found in it, the same arrays
            # written to funet_beats.npy / nst_beats.npy and used for every HR number.
            act_series = [(signal_label, t_fib, activity, in_window(beats_f)),
                          ("NST (bandpassed)", t_nst_plot[nm], nst[nm], in_window(beats_n))]
            if ACTIVITY_CHECK_LAYOUT == "panels":
                save_interactive_multi(str(OUT_DIR / "activity_check.html"), act_title,
                                       act_series, max_points=INTERACTIVE_MAX_POINTS)
            elif ACTIVITY_CHECK_LAYOUT == "offset":
                # colours pinned to the other two layouts' (fiber green, NST blue) rather
                # than left to the overlay palette's own order. beat_span="full" runs each
                # beat line the whole panel height, which is what makes it visible at a
                # glance whether the two trains land on the same instants.
                save_interactive_overlay(str(OUT_DIR / "activity_check.html"), act_title,
                                         act_series, offset=ACTIVITY_CHECK_OFFSET,
                                         colors=["green", "royalblue"], beat_span="full",
                                         max_points=INTERACTIVE_MAX_POINTS)
            else:
                save_interactive(str(OUT_DIR / "activity_check.html"), act_title,
                                 t_fib, activity, signal_label,
                                 t_nst_plot[nm], nst[nm], "NST (bandpassed)",
                                 y_a_title=signal_label, y_b_title="NST amplitude",
                                 beats_a=in_window(beats_f), beats_b=in_window(beats_n),
                                 max_points=INTERACTIVE_MAX_POINTS)
            outputs += f", activity_check.html ({ACTIVITY_CHECK_LAYOUT})"
            # raw model input: the windowed fibers + NST ground truth, one x-linked
            # panel each (zoom/pan/drag).
            fiber_panels = [(f"fiber {name}", t_fib, x[i]) for i, name in enumerate(FIBERS)]
            save_interactive_multi(
                str(OUT_DIR / "raw_fibers_check.html"),
                f"{DATA_DIR.name} raw fibers {'+'.join(FIBERS)} "
                f"({MODEL_VERSION} reads {'+'.join(fibers_used)}) + NST "
                f"({drift_label}) — {WINDOW[0]:.0f}-{WINDOW[1]:.0f}s",
                fiber_panels + [("NST (bandpassed mic)", t_nst_plot[nm], nst[nm])],
                max_points=INTERACTIVE_MAX_POINTS)
            outputs += ", raw_fibers_check.html"
        except ImportError:
            print("  [interactive] plotly not installed — skipping activity_check.html "
                  "(poetry run pip install plotly)")
    print(f"Wrote {OUT_DIR}/  ({outputs})")


if __name__ == "__main__":
    main()
