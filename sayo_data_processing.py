"""Offline processing of the "sayo" rtmon sessions (.out/rtmon/sessions/sayo_*).

Default output is a two-track HR comparison. Each track (HR_TRACK_A on fiber
1A, HR_TRACK_B on whichever fibers you pick in the SELECTION block) is a
(family, version, fibers, detector) pipeline matching the live rig's numbers
(rtmon.processors):

  - "funet" / "tslnet": stacked fibers -> beat-activity envelope -> peak-picked
    beats.
  - "ssnet": one fiber -> separated heart waveform -> narrow bandpass -> beat
    detector.
  - "bandpass": no checkpoint at all -- each fiber bandpassed to BANDPASS_BAND
    (selectable), smoothed amplitude envelope, averaged across fibers. A
    detector, if given, picks beats from the mean bandpassed waveform (the
    role it plays after ssnet's separator); with None the envelope is
    peak-picked. The activity is the envelope rather than the carrier: a beat
    is a burst of band energy, and at beat-length lags the carrier phase is
    incoherent -- measured on sayo_short, carrier autocorrelation never scored
    above 0.27 at any band, while the envelope locks onto the PPG-confirmed
    pulse. On sayo data (20-60) Hz puts the median within 2 bpm of the PPG
    truth; (190-220) is the fetal band.

Both tracks' HR is then computed two ways -- one figure per method, both curves
smoothed over ~HR_SMOOTH_S seconds, with the Pearson correlation between the
two curves (on a common 1 s grid over their overlap) printed on the figure:

  hr_compare_ibi.png       HR = 60 / inter-beat interval at each beat
  hr_compare_autocorr.png  HR by windowed autocorrelation of the beat activity

Channel/device mapping recorded by rtmon.recorder:
    ps4000.npy  (N, 3): [t, 1A, 1B]           (chest device)
    ps3000a.npy (N, 5): [t, 2A, 2B, 2C, 2D]   (abdomen device)
i.e. the "1" fibers live on the ps4000, the "2" fibers on the ps3000a. The
mapping is read from session.json, so channels are always the ones *labeled*
1A/1B/2A-2D, whichever file they came from. The two scopes have independent
clocks, hence one time axis per family.

LEGACY_FIGURES = True additionally writes the original outputs: the raw-fiber
overview, the single-fiber autocorrelation-HR figure (with model and PPG-strap
overlays), the models_autocorr_hr comparison of the same two tracks, and
hr_results.npz.

Run:  poetry run python sayo_data_processing.py [session-name]
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, correlate, find_peaks, hilbert, sosfiltfilt

from analyze.constants import (
    FETAL_ACOUSTIC_BAND_HZ, FETAL_BPM_RANGE, MATERNAL_BPM_RANGE, PROJECT_DIR,
    XCORR_TARGET_FS,
)
from analyze.data import Audio
from analyze.filters import bp_filter

# ---------------------------------------------------------------------------
# SELECTION -- edit these
# ---------------------------------------------------------------------------

SESSION = "sayo_long"        # .out/rtmon/sessions/<SESSION>; argv[1] overrides
WINDOW = None                # (start_s, end_s) crop, or None for the whole recording

# Model: family "funet" | "tslnet" (stacked fibers -> beat activity), "ssnet"
# (one fiber -> separated heart -> detector), or "bandpass" (checkpoint-free:
# fibers -> BANDPASS_BAND envelope average). The MODEL_* selection itself only
# feeds the legacy figures (None skips it); MODEL_BPM_RANGE gates every
# track's beats.
MODEL_FAMILY = "ssnet"
MODEL_VERSION = "tuned-model-v13"
MODEL_FIBERS = ["1A"]    # inputs, in training order; count must match the checkpoint
MODEL_BPM_RANGE = FETAL_BPM_RANGE    # plausible-rate gate for the model's beats
SSNET_DETECTOR = "v9_beat_detector"  # beat detector for the ssnet path (what the rig ran)
MODEL_HR_SMOOTH = 10                 # moving-average width (beats) for the model HR line; 1 = raw
BANDPASS_BAND = (190.0, 220.0)         # Hz -- the "bandpass" family's band, up for selection.
                                     # (20,60) tracks the subject's own pulse on sayo data;
                                     # FETAL_ACOUSTIC_BAND_HZ (190,220) for the fetal band.

# Autocorrelation HR
AC_FIBER = "1A"                      # channel to autocorrelate
AC_BAND = (190.0, 220.0)               # bandpass (Hz) -- TBD. (20,60) tracks the subject's own
                                     # pulse on sayo data; FETAL_ACOUSTIC_BAND_HZ for fetal.
AC_BPM_RANGE = (45.0, 200.0)         # lag search range; wide enough to see either rhythm
AC_ENVELOPE = True                   # autocorrelate the band's amplitude envelope (see docstring)
AC_ENV_LP_HZ = 10.0                  # envelope smoothing lowpass
AC_WINDOW_S = 15.0                   # seconds of signal per HR estimate
AC_STEP_S = 5.0                      # hop between estimates
AC_MIN_SCORE = 0.3                   # windows scoring below this are drawn but not connected
AC_SUBHARMONIC_FIX = 0.7             # if a peak at half the winning lag is >= this fraction of
                                     # it, take that one (octave-error fix); None disables

SHOW_PPG_REF = True                  # overlay the PPG strap's pulse if recorded and worn

# Two-track HR comparison -- the default output (hr_compare_{ibi,autocorr}.png).
# Track A runs on fiber 1A; track B's fibers (how many and which) are up to you.
# Each track is (family, version, fibers, detector). The detector picks beats
# for ssnet (required) and bandpass (optional -- it runs on the bandpassed
# waveform; None peak-picks the envelope instead); funet/tslnet ignore it.
# Fiber count must match the checkpoint (ssnet: always 1; funet-v30: 2, most
# funet/tslnet: 3 or 5 -- see lib/<family>); bandpass takes any number.
# Templates (versions = checkpoint names under lib/, detectors v1-v9 exist):
#   ("ssnet",    "tuned-model-v13", ["1A"], "v9_beat_detector")    # separator + detector
#   ("ssnet",    "maternal-tuned-model-v2", ["1A"], "v7_beat_detector")
#   ("funet",    "funet-v30", ["2A", "2B"], None)                  # 2-fiber activity model
#   ("funet",    "funet-v36", ["2A", "2B", "2C"], None)            # 3-fiber, training order
#   ("tslnet",   "tslnet-v9", ["2A", "2B", "2C"], None)            # 3-fiber, training order
#   ("bandpass", None, ["1A"], None)                     # BANDPASS_BAND envelope, peak-picked
#   ("bandpass", None, ["1A"], "v9_beat_detector")       # detector on the bandpassed waveform
HR_TRACK_A = ("bandpass", None, ["1A"], "v9_beat_detector")
HR_TRACK_B = ("ssnet", "tuned-model-v13", ["1A"], "v9_beat_detector")

# HR_TRACK_B = ("funet", "funet-v30", ["2A", "2B"], None)
HR_BPM_RANGE = FETAL_BPM_RANGE       # plausible-rate range for both tracks' HR
HR_SMOOTH_S = 15.0                   # moving-average width (s) applied to every HR curve
LEGACY_FIGURES = False               # True: also write the original figures + hr_results.npz

# The legacy models_autocorr_hr figure (LEGACY_FIGURES only) draws the same two
# HR_TRACK_A/B tracks, but scores every window: ssnet tracks autocorrelate their
# detector's beat times as a gaussian-smoothed train; the rest their activity signal.
BEAT_TRAIN_SIGMA_S = 0.05            # gaussian width when beats become an activity series
TRACK_COLORS = ["#2a78d6", "#eb6834", "#3fa45b", "#8a5cc9"]

SESSIONS_DIR = Path(PROJECT_DIR) / ".out/rtmon/sessions"
OUT_DIR = Path(PROJECT_DIR) / "out/sayo_processing/bandpass+ssnet"

# Series colors, matching the rig UI (bandpass track blue, model track orange).
COLOR_AC = "#2a78d6"
COLOR_MODEL = "#eb6834"

# Fallback column layout when a session has no session.json (rtmon.recorder's).
DEFAULT_COLUMNS = {"ps4000": ["1A", "1B"], "ps3000a": ["2A", "2B", "2C", "2D"]}


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_session(session: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load both PicoScope streams of ``session`` as ``{label: (t, x)}``.

    Column labels come from session.json's ``streams`` block (authoritative);
    a session without one gets the recorder's fixed layout. Each channel keeps
    its own device's time axis.
    """
    sdir = SESSIONS_DIR / session
    if not sdir.is_dir():
        raise FileNotFoundError(f"no session at {sdir}")

    meta = {}
    if (sdir / "session.json").is_file():
        meta = json.loads((sdir / "session.json").read_text()).get("streams", {})

    channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for stream, fallback in DEFAULT_COLUMNS.items():
        path = sdir / f"{stream}.npy"
        if not path.is_file():
            print(f"[load] {stream}.npy missing -- skipped")
            continue
        arr = np.load(path)
        names = meta.get(stream, {}).get("columns", fallback)
        if arr.shape[1] != len(names) + 1:
            raise ValueError(f"{path.name} has {arr.shape[1]} columns, "
                             f"expected time + {names}")
        t = arr[:, 0]
        print(f"[load] {path.name}: {arr.shape[0]} rows, {t[-1] - t[0]:.1f} s "
              f"@ {hz_of(t):.0f} Hz -> {names}")
        for k, name in enumerate(names):
            channels[name] = (t, arr[:, k + 1])
    return channels


def crop(channels, window):
    if window is None:
        return channels
    lo, hi = window
    out = {}
    for name, (t, x) in channels.items():
        m = (t >= lo) & (t <= hi)
        out[name] = (t[m], x[m])
    return out


def hz_of(t: np.ndarray) -> float:
    return 1.0 / float(np.median(np.diff(t[: min(t.size, 10000)])))


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

def _pick_device():
    import os
    import torch
    if "RTMON_DEVICE" in os.environ:
        return torch.device(os.environ["RTMON_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_PIPELINE_CACHE: dict = {}


def model_pipeline(channels, family, version, fibers, detector=None):
    """Run one model track and return ``(t, signal, beats)``, cached per track.

    ``signal`` is the track's beat-carrying series on its own (absolute) time
    axis -- ssnet: the separated, narrow-bandpassed heart waveform; funet /
    tslnet: the beat-activity envelope; bandpass: the fibers' averaged
    BANDPASS_BAND amplitude envelope. ``beats`` are absolute beat times
    (ssnet: from ``detector``; bandpass: ``detector`` on the bandpassed
    waveform when given, else envelope peaks; funet/tslnet: peak-picked).
    Mirrors rtmon.processors run over the whole recording at once instead of
    10 s chunks -- the model wrappers window internally.
    """
    key = (family, version, tuple(fibers), detector)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]
    # rtmon pulls in torch at import time, so only reach for it when a model is asked for.
    from rtmon.models import find as find_model
    from rtmon.processors import _load_ssnet, _ssnet_heart, activity_beats, align

    missing = [f for f in fibers if f not in channels]
    if missing:
        raise ValueError(f"model fibers {missing} not in this session (have {sorted(channels)})")

    if family != "bandpass":
        entry = find_model(family, version)
        if entry is None:
            raise KeyError(f"no {family} model named {version!r}")
    if family == "bandpass":
        # Checkpoint-free track: align the fibers onto one grid, bandpass each
        # to BANDPASS_BAND, and average the smoothed amplitude envelopes into
        # one beat-activity series (envelope, not carrier -- see module docstring).
        # Beats: a given ``detector`` runs on the mean bandpassed waveform
        # (bandpass standing in for ssnet's separated heart); with None the
        # envelope is peak-picked like the activity models.
        print(f"[model] bandpass {BANDPASS_BAND[0]:g}-{BANDPASS_BAND[1]:g} Hz on {fibers}"
              + (f" + {detector}" if detector else "") + " ...")
        grid, stack = align([channels[f] for f in fibers])
        if not grid.size:
            raise ValueError(f"fibers {fibers} do not overlap in time")
        hz = hz_of(grid)
        sos = butter(3, AC_ENV_LP_HZ, fs=hz, btype="low", output="sos")
        waves, envs = [], []
        for row in stack:
            filt = bp_filter(Audio(grid - grid[0], hz, row), *BANDPASS_BAND)
            waves.append(filt.data)
            env = np.abs(hilbert(filt.data, next_fast_len(filt.data.size))[:filt.data.size])
            envs.append(sosfiltfilt(sos, env))
        t_sig, sig = grid, np.mean(envs, axis=0)
        if detector:
            from beat_app.detectors import run_detector
            mean_wave = Audio(grid - grid[0], hz, np.mean(waves, axis=0))
            beats = run_detector(detector, mean_wave, MODEL_BPM_RANGE) + grid[0]
        else:
            beats = activity_beats(sig, t_sig, MODEL_BPM_RANGE)
    elif family == "ssnet":
        # One fiber -> separated heart waveform -> narrow bandpass -> detector.
        # _load_ssnet runs on rtmon's device: cpu unless RTMON_DEVICE says otherwise.
        from beat_app.detectors import run_detector
        from rtmon.processors import BANDS, device as _rtmon_device
        print(f"[model] {version} on {fibers} ({_rtmon_device().type}) ...")
        t, x = channels[fibers[0]]
        hz = hz_of(t)
        wide, narrow = BANDS["fetal"]["acoustic"], BANDS["fetal"]["narrow"]
        fiber = bp_filter(Audio(t - t[0], hz, np.asarray(x, float)), *wide, filter_type="butter")
        heart = _ssnet_heart(_load_ssnet(entry), fiber.data, hz)
        separated = bp_filter(Audio(fiber.time, hz, heart), *narrow, filter_type="butter")
        t_sig, sig = separated.time + t[0], separated.data
        beats = run_detector(detector or SSNET_DETECTOR, separated, MODEL_BPM_RANGE) + t[0]
    else:
        device = _pick_device()
        print(f"[model] {version} on {fibers} ({device.type}) ...")
        series = [channels[f] for f in fibers]
        grid, stack = align(series)
        if stack.shape[0] != entry.channels:
            raise ValueError(f"{version} expects {entry.channels} fiber(s), "
                             f"got {stack.shape[0]} ({', '.join(fibers)})")
        if family == "funet":
            from funet.config import load_config
            from funet.inference import load_funet as load, run_funet as infer
        else:
            from tslnet.config import load_config
            from tslnet.inference import load_tslnet as load, run_tslnet as infer
        config = load_config(entry.config)
        model = load(config, entry.checkpoint, device)
        activity = np.asarray(
            infer(stack.astype(np.float32), int(round(hz_of(grid))), model, config, device),
            dtype=float)
        n = min(activity.size, grid.size)
        t_sig, sig = grid[:n], activity[:n]
        beats = activity_beats(sig, t_sig, MODEL_BPM_RANGE)

    print(f"[model] {beats.size} beats"
          + (f", {60.0 / np.mean(np.diff(beats)):.1f} bpm mean" if beats.size > 1 else ""))
    result = (t_sig, sig, beats)
    _PIPELINE_CACHE[key] = result
    return result


def run_model(channels) -> np.ndarray | None:
    """Beat times from the single MODEL_* selection, for the main figure."""
    if MODEL_FAMILY is None:
        return None
    return model_pipeline(channels, MODEL_FAMILY, MODEL_VERSION, MODEL_FIBERS,
                          SSNET_DETECTOR if MODEL_FAMILY == "ssnet" else None)[2]


def inst_hr(beats: np.ndarray, bpm_range, smooth: int):
    """Instantaneous HR (60/IBI) at each pair's second beat, clipped and smoothed."""
    beats = np.sort(np.asarray(beats, float))
    if beats.size < 2:
        return np.array([]), np.array([])
    bpm = np.clip(60.0 / np.diff(beats), *bpm_range)
    if smooth > 1 and bpm.size > 1:
        bpm = uniform_filter1d(bpm, size=min(smooth, bpm.size))
    return beats[1:], bpm


def track_label(family, version, fibers, detector=None):
    """Legend label for a track: fibers · version-or-band, + detector if one runs."""
    core = (f"bandpass {BANDPASS_BAND[0]:g}-{BANDPASS_BAND[1]:g} Hz"
            if family == "bandpass" else str(version))
    return f"{'+'.join(fibers)} · {core}" + (f" + {detector}" if detector else "")


# ---------------------------------------------------------------------------
# 3. Autocorrelation HR
# ---------------------------------------------------------------------------

def ac_prepare(t, x):
    """Bandpass AC_FIBER and reduce it to the series the autocorrelation runs on.

    Returns ``(filtered, t_ac, x_ac, hz_ac)``: the bandpassed Audio (for
    plotting) plus the analysis series -- the smoothed amplitude envelope
    decimated to ~XCORR_TARGET_FS when AC_ENVELOPE, else the carrier itself.
    """
    hz = hz_of(t)
    filtered = bp_filter(Audio(t, hz, np.asarray(x, float)), *AC_BAND)
    if not AC_ENVELOPE:
        return filtered, t, filtered.data, hz
    n = filtered.data.size
    env = np.abs(hilbert(filtered.data, next_fast_len(n))[:n])
    sos = butter(3, AC_ENV_LP_HZ, fs=hz, btype="low", output="sos")
    env = sosfiltfilt(sos, env)
    dec = max(1, int(round(hz / XCORR_TARGET_FS)))
    return filtered, t[::dec], env[::dec], hz / dec


def autocorr_hr(t, x, hz, bpm, win_s, step_s):
    """HR by windowed autocorrelation of the series ``(t, x)`` at rate ``hz``.

    Per window: unbiased autocorrelation coefficient, then the highest local
    peak among lags inside the plausible IBI range [60/bpm_hi, 60/bpm_lo],
    refined by parabolic interpolation. Returns (t_mid, bpm_est, score);
    score is that peak's value (0..1, 1 = perfectly periodic), NaN bpm where
    no peak exists. Same measurement as rtmon.processors.ppg_periodicity,
    keeping the lag rather than just the maximum.
    """
    win, step = int(round(win_s * hz)), int(round(step_s * hz))
    lag_lo = max(1, int(np.ceil(hz * 60.0 / bpm[1])))
    lag_hi = int(np.floor(hz * 60.0 / bpm[0]))

    mids, rates, scores = [], [], []
    for i0 in range(0, x.size - win + 1, step):
        w = x[i0:i0 + win]
        mids.append(float(t[i0 + win // 2]))
        w = w - w.mean()
        rate, score = np.nan, 0.0
        if np.any(w):
            ac = correlate(w, w, mode="full", method="fft")[win - 1:win + lag_hi + 1]
            ac /= (win - np.arange(ac.size))      # unbiased: average over each lag's overlap
            if ac[0] > 0:
                ac /= ac[0]
                seg = ac[lag_lo:lag_hi + 1]
                peaks, _ = find_peaks(seg)
                if peaks.size:
                    b = lag_lo + peaks[np.argmax(seg[peaks])]
                    # A rhythm at IBI T also peaks at lag 2T, and window noise can put
                    # that subharmonic on top -- halving the reported rate. If a nearly
                    # as strong peak sits at half the winning lag, that is the fundamental.
                    if AC_SUBHARMONIC_FIX is not None:
                        target = b / 2.0
                        tol = max(2.0, 0.08 * target)
                        cands = [q for q in lag_lo + peaks
                                 if abs(q - target) <= tol
                                 and ac[q] >= AC_SUBHARMONIC_FIX * ac[b]]
                        if cands:
                            b = min(cands, key=lambda q: abs(q - target))
                    # Parabolic refinement around the integer-lag peak.
                    delta = 0.0
                    if b + 1 < ac.size:
                        y0, y1, y2 = ac[b - 1], ac[b], ac[b + 1]
                        den = y0 - 2.0 * y1 + y2
                        if den < 0:
                            delta = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
                    rate, score = 60.0 * hz / (b + delta), float(ac[b])
        rates.append(rate)
        scores.append(score)
    return np.asarray(mids), np.asarray(rates), np.asarray(scores)


# ---------------------------------------------------------------------------
# 4. PPG reference
# ---------------------------------------------------------------------------

def ppg_reference(session: str, window) -> np.ndarray | None:
    """Pulse beat times from the strap's best PPG channel, or None if absent/unworn."""
    path = SESSIONS_DIR / session / "pvs.npy"
    if not (SHOW_PPG_REF and path.is_file()):
        return None
    from rtmon.processors import PPG_MIN_PERIODICITY, ppg_beats, ppg_periodicity
    arr = np.load(path)
    t = arr[:, 0]
    if window is not None:
        arr = arr[(t >= window[0]) & (t <= window[1])]
        t = arr[:, 0]
    if t.size < 64:
        return None
    hz = 1.0 / float(np.median(np.diff(t)))
    scored = [(ppg_periodicity(arr[:, k], hz, MATERNAL_BPM_RANGE), k)
              for k in (1, 2, 3)]      # PPG0..PPG2; column 4 is ambient
    strength, k = max(scored)
    if strength < PPG_MIN_PERIODICITY:
        print(f"[ppg] best periodicity {strength:.2f} -- strap not worn, no reference")
        return None
    print(f"[ppg] using PPG{k - 1} (periodicity {strength:.2f}) as pulse reference")
    return ppg_beats(arr[:, k], t, hz, MATERNAL_BPM_RANGE)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(channels, out: Path):
    """Every fiber, decimated -- a did-the-recording-work sanity figure."""
    names = sorted(channels)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 1.4 * len(names)),
                             sharex=True, constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        t, x = channels[name]
        ax.plot(t[::10], x[::10], lw=0.3, color="0.4")
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Session time (s)")
    fig.suptitle("Raw fibers (10x decimated)", fontsize=10)
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)
    print(f"[plot] {out / 'overview.png'}")


def plot_hr(prep, ac_t, ac_bpm, ac_score, model_beats, ppg_beat_times, out: Path):
    filtered, env_t, env_x, _ = prep

    fig, (ax_sig, ax_hr, ax_sc) = plt.subplots(
        3, 1, figsize=(16, 4.5), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 2.4, 1]})

    ax_sig.plot(filtered.time[::5], filtered.data[::5], lw=0.3, color="0.7",
                label="bandpassed")
    if AC_ENVELOPE:
        ax_sig.plot(env_t, env_x, lw=0.7, color=COLOR_AC, label="envelope")
        ax_sig.legend(loc="upper right", fontsize=7)
    ax_sig.set_ylabel(AC_FIBER)
    ax_sig.set_title(f"{AC_FIBER} @ {AC_BAND[0]:g}-{AC_BAND[1]:g} Hz", fontsize=9)

    # HR: autocorrelation estimate, each point colored by its correlation score.
    ok = np.isfinite(ac_bpm)
    good = ok & (ac_score >= AC_MIN_SCORE)
    # Connect confident points, but not across a run of unconfident ones.
    gt, gb = ac_t[good], ac_bpm[good]
    if gt.size:
        brk = np.where(np.diff(gt) > 2.5 * AC_STEP_S)[0]
        gt = np.insert(gt, brk + 1, np.nan)
        gb = np.insert(gb, brk + 1, np.nan)
    ax_hr.plot(gt, gb, lw=1.0, color=COLOR_AC, alpha=0.5, zorder=1)
    sc = ax_hr.scatter(ac_t[ok], ac_bpm[ok], c=ac_score[ok], cmap="viridis",
                       vmin=0.0, vmax=1.0, s=22, zorder=2,
                       label=f"autocorr HR ({AC_FIBER})")
    if model_beats is not None and model_beats.size >= 2:
        mt, mbpm = inst_hr(model_beats, MODEL_BPM_RANGE, MODEL_HR_SMOOTH)
        ax_hr.plot(mt, mbpm, lw=1.4, color=COLOR_MODEL,
                   label=track_label(MODEL_FAMILY, MODEL_VERSION, MODEL_FIBERS))
    if ppg_beat_times is not None and ppg_beat_times.size >= 2:
        pt, pbpm = inst_hr(ppg_beat_times, MATERNAL_BPM_RANGE, MODEL_HR_SMOOTH)
        ax_hr.plot(pt, pbpm, lw=1.2, color="0.35", ls="--", label="PPG pulse (ref)")
    lo = min(AC_BPM_RANGE[0], MODEL_BPM_RANGE[0])
    hi = max(AC_BPM_RANGE[1], MODEL_BPM_RANGE[1])
    ax_hr.set_ylim(lo - 10, hi + 10)
    ax_hr.set_ylabel("HR (bpm)")
    ax_hr.legend(loc="upper right", fontsize=8)
    ax_hr.set_title(
        f"HR -- autocorrelation of {'envelope' if AC_ENVELOPE else 'carrier'} "
        f"({AC_WINDOW_S:g} s window / {AC_STEP_S:g} s step), color = correlation score",
        fontsize=9)

    ax_sc.plot(ac_t, ac_score, lw=1.0, color=COLOR_AC)
    ax_sc.axhline(AC_MIN_SCORE, color="0.6", lw=0.8, ls="--")
    ax_sc.set_ylim(0, 1)
    ax_sc.set_ylabel("Corr. score")
    ax_sc.set_xlabel("Session time (s)")

    fig.colorbar(sc, ax=[ax_sig, ax_hr, ax_sc], pad=0.01,
                 label="autocorrelation peak (0-1)")
    fig.suptitle(f"{SESSION} -- median {np.nanmedian(ac_bpm[good]):.0f} bpm, "
                 f"median score {np.median(ac_score[ok]):.2f}" if good.any()
                 else f"{SESSION} -- no autocorr window scored >= {AC_MIN_SCORE}",
                 fontsize=11)
    fig.savefig(out / "hr_autocorr.png", dpi=150)
    plt.close(fig)
    print(f"[plot] {out / 'hr_autocorr.png'}")


def track_activity(t_sig, sig, beats, family):
    """A track's beat-activity series at ~XCORR_TARGET_FS, for autocorrelation.

    Activity models already output one; a detector track's beat times become a
    gaussian-smoothed impulse train, so beat-time jitter widens the
    autocorrelation peak instead of splitting it.
    """
    if family == "ssnet":
        hz = XCORR_TARGET_FS
        grid = np.arange(t_sig[0], t_sig[-1], 1.0 / hz)
        series = np.zeros(grid.size)
        inside = beats[(beats >= grid[0]) & (beats <= grid[-1])]
        series[np.round((inside - grid[0]) * hz).astype(int)] = 1.0
        return grid, gaussian_filter1d(series, BEAT_TRAIN_SIGMA_S * hz), hz
    hz = hz_of(t_sig)
    dec = max(1, int(round(hz / XCORR_TARGET_FS)))
    return t_sig[::dec], uniform_filter1d(sig, dec)[::dec], hz / dec


def plot_model_autocorr_hr(channels, out: Path):
    """HR_TRACK_A/B side by side, each track's HR from the same windowed
    autocorrelation (score in the bottom panel; hollow points score < AC_MIN_SCORE)."""
    fig, (ax_hr, ax_sc) = plt.subplots(
        2, 1, figsize=(16, 4.5), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 1]})
    summary = []
    for (family, version, fibers, detector), color in zip((HR_TRACK_A, HR_TRACK_B), TRACK_COLORS):
        t_sig, sig, beats = model_pipeline(channels, family, version, fibers, detector)
        grid, series, hz = track_activity(t_sig, sig, beats, family)
        ac_t, ac_bpm, ac_score = autocorr_hr(
            grid, series, hz, HR_BPM_RANGE, AC_WINDOW_S, AC_STEP_S)

        label = track_label(family, version, fibers, detector)
        ok = np.isfinite(ac_bpm)
        good = ok & (ac_score >= AC_MIN_SCORE)
        gt, gb = ac_t[good], ac_bpm[good]
        if gt.size:
            brk = np.where(np.diff(gt) > 2.5 * AC_STEP_S)[0]
            gt = np.insert(gt, brk + 1, np.nan)
            gb = np.insert(gb, brk + 1, np.nan)
        ax_hr.plot(gt, gb, lw=1.2, color=color, zorder=2)
        ax_hr.scatter(ac_t[good], ac_bpm[good], s=16, color=color, zorder=3, label=label)
        weak = ok & ~good
        ax_hr.scatter(ac_t[weak], ac_bpm[weak], s=16, facecolors="none",
                      edgecolors=color, lw=0.8, alpha=0.6, zorder=1)
        ax_sc.plot(ac_t, ac_score, lw=1.0, color=color)
        if good.any():
            summary.append(f"{label}: {np.nanmedian(ac_bpm[good]):.0f} bpm "
                           f"({good.sum()}/{ok.sum()} confident)")
        print(f"[compare] {label}: median "
              f"{np.nanmedian(ac_bpm[good]) if good.any() else float('nan'):.1f} bpm, "
              f"{good.sum()}/{ac_t.size} windows >= {AC_MIN_SCORE}")

    ax_hr.set_ylim(HR_BPM_RANGE[0] - 10, HR_BPM_RANGE[1] + 10)
    ax_hr.set_ylabel("HR (bpm)")
    ax_hr.legend(loc="upper right", fontsize=8)
    ax_hr.set_title(
        f"Model HR by autocorrelation ({AC_WINDOW_S:g} s window / {AC_STEP_S:g} s step); "
        f"hollow = score < {AC_MIN_SCORE:g}", fontsize=9)
    ax_sc.axhline(AC_MIN_SCORE, color="0.6", lw=0.8, ls="--")
    ax_sc.set_ylim(0, 1)
    ax_sc.set_ylabel("Corr. score")
    ax_sc.set_xlabel("Session time (s)")
    fig.suptitle(f"{SESSION} -- " + ("   |   ".join(summary) if summary
                                     else "no confident windows"), fontsize=10)
    fig.savefig(out / "models_autocorr_hr.png", dpi=150)
    plt.close(fig)
    print(f"[plot] {out / 'models_autocorr_hr.png'}")


# ---------------------------------------------------------------------------
# Two-track HR comparison (the default output)
# ---------------------------------------------------------------------------

def track_hr_ibi(beats):
    """Smoothed IBI HR: 60/interval at each beat, averaged over ~HR_SMOOTH_S."""
    b = np.sort(np.asarray(beats, float))
    if b.size < 2:
        return np.array([]), np.array([])
    smooth = max(1, int(round(HR_SMOOTH_S / float(np.median(np.diff(b))))))
    return inst_hr(b, HR_BPM_RANGE, smooth)


def track_hr_autocorr(t_sig, sig, beats, family):
    """Smoothed autocorrelation HR of the track's beat activity (finite windows)."""
    grid, series, hz = track_activity(t_sig, sig, beats, family)
    t, bpm, _score = autocorr_hr(grid, series, hz, HR_BPM_RANGE, AC_WINDOW_S, AC_STEP_S)
    ok = np.isfinite(bpm)
    t, bpm = t[ok], bpm[ok]
    smooth = max(1, int(round(HR_SMOOTH_S / AC_STEP_S)))
    if smooth > 1 and bpm.size > 1:
        bpm = uniform_filter1d(bpm, size=min(smooth, bpm.size))
    return t, bpm


def hr_pearson(ta, ya, tb, yb, step=1.0):
    """Pearson r between two HR curves, interpolated onto a common ``step``-s
    grid over their overlap; NaN without overlap or when either is constant."""
    if ta.size < 2 or tb.size < 2:
        return float("nan")
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if hi - lo < 2 * step:
        return float("nan")
    grid = np.arange(lo, hi, step)
    a, b = np.interp(grid, ta, ya), np.interp(grid, tb, yb)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def plot_hr_compare(curves, method, fname, out: Path):
    """Both tracks' smoothed HR over time in one figure, Pearson r printed on it.

    ``curves`` is [(t, bpm, label), (t, bpm, label)].
    """
    (ta, ya, _), (tb, yb, _) = curves
    r = hr_pearson(ta, ya, tb, yb)
    fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    for (t, bpm, label), color in zip(curves, TRACK_COLORS):
        if t.size:
            ax.plot(t, bpm, lw=1.5, color=color, label=label)
        else:
            print(f"[hr] {label}: too few beats/windows for a curve")
    ax.set_ylabel("HR (bpm)")
    ax.set_xlabel("Session time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{SESSION} -- {method}, {HR_SMOOTH_S:g} s smoothing", fontsize=10)
    ax.text(0.015, 0.05, (f"Pearson r = {r:.3f}" if np.isfinite(r)
                          else "Pearson r: n/a (no overlapping curves)"),
            transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.85))
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)
    print(f"[plot] {out / fname} (Pearson r = {r:.3f})")


# ---------------------------------------------------------------------------

def legacy_figures(channels, out: Path):
    """The original outputs: overview, single-fiber AC HR figure with model and
    PPG overlays, models_autocorr_hr comparison of HR_TRACK_A/B, hr_results.npz."""
    # The named data, exactly as recorded (1-family = ps4000, 2-family = ps3000a).
    time1, fiber1A = channels["1A"]
    _, fiber1B = channels["1B"]
    time2, fiber2A = channels["2A"]
    _, fiber2B = channels["2B"]
    _, fiber2C = channels["2C"]
    _, fiber2D = channels["2D"]
    print(f"[data] fiber1A/1B: {fiber1A.size} samples on time1; "
          f"fiber2A-2D: {fiber2A.size} samples on time2")

    plot_overview(channels, out)

    model_beats = run_model(channels)

    prep = ac_prepare(*channels[AC_FIBER])
    _, ac_series_t, ac_series_x, ac_hz = prep
    ac_t, ac_bpm, ac_score = autocorr_hr(
        ac_series_t, ac_series_x, ac_hz, AC_BPM_RANGE, AC_WINDOW_S, AC_STEP_S)
    ok = np.isfinite(ac_bpm) & (ac_score >= AC_MIN_SCORE)
    print(f"[ac] {ok.sum()}/{ac_t.size} windows >= {AC_MIN_SCORE} score; "
          f"median {np.nanmedian(ac_bpm[ok]):.1f} bpm" if ok.any()
          else f"[ac] no window reached score {AC_MIN_SCORE}")

    ppg_beat_times = ppg_reference(SESSION, WINDOW)

    plot_hr(prep, ac_t, ac_bpm, ac_score, model_beats, ppg_beat_times, out)

    plot_model_autocorr_hr(channels, out)

    np.savez(out / "hr_results.npz",
             ac_time=ac_t, ac_bpm=ac_bpm, ac_score=ac_score,
             ac_fiber=AC_FIBER, ac_band=np.asarray(AC_BAND),
             ac_envelope=AC_ENVELOPE,
             model_beats=model_beats if model_beats is not None else np.array([]),
             ppg_beats=ppg_beat_times if ppg_beat_times is not None else np.array([]),
             model=f"{MODEL_VERSION}:{'+'.join(MODEL_FIBERS)}" if MODEL_FAMILY else "")
    print(f"[save] {out / 'hr_results.npz'}")


def main():
    global SESSION
    if len(sys.argv) > 1:
        SESSION = sys.argv[1]
    out = OUT_DIR / SESSION
    out.mkdir(parents=True, exist_ok=True)

    channels = crop(load_session(SESSION), WINDOW)

    if LEGACY_FIGURES:
        legacy_figures(channels, out)

    # The two comparison tracks, each run once (model_pipeline caches), then
    # one figure per HR method: IBI and windowed autocorrelation.
    tracks = (HR_TRACK_A, HR_TRACK_B)
    runs = [model_pipeline(channels, *track) for track in tracks]
    labels = [track_label(*track) for track in tracks]

    ibi_curves = [(*track_hr_ibi(beats), label)
                  for (_t, _sig, beats), label in zip(runs, labels)]
    ac_curves = [(*track_hr_autocorr(t_sig, sig, beats, track[0]), label)
                 for track, (t_sig, sig, beats), label in zip(tracks, runs, labels)]

    plot_hr_compare(ibi_curves, "HR from IBI (60 / inter-beat interval)",
                    "hr_compare_ibi.png", out)
    plot_hr_compare(
        ac_curves,
        f"HR from autocorrelation ({AC_WINDOW_S:g} s window / {AC_STEP_S:g} s step)",
        "hr_compare_autocorr.png", out)


if __name__ == "__main__":
    main()
