#!/usr/bin/env python3
"""Build a paired fetal HR + SOT snippet dataset from a yaml window spec.

Yaml format (same as lib/tune-ssnet/training_clips_mono.yaml):

    mode: stereo      # required: 'stereo' (all fibers as channels) or 'mono' (one snippet per fiber)
    length: 10        # snippet length in seconds (omit to use each section whole)
    overlap: 3        # overlap in seconds between adjacent snippets within a section
    impulse_mode: gate  # heart target: 'gate' (gated band-passed fiber) or 'gaussian'
    time_warp: False  # randomly stretch/compress the gaps between beats (default off)
    warp_strength: 12 # max random stretch factor for inter-beat gaps
    warp_pad: 300     # samples left untouched around each beat when warping
    gate_width_ibi_fraction: 0.3  # heartbeat gate half-width as a fraction of the mean IBI
    snap_to_energy: True  # snap beat labels to the fiber's nearby energy peak
    lung: True        # write the NeoSSNet lung + NLMS noise targets (off => only mix + heart)
    mic_beats: False  # when True, prefer hand-marked times from mic_beats.npy (written
                      # beside microphone.wav by the beat-marking app) for patients that
                      # have them, and fall back to the v7 detector for those that don't.
                      # When False, every patient uses the v7 detector.
    data:
      "PT12_1":
        mode: train   # required: routes this patient's snippets to fetal-train/ or fetal-test/
        fibers: ["2A", "2B"]
        sections: ["130-150", "170-178"]

Each "start-end" section is cut into `length`-second snippets stepping by
`length - overlap`. A patient's mode ('train'|'test') routes all of its snippets
to <out-dir>/fetal-train/ or <out-dir>/fetal-test/ -- whole patients are held
out for testing, with no within-patient leakage. In stereo mode all listed
fibers are stacked as channels into one snippet per window; in mono mode each
fiber is its own snippet. A window that runs past the end of the recorded data
(mic or fibers) is dropped along with the rest of that patient's windows, with
a warning giving the actual data end.

Snippets are generated in parallel (--jobs threads). Indices are assigned up
front per split, so a skipped snippet (too few detectable beats) leaves a gap
in the numbering; the training loader keys off filenames, so gaps are harmless.
--no-plots skips the detector and heart-target debug pngs.
"""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from logging import warning
from pathlib import Path

import matplotlib
import numpy as np
import scipy.signal
import torch
import yaml
from scipy.io import wavfile
from scipy.signal import detrend, resample_poly

from analyze.hr.detect_v7 import v7_beat_detector

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory

from analyze.anc import nlms_filter  # noqa: E402
from analyze.data import Audio, load_fibers  # noqa: E402
from analyze.filters import bp_filter  # noqa: E402
# from analyze.hr.detect_v9 import v9_beat_detector  # noqa: E402
from analyze.sot import _moving_avg, _robust_clip  # noqa: E402
from analyze.util import normalize_path  # noqa: E402
from analyze.constants import (  # noqa: E402
    ABDOMEN_FIBER_NAMES, DEFAULT_DATA_DIR, FETAL_ACOUSTIC_BAND_HZ, FETAL_BPM_RANGE,
    MIC_BEATS_FILE, MIC_FILE, NEOSSNET_MODEL_CFG, NEOSSNET_MODEL_HZ, NEOSSNET_MODEL_PATH,
    PVS_FILE,
)
from utils import load_model  # noqa: E402  (top-level name from the lib/neossnet submodule)

WINDOW_IBI_FRACTION = 0.3  # half-window per beat = this * mean IBI
SNAP_TOLERANCE_S = 0.040   # how far a beat may be nudged to the fiber's own energy peak

# pyplot is not thread-safe: every figure is built and saved under this lock.
PLOT_LOCK = threading.Lock()


class NoBeatException(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    mode: str            # 'stereo' | 'mono'
    length: float | None  # None => one snippet per section
    overlap: float
    impulse_mode: str    # 'gate' | 'gaussian'
    time_warp: bool
    warp_strength: float
    warp_pad: int
    gate_width_ibi_fraction: float
    snap_to_energy: bool
    use_mic_beats: bool
    write_lung: bool
    write_plots: bool


@dataclass(eq=False)
class Job:
    split: str           # 'train' | 'test'
    idx: int
    out_dir: Path
    dir_name: str
    window: str          # "start-end" label for the index map
    fiber_label: str
    fibers: list[Audio]  # windowed, one per channel
    sot: Audio           # windowed microphone
    mic_beats: np.ndarray | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mic(path: str) -> Audio:
    fs, arr = wavfile.read(path + MIC_FILE)
    arr = arr if arr.ndim == 1 else arr[:, 0]
    return Audio(np.arange(len(arr)) / float(fs), fs, arr)


def load_ppg(path: str, ppg_col: int = 1) -> Audio:
    """Maternal PPG from pvs.npy. Column 0 is time; ``ppg_col`` picks the signal
    column, the same index analyze_waveforms.py labels as pvs[i].

    Left unfiltered on purpose -- callers that want the 0.7-4 Hz maternal band
    (analyze.sot.load_sot) filter it themselves; the plotting callers want raw.
    """
    pvs = np.load(path + PVS_FILE)
    t = pvs[:, 0].astype(float)
    if not 0 < ppg_col < pvs.shape[1]:
        raise ValueError(
            f"ppg_col={ppg_col} out of range for {path + PVS_FILE}: "
            f"it has {pvs.shape[1]} columns, and column 0 is time"
        )
    hz = round(1.0 / float(np.median(np.diff(t))))
    return Audio(t, hz, pvs[:, ppg_col].astype(float))


def load_mic_beats(path: str) -> np.ndarray:
    """Hand-marked fetal beat times (absolute seconds), sorted, from mic_beats.npy.

    Accepts a plain 1-D array of times, or a 2-D array whose first column is time.
    """
    arr = np.asarray(np.load(path + MIC_BEATS_FILE, allow_pickle=False), dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 1:
        arr = arr[:, 0]
    return np.sort(arr.ravel())


def parse_window(spec: str) -> tuple[float, float]:
    start_s, end_s = spec.split("-")
    return float(start_s), float(end_s)


def snippet_starts(start: float, end: float, length: float, overlap: float) -> list[float]:
    """Start times of fixed-length snippets stepping by (length - overlap);
    a trailing remainder too short for a full snippet is dropped."""
    starts = []
    t = start
    while t + length <= end + 1e-9:
        starts.append(t)
        t += length - overlap
    return starts


def window_audio(audio: Audio, start: float, end: float) -> Audio:
    mask = (audio.time >= start) & (audio.time < end)
    return Audio(audio.time[mask], audio.hz, audio.data[mask])


def spans_window(audio: Audio, length_s: float) -> bool:
    # A full window holds ~length_s * hz samples; allow a couple samples of slack
    # for float boundary rounding.
    return len(audio.data) + 2 >= length_s * audio.hz


def fiber_groups(named_fibers, mode):
    # stereo: one group with every fiber as a channel; mono: one group per fiber.
    return [named_fibers] if mode == "stereo" else [[nf] for nf in named_fibers]


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def resample(X, hz, target):
    g = np.gcd(target, int(hz))
    up, down = target // g, int(hz) // g
    return resample_poly(X, up, down) if up != down else X


def stack_resampled(fibers):
    # All fibers resampled to NEOSSNET_MODEL_HZ and stacked as (channels, time).
    channels = [resample(fiber.data, fiber.hz, NEOSSNET_MODEL_HZ) for fiber in fibers]
    length = min(len(channel) for channel in channels)
    return np.vstack([channel[:length] for channel in channels])


_model_lock = threading.Lock()
_model = None


def neossnet_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = load_model(NEOSSNET_MODEL_PATH, NEOSSNET_MODEL_CFG)
            _model.eval()
    return _model


def lung_sound(X):
    x = np.asarray(X, dtype=float).ravel()
    x = x / (float(np.max(np.abs(x))) + 1e-12)
    batch = torch.tensor(x, dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        output = neossnet_model()(batch)
    return output[0, 1, :].numpy()


def noise_sound(X, heart, lung):
    X = nlms_filter(X, heart, round(0.5 * NEOSSNET_MODEL_HZ))
    X = nlms_filter(X, lung, round(0.5 * NEOSSNET_MODEL_HZ))
    return X


# ---------------------------------------------------------------------------
# Heart target
# ---------------------------------------------------------------------------

def beat_centers(beat_times, start_time, sample_rate):
    return [int(round((beat_time - start_time) * sample_rate)) for beat_time in beat_times]


def fiber_energy(signal, sample_rate, window_s=0.02):
    return _moving_avg(np.abs(signal), max(1, int(round(window_s * sample_rate))))


def snap_centers_to_energy(energy, centers, tolerance):
    # Nudge each beat center to the nearest fiber energy peak within +/-tolerance samples.
    snapped = []
    for center in centers:
        lo, hi = max(0, center - tolerance), min(len(energy), center + tolerance)
        snapped.append(lo + int(np.argmax(energy[lo:hi])) if hi > lo else center)
    return snapped


def beat_gate(centers, X, half_width):
    # Smooth (Hann) window around each beat, zero between beats.
    length = len(X)
    window = np.hanning(2 * half_width)
    gate = np.zeros(length)
    for center in centers:
        start, stop = center - half_width, center + half_width
        win_start, win_stop = 0, 2 * half_width
        if start < 0:
            win_start, start = -start, 0
        if stop > length:
            win_stop, stop = win_stop - (stop - length), length
        if stop <= start:
            continue
        amp = window[win_start:win_stop] / (np.max(np.abs(X[start:stop])) + 1e-12)
        gate[start:stop] = np.maximum(gate[start:stop], amp)
    return gate


def gaussian_heart(length, centers, half_width):
    heart = np.zeros(length)
    sigma = half_width / 4  # tight: pulse essentially decayed to 0 by the window edge
    for center in centers:
        a, b = max(0, center - half_width), min(length, center + half_width)
        x = np.arange(a, b)
        heart[a:b] += np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return heart


def heart_target(beat_evaluator, mix, t, sot, band, s: Settings):
    """Single-channel clean heartbeat, centered on the reference channel's own beats."""
    beat_times = beat_evaluator(sot)["times"]
    if len(beat_times) < 2:  # one beat => empty np.diff => NaN half-width downstream
        raise NoBeatException(f"fewer than 2 beats detected in {t[0]}-{t[-1]}")

    sample_rate = round(1 / (t[1] - t[0]))
    mean_ibi = float(np.mean(np.diff(beat_times)))
    half_width = int(round(mean_ibi * s.gate_width_ibi_fraction * sample_rate))
    filtered = bp_filter(Audio(t, sample_rate, mix[0]), band[0], band[1], filter_type="butter").data

    centers = beat_centers(beat_times, t[0], sample_rate)
    if s.snap_to_energy:
        # Can shift a label off the acoustic onset toward the loudest lobe.
        centers = snap_centers_to_energy(fiber_energy(filtered, sample_rate), centers,
                                         int(round(SNAP_TOLERANCE_S * sample_rate)))

    if s.impulse_mode == "gaussian":
        heart = gaussian_heart(len(mix[0]), centers, half_width)
    else:
        heart = filtered * beat_gate(centers, filtered, half_width)
    return heart, beat_times, half_width


def fetal_detector(plot_dir, idx):
    # Match load_sot's detection signal: detrend + robust-clip before peak finding.
    def detect(sot: Audio):
        prepared = Audio(sot.time, sot.hz, _robust_clip(detrend(sot.data)))
        if plot_dir is None:
            return v7_beat_detector(prepared, FETAL_BPM_RANGE)
        with PLOT_LOCK:  # v7's debug plot uses pyplot
            return v7_beat_detector(prepared, FETAL_BPM_RANGE, plot_dir, tag=f"{idx}_detections")
    return detect


def mic_beats_evaluator(beat_times):
    """Drop-in for ``fetal_detector``: return the hand-marked beat times that
    fall inside the snippet window instead of detecting on the mic."""
    beat_times = np.asarray(beat_times, dtype=float)

    def evaluate(mic_window: Audio):
        t0, t1 = float(mic_window.time[0]), float(mic_window.time[-1])
        return {"times": beat_times[(beat_times >= t0) & (beat_times <= t1)]}

    return evaluate


# ---------------------------------------------------------------------------
# Time warp augmentation
# ---------------------------------------------------------------------------

def time_warp(X, sections, resampler=lambda x, n: scipy.signal.resample(x, n)):
    """
    >>> time_warp(np.array([0, 1, 2, 3, 4, 5]), []).astype(int).tolist()
    [0, 1, 2, 3, 4, 5]
    >>> time_warp(np.array([0, 1, 2, 3, 4, 5]), [(0, 1, 3)]).astype(int).tolist()
    [0, 0, 0, 1, 2, 3, 4, 5]
    >>> time_warp(np.array([0, 1, 2, 3, 4, 5]), [(2, 4, 5)]).astype(int).tolist()
    [0, 1, 2, 2, 2, 2, 2, 4, 5]
    """
    # Each (start, end, new_size) section resamples X[start:end] into new_size samples.
    length = len(X) + sum(new_size - (end - start) for start, end, new_size in sections)
    data = np.zeros(length)
    in_pos = 0
    out_pos = 0
    for start, end, new_size in sections:
        gap = start - in_pos
        data[out_pos:out_pos + gap] = X[in_pos:start]
        out_pos += gap
        data[out_pos:out_pos + new_size] = resampler(X[start:end], new_size)
        out_pos += new_size
        in_pos = end
    data[out_pos:] = X[in_pos:]
    return data


def warp_sections(t, beat_times, padded_half_width, strength):
    # Randomly rescale each inter-beat gap, leaving padded_half_width around beats untouched.
    rng = np.random.default_rng(42)
    sections = []
    i = 0
    for beat_idx in np.searchsorted(t, beat_times):
        start = max(0, beat_idx - padded_half_width)
        end = min(len(t), beat_idx + padded_half_width)
        if start > i:
            scale = max(rng.random() * strength, 0.1)
            sections.append((i, start, round((start - i) * scale)))
        i = end
    return sections


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_mono(path: Path, hz: int, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data, dtype=np.float32)
    x = x - x.mean()  # remove DC
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    wavfile.write(path, hz, x)


def write_multichannel(path: Path, hz: int, channels: np.ndarray) -> None:
    # DC-removed per channel, globally peak-normalised.
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(channels, dtype=np.float32)
    data = data - data.mean(axis=1, keepdims=True)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    wavfile.write(path, hz, data.T)


def beat_lines(axis, beat_times):
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    axis.vlines(beat_times, 0, 1, transform=transform, color="0.3", lw=0.8, ls="--", alpha=0.7)


def plot_heart(path, sot, raw, heart, heart_hz, start_time, beat_times, title):
    # Three-panel waveform: SOT, generated heart, raw fiber; SOT beats marked.
    heart_time = start_time + np.arange(len(heart)) / heart_hz
    with PLOT_LOCK:
        figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(12, 5), sharex=True)

        top.plot(sot.time, sot.data, color="tab:red", lw=0.6)
        top.set_ylabel("SOT")
        top.set_title(title, fontsize=9)

        middle.plot(heart_time, heart, color="tab:green", lw=0.6)
        middle.set_ylabel("generated heart")
        middle.set_xlabel("time (s)")

        bottom.plot(heart_time, raw, color="tab:blue", lw=0.6)
        bottom.set_ylabel("raw")
        bottom.set_xlabel("time (s)")

        for axis in (top, bottom):
            beat_lines(axis, beat_times)
        bottom.set_xlim(float(heart_time[0]), float(heart_time[-1]))

        figure.tight_layout()
        figure.savefig(path, dpi=120)
        plt.close(figure)


def write_snippet(job: Job, s: Settings) -> bool:
    """One mix (mono or multi-channel) plus its mono targets; False if skipped."""
    # Hand-marked beats when this snippet's patient has them, else detect on the mic (v7).
    evaluator = (mic_beats_evaluator(job.mic_beats) if job.mic_beats is not None
                 else fetal_detector(job.out_dir if s.write_plots else None, job.idx))

    mix = stack_resampled(job.fibers)
    start_time = job.fibers[0].time[0]
    t = start_time + np.arange(mix.shape[1]) / NEOSSNET_MODEL_HZ

    try:
        heart, beat_times, half_width = heart_target(
            evaluator, mix, t, job.sot, FETAL_ACOUSTIC_BAND_HZ, s)
    except NoBeatException as error:
        warning(f"skipping {job.split} snippet {job.idx}: {error}")
        return False

    targets = {"heart": heart}
    if s.write_lung:
        reference = mix.mean(axis=0)
        lung = lung_sound(reference)
        targets["lung"] = lung
        targets["noise"] = noise_sound(reference, heart, lung)

    if s.time_warp:
        sections = warp_sections(t, beat_times, half_width + s.warp_pad, s.warp_strength)
        mix = np.stack([time_warp(channel, sections) for channel in mix])
        targets = {name: time_warp(data, sections) for name, data in targets.items()}

    write_multichannel(job.out_dir / f"{job.idx}_mix.wav", NEOSSNET_MODEL_HZ, mix)
    for name, data in targets.items():
        write_mono(job.out_dir / f"{job.idx}_{name}.wav", NEOSSNET_MODEL_HZ, data)
    if s.write_plots:
        plot_heart(job.out_dir / f"{job.idx}_heart.png", job.sot, mix[0], targets["heart"],
                   NEOSSNET_MODEL_HZ, start_time, beat_times, f"{job.out_dir.name} snippet {job.idx}")

    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_settings(cfg: dict, args) -> Settings:
    mode = cfg.get("mode")
    if mode not in ("mono", "stereo"):
        raise ValueError(f"config 'mode' is required and must be 'mono' or 'stereo', got {mode!r}")
    impulse_mode = cfg.get("impulse_mode", "gate")
    if impulse_mode not in ("gaussian", "gate"):
        raise ValueError(f"config 'impulse_mode' must be 'gaussian' or 'gate', got {impulse_mode!r}")
    length = cfg.get("length")
    overlap = cfg.get("overlap", 0.0)
    if length is not None and overlap >= length:
        raise ValueError(f"overlap ({overlap}) must be less than length ({length})")

    return Settings(
        mode=mode,
        length=length,
        overlap=overlap,
        impulse_mode=impulse_mode,
        time_warp=cfg.get("time_warp", False),
        warp_strength=cfg.get("warp_strength", 12),
        warp_pad=cfg.get("warp_pad", 300),
        gate_width_ibi_fraction=cfg.get("gate_width_ibi_fraction", WINDOW_IBI_FRACTION),
        snap_to_energy=cfg.get("snap_to_energy", True),
        use_mic_beats=cfg.get("mic_beats", False),
        write_lung=cfg.get("lung", True),
        write_plots=not args.no_plots,
    )


def require_split_mode(name, patient_spec):
    mode = patient_spec.get("mode")
    if mode not in ("train", "test"):
        raise ValueError(f"patient {name!r} must declare mode: 'train' or 'test' (got {mode!r})")
    return mode


def build_jobs(spec: dict, s: Settings, data_dir: str, out_dir: Path) -> list[Job]:
    """Load each patient and expand its sections into per-snippet jobs with
    pre-assigned indices (contiguous per split, restarting from 0)."""
    split_dirs = {"train": out_dir / "fetal-train", "test": out_dir / "fetal-test"}
    counters = {"train": 0, "test": 0}
    jobs: list[Job] = []

    for dir_name, patient_spec in spec.items():
        split = require_split_mode(dir_name, patient_spec)
        fiber_names = patient_spec["fibers"]
        unknown = [f for f in fiber_names if f not in ABDOMEN_FIBER_NAMES]
        if unknown:
            raise ValueError(
                f"{dir_name}: unknown fiber(s) {unknown}, must be one of {ABDOMEN_FIBER_NAMES}")

        patient_path = normalize_path(f"{data_dir}{dir_name}")
        print(f"Loading {dir_name} ... (-> {split})")
        fibers = load_fibers(Path(patient_path))
        mic = load_mic(patient_path)
        # mic_beats: prefer this patient's hand-marked beats when present; otherwise
        # leave None so write_snippet detects on the mic (v7).
        mic_beats = None
        if s.use_mic_beats:
            if Path(patient_path + MIC_BEATS_FILE).exists():
                mic_beats = load_mic_beats(patient_path)
                print(f"  using {len(mic_beats)} hand-marked beats from {MIC_BEATS_FILE}")
            else:
                print(f"  no {MIC_BEATS_FILE}; falling back to the v7 mic detector")

        data_end = min(float(source.time[-1])
                       for source in [mic, *(fibers.abdomen[name] for name in fiber_names)])

        out_of_data = False
        for window_spec in patient_spec["sections"]:
            start, end = parse_window(window_spec)
            if end <= start:
                warning(f"{dir_name} [{window_spec}] has end <= start; skipping")
                continue

            win_length = s.length if s.length is not None else end - start
            starts = snippet_starts(start, end, win_length, s.overlap)
            if not starts:
                warning(f"{dir_name} [{window_spec}] is shorter than "
                        f"length {win_length}s; no snippets produced")

            for window_start in starts:
                window_end = window_start + win_length
                named = [(name, window_audio(fibers.abdomen[name], window_start, window_end))
                         for name in fiber_names]
                mic_window = window_audio(mic, window_start, window_end)

                if not all(spans_window(w, win_length) for w in (mic_window, *(f for _, f in named))):
                    warning(f"{dir_name}: window {window_start:.3f}-{window_end:.3f} runs past the "
                            f"end of the data ({data_end:.3f}s); dropping it and the rest of this patient")
                    out_of_data = True
                    break

                for group in fiber_groups(named, s.mode):
                    idx = counters[split]
                    counters[split] += 1
                    jobs.append(Job(
                        split=split,
                        idx=idx,
                        out_dir=split_dirs[split],
                        dir_name=dir_name,
                        window=f"{window_start:.3f}-{window_end:.3f}",
                        fiber_label="+".join(name for name, _ in group),
                        fibers=[fiber for _, fiber in group],
                        sot=mic_window,
                        mic_beats=mic_beats,
                    ))
            if out_of_data:
                break
    return jobs


def run_jobs(jobs: list[Job], s: Settings, workers: int) -> list[Job]:
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(write_snippet, job, s): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            if future.result():
                completed.append(job)
                print(f"[{job.split}] snippet {job.idx} ({job.dir_name}) written")
    return sorted(completed, key=lambda job: (job.split, job.idx))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice Banner_data directories into paired fetal HR + SOT wav snippets")
    parser.add_argument("yaml_path", type=Path, help="Window spec yaml (see module docstring)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Base directory containing the patient subdirectories (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output base directory")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="Snippet worker threads (default: CPU count)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip the detector and heart-target debug pngs")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        cfg: dict = yaml.safe_load(f)
    settings = load_settings(cfg, args)
    if settings.use_mic_beats:
        print("----- PREFERRING HANDPICKED SOT BEATS (v7 detector where absent) -----")

    workers = max(1, args.jobs or 1)
    # Torch's intra-op pool is shared across threads; keep workers * torch threads ~ CPUs.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))

    jobs = build_jobs(cfg["data"], settings, normalize_path(args.data_dir), args.out_dir)
    print(f"\nGenerating {len(jobs)} snippets with {workers} threads ...")
    completed = run_jobs(jobs, settings, workers)

    counts = {split: sum(1 for job in completed if job.split == split) for split in ("train", "test")}
    skipped = len(jobs) - len(completed)
    print(f"\nCreated {counts['train']} train + {counts['test']} test snippets in {args.out_dir}/"
          + (f" ({skipped} skipped)" if skipped else ""))
    print("\nIndex map:")
    for job in completed:
        print(f"  [{job.split}] {job.idx}: {job.dir_name} [{job.window}] fiber={job.fiber_label}")


if __name__ == "__main__":
    main()
