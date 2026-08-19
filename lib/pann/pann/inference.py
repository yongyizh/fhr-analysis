"""Run a trained PANNet on a raw waveform and get a beat-activity signal over time.

Mirrors ``funet.inference`` and ``tslnet.inference`` -- same signature, same return contract --
so the analyze pipeline can swap one model for another. The difference is only the front-end:
PANNet hands the 4 kHz waveform straight to Cnn14's mel extractor rather than building its own
spectrogram or decimating.

The model trains on fixed crop_len-second crops, so inference processes the series in equal
windows (matching the training sample count) and stitches the per-window activity back together
-- see common.phases.inference.run_windowed. The frame-rate activity is then mapped onto the
input's own time axis so it lines up sample-for-sample with the source waveform.
"""

import numpy as np
import torch

from common.audio import SAMPLE_RATE, resample
from common.phases.inference import (
    activity_postprocess, frames_to_native, load_model, normalize_blocks, run_windowed,
)
from common.preprocess import Preprocessor

from pann.data import crop_samples
from pann.model import PANNet
from pann.task import PANNTask


def load_pann(config, checkpoint: str, device: torch.device = None) -> PANNet:
    """Build a PANNet matching ``config`` and load head weights from ``checkpoint``.

    ``checkpoint`` is a head-only file (see PANNet.state_dict); the frozen backbone comes from
    ``config.model.checkpoint`` via the Hugging Face cache.
    """
    return load_model(PANNTask(), config, checkpoint, device)


@torch.no_grad()
def run_pann(
        x: np.ndarray,
        src_hz: int,
        model: PANNet,
        config,
        device: torch.device = None,
) -> np.ndarray:
    """Beat-activity over time for waveform ``x`` (``(T,)`` or ``(channels, T)``).

    Returns a non-negative activity signal the same length as ``x`` and sampled at ``src_hz``:
    high where the model thinks a fetal beat occurs. Relative peaks (not absolute scale) carry
    the beats -- under the affine-invariant losses the units are standard deviations above the
    window's own floor.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]                      # (1, T)
    n_native = x.shape[-1]

    channels = x.shape[0]
    if channels != config.model.channels:
        raise ValueError(
            f"waveform has {channels} channel(s) but the model expects "
            f"{config.model.channels} (config.model.channels)")

    # Match training preprocessing: resample to the model rate, then peak-normalise on the same
    # time scale a training snippet was normalised on -- not across the whole recording, which
    # lets one loud transient rescale everything else (see normalize_blocks).
    x = resample(x, src_hz, SAMPLE_RATE)
    window = crop_samples(config)
    x = normalize_blocks(x, window)

    # The same deterministic transforms the dataset applied, at the same rate, or the model
    # meets an input distribution it never trained on (see common.preprocess). This is also
    # where band-limiting happens: Cnn14's front-end does none of its own.
    waveform = Preprocessor(config.data.preprocess)(
        torch.from_numpy(np.ascontiguousarray(x)))

    # How a window of raw output becomes activity depends on what the loss pinned down, so the
    # readout is chosen from the loss rather than fixed here. Inference-only; training
    # optimizes the raw head output.
    postprocess = activity_postprocess(config.train.loss)

    activity = run_windowed(model, waveform, window, device=device, postprocess=postprocess)

    # Frame t is centred at sample t*hop of the 4 kHz signal, so that is the hop that maps the
    # result back onto the source waveform's own time axis.
    hop = config.model.hop_length * (config.model.time_pool ** 6)
    return frames_to_native(activity, hop, SAMPLE_RATE, n_native, src_hz)
