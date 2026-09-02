"""The inference phase: load a trained checkpoint and run it over a long input.

Models here train on fixed-length crops, so inference processes a long recording in equal
windows matching the training extent (GroupNorm and friends then see the statistics they
trained on) and stitches the per-window output back together.

What stays in the task: how a waveform becomes model input, and how a window of raw output
becomes an activity value. What lives here: loading, windowing, and mapping a frame-rate
result back onto the input's own time axis. This module must never import optuna.
"""

from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


# Loss name -> how to read that model's raw output as a non-negative activity envelope.
#
# Keyed by loss, because the loss is what does (or does not) pin the output's scale, and the
# postprocess has to be invariant to exactly what the loss was invariant to:
#
#   kldiv       the model already emitted log-probabilities, so exp is the real distribution.
#   mse/huber   regressed to a unit-peak comb, so the scale is calibrated -- keep it, just
#               drop the sub-zero floor. Huber differs from MSE only in how it charges a
#               large residual, so it pins the same scale and reads out the same way.
#   snr/corr/   all affine-invariant (CorrelationLoss divides by both norms; CorrAmpLoss's d'
#   corr_amp    is documented as invariant to affine scaling of the output). Nothing in
#               training pins the scale, so the readout must not care about it either.
#
# This table replaced a blanket softmax for every signal-head loss, which was the worst
# possible choice for the affine-invariant ones: softmax is shift-invariant but responds to
# scale *exponentially*, so an output that drifted to a large scale collapsed to a near
# argmax. A perfect 16-beat window came out as 3 spikes at scale 16 and 8 at scale 8, while
# the scale itself was free to be anything.
ACTIVITY_POSTPROCESS = {
    "kldiv":    "exp",
    "mse":      "clamp",
    "huber":    "clamp",
    "snr":      "standardize",
    "corr":     "standardize",
    "corr_amp": "standardize",
}


def activity_postprocess(loss: str, eps: float = 1e-8) -> Callable[[torch.Tensor], torch.Tensor]:
    """The per-window readout for a model trained under ``loss`` (see ACTIVITY_POSTPROCESS).

    Every variant returns a non-negative envelope, which is the contract the beat detectors
    downstream expect; only relative peak height carries information.
    """
    try:
        kind = ACTIVITY_POSTPROCESS[loss]
    except KeyError:
        raise ValueError(
            f"no activity readout for loss {loss!r} (known: {sorted(ACTIVITY_POSTPROCESS)})"
        ) from None

    if kind == "exp":
        return lambda out: out.exp()
    if kind == "clamp":
        return lambda out: out.clamp_min(0)

    # standardize: scale- and shift-invariant, matching the loss. The clamp drops the
    # below-mean floor, leaving peaks measured in standard deviations above it. A constant
    # window yields all zeros rather than a NaN.
    def standardize(out: torch.Tensor) -> torch.Tensor:
        return ((out - out.mean()) / (out.std() + eps)).clamp_min(0)

    return standardize


def normalize_blocks(x: np.ndarray, block_samples: int, eps: float = 1e-12) -> np.ndarray:
    """Peak-normalise a ``(channels, time)`` waveform in blocks of ``block_samples``.

    Training normalises per *snippet* (generate_training_snippets.write_multichannel peak-
    normalises each one), so a model only ever saw waveforms scaled by a local peak. Scaling a
    whole recording by its single global peak -- which is what inference used to do -- lets one
    loud transient shrink everything else: a 5x gap on a recording with one burst, which
    survives into the input because the log1p envelope is nonlinear, so the backbone's own
    z-scoring cannot undo it.

    The gain is interpolated between block centres rather than applied per block. A hard block
    boundary is a step in amplitude, and a step is broadband -- inside a 100-300 Hz passband it
    would look like a beat, manufacturing one false peak every block.

    One peak across all channels, as in training: relative fiber loudness is signal.
    """
    # Coerced because a caller deriving this from a float crop_len (TSLNet's is 2.56 s) would
    # otherwise hand in 10240.0 and fail on the slice indices below, several lines later.
    block_samples = int(block_samples)

    total = x.shape[-1]
    if block_samples <= 0 or total <= block_samples:
        return x / (float(np.max(np.abs(x))) + eps)

    starts = np.arange(0, total, block_samples)
    peaks = np.array([float(np.max(np.abs(x[..., s:s + block_samples]))) for s in starts])
    centres = np.minimum(starts + block_samples / 2.0, total - 1)

    gains = np.interp(np.arange(total), centres, 1.0 / (peaks + eps))
    return (x * gains).astype(np.float32, copy=False)


def load_model(task, config, checkpoint: str, device: Optional[torch.device] = None) -> nn.Module:
    """Build the model ``config`` describes and load ``checkpoint`` into it, in eval mode."""
    device = device or torch.device("cpu")
    model = task.build_model(config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def run_windowed(
        model: nn.Module,
        x: torch.Tensor,
        window: int,
        *,
        device: Optional[torch.device] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> np.ndarray:
    """Run ``model`` over ``x`` in consecutive ``window``-sized slices of its last axis.

    ``x`` is a single unbatched example, e.g. ``(channels, freq, frames)`` or
    ``(channels, samples)``; the model is expected to map a batch of those to
    ``(batch, window)``. Returns the stitched ``(len,)`` result, trimmed back to ``x``'s
    original length.

    The last axis is padded up to a whole number of windows first, so EVERY window is exactly
    ``window`` wide: a short final window is out-of-distribution (fewer frames shift a
    normalisation layer's per-sample statistics), which visibly inflates the tail of the
    output. Padding reflects so the boundary looks like signal continuing, falling back to
    zeros when the input is too short to reflect that far.
    """
    device = device or next(model.parameters()).device
    total = x.shape[-1]

    pad = (-total) % window
    if pad:
        mode = "reflect" if total > pad else "constant"
        x = torch.nn.functional.pad(x, (0, pad), mode=mode)
    padded = total + pad

    out = np.zeros(padded, dtype=np.float32)
    x = x.to(device)
    for start in range(0, padded, window):
        chunk = x[..., start:start + window]          # always exactly `window` wide
        y = model(chunk.unsqueeze(0))[0]              # (window,)
        if postprocess is not None:
            y = postprocess(y)
        out[start:start + window] = y.cpu().numpy()

    return out[:total]


#: How ``frames_to_native`` fills the gap between frames. Neither adds information -- one
#: frame every ``hop_length`` samples is all there is -- they differ in what they assume lies
#: between: straight lines, or a smooth shape-preserving curve.
INTERPOLATIONS = ("linear", "pchip")


def frames_to_native(
        activity: np.ndarray,
        hop_length: int,
        model_hz: int,
        n_native: int,
        src_hz: int,
        interpolation: str = "linear",
) -> np.ndarray:
    """Map a frame-rate signal onto the input's own sample grid.

    Frame ``t`` is centred at sample ``t * hop_length`` of the ``model_hz`` signal; the result
    is ``n_native`` samples at ``src_hz``, so it lines up with the source waveform.

    ``linear`` (the default, and what every model shipped with) joins consecutive frames with
    straight lines. The result is a polyline: at hop 256 that is one corner every 64 ms, and
    255 of every 256 samples are redraw.

    ``pchip`` fits a shape-preserving cubic instead, giving a continuous first derivative. It
    is chosen over a bandlimited (sinc) resample deliberately: sinc rings, and its undershoot
    would push a beat envelope negative between beats, which breaks the non-negativity every
    downstream beat detector assumes (see ACTIVITY_POSTPROCESS). PCHIP cannot overshoot the
    surrounding frame values, so non-negative input stays non-negative.

    Both clamp outside the frame range rather than extrapolating: ``n_native`` covers up to
    ``hop_length`` samples past the last frame centre, and a cubic let loose there can swing
    far away from the data.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(
            f"Unknown interpolation: {interpolation!r} (expected one of {list(INTERPOLATIONS)})")

    frame_times = np.arange(len(activity)) * hop_length / model_hz
    native_times = np.arange(n_native) / src_hz

    # PCHIP needs at least two points to define a curve; with fewer, linear (which degrades to
    # a constant) is the only sensible reading anyway.
    if interpolation == "linear" or len(activity) < 2:
        return np.interp(native_times, frame_times, activity).astype(np.float32)

    from scipy.interpolate import PchipInterpolator
    curve = PchipInterpolator(frame_times, activity, extrapolate=False)
    clamped = np.clip(native_times, frame_times[0], frame_times[-1])
    return curve(clamped).astype(np.float32)
