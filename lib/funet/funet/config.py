from dataclasses import dataclass, field
from typing import List, Optional

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config


@dataclass(kw_only=True)
class FUNetModelConfig:
    """Architecture plus the input contract.

    n_fft/hop_length live here, not in `data`: they fix the spectrogram the network sees, and
    a checkpoint cannot be loaded or run without them (see stft_output_shape / inference) --
    the same test that puts base_channels here. crop_len stays in `train`, because FUNet is
    fully convolutional and runs on any length divisible by 2**depth.

    freq_crop_hz is a model field by that same test: it changes how many spectrogram rows
    reach the first conv, so a checkpoint trained with a crop cannot be run without it. It is
    deliberately *not* in `data.preprocess` -- that list is shared verbatim by every model,
    and the row range depends on this model's own n_fft.
    """

    channels: int = 4
    dilations: List[int] = field(default_factory=lambda: [1, 1, 1, 2, 2, 4, 4])
    bottleneck_dilation: int = 8
    bottleneck_convs: int = 3    # conv-norm-relu blocks in the bottleneck stack
    codec_convolutions: int = 3  # conv-norm-relu blocks per encoder AND per decoder level
    base_channels: int = 64      # first-level width; every level doubles from here
    dropout: float = 0.0         # Dropout2d p in the bottleneck + deepest enc/dec level; 0 = off
    n_fft: int = 1024
    hop_length: int = 256

    # How the frame-rate activity is filled back onto the sample grid at readout:
    # 'linear' (piecewise straight lines, a corner every hop) or 'pchip' (shape-preserving
    # cubic, continuous first derivative, still non-negative). See
    # common.phases.inference.frames_to_native. Weights are untouched either way -- this is a
    # readout choice, so a checkpoint stays loadable -- but it moves detected beat times, and
    # therefore every HR number downstream. Defaults to 'linear', which is what every existing
    # model under models/ was measured with.
    interpolation: str = "linear"

    # Passband crop: keep only the spectrogram rows spanning [low, high] Hz, discarding the
    # bands `data.preprocess`'s bandpass already emptied (log1p(0) == 0, so an out-of-band row
    # is a constant-zero field every convolution still pays for). None = full height, which is
    # the default so every existing checkpoint under models/ stays loadable and byte-identical.
    # See data.freq_crop_bins for the exact rows this maps to.
    freq_crop_hz: Optional[List[float]] = None

    # Kill switch for the crop, independent of the band above. True restores the pre-crop
    # front-end exactly -- full-height spectrogram, unchanged feasibility check, unchanged
    # mask ordering -- while leaving freq_crop_hz in the file as a record of the band that
    # was being used. Setting freq_crop_hz to null does the same thing but discards that.
    # It is a model field for the same reason the band is: it decides how many rows reach
    # the first conv, so a checkpoint cannot be run without knowing which way it was set.
    disable_freq_crop: bool = False

    # How every 'same'-padded convolution fills outside the feature map -- and, because the two
    # share a cause, which upsampler the decoders use.
    #
    # 'zeros' is the default and reproduces every checkpoint under models/ exactly: zero
    # padding plus ConvTranspose2d decoders. Leaving this key out of a config therefore changes
    # nothing, which is the point -- an existing run must stay loadable.
    #
    # Anything else pads continuously *and* replaces the decoders' ConvTranspose2d with
    # Upsample + Conv2d. Both target the same measured failure: fed all-zero audio, FUNet emits
    # a fixed per-window pattern the detector reads as a confident ~206 bpm -- a boundary
    # artefact recurring every inference window (zero padding), plus a checkerboard oscillation
    # inside it (transposed convolution). A model that invents a plausible heart rate from
    # silence gives no way to tell "no signal" from "a beat", so this is worth an arm.
    #
    # 'reflect' | 'replicate' | 'circular' are torch's other Conv2d modes. 'circular' is a poor
    # fit here -- a window's end is not continuous with its start. 'reflect' additionally
    # requires the pad to be smaller than the feature map; check_feasible rejects a
    # depth/dilation combination that violates it rather than letting the forward pass crash.
    #
    # Changing this invalidates existing weights (the decoders' module layout differs), so it
    # belongs in `model` by the usual test.
    padding_mode: str = "zeros"   # 'zeros' | 'reflect' | 'replicate' | 'circular'


@dataclass(kw_only=True)
class FUNetTrainConfig(TrainConfig):
    """Base knobs plus FUNet's loss options and SpecAugment masks."""

    loss: str = "kldiv"   # 'kldiv' | 'snr' | 'corr' | 'corr_amp' | 'mse' | 'huber'; see task.LOSSES
    amp_weight: float = 0.0          # corr_amp only: weight on the d' peak-contrast term
    amp_beat_threshold: float = 0.1  # corr_amp only: frac of per-item target peak counting as a beat
    # huber only: residual (in fractions of a beat peak, since the target is rescaled to unit
    # peak) beyond which the loss goes linear. Must stay well under 1 or huber degenerates
    # into 0.5 * MSE -- see common.losses.HuberLoss.
    huber_delta: float = 0.1
    # SpecAugment (train-only), max width of one zeroed band per sample; 0 = off. Regularisation,
    # so it sits alongside `augment` rather than in `data`.
    freq_mask: int = 0    # freq bins zeroed
    time_mask: int = 0    # time frames zeroed


# No FUNetDataConfig: common.DataConfig covers train_dir/val_dir/val_fraction/num_workers, and
# nothing about where the snippets live is FUNet-specific.


@dataclass(kw_only=True)
class FUNetConfig(Config):
    model: FUNetModelConfig = field(default_factory=FUNetModelConfig)
    train: FUNetTrainConfig = field(default_factory=FUNetTrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> FUNetConfig:
    """Load a FUNet config. Kept at this signature so external callers (src/analyze/funet.py,
    bin/record_with_realtime_tracking) don't need to know about tasks."""
    from funet.task import FUNetTask   # local import: task.py imports this module
    return _load_config(path, FUNetTask())
