from dataclasses import dataclass, field

from common.config import Config, DataConfig, TrainConfig
from common.config import load_config as _load_config

from pann.model import DEFAULT_CHECKPOINT


@dataclass(kw_only=True)
class PANNModelConfig:
    """Architecture plus the input contract.

    ``hop_length`` and ``time_pool`` live here rather than in ``data`` for the same reason
    FUNet's spectrogram geometry does: together they fix the frame grid the model emits, and a
    checkpoint cannot be run without them.
    """

    channels: int = 3            # abdomen fibers, stacked and run through the backbone each
    checkpoint: str = DEFAULT_CHECKPOINT

    # The control for this model's whole premise. False keeps the architecture but randomises
    # the conv weights, so a run measures what the head can do over a random projection of the
    # same shape. The fixed STFT/mel buffers are loaded in both arms -- they are a transform,
    # not a learned feature, and randomising them would test a different question.
    #
    # Not searched, deliberately: an experiment arm, not a hyperparameter.
    pretrained: bool = True
    backbone_seed: int = 0       # makes the control reproducible; ignored when pretrained

    # STFT stride, in samples of the 4 kHz signal, and therefore the output frame period:
    # 320 -> 80 ms, 256 -> 64 ms (FUNet's). Lower buys finer beat timing at linear cost. Not a
    # weight, so it is free to differ from the 320 Cnn14 was pretrained with.
    hop_length: int = 320
    # Per-block time pooling. 1 keeps full time resolution and is the point of this model;
    # 2 reproduces stock Cnn14 and downsamples time 64x over six blocks, which check_feasible
    # rejects as unable to localise a beat.
    time_pool: int = 1

    head_hidden: int = 256       # bottleneck width of the trainable MLP
    head_layers: int = 2         # Linear layers; 1 = a plain linear probe over frozen features
    dropout: float = 0.0


@dataclass(kw_only=True)
class PANNTrainConfig(TrainConfig):
    """Base knobs plus the loss options shared with FUNet's and TSLNet's beat-activity heads."""

    loss: str = "mse"                # 'snr' | 'corr' | 'corr_amp' | 'mse'; see task.LOSSES
    amp_weight: float = 0.1          # corr_amp only: weight on the d' peak-contrast term
    amp_beat_threshold: float = 0.1  # corr_amp only: frac of per-item target peak counting as a beat


@dataclass(kw_only=True)
class PANNConfig(Config):
    model: PANNModelConfig = field(default_factory=PANNModelConfig)
    train: PANNTrainConfig = field(default_factory=PANNTrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> PANNConfig:
    """Load a PANN config. Same signature as funet.config.load_config so external callers
    (the analyze pipeline) don't need to know about tasks."""
    from pann.task import PANNTask   # local import: task.py imports this module
    return _load_config(path, PANNTask())
