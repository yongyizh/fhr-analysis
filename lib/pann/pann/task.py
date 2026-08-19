"""PANNet's Task: the only glue between the model and common's phases.

Same seam FUNet, SSNet and TSLNet implement, so PANNet inherits the training loop, atomic
checkpointing, config archiving, all three LR schedules and the Optuna search unchanged.
"""

import copy

from common.errors import InfeasibleConfig
from common.losses import CorrAmpLoss, CorrelationLoss, MSELoss, SNRLoss
from common.optim import OPTIMIZERS
from common.preprocess import BANDPASS_HZ
from common.task import Task

from pann.config import PANNConfig
from pann.data import make_dataloader, model_frames
from pann.model import BLOCK_CHANNELS, MEL_BINS, PANNet

# loss name -> config -> loss module. No 'kldiv'/head coupling: PANNet emits a raw per-frame
# signal only, so the losses are the affine-invariant subset FUNet and TSLNet share.
LOSSES = {
    "snr":      lambda cfg: SNRLoss(),
    "corr":     lambda cfg: CorrelationLoss(),
    "corr_amp": lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,
                                        beat_threshold=cfg.train.amp_beat_threshold),
    "mse":      lambda cfg: MSELoss(),
}

# A frame is the finest thing the model can place a beat within, so it must be well under one
# beat interval. The fastest plausible fetal rate is ~200 bpm = 0.3 s.
MAX_FRAME_BEAT_FRACTION = 0.5
FASTEST_FETAL_INTERVAL = 60.0 / 200.0   # seconds
TYPICAL_FETAL_BPM = 140.0               # only for reporting how many beats a crop covers

# The 8x shift the 4 kHz-into-32 kHz front-end produces; see pann.model. Used only to report
# where the fetal band actually lands, which is the single most surprising thing about this
# model and worth printing on every run.
PANN_NATIVE_HZ = 32000


class PANNTask(Task):
    name = "pann"
    ConfigType = PANNConfig
    device_env_vars = ("PANN_DEVICE",)

    # ---------------------------------------------------------------- required
    def build_model(self, config) -> PANNet:
        m = config.model
        model = PANNet(
            channels=m.channels,
            checkpoint=m.checkpoint,
            pretrained=m.pretrained,
            backbone_seed=m.backbone_seed,
            hop_length=m.hop_length,
            time_pool=m.time_pool,
            head_hidden=m.head_hidden,
            head_layers=m.head_layers,
            dropout=m.dropout,
        )

        # The dataset builds its target from model_frames(config); if the model disagrees, every
        # label is misaligned with the prediction and the loss is meaningless. Catch it here
        # rather than through a broadcast error deep in the loss.
        declared, actual = model_frames(config), model.frames_for(
            int(round(config.train.crop_len * 4000)))
        if declared != actual:
            raise ValueError(
                f"frame-count mismatch: data.model_frames says {declared}, the model emits "
                f"{actual} for a {config.train.crop_len}s crop -- hop_length/time_pool are "
                "interpreted differently by pann.data and pann.model")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        arm = "pretrained" if m.pretrained else f"RANDOM CONTROL seed {m.backbone_seed}"
        print(f"PANNet: {trainable:,} trainable head params over {frozen:,} frozen "
              f"backbone params ({m.checkpoint}, {arm})")
        return model

    def build_loss(self, config):
        try:
            factory = LOSSES[config.train.loss]
        except KeyError:
            raise ValueError(
                f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None
        print(f"Loss: {config.train.loss}")
        return factory(config)

    def make_train_loader(self, config):
        return make_dataloader(config, config.data.train_dir, train=True)

    def make_val_loader(self, config):
        # With no val_dir the split comes out of train_dir by index (val_fraction);
        # check_feasible has already rejected the case where neither is set.
        return make_dataloader(config, config.data.val_dir or config.data.train_dir, train=False)

    # ---------------------------------------------------------------- optional
    def check_feasible(self, config) -> None:
        """Reject a config that cannot localise a beat, using declared values only -- no
        checkpoint download -- so the optimize phase can prune a bad trial for free."""
        m, data = config.model, config.data

        if m.head_layers < 1:
            raise InfeasibleConfig(
                f"model.head_layers must be at least 1, got {m.head_layers}; 1 is a linear "
                "probe straight from the frame embeddings to the activity")
        if m.hop_length < 1:
            raise InfeasibleConfig(f"model.hop_length must be positive, got {m.hop_length}")
        if m.time_pool < 1:
            raise InfeasibleConfig(f"model.time_pool must be at least 1, got {m.time_pool}")

        if not data.val_dir and not 0 < data.val_fraction < 1:
            raise InfeasibleConfig(
                "no validation split: set data.val_dir (a held-out patient, preferred) or "
                "data.val_fraction in (0, 1) to carve one out of train_dir")

        frames = model_frames(config)
        if frames < 1:
            raise InfeasibleConfig(
                f"a {config.train.crop_len}s crop at hop {m.hop_length} with time_pool "
                f"{m.time_pool} yields {frames} frames; lengthen crop_len, lower hop_length, "
                "or set time_pool to 1")

        # The whole reason this model pools frequency-only. Stock Cnn14's (2,2) schedule lands
        # here and is rejected: 64x on the time axis puts ten beats in one output cell.
        frame_seconds = config.train.crop_len / frames
        if frame_seconds > MAX_FRAME_BEAT_FRACTION * FASTEST_FETAL_INTERVAL:
            raise InfeasibleConfig(
                f"hop_length {m.hop_length} with time_pool {m.time_pool} makes one output frame "
                f"{frame_seconds:.3f}s, over {MAX_FRAME_BEAT_FRACTION:g} of the "
                f"{FASTEST_FETAL_INTERVAL:.2f}s fastest fetal beat interval; adjacent beats "
                "would share a frame. Lower hop_length or set time_pool to 1")

        if "bandpass" not in config.data.preprocess:
            print("WARNING: data.preprocess has no 'bandpass'. Cnn14's mel front-end does no "
                  "band-limiting of its own, so maternal sounds and motion below the fetal "
                  "band reach the model unchanged.")

        # Where the fetal band actually lands. This is the single most counter-intuitive thing
        # about the model, so it is printed on every run rather than left in a docstring.
        shift = PANN_NATIVE_HZ / 4000
        lo, hi = BANDPASS_HZ
        beats = config.train.crop_len * TYPICAL_FETAL_BPM / 60.0
        print(f"Front-end: {MEL_BINS} mel bins over {len(BLOCK_CHANNELS)} conv blocks; "
              f"hop {m.hop_length} -> {frames} frames of {frame_seconds * 1000:.0f} ms; "
              f"{config.train.crop_len}s ~ {beats:.1f} beats")
        print(f"           4 kHz audio into a {PANN_NATIVE_HZ // 1000} kHz filterbank shifts the "
              f"{lo:.0f}-{hi:.0f} Hz fetal band to {lo * shift:.0f}-{hi * shift:.0f} Hz "
              f"as Cnn14 sees it")

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """This trial's searched hyperparameters. The backbone is frozen and never searched --
        one checkpoint, no architecture to vary. What is left is the head's capacity, the frame
        grid, and the usual optimisation knobs. Keep in sync with ``searched_fields``."""
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # 1 (a linear probe) is a real hypothesis here, not a degenerate corner: if the frozen
        # features are already linearly separable, extra layers are capacity to overfit with.
        model.head_layers = trial.suggest_int("head_layers", 1, 4)
        model.head_hidden = trial.suggest_categorical("head_hidden", [64, 128, 256, 512])
        model.dropout = trial.suggest_float("dropout", 0.0, 0.5)
        # Frame period: 64 ms (FUNet's) to 128 ms. time_pool stays 1 -- anything else is
        # rejected by check_feasible at these hops, so searching it only wastes trials.
        model.hop_length = trial.suggest_categorical("hop_length", [256, 320, 512])

        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        # A head over frozen features tolerates a higher LR than a net trained from scratch.
        train.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        # Fraction of the peak LR, so the cosine floor can never land above the peak.
        train.min_lr = train.learning_rate * trial.suggest_float("min_lr_frac", 1e-3, 1e-1, log=True)

        return config

    def searched_fields(self, config) -> dict:
        """The searched fields, shaped like the config YAML. Mirrors ``suggest`` -- keep together."""
        return {
            "model": {
                "head_layers": config.model.head_layers,
                "head_hidden": config.model.head_hidden,
                "dropout": config.model.dropout,
                "hop_length": config.model.hop_length,
            },
            "train": {
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
            },
        }

    def baseline_params(self, base) -> dict:
        """The ``suggest`` parameters reproducing ``base``, enqueued as the study's first trial
        so the search answers "can it beat the config I already have?" rather than reporting a
        winner nobody compared against."""
        m, t = base.model, base.train
        return {
            "head_layers": m.head_layers,
            "head_hidden": m.head_hidden,
            "dropout": m.dropout,
            "hop_length": m.hop_length,
            "optimizer": t.optimizer,
            "learning_rate": t.learning_rate,
            "weight_decay": t.weight_decay,
            "min_lr_frac": max(1e-3, min(1e-1, t.min_lr / t.learning_rate)),
        }
