"""FUNet's Task: the only glue between the model and common's phases."""

import copy

from torch import nn

import numpy as np

from common.audio import SAMPLE_RATE
from common.errors import InfeasibleConfig
from common.losses import CorrAmpLoss, CorrelationLoss, HuberLoss, MSELoss, SNRLoss
from common.metrics import FETAL_BPM_RANGE, HRMetrics
from common.optim import OPTIMIZERS
from common.phases.inference import activity_postprocess
from common.task import Task

from funet.config import FUNetConfig
from funet.data import freq_crop_bins, make_dataloader, stft_output_shape
from funet.model import FUNet

# loss name -> (config -> loss module, matching model output head). Factories take the config
# so amplitude-aware losses can read their hyperparameters from it.
#
# Single source of truth for the loss<->head coupling: the train entry point and inference
# used to keep two hand-maintained copies of this mapping, which could drift.
LOSSES = {
    "kldiv":    (lambda cfg: nn.KLDivLoss(reduction="batchmean"), "logprob"),
    "snr":      (lambda cfg: SNRLoss(), "signal"),
    "corr":     (lambda cfg: CorrelationLoss(), "signal"),   # sign-sensitive; fixes the SI-SNR sign-flip
    "corr_amp": (lambda cfg: CorrAmpLoss(amp_weight=cfg.train.amp_weight,   # corr + d' peak-contrast
                                         beat_threshold=cfg.train.amp_beat_threshold), "signal"),
    "mse":      (lambda cfg: MSELoss(), "signal"),   # per-frame regression to a unit-peak comb
    # Same target and readout as mse, but the per-frame gradient is capped at huber_delta, so
    # the worst-placed beats stop dominating the batch. See common.losses.HuberLoss.
    "huber":    (lambda cfg: HuberLoss(delta=cfg.train.huber_delta), "signal"),
}


class FUNetTask(Task):
    name = "funet"
    ConfigType = FUNetConfig
    device_env_vars = ("FUNET_DEVICE",)

    # The passband crop is a deliberate, measured choice about which rows are signal, not a
    # knob to sweep: the band follows from data.preprocess's bandpass, and letting a search
    # wander it would trade a known 8x compute saving for trials that differ in what they can
    # see. Both are inherited from the base config verbatim; common.phases.optimize enforces
    # it per trial (see Task.frozen_fields).
    #
    # interpolation joins these for the same reason: it is a readout decision, and letting a
    # search vary it would let a trial win by picking the readout that localises beats best
    # rather than the model that does.
    frozen_fields = ("model.freq_crop_hz", "model.disable_freq_crop", "model.interpolation",
                     "model.padding_mode")

    # n_fft/hop_length are searched, but they change what the loss *means*, not just the model:
    # data.__getitem__ builds the target comb on the spectrogram's own frame grid, so halving
    # the hop doubles the frame count and spreads the same beats over more (mostly-empty)
    # frames -- mechanically lowering the MSE. Measured on real snippets, an all-zeros model
    # scores 0.116 at hop 64 but 0.139 at hop 256: a 20% "win" for any trial that merely shrinks
    # the hop. The funet-v33 search took exactly that (all 7 top trials chose hop 64, the
    # minimum) and returned a config that beat the hand-tuned one on paper while being worse in
    # practice. Ranking by hr_corr removes the loophole -- BPM traces are in bpm against
    # seconds, so the score does not move with the frame rate -- and declaring the fields here
    # makes the optimize phase enforce that pairing instead of trusting anyone to remember it.
    loss_scale_fields = ("model.n_fft", "model.hop_length")

    # FUNet.forward ends by scoring every (freq, time) cell, softmaxing over freq and summing:
    # `project` is the first module of that collapse, so its input is the last thing that still
    # has a frequency axis -- the map the model is about to compress into its output vector.
    prepool_attr = "project"

    def head_for(self, config) -> str:
        """Which output head this config's loss trains."""
        try:
            return LOSSES[config.train.loss][1]
        except KeyError:
            raise ValueError(
                f"Unknown loss: {config.train.loss!r} (expected one of {list(LOSSES)})") from None

    # ------------------------------------------------------------------ required
    def build_model(self, config) -> FUNet:
        m = config.model
        return FUNet(
            channels=m.channels,
            dilations=m.dilations,
            bottleneck_dilation=m.bottleneck_dilation,
            bottleneck_convs=m.bottleneck_convs,
            codec_convolutions=m.codec_convolutions,
            base_channels=m.base_channels,
            head=self.head_for(config),
            # Inactive under eval(), but dropout>0 shifts Sequential state_dict keys, so the
            # architecture must match the checkpoint's training config to load it.
            dropout=m.dropout,
            # Same story: anything but 'zeros' also swaps the decoders' upsampler, which shifts
            # their Sequential keys. See FUNetModelConfig.padding_mode.
            padding_mode=m.padding_mode,
        )

    def build_loss(self, config):
        head = self.head_for(config)   # validates the name
        print(f"Loss: {config.train.loss} (model head: {head})")
        return LOSSES[config.train.loss][0](config)

    def make_train_loader(self, config):
        return make_dataloader(config, config.data.train_dir, train=True)

    def make_val_loader(self, config):
        return make_dataloader(config, config.data.val_dir, train=False)

    # ------------------------------------------------------------------ optional
    def check_feasible(self, config) -> None:
        """Reject a network deeper than the spectrogram can support.

        FUNet halves freq and time once per level, so it needs 2**depth <= both the freq-bin
        count and the frame count (see FUNet.forward / data.__getitem__). A deep net combined
        with a small n_fft or a large hop violates that -- and so does a narrow
        model.freq_crop_hz, which stft_output_shape has already applied to freq_bins.
        """
        mode = config.model.padding_mode
        if mode not in ("zeros", "reflect", "replicate", "circular"):
            raise InfeasibleConfig(
                f"model.padding_mode must be one of zeros/reflect/replicate/circular, "
                f"got {mode!r}")
        if mode != "zeros":
            print(f"Padding: '{mode}' -- decoders use Upsample+Conv2d instead of "
                  f"ConvTranspose2d, so this cannot load a 'zeros' checkpoint.")
        freq_bins, time_frames = stft_output_shape(config)
        divisor = 2 ** len(config.model.dilations)
        if divisor > freq_bins or divisor > time_frames:
            # The effective crop, not the raw field: disable_freq_crop may have turned it off,
            # and a message naming a band that was not applied sends the reader the wrong way.
            crop = freq_crop_bins(config)
            band = (f" (freq is the {freq_bins} row(s) kept by freq_crop_hz "
                    f"{list(config.model.freq_crop_hz)})" if crop else "")
            raise InfeasibleConfig(
                f"depth {len(config.model.dilations)} needs freq and time >= {divisor}, but "
                f"this config yields freq={freq_bins}, time={time_frames}{band}")

        # After the depth check on purpose: a network too deep for its input floors the maps to
        # zero, and reporting "pads 1 into a 0x0 map" would send the reader after the wrong bug.
        if mode == "reflect":
            problems = self._reflect_pad_problems(config)
            if problems:
                raise InfeasibleConfig(
                    "padding_mode 'reflect' cannot pad wider than the map it pads, but "
                    + "; ".join(problems)
                    + ". Use 'replicate' (no such limit), lower the dilations, or shorten the "
                      "network")

    def _reflect_pad_problems(self, config) -> list:
        """Layers whose 'same' padding would meet or exceed the feature map it pads.

        ``reflect`` mirrors the interior, so it cannot supply more padding than there is map to
        mirror -- torch raises "Padding size should be less than the corresponding input
        dimension". A shallow net never comes close; FUNet's *default* dilations
        ([1,1,1,2,2,4,4] with bottleneck_dilation 8) do, because the bottleneck sees a
        freq/128 grid and asks for 12 rows of padding. Caught here rather than left to explode
        part-way through the first forward pass.

        ``replicate`` and ``circular`` have no such limit, so this only gates ``reflect``.
        """
        freq, time = stft_output_shape(config)
        depth = len(config.model.dilations)
        divisor = 2 ** depth
        freq -= freq % divisor
        time -= time % divisor

        problems = []
        # Encoder level i works on the map halved i times; kernel 3 at dilation d pads d a side.
        # The decoders mirror it, so checking the encoders covers both.
        for i, d in enumerate(config.model.dilations):
            f, t = freq >> i, time >> i
            if d >= min(f, t):
                problems.append(f"encoder/decoder level {i}: dilation {d} pads {d} into a "
                                f"{f}x{t} map")
        # The bottleneck is the tightest: kernel 4, so it pads ceil(3*dilation/2) a side.
        f, t = freq >> depth, time >> depth
        pad = -(-config.model.bottleneck_dilation * 3 // 2)
        if pad >= min(f, t):
            problems.append(f"bottleneck: dilation {config.model.bottleneck_dilation} with "
                            f"kernel 4 pads {pad} into a {f}x{t} map")
        return problems

    def make_input(self, config, x, src_hz):
        """The log1p spectrogram, preprocessed and passband-cropped, for ONE window.

        ``spectrogram_input`` floors only the frequency axis, because ``run_funet`` follows it
        by cutting the time axis into training-sized windows anyway. A single window has no
        such step after it, so the time axis is floored here instead -- exactly as
        ``funet.data.FetalPairs.__getitem__`` floors both axes -- or FUNet.forward rejects the
        input for not halving cleanly at every level.

        Lazy import: funet.inference imports this module.
        """
        from funet.inference import spectrogram_input

        S = spectrogram_input(config, x, src_hz)
        divisor = 2 ** len(config.model.dilations)
        return S[..., :S.shape[-1] - S.shape[-1] % divisor]

    def run_on_waveform(self, config, model, x, src_hz, device=None):
        """Beat activity over a whole recording, via the real inference path.

        Imported here rather than at module scope because funet.inference imports this module:
        at import time the cycle would be unresolvable, at call time it is already loaded.
        """
        from funet.inference import run_funet
        return run_funet(x, src_hz, model, config, device=device)

    def make_val_scorer(self, config):
        """Score validation by HR-trace correlation, along the real inference path.

        The detector is ``analyze.hr.detect_v2.v2_beat_detector`` -- literally the one
        ``analyze.funet_runner`` runs on a deployed model -- so the score cannot drift from
        what the pipeline actually does. HRMetrics supplies the rest of that path
        (``activity_postprocess`` then ``frames_to_native``); see its docstring for why picking
        peaks off the frame grid instead quantises the BPM trace into uselessness.

        Imported lazily and locally: ``analyze`` is the analysis application and pulls in
        matplotlib and the neossnet utils, which a plain ``--objective loss`` training run has
        no business loading. Doing it here also keeps the import out of ``common``, which must
        not depend on the analysis stack.
        """
        postprocess = activity_postprocess(config.train.loss)

        def detect(activity, hz):
            from analyze.data import Audio
            from analyze.hr.detect_v2 import v2_beat_detector
            time = np.arange(activity.size) / hz
            # out=None suppresses the detector's diagnostic PNG, making it pure and in-memory.
            return v2_beat_detector(Audio(time, hz, activity), FETAL_BPM_RANGE, None)["times"]

        # Shared across epochs so each target's beats are detected once, not every epoch.
        reference_beats: dict = {}
        return lambda: HRMetrics(
            detect=detect,
            hop_length=config.model.hop_length,
            sample_rate=SAMPLE_RATE,
            postprocess=postprocess,
            reference_beats=reference_beats,
            interpolation=config.model.interpolation,
        )

    def period_probe(self, config):
        """Expose the detector's period choice and how close the runner-up was.

        Lazy and local for the same reason ``make_val_scorer``'s detector is: ``analyze`` pulls
        in matplotlib and the neossnet utils, which a training run has no business loading.
        """
        from analyze.hr.period import period_probe
        return period_probe

    # ------------------------------------------------------- optimize phase only
    def suggest(self, trial, base):
        """Return a copy of ``base`` with the searched hyperparameters replaced for this trial.

        Only the fields set here are touched; every other field on ``base`` is inherited as-is.
        Keep this in sync with ``searched_fields`` (which serialises the same set back to YAML).
        """
        config = copy.deepcopy(base)
        model, train = config.model, config.train

        # -- Architecture --
        # `dilations` encodes both the network depth (its length) and the per-level dilation,
        # so we sample a depth and then one dilation per level. Depth drives the 2**depth
        # downsampling that must divide the spectrogram (enforced by check_feasible); it is
        # capped at 6 because deeper nets stop fitting the smaller n_fft / larger hops below.
        depth = trial.suggest_int("depth", 3, 6)
        model.dilations = [
            trial.suggest_categorical(f"dilation_l{i}", [1, 2, 4, 8]) for i in range(depth)
        ]
        model.bottleneck_dilation = trial.suggest_categorical("bottleneck_dilation", [1, 2, 4, 8])
        model.bottleneck_convs = trial.suggest_int("bottleneck_convs", 1, 4)
        model.codec_convolutions = trial.suggest_int("codec_convolutions", 1, 4)
        model.base_channels = trial.suggest_categorical("base_channels", [8, 12, 16, 24, 32])
        model.dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # -- Spectrogram (part of the input contract, hence model config) --
        # Only rankable by an objective that does not move with the frame rate; see
        # loss_scale_fields, which the optimize phase enforces.
        model.n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048])
        model.hop_length = trial.suggest_categorical("hop_length", [64, 128, 256, 512])
        # freq_crop_hz / disable_freq_crop stay frozen (see frozen_fields) -- never assign them
        # here. Being stated in Hz, the band needs no per-trial adjustment anyway: whatever
        # n_fft this trial drew resolves it to the right rows (see freq_crop_bins).

        # -- Optimisation --
        train.optimizer = trial.suggest_categorical("optimizer", list(OPTIMIZERS))
        train.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        # Upper bound is 1.0, not 1e-1: the hand-tuned config runs weight_decay 0.2, so a 1e-1
        # ceiling put the best known setting outside the search space entirely.
        train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1.0, log=True)
        # min_lr is the cosine floor, so it is only meaningful below the peak LR. Sampling it
        # as a fraction of learning_rate guarantees min_lr < learning_rate for every trial
        # (sampling the two independently could put the floor above the peak). Only used when
        # the base config's lr_schedule is 'cosine'; harmless otherwise.
        train.min_lr = train.learning_rate * trial.suggest_float("min_lr_frac", 1e-3, 1e-1, log=True)

        # -- SpecAugment (regularisation, hence train config) --
        train.freq_mask = trial.suggest_int("freq_mask", 0, 64)   # max freq bins zeroed; 0 = off
        train.time_mask = trial.suggest_int("time_mask", 0, 8)    # max time frames zeroed; 0 = off

        return config

    def baseline_params(self, base) -> dict:
        """The params reproducing ``base`` under ``suggest``, so the hand-tuned config runs as
        the study's anchor trial. Mirrors ``suggest`` -- keep the two together."""
        model, train = base.model, base.train
        return {
            "depth": len(model.dilations),
            **{f"dilation_l{i}": d for i, d in enumerate(model.dilations)},
            "bottleneck_dilation": model.bottleneck_dilation,
            "bottleneck_convs": model.bottleneck_convs,
            "codec_convolutions": model.codec_convolutions,
            "base_channels": model.base_channels,
            "dropout": model.dropout,
            "n_fft": model.n_fft,
            "hop_length": model.hop_length,
            "optimizer": train.optimizer,
            "learning_rate": train.learning_rate,
            "weight_decay": train.weight_decay,
            # suggest samples the cosine floor as a fraction of the peak LR, so invert that.
            "min_lr_frac": train.min_lr / train.learning_rate,
            "freq_mask": train.freq_mask,
            "time_mask": train.time_mask,
        }

    def searched_fields(self, config) -> dict:
        """The searched fields of ``config``, shaped like the config YAML. Mirrors the set
        ``suggest`` assigns -- keep the two together."""
        return {
            "model": {
                "dilations": config.model.dilations,
                "bottleneck_dilation": config.model.bottleneck_dilation,
                "bottleneck_convs": config.model.bottleneck_convs,
                "codec_convolutions": config.model.codec_convolutions,
                "base_channels": config.model.base_channels,
                "dropout": config.model.dropout,
                # Must be emitted: they are the input contract, so a best-config.yaml carrying
                # the base geometry would not load the checkpoint trained beside it.
                "n_fft": config.model.n_fft,
                "hop_length": config.model.hop_length,
            },
            "train": {
                "optimizer": config.train.optimizer,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "min_lr": config.train.min_lr,
                "freq_mask": config.train.freq_mask,
                "time_mask": config.train.time_mask,
            },
        }
