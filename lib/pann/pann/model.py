"""PANNet: a frozen PANNs Cnn14 audio backbone under a small trainable head.

The bet is transfer, like TSLNet's -- but from a much closer domain. TimesFM was pretrained on
generic real-world *time series* (traffic, retail, weather); the fiber signal is **acoustic**,
and Cnn14 was pretrained on AudioSet (~1.9M clips, 527 classes) which includes heartbeat and
body sounds. So this is a genuinely different hypothesis, not a rerun of the TSLNet experiment.

Two design decisions carry this model, and both are worth understanding before reading forward:

* **The audio is fed in at 4 kHz through a front-end built for 32 kHz, deliberately.** Cnn14's
  mel filterbank spans 50 Hz - 14 kHz assuming a 32 kHz input. Handing it 4 kHz samples without
  resampling shifts everything up by 8x *in the filterbank's frame of reference*: the 100-300 Hz
  fetal band lands at 800-2400 Hz, in the dense, well-trained middle of the filterbank, instead
  of being crushed into the lowest one or two mel bins. Nothing is resampled and no information
  is created or destroyed -- only the bins the energy lands in change. The cost is that the time
  axis shifts by the same factor: one hop of 320 samples is 10 ms to Cnn14 and 80 ms here.

* **Pooling is frequency-only.** Stock Cnn14 pools (2, 2) after each of its six conv blocks,
  which downsamples time by 64x -- at 80 ms a frame that is a 5.1 s output step, over ten beats
  in a single cell, and useless for saying *when* a beat happened. This model pools (1, 2)
  instead: the 64 mel bins collapse to 1 over six blocks while the time axis is untouched. The
  pretrained weights are indifferent to this, because pooling is not convolution -- every conv
  is 3x3 with padding 1 and sees the same local neighbourhood either way. What changes is the
  *scale* those filters are applied at, which is the price of dense output from a backbone that
  was trained to emit one label per clip.

The backbone is frozen -- ``requires_grad_(False)``, pinned to eval, excluded from the optimiser
(``common.optim.build_optimizer`` filters on ``requires_grad``) and from the checkpoint (see
``state_dict``). Only the head trains, exactly as in TSLNet.

Checkpoint: ``nicofarr/panns_Cnn14`` on the Hugging Face hub, which is the published
``Cnn14_mAP=0.431`` weights re-uploaded as safetensors. Its keys are prefixed ``backbone.``
(the uploader wrapped the module), and that prefix is stripped on load.
"""

import functools
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_CHECKPOINT = "nicofarr/panns_Cnn14"

# Cnn14's front-end geometry, fixed by the published weights: the STFT is stored as a pair of
# (n_fft//2+1, 1, n_fft) convolution kernels and the mel matrix as (n_fft//2+1, mel_bins).
N_FFT = 1024
MEL_BINS = 64
# Channel widths of the six conv blocks. 1 -> 64 -> ... -> 2048, doubling each level.
BLOCK_CHANNELS = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
EMBED_DIM = BLOCK_CHANNELS[-1][1]   # 2048, the per-frame feature width the head reads


class ConvBlock(nn.Module):
    """Cnn14's repeating unit: conv-bn-relu twice, then pooling.

    Named ``conv1/bn1/conv2/bn2`` because those are the key names in the published checkpoint;
    renaming any of them silently orphans the pretrained weights for this block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        # Average pooling, matching Cnn14's pool_type='avg'. Max pooling here would still run
        # but would not be the operation the downstream weights were trained under.
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """The published Cnn14, reimplemented to match the checkpoint's key names exactly.

    Only the parts this model uses are kept: the STFT/mel front-end, ``bn0``, and the six conv
    blocks. ``fc1``/``fc_audioset`` are the AudioSet classifier head and are deliberately absent
    -- they map a globally-pooled clip embedding to 527 tags, which is the opposite of what a
    per-frame beat model wants. Their absence is why ``load_pretrained`` tolerates exactly those
    two keys being unused.

    The STFT and mel filterbank are stored as buffers rather than recomputed with torchaudio:
    the published tensors *are* the transform the network was trained under, and rebuilding them
    from parameters risks a subtly different window or normalisation that the frozen conv stack
    has no chance to adapt to.
    """

    def __init__(self):
        super().__init__()
        # (n_fft//2+1, 1, n_fft) each -- the DFT basis stored as conv kernels, as torchlibrosa
        # writes it. Buffers, not parameters: fixed transforms with nothing to learn.
        self.register_buffer("stft_real", torch.zeros(N_FFT // 2 + 1, 1, N_FFT))
        self.register_buffer("stft_imag", torch.zeros(N_FFT // 2 + 1, 1, N_FFT))
        self.register_buffer("mel_w", torch.zeros(N_FFT // 2 + 1, MEL_BINS))

        self.bn0 = nn.BatchNorm2d(MEL_BINS)
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(i, o) for i, o in BLOCK_CHANNELS])

    # ------------------------------------------------------------------ front-end
    def logmel(self, waveform: torch.Tensor, hop_length: int) -> torch.Tensor:
        """``(batch, samples)`` -> ``(batch, 1, frames, mel)`` log-mel, torchlibrosa's way.

        ``hop_length`` is a stride, not a weight, so it is free to differ from the 320 Cnn14
        trained with -- lowering it buys finer beat timing at linear cost.
        """
        # Reflect-pad by n_fft//2 so frame t is *centred* on sample t*hop (center=True).
        x = F.pad(waveform.unsqueeze(1), (N_FFT // 2, N_FFT // 2), mode="reflect")
        real = F.conv1d(x, self.stft_real, stride=hop_length)
        imag = F.conv1d(x, self.stft_imag, stride=hop_length)
        power = real.pow(2) + imag.pow(2)                    # (batch, freq, frames)
        mel = torch.matmul(power.transpose(1, 2), self.mel_w)  # (batch, frames, mel)
        # power_to_db with amin=1e-10, ref=1.0 -- torchlibrosa's defaults. The clamp is what
        # keeps silent frames from producing -inf and poisoning every downstream batchnorm.
        log_mel = 10.0 * torch.log10(mel.clamp(min=1e-10))
        return log_mel.unsqueeze(1)                          # (batch, 1, frames, mel)

    # --------------------------------------------------------------------- forward
    def forward(self, waveform: torch.Tensor, hop_length: int,
                time_pool: int = 1) -> torch.Tensor:
        """``(batch, samples)`` -> ``(batch, EMBED_DIM, frames // time_pool)``.

        ``time_pool`` is the per-block time pooling factor: 1 keeps full time resolution (the
        default and the point of this model), 2 reproduces stock Cnn14's schedule and costs a
        64x downsample over six blocks.
        """
        x = self.logmel(waveform, hop_length)   # (batch, 1, frames, mel)

        # bn0 normalises per mel bin, so the mel axis is rotated into the channel slot for it
        # and back afterwards -- exactly what Cnn14 does, and why bn0 has MEL_BINS features.
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)

        # Axes are (batch, channels, TIME, MEL), so pool_size is (time, freq). Pooling only the
        # frequency axis is what makes a per-frame output possible; see the module docstring.
        for block in self.conv_blocks:
            x = block(x, pool_size=(time_pool, 2))

        # Six halvings take 64 mel bins to 1; mean() rather than squeeze() so a different
        # MEL_BINS or block count degrades to an average instead of a shape error.
        return x.mean(dim=3)                    # (batch, EMBED_DIM, frames)


def head_mlp(in_features: int, hidden: int, layers: int, dropout: float) -> nn.Sequential:
    """The trainable head: ``layers`` Linear layers mapping frame features -> one activity value.

    Mirrors ``tslnet.model.head_mlp`` deliberately, so the two frozen-backbone models differ in
    their backbone and nothing else. ``layers=1`` is a plain linear probe, the standard baseline
    for a frozen backbone: if it matches a deeper head, the features are already linearly
    separable and depth is not the limitation.

    No activation after the last layer -- the output is a signal, not a rate, and a trailing
    ReLU would zero every frame whose pre-activation went negative and kill its gradient.
    """
    if layers < 1:
        raise ValueError(f"head_layers must be at least 1, got {layers}")
    modules: list[nn.Module] = []
    width = in_features
    for _ in range(layers - 1):
        modules += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    modules.append(nn.Linear(width, 1))
    return nn.Sequential(*modules)


@functools.cache
def _load_backbone(checkpoint: str, pretrained: bool, seed: int) -> Cnn14:
    """Build a Cnn14 and fill it from ``checkpoint``, or randomly. Cached per process.

    With ``pretrained=False`` the architecture is identical but the conv weights are random --
    the control arm for this model's entire premise, the same one ``tslnet.model`` provides.
    Run it. A frozen backbone that does not beat its own random twin is an expensive random
    feature map, and that is a cheap thing to find out.

    The front-end buffers (STFT basis, mel matrix) are loaded in **both** arms. They are fixed
    mathematical transforms, not learned features, so randomising them would test "does a mel
    spectrogram help" rather than "does AudioSet pretraining help", which is not the question.

    Cached because ``build_model`` runs per Optuna trial and per inference call; the backbone is
    frozen, kept in eval, and never checkpointed, so one shared instance cannot be mutated
    between users.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = Cnn14()

    # Always fetch: even the random arm needs the real STFT/mel buffers (and the file is the
    # only place they exist). ~330 MB, cached by huggingface_hub after the first call.
    path = hf_hub_download(checkpoint, "model.safetensors")
    raw = load_file(path)
    # The uploader wrapped Cnn14 in an outer module, so every key carries a 'backbone.' prefix.
    state = {k[len("backbone."):]: v for k, v in raw.items() if k.startswith("backbone.")}

    front_end = {
        "stft_real": state["spectrogram_extractor.stft.conv_real.weight"],
        "stft_imag": state["spectrogram_extractor.stft.conv_imag.weight"],
        "mel_w": state["logmel_extractor.melW"],
    }

    if pretrained:
        weights = {k: v for k, v in state.items()
                   if k.startswith(("bn0.", "conv_block"))}
        # conv_blockN.* -> conv_blocks.{N-1}.* : the published names are 1-based and flat,
        # this module holds them in a ModuleList.
        remapped = {}
        for k, v in weights.items():
            if k.startswith("conv_block"):
                n, rest = k[len("conv_block"):].split(".", 1)
                k = f"conv_blocks.{int(n) - 1}.{rest}"
            remapped[k] = v
        missing, unexpected = model.load_state_dict({**remapped, **front_end}, strict=False)
        # fc1/fc_audioset are intentionally dropped (see Cnn14's docstring); nothing else may be.
        assert not unexpected, f"unexpected keys in {checkpoint}: {unexpected}"
        assert not missing, f"checkpoint {checkpoint} is missing: {missing}"
        print(f"PANNet backbone: '{checkpoint}' loaded (AudioSet-pretrained Cnn14)")
    else:
        # Seeded and fork_rng'd so it neither depends on nor disturbs the RNG driving shuffling
        # and augmentation -- which keeps head-only checkpoints valid for this arm too.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    m.reset_parameters()
        model.load_state_dict(front_end, strict=False)
        print(f"PANNet backbone: *** CONTROL ARM: RANDOM WEIGHTS, seed {seed} *** "
              f"(architecture from '{checkpoint}'; pretrained weights NOT loaded)")

    return model


def load_backbone(checkpoint: str = DEFAULT_CHECKPOINT, pretrained: bool = True,
                  seed: int = 0) -> Cnn14:
    """The cached backbone for ``checkpoint``. Defaults applied here rather than on the cached
    function so ``load_backbone()`` and ``load_backbone(DEFAULT_CHECKPOINT)`` share one entry."""
    return _load_backbone(checkpoint, pretrained, seed)


class PANNet(nn.Module):
    """``(batch, channels, samples)`` waveform -> ``(batch, frames)`` beat activity.

    Each fiber is run through the shared backbone independently and their per-frame embeddings
    are concatenated, so the head sees all channels side by side and can learn which fiber to
    trust. This mirrors ``tslnet.model.TSLNet``; as there, **channel order is part of the input
    contract** -- slot 0 must be the same fiber at inference as during training.
    """

    def __init__(
        self,
        channels: int = 3,
        checkpoint: str = DEFAULT_CHECKPOINT,
        pretrained: bool = True,
        backbone_seed: int = 0,
        hop_length: int = 320,
        time_pool: int = 1,
        head_hidden: int = 256,
        head_layers: int = 2,
        dropout: float = 0.0,
        backbone: Optional[Cnn14] = None,
    ):
        super().__init__()
        self.channels = channels
        self.hop_length = hop_length
        self.time_pool = time_pool

        self.backbone = (backbone if backbone is not None
                         else load_backbone(checkpoint, pretrained, backbone_seed))
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.mlp = head_mlp(channels * EMBED_DIM, head_hidden, head_layers, dropout)

    def frames_for(self, samples: int) -> int:
        """Output frames for an input of ``samples``. Centre-padded STFT gives
        ``1 + samples // hop``, then six blocks each divide time by ``time_pool``."""
        return (1 + samples // self.hop_length) // (self.time_pool ** len(BLOCK_CHANNELS))

    # --------------------------------------------------------------- frozen-backbone glue
    def train(self, mode: bool = True):
        """Train/eval the head only; the backbone stays in eval permanently so its BatchNorm
        running statistics are never updated by this data."""
        super().train(mode)
        self.backbone.eval()
        return self

    def state_dict(self, *args, **kwargs):
        """The head only -- the backbone is frozen and identical to the published checkpoint,
        so writing it would put ~330 MB of unchanged weights into every model_best.pt."""
        full = super().state_dict(*args, **kwargs)
        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        return type(full)((k, v) for k, v in full.items()
                          if not k.startswith(f"{prefix}backbone."))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a head-only checkpoint. Backbone keys are absent by construction, so those are
        the one thing allowed to be missing; everything else is still held to ``strict``. A
        blanket strict=False would let a differently-shaped head load nothing at all and leave
        random weights behind, silently."""
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [k for k in result.missing_keys if not k.startswith("backbone.")]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "head checkpoint does not match this config"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {list(result.unexpected_keys)}" if result.unexpected_keys else "")
                + " -- check model.head_layers / head_hidden / channels against the config "
                  "archived next to the checkpoint")
        return type(result)(missing, result.unexpected_keys)

    # ------------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PANNet expects (batch, channels, samples), got {tuple(x.shape)}")
        batch, channels, samples = x.shape
        if channels != self.channels:
            raise ValueError(
                f"PANNet was built for {self.channels} channel(s) but got {channels}; "
                "config.model.channels must match the fibers being stacked")

        # Fibers ride in the batch dimension: index (b, c) lands at b*channels + c, which the
        # un-fold below relies on.
        with torch.no_grad():   # frozen; also keeps the conv stack's activations off the tape
            feats = self.backbone(x.reshape(batch * channels, samples),
                                  hop_length=self.hop_length, time_pool=self.time_pool)

        frames = feats.shape[-1]
        # -> (batch, frames, channels*EMBED_DIM): every fiber's view of the same frame, side by
        # side, so the head can weight them against each other.
        feats = feats.reshape(batch, channels, EMBED_DIM, frames)
        feats = feats.permute(0, 3, 1, 2).reshape(batch, frames, channels * EMBED_DIM)

        return self.mlp(feats).squeeze(-1)      # (batch, frames)
