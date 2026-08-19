"""The waveform front-end and paired dataset for PANNet.

Unlike TSLNet there is no decimation: Cnn14's mel front-end consumes the 4 kHz signal directly,
which is exactly what places the 100-300 Hz fetal band at 800-2400 Hz in a filterbank built for
32 kHz (see ``pann.model``). So a training item is the raw stacked fibers, cropped.

The target is the heart comb pooled onto the model's own frame grid, *centred* the way a
centre-padded STFT frames the signal: frame t covers samples [t*hop - hop/2, t*hop + hop/2).
Pooling rather than resampling is deliberate -- a comb put through an anti-alias lowpass rings
and smears, and beat *timing* is the label.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.audio import (
    SAMPLE_RATE, crop_time, holdout_split, load_wav, snippet_indices,
)
from common.augment import Augmenter
from common.preprocess import Preprocessor


def crop_samples(config) -> int:
    """Length of one training crop, in samples of the 4 kHz signal."""
    return int(round(config.train.crop_len * SAMPLE_RATE))


def model_frames(config) -> int:
    """Frames the model emits for one crop.

    A centre-padded STFT gives ``1 + samples // hop``; six conv blocks then divide the time axis
    by ``time_pool`` each. Kept here (not only on the model) so ``check_feasible`` can reject an
    unusable geometry without downloading 330 MB of weights.
    """
    m = config.model
    return (1 + crop_samples(config) // m.hop_length) // (m.time_pool ** 6)


def pool_to_frames(signal: torch.Tensor, hop: int, frames: int) -> torch.Tensor:
    """Pool a ``(samples,)`` signal onto ``frames`` STFT frames, centred on frame times.

    Frame t is centred at sample ``t*hop``, so the pooling window is offset by half a hop --
    padding the front by ``hop//2`` and then taking non-overlapping windows of ``hop`` reproduces
    exactly that alignment. Getting this wrong shifts every label by up to half a frame, which
    at hop 320 is 40 ms, a tenth of a beat interval.
    """
    need = frames * hop
    padded = F.pad(signal, (hop // 2, max(0, need - hop // 2 - signal.shape[-1])))
    return padded[:need].reshape(frames, hop).mean(dim=-1)


class PANNPairs(Dataset):
    """Paired snippet dataset in the shared training layout: ``{i}_mix.wav`` (multi-channel)
    plus mono ``{i}_heart.wav``.

    mix    -> (channels, samples) at 4 kHz
    target -> (frames,) per-frame beat activity, normalised to sum to 1
    """

    def __init__(self, snippet_dir: str, indices: list, crop_length: int, train: bool,
                 hop_length: int, frames: int, augment=(), preprocess=()):
        self.dir = snippet_dir
        self.indices = indices
        self.crop_length = crop_length
        self.train = train   # train => random crop offset + augmentation; eval => deterministic
        self.hop_length = hop_length
        self.frames = frames
        self.augmenter = Augmenter(augment)
        # Unlike the augmenter, applied to validation too -- see common.preprocess.
        self.preprocessor = Preprocessor(preprocess)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        mix = load_wav(f"{self.dir}/{idx}_mix.wav")
        heart = load_wav(f"{self.dir}/{idx}_heart.wav")

        # One shared offset so mix and target stay aligned, and a fixed length so the frame
        # count (and the default collate) is consistent across the batch.
        mix, heart = crop_time([mix, heart], self.crop_length, random_offset=self.train)
        mix = self.preprocessor(self.augmenter(mix))

        # clamp_min(0) drops the gated/negative lobes; normalising to sum 1 keeps the target on
        # the same scale as FUNet's and TSLNet's, so the shared losses behave identically.
        target = pool_to_frames(heart[0].clamp_min(0), self.hop_length, self.frames)
        target = target / (target.sum() + 1e-12)

        return mix.float(), target.float()


def make_dataloader(config, snippet_dir: str, *, train: bool) -> DataLoader:
    """Build a loader over ``snippet_dir``.

    Two split strategies, as in the other models: a separate ``val_dir`` holds out a whole
    patient (preferred -- no within-patient leakage), and ``val_fraction`` carves the tail off
    ``train_dir`` by index. ``val_fraction`` is used only when set and no ``val_dir`` is given.
    """
    indices = snippet_indices(snippet_dir)
    split_note = ""
    if not config.data.val_dir and config.data.val_fraction > 0:
        train_idx, val_idx = holdout_split(indices, config.data.val_fraction)
        chosen = train_idx if train else val_idx
        split_note = f" -> {len(chosen)}"
    else:
        chosen = indices

    ds = PANNPairs(snippet_dir, chosen, crop_samples(config), train=train,
                   hop_length=config.model.hop_length, frames=model_frames(config),
                   augment=config.train.augment if train else (),
                   preprocess=config.data.preprocess)   # every split, not just train

    print(f"Loaded {len(indices)} snippets from {snippet_dir}{split_note} "
          f"({'train' if train else 'validation'})")
    return DataLoader(ds, batch_size=config.train.batch_size, shuffle=train,
                      num_workers=config.data.num_workers, pin_memory=True)
