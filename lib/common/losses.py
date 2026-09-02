"""Beat-activity losses, shared by every model that predicts a per-frame activity signal.

All of these take ``(batch, time)`` output and target and are lower-is-better, which is what
lets the optimize phase minimise validation loss directly. Nothing here knows which model
produced the output, so a new beat-activity model gets the whole set for free -- the same
reason the training loop and config loader live in ``common``.

Two families, and the distinction matters when picking one:

* **shape** objectives (``CorrelationLoss``, ``SNRLoss``) are scale-invariant -- they score
  *where* the beats are and ignore absolute amplitude.
* **reconstruction** objectives (``MSELoss``, ``HuberLoss``) pin an absolute scale.

``CorrAmpLoss`` sits between: correlation for placement plus a d' term for contrast.

Which head a loss needs is the model's business, not this module's: ``kldiv`` wants
log-probabilities, the rest want a raw signal. Each task's ``LOSSES`` table records that
coupling (see ``funet.task`` / ``tslnet.task``).
"""

import torch
from torch import nn
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio


class SNRLoss(nn.Module):
    """Negative SI-SNR (scale-invariant signal-to-distortion ratio), the SNR loss
    that worked best in neossnet (see lib/neossnet/loss_fn.py SNR_Loss).

    Treats the model output as a signal estimate of the target comb rather than a
    probability distribution. SI-SNR is scale-invariant, so the target staying
    normalized (sum to 1) is fine - only its shape matters. Expects output/target
    of shape (batch, time); returns the negated mean SI-SNR over the batch.
    """

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        snr = scale_invariant_signal_distortion_ratio(preds=output, target=target, zero_mean=False)
        return -snr.mean()


class CorrelationLoss(nn.Module):
    """1 - Pearson correlation between output and target (per item, averaged over batch).

    Like SI-SNR it is scale-invariant -- only the *shape* of the output matters, which
    is what worked well for signal-regression here. Unlike SI-SDR it is NOT sign-
    invariant: maximizing correlation forces the beats to be positive peaks, so the
    model can't settle on the negated comb (SI-SDR could, which clamp_min(0) at
    inference then discarded). Expects output/target of shape (batch, time).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        o = output - output.mean(dim=-1, keepdim=True)
        t = target - target.mean(dim=-1, keepdim=True)
        corr = (o * t).sum(dim=-1) / (o.norm(dim=-1) * t.norm(dim=-1) + self.eps)
        return (1 - corr).mean()


class MSELoss(nn.Module):
    """Per-frame MSE against a unit-peak comb -- a reconstruction objective (contrast with
    corr/snr, which are scale-invariant shape objectives).

    MSE pins absolute amplitude: beat frames are driven to 1, floor frames to 0. The
    dataset target is sum-normalized, so its peak height depends on how many beats fall
    in the window -- an inconsistent regression target. Each item's comb is therefore
    rescaled to unit peak before the MSE, giving a window-independent beats->1/floor->0
    target.

    Caveats (see corr vs MSE discussion): it pins an absolute scale that the signal-head
    inference then softmax-normalizes away (relative peaks survive, calibration is partly
    unused), and it penalizes overshoot symmetrically (a beat at 1.4 is as "wrong" as
    0.6), so unlike the d' term in corr_amp it does not push for emphatic peaks. To MSE
    against the raw sum-normalized target instead, drop the rescale line.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Rescale to unit peak per item; clamp_min guards an all-zero (silent) target.
        target = target / target.amax(dim=-1, keepdim=True).clamp_min(self.eps)
        return nn.functional.mse_loss(output, target)


class HuberLoss(nn.Module):
    """Huber regression against the same unit-peak comb ``MSELoss`` uses: quadratic while the
    residual is inside ``delta``, linear beyond it.

    Same objective and same readout as MSE (see ``MSELoss`` for why the target is rescaled to
    unit peak), differing only in how a large residual is charged. MSE weights one frame that
    is off by 0.8 as heavily as sixteen frames off by 0.2, so the gradient is dominated by the
    worst frames -- which here are systematically the *beat* frames (the ones the target drives
    to 1, and the ones a mistimed or mislabelled beat puts in the wrong place). Huber caps the
    per-frame gradient at ``delta``, so a handful of badly-placed beats can no longer outvote
    the rest of the window.

    ``delta`` is in the units of the rescaled target, i.e. fractions of a beat peak, and the
    useful range is well below 1: the target's peak *is* 1, so every residual is O(1) and
    torch's default delta of 1.0 keeps the loss in its quadratic branch nearly everywhere --
    that is 0.5 * MSE, not a robust loss. Start around 0.1 (a frame more than a tenth of a
    peak off the target is treated as an outlier) and sweep 0.02-0.5 to see the knob move.

    Scale note: ``nn.functional.huber_loss`` is the delta-scaled form, so for small residuals
    it equals 0.5 * the MSE, not the MSE. Loss values are therefore ~half an MSE run's at the
    same fit quality; compare huber runs to each other, and compare across losses with the HR
    metric (``train.measure_hr``), never by the loss number.
    """

    def __init__(self, delta: float = 0.1, eps: float = 1e-8):
        super().__init__()
        if delta <= 0:
            raise ValueError(f"HuberLoss delta must be > 0, got {delta}")
        self.delta = delta
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Identical rescale to MSELoss: the dataset target is sum-normalised, so its peak
        # height depends on how many beats landed in the window. Without this, `delta` would
        # mean a different fraction of a beat in every window.
        target = target / target.amax(dim=-1, keepdim=True).clamp_min(self.eps)
        return nn.functional.huber_loss(output, target, delta=self.delta)


class CorrAmpLoss(nn.Module):
    """Correlation (shape) + d' detection-margin (amplitude/contrast).

    Correlation is scale-invariant over the whole window, so it rewards beats being
    in the right *place* but is blind to how far they rise above the floor -- the
    output can sit barely above a noisy floor and still score well, which is why some
    beats come out weak and hard to distinguish. The d' term adds the missing contrast
    incentive: it is the signal-detection separability of the output at beat frames vs
    floor frames, (mean_beat - mean_floor) / std_floor per item. Maximizing it raises
    the beats *and* flattens the floor, i.e. pushes for high-SNR peaks everywhere.

    total = (1 - Pearson corr)  -  amp_weight * d'      (both averaged over the batch)

    Beat vs floor frames come from the target comb: a frame is a beat when the
    (nonnegative) target exceeds beat_threshold * its per-item peak. Like correlation,
    d' is invariant to affine scaling of the output, so it does not reintroduce the
    absolute-scale sensitivity the signal head was chosen to avoid. amp_weight = 0
    recovers plain CorrelationLoss exactly (the d' term is not even evaluated).
    """

    # Floor the denominator variance at this fraction of the output's total variance. A
    # near-flat floor otherwise sends var_floor -> 0 and d' -> inf, spiking the loss and
    # its gradient exactly when the model starts quieting the floor. Flooring caps d' at
    # ~1/sqrt(this) std of separation (a beat many std above the floor is already perfect
    # detection; "more" is not worth an exploding gradient) and only bites when the floor
    # is anomalously flat; it is scale-invariant since var_total scales with the output.
    VAR_FLOOR_FRAC = 1e-2

    def __init__(self, amp_weight: float = 0.0, beat_threshold: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.corr = CorrelationLoss(eps=eps)
        self.amp_weight = amp_weight
        self.beat_threshold = beat_threshold
        self.eps = eps

    def _dprime(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Split each item's frames into beat vs floor by its own target comb. The masks
        # depend only on the (constant) target, so gradients flow purely through output.
        peak = target.amax(dim=-1, keepdim=True)
        beat = target > self.beat_threshold * peak            # (batch, time) bool
        floor = ~beat
        # clamp_min(1) guards the degenerate all-beat / all-floor item against divide-by-zero.
        beat_n = beat.sum(dim=-1).clamp_min(1)
        floor_n = floor.sum(dim=-1).clamp_min(1)

        mean_beat = (output * beat).sum(dim=-1) / beat_n
        mean_floor = (output * floor).sum(dim=-1) / floor_n
        var_floor = ((output - mean_floor.unsqueeze(-1)) ** 2 * floor).sum(dim=-1) / floor_n

        var_total = output.var(dim=-1, unbiased=False)
        var_floor = torch.maximum(var_floor, self.VAR_FLOOR_FRAC * var_total)

        dprime = (mean_beat - mean_floor) / (var_floor.sqrt() + self.eps)
        return dprime.mean()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.corr(output, target)
        if self.amp_weight > 0:
            loss = loss - self.amp_weight * self._dprime(output, target)
        return loss
