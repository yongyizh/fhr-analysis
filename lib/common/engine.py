"""The training loop, shared by every model.

Deliberately knows nothing about optuna or YAML: pruning enters through the plain
``on_epoch(epoch, val_loss)`` callback and config archiving through the ``save_config``
callback, which is what keeps the train phase usable without the optimize phase installed.

Terminology: ``val_*`` throughout. The split scored here selects ``model_best.pt`` (and, in
the optimize phase, is the objective the search maximises against), which makes it a
validation set -- not a test set.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.io import atomic_save, plot_training_curves
from common.metrics import HRScore


@dataclass
class FitResult:
    """What a training run produced, for the caller to select or score on.

    ``best_val_loss`` is the selection criterion (the epoch saved as model_best.pt).
    ``best_score`` is the HR-correlation *at that same epoch* -- not the best score seen, which
    would describe a checkpoint nobody kept. Scoring the val-loss-selected epoch is what makes
    the number match the model you would actually deploy.
    """

    best_val_loss: float    # the selection loss: training loss when there was no val split
    best_epoch: int
    best_score: Optional[HRScore] = None
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[Optional[float]] = field(default_factory=list)
    scores: List[Optional[HRScore]] = field(default_factory=list)


def train_one_epoch(
        model: nn.Module,
        data: DataLoader,
        optimiser: optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        clip: Optional[float] = None,
):
    """One pass over ``data``. Returns (mean batch loss, max pre-clip grad norm).

    The grad norm is a free by-product of clipping, so it is ``None`` when ``clip`` is None
    rather than 0.0 -- reporting an unmeasured norm as zero reads like vanished gradients.
    """
    model.to(device)
    model.train()

    total_loss = 0.0
    max_grad_norm = None if clip is None else 0.0

    for inpt, target in data:
        inpt = inpt.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimiser.zero_grad()

        output = model(inpt)
        loss = loss_fn(output, target)

        loss.backward()
        if clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            max_grad_norm = max(max_grad_norm, grad_norm)
        optimiser.step()

        total_loss += loss.item()

    # Average over batches so it is comparable to evaluate() and not dominated by one noisy batch.
    return total_loss / len(data), max_grad_norm


def evaluate(
        model: nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        loss_fn: Callable,
        scorer=None,
) -> float:
    """Mean loss over ``dataloader`` with grads disabled.

    ``scorer``, when given (a ``common.metrics.HRMetrics``), is fed every batch as it goes
    by, so the HR-correlation costs one shared forward pass rather than a second sweep.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inpt, target in dataloader:
            inpt = inpt.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(inpt)
            total_loss += loss_fn(output, target).item()
            if scorer is not None:
                scorer.update(output, target)

    return total_loss / len(dataloader)


def fit(
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader],
        optimiser: optim.Optimizer,
        loss_fn: Callable,
        epochs: int,
        device: torch.device,
        clip: Optional[float] = None,
        scheduler=None,                           # common.optim.Scheduler, or None
        best_model_path: Optional[str] = None,    # best-epoch weights, re-saved on each improvement
        last_model_path: Optional[str] = None,    # final-epoch weights, saved once at the end
        curves_path: Optional[str] = None,        # train/validation loss plot, saved at the end
        save_config: Optional[Callable[[], None]] = None,
        early_stop_patience: Optional[int] = None,
        on_epoch: Optional[Callable[[int, float], None]] = None,
        make_scorer: Optional[Callable[[], object]] = None,
) -> FitResult:
    """Train for ``epochs`` and return a :class:`FitResult`.

    ``make_scorer`` builds a fresh ``common.metrics.HRMetrics`` per epoch (fresh because
    one instance accumulates one pass over the split). The score is measured every epoch so the
    curve can be plotted, but it does **not** select the checkpoint -- validation loss does,
    unchanged. What the caller gets back is the score at the val-loss-selected epoch.

    Each ``*_path`` is saved only when given, and saved atomically, so an interrupted run
    never corrupts it. The best-epoch checkpoint is rewritten every time the validation loss
    improves, so it is always on disk even if training is cut short.

    ``save_config``, when given, is invoked immediately after every checkpoint write. That is
    what guarantees a checkpoint and the config that produced it are never out of sync on
    disk: writing the config only at the end (the old ssnet behaviour) leaves a preempted run
    with an orphaned ``model_best.pt`` and no way to rebuild the architecture that loads it.

    ``early_stop_patience`` stops the run after that many epochs with no improvement. Note
    this counts epochs-without-improvement; neossnet's original loop instead counted LR
    reductions, which coupled early stopping to the scheduler's behaviour.

    ``on_epoch``, when given, is called after every epoch as ``on_epoch(epoch, val_loss)`` and
    may raise to stop the run early -- the Optuna search uses this to report intermediate
    losses and prune unpromising trials, and keeping it a plain callback means this module
    never imports optuna.

    ``val_data`` None means there is no held-out split (every patient trains). The training
    loss then stands in for validation loss everywhere it is used -- checkpoint selection,
    early stopping, the plateau schedule and ``on_epoch`` -- so model_best.pt is the
    lowest-training-loss epoch. No scorer runs, since there is nothing held out to score.
    """
    lowest_loss = float("inf")
    best_epoch = -1
    best_score: Optional[HRScore] = None
    epochs_since_improvement = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    scores: List[Optional[HRScore]] = []

    for epoch in range(epochs):
        train_loss, max_grad_norm = train_one_epoch(
            model, train_data, optimiser, loss_fn, device, clip)
        if val_data is not None:
            scorer = make_scorer() if make_scorer is not None else None
            val_loss = evaluate(model, device, val_data, loss_fn, scorer)
            score = scorer.result() if scorer is not None else None
        else:
            val_loss = score = None
        # What selects the checkpoint and drives the schedule / early stop / pruning.
        select_loss = train_loss if val_loss is None else val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scores.append(score)

        if select_loss < lowest_loss:
            lowest_loss = select_loss
            best_epoch = epoch
            best_score = score
            epochs_since_improvement = 0
            if best_model_path is not None:
                atomic_save(lambda p: torch.save(model.state_dict(), p), best_model_path)
                if save_config is not None:
                    save_config()
        else:
            epochs_since_improvement += 1

        # Report the LR this epoch actually ran at, then step the schedule for the next one.
        lr_note = f', LR: {optimiser.param_groups[0]["lr"]:.2e}' if scheduler is not None else ''
        if scheduler is not None:
            scheduler.step(select_loss)

        grad_note = '' if max_grad_norm is None else \
            f', Max grad norm (pre-clip): {max_grad_norm:.4f}'
        score_note = '' if score is None else f', HR: {score}'
        val_note = 'n/a' if val_loss is None else f'{val_loss:.6f}'
        print(f'[{epoch+1}|{epochs}] Train loss: {train_loss:.6f}, '
              f'Val loss: {val_note}{score_note}{grad_note}{lr_note}')

        # After the epoch is fully logged so a pruning exception can't skip the print above.
        if on_epoch is not None:
            on_epoch(epoch, select_loss)

        if early_stop_patience is not None and epochs_since_improvement >= early_stop_patience:
            print(f'Early stopping: no {"training" if val_data is None else "validation"} improvement for {early_stop_patience} epoch(s).')
            break

    if last_model_path is not None:
        atomic_save(lambda p: torch.save(model.state_dict(), p), last_model_path)
        if save_config is not None:
            save_config()
    if curves_path is not None:
        # The agreement fraction, not the correlation: it is the metric a search ranks by,
        # and unlike r it stays meaningful on a near-constant rate (see common.metrics).
        agreement = [None if s is None else s.within_tol for s in scores]
        atomic_save(
            lambda p: plot_training_curves(train_losses, val_losses, p, scores=agreement),
            curves_path)
        print(f'Saved training curves to {curves_path}')

    return FitResult(
        best_val_loss=lowest_loss,
        best_epoch=best_epoch,
        best_score=best_score,
        train_losses=train_losses,
        val_losses=val_losses,
        scores=scores,
    )
