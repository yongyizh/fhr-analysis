"""Estimate and correct the accumulating time drift between fiber and NST.

The NST (.wav) drops samples during recording, so its clock falls progressively
behind the continuous fiber clock — a lag that ACCUMULATES over the window. This
measures that lag from the two beat trains and lets the caller shift the NST onto
the fiber clock before computing HR.

Method (per the validation spec):
  1. Turn each beat train (fiber model-output beats, NST beats) into a smooth pulse
     train — each beat becomes a difference-of-two-sigmoids soft pulse centred on
     the beat (`sigmoid_pulse_train`), on a fine common grid.
  2. Slide a segment of length `seg_len_s` (step `seg_step_s`) across the window;
     in each segment cross-correlate the two pulse trains over candidate shifts.
     The shift with the highest cross-correlation is the fiber<->NST lag there.
  3. Enforce the lag is monotonically NON-DECREASING with time (drift accumulates).
  4. `make_shift_fn` turns the per-segment lags into a continuous shift(t); the
     caller adds it to NST beat/time so NST moves onto the fixed fiber clock.

Sign convention: the returned lag is the shift to ADD to NST times to align them
to the fiber (`nst_corrected = nst + shift(nst)`), and it comes out increasing.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid_pulse_train(beats, grid, width_s=0.03, steep_s=0.01):
    """Sum of soft pulses, one centred on each beat: a difference of two sigmoids
    (rise at beat-width, fall at beat+width) — a smooth top-hat centred on the beat."""
    beats = np.asarray(beats, dtype=float)
    grid = np.asarray(grid, dtype=float)
    out = np.zeros_like(grid)
    steep_s = max(steep_s, 1e-6)
    half = width_s + 8.0 * steep_s          # compact local support (tails die off)
    for c in beats:
        lo = np.searchsorted(grid, c - half)
        hi = np.searchsorted(grid, c + half)
        if hi <= lo:
            continue
        g = grid[lo:hi]
        out[lo:hi] += (1.0 / (1.0 + np.exp(-(g - (c - width_s)) / steep_s))
                       - 1.0 / (1.0 + np.exp(-(g - (c + width_s)) / steep_s)))
    return out


def _isotonic(y):
    """Least-squares non-decreasing fit (pool-adjacent-violators).

    Used instead of a running maximum to enforce "lag always increases". A running
    max RATCHETS on noise — on a truly flat lag it climbs to the largest outlier and
    manufactures drift that isn't there. Isotonic regression returns the best
    monotonic fit, so flat-but-noisy input stays flat and real drift is still tracked.
    """
    y = np.asarray(y, dtype=float)
    vals, wts = [], []
    for v in y:
        v, w = float(v), 1.0
        while vals and vals[-1] > v:                 # pool adjacent violators
            pv, pw = vals.pop(), wts.pop()
            v = (pv * pw + v * w) / (pw + w)
            w += pw
        vals.append(v); wts.append(w)
    out = np.empty(y.size, dtype=float)
    i = 0
    for v, w in zip(vals, wts):
        k = int(round(w)); out[i:i + k] = v; i += k
    return out


def coarse_lag(beats_f, beats_n, anchor_window, max_lag_s=30.0,
               rate_win_s=10.0, grid_fs=10.0):
    """Coarse absolute NST->fiber lag over ``anchor_window``, from the beat-RATE curves.

    The beat pulse trains repeat every beat (~0.4 s), so a WIDE search on them is
    hopelessly ambiguous. The beat RATE (beats per ``rate_win_s``) varies slowly and
    is NOT periodic at that scale, so it can be matched unambiguously across tens of
    seconds. This anchors the fine per-segment search when the starting lag is large
    (seconds to tens of seconds); the fine stage then refines it to sub-second.

    Returns ``(lag_s, score)`` — lag_s is the shift to ADD to NST times.
    """
    beats_f = np.sort(np.asarray(beats_f, dtype=float))
    beats_n = np.sort(np.asarray(beats_n, dtype=float))
    a0, a1 = anchor_window
    grid = np.arange(a0 - max_lag_s, a1 + max_lag_s, 1.0 / grid_fs)

    def rate(b):
        half = rate_win_s / 2.0
        return (np.searchsorted(b, grid + half) - np.searchsorted(b, grid - half)).astype(float)

    rf, rn = rate(beats_f), rate(beats_n)
    w = (grid >= a0) & (grid <= a1)
    ref = rf[w]
    if ref.size < 4 or np.ptp(ref) == 0:
        return 0.0, 0.0
    i0 = int(np.argmax(w))                       # grid index of a0
    a = ref - ref.mean()
    da = np.sqrt(float((a * a).sum()))

    best, best_S = -np.inf, 0
    for S in range(-int(max_lag_s * grid_fs), int(max_lag_s * grid_fs) + 1):
        s0 = i0 - S                              # NST shifted right by S (same convention)
        if s0 < 0 or s0 + ref.size > rn.size:
            continue
        seg = rn[s0:s0 + ref.size]
        b_ = seg - seg.mean()
        d = da * np.sqrt(float((b_ * b_).sum()))
        if d <= 0:
            continue
        sc = float((a * b_).sum()) / d
        if sc > best:
            best, best_S = sc, S
    return best_S / grid_fs, best


def estimate_drift_lag(
    beats_f, beats_n, window,
    seg_len_s=8.0, seg_step_s=4.0,
    max_lag_s=30.0, max_lag_step_s=0.15,
    grid_fs=200.0, pulse_width_s=0.04, pulse_steep_s=0.01,
    monotonic=True, min_seg_beats=5,
    score_frac=0.9, initial_lag_guess=None, anchor_halfwin_s=0.5,
    anchor_len_s=60.0,
):
    """Return (seg_centers, lag_applied, lag_measured, valid).

    lag_applied is the monotonic shift-to-add-to-NST (s) at each segment centre;
    lag_measured is the raw per-segment pick (before the monotonic constraint).

    Beat trains are periodic, so the cross-correlation has near-equal peaks a beat
    period apart. To avoid octave errors we pick, among peaks within `score_frac`
    of the segment max, the one CLOSEST to the expected lag (the previous segment's
    lag, or `initial_lag_guess` for the first). A distinct correct peak (from real
    HR variation) still wins outright; the tie-break only matters when steady HR
    makes the octaves genuinely ambiguous.
    """
    beats_f = np.sort(np.asarray(beats_f, dtype=float))
    beats_n = np.sort(np.asarray(beats_n, dtype=float))

    # Anchor: the fine per-segment search only looks +/- anchor_halfwin_s around the
    # starting guess, which cannot find a lag of seconds-to-tens-of-seconds. When no
    # guess is given, get one from the beat-RATE cross-correlation, which IS
    # unambiguous over a wide range (see coarse_lag).
    if initial_lag_guess is None:
        # A coarse estimate over a span gives the lag at that span's CENTRE. Taking two
        # (start and end of the window) yields the drift slope, so we can extrapolate to
        # the first fine segment. Without this the guess is off by ~half the anchor span
        # worth of drift, which is easily a whole beat period -> an octave slip.
        L = min(anchor_len_s, max(4.0, (window[1] - window[0]) / 2.0))
        lag1, s1 = coarse_lag(beats_f, beats_n, (window[0], window[0] + L), max_lag_s=max_lag_s)
        lag2, s2 = coarse_lag(beats_f, beats_n, (window[1] - L, window[1]), max_lag_s=max_lag_s)
        c1, c2 = window[0] + L / 2.0, window[1] - L / 2.0
        slope = (lag2 - lag1) / (c2 - c1) if c2 > c1 else 0.0
        c0 = window[0] + seg_len_s / 2.0                    # first fine segment centre
        initial_lag_guess = lag1 + slope * (c0 - c1)
        print(f"[lag] coarse anchor: {lag1:+.2f}s @{c1:.0f}s, {lag2:+.2f}s @{c2:.0f}s "
              f"(scores {s1:.2f}/{s2:.2f}) -> start guess {initial_lag_guess:+.2f}s, "
              f"drift {slope*1000:.1f} ms/s")

    margin = max_lag_s + pulse_width_s + 8.0 * pulse_steep_s + 0.05
    t0, t1 = window[0] - margin, window[1] + margin
    grid = np.arange(t0, t1, 1.0 / grid_fs)
    f = sigmoid_pulse_train(beats_f, grid, pulse_width_s, pulse_steep_s)
    n = sigmoid_pulse_train(beats_n, grid, pulse_width_s, pulse_steep_s)

    seg_n = int(round(seg_len_s * grid_fs))
    seg_centers = np.arange(window[0] + seg_len_s / 2,
                            window[1] - seg_len_s / 2 + 1e-9, seg_step_s)

    lag_measured = np.full(seg_centers.size, np.nan)
    lag_applied = np.full(seg_centers.size, np.nan)
    valid = np.zeros(seg_centers.size, dtype=bool)

    prev = None
    for i, c in enumerate(seg_centers):
        fs0 = int(round((c - seg_len_s / 2 - t0) * grid_fs))
        fs1 = fs0 + seg_n
        f_seg = f[fs0:fs1]

        # enough fiber beats in this segment to trust the alignment?
        nb = int(np.sum((beats_f >= c - seg_len_s / 2) & (beats_f <= c + seg_len_s / 2)))
        if nb < min_seg_beats or not np.any(f_seg > 0):
            lag_applied[i] = prev if prev is not None else initial_lag_guess
            prev = lag_applied[i]
            continue

        # candidate shift range S (samples): S>0 shifts NST content later to meet fiber.
        # Search a SMALL window around the running estimate (`initial_lag_guess` on the
        # first segment, `prev` after) — narrower than a beat period, so octave
        # (period-multiple) picks are impossible. The window is symmetric so drift can be
        # tracked without an upper-edge ratchet; monotonicity is enforced afterwards.
        center = initial_lag_guess if prev is None else prev
        half = anchor_halfwin_s if prev is None else max_lag_step_s
        lo = int(round((center - half) * grid_fs))
        hi = int(round((center + half) * grid_fs))
        target_S = center * grid_fs
        lo = max(lo, int(round(-max_lag_s * grid_fs)))
        hi = min(hi, int(round(max_lag_s * grid_fs)))
        hi = min(hi, fs0)                      # keep n[fs0-hi:] in bounds
        lo = max(lo, fs1 - n.size)            # keep n[:fs1-lo] in bounds
        if hi < lo:
            lag_applied[i] = prev if prev is not None else initial_lag_guess
            prev = lag_applied[i]
            continue

        # NORMALIZED cross-correlation over all S in [lo, hi] via one correlation:
        #   score(S) = <f_seg, n_slice> / (||f_seg|| ||n_slice||),  S = hi - k
        # Normalizing by the sliding window energy removes the bias of the raw dot
        # product toward shifts that merely overlap more pulse mass.
        region = n[fs0 - hi: fs1 - lo]
        num = np.correlate(region, f_seg, mode="valid")      # length hi-lo+1
        r2 = np.concatenate([[0.0], np.cumsum(region * region)])
        energy = r2[seg_n:] - r2[:-seg_n]                    # sliding window energy, same length
        denom = np.sqrt(np.sum(f_seg * f_seg)) * np.sqrt(np.maximum(energy, 1e-12))
        scores = num / denom
        S_range = hi - np.arange(scores.size)                # samples

        mx = scores.max()
        if mx <= 0:
            lag_applied[i] = prev if prev is not None else initial_lag_guess
            prev = lag_applied[i]
            continue
        near = np.where(scores >= score_frac * mx)[0]
        j = near[np.argmin(np.abs(S_range[near] - target_S))]  # closest near-tie to expected
        meas = S_range[j] / grid_fs

        lag_measured[i] = meas
        lag_applied[i] = meas
        valid[i] = True
        prev = meas                          # track the raw estimate for the next search centre

    if monotonic:
        m = ~np.isnan(lag_measured)
        if m.sum() >= 2:
            # best monotonic FIT to the raw measurements (not a running max, which
            # would ratchet upward on noise and invent drift on a flat lag)
            lag_applied = np.interp(seg_centers, seg_centers[m], _isotonic(lag_measured[m]))
        else:
            lag_applied = np.maximum.accumulate(lag_applied)
    return seg_centers, lag_applied, lag_measured, valid


def shift_beats(beats, shift_fn, gap_tol_s=0.02):
    """Shift beats onto the fiber clock; flag intervals broken by the shift.

    Returns ``(shifted_beats, pair_bad)``. ``pair_bad[i]`` marks the interval
    between shifted beats i and i+1 whose length the correction changed by more
    than ``gap_tol_s`` — i.e. the pair straddles a jump in the lag curve, so the
    gap it opens is an artefact of the shift, not physiology. Callers should show
    a GAP there instead of computing 60/IBI across it (which would put a fake dip
    in the HR curve). A smoothly-varying lag changes an interval by only
    drift_rate x IBI (~0.002 s), well under the default tolerance, so nothing is
    dropped when there is no jump.
    """
    b = np.sort(np.asarray(beats, dtype=float))
    s = np.asarray(shift_fn(b), dtype=float)
    pair_bad = np.abs(np.diff(s)) > gap_tol_s
    return b + s, pair_bad


def make_shift_fn(seg_centers, lag_applied):
    """Continuous shift(t) (s to add to NST times), edge-held outside the segments."""
    seg_centers = np.asarray(seg_centers, dtype=float)
    lag_applied = np.asarray(lag_applied, dtype=float)
    if seg_centers.size == 0:
        return lambda t: np.zeros_like(np.asarray(t, dtype=float))
    return lambda t: np.interp(np.asarray(t, dtype=float), seg_centers, lag_applied)


def plot_lag(seg_centers, lag_applied, lag_measured, valid, out_png, title=""):
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    ax.plot(seg_centers, lag_applied, "-o", color="tab:purple", ms=4,
            label="applied lag (monotonic)")
    m = ~np.isnan(lag_measured)
    ax.plot(seg_centers[m], lag_measured[m], "x", color="0.5", ms=6,
            label="measured per-segment lag")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("NST→fiber lag (s)")
    ax.set_title(title or "Fiber↔NST drift lag (cross-correlation of beat pulse trains)")
    ax.legend(fontsize=8); ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --- self-test: recover a known accumulating drift --------------------------
if __name__ == "__main__":
    win = (10.0, 250.0)
    drift = lambda q: 0.5 + 3.0 * (q - win[0]) / (win[1] - win[0])   # 0.5 -> 3.5 s, accumulating

    def make_beats(hr_fn, seed):
        rng = np.random.default_rng(seed)
        t, out = win[0], []
        while t < win[1] + 5:
            t += 60.0 / hr_fn(t) + 0.008 * rng.standard_normal()
            out.append(t)
        return np.array(out)

    hr_varying = lambda q: 150 + 20 * np.sin(2 * np.pi * (q - win[0]) / 120)
    hr_steady = lambda q: 150.0

    # (a) small lag, explicit guess
    for label, hr_fn in [("varying HR (130-170)", hr_varying), ("steady HR (~150)", hr_steady)]:
        B = make_beats(hr_fn, 1)
        Bn = B - drift(B)                       # NST clock behind by drift(q)
        sc, la, lm, v = estimate_drift_lag(B, Bn, win, initial_lag_guess=0.5)
        true = drift(sc)
        err = np.nanmax(np.abs(la - true))
        print(f"[{label}] max |applied-true| = {err:.3f}s "
              f"(applied {la[0]:.2f}->{la[-1]:.2f}, true {true[0]:.2f}->{true[-1]:.2f})  "
              f"{'PASS' if err < 0.2 else 'FAIL'}")

    # (b) LARGE starting lag, auto coarse anchor (initial_lag_guess=None)
    for start in (4.0, 20.0):
        big = lambda q, s=start: s + 3.0 * (q - win[0]) / (win[1] - win[0])
        B = make_beats(hr_varying, 2)
        Bn = B - big(B)
        sc, la, lm, v = estimate_drift_lag(B, Bn, win)      # auto anchor
        true = big(sc)
        err = np.nanmax(np.abs(la - true))
        print(f"[large lag start={start:.0f}s, auto anchor] max |applied-true| = {err:.3f}s "
              f"(applied {la[0]:.2f}->{la[-1]:.2f}, true {true[0]:.2f}->{true[-1]:.2f})  "
              f"{'PASS' if err < 0.3 else 'FAIL'}")
