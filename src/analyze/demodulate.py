"""Carrier demodulation: read the 200 Hz band as a tone the heartbeat modulates.

Every other method in this package treats that band as acoustics -- a beat is a sound, and
the detectors look for its energy. This one treats it as a *carrier*, so the beat lives in
the tone's amplitude rather than in its power:

    x(t) -> bandpass -> Hilbert -> mix down by f_c and low-pass -> A(t) = |u(t)|

A(t) lands on the input's own time axis, sample for sample, and is non-negative by
construction -- the same contract a beat-activity envelope from FUNet satisfies, which is
what lets a detector consume it unchanged.

Only the signal processing lives here. The figures, HTML output and HR outlier rejection
from the original experiment script are deliberately not carried over: this module is
imported by the runner, so it must stay free of matplotlib and plotly.

One parameter is *measured* rather than configured -- the carrier frequency, via the
two-pass estimate in demodulate_audio. Everything else below is a fixed choice.
"""

from typing import Tuple

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from analyze.data import Audio

# Defaults, retuned for a ~200-230 Hz carrier at 5 kHz. The baseband cutoff only has to pass
# the beat itself (1.8-3 Hz), so 25 Hz is generous and keeps the envelope smooth.
DEMOD_BANDPASS_HZ = (200.0, 230.0)
DEMOD_BANDPASS_ORDER = 4
DEMOD_BASEBAND_HZ = 25.0
DEMOD_LOWPASS_ORDER = 4

# Samples this close to either end are excluded from the phase fit (filter transients), and
# samples below this percentile of |z| are excluded as well (phase is meaningless where there
# is no amplitude).
DEMOD_EDGE_EXCLUSION_S = 0.5
DEMOD_AMPLITUDE_FLOOR_PCT = 25.0


def _check_below_nyquist(label: str, value_hz: float, hz: float) -> None:
    """Reject a cutoff at or above Nyquist before scipy turns it into a cryptic error."""
    nyquist = hz / 2.0
    if not (0.0 < value_hz < nyquist):
        raise ValueError(
            f"{label} must satisfy 0 < f < Nyquist ({nyquist:g} Hz) for a {hz:g} Hz "
            f"signal; got {value_hz:g} Hz.")


def check_band_below_nyquist(label: str, band: Tuple[float, float], hz: float) -> None:
    low, high = float(band[0]), float(band[1])
    nyquist = hz / 2.0
    if not (0.0 < low < high < nyquist):
        raise ValueError(
            f"{label} must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz) for a "
            f"{hz:g} Hz signal; got {low:g}-{high:g} Hz.")


def _interior_mask(n: int, hz: float, edge_s: float) -> np.ndarray:
    """Samples at least ``edge_s`` from either end, or everything if that leaves nothing.

    The fallback keeps a short clip usable: an empty mask would make the fit undefined
    rather than merely edge-contaminated.
    """
    t_rel = np.arange(n, dtype=float) / hz
    mask = (t_rel >= edge_s) & (t_rel <= t_rel[-1] - edge_s if n else False)
    if np.count_nonzero(mask) < 2:
        return np.ones(n, dtype=bool)
    return mask


def estimate_carrier_hz(
        analytic: np.ndarray,
        hz: float,
        edge_s: float = DEMOD_EDGE_EXCLUSION_S,
        amplitude_floor_pct: float = DEMOD_AMPLITUDE_FLOOR_PCT,
) -> Tuple[float, np.ndarray]:
    """Carrier frequency from the slope of the unwrapped analytic phase.

    A tone at f_c has phase 2*pi*f_c*t + phi_0, so a line fit gives f_c = slope / (2*pi).
    Over a real band this is the amplitude-weighted mean instantaneous frequency -- fine
    while one component dominates, misleading if two comparable tones share the band.

    Unwrapping happens over the whole array before anything is dropped: unwrap needs every
    consecutive difference, so masking first could hide a 2*pi step. Only the fit is
    restricted -- to samples clear of the edges and above ``amplitude_floor_pct`` of |z|.

    Returns ``(carrier_hz, fit_mask)``.
    """
    analytic = np.asarray(analytic)
    n = analytic.size
    if n < 2:
        raise ValueError("Carrier estimation needs at least 2 samples.")

    phase = np.unwrap(np.angle(analytic))
    t_rel = np.arange(n, dtype=float) / hz

    valid = _interior_mask(n, hz, edge_s)
    magnitude = np.abs(analytic)
    if 0.0 < amplitude_floor_pct < 100.0:
        floor = np.percentile(magnitude[valid], amplitude_floor_pct)
        strong = valid & (magnitude >= floor)
        if np.count_nonzero(strong) >= 2:   # only tighten if a line is still defined
            valid = strong

    slope = np.polyfit(t_rel[valid], phase[valid], 1)[0]
    return float(slope / (2.0 * np.pi)), valid


def demodulate_to_baseband(
        analytic: np.ndarray,
        hz: float,
        carrier_hz: float,
        cutoff_hz: float = DEMOD_BASEBAND_HZ,
        order: int = DEMOD_LOWPASS_ORDER,
) -> np.ndarray:
    """Mix an analytic signal down to 0 Hz and low-pass I and Q identically.

        u(t) = LPF{ z(t) * e^(-j*2*pi*f_c*t) }

    No factor of 2: z is single-sided already, so there is no image at -2*f_c to halve away.
    Time runs from the start of the array so the exponent stays small.

    I and Q get the same zero-phase filter. Filtering them differently would rotate the
    phase and corrupt the residual the second pass is trying to measure.
    """
    _check_below_nyquist("Baseband low-pass cutoff", cutoff_hz, hz)
    n = np.asarray(analytic).size
    t_rel = np.arange(n, dtype=float) / hz
    baseband = np.asarray(analytic) * np.exp(-1j * 2.0 * np.pi * carrier_hz * t_rel)
    sos = butter(order, cutoff_hz, btype="lowpass", fs=hz, output="sos")
    return sosfiltfilt(sos, np.real(baseband)) + 1j * sosfiltfilt(sos, np.imag(baseband))


def residual_offset_hz(baseband: np.ndarray, hz: float, valid: np.ndarray) -> float:
    """How far the mix-down missed the true carrier.

    Mixing at f_c + delta leaves the baseband turning at 2*pi*delta rad/s, so fitting that
    slope recovers delta. Measured after the low-pass, which is why it beats the wideband
    estimate: the noise that biased the first fit is gone by then.
    """
    n = np.asarray(baseband).size
    t_rel = np.arange(n, dtype=float) / hz
    phase = np.unwrap(np.angle(baseband))
    return float(np.polyfit(t_rel[valid], phase[valid], 1)[0] / (2.0 * np.pi))


def demodulate_audio(
        bandpassed: Audio,
        baseband_hz: float = DEMOD_BASEBAND_HZ,
        edge_s: float = DEMOD_EDGE_EXCLUSION_S,
        amplitude_floor_pct: float = DEMOD_AMPLITUDE_FLOOR_PCT,
        lowpass_order: int = DEMOD_LOWPASS_ORDER,
) -> Tuple[Audio, float, float, float, float]:
    """Two-pass complex demodulation of an already-bandpassed Audio.

    Pass 1 mixes at the phase-fit estimate and measures the leftover; pass 2 mixes at the
    corrected carrier. This is the only fitted quantity in the module -- band and cutoffs are
    configured, the carrier is measured.

    Returns ``(amplitude, initial, residual, corrected, final_residual)``. Every step is
    sample-wise, so A(t) lands on the input's own time axis unchanged.
    """
    hz = float(bandpassed.hz)
    _check_below_nyquist("Baseband low-pass cutoff", baseband_hz, hz)

    analytic = hilbert(np.asarray(bandpassed.data, dtype=float))
    initial_carrier, valid = estimate_carrier_hz(analytic, hz, edge_s, amplitude_floor_pct)

    first = demodulate_to_baseband(analytic, hz, initial_carrier, baseband_hz, lowpass_order)
    residual = residual_offset_hz(first, hz, valid)
    corrected_carrier = initial_carrier + residual

    second = demodulate_to_baseband(analytic, hz, corrected_carrier, baseband_hz, lowpass_order)
    final_residual = residual_offset_hz(second, hz, valid)

    amplitude = np.abs(second)   # A(t) = sqrt(I^2 + Q^2), non-negative by construction
    return (
        Audio(bandpassed.time, bandpassed.hz, amplitude),
        initial_carrier,
        residual,
        corrected_carrier,
        final_residual,
    )
