import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import resample_poly

from analyze.constants import NEOSSNET_MODEL_HZ
from analyze.data import FiberPair

# Top-level name from the lib/neossnet submodule (see pyproject package-dir).
from utils import generate_output

def normalize_path(path):
    return path.removesuffix("/") + "/"

def plot_amplitudes(signals, time, file):
    num = signals.shape[0]

    plt.figure(figsize=(10, 4 * num))
    for i in range(num):
        plt.subplot(num, 1, i + 1)
        plt.plot(time, signals[i], 'g', linewidth=1)
        # plt.xlim(360, 364)

        plt.title(f"Signal {i + 1}")

    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(file)
    plt.close()

# def moving_average(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
#     n = max(1, int(round(seconds * fs)))
#     if n <= 1:
#         return x.copy()
#     kernel = np.ones(n, dtype=float) / n
#     return np.convolve(x, kernel, mode="same")

def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.copy()
    kernel = np.ones(n, dtype=float) / n
    return np.convolve(x, kernel, mode='same')

def moving_average_v2(y, window):
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or y.size == 0:
        return y.copy()
    # Clamp to the array length. np.convolve(mode="same") returns max(M, N) elements, so a
    # window wider than the trace yields MORE smoothed values than there are timestamps --
    # silently desyncing t from bpm on a sparse beat trace (few beats + the hardcoded
    # 20-point HR window is exactly that case) and turning any correlation into noise.
    w = min(w, y.size)
    kernel = np.ones(w)
    counts = np.convolve(np.ones_like(y), kernel, mode="same") # points per window
    sums = np.convolve(y, kernel, mode="same")
    return sums / counts

def running_rms(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    n = max(1, int(round(seconds * fs)))
    return np.sqrt(np.maximum(moving_average(np.square(x), n), 0.0))

def suppress_transients(x: np.ndarray, fs: float, window_s: float = 0.12) -> np.ndarray:
    local = running_rms(x, fs, window_s)
    scale = np.median(local) + 3.0 * (1.4826 * np.median(np.abs(local - np.median(local))) + 1e-9)
    gain = 1.0 / np.maximum(1.0, local / max(scale, 1e-6))
    return x * gain



def run_neossnet(
        x: np.ndarray,
        src_hz: int,
        model,
        config
):
    """Run NeoSSNet on a single-channel waveform sampled at ``src_hz``.

    Matches the model's training distribution: peak-normalise to [-1, 1] and
    resample to 4 kHz before inference, then resample the heart/lung outputs
    back to ``src_hz`` and restore the original amplitude scale. Returns
    (heart, lung) as numpy arrays the same length as ``x``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    # Peak-normalise to [-1, 1] (training used torchaudio normalize=True).
    peak = float(np.max(np.abs(x))) + 1e-12
    xn = x / peak

    # Resample src_hz -> 4 kHz so the model's learned filters are correctly tuned.
    g = np.gcd(NEOSSNET_MODEL_HZ, int(src_hz))
    up, down = NEOSSNET_MODEL_HZ // g, int(src_hz) // g
    x_model = resample_poly(xn, up, down) if up != down else xn

    tensor = torch.tensor(x_model, dtype=torch.float32).unsqueeze(0)  # (1, T)
    heart, lung = generate_output(tensor, model, config)

    def to_native(y: torch.Tensor) -> np.ndarray:
        y = resample_poly(y.numpy(), down, up) if up != down else y.numpy()
        if len(y) >= n:
            y = y[:n]
        else:
            y = np.pad(y, (0, n - len(y)))
        return y * peak

    return to_native(heart), to_native(lung)

def abdomen_sound(out, tag=None):
    def generate_abdomen_sound(pair: FiberPair):
        name = f"abdomen.wav"
        if tag is not None:
            name = f"abdomen_{tag}.wav"
        pair.abdomen.write(out / name)

        return pair
    return generate_abdomen_sound