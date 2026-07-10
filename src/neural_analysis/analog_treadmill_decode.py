"""Utilities for analog treadmill voltage inspection and decoding."""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy import signal


def volts2speed(millivolts: np.ndarray) -> np.ndarray:
    """Convert treadmill analog voltage to treadmill speed.

    Args:
        millivolts:
            Treadmill voltage signal in millivolts, shape ``(...,)``.

    Returns:
        Treadmill speed in millimeters per second with the same shape as
        ``millivolts``.
    """
    max_dac_value = 2.5 / 3.3 * 4095
    dac_value = np.asarray(millivolts, dtype=float) * 4095 / 3300
    return (dac_value / max_dac_value - 0.5) * 2 * 1000


def analyze_treadmill_signal(treadmill_analog: np.ndarray, sample_rate: float) -> None:
    """Plot spectral diagnostics for an analog treadmill signal.

    Args:
        treadmill_analog:
            One-dimensional treadmill analog signal, shape ``(n_samples,)``.
            Units are expected to match the caller's acquisition units.
        sample_rate:
            Sampling rate in Hz.

    Returns:
        None. Displays matplotlib figures for manual inspection.
    """
    treadmill_signal = np.asarray(treadmill_analog, dtype=float).squeeze()
    if treadmill_signal.ndim != 1 or treadmill_signal.size < 2:
        raise ValueError("treadmill_analog must be a 1D signal with at least 2 samples.")

    treadmill_signal = treadmill_signal - np.nanmedian(treadmill_signal)

    nperseg = int(min(4096, treadmill_signal.size))
    if nperseg < 128:
        nperseg = treadmill_signal.size
    noverlap = int(nperseg // 2)

    f_spec, t_spec, sxx = sp.signal.spectrogram(
        treadmill_signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    f_per, pxx = sp.signal.periodogram(
        treadmill_signal,
        fs=sample_rate,
        window="hann",
        detrend="constant",
        scaling="density",
    )
    f_welch, pxx_welch = sp.signal.welch(
        treadmill_signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    sxx_db = 10 * np.log10(sxx + 1e-20)
    pxx_db = 10 * np.log10(pxx + 1e-20)
    pxx_welch_db = 10 * np.log10(pxx_welch + 1e-20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    pcm = axes[0].pcolormesh(t_spec, f_spec, sxx_db, shading="auto")
    axes[0].set_title("Treadmill Analog Spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm, ax=axes[0], label="Power/Frequency (dB)")

    axes[1].plot(f_per, pxx_db, lw=1.0)
    axes[1].set_title("Treadmill Analog Periodogram")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power/Frequency (dB)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(f_welch, pxx_welch_db, lw=1.0, color="tab:orange")
    axes[2].set_title("Treadmill Analog Welch PSD")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power/Frequency (dB)")
    axes[2].grid(alpha=0.3)

    plt.show()


def wavelet_hard_threshold(
    treadmill_signal: np.ndarray,
    wavelet: str = "sym8",
    level: int = 4,
) -> np.ndarray:
    """Remove baseline noise while preserving sharp peaks.

    Args:
        treadmill_signal:
            One-dimensional analog signal, shape ``(n_samples,)``.
        wavelet:
            PyWavelets wavelet name.
        level:
            Wavelet decomposition level.

    Returns:
        Denoised signal with shape ``(n_samples,)`` and the same units as the
        input signal.
    """
    import pywt

    signal_array = np.asarray(treadmill_signal, dtype=float)
    coeffs = pywt.wavedec(signal_array, wavelet, mode="per", level=level)

    detail_coeffs = coeffs[-1]
    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    sigma = mad / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_array)))

    denoised_coeffs = [coeffs[0]]
    for coefficient_array in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(coefficient_array, value=threshold, mode="hard"))

    clean_signal = pywt.waverec(denoised_coeffs, wavelet, mode="per")
    return clean_signal[:len(signal_array)]


def robust_spectral_subtraction(
    noisy_signal: np.ndarray,
    noise_reference: np.ndarray,
    sample_rate: int = 25000,
) -> np.ndarray:
    """Subtract a noise spectrum from a noisy signal.

    Args:
        noisy_signal:
            One-dimensional signal to clean, shape ``(n_samples,)``.
        noise_reference:
            One-dimensional reference noise signal, shape
            ``(n_reference_samples,)``.
        sample_rate:
            Sampling rate in Hz.

    Returns:
        Cleaned signal in the same units as ``noisy_signal``. The returned
        length is determined by SciPy's inverse STFT reconstruction.
    """
    freqs, noise_psd = signal.welch(noise_reference, fs=sample_rate, nperseg=1024)
    noise_mag_template = np.sqrt(noise_psd)

    _, _, stft_values = signal.stft(noisy_signal, fs=sample_rate, nperseg=1024)
    magnitude = np.abs(stft_values)
    phase = np.angle(stft_values)

    beta = 1.5
    clean_mag = np.maximum(magnitude - beta * noise_mag_template[:, np.newaxis], 0)

    stft_clean = clean_mag * np.exp(1j * phase)
    _, cleaned_signal = signal.istft(stft_clean, fs=sample_rate)

    return cleaned_signal
