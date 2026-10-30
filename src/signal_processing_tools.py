"""
Signal processing building blocks for the magnetometer data: Butterworth filters, FIR band
envelopes for the constant frequency (harmonic) analysis, the synchrosqueezed STFT, distribution
fits and SNR. Nothing in here plots; the plots are made in the notebooks.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, firwin, freqz, hilbert, kaiserord
from scipy.stats import norm, normaltest, truncnorm
from ssqueezepy import ssq_stft

FS = 200          # Hz, magnetometer sampling frequency
SENSORS = range(1, 6)


def select_sensor(survey: pd.DataFrame, sensor: int, on_track: bool = False) -> pd.DataFrame:
    """Rows of one sensor, optionally only the samples labelled on_track."""
    mask = survey["Sensor ID"] == sensor
    if on_track:
        mask &= survey["Track_status"] == "on_track"
    return survey[mask]


def butter_filter(data, cutoff, fs: float, btype: str, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth filter. cutoff is one frequency [Hz], or [low, high] for a bandpass."""
    b, a = butter(order, cutoff, btype=btype, fs=fs)
    return filtfilt(b, a, data)


def butter_highpass_filter(data, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    return butter_filter(data, cutoff, fs, "highpass", order)


def butter_lowpass_filter(data, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    return butter_filter(data, cutoff, fs, "lowpass", order)


def butter_bandpass_filter(data, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    return butter_filter(data, [lowcut, highcut], fs, "bandpass", order)


def fir_bandpass_taps(lowcut: float, highcut: float, fs: float, transition_width: float = 0.4,
                      ripple_db: float = 60, window: str = "kaiser") -> np.ndarray:
    """
    Linear phase FIR bandpass.

    Notes:
    ------------
    The number of taps follows from the Kaiser design rule for the transition width [Hz] and the
    stopband attenuation [dB], rounded up to an even number. Other windows use the same number of
    taps, so they can be compared with the Kaiser design.
    """
    ntaps, beta = kaiserord(ripple_db, transition_width / (0.5 * fs))
    ntaps += ntaps % 2
    window = ("kaiser", beta) if window == "kaiser" else window
    return firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False, window=window, scale=True)


def band_envelope(data, fs: float, lowcut: float, highcut: float, transition_width: float = 0.4,
                  ripple_db: float = 60, window: str = "kaiser") -> dict:
    """
    Isolate one frequency band and describe it with the analytic signal.

    Returns:
    ------------
    dict with
        taps: FIR coefficients (see fir_bandpass_taps)
        filtered: zero phase bandpassed signal
        envelope: Hilbert envelope (instantaneous amplitude) [nT]
        inst_freq: instantaneous frequency [Hz], one sample shorter than the signal
    """
    taps = fir_bandpass_taps(lowcut, highcut, fs, transition_width, ripple_db, window)
    filtered = filtfilt(taps, 1.0, data)
    analytic = hilbert(filtered)
    phase = np.unwrap(np.angle(analytic))
    return {
        "taps": taps,
        "filtered": filtered,
        "envelope": np.abs(analytic),
        "inst_freq": np.diff(phase) / (2 * np.pi) * fs,
    }


def frequency_response(taps: np.ndarray, fs: float, n: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies [Hz] and complex response of a FIR filter."""
    return freqz(taps, worN=n, fs=fs)


# -------------------------
# Time-frequency
# -------------------------
def calculate_ssq_stft(signal, n_fft: int, fs: float = FS):
    """
    Synchrosqueezed short time Fourier transform.

    Parameters:
    ------------
    n_fft: int
        Window length, a power of 2. Sets the trade-off between time and frequency resolution.

    Returns:
    ------------
    Tx: synchrosqueezed STFT (frequency x time)
    Sx: plain STFT
    ssq_freqs: frequency of every row of Tx [Hz]
    scales: scales of the transform, needed for extract_ridges

    Notes:
    ------------
    hop_len=1 and modulated=True keep the transform invertible.
    """
    return ssq_stft(signal, fs=fs, n_fft=n_fft, hop_len=1, modulated=True, padtype="zero")


# -------------------------
# Distributions
# -------------------------
def normal_fit(signal) -> dict:
    """Gaussian fit and D'Agostino-Pearson normality test."""
    mu, sigma = norm.fit(signal)
    statistic, p_value = normaltest(signal)
    return {"mu": mu, "sigma": sigma, "dist": norm(mu, sigma), "statistic": statistic, "p_value": p_value}


def truncated_normal_fit(x) -> dict:
    """Gaussian fit of x, truncated to the range of x (e.g. for envelopes, which can't be negative)."""
    mu, sigma = norm.fit(x)
    a, b = (np.min(x) - mu) / sigma, (np.max(x) - mu) / sigma
    return {"mu": mu, "sigma": sigma, "dist": truncnorm(a, b, loc=mu, scale=sigma)}


# -------------------------
# SNR
# -------------------------
def compute_snr_per_sensor(anomaly_df: pd.DataFrame,
                           components=("Mag-X [nT]", "Mag-Y [nT]", "Mag-Z [nT]", "Total field [nT]"),
                           signal_window_size: int = 700, total_window_size: int = 2000,
                           filtered: bool = False) -> pd.DataFrame:
    """
    SNR of every component and sensor in a window centred on an anomaly.

    The signal is the peak-to-peak value of the central signal_window_size samples, the noise the
    standard deviation of the samples before and after it.

    Notes:
    ------------
    The windows are selected with .loc, which includes the end label, so the signal window shares
    its first and last sample with the noise windows.
    """
    if filtered:
        components = [c.replace("[nT]", "[nT]_2Hz") for c in components]

    signal_start = (total_window_size - signal_window_size) // 2
    signal_end = signal_start + signal_window_size

    noise_before = slice(0, signal_start)
    signal = slice(signal_start, signal_end)
    noise_after = slice(signal_end, total_window_size)

    results = []
    for sensor_id in anomaly_df["Sensor ID"].unique():
        df_sensor = anomaly_df[anomaly_df["Sensor ID"] == sensor_id].reset_index(drop=True)

        for comp in components:
            sig_window = df_sensor.loc[signal, comp]
            noise_window = pd.concat([df_sensor.loc[noise_before, comp], df_sensor.loc[noise_after, comp]])

            peak_to_peak = sig_window.max() - sig_window.min()
            std_noise = noise_window.std()

            results.append({
                "Sensor ID": sensor_id,
                "Component": comp,
                "SNR": peak_to_peak / std_noise if std_noise != 0 else np.nan,
                "Peak-to-peak [nT]": peak_to_peak,
                "Noise StdDev [nT]": std_noise,
            })

    return np.round(pd.DataFrame(results), 2)
