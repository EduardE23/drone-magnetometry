"""
Constant frequency noise in the survey data: sensor offsets, highpass noise level, the narrow
harmonic bands (16.7 Hz family, 20 Hz) and an ARMA model of a harmonic's envelope.
"""
import numpy as np
from scipy.signal import hilbert
from statsmodels.tsa.arima.model import ARIMA

from signal_processing_tools import (FS, SENSORS, band_envelope, butter_bandpass_filter, butter_highpass_filter,
                                     butter_lowpass_filter, frequency_response, select_sensor, truncated_normal_fit)

HARMONIC_BASE = 16.7   # Hz
HALF_WIDTH = 0.5       # Hz, half width of the harmonic bands
TRIM = 6000            # samples removed at both ends (take-off and landing)


# Low frequency
# -------------------------
def lowpass_sensors(survey, column="Total field [nT]", cutoff=2, fs=FS) -> dict[int, np.ndarray]:
    """Lowpass filtered on-track signal per sensor."""
    return {s: butter_lowpass_filter(select_sensor(survey, s, on_track=True)[column].to_numpy(), cutoff, fs)
            for s in SENSORS}


def sensor_offsets(lowpassed: dict[int, np.ndarray], reference_sensor=4) -> dict[str, float]:
    """Median offset of each sensor w.r.t. the reference sensor [nT]."""
    ref = lowpassed[reference_sensor]
    return {f"Sensor {s} - Sensor {reference_sensor}": np.median(signal - ref)
            for s, signal in lowpassed.items() if s != reference_sensor}


def highpass_noise(survey, cutoff=2, fs=FS) -> dict[int, dict]:
    """Highpass filtered on-track total field per sensor, with its mean |noise|, std and linear trend."""
    results = {}
    for s in SENSORS:
        noise = butter_highpass_filter(select_sensor(survey, s, on_track=True)["Total field [nT]"], cutoff, fs)
        x = np.arange(len(noise))
        results[s] = {
            "noise": noise,
            "mean_abs": np.mean(np.abs(noise)),
            "std": np.std(noise),
            "trend": np.polyval(np.polyfit(x, noise, 1), x),
        }
    return results


def compare_windows(signal, lowcut, highcut, windows=("kaiser",), transition_width=0.4, ripple_db=60, fs=FS):
    """band_envelope for each FIR window, with the frequency response (w, h) of its taps."""
    results = {}
    for window in windows:
        result = band_envelope(signal, fs, lowcut, highcut, transition_width, ripple_db, window)
        result["w"], result["h"] = frequency_response(result["taps"], fs)
        results[window] = result
    return results


def band_per_sensor(survey, lowcut, highcut, on_track=False, trim=0, fs=FS, **fir) -> dict[int, dict]:
    """band_envelope of the total field of every sensor, trim samples removed at both ends."""
    results = {}
    for s in SENSORS:
        signal = select_sensor(survey, s, on_track)["Total field [nT]"]
        if trim:
            signal = signal.iloc[trim:-trim]
        results[s] = band_envelope(signal, fs, lowcut, highcut, **fir)
    return results


def harmonic_bands(survey, base_freq, harmonics, half_width=HALF_WIDTH, trim=TRIM, **fir) -> dict[int, dict]:
    """band_per_sensor around n * base_freq for every harmonic n."""
    return {n: band_per_sensor(survey, n * base_freq - half_width, n * base_freq + half_width, trim=trim, **fir)
            for n in harmonics}


def envelope_distributions(survey, harmonics, base_freq=HARMONIC_BASE, band_center=None,
                           half_width=HALF_WIDTH, fs=FS) -> dict[tuple[int, int], dict]:
    """
    Envelope of each harmonic band per sensor (full record) with a truncated Gaussian fit.
    With band_center, that band is used for every harmonic instead of n * base_freq.

    Returns dict[(sensor, n)] with center, envelope and the truncated_normal_fit (mu, sigma, dist).
    """
    results = {}
    for s in SENSORS:
        signal = select_sensor(survey, s)["Total field [nT]"].values
        for n in harmonics:
            center = n * base_freq if band_center is None else band_center
            envelope = band_envelope(signal, fs, center - half_width, center + half_width)["envelope"]
            results[(s, n)] = {"center": center, "envelope": envelope, **truncated_normal_fit(envelope)}
    return results


def _bandpassed(survey, sensor, trim, freq, half_width, highpass, fs):
    signal = select_sensor(survey, sensor)["Total field [nT]"].iloc[slice(*trim)]
    signal = butter_highpass_filter(signal, highpass, fs)
    return butter_bandpass_filter(signal, freq - half_width, freq + half_width, fs)


def arma_harmonic_model(train_survey, test_survey, freq, sensor=3, order=(2, 0, 0), seed=42,
                        train_trim=(TRIM, -TRIM), test_trim=(TRIM, -7000), half_width=HALF_WIDTH,
                        highpass=2, fs=FS) -> dict:
    """
    Model one harmonic as a sine modulated by an ARMA envelope.

    The ARMA model is fitted on the Hilbert envelope of the bandpassed train survey. A simulated
    envelope times sin(2 pi f t) is compared with the bandpassed test survey.

    Returns dict with t, actual, synthetic, rmse and the fitted model.
    """
    train = _bandpassed(train_survey, sensor, train_trim, freq, half_width, highpass, fs)
    test = _bandpassed(test_survey, sensor, test_trim, freq, half_width, highpass, fs)

    envelope = np.abs(hilbert(butter_bandpass_filter(train, freq - half_width, freq + half_width, fs)))
    fit = ARIMA(envelope, order=order).fit()

    # simulate takes the shocks from random_state but the initial state from the global numpy RNG,
    # so both are seeded to get the same simulation every run
    rng_state = np.random.get_state()
    np.random.seed(seed)
    simulated = fit.simulate(nsimulations=len(test), random_state=seed)
    np.random.set_state(rng_state)

    t = np.arange(len(test)) / fs
    synthetic = simulated * np.sin(2 * np.pi * freq * t)
    rmse = np.sqrt(np.mean((test - synthetic) ** 2))
    return {"t": t, "actual": test, "synthetic": synthetic, "rmse": rmse, "fit": fit}
