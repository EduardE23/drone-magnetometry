from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import linregress
from ssqueezepy import extract_ridges

from dataloader import load_survey
from signal_processing_tools import FS, SENSORS, butter_bandpass_filter, calculate_ssq_stft, select_sensor

PROPELLER_BAND = (37, 60)    # Hz
VELOCITY_FLIGHTS = range(7, 12)
SURVEY_SPEED = 0.5           # m/s, low enough to label the slowest flight
EDGE = 200                   # samples removed at both ends of the transform (edge effects)
VELOCITY_TOLERANCE = 0.2     # m/s

# Part of each flight that is used (removes take-off and landing)
TRIM = {7: (3000, -5000), 8: (10000, None), 9: (2100, -5500), 10: (2200, -5000), 11: (2000, -3200)}


def flight_velocity(survey_index: int) -> float:
    """Target velocity of a velocity flight [m/s]: 1 m/s for survey 7, +0.5 m/s per flight."""
    return 1 + 0.5 * (survey_index - 7)


def ridge_settings(survey_index: int, sensor: int) -> tuple[int, int]:
    """Number of ridges and ridge bandwidth (bins) for extract_ridges, tuned per flight and sensor."""
    n_ridges = 3 if sensor == 5 and survey_index in (7, 8, 9) else 4
    if sensor == 5 and survey_index == 9:
        bw = 8
    elif sensor == 4 and survey_index in (7, 9):
        bw = 5
    else:
        bw = 10
    return n_ridges, bw


def track_data(survey: pd.DataFrame, sensor: int, trim=(None, None), exclude_tracks=()) -> pd.DataFrame:
    """Samples of one sensor on the labelled tracks, trimmed, without the excluded tracks."""
    data = survey[survey["Track_number"] > 0]
    data = data[data["Sensor ID"] == sensor].iloc[slice(*trim)]
    return data[~data["Track_number"].isin(exclude_tracks)]


def propeller_ssq(signal, n_fft: int, fs: float = FS):
    """Bandpass to the propeller band and compute the SSQ-STFT. Returns filtered, Tx, Sx, ssq_freqs, scales."""
    filtered = butter_bandpass_filter(signal, *PROPELLER_BAND, fs)
    return (filtered, *calculate_ssq_stft(filtered, n_fft, fs))


def propeller_ridges(Tx, scales, n_ridges: int = 4, bw: int = 10, penalty: float = 80) -> np.ndarray:
    """Ridge (row) index of Tx per sample, shape (time, n_ridges)."""
    return extract_ridges(Tx, scales, penalty, n_ridges=n_ridges, bw=bw, transform="stft")


def flight_ridges(survey: pd.DataFrame, survey_index: int, sensor: int, n_fft: int = 2048,
                  edge: int = EDGE) -> list[pd.DataFrame]:
    """
    Ridges of the propeller band for one sensor of a velocity flight, from low to high frequency.

    Returns one DataFrame per ridge with Velocity [m/s], Ridge Index, Amplitude and Frequencies
    of every sample, without the edge samples.
    """
    data = select_sensor(survey, sensor).iloc[slice(*TRIM[survey_index])]
    _, Tx, _, ssq_freqs, scales = propeller_ssq(data["Total field [nT]"], n_fft)
    ridge_idxs = propeller_ridges(Tx, scales, *ridge_settings(survey_index, sensor))

    Tx = Tx[:, edge:-edge]
    time_idxs = np.arange(Tx.shape[1])
    velocity = data["Velocity"].iloc[edge:-edge]

    ridges = []
    for ridge in np.argsort(ridge_idxs.mean(axis=0)):
        idx = ridge_idxs[edge:-edge, ridge].astype(int)
        ridges.append(pd.DataFrame({
            "Velocity [m/s]": velocity,
            "Ridge Index": idx,
            "Amplitude": np.abs(Tx[idx, time_idxs]),
            "Frequencies": ssq_freqs[idx],
        }))
    return ridges


def velocity_ridges(survey_indices=VELOCITY_FLIGHTS, sensors=SENSORS,
                    tolerance: float = VELOCITY_TOLERANCE) -> dict[tuple[int, int], pd.DataFrame]:
    """
    Ridges of all velocity flights, only the samples within tolerance of the flight's target velocity.

    Returns dict[(sensor, ridge)] with the samples of all flights combined, ridge 1 = lowest frequency.
    """
    grouped = defaultdict(list)
    for survey_index in survey_indices:
        survey = load_survey(survey_index, SURVEY_SPEED)
        v_target = flight_velocity(survey_index)

        for sensor in sensors:
            for ridge, df in enumerate(flight_ridges(survey, survey_index, sensor), start=1):
                on_speed = df["Velocity [m/s]"].between(v_target - tolerance, v_target + tolerance)
                grouped[(sensor, ridge)].append(df[on_speed])

    return {key: pd.concat(dfs, ignore_index=True) for key, dfs in grouped.items()}


def fit_line(x, y) -> tuple[float, float]:
    """Least squares slope and intercept."""
    fit = linregress(x, y)
    return fit.slope, fit.intercept


def amplitude_db(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and ridge amplitude in dB (20 log10), for the samples with a non-zero amplitude."""
    df = df[df["Amplitude"] > 0]
    return df["Velocity [m/s]"].values, 20 * np.log10(df["Amplitude"].values)
