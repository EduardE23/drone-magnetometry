import numpy as np
import pandas as pd
import pywt
from scipy.signal import convolve
from scipy.spatial import KDTree

from signal_processing_tools import butter_lowpass_filter, compute_snr_per_sensor

COMPONENTS = ["Mag-X [nT]", "Mag-Y [nT]", "Mag-Z [nT]", "Total field [nT]"]
FILTER_SUFFIX = "_2Hz"

SENSOR_BIAS = {1: 53.80, 2: 75.14, 3: 131.90, 5: 47.58}


def object_spacing(objects: pd.DataFrame, i: int = 0, j: int = 1) -> float:
    """Distance between two objects."""
    return float(np.hypot(objects.loc[i, "Longitude [m]"] - objects.loc[j, "Longitude [m]"],
                          objects.loc[i, "Latitude [m]"] - objects.loc[j, "Latitude [m]"]))

def match_nearest_samples(survey: pd.DataFrame, objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find the survey sample closest to each object.

    """
    tree = KDTree(survey[["Longitude [m]", "Latitude [m]"]].values)
    distances, indices = tree.query(objects[["Longitude [m]", "Latitude [m]"]].values)

    objects = objects.copy()
    objects["Nearest_Point_Index"] = indices
    objects["Distance_to_Nearest_Point"] = distances
    nearest_points = survey.iloc[indices].reset_index(drop=True)
    return objects, nearest_points


def extract_dipole_windows(survey: pd.DataFrame, nearest_points: pd.DataFrame, window_size: int = 1000,
                           filter: bool = True, fs: float = 200, cutoff: float = 2) -> pd.DataFrame:
    """
    Extract ±window_size samples around the first anomaly for every sensor.

    """
    assert len(nearest_points) >= 1, "At least one anomaly must be provided."

    primary_time = nearest_points.iloc[0]["GPS Time [ms]"]
    other_times = nearest_points["GPS Time [ms]"].values[1:]

    snippets = []
    for sensor_id in survey["Sensor ID"].unique():
        sensor_data = survey[survey["Sensor ID"] == sensor_id].reset_index(drop=True)

        if filter:
            for comp in COMPONENTS:
                if comp in sensor_data.columns:
                    sensor_data[comp + FILTER_SUFFIX] = butter_lowpass_filter(sensor_data[comp], fs=fs, cutoff=cutoff)

        primary_idx = np.abs(sensor_data["GPS Time [ms]"] - primary_time).idxmin()
        start = max(primary_idx - window_size, 0)
        end = min(primary_idx + window_size, len(sensor_data))

        snippet = sensor_data.iloc[start:end].copy()
        snippet["Sensor ID"] = sensor_id
        snippet["Center_Idx"] = primary_idx - start

        other_offsets = [np.abs(sensor_data["GPS Time [ms]"] - t).idxmin() - start for t in other_times]
        snippet["Other_Anomaly_Offsets"] = [other_offsets] * len(snippet)

        snippets.append(snippet)

    return pd.concat(snippets, ignore_index=True)


def normalize_windows(windows: pd.DataFrame, filtered: bool = True) -> pd.DataFrame:
    """Min-max normalise X, Y, Z and total field to [0, 1] per sensor."""
    cols = [c + FILTER_SUFFIX if filtered else c for c in COMPONENTS]

    normalized = windows.copy()
    grouped = normalized.groupby("Sensor ID")[cols]
    col_min, col_max = grouped.transform("min"), grouped.transform("max")
    normalized[cols] = (normalized[cols] - col_min) / (col_max - col_min)
    return normalized


def remove_sensor_bias(windows: pd.DataFrame, columns: list[str], bias: dict[int, float] = SENSOR_BIAS) -> pd.DataFrame:
    """Subtract the constant offset of each sensor from the given columns."""
    corrected = windows.copy()
    offsets = corrected["Sensor ID"].map(bias).fillna(0)
    corrected[columns] = corrected[columns].sub(offsets, axis=0)
    return corrected


def sensor_signal(windows: pd.DataFrame, sensor_id: int, column: str = "Total field [nT]_2Hz") -> np.ndarray:
    return windows.loc[windows["Sensor ID"] == sensor_id, column].values



# SNR
# -------------------------
def snr_table(windows: pd.DataFrame, filtered: bool = False, window_size: int = 1000,
              signal_window_size: int = 700) -> pd.DataFrame:

    snr = compute_snr_per_sensor(windows, filtered=filtered, signal_window_size=signal_window_size,
                                 total_window_size=2 * window_size)
    snr["Component"] = snr["Component"].str.extract(r"(Mag-[XYZ]|Total field)", expand=False)
    snr = snr[snr["Component"] == "Total field"]
    return snr[["Sensor ID", "SNR", "Peak-to-peak [nT]", "Noise StdDev [nT]"]].reset_index(drop=True)


# Testing wavelets
# -------------------------
def wavelet_convolution(signal: np.ndarray, wavelet: str = "sym7", level: int = 6) -> np.ndarray:
    kernel = pywt.Wavelet(wavelet).wavefun(level=level)[0][::-1]
    return convolve(signal, kernel, mode="same")


def wavelet_kernel(wavelet: str = "sym7", level: int = 7) -> np.ndarray:
    return pywt.Wavelet(wavelet).wavefun(level=level)[1][::-1]


def sliding_wavelet_energy(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    pad = len(kernel) // 2
    padded = np.pad(signal, pad_width=pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel))[:len(signal)]
    return np.sum((windows * kernel) ** 2, axis=1)
