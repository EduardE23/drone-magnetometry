import sys

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.stats import laplace, norm
from shapely import vectorized
from shapely.geometry import MultiPoint, Polygon

from paths import PYMEL_DIR
from signal_processing_tools import FS, SENSORS, butter_lowpass_filter

SURVEY_INDEX = 2
TIELINE_INDEX = 3
SURVEY_SPEED = 1.1           # m/s
EXCLUDED_TRACKS = (13, 14, 15)
MATCH_DISTANCE = 0.01        # m, max distance between a tie-line and a survey sample at a crossing
PLOT_TRIM = (5075, -900)     # samples of each sensor used for the before/after plot and sensor grids

# Survey area for the per sensor grids
SURVEY_POLYGON = Polygon([(553876, 5645508), (553895, 5645521), (553925, 5645474), (553910, 5645457), (553876, 5645508)])

# Inset around the objects in the per sensor grids
ZOOM_BOX = {"lat_min": 5645500, "lat_max": 5645512, "lon_min": 553888, "lon_max": 553895}
ZOOM_POLYGON = Polygon([
    (553885.5, 5645516.5), (553888, 5645516.5), (553898, 5645502.5), (553898, 5645502.5),
    (553897.4, 5645497.5), (553895.5, 5645497.5), (553884.5, 5645515), (553884.5, 5645517.5),
])

# Zoom on the anomaly in the all sensor grid
ANOMALY_POLYGON = Polygon([
    (553885.5, 5645515.8), (553887.5, 5645515.8), (553897.5, 5645501.5), (553897.4, 5645497.5),
    (553895.5, 5645497.5), (553884.5, 5645515.0), (553884.8, 5645515.8),
])


# Levelling
# -------------------------
def detrend_tracks(data: pd.DataFrame, exclude_tracks=(), column="Total field [nT]") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove a linear trend (vs. sample number) from every track of every sensor.

    Returns:
    ------------
    detrended: the on-track samples with the column 'detrended'
    slopes: |slope| [nT/sample] per sensor and track
    """
    data = data[data["Track_number"] > 0]
    data = data[~data["Track_number"].isin(exclude_tracks)]

    sensors, slopes = [], []
    for sensor in SENSORS:
        sensor_data = data[data["Sensor ID"] == sensor]
        tracks = []
        for track_number in sensor_data["Track_number"].unique():
            track = sensor_data[sensor_data["Track_number"] == track_number].copy()
            x = np.arange(len(track))
            slope, intercept = np.polyfit(x, track[column].values, 1)
            track["detrended"] = track[column].values - (slope * x + intercept)
            slopes.append({"Sensor ID": sensor, "Track_number": track_number, "slope": abs(slope)})
            tracks.append(track)
        sensors.append(pd.concat(tracks, ignore_index=True))

    return pd.concat(sensors, ignore_index=True), pd.DataFrame(slopes)


def tieline_correction(survey: pd.DataFrame, tieline: pd.DataFrame, column="detrended",
                       max_distance=MATCH_DISTANCE, cutoff=2, fs=FS) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Level the survey to the tie-line, per sensor.

    Notes:
    ------------
    Tie-line samples within max_distance of a survey sample are crossings. The residual
    tie-line - survey at the crossings (both lowpass filtered) is linearly interpolated over the
    survey, 0 outside the crossings, and subtracted.

    Returns:
    ------------
    survey: with the columns 'residuals' and 'Total field_corrected'
    matched: tie-line samples at the crossings, per sensor
    """
    survey = survey.copy()
    matched = {}

    for sensor in SENSORS:
        survey_sensor = survey[survey["Sensor ID"] == sensor]
        tieline_sensor = tieline[tieline["Sensor ID"] == sensor]
        survey_coords = survey_sensor[["Longitude [m]", "Latitude [m]"]].values

        distances, indices = cKDTree(survey_coords).query(
            tieline_sensor[["Longitude [m]", "Latitude [m]"]].values, k=1, distance_upper_bound=max_distance)
        valid = np.isfinite(distances)
        crossings = tieline_sensor.iloc[valid]
        matched[sensor] = crossings.copy()

        residuals = (butter_lowpass_filter(crossings[column].values, cutoff, fs)
                     - butter_lowpass_filter(survey_sensor.iloc[indices[valid]][column].values, cutoff, fs))
        interp_residuals = griddata(crossings[["Longitude [m]", "Latitude [m]"]].values, residuals,
                                    survey_coords, method="linear", fill_value=0)

        survey.loc[survey_sensor.index, "residuals"] = interp_residuals
        survey.loc[survey_sensor.index, "Total field_corrected"] = survey_sensor[column].values - interp_residuals

    return survey, matched


def before_after(survey: pd.DataFrame, trim=PLOT_TRIM, cutoff=2, fs=FS) -> dict[int, dict]:
    """Lowpass filtered total field (before) and detrended field (after) per sensor, with their std."""
    results = {}
    for sensor in SENSORS:
        data = survey[survey["Sensor ID"] == sensor].iloc[slice(*trim)]
        before = butter_lowpass_filter(data["Total field [nT]"], cutoff, fs)
        after = butter_lowpass_filter(data["detrended"], cutoff, fs)
        results[sensor] = {"before": before, "after": after, "std_before": before.std(), "std_after": after.std()}
    return results


def residual_fits(survey: pd.DataFrame, laplace_sensors=(1, 2, 3, 4)) -> dict[int, dict]:
    """Non-zero levelling residuals per sensor with a Laplace fit (laplace_sensors) or a Gaussian fit."""
    corrected = survey[survey["residuals"] != 0]
    results = {}
    for sensor in SENSORS:
        residuals = corrected.loc[corrected["Sensor ID"] == sensor, "residuals"].dropna()
        dist = laplace if sensor in laplace_sensors else norm
        results[sensor] = {"residuals": residuals, "dist": dist, "params": dist.fit(residuals)}
    return results


def _nn_grid(lat, lon, values, cell_size) -> pd.DataFrame:
    if str(PYMEL_DIR) not in sys.path:
        sys.path.append(str(PYMEL_DIR))
    from PyMEL.raster import blank, nearest_neighbor as nn

    result = nn.nn_interpolate(lat, lon, values, blank.create_2d(lat, lon, cell_size=cell_size), df_out=True)

    return result[1]


def mask_polygon(grid: pd.DataFrame, polygon: Polygon | None) -> pd.DataFrame:
    """Grid cells inside the polygon (all cells if polygon is None)."""
    return grid[vectorized.contains(polygon, grid["lon"].values, grid["lat"].values)].copy()


def mask_by_neighbors(grid: pd.DataFrame, lon, lat, k=6, radius=1.0) -> pd.DataFrame:
    """Grid cells with at least k data points within radius [m]."""
    dists, _ = cKDTree(np.c_[lon, lat]).query(np.c_[grid["lon"].values, grid["lat"].values],
                                               k=k, distance_upper_bound=radius)
    return grid[np.sum(np.isfinite(dists), axis=1) >= k].copy()


def sensor_grids(survey: pd.DataFrame, column="detrended", trim=PLOT_TRIM, cell_size=0.5) -> dict[int, dict]:
    """Grid of the raw (column) and corrected field per sensor."""
    grids = {}
    for sensor in SENSORS:
        data = survey[survey["Sensor ID"] == sensor].iloc[slice(*trim)]
        lat, lon = data["Latitude [m]"], data["Longitude [m]"]
        grids[sensor] = {"raw": _nn_grid(lat, lon, data[column], cell_size),
                         "corrected": _nn_grid(lat, lon, data["Total field_corrected"], cell_size)}
    return grids


def in_box(df: pd.DataFrame, box=ZOOM_BOX) -> pd.Series:
    return (df["Latitude [m]"].between(box["lat_min"], box["lat_max"])
            & df["Longitude [m]"].between(box["lon_min"], box["lon_max"]))


def zoom_grids(survey: pd.DataFrame, column="detrended", box=ZOOM_BOX, polygon=ZOOM_POLYGON,
               cell_size=0.025) -> dict[int, pd.DataFrame | None]:
    """Fine grid per sensor of the samples inside the box, clipped to the polygon."""
    grids = {}
    for sensor in SENSORS:
        data = survey[survey["Sensor ID"] == sensor]
        data = data[in_box(data, box)]
        if data.empty:
            grids[sensor] = None
            continue
        grid = _nn_grid(data["Latitude [m]"].values, data["Longitude [m]"].values, data[column].values, cell_size)
        grids[sensor] = mask_polygon(grid, polygon)
    return grids


def all_sensor_grid(survey: pd.DataFrame, column="detrended", cell_size=0.5, k=6, radius=1.5) -> pd.DataFrame:
    """
    Grid of all sensors together, cut to the convex hull of the data (eroded by 1.5 cells to trim
    the ragged interpolation at the edges) and without cells that have too few data points nearby.
    """
    lat, lon = survey["Latitude [m]"].values, survey["Longitude [m]"].values
    grid = _nn_grid(lat, lon, survey[column].values, cell_size)

    hull = MultiPoint(list(zip(lon, lat))).convex_hull
    hull = hull.buffer(-1.5 * cell_size) if hull.area > 0 else hull
    grid = mask_polygon(grid, hull)
    return mask_by_neighbors(grid, lon, lat, k=k, radius=radius)


def anomaly_grid(survey: pd.DataFrame, polygon=ANOMALY_POLYGON, column="detrended", cell_size=0.025,
                 buffer=0.5, k=6, radius=0.12) -> pd.DataFrame:
    """
    Fine grid of all sensors around the anomaly. The samples within the buffered polygon are
    interpolated (so the edges are supported), then the grid is clipped to the polygon.
    """
    lat, lon = survey["Latitude [m]"].values, survey["Longitude [m]"].values
    inside = vectorized.contains(polygon.buffer(buffer), lon, lat)

    grid = _nn_grid(lat[inside], lon[inside], survey[column].values[inside], cell_size)
    grid = mask_polygon(grid, polygon)
    return mask_by_neighbors(grid, lon[inside], lat[inside], k=k, radius=radius)


def objects_in(objects: pd.DataFrame, polygon: Polygon) -> pd.DataFrame:
    return objects[vectorized.contains(polygon, objects["Longitude [m]"].values, objects["Latitude [m]"].values)]
