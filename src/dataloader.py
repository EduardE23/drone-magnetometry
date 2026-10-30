from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from pyproj import Transformer

from paths import DATA_DIR, OBJECTS_FILE

class SurveyDataLoader:
    """
    Pipeline that loads survey files, aligns timestamps and positioning,
    applies sensor offsets, labels tracks, and aligns magnetometer data.

    Usage:
        loader = SurveyDataLoader(data_dir="path/to/csvs", survey_index=0, velocity_threshold=1.25)
        final_df = loader.process_all()
    """

    def __init__(self, data_dir: str | Path, survey_index: int, survey_speed: float ):
        self.data_dir = Path(data_dir)
        self.survey_index = int(survey_index)
        self.velocity_threshold = float(survey_speed)
        self.max_course_deviation = 10.0  # degrees deviation from the survey line axis
        self.min_speed_factor = 0.77       
        self.survey_data: dict[str, pd.DataFrame] = {}
        self._keywords = ["gga", "pps", "zda", "config", "magdata", "position"]

    def list_flight_files(self) -> list[str]:
        """
        List all flight data files in the directory. 

        Parameters:
        ------------
        data_dir: Path
            Path to main flight directory
        
        Returns:
        ------------
        list
            Strings of all the names of all the flight files


        """
        return [f for f in os.listdir(self.data_dir) if f != ".DS_Store"]

    def filter_files_by_keyword(self, files: list[str], keyword: str) -> np.ndarray:
        """
        Return sorted list of files containing the given keyword.

        Parameters:
        ------------
        files_list: 
            list from list_flight_files
        
        Returns:
        ------------
        Sorted list according to time of specific keyword

        """
        return np.sort([f for f in files if keyword in f.lower()])

    def _overlapping_position_file(self, position_files: list[str], gga_file: str) -> str:
        """
        Find the position file that overlaps the most in time with the given GGA file.
        
        Parameters:
        ------------
        position_files: 
            list of all position files

        Notes:
        ------------
        This is done because the gga (XYZ coords) datalogger is started manually 
        1 to 2 minutes later than the magnetic/roll/yaw etc data recording of the drone.
        """
        gga = pd.read_csv(self.data_dir / gga_file, delimiter=";")
        gga.columns = gga.columns.str.strip()
        gga_start, gga_end = gga["GPS Time [ms]"].min(), gga["GPS Time [ms]"].max()

        overlaps = []
        for f in position_files:
            pos = pd.read_csv(self.data_dir / f, usecols=["Elapsed", "Date", "Latitude"]).dropna(subset=["Latitude"])

            # The timestamps show in the files are the time elapsed from midnight
            midnight = pd.Timestamp(pos["Date"].iloc[0].replace("/", "-"), tz="UTC").value // 10**6
            pos_ms = pos["Elapsed"].to_numpy() - midnight
            overlaps.append(max(0, min(gga_end, pos_ms.max()) - max(gga_start, pos_ms.min())))

        if max(overlaps) == 0:
            raise FileNotFoundError(f"No position file overlaps in time with {gga_file}")
        return position_files[int(np.argmax(overlaps))]

    def load_survey_files(self) -> dict[str, pd.DataFrame]:
        """
        Load raw survey files for the given index.
        Handles automatic matching between GGA and the nearest position file.
        """
        all_files = self.list_flight_files()
        survey_data = {}

        for key in self._keywords:
            matches = self.filter_files_by_keyword(all_files, key)

            # Handle position file matching
            if key == "position":
                gga_file = self.filter_files_by_keyword(all_files, "gga")[self.survey_index]
                closest_pos = self._overlapping_position_file(list(matches), gga_file)
                filepath = self.data_dir / closest_pos
                # Just as info to know which datalogger file got matched to the 
                print(f"[INFO] Matched position file: {closest_pos} → overlapping with {gga_file}")
            else:
                if len(matches) <= self.survey_index:
                    raise IndexError(f"No '{key}' file found for survey index {self.survey_index}")
                filepath = self.data_dir / matches[self.survey_index]

            # Use comma for position, semicolon for everything else
            delimiter = "," if key == "position" else ";"
            skiprows = 5 if key == "config" else 0

            survey_data[key] = pd.read_csv(filepath, delimiter=delimiter, skiprows=skiprows)

        self.survey_data = survey_data
        return survey_data

    def format_survey_data(self) -> pd.DataFrame:
        """
        Clean, align, and merge GGA/ZDA/PPS/positioning data into a formatted dataset.
        Returns aligned_data (DataFrame) with columns:
          ["Date", "Timestamp [µs]", "GPS Time [ms]", "Time", "Longitude [m]", "Latitude [m]",
           "Altitude", "Roll", "Pitch", "Heading", "Velocity"]

        Parameters:
        ------------
        survey_data: dictionary
            Contains the pps, zda, gga, position, config, magdata. 

            pps: pd.DataFrame
                Reference internal clock for the drone with column: Timestamp [µs]

            zda: pd.DataFrame
                Reference GPS time for the drone with columns: [Timestamp [µs];
                                                                Timestamp GPS UTC [ms]]
            
            gga: pd.DataFrame
                Correct coordinates with columns: [Timestamp [µs];
                                                     GPS Time [ms];
                                                     Latitude [°];
                                                     Longitude [°];
                                                     Mean Sea Level [m];
                                                     Quality [];
                                                     Satellites [];
                                                     DGPS Age [s];
                                                     Geoid Separation [m]]
            
            position: pd.DataFrame 
                Contains telemetric data from drone with columns: 
                [Elapsed,
                Date,
                Time,
                Pitch,
                Roll,
                Heading,
                Latitude,
                Longitude,
                Altitude,
                Next WP,
                Velocity,
                RTK Status,
                Latitude RTK,
                Longitude RTK,
                Altitude RTK,
                ALT:Altitude,
                ALT:ID,
                OBS:Sectors1,
                OBS:Sectors2,
                OBS:Sectors3,
                OBS:ID]    
            
            config: pd.DataFrame
                Sensors configuration with columns: [OffsetX [m]; OffsetY [m]; OffsetZ [m]]
        
        Returns:
        ------------
        aligned_data: pd.DataFrame
            Dataframe which contains the correct Timestamp from pps, the GPS time from zda,
            the coordinates from gga and the pitch, roll, yaw, velocity, altitude from the positioning file.
            Contains the following columns: 
            ["Date", "Timestamp [µs]", "GPS Time [ms]", "Time", "Longitude [m]", "Latitude [m]", "Altitude", "Roll", "Pitch", "Heading", "Velocity"]

        Notes:
        ------------
        The positioning file is alternating in sampling frequency between approximately 26 Hz 38 Hz. 
        The data was thus downsampled to the gga sampling frequency 5 Hz by finding the nearest neighbour
        in time (GPS Time [ms]). GGA rows outside the time span of the position file are dropped.
        """
        gga = self.survey_data["gga"].copy()
        pps = self.survey_data["pps"].copy()
        zda = self.survey_data["zda"].copy()
        position = self.survey_data["position"].copy().dropna(subset=['Latitude', 'Longitude'])

        # gga cleanup 
        # ------------------------------------------------------------------------------------

        gga.columns = gga.columns.str.strip()
        gga = gga.drop(columns=[
            "Quality []", "Mean Sea Level [m]", "Satellites []",
            "DGPS Age [s]", "Geoid Separation [m]"
        ], errors="ignore")

        # Timestamp alignment: replace zda timestamps with pps then interpolate into GGA from 1 to 5 Hz 
        #------------------------------------------------------------------------------------

        zda.columns = zda.columns.str.strip()
        # zda arrives a few ms after pps pulse
        pps_time = np.sort(pd.to_numeric(pps['Timestamp [µs]'], errors='coerce').values)
        zda_raw = pd.to_numeric(zda['Timestamp [µs]'], errors='coerce').values
        nearest = np.abs(zda_raw[:, None] - pps_time[None, :]).argmin(axis=1)
        paired = np.abs(zda_raw - pps_time[nearest]) < 5e5

        zda_gps = pd.to_numeric(zda['Timestamp GPS UTC [ms]'], errors='coerce').values[paired]
        zda_time = pps_time[nearest[paired]]

        clock_fit = np.polyfit(zda_gps, zda_time, 1)
        gga['Timestamp [µs]'] = np.round(np.polyval(clock_fit, gga['GPS Time [ms]'].values))

        # Clean position file 
        # ------------------------------------------------------------------------------------

        position.columns = position.columns.str.strip()
        position = position.drop(columns=[
            'RTK Status', 'Altitude', 'Latitude RTK', 'Longitude RTK',
            'Altitude RTK', 'Next WP', 'ALT:ID', 'OBS:Sectors1', 'OBS:Sectors2',
            'OBS:Sectors3', 'OBS:ID'
        ], errors="ignore")
        
        # Heading contains angles between 0 to -180 and 0 to 180
        position["Heading"] = position["Heading"].apply(lambda x: x + 360 if (isinstance(x, (int, float)) and x < 0) else x)

        # Nearest neighbour time matching from gga to position
        # ------------------------------------------------------------------------------------
        # Both are UTC: GGA in ms since midnight, Elapsed in Unix ms. Matching on coordinates instead picks
        # samples from another moment when the drone hovers or crosses its own path.

        midnight = pd.Timestamp(position['Date'].iloc[0].replace('/', '-'), tz='UTC').value // 10**6
        position['GPS Time [ms]'] = (position['Elapsed'] - midnight).astype('int64')
        gga['GPS Time [ms]'] = gga['GPS Time [ms]'].astype('int64')
        position = position.drop(columns=['Elapsed']).sort_values('GPS Time [ms]')

        # Position is logged at 30 ish Hz
        merged = pd.merge_asof(
            gga.sort_values('GPS Time [ms]'), position,
            on='GPS Time [ms]', direction='nearest', tolerance=100
        ).dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)

        # Convert degrees to meters
        # ------------------------------------------------------------------------------------

        # Conversion done after all the datasets merging to avoid conversion error from degrees to meters
        merged = self.convert_degrees_to_meters(merged)

        # Remove and reorder columns
        # ------------------------------------------------------------------------------------
        drop_cols = ['Latitude', 'Longitude', 'Latitude [°]', 'Longitude [°]']
        aligned_data = (
            merged.rename(columns={"ALT:Altitude": "Altitude"})
                  .drop(columns=drop_cols, errors="ignore")
                  .loc[:, ["Date", "Timestamp [µs]", "GPS Time [ms]", "Time",
                           "Longitude [m]", "Latitude [m]", "Altitude",
                           "Roll", "Pitch", "Heading", "Velocity"]]
        )
        return aligned_data
    
    @staticmethod
    def convert_degrees_to_meters(df: pd.DataFrame) -> pd.DataFrame:
        """
        Function that converts degree coordinates WGS84 (EPSG:4326) to meters UTM31N (EPSG:32631).

        Parameters:
        ------------------------------------------------------------
        gga: pd.DataFrame
            See function format_survey_data
        
        Returns:
        ------------------------------------------------------------
        gga: pd.DataFrame
            gga data but with meter coordinates instead of degrees coordinates
        """

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
        lon_m, lat_m = transformer.transform(df["Longitude [°]"].values, df["Latitude [°]"].values)
        df["Longitude [m]"] = lon_m
        df["Latitude [m]"] = lat_m
        return df


    # Offsets
    # ------------------------------------------------------------------------------------
        
    def apply_drone_center_offset(self, aligned_data: pd.DataFrame, offsets: list[float] | None = None):
        """
        Apply the fixed offset from the GPS antenna to the drone center sensor.
        Offsets are [dx, dy, dz] in the cilidrical frame in meters according to the config file.

        Parameters:
        ------------------------------------------------------------
        aligned_data: pd.DataFrame
            See function format_survey_data
        
        Returns:
        ------------------------------------------------------------
        gps_data: pd.DataFrame
            Original gga data coming from the Skyhub without rotation.
        
        centered: pd.DataFrame
            The skyhub data translated according to the dx, dy, dz offsets. It replaces the 
            original coordinates which are attributed to the Skyhub. Hence the extra gps_data output
            to preserve that data in case it is needed. 

        Notes:
        ------------------------------------------------------------
        If you calculate the euclidean distance at every timestamp between the skyhub and sensor 3,
        it should always be constant. You can verify the 3 dimensional distance by measuring the drone.
        The rotation matrices are present in https://msl.cs.uiuc.edu/planning/node102.html, if interested.
        There's 2 rotations happening for the correction of the coordinates. The first is a "linear" 
        displacement from the position of the skyhub to sensor 3(it's the center of the setup). 
        The second (in apply_center_to_sensor_offset) is a rotation of all the sensors w.r.t. sensor 3.

        """
        
        # This need actual measurements, I assumed that sensor 3 is positioned 10 centimeters in front and to the left of the skyhub
        # and 20 centimeters lower from the original altitude of the skyhub.
        if offsets is None:
            offsets = [0.1, 0.1, -0.2]

        gps_data = aligned_data.copy()
        centered = aligned_data.copy()

        # Angles 
        # ------------------------------------------------------------------------------------
        roll = np.zeros(len(centered))
        pitch = centered['Pitch'].to_numpy()
        heading = centered['Heading'].to_numpy()
        angles = np.vstack([heading, pitch, roll]).T

        # Rotation and offset correction
        # ------------------------------------------------------------------------------------
        rot = R.from_euler('zyx', angles, degrees=True)

        offsets_arr = np.array(offsets)
        rotated = rot.apply(offsets_arr)

        centered[['Longitude [m]', 'Latitude [m]', 'Altitude']] = centered[['Longitude [m]', 'Latitude [m]', 'Altitude']].values + rotated

        return gps_data, centered

    def apply_center_to_sensor_offset(self, centered_data: pd.DataFrame, offsets: np.ndarray) -> pd.DataFrame:
        """
        Apply the offset from the centered coordinates to each individual sensor.
        offsets are the same [dx, dy, dz] according to config file.

        
        Notes:
          Pitch is not taken into account anymore given that the altitude was corrected in the previous rotation matrix.
          The sensor heading is -90 degrees from the drone's heading, so that it aligns on a line. 
        """
        corrected = centered_data.copy()

        # Angles
        # ------------------------------------------------------------------------------------
        roll = corrected['Roll'].to_numpy()
        pitch = np.zeros(len(corrected))
        heading = (-(corrected['Heading'].to_numpy() - 90.0)) % 360.0

        angles = np.vstack([heading, pitch, roll]).T

        # Rotation and offset correction
        # ------------------------------------------------------------------------------------
        rot = R.from_euler('zyx', angles, degrees=True)
        offsets_arr = np.array(offsets)
        rotated = rot.apply(offsets_arr)  # shape (N,3)

        corrected[['Longitude [m]', 'Latitude [m]', 'Altitude']] = corrected[['Longitude [m]', 'Latitude [m]', 'Altitude']].values + rotated
        return corrected

    def generate_sensor_data(self, centered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate for each sensor latitude and longitude coordinates according
        their respective offsets. 

        Parameters:
        ------------------------------------------------------------
        centered_data: pd.DataFrame
            See apply_drone_center_offset

        Returns:

        """
        config_df = self.survey_data.get("config")

        offsets_array = np.array(config_df.values, dtype=float)

        corrected_sensors = []
        for i, offsets in enumerate(offsets_array):
            corrected = self.apply_center_to_sensor_offset(centered_data, offsets)
            corrected = corrected.copy()
            corrected["Sensor ID"] = i + 1
            corrected_sensors.append(corrected)

        all_sensor_coordinates = pd.concat(corrected_sensors, ignore_index=True)
        return all_sensor_coordinates

    # -------------------------
    # Track labelling
    # -------------------------

    def label_off_track_points(self, centered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Label datapoints as on_track if the drone flies along the survey lines at survey speed.
        Adds the columns "Course" and "Track_status" to the dataset.

        Notes:
        ------------------------------------------------------------
        E.g. drone flies at 1.5 m/s, if I give a threshold of 1.25 m/s then all the datapoints
        in the curbs are selected because the drone slows down in those regions to turn.
        Speed alone also keeps the transit to and from the field (3-5 m/s), so a point is only
        on_track if its course is within max_course_deviation of the survey line axis and its
        speed is below 1/min_speed_factor x the survey line speed.

        """
        df = centered_data.copy()

        # Course over ground: direction of travel from the (smoothed) coordinates. The heading can't be used
        # since the drone keeps the same yaw on the way back and flies the return lines backwards.
        x = df["Longitude [m]"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        y = df["Latitude [m]"].rolling(5, center=True, min_periods=1).mean().to_numpy()
        df["Course"] = np.degrees(np.arctan2(np.gradient(x), np.gradient(y))) % 360

        # Survey line axis: most common course (modulo 180) among moving points
        moving = df["Velocity"] >= self.velocity_threshold
        hist, edges = np.histogram(df.loc[moving, "Course"] % 180, bins=36, range=(0, 180))
        self.line_axis = edges[hist.argmax()] + 2.5

        # Parallel to the survey lines, in both directions
        deviation = np.abs((df["Course"] - self.line_axis + 90) % 180 - 90)
        parallel = deviation < self.max_course_deviation

        # Refine the axis (5 degree bins are too coarse for the cross-track offsets in the labelling):
        # circular mean of the doubled course, so both flight directions count for the same axis
        doubled = np.radians(2 * df.loc[moving & parallel, "Course"])
        self.line_axis = np.degrees(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2 % 180
        deviation = np.abs((df["Course"] - self.line_axis + 90) % 180 - 90)
        parallel = deviation < self.max_course_deviation

        # Survey speed on the lines. Transit to and from the field is flown a lot faster.
        self.line_speed = df.loc[moving & parallel, "Velocity"].median()
        line_speed_ok = df["Velocity"] <= self.line_speed/self.min_speed_factor

        df["Track_status"] = "off_track"
        df.loc[moving & parallel & line_speed_ok, "Track_status"] = "on_track"
        return df

    def _label_tracks_for_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the dataset with a column 'Track_number'.

        Notes
        ------------------------------------------------------------
        Removes the off_track datapoints from label_off_track_points first. The remaining on_track
        points are split into segments at every time gap (> gap_threshold) or change of direction.
        When the drone pauses halfway a line and continues, both segments fly the same direction
        and lie on the same line, so they are merged again if their cross-track offset is smaller
        than cross_track_tol. Segments with fewer than min_points samples are set to off-track.
        """
        gap_threshold = 2000   # ms
        min_points = 20        # minimum samples for a valid track (4 s at 5 Hz)
        cross_track_tol = 1.0  # m, max offset between two segments of the same line

        group = group.copy()
        group['Track_number'] = -999

        on_mask = group['Track_status'] == 'on_track'
        if on_mask.sum() == 0:
            return group

        on_group = group.loc[on_mask].sort_values('GPS Time [ms]').reset_index()

        # Direction along the line axis: +1 or -1
        along = np.cos(np.radians(on_group['Course'] - self.line_axis))
        on_group['direction'] = np.sign(along).astype(int)

        # Offset perpendicular to the line axis
        axis = np.radians(self.line_axis)
        on_group['cross_track'] = (on_group['Longitude [m]'] * np.cos(axis)
                                   - on_group['Latitude [m]'] * np.sin(axis))

        # Split at time gaps or direction changes
        new_segment = (on_group['GPS Time [ms]'].diff() > gap_threshold) | (on_group['direction'].diff() != 0)
        on_group['segment'] = new_segment.cumsum()

        # Merge consecutive segments flown in the same direction on the same line
        # (end of the previous segment vs start of the next, where the drone paused)
        segments = on_group.groupby('segment').agg(direction=('direction', 'first'),
                                                   start=('cross_track', lambda c: c.head(5).median()),
                                                   end=('cross_track', lambda c: c.tail(5).median()))
        same_line = ((segments['direction'] == segments['direction'].shift())
                     & ((segments['start'] - segments['end'].shift()).abs() < cross_track_tol))
        segments['track_id'] = (~same_line).cumsum()
        on_group['track_id'] = on_group['segment'].map(segments['track_id'])

        # Short tracks become off-track (-999)
        track_counts = on_group['track_id'].value_counts()
        short_tracks = track_counts[track_counts < min_points].index
        on_group.loc[on_group['track_id'].isin(short_tracks), 'track_id'] = -999

        # Relabel remaining valid tracks sequentially (1..N)
        valid_ids = sorted(on_group.loc[on_group['track_id'] != -999, 'track_id'].unique())
        relabel_map = {old: new for new, old in enumerate(valid_ids, start=1)}
        on_group['track_id'] = on_group['track_id'].map(relabel_map).fillna(-999).astype(int)

        # Map back to the main dataframe
        group.loc[on_group['index'], 'Track_number'] = on_group['track_id'].values
        group.loc[group['Track_number'] == -999, 'Track_status'] = 'off_track'

        return group


    def apply_labelling(self, all_sensor_coordinates: pd.DataFrame) -> pd.DataFrame:
        """
        Label tracks on the center sensor and copy the Track_number to all sensors by GPS time.
        Sets Track_number for on-track rows, and leaves off-track rows with -999.

        Notes
        ------------------------------------------------------------
        All sensors are sampled at the same time, so they belong to the same track at the same moment.
        Labelling each sensor separately can give different tracks, since roll shifts the outer
        sensors' cross-track offset slightly.
        """
        all_sensor_coordinates = all_sensor_coordinates.copy()
        if 'Track_status' not in all_sensor_coordinates.columns:
            raise KeyError("'Track_status' required before labelling. Call label_off_track_points first.")

        sensor_ids = np.sort(all_sensor_coordinates['Sensor ID'].unique())
        center_id = sensor_ids[len(sensor_ids) // 2]
        center = all_sensor_coordinates[all_sensor_coordinates['Sensor ID'] == center_id]
        center = self._label_tracks_for_group(center)

        tracks = center.set_index('GPS Time [ms]')[['Track_number', 'Track_status']]
        labelled = all_sensor_coordinates.drop(columns=['Track_status']).join(tracks, on='GPS Time [ms]')
        return labelled.reset_index(drop=True)

    # -------------------------
    # Magnetometer alignment
    # -------------------------
    def align_magdata_time(self):
        """
        Returns the magdata with numeric timestamps.

        Notes
        ------------------------------------------------------------
        Magdata, PPS and the raw GGA timestamps are all logged on the same internal clock [µs], and
        format_survey_data puts the GGA timestamps on the PPS-corrected version of that clock.
        So the magdata timestamps need no correction to be compared with the GGA timestamps.
        """
        magdata = self.survey_data["magdata"].copy()
        magdata['Timestamp [us]'] = pd.to_numeric(magdata['Timestamp [us]'], errors='coerce')
        magdata = magdata.dropna(subset=['Timestamp [us]'])

        self.survey_data["magdata"] = magdata
        return magdata


    def align_gga_to_mag(self, all_sensor_coordinates: pd.DataFrame, magdata: pd.DataFrame, sensor: int) -> pd.DataFrame:
        """
        For one sensor (1..5), interpolate the GGA-derived fields (5 Hz) onto the magdata timestamps (200 Hz).
        Returns an aligned DataFrame for that sensor, with friendly column names.

        Notes
        ------------------------------------------------------------
        Magdata outside the time span of the GGA data has no position and is dropped.
        Track_number and Track_status are labels, so they are taken from the nearest GGA sample in time.
        """
        sensor_coords = (all_sensor_coordinates[all_sensor_coordinates["Sensor ID"] == sensor]
                         .sort_values('Timestamp [µs]').reset_index(drop=True))

        X = f" B{sensor}x [nT]"
        Y = f" B{sensor}y [nT]"
        Z = f" B{sensor}z [nT]"

        mag_cols = ["Timestamp [us]"]
        for col in (X, Y, Z):
            mag_cols.append(col)

        magdata_sensor = magdata[mag_cols].copy()
        magdata_sensor.rename(columns={X: "Mag-X [nT]", Y: "Mag-Y [nT]", Z: "Mag-Z [nT]"}, inplace=True)
        magdata_sensor["Total field [nT]"] = np.sqrt(
            magdata_sensor["Mag-X [nT]"]**2 + magdata_sensor["Mag-Y [nT]"]**2 + magdata_sensor["Mag-Z [nT]"]**2
        )

        # Interpolate GGA-derived numeric fields onto mag timestamps
        gga_t = sensor_coords['Timestamp [µs]'].astype(float).values
        mag_t = magdata_sensor['Timestamp [us]'].astype(float).values

        fields_to_interp = [
            'GPS Time [ms]', 'Longitude [m]', 'Latitude [m]',
            'Altitude', 'Roll', 'Pitch', 'Heading', 'Velocity'
        ]

        interpolated_data = {}
        for col in fields_to_interp:
            values = sensor_coords[col].astype(float).values
            if col == 'Heading':
                # Unwrap so that e.g. 359 -> 1 degrees isn't interpolated through 180
                values = np.degrees(np.unwrap(np.radians(values)))
            f_interp = interp1d(gga_t, values, kind='linear', bounds_error=False, fill_value=np.nan)
            interpolated_data[col] = f_interp(mag_t)
        interpolated_data['Heading'] = interpolated_data['Heading'] % 360

        # Labels: nearest GGA sample in time
        nearest = np.clip(np.searchsorted(gga_t, mag_t), 1, len(gga_t) - 1)
        nearest -= (mag_t - gga_t[nearest - 1]) < (gga_t[nearest] - mag_t)
        track_status = sensor_coords['Track_status'].values[nearest]
        track_number = sensor_coords['Track_number'].values[nearest]

        # Assemble final aligned DataFrame for this sensor
        aligned_df = magdata_sensor.copy()
        aligned_df["Timestamp [µs]"] = mag_t
        aligned_df["GPS Time [ms]"] = interpolated_data["GPS Time [ms]"]
        aligned_df["Time [hh:mm:ss.sss]"] = pd.to_datetime(aligned_df["GPS Time [ms]"], unit="ms").dt.strftime("%H:%M:%S.%f").str[:-3]
        aligned_df["Sensor ID"] = sensor
        aligned_df["Date"] = sensor_coords["Date"].iloc[0] if "Date" in sensor_coords.columns else ""
        aligned_df["Track_number"] = track_number

        # Numeric fields
        aligned_df["Latitude [m]"] = interpolated_data["Latitude [m]"]
        aligned_df["Longitude [m]"] = interpolated_data["Longitude [m]"]
        aligned_df["Altitude [m]"] = interpolated_data["Altitude"]
        aligned_df["Roll"] = interpolated_data["Roll"]
        aligned_df["Pitch"] = interpolated_data["Pitch"]
        aligned_df["Heading"] = interpolated_data["Heading"]
        aligned_df["Velocity"] = interpolated_data["Velocity"]
        aligned_df["Track_status"] = track_status

        # Reorder columns to clean final output
        final_cols = [
            "Date", "GPS Time [ms]", "Time [hh:mm:ss.sss]",
            "Sensor ID", 
            "Latitude [m]", "Longitude [m]", "Altitude [m]", "Roll", "Pitch", "Heading", "Velocity",
            "Total field [nT]", "Mag-X [nT]", "Mag-Y [nT]", "Mag-Z [nT]",
            "Track_status", "Track_number"
        ]
        # Some columns may be missing due to input differences; keep only available columns preserving order
        final_cols = [c for c in final_cols if c in aligned_df.columns]
        aligned_df = aligned_df[final_cols]

        # Drop magdata recorded before the first or after the last GGA sample (no position)
        aligned_df = aligned_df.dropna(subset=["Latitude [m]", "Longitude [m]"]).reset_index(drop=True)
        return aligned_df

    # -------------------------
    # End-to-end pipeline
    # -------------------------
    def process_all(self) -> pd.DataFrame:
        """
        Run the full pipeline end-to-end and return the final concatenated DataFrame
        with mag-aligned data for all sensors.
        """
        # 1) Load raw survey files
        self.load_survey_files()

        # 2) Format and align GPS/positioning
        aligned_data = self.format_survey_data()

        # 3) Apply GPS -> drone center offset
        _, centered_data = self.apply_drone_center_offset(aligned_data)

        # 4) Label on/off-track on centered_data using velocity threshold
        centered_data = self.label_off_track_points(centered_data)

        # 5) Generate sensor coordinates (center -> each sensor)
        all_sensor_coords = self.generate_sensor_data(centered_data)

        # 6) Label tracks per sensor
        all_sensor_coords = self.apply_labelling(all_sensor_coords)

        # 7) Align magdata timestamps to PPS-corrected ZDA
        magdata = self.align_magdata_time()

        # 8) For each sensor, align its GGA-derived fields to magdata timestamps
        all_aligned = []
        for sensor in range(1, 6):
            aligned_sensor_df = self.align_gga_to_mag(all_sensor_coords, magdata, sensor)
            all_aligned.append(aligned_sensor_df)

        final_dataset = pd.concat(all_aligned, ignore_index=True)
        return final_dataset


def load_survey(survey_index: int, survey_speed: float, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Run the full SurveyDataLoader pipeline for one survey."""
    return SurveyDataLoader(data_dir, survey_index, survey_speed).process_all()


def dms_to_degrees(coordinate: str) -> float:
    """Convert a coordinate like '050d57m31.177s' to decimal degrees."""
    deg, minutes, seconds = re.split("[dms]", coordinate)[:3]
    return float(deg) + float(minutes) / 60 + float(seconds) / 3600


def load_object_locations(path: str | Path = OBJECTS_FILE) -> pd.DataFrame:
    """
    Load the RTK locations of the buried iron objects.

    Returns:
    ------------
    DataFrame with the columns Name, Longitude [m], Latitude [m] (UTM31N)
    """
    objects = pd.read_csv(path, delimiter=",")
    objects.columns = objects.columns.str.strip()
    objects = objects[["Name", "Org_Latitude", "Org_Longitude"]].copy()

    objects["Latitude [°]"] = objects["Org_Latitude"].apply(dms_to_degrees)
    objects["Longitude [°]"] = objects["Org_Longitude"].apply(dms_to_degrees)
    objects = SurveyDataLoader.convert_degrees_to_meters(objects)

    return objects.drop(columns=["Longitude [°]", "Latitude [°]", "Org_Longitude", "Org_Latitude"])
