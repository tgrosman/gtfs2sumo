import copy
import datetime
import os
import random
import re
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import timedelta
from itertools import groupby
from typing import Any, Iterable, Literal, Union

import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sumolib
import traci
from folium.plugins import Geocoder, HeatMap
from shapely.geometry import Point
from shapely.ops import nearest_points

# ===========================================================
# TABLE OF CONTENTS
# ===========================================================
# SECTION 1: GTFS Transformation
# SECTION 2: Coordinates Transformation
# SECTION 3: Route Generation & Correction
# SECTION 4: Visualization
# SECTION 5: Write SUMO Files
# SECTION 6: SUMO Tools
# SECTION 7: Output Analysis
# SECTION 8: XML Manipulation
# ===========================================================


# ===========================================================
# SECTION 1: GTFS Transformation
# ===========================================================


def compute_interval_dict(
    trip_list: pd.DataFrame, generate_csv: bool = False
) -> dict[str, dict[str, Union[float, str]]]:
    """
    Computes the median interval between trips for each main shape, along with the earliest and latest departures.

    Parameters
    ----------
    trip_list : pd.DataFrame
        DataFrame containing trip information including main shape ID and departure times.
    generate_csv : bool, optional
        If True, saves a CSV file with the processed trip list for debugging. Default is False.

    Returns
    -------
    dict[str, dict[str, Union[float, str]]]
        A dictionary with main shape IDs as keys and a dictionary of median interval, earliest departure,
        and last departure times as values.
    """
    # Sort trips by 'main_shape_id' and 'departure_fixed'
    sorted_trip_list = trip_list.sort_values(
        by=["main_shape_id", "departure_fixed"]
    )
    # only consider trips where main shape was choosen
    # TODO: ask Timo if this is feasable
    sorted_trip_list = sorted_trip_list[
        sorted_trip_list["shape_id"] == sorted_trip_list["main_shape_id"]
    ]

    # Remove trips marked as 'trimmed'
    sorted_trip_list = sorted_trip_list[
        ~sorted_trip_list["trip_id"].str.contains(".trimmed")
    ]

    if generate_csv:
        # Save the sorted and filtered trip list to CSV for debugging
        sorted_trip_list.to_csv("trip_list_with_main_shapes.csv", index=False)

    # Calculate the time difference between consecutive trips
    sorted_trip_list["time_diff"] = sorted_trip_list.groupby("main_shape_id")[
        "departure_fixed"
    ].diff()

    # Convert the time difference to minutes
    sorted_trip_list["time_diff"] = (
        sorted_trip_list["time_diff"].dt.total_seconds() / 60
    )

    # Drop NaN values which occur for the first trip in each group
    sorted_trip_list = sorted_trip_list.dropna(subset=["time_diff"])

    # Calculate the median interval between trips for each main shape ID
    median_intervals = (
        sorted_trip_list.groupby("main_shape_id")["time_diff"]
        .median()
        .reset_index()
    )

    # Get the earliest and latest trip for each main shape ID
    earliest_trips = (
        sorted_trip_list.groupby("main_shape_id").first().reset_index()
    )
    latest_trips = (
        sorted_trip_list.groupby("main_shape_id").last().reset_index()
    )

    # Merge the median intervals with earliest and latest departure times
    summary_df = pd.merge(
        median_intervals,
        earliest_trips[["main_shape_id", "departure_fixed"]],
        on="main_shape_id",
    )
    summary_df = pd.merge(
        summary_df,
        latest_trips[["main_shape_id", "departure_fixed"]],
        on="main_shape_id",
        suffixes=("_earliest", "_latest"),
    )

    # Rename columns for clarity
    summary_df.columns = [
        "main_shape_id",
        "median_interval_minutes",
        "earliest_departure",
        "last_departure",
    ]

    # Convert departure times to 'HH:MM:SS' format
    summary_df["earliest_departure"] = summary_df["earliest_departure"].apply(
        lambda x: str(x)[-8:]
    )
    summary_df["last_departure"] = summary_df["last_departure"].apply(
        lambda x: str(x)[-8:]
    )

    # Sort DataFrame by earliest departure time
    summary_df = summary_df.sort_values(by="earliest_departure")

    # Convert the DataFrame to a dictionary with 'main_shape_id' as keys
    summary_dict = summary_df.set_index("main_shape_id").to_dict(orient="index")

    return summary_dict


def compute_median_travel_times(
    shape_id: str,
    gtfs_data: pd.DataFrame,
    main_shape_stop_sequence: pd.DataFrame,
    main_shape_interval_dict: dict,
    offset_sec: float,
    travel_time_until_first_stop: float = 0.0,
) -> pd.DataFrame:
    """
    Computes the median travel times between stops for a given shape and updates the stop sequence DataFrame with these times, based on the GTFS data.

    Parameters
    ----------
    shape_id : str
        The ID of the shape for which to compute travel times.
    gtfs_data : pd.DataFrame
        DataFrame containing GTFS data including stop times and shape information.
    main_shape_stop_sequence : pd.DataFrame
        DataFrame containing the main shape's stop sequence with relevant stop details.
    main_shape_interval_dict : dict
        Dictionary containing interval information, specifically the earliest departure time.
    offset_sec : float
        The offset in seconds to apply to the earliest departure time (stop duration)
    travel_time_until_first_stop : float, optional
        The travel time (in seconds) from the start point to the first stop. Default is 0.0.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame containing the stop sequence with computed median travel times and calculated departure times.
    """
    # Filter relevant trips and sort by trip_id and stop_sequence
    filtered_df: pd.DataFrame = gtfs_data[
        gtfs_data["shape_id"] == shape_id
    ].sort_values(by=["trip_id", "stop_sequence"])

    # Compute travel time between consecutive stops for each trip
    filtered_df["travel_time"] = filtered_df.groupby("trip_id")[
        "departure_fixed"
    ].diff()

    # Drop NaN-values which occur only for the first stop of each trip
    filtered_df = filtered_df.dropna(subset=["travel_time"])

    # Create a new column for the previous stop sequence
    filtered_df["prev_stop_sequence"] = filtered_df["stop_sequence"] - 1

    # Group by stop pairs and compute the median travel time
    stop_pairs = (
        filtered_df.groupby(["prev_stop_sequence", "stop_sequence"])[
            "travel_time"
        ]
        .median()
        .reset_index()
    )

    # Rename columns for clarity
    stop_pairs = stop_pairs.rename(
        columns={
            "prev_stop_sequence": "from_stop_sequence",
            "stop_sequence": "to_stop_sequence",
            "travel_time": "median_travel_time",
        }
    )

    # Create a copy of the stop sequence DataFrame to avoid altering the original
    shape_stop_sequence = copy.deepcopy(main_shape_stop_sequence)

    # Merge median travel times with the stop sequence DataFrame
    shape_stop_sequence = shape_stop_sequence.merge(
        stop_pairs,
        left_on="stop_sequence",
        right_on="to_stop_sequence",
        how="left",
    )

    # Set median travel time to 0 for the first stop
    shape_stop_sequence.loc[
        shape_stop_sequence["stop_sequence"] == 0.0, "median_travel_time"
    ] = pd.Timedelta(0)

    # Convert median travel time to seconds
    shape_stop_sequence["median_travel_time_sec"] = shape_stop_sequence[
        "median_travel_time"
    ].dt.total_seconds()

    # Initialize the departure time in seconds
    shape_stop_sequence["departure_time_sec"] = (
        offset_sec + travel_time_until_first_stop
    )

    # Compute cumulative departure times by adding median travel times
    for i in range(1, len(shape_stop_sequence)):
        shape_stop_sequence.loc[i, "departure_time_sec"] = (
            shape_stop_sequence.loc[i - 1, "departure_time_sec"]
            + shape_stop_sequence.loc[i, "median_travel_time_sec"]
        )

    # Get the earliest departure time from the dictionary and convert it to a timestamp
    earliest_departure_str = main_shape_interval_dict.get(
        "earliest_departure", "1970-01-01 00:00:00"
    )
    earliest_departure = pd.to_datetime(earliest_departure_str)

    # Initialize departure time as human-readable timestamp
    shape_stop_sequence["departure_time_human"] = earliest_departure

    # Compute cumulative departure times as human-readable timestamps
    shape_stop_sequence["departure_time_human"] = (
        shape_stop_sequence["median_travel_time"].cumsum() + earliest_departure
    )

    return shape_stop_sequence


def filter_gtfs(
    routes: pd.DataFrame,
    trips_on_day: pd.DataFrame,
    shapes: pd.DataFrame,
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    net: sumolib.net,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]
]:
    """
    Filters and processes GTFS data based on the network boundaries and shapes.
    Credit: SUMO gtfs2pt.py

    Parameters
    ----------
    routes : pd.DataFrame
        DataFrame containing route information.
    trips_on_day : pd.DataFrame
        DataFrame containing trips for the day.
    shapes : pd.DataFrame
        DataFrame containing shape information. Can be None.
    stops : pd.DataFrame
        DataFrame containing stop information.
    stop_times : pd.DataFrame
        DataFrame containing stop times information.
    net : sumolib.net
        SUMO network object used to get the bounding box.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]
        A tuple containing:
        - The filtered GTFS data DataFrame.
        - A DataFrame with the list of trips.
        - A DataFrame with filtered stops.
        - The shapes DataFrame.
        - A dictionary mapping shape IDs to their main shape ID.

    Notes
    -----
    - The function filters stops and shapes based on their location within the bounding box defined by the SUMO network.
    - The output DataFrame includes filtered GTFS data with relevant columns, and a dictionary mapping secondary shapes to their main shapes.
    """
    # Retrieve the bounding box coordinates from the SUMO network
    net_box = net.getBBoxXY()
    bbox = net.convertXY2LonLat(*net_box[0]) + net.convertXY2LonLat(*net_box[1])

    # Ensure latitude and longitude columns are of float type
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)

    if shapes is not None:
        # Ensure shape points are of float type
        shapes["shape_pt_lat"] = shapes["shape_pt_lat"].astype(float)
        shapes["shape_pt_lon"] = shapes["shape_pt_lon"].astype(float)
        shapes["shape_pt_sequence"] = shapes["shape_pt_sequence"].astype(float)

        # Filter shapes to include only those within the bounding box
        shapes = shapes[
            (bbox[1] <= shapes["shape_pt_lat"])
            & (shapes["shape_pt_lat"] <= bbox[3])
            & (bbox[0] <= shapes["shape_pt_lon"])
            & (shapes["shape_pt_lon"] <= bbox[2])
        ]

    # Merge GTFS data from trips, stop times, stops, and routes
    gtfs_data = pd.merge(
        pd.merge(
            pd.merge(trips_on_day, stop_times, on="trip_id"),
            stops,
            on="stop_id",
        ),
        routes,
        on="route_id",
    )

    # Filter relevant columns in the GTFS data
    gtfs_data = gtfs_data[
        [
            "route_id",
            "shape_id",
            "trip_id",
            "stop_id",
            "route_short_name",
            "route_type",
            "trip_headsign",
            "direction_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "stop_sequence",
            "arrival_fixed",
            "departure_fixed",
            "outside_bbox",
            "edge_id",
            "lane_id",
            "mapped",
        ]
    ]

    # Filter data to include only stops within the bounding box
    gtfs_data = gtfs_data[
        (bbox[1] <= gtfs_data["stop_lat"])
        & (gtfs_data["stop_lat"] <= bbox[3])
        & (bbox[0] <= gtfs_data["stop_lon"])
        & (gtfs_data["stop_lon"] <= bbox[2])
    ]

    # Create a DataFrame with the list of trips with the earliest stop_sequence
    trip_list = gtfs_data.loc[
        gtfs_data.groupby("trip_id").stop_sequence.idxmin()
    ]

    # Add a column for unambiguous stop_id (initially set to None)
    gtfs_data["stop_item_id"] = None

    shapes_dict = {}

    # Determine the most frequent shape for each route and direction
    filtered_stops = (
        gtfs_data.groupby(["route_id", "direction_id", "shape_id"])["shape_id"]
        .size()
        .reset_index(name="counts")
    )
    group_shapes = (
        filtered_stops.groupby(["route_id", "direction_id"])
        .shape_id.aggregate(set)
        .reset_index()
    )
    filtered_stops = filtered_stops.loc[
        filtered_stops.groupby(["route_id", "direction_id"])["counts"].idxmax()
    ][["route_id", "shape_id", "direction_id"]]
    filtered_stops = pd.merge(
        filtered_stops, group_shapes, on=["route_id", "direction_id"]
    )
    # Map each shape to its most frequent main shape
    for row in filtered_stops.itertuples():
        for sec_shape in row.shape_id_y:
            shapes_dict[sec_shape] = row.shape_id_x
    # Create a DataFrame with main shapes and their associated stops
    filtered_stops = gtfs_data[
        gtfs_data["shape_id"].isin(filtered_stops.shape_id_x)
    ]
    filtered_stops = filtered_stops[
        [
            "route_id",
            "shape_id",
            "stop_id",
            "route_short_name",
            "route_type",
            "trip_headsign",
            "direction_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
        ]
    ].drop_duplicates()

    return gtfs_data, trip_list, filtered_stops, shapes, shapes_dict


def get_main_shapes(shape_list: list[str], shapes_dict: dict[str]) -> list[str]:
    """
    Extracts a list of unique main shapes from a dictionary of shapes.

    Parameters
    ----------
    shape_list : list[str]
        A list of shape IDs to be processed.
    shapes_dict : dict[str]
        A dictionary where keys are shape IDs and values are the main shape IDs associated with those shapes.

    Returns
    -------
    list[str]
        A list of unique main shape IDs, preserving the order of their first occurrence in `shape_list`.
    """
    main_shape_ids = []

    for shape in shape_list:
        # Retrieve the main shape ID from the dictionary for the current shape ID
        main_shape = shapes_dict.get(shape)

        # Add the main shape ID to the list if it is not already present
        if main_shape and main_shape not in main_shape_ids:
            main_shape_ids.append(main_shape)

    return main_shape_ids


def get_shape_stop_sequences(
    shapes_list: list, gtfs_data: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Retrieves stop sequences for each shape from GTFS data. The function only retrieves the stop sequence for the first trip associated with each shape ID.
    Stop sequence for all trips with the same shape ID is the same.

    Parameters
    ----------
    shapes_list : list
        A list of shape IDs for which stop sequences need to be retrieved.
    gtfs_data : pd.DataFrame
        A DataFrame containing GTFS data with columns including "shape_id", "trip_id", "stop_sequence", "stop_id",
        "stop_name", "stop_lat", "stop_lon", "lane_id", "edge_id", and "mapped".

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary where each key is a shape ID and each value is a DataFrame containing the stop sequences for that shape.
        The DataFrame includes columns "stop_sequence", "stop_id", "stop_name", "stop_lat", "stop_lon", "lane_id",
        "edge_id", and "mapped", sorted by "stop_sequence".
    """
    shape_stop_sequences = {}

    for shape_id in shapes_list:
        # Filter GTFS data for the current shape_id
        trips_per_shape = gtfs_data[gtfs_data["shape_id"] == shape_id]

        # Select the stop sequence for the first trip associated with this shape_id
        first_trip_id = (
            trips_per_shape["trip_id"].drop_duplicates().values.tolist()[0]
        )
        stop_sequence = trips_per_shape[
            trips_per_shape["trip_id"] == first_trip_id
        ][
            [
                "stop_sequence",
                "stop_id",
                "stop_name",
                "stop_lat",
                "stop_lon",
                "lane_id",
                "edge_id",
                "mapped",
            ]
        ].sort_values(
            by="stop_sequence"
        )

        # Store the DataFrame in the dictionary with shape_id as the key
        shape_stop_sequences[shape_id] = stop_sequence

    return shape_stop_sequences


def get_shape_dict(
    main_shapes: list, shapes: pd.DataFrame
) -> dict[str, tuple[float, float]]:
    """
    Constructs a dictionary mapping shape IDs to lists of geo coordinates.

    Parameters
    ----------
    main_shapes : list
        A list of shape IDs to be processed.
    shapes : pd.DataFrame
        A DataFrame containing shape information with columns "shape_id", "shape_pt_lon", and "shape_pt_lat".

    Returns
    -------
    dict[str, list[float, float]]
        A dictionary where keys are shape IDs and values are lists of [longitude, latitude] pairs representing the shape's coordinates.

    """
    shapes_collection = {}
    for shape_id in main_shapes:
        # Filter the DataFrame for the current shape_id
        shape = shapes.loc[shapes["shape_id"] == shape_id]

        # Create a list of [longitude, latitude] pairs from the filtered shape DataFrame
        tmp_list = [
            (x, y) for x, y in zip(shape["shape_pt_lon"], shape["shape_pt_lat"])
        ]

        # Assign the list to the corresponding shape_id in the dictionary
        shapes_collection[shape_id] = tmp_list

    return shapes_collection


def import_gtfs(
    filter_date: str, gtfsZip
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Imports and processes GTFS data from a ZIP file for a given date ("YYYYMMDD").
    Credit: gtfs2pt.py

    Parameters
    ----------
    filter_date : str
        The date for filtering trips, in the format "YYYYMMDD".
    gtfsZip : zipfile.ZipFile
        A ZipFile object containing GTFS files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame]
        A tuple containing the following pandas DataFrames:
        - routes : pd.DataFrame
            DataFrame containing route information.
        - trips_on_day : pd.DataFrame
            DataFrame containing trip information filtered for the specified date.
        - shapes : pd.DataFrame | None
            DataFrame containing shape information
        - stops : pd.DataFrame
            DataFrame containing stop information.
        - stop_times : pd.DataFrame
            DataFrame containing stop times information, adjusted for a full day.
    """
    # Read GTFS files into DataFrames
    routes = pd.read_csv(gtfsZip.open("routes.txt"), dtype=str)
    stops = pd.read_csv(gtfsZip.open("stops.txt"), dtype=str)
    stop_times = pd.read_csv(gtfsZip.open("stop_times.txt"), dtype=str)
    trips = pd.read_csv(gtfsZip.open("trips.txt"), dtype=str)
    shapes = (
        pd.read_csv(gtfsZip.open("shapes.txt"), dtype=str)
        if "shapes.txt" in gtfsZip.namelist()
        else None
    )
    calendar_dates = pd.read_csv(gtfsZip.open("calendar_dates.txt"), dtype=str)
    calendar = pd.read_csv(gtfsZip.open("calendar.txt"), dtype=str)

    # Ensure 'trip_headsign' and 'route_short_name' columns exist
    if "trip_headsign" not in trips:
        trips["trip_headsign"] = ""
    if "route_short_name" not in routes:
        routes["route_short_name"] = routes["route_long_name"]

    # Convert stop_sequence to float to handle potential float values in the index
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(
        float, copy=False
    )

    # Define the full day as 24 hours
    full_day = pd.to_timedelta("24:00:00")

    # Convert arrival_time and departure_time to timedelta
    stop_times["arrival_fixed"] = pd.to_timedelta(stop_times.arrival_time)
    stop_times["departure_fixed"] = pd.to_timedelta(stop_times.departure_time)

    # Identify and adjust trips spanning over midnight
    fix_trips = stop_times[
        (stop_times["arrival_fixed"] >= full_day)
        & (stop_times["stop_sequence"] == stop_times["stop_sequence"].min())
    ].trip_id.values.tolist()
    stop_times.loc[stop_times.trip_id.isin(fix_trips), "arrival_fixed"] = (
        stop_times.loc[stop_times.trip_id.isin(fix_trips), "arrival_fixed"]
        % full_day
    )
    stop_times.loc[stop_times.trip_id.isin(fix_trips), "departure_fixed"] = (
        stop_times.loc[stop_times.trip_id.isin(fix_trips), "departure_fixed"]
        % full_day
    )

    # Handle extra stop times that exceed the 24-hour period
    extra_stop_times = stop_times.loc[stop_times.arrival_fixed > full_day]
    extra_stop_times.loc[:, "arrival_fixed"] = (
        extra_stop_times["arrival_fixed"] % full_day
    )
    extra_stop_times.loc[:, "departure_fixed"] = (
        extra_stop_times["departure_fixed"] % full_day
    )
    extra_trips_id = extra_stop_times.trip_id.values.tolist()
    extra_stop_times.loc[:, "trip_id"] = (
        extra_stop_times["trip_id"] + ".trimmed"
    )
    stop_times = pd.concat((stop_times, extra_stop_times))

    # Update trips DataFrame to include the trimmed trips
    extra_trips = trips.loc[trips.trip_id.isin(extra_trips_id)]
    extra_trips.loc[:, "trip_id"] = extra_trips["trip_id"] + ".trimmed"
    trips = pd.concat((trips, extra_trips))

    # Define time range for filtering
    time_begin = 0
    time_end = 86400
    time_interval = time_end - time_begin
    start_time = pd.to_timedelta(
        time.strftime("%H:%M:%S", time.gmtime(time_begin))
    )

    # Apply time filtering based on the interval
    if time_interval < 86400 and time_end <= 86400:
        end_time = pd.to_timedelta(
            time.strftime("%H:%M:%S", time.gmtime(time_end))
        )
        stop_times = stop_times[
            (start_time <= stop_times["departure_fixed"])
            & (stop_times["departure_fixed"] <= end_time)
        ]
    elif time_interval < 86400 < time_end:
        end_time = pd.to_timedelta(
            time.strftime("%H:%M:%S", time.gmtime(time_end - 86400))
        )
        stop_times = stop_times[
            ~(
                (stop_times["departure_fixed"] > end_time)
                & (stop_times["departure_fixed"] < start_time)
            )
        ]

    # Filter trips based on the provided date
    weekday = (
        "monday tuesday wednesday thursday friday saturday sunday".split()[
            datetime.datetime.strptime(filter_date, "%Y%m%d").weekday()
        ]
    )
    removed = calendar_dates[
        (calendar_dates.date == filter_date)
        & (calendar_dates.exception_type == "2")
    ]
    services = calendar[
        (calendar.start_date <= filter_date)
        & (calendar.end_date >= filter_date)
        & (calendar[weekday] == "1")
        & (~calendar.service_id.isin(removed.service_id))
    ]
    added = calendar_dates[
        (calendar_dates.date == filter_date)
        & (calendar_dates.exception_type == "1")
    ]
    trips_on_day = trips[
        trips.service_id.isin(services.service_id)
        | trips.service_id.isin(added.service_id)
    ]

    return routes, trips_on_day, shapes, stops, stop_times


def offset_datetime_str(time_str: str, seconds: int, sub: bool = False) -> str:
    """
    Adjusts a time string by adding or subtracting a specified number of seconds.

    Parameters
    ----------
    time_str : str
        A time string in the format "HH:MM:SS" to be adjusted.
    seconds : int
        The number of seconds to add or subtract from the given time.
    sub : bool, optional
        If True, subtracts the seconds from the time. If False, adds the seconds. Default is False.

    Returns
    -------
    str
        The adjusted time as a string in the format "HH:MM:SS".
    """
    # Define the time format
    time_format = "%H:%M:%S"

    # Convert the time string to a datetime object
    time_obj = datetime.datetime.strptime(time_str, time_format)

    if sub:
        # Subtract the specified number of seconds
        new_time_obj = time_obj - timedelta(seconds=seconds)
    else:
        # Add the specified number of seconds
        new_time_obj = time_obj + timedelta(seconds=seconds)

    # Convert the adjusted datetime object back to a string
    new_time_str = new_time_obj.strftime(time_format)

    return new_time_str


def reset_stop_sequence(
    main_shape_ids: list[str],
    main_shape_stop_sequences: dict[str, pd.DataFrame],
) -> None:
    """
    Reset the stop_sequence for each shape_id in main_shape_ids.

    The stop_sequence is reset to a sequential range starting from 0 if the first
    entry is not already 0.0.

    Parameters
    ----------
    main_shape_ids : list[str]
        A list of shape IDs for which the stop sequences need to be reset.
    main_shape_stop_sequences : dict[str, pd.DataFrame]
        A dictionary mapping shape IDs to their corresponding DataFrame
        containing stop sequences.

    Returns
    -------
    None
    """
    counter = 0
    for shape_id in main_shape_ids:
        counter = 0
        for i, row in main_shape_stop_sequences[shape_id].iterrows():
            if counter == 0 and row.stop_sequence != 0.0:
                # print(shape_id)
                new_stop_sequence = [
                    float(x)
                    for x in list(
                        range(0, len(main_shape_stop_sequences[shape_id].index))
                    )
                ]
                main_shape_stop_sequences[shape_id][
                    "stop_sequence"
                ] = new_stop_sequence
            counter += 1
        counter = 0


# ===========================================================
# SECTION 2: Coordinates Transformation (Geo <-> SUMO)
# ===========================================================


def bbox_XY_2_lon_lat(
    bbox: list[tuple[float, float]], net: sumolib.net
) -> list[tuple[float, float]]:
    """
    Convert a bounding box from SUMO network XY coordinates to geographic coordinates (longitude, latitude).

    Parameters
    ----------
    bbox : list[tuple[float, float]]
        A list containing two tuples representing the bounding box in XY coordinates:
        - bbox[0]: Tuple of (X, Y) coordinates for the bottom-left (southwest) corner.
        - bbox[1]: Tuple of (X, Y) coordinates for the top-right (northeast) corner.
    net : sumolib.net
        The SUMO network object used to convert XY coordinates to geographic coordinates (longitude, latitude).

    Returns
    -------
    list[tuple[float, float]]
        A list containing two tuples representing the geographic coordinates (longitude, latitude):
        - [southwest, northeast]:
          - southwest: Tuple (longitude, latitude) for the southwest corner.
          - northeast: Tuple (longitude, latitude) for the northeast corner.

    Notes
    -----
    The coordinate formats in different systems:
    - Folium: (southwest) (northeast) with (latitude, longitude) order.
    - Overpass API: (southwest) (northeast) with (latitude, longitude) order.
    - SUMO: (southwest) (northeast) with (longitude, latitude) order.
    """
    # Convert the southwest (bottom-left) corner from XY to geographic (longitude, latitude)
    southwest = net.convertXY2LonLat(bbox[0][0], bbox[0][1])

    # Convert the northeast (top-right) corner from XY to geographic (longitude, latitude)
    northeast = net.convertXY2LonLat(bbox[1][0], bbox[1][1])

    return [southwest, northeast]


def calculate_bbox_with_shapes(
    net_file: sumolib.net,
) -> tuple[float, float, float, float]:
    """
    Calculate the bounding box of a SUMO network based on the lane shapes. The function extracts lane shapes from the network file, parses the points, and calculates the extreme coordinates
    to determine the bounding box. Lanes without shape data are ignored.

    Parameters
    ----------
    net_file : sumolib.net
        The SUMO network file in XML format that contains lane shape data.

    Returns
    -------
    tuple[float,float,float,float]
        A tuple of four floats representing the bounding box of the network in the format (min_x, min_y, max_x, max_y):
        - min_x: Minimum x-coordinate (longitude).
        - min_y: Minimum y-coordinate (latitude).
        - max_x: Maximum x-coordinate (longitude).
        - max_y: Maximum y-coordinate (latitude).
    """
    # Parse the XML network file
    tree = ET.parse(net_file)
    root = tree.getroot()

    # Initialize bounding box values with extreme initial values
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    # Iterate over all lanes in the network
    for lane in root.findall(".//lane"):
        shape = lane.get("shape")  # Get the shape attribute for each lane

        # Process the shape if it exists
        if shape:
            points = shape.split()  # Split shape into individual points
            for point in points:
                # Extract x and y coordinates from the point
                x, y = map(float, point.split(","))

                # Update the bounding box coordinates based on the current point
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    # Return the calculated bounding box as a tuple (min_x, min_y, max_x, max_y)
    return (min_x, min_y, max_x, max_y)


def _get_center_bbox(bbox: list[tuple[float, float]]) -> list[float, float]:
    """
    Internal helper function that calculates the center point (latitude and longitude) of a bounding box.

    Parameters
    ----------
    bbox : list of tuple of float
        A list containing two tuples representing the bounding box coordinates:
        - bbox[0]: Tuple of (longitude, latitude) representing the bottom-left corner.
        - bbox[1]: Tuple of (longitude, latitude) representing the top-right corner.

    Returns
    -------
    list[float,float]
        A list containing the center point coordinates [center_latitude, center_longitude]:
        - center_lat: The latitude of the center of the bounding box.
        - center_lon: The longitude of the center of the bounding box.
    """
    # Calculate the center latitude by averaging the latitudes of the bottom-left and top-right corners
    center_lat = (bbox[0][1] + bbox[1][1]) / 2

    # Calculate the center longitude by averaging the longitudes of the bottom-left and top-right corners
    center_lon = (bbox[0][0] + bbox[1][0]) / 2

    # Return the center point as a list [center_latitude, center_longitude]
    return [center_lat, center_lon]


def get_edge_from_lon_lat(
    lon: float, lat: float, net: sumolib.net, radius: int = 100
) -> str:
    """
    Get the edge ID of the nearest lane to the given geographic coordinates (longitude, latitude)
    by first converting the coordinates to the network's X and Y positions.

    Parameters
    ----------
    lon : float
        The longitude of the geographic coordinate.
    lat : float
        The latitude of the geographic coordinate.
    net : sumolib.net
        The SUMO network object used to map the geographic coordinates to network coordinates.
    radius : int, optional
        The search radius in meters around the given coordinates to find the nearest lane (default is 100).

    Returns
    -------
    str
        The ID of the edge corresponding to the nearest lane to the given geographic coordinates.
    """
    # Find the nearest lane based on the geographic coordinates, converting them to network X, Y coordinates
    closest_lane = get_lane_from_lon_lat(
        lon, lat, net, radius, return_candidates=False
    )

    # Return the ID of the edge associated with the closest lane
    return closest_lane.getEdge().getID()


def get_lane_from_lon_lat(
    lon: float,
    lat: float,
    net: sumolib.net,
    radius: int = 100,
    return_candidates: bool = False,
) -> Union[str, list[tuple[Any, float]]]:
    """
    Maps the nearest lane to the given geographic coordinates.

    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    net : sumolib.net
        SUMO network object used to convert coordinates and find lanes.
    radius : int, optional
        Radius in meters to search for neighboring lanes. Default is 100 meters.
    return_candidates : bool, optional
        If True, returns a list of candidate lanes with their distances. Default is False.

    Returns
    -------
     Union[str, list[tuple[Any, float]]]
        If `return_candidates` is False, returns the ID of the closest lane as a string.
        If `return_candidates` is True, returns a list of tuples where each tuple contains a lane and its distance from the point.
    """
    # Convert geographic coordinates to network coordinates
    x, y = net.convertLonLat2XY(lon, lat)

    # Find neighboring lanes within the specified radius
    lanes = net.getNeighboringLanes(x, y, radius)
    # If no lanes are found, return None
    if not lanes:
        return None
    # Sort lanes by their distance to the given coordinates
    distances_and_lanes = sorted(lanes, key=lambda lane: lane[1])
    # If return_candidates is True, return the list of candidates
    if return_candidates:
        return distances_and_lanes
    # Return the ID of the closest lane
    closest_lane, _ = distances_and_lanes[0]
    return closest_lane.getID()


def _get_lane_from_XY(
    x: float,
    y: float,
    net: sumolib.net,
    radius: int = 100,
    return_candidates: bool = False,
) -> Union[str, list[tuple[Any, float]]]:
    """
    Internal helper function that maps the nearest lane to the given X, Y coordinates in the SUMO network.

    Parameters
    ----------
    x : float
        X coordinate in the SUMO network's coordinate system.
    y : float
        Y coordinate in the SUMO network's coordinate system.
    net : sumolib.net
        SUMO network object used to find neighboring lanes.
    radius : int, optional
        Radius in meters to search for neighboring lanes. Default is 100 meters.
    return_candidates : bool, optional
        If True, returns a list of candidate lanes with their distances. Default is False.

    Returns
    -------
    Union[str, list[tuple[sumolib.Lane, float]]]
        If `return_candidates` is False, returns the ID of the closest lane as a string.
        If `return_candidates` is True, returns a list of tuples where each tuple contains a lane and its distance from the point.
    """
    # Find neighboring lanes within the specified radius
    lanes = net.getNeighboringLanes(x, y, radius)

    # If no lanes are found, return None
    if not lanes:
        return None

    # Sort lanes by their distance to the given coordinates
    distances_and_lanes = sorted(lanes, key=lambda lane: lane[1])

    # If return_candidates is True, return the list of candidates
    if return_candidates:
        return distances_and_lanes

    # Return the ID of the closest lane
    closest_lane, _ = distances_and_lanes[0]
    return closest_lane.getID()


def get_lane_XY(lane_id: str, net: sumolib.net) -> tuple[float, float]:
    """
    Retrieves the average X and Y coordinates for a given lane ID in the SUMO network.

    Handles normal lanes as well as special junction lanes.

    Parameters
    ----------
    lane_id : str
        The ID of the lane. This can be a normal lane ID or an internal junction lane ID.

    net : sumolib.net
        SUMO network object used to access lane and node information.

    Returns
    -------
    tuple[float, float]
        The average X and Y coordinates of the lane.

    Raises
    ------
    ValueError
        If the lane shape has no points.
    """
    # Determine if the lane ID is for a cluster junction, internal junction, or a normal lane
    if lane_id.startswith(":cluster"):
        # Extract the base ID from the cluster junction ID
        parts = lane_id.split(":")[1].split("_")[:-2]
        base_id = "_".join(parts)
        shape = net.getNode(base_id).getShape()
    elif lane_id.startswith(":"):
        # Extract the base ID from the internal junction ID
        base_id = lane_id.rsplit("_", 2)[0].split(":")[1]
        shape = net.getNode(base_id).getShape()
    else:
        # For normal lanes, get the shape directly
        try:
            shape = net.getLane(lane_id).getShape()
        except:
            edge_id = lane_id.rsplit("_", 1)[0]
            shape = net.getEdge(edge_id).getShape()

    # Ensure that the shape contains points
    if not shape:
        raise ValueError("The lane shape has no points")

    # Calculate the average X and Y coordinates
    avg_x = sum(point[0] for point in shape) / len(shape)
    avg_y = sum(point[1] for point in shape) / len(shape)

    return avg_x, avg_y


def get_edge_XY(edge_id: str, net: sumolib.net) -> tuple[float, float]:
    """
    Retrieves the average X and Y coordinates for a given edge ID in the SUMO network.

    Handles both normal edges and internal junction nodes.

    Parameters
    ----------
    edge_id : str
        The ID of the edge. This can be a normal edge ID or an internal junction node ID.

    net : sumolib.net
        SUMO network object used to access edge and node information.

    Returns
    -------
    tuple[float, float]
        The average X and Y coordinates of the edge.

    Raises
    ------
    ValueError
        If the edge shape has no points.
    """
    # Check if the edge exists in the network
    if net.hasEdge(edge_id):
        shape = net.getEdge(edge_id).getShape()
    else:
        # If the edge doesn't exist, treat the edge_id as a node ID
        shape = net.getNode(edge_id).getShape()

    # Ensure that the shape contains points
    if not shape:
        raise ValueError("The edge or node shape has no points")

    # Calculate the average X and Y coordinates
    avg_x = sum(point[0] for point in shape) / len(shape)
    avg_y = sum(point[1] for point in shape) / len(shape)

    return avg_x, avg_y


def outside_bbox(
    point: tuple[float, float], bbox: list[tuple[float, float]]
) -> bool:
    """
    Checks if a given point is outside the bounding box.

    Parameters
    ----------
    point : tuple[float, float]
        The coordinates of the point to check (latitude, longitude).

    bbox : list[tuple[float, float]]
        The bounding box defined by two tuples:
        - The first tuple represents the lower-left corner (min_lon, min_lat).
        - The second tuple represents the upper-right corner (max_lon, max_lat).

    Returns
    -------
    bool
        True if the point is outside the bounding box, False otherwise.
    """
    lat, lon = point
    min_lon, min_lat = bbox[0]
    max_lon, max_lat = bbox[1]

    return not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon)


def transform_edge_dict_to_geo_dict(
    coord_dict: dict[str, str], net: sumolib.net
) -> dict[str, list[tuple[float, float]]]:
    """
    Transforms a dictionary of edge IDs into geographic coordinates.

    Parameters
    ----------
    coord_dict : dict[str, str]
        A dictionary where the keys are edge IDs and the values are space-separated strings of edge IDs.

    net : sumolib.net
        The SUMO network object.

    Returns
    -------
    dict[str, list[tuple[float, float]]]
        A dictionary where the keys are edge IDs and the values are lists of geographic coordinates (latitude, longitude).
    """
    transformed_geo_dict = {}

    for k, mappedRoute in coord_dict.items():
        geo_sumo_route = transform_edge_to_geo_points(mappedRoute, net)
        transformed_geo_dict[k] = geo_sumo_route

    return transformed_geo_dict


def transform_edge_to_geo_points(
    mappedRoute: str, net: sumolib.net
) -> list[tuple[float, float]]:
    """
    Converts a route represented by a string of edge IDs into geographic coordinates.

    Parameters
    ----------
    mappedRoute : str
        A space-separated string of edge IDs.

    net : sumolib.net
        The SUMO network object containing edge information.

    Returns
    -------
    list[tuple[float, float]]
        A list of geographic coordinates (latitude, longitude) corresponding to the route.
    """
    geo_sumo_route = []

    # Split the route into edge IDs
    edge_ids = mappedRoute.split(" ")

    # Retrieve the shape of each edge and convert it to geographic coordinates
    for edge_id in edge_ids:
        edge = net.getEdge(edge_id)
        if edge:
            shape = edge.getShape()
            geo_sumo_route.extend(
                [net.convertXY2LonLat(x, y) for x, y in shape]
            )

    return geo_sumo_route


# ===========================================================
# SECTION 3: Route Generation & Correction
# ===========================================================


def adjust_stop_length(
    lane_id: str,
    start_pos: float,
    end_pos: float,
    net: sumolib.net,
    orig_stop_length: float = 13.0,
    new_stop_length: float = 26.0,
) -> tuple[float, float]:
    """
    Adjust the start and end positions of a bus stop on a lane to accommodate a new stop length.

    Parameters
    ----------
    lane_id : str
        The ID of the lane on which the bus stop is located.
    start_pos : float
        The original starting position of the bus stop on the lane (in meters).
    end_pos : float
        The original ending position of the bus stop on the lane (in meters).
    net : sumolib.net
        The SUMO network object used to retrieve lane information.
    orig_stop_length : float, optional
        The original length of the bus stop (in meters). Default is 13.0 meters.
    new_stop_length : float, optional
        The desired new length of the bus stop (in meters). Default is 26.0 meters.

    Returns
    -------
    tuple[float, float]
        A tuple containing the adjusted start and end positions of the bus stop (in meters).

    Notes
    -----
    - If the lane length is less than the new stop length, the stop will span the entire lane.
    - If the stop is at the beginning or end of the lane, it will be adjusted accordingly.
    - The function ensures that the adjusted stop does not extend beyond the lane's boundaries.

    """
    # print(start_pos, end_pos)  # Debug: Original start and end positions

    lane_length = net.getLane(lane_id).getLength()

    # Case where the lane is shorter than the new stop length
    if lane_length < new_stop_length:
        # print("lane")  # Debug: Lane is shorter than new stop length
        new_start_pos = 0.0
        new_end_pos = lane_length
        # print(new_start_pos, new_end_pos)  # Debug: Adjusted start and end positions
        return new_start_pos, new_end_pos

    orig_center_point = round(start_pos + orig_stop_length / 2, 2)

    # Case where the bus stop is at the beginning of the lane
    if start_pos == 0.0:
        # print("if1")  # Debug: Bus stop is at the beginning of the lane
        new_start_pos = 0.0
        new_end_pos = new_stop_length
        # print(new_start_pos, new_end_pos)  # Debug: Adjusted start and end positions
        return new_start_pos, new_end_pos

    # Case where the bus stop is at the end of the lane
    elif end_pos == lane_length:
        # print("elif")  # Debug: Bus stop is at the end of the lane
        new_start_pos = round(end_pos - new_stop_length, 2)
        new_end_pos = end_pos
        # print(new_start_pos, new_end_pos)  # Debug: Adjusted start and end positions
        return new_start_pos, new_end_pos

    # Case where the bus stop is in the middle of the lane
    else:
        # print("else")  # Debug: Bus stop is in the middle of the lane
        new_start_pos = round(orig_center_point - new_stop_length / 2, 2)
        new_end_pos = round(orig_center_point + new_stop_length / 2, 2)
        # print("New positions", new_start_pos, new_end_pos)  # Debug: New positions before boundary checks

        # Adjust if the new start position is less than 0
        if new_start_pos < 0.0:
            # print("new_start_pos < 0")  # Debug: New start position is less than 0
            new_start_pos = start_pos
            new_end_pos = round(start_pos + new_stop_length, 2)
            # print("new_start_pos < 0", new_start_pos, new_end_pos)  # Debug: Adjusted start and end positions
            return new_start_pos, new_end_pos

        # Adjust if the new end position exceeds the lane length
        if new_end_pos > lane_length:
            # print("new_end_pos > lane_length")  # Debug: New end position exceeds lane length
            new_end_pos = lane_length
            new_start_pos = round(lane_length - new_stop_length, 2)
            # print("new_end_pos > lane_length", new_start_pos, new_end_pos)  # Debug: Adjusted start and end positions
            return new_start_pos, new_end_pos

        # print(new_start_pos, new_end_pos)  # Debug: Final adjusted start and end positions
        return new_start_pos, new_end_pos


def _check_boundaries(
    before_edge_index: int, after_edge_index: int, route: list
) -> tuple[int, int]:
    """
    This internal helper function ensures that the indices for edges in a route stay within the valid boundaries of the route list.

    Parameters
    ----------
    before_edge_index : int
        The index of the edge before the correction point in the route list.
    after_edge_index : int
        The index of the edge after the correction point in the route list.
    route : list
        The list of edges representing the route.

    Returns
    -------
    tuple[int, int]
        A tuple containing the adjusted indices for the edges, ensuring they are within the valid range of the route list.

    Notes
    -----
    - If `before_edge_index` is less than 0, it is adjusted to 0.
    - If `after_edge_index` is greater than or equal to the length of the route list, it is adjusted to the last valid index (`len(route) - 1`).
    """
    # Ensure before_edge_index is not less than 0
    if before_edge_index < 0:
        before_edge_index = 0

    # Ensure after_edge_index is not greater than the last index of the route list
    if after_edge_index >= len(route):
        after_edge_index = len(route) - 1

    return before_edge_index, after_edge_index


def _check_incoming(
    route_candidates: list, curr_edge: str, net: sumolib.net
) -> bool:
    """
    This internal helper function checks if any of the route candidates are incoming edges to the specified current edge.

    Parameters
    ----------
    route_candidates : list
        A list of edge IDs that are potential candidates for being part of the route.
    curr_edge : str
        The ID of the current edge for which incoming edges are being checked.
    net : sumolib.net
        The SUMO network object used to retrieve edge information.

    Returns
    -------
    bool
        True if any of the route candidates are incoming edges to the current edge, False otherwise.

    Notes
    -----
    - This function uses the SUMO network object to get the list of incoming edges for the specified current edge.
    - It then checks if any of the route candidates are among these incoming edges.
    """
    # Retrieve the list of incoming edges for the current edge
    incoming_edges = net.getEdge(curr_edge).getIncoming()

    # Check if any of the route candidates are in the list of incoming edges
    for edge in route_candidates:
        if edge in incoming_edges:
            return True

    return False


def correct_start(
    shape_id: str,
    from_edge_id: str,
    to_edge_id: str,
    to_edge_index: str,
    route: list[str],
    search_depth_index: int,
    net: sumolib.net,
) -> tuple[None, None] | tuple[list[str], list[str]]:
    """
    Corrects the start of a route if an error is raised by Routechecker, typically due to a U-turn. Find the next possible pair of edges that have a valid connection.
    Update the route and return the corrected route and a list of cutoff edges.

    Parameters
    ----------
    shape_id : str
        The ID of the shape for which the route is being corrected.
    from_edge_id : str
        The ID of the starting edge of the problematic route segment.
    to_edge_id : str
        The ID of the ending edge of the problematic route segment.
    to_edge_index : str
        The index of the ending edge in the route list.
    route : list[str]
        The current route list that needs correction.
    search_depth_index : int
        The index up to which connections are searched.
    net : sumolib.net
        The SUMO network object used to retrieve edge information.

    Returns
    -------
    tuple[None, None] | tuple[list[str], list[str]]
        A tuple containing:
        - The corrected route list.
        - A list of cutoff edges if a correction was made, or None if no correction was needed.

    Raises
    ------
    ValueError
        If a connection between `from_edge_id` and `to_edge_id` already exists, indicating a potential U-turn.

    Notes
    -----
    - The function checks if the connection between `from_edge_id` and `to_edge_id` is valid.
    - If valid, it searches for the next possible edge in the route that has a valid connection.
    - If a valid connection is found, it updates the route and returns the corrected route and a list of cutoff edges.
    - If no valid connection is found, it returns None for both the route and cutoff edges.
    """
    # Check if connection between from_edge and to_edge exist. If true, raise Error
    if net.getEdge(from_edge_id) in net.getEdge(to_edge_id).getIncoming():
        raise ValueError(
            "Connection between from_edge and to_edge already exist."
        )

    # Create two lists that can be accessed via a common counter. The first represents the possible to_edges and the other list represents the possible next_edges.
    pos_to_edge_index = list(range(int(to_edge_index), search_depth_index))
    pos_next_edge_index = list(
        range(int(to_edge_index) + 1, search_depth_index + 1)
    )

    next_edge_index = None
    i = 0
    while i < len(pos_next_edge_index):
        # Get edge if a connection exist between both edges.
        next_edge = _search_connection(
            route[pos_to_edge_index[i]], route[pos_next_edge_index[i]], net
        )
        # If a connection exist between the to_edge and the next_edge on the route, set flag and exit loop.
        if next_edge:
            # Set the index of next_edge
            next_edge_index = pos_next_edge_index[i]
            # Exit the loop
            break
        # If not add 1 to the common counter
        i += 1

    # If no connection exists for any possibility, print shape_id and return None for all return values.
    if not next_edge_index:
        # print("Bazinga", shape_id)
        return None, None

    # If a connection exists, update the route, and create a list of cutoff_edges. Important to replace bus stops.
    if next_edge_index:
        cutoff_edges = route[: next_edge_index - 1]
        route = route[next_edge_index - 1 :]

    return route, cutoff_edges


def correct_end(
    shape_id: str,
    from_edge_id: str,
    from_edge_index: int,
    to_edge_id: str,
    route: list,
    search_depth_index: int,
    net: sumolib.net,
) -> tuple[None, None] | tuple[list[str], list[str]]:
    """
    Corrects the end of a route if an error is raised by the route checker, typically due to a U-turn.Find the next possible pair of edges that have a valid connection.
    Update the route and return the corrected route and a list of cutoff edges.

    Parameters
    ----------
    shape_id : str
        The ID of the shape for which the route is being corrected.
    from_edge_id : str
        The ID of the starting edge of the problematic route segment.
    from_edge_index : int
        The index of the starting edge in the route list.
    to_edge_id : str
        The ID of the ending edge of the problematic route segment.
    route : list
        The current route list that needs correction.
    search_depth_index : int
        The index up to which connections are searched.
    net : sumolib.net
        The SUMO network object used to retrieve edge information.

    Returns
    -------
    tuple[None, None] | tuple[list[str], list[str]]
        A tuple containing:
        - The corrected route list.
        - A list of cutoff edges if a correction was made, or None if no correction was needed.

    Raises
    ------
    ValueError
        If a connection between `from_edge_id` and `to_edge_id` already exists, indicating a potential U-turn.
    """
    # Check if connection between from_edge and to_edge exist. If true, raise Error
    if net.getEdge(from_edge_id) in net.getEdge(to_edge_id).getIncoming():
        raise ValueError("Connection between from_edge and to_edge exist.")

    pos_to_edge_index = list(
        range(len(route) - search_depth_index, int(from_edge_index) + 1)
    )[::-1]
    pos_next_edge_index = list(
        range(len(route) - 1 - search_depth_index, int(from_edge_index))
    )[::-1]

    next_edge_index = None
    i = 0
    while i < len(pos_next_edge_index):
        next_edge = _search_connection(
            route[pos_next_edge_index[i]], route[pos_to_edge_index[i]], net
        )
        # If a connection exist between the to_edge and the next_edge on the route, set flag and exit loop.
        if next_edge:
            next_edge_index = pos_to_edge_index[i]
            break
        i += 1

    if not next_edge_index:
        # print("Bazinga", shape_id)
        return None, None

    if next_edge_index:
        cutoff_edges = route[next_edge_index + 1 :]
        route = route[: next_edge_index + 1]

    return route, cutoff_edges


def create_human_readable_bus_line_info(
    trip_list: pd.DataFrame,
    main_shape_ids: list[str],
    completeness_dict: dict[str, float],
) -> dict[str, dict[str, str, str, str, float]]:
    """
    Create a DataFrame with human-readable bus line info including route string, completeness, bus line name, and direction.

    Parameters
    ----------
    trip_list : pd.DataFrame
        DataFrame containing trip information.
    main_shape_ids : list[str]
        List of shape IDs to filter the trips.

    Returns
    -------
    dict[str, dict[str, str,str,str]
        Dict holding human-readable bus line info with shape ID as key.
    """
    # Filter the DataFrame to only include the main shapes
    filtered_df = trip_list[trip_list["shape_id"].isin(main_shape_ids)]
    # Create a reduced DataFrame with only the necessary columns
    reduced_df = (
        filtered_df[
            ["shape_id", "route_short_name", "direction_id", "trip_headsign"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Create a dictionary to map route_short_name and direction_id to trip_headsign
    trip_headsign_map = _create_trip_headsign_map(reduced_df)
    # Create the name_str column with a human-readable representation of the route
    reduced_df["name_str"] = reduced_df.apply(
        lambda row: _create_name_str(
            trip_headsign_map, row.route_short_name, row.direction_id
        ),
        axis=1,
    )
    # Create completeness column
    reduced_df["completeness"] = reduced_df["shape_id"].map(completeness_dict)
    # Convert DataFrame to dictionary with shape_id as key
    bus_line_info_dict = reduced_df.set_index("shape_id").to_dict(
        orient="index"
    )

    return bus_line_info_dict


def _create_name_str(
    trip_headsign_map: dict[str, dict[str, str]],
    route_short_name: str,
    direction_id: str,
) -> str:
    """
    Create a human-readable string for a bus line.

    Parameters
    ----------
    trip_headsign_map : dict
        Dictionary mapping (route_short_name, direction_id) to trip_headsign.
    route_short_name : str
        Route short name.
    direction_id : str
        Direction ID.

    Returns
    -------
    str
        Human-readable name for the bus line.
    """
    route = route_short_name
    direction = direction_id
    # Check if the route is in the map
    if route in trip_headsign_map:
        # Check if both directions are available and create the string accordingly
        if "1" in trip_headsign_map[route] and "0" in trip_headsign_map[route]:
            if direction == "1":
                return f"{route}: {trip_headsign_map[route]['0']} =&gt; {trip_headsign_map[route]['1']}"
            elif direction == "0":
                return f"{route}: {trip_headsign_map[route]['1']} =&gt; {trip_headsign_map[route]['0']}"
        # If only one direction is available, use that. Relevant for circular routes and one-way routes.
        else:
            # If no pair exists, use the trip_headsign for the given direction
            return f"{route}: {trip_headsign_map[route].get(direction)}"

    # If the route is not in the map, return an empty string
    return ""


def create_routechecker_df(
    routechecker_output: list[tuple[str, str, str, str]], routes: dict[str, str]
) -> pd.DataFrame:
    """
    Creates a DataFrame from the output of Routechecker, with additional details about edge indices and route lengths. The DataFrame is sorted by shape_id for easier readability.
    The indices of the problematic edges in the route are computed and added to the DataFrame.

    Parameters
    ----------
    routechecker_output : list[tuple[str,str,str,str]]
        A list of route checker results, where each entry is a tuple containing:
        - shape_id (str): The ID of the route shape.
        - reason (str): The reason for the route checker error.
        - from_edge (str): The starting edge of the problematic segment.
        - to_edge (str): The ending edge of the problematic segment.
    routes : dict[str,str])
        A dictionary where keys are route IDs (shape_id) and values are strings representing the route as a sequence of edge IDs.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - shape_id (int): The ID of the route shape.
        - reason (str): The reason for the route checker error.
        - from_edge (str): The starting edge of the problematic segment.
        - to_edge (str): The ending edge of the problematic segment.
        - from_index (int): The index of the starting edge in the route.
        - to_index (int): The index of the ending edge in the route.
        - route_length (int): The total number of edges in the route.
    """
    # Create DataFrame from the routechecker output
    routechecker_df = pd.DataFrame(
        routechecker_output,
        columns=["shape_id", "reason", "from_edge", "to_edge"],
    )

    # Convert shape_id to numeric type for sorting and comparison
    routechecker_df["shape_id"] = pd.to_numeric(routechecker_df["shape_id"])

    # Sort DataFrame by shape_id for easier readability
    routechecker_df = (
        routechecker_df.groupby("shape_id")
        .apply(lambda x: x.sort_values(by="shape_id"))
        .reset_index(drop=True)
    )

    # Initialize lists to store additional information
    route_length_list = []
    from_edge_index_list = []
    to_edge_index_list = []

    # Iterate over each row in the DataFrame
    for i, row in routechecker_df.iterrows():
        # Get the route as a list of edge IDs
        split_route = routes[str(row.shape_id)].split(" ")

        # Append the length of the route to the list
        route_length_list.append(len(split_route))

        # Find indices of the from_edge and to_edge in the route
        indices = _find_indices_consecutive_edge_pair(
            row.from_edge, row.to_edge, split_route
        )

        # Append the indices to their respective lists
        from_edge_index_list.append(indices[0])
        to_edge_index_list.append(indices[1])

    # Add the computed information to the DataFrame
    routechecker_df = routechecker_df.assign(from_index=from_edge_index_list)
    routechecker_df = routechecker_df.assign(to_index=to_edge_index_list)
    routechecker_df = routechecker_df.assign(route_length=route_length_list)

    return routechecker_df


def _create_trip_headsign_map(
    reduced_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """
    Create a mapping from route_short_name and direction_id to trip_headsign.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: shape_id, route_short_name, direction_id, trip_headsign.

    Returns
    -------
    dict
        Dictionary mapping (route_short_name, direction_id) to trip_headsign.
    """
    trip_headsign_map = {}

    # Populate the dictionary
    for _, row in reduced_df.iterrows():
        # Extract the necessary information
        route = row["route_short_name"]
        direction = row["direction_id"]
        headsign = row["trip_headsign"]
        # Create a nested dictionary if the route is not already in the map
        if route not in trip_headsign_map:
            trip_headsign_map[route] = {}
        # Add the headsign to the dictionary (both directions if available)
        trip_headsign_map[route][direction] = headsign

    return trip_headsign_map


def cut_shape_points(
    shape_id: str,
    target_point: Point,
    shape_list: dict[str, list[float, float]],
    bbox_lon_lat: list[tuple[float, float]],
    net: sumolib.net,
    start_flag: bool = False,
    generate_map: bool = False,
    directory: str = ".",
) -> list[float]:
    """
    Cuts a list of shape points to include only those points that are before or after a specified target point.

    Parameters
    ----------
    shape_id : str
        Identifier for the shape being processed.
    target_point : shapely.geometry.Point
        The target point used to determine where to cut the shape.
    shape_list : dict[str, list]
        A list of geographic points representing the shape. Each point should be a tuple of (longitude, latitude).
    bbox_lon_lat : list[tuple[float, float]]
        Bounding box coordinates used for map generation. Contains two tuples: (min_lon, min_lat) and (max_lon, max_lat).
    net : sumolib.net
        The SUMO network object used to convert coordinates.
    start_flag : bool, optional
        If True, the shape is cut from the start to the nearest point. If False, cut includes points up to and after the nearest point.
    generate_map : bool, optional
        If True, generates a map with the original and cut shapes for debugging purposes.
    directory
        directory to store generated maps in

    Returns
    -------
    list[float]
        A list of points representing the cut shape.
    """
    # Convert geo coordinates to x, y
    point_list = [
        Point(net.convertLonLat2XY(geo_point[0], geo_point[1]))
        for geo_point in shape_list
    ]

    # Find the nearest Point
    nearest = _find_nearest_point(target_point, point_list)
    # print("Nearest point:", nearest)

    # Get the index of nearest
    nearest_index = point_list.index(nearest)
    # print("nearest_index", nearest_index)

    # If non-mappable stops are the beginning of the route, cut from start
    if start_flag:
        cut_shape = shape_list[nearest_index - 1 :]
        # print(len(cut_shape))
    # If non-mappable stops are at the end of route, cut until.
    else:
        # +1 to include the nearest_index and +1 to include the next point as well. Basically to go past the bus stop.
        cut_shape = shape_list[: nearest_index + 2]
        # print(len(cut_shape))

    # Optional: Generate a map with the old shape and the cut shape. Helpful for debugging.
    if generate_map:
        map_name = f"{shape_id}_cut"
        shape_collection = {shape_id: shape_list, map_name: cut_shape}
        generate_shape_map(
            shape_collection,
            map_name=map_name,
            bbox_lon_lat=bbox_lon_lat,
            path=directory,
            show_flag=True,
        )

    return cut_shape


def detect_pairwise_duplicates(edge_list: list[str]) -> list[tuple[str, str]]:
    """
    Detect pairwise duplicates of consecutive edge IDs in the provided list.

    Parameters
    ----------
    edge_list : list[str]
        A list of edge IDs where consecutive pairs will be analyzed.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples where each tuple represents a consecutive edge pair that appears more than once.
    """
    # Split the route string into a list of edge IDs
    # edges = route_str.split()

    # Create a dictionary to track the pairs and their counts
    pair_counts = {}

    # Iterate through the list and form consecutive pairs
    for i in range(len(edge_list) - 1):
        pair = (edge_list[i], edge_list[i + 1])

        # If the pair is already in the dictionary, increment its count
        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            # If not, add it to the dictionary with count 1
            pair_counts[pair] = 1

    # Filter pairs that appear more than once (exact duplicates)
    duplicates = [pair for pair, count in pair_counts.items() if count > 1]

    return duplicates


def evaluate_shape_to_route_conversion(
    shapes: list[str],
    remaining_faulty_routes: list[str],
    shape_ids_non_mappable: list[str],
) -> dict[str, int]:
    """
    Evaluates the conversion of shapes to routes by categorizing and counting them based on their status.

    Parameters
    ----------
    shapes : list[str]
        List of shape IDs that are considered for conversion.
    remaining_faulty_routes : list[str]
        List of shape IDs that have been identified as having faults.
    shape_ids_non_mappable : list[str]
        List of shape IDs that correspond to non-mappable stops.

    Returns
    -------
    dict[str, int]
        A dictionary with counts of various categories:
        - "total": Total number of shapes.
        - "clean": Number of shapes that are neither faulty nor non-mappable.
        - "remaining_faulty_routes": Number of shapes identified as faulty.
        - "non_mappable_stops": Number of shapes that are non-mappable.
    """
    counts = {
        "total": len(shapes),
        "clean": sum(
            1
            for shape_id in shapes
            if shape_id not in remaining_faulty_routes
            and shape_id not in shape_ids_non_mappable
        ),
        "remaining_faulty_routes": sum(
            1 for shape_id in shapes if shape_id in remaining_faulty_routes
        ),
        "non_mappable_stops": sum(
            1 for shape_id in shapes if shape_id in shape_ids_non_mappable
        ),
    }

    return counts


def execute_routechecker(
    routechecker_path: str, net_path: str, route_path: str
) -> list[tuple[str, str, str, str]]:
    """
    Executes Routechecker and retrieves route validation results from its output. The function uses `subprocess.Popen` to run the routechecker.py with the specified parameters.
    The script’s output is processed to extract relevant route information using regular expressions.

    Parameters
    ----------
    routechecker_path : str
        The file path to the routechecker script.
    net_path : str
        The file path to the network file required by the routechecker.
    route_path : str
        The file path to the routes file to be checked.

    Returns
    -------
    list[tuple[str, str, str, str]]
        A list of tuples containing route validation results. Each tuple contains:
        - Route ID
        - Validation result (e.g., error type)
        - From edge ID
        - To edge ID
    """
    # Define input parameters
    param_flag = "--net"
    # net_path defined above

    verbose = "-v"
    # Execute the script with input parameters
    process = subprocess.Popen(
        [
            "python",
            routechecker_path,
            param_flag,
            net_path,
            route_path,
            verbose,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Get the output and error (if any)
    output, error = process.communicate()
    if error:
        print(error.decode("utf-8"))
        exit(-1)
    # Decode output to string
    output_str = output.decode("utf-8")
    # Retrieve information
    pattern = r"Route (\S+) (\w+) between (-?\S+#?\S*) and (-?\S+#?\S*)"
    # Find all matches of the pattern in the output
    routechecker_output = re.findall(pattern, output_str)

    return routechecker_output


def filter_duplicate_pairs_at_start_end(
    route: list[str], number_of_ids: int
) -> list[str]:
    """
    Remove consecutive duplicate edge pairs from a list of edges at the start/end of route.

    Parameters
    ----------
    edge_ids : list[str]
        A list of edge IDs representing the route.

    Returns
    -------
    list[str]
        A list of edge IDs with pairwise duplicates removed.
    """
    # Check if the list contains at least 'number_of_ids' elements
    if len(route) < number_of_ids:
        raise ValueError("Not enough elements in route")

    # Get the first pairs of the route
    first_pairs = route[
        :number_of_ids
    ]  # Extract every other element from the first n elements
    # Get the last pairs of the route
    last_pairs = route[
        -number_of_ids:
    ]  # Extract every other element from the last n elements
    # Get the rest of the route
    route_rest = route[number_of_ids:-number_of_ids]

    # Check if splitting was correct
    if (len(first_pairs) + len(last_pairs) + len(route_rest)) == len(route):
        cleaned_first_pairs = _remove_duplicate_pairs(first_pairs)
        cleaned_last_pairs = _remove_duplicate_pairs(last_pairs)

    # Concatenate the first and last five pairs
    cleaned_route = cleaned_first_pairs + route_rest + cleaned_last_pairs

    return cleaned_route


def _find_indices_consecutive_edge_pair(
    from_edge: str, to_edge: str, route: list[str]
) -> tuple[int, int] | tuple[None, None]:
    """
    Internal helper function that finds the indices of consecutive edges in a route where a specific pair occurs.

    Parameters
    ----------
    from_edge : str
        The ID of the edge that should be followed by the `to_edge`.
    to_edge : str
        The ID of the edge that should follow the `from_edge`.
    route : list[str]
        A list of edge IDs representing the route.

    Returns
    -------
    tuple[int, int] | tuple[None, None]
        A tuple containing the indices of `from_edge` and `to_edge` if they are consecutive in the route.
        Returns `(None, None)` if the pair is not found.
    """
    # Iterate through the route to find the pair of consecutive edges
    for i in range(len(route) - 1):
        if route[i] == from_edge and route[i + 1] == to_edge:
            return i, i + 1

    # Return None, None if the consecutive edges are not found
    return None, None


def _find_nearest_point(
    target_point: Point, point_list: list[Point]
) -> None | Point:
    """
    Internal helper function that finds the nearest Point in a list of Shapely Points to a given Shapely Point.
    The function uses Shapely's `nearest_points` function to determine the closest point.

    Parameters
    ----------
    target_point : shapely.geometry.Point
        The Shapely Point to which the nearest point is to be found.
    point_list : list[shapely.geometry.Point]
        A list of Shapely Points among which the nearest point to `target_point` is to be found.

    Returns
    -------
    shapely.geometry.Point
        The nearest Point from `point_list` to the `target_point`.
    """
    nearest_point = None
    min_distance = float("inf")

    for point in point_list:
        # Find the nearest points
        nearest = nearest_points(target_point, point)
        distance = target_point.distance(
            nearest[1]
        )  # Distance between target_point and nearest point in point_list

        # Update the nearest point if a closer one is found
        if distance < min_distance:
            min_distance = distance
            nearest_point = nearest[1]

    return nearest_point


def get_change_index(
    pattern: str, stop_sequence_str: str
) -> tuple[Literal[True], int] | tuple[Literal[False], int] | tuple[None, None]:
    """
    Find the index of the first occurrence of a change pattern in a given string.

    This function looks for a change in the string where the pattern transitions from "0" to "1" or from "1" to "0".
    It returns the index of this change and a boolean flag indicating whether the change is "01" or "10".

    Parameters
    ----------
    pattern : str
        A regex pattern to match against the string. If the pattern does not match, the function returns None.
    stop_sequence_str : str
        The string in which to find the change pattern.

    Returns
    -------
    tuple[Literal[True], int] | tuple[Literal[False], int] | tuple[None, None]
        - If the pattern matches and a "01" transition is found, returns (True, index) where index is the position of the "01" transition.
        - If the pattern matches and a "10" transition is found, returns (False, index) where index is the position of the "10" transition.
        - If the pattern does not match or no transition is found, returns (None, None).
    """
    # Set a boolean flag whether the change between 0 and 1 occurs at the beginning or end of the string
    start_flag = False

    # Match the string against the provided pattern
    match = re.fullmatch(pattern, stop_sequence_str)

    if match:
        if "01" in stop_sequence_str:
            # Find the index of the "01" change pattern
            change_index = stop_sequence_str.find("01")
            start_flag = True
            return (start_flag, change_index)

        else:
            # Find the index of the "10" change pattern
            change_index = stop_sequence_str.find("10")
            return (start_flag, change_index)

    # Return None if the string does not match the pattern or if no change is found
    return (None, None)


def _get_lane_on_route(
    edge_lanes_dict: dict[str, list[str]],
    x: float,
    y: float,
    net: sumolib.net,
    radius: int = 30,
    bus_stop_length: float = 13.0,
) -> tuple[str, float, float] | None:
    """
    Internal helper function that finds the lane on the route that corresponds to the given coordinates and bus stop length.

    This function searches for lane candidates near the specified (x, y) coordinates within a given radius.
    It then checks if these candidates belong to the specified edge and calculates the start and end positions
    of the bus stop on the identified lane.

    Parameters
    ----------
    edge_lanes_dict : dict[str, list[str]]
        A dictionary mapping edge IDs to lane IDs.
    x : float
        The x-coordinate of the location to search for lanes.
    y : float
        The y-coordinate of the location to search for lanes.
    net : sumolib.net
        The SUMO network object used to retrieve lane and edge information.
    radius : int, optional
        The radius around the (x, y) coordinates to search for lane candidates (default is 30).
    bus_stop_length : float, optional
        The length of the bus stop (default is 13.0).

    Returns
    -------
    tuple[str, float, float] | None
        - If a suitable lane is found, returns a tuple containing the lane ID, start position, and end position.
        - If no suitable lane is found, returns None.
    """
    # Find lane candidates near the given coordinates
    candidates = _get_lane_from_XY(x, y, net, radius, return_candidates=True)

    if candidates is None:
        print("No candidates found.")
        return None

    # Create a list of tuples containing lane IDs and corresponding edge IDs
    candidates = [
        (c_lane[0].getID(), net.getLane(c_lane[0].getID()).getEdge().getID())
        for c_lane in candidates
    ]
    # print('Candidates:', candidates)

    # Check if any candidate lane belongs to the specified edge
    for candidate in candidates:
        if candidate[1] in edge_lanes_dict.keys():
            start_pos, end_pos = _get_start_and_end_pos(
                x,
                y,
                net.getLane(candidate[0]).getShape(),
                net.getLane(candidate[0]).getLength(),
                bus_stop_length,
            )
            # print(f'Found candidates: {candidate[0], start_pos, end_pos}')
            return [candidate[0], start_pos, end_pos]

    # No suitable lane found
    # print(f'No candidates found.')
    return None


def _get_start_and_end_pos(
    x: float,
    y: float,
    lane_shape: Any,
    lane_length: float,
    bus_stop_length: float = 13.0,
) -> tuple[float, float]:
    """
    Internal helper function that calculates the start and end positions of a bus stop on a lane based on its length and position.

    This function determines the start and end positions of a bus stop given the coordinates of a point
    on the lane, the shape of the lane, and the length of the lane. It adjusts these positions to ensure
    they stay within the boundaries of the lane.

    Parameters
    ----------
    x : float
        The x-coordinate of the point on the lane.
    y : float
        The y-coordinate of the point on the lane.
    lane_shape : Any
        The shape of the lane (used for geometric calculations).
    lane_length : float
        The total length of the lane.
    bus_stop_length : float, optional
        The length of the bus stop (default is 13.0).

    Returns
    -------
    tuple[float, float]
        A tuple containing the start and end positions of the bus stop on the lane.
    """
    # Calculate the offset and distance of the point (x, y) from the lane shape
    lanePos, dist = sumolib.geomhelper.polygonOffsetAndDistanceToPoint(
        (x, y), lane_shape
    )

    # Compute start and end positions based on the bus stop length
    start_pos = round(lanePos - bus_stop_length / 2, 2)
    end_pos = round(lanePos + bus_stop_length / 2, 2)

    # Adjust positions if they fall outside the lane boundaries
    if start_pos < 0.0:
        start_pos = 0.0
        end_pos = bus_stop_length
    elif lane_length < bus_stop_length:
        start_pos = 0.0
        end_pos = lane_length
    elif end_pos > lane_length:
        start_pos = lane_length - bus_stop_length
        end_pos = lane_length

    return start_pos, end_pos


def map_stops_to_route(
    shape_id: str,
    route: list[str],
    cutoff_edges: list[str],
    main_shape_stop_sequence: pd.DataFrame,
    net: sumolib.net,
    radius: int = 100,
    stop_length: float = 13.0,
    cutoff_at_start: bool = False,
    cap_adjustment: int = 4,
) -> dict[str, tuple[str, float, float]] | None:
    """
    Map bus stops to a given route, adjusting for possible route cuts due to U-turns or other errors.

    This function places bus stops along a route, adjusting for cases where the route might be cut
    or altered. If an edge on which a stop was initially placed is cut, it finds the most appropriate
    alternative edge and adjusts the position accordingly.

    Parameters
    ----------
    shape_id : str
        Identifier for the shape or route being processed.
    route : list[str]
        List of edge IDs representing the route.
    cutoff_edges : list[str]
        List of edge IDs that have been cut or altered.
    main_shape_stop_sequence : pd.DataFrame
        DataFrame containing stop information, including lane and edge IDs, and stop sequence.
    net : sumolib.net
        The SUMO network object.
    radius : int, optional
        Radius for finding lane candidates (default is 100).
    stop_length : float, optional
        Length of the bus stop (default is 13.0).
    cutoff_at_start : bool, optional
        Flag indicating if adjustments should be made at the start of the route (default is False).
    cap_adjustment : int, optional
        Adjustment value for handling cutoff and U-turn cases (default is 4).

    Returns
    -------
    dict[str, tuple[str, float, float]]
        A dictionary mapping stop IDs to tuples of (lane ID, start position, end position).
    """
    stops = {}
    counter = 0
    last_stop = None

    for i, row in main_shape_stop_sequence.iterrows():
        # Build a dictionary of edge IDs to their lane IDs
        edge_lanes_dict = {
            edge_id: net.getEdge(edge_id).getLanes() for edge_id in route
        }
        # Find the appropriate lane and position for the stop
        found_lane = _get_lane_on_route(
            edge_lanes_dict,
            row.geometry.x,
            row.geometry.y,
            net,
            radius,
            stop_length,
        )

        # print("found_lane", found_lane)

        if not found_lane:
            # print(
            #     "None value found in shape",
            #     shape_id,
            #     row.stop_id,
            #     row.stop_sequence,
            #     main_shape_stop_sequence["stop_sequence"].max(),
            #     row.edge_id,
            # )
            return None

        # Set the lane to 0. Important for routes with fixed U-turns where a stop on the opposite side now is mapped to the other side. Obviously lane 3 is nearer to the original stop than 0. Thus, adjust it.
        found_lane[0] = found_lane[0].split("_")[0] + "_0"
        # junction_id = lane_id.rsplit("_", 2)[0].split(":")[1]

        # Get lane length in order to place stop at the end in next step.
        lane_length = net.getLane(found_lane[0]).getLength()

        # print("Curr", found_lane, row.edge_id, row.stop_sequence, cutoff_at_start)
        # Handle cases where the route is cut at the start
        if (
            row.edge_id in cutoff_edges
            and cutoff_at_start
            and row.stop_sequence < cap_adjustment
        ):
            if last_stop is not None and last_stop[0] != found_lane[0]:
                found_lane[0] = route[0] + "_0"
                found_lane[1] = last_stop[2]
                found_lane[2] = found_lane[1] + stop_length
            else:
                found_lane[0] = route[0] + "_0"
                found_lane[1] = 0.0
                found_lane[2] = stop_length

            # print('Adjusted lane', found_lane)

        # Handle cases where the stop sequence is near the start of the route
        # 11106, 10548
        elif (
            last_stop is not None
            and row.stop_sequence < cap_adjustment
            and cutoff_at_start
            and row.edge_id not in cutoff_edges
            and row.edge_id not in route
        ):
            incoming_flag = _check_incoming(
                route[: int(len(route) * 20)], found_lane[0].split("_")[0], net
            )
            if not incoming_flag:
                found_lane[0] = route[0] + "_0"
                found_lane[1] = last_stop[2]
                found_lane[2] = found_lane[1] + stop_length

        # Handle cases where the stop is at the start of the route
        # 10099
        elif (
            last_stop is None
            and route.index(found_lane[0].split("_")[0]) >= cap_adjustment
        ):
            found_lane[0] = route[0] + "_0"
            found_lane[1] = 0.0
            found_lane[2] = stop_length

        # Place last stop after second last. Important for cutoff U-turns where the mapped geo location comes before the second last stop.
        if (
            last_stop is not None
            and found_lane[0] == last_stop[0]
            and found_lane[1] < last_stop[2]
            and row.stop_sequence
            >= main_shape_stop_sequence["stop_sequence"].max() - cap_adjustment
        ):
            found_lane[1] = lane_length - stop_length
            found_lane[2] = lane_length
        # Handle cases where the stop sequence is near the end of the route
        # 10994
        elif (
            row.edge_id in cutoff_edges
            and last_stop is not None
            and row.stop_sequence
            >= main_shape_stop_sequence["stop_sequence"].max() - cap_adjustment
            and last_stop[0] == route[-1] + "_0"
        ):
            found_lane[0] = route[-1] + "_0"
            lane_length = net.getLane(found_lane[0]).getLength()
            found_lane[1] = lane_length - stop_length
            found_lane[2] = lane_length
        # Handle cases where the stop is the last in the sequence
        # 9769
        if (
            row.stop_sequence == main_shape_stop_sequence["stop_sequence"].max()
            and not found_lane[0].split("_")[0] in route[-cap_adjustment:]
        ):
            found_lane[0] = route[-1] + "_0"
            lane_length = net.getLane(found_lane[0]).getLength()
            found_lane[1] = lane_length - stop_length
            found_lane[2] = lane_length
        # Adjust the stop position if it overlaps with the previous stop
        if (
            last_stop != None
            and last_stop[0] == found_lane[0]
            and found_lane[1] < last_stop[2]
        ):
            if last_stop[2] + stop_length > lane_length:
                found_lane[1] = last_stop[1]
                found_lane[2] = last_stop[2]
            else:
                found_lane[1] = last_stop[2]
                found_lane[2] = found_lane[1] + stop_length

        stops[f"{shape_id}_{counter}"] = (
            found_lane[0],
            found_lane[1],
            found_lane[2],
        )
        last_stop = (found_lane[0], found_lane[1], found_lane[2])
        counter += 1

    return stops


def remove_consecutive_edge_duplicates(edge_list: list[str]) -> list[str]:
    """
    Remove consecutive duplicate edge IDs from the provided list.

    Parameters
    ----------
    edge_list : list[str]
        A list of edge IDs where consecutive duplicates should be removed.

    Returns
    -------
    list[str]
        A list of edge IDs with consecutive duplicates removed.
    """
    return [edge_id for edge_id, _ in groupby(edge_list)]


def remove_detour(
    edge_list: list[str],
    network: sumolib.net.Net,
    detour_length_threshold: float = 20,
) -> str:
    """
    Remove the section of the route between the first occurrence of a pair of consecutive edges
    and the second occurrence of that pair, inclusive.

    Parameters
    ----------
    edge_list : list[str]
        A list of edge IDs where consecutive duplicates are to be detected and removed.
    detour_length_threshold : int
        the maximum length of the detour. Longer detours will be considered as "real loops"

    Returns
    -------
    str
        The route with the section between duplicate edge pairs removed, joined back into a single string.
    """
    # Split the route string into a list of edge IDs
    # edges = route_str.split()

    # Create a dictionary to track pairs and their positions
    pair_positions = {}

    # Initialize variables to track where the loop starts and ends
    loop_start = -1
    loop_end = -1

    # Iterate through the list and form consecutive pairs
    for i in range(len(edge_list) - 1):
        pair = (edge_list[i], edge_list[i + 1])

        # If the pair is already seen, it means we found the start of a loop
        if pair in pair_positions:
            # The loop should start after the second occurrence of the first edge of the duplicate pair
            loop_start = i + 1
            loop_end = pair_positions[pair] + 1
            break
        else:
            # Store the first occurrence of the pair
            pair_positions[pair] = i

    detour_length = _get_length_of_detour(
        edge_list[loop_end + 1 : loop_start], network
    )
    # If a loop is detected, remove all edges between the second occurrence of the first duplicate pair
    if (
        loop_start != -1
        and loop_end != -1
        and detour_length <= detour_length_threshold
    ):
        cleaned_edges = edge_list[: loop_end + 1] + edge_list[loop_start + 1 :]
    else:
        # If no loop is detected, return the original route
        cleaned_edges = edge_list

    # Join the cleaned edges back into a string
    return " ".join(cleaned_edges)


def _get_length_of_detour(
    detour_edge_list: list[str], net: sumolib.net.Net
) -> float:
    length = 0
    for edge in detour_edge_list:
        if net.hasEdge(edge):
            length += net.getEdge(edge).getLength()
    return length


def _remove_duplicate_pairs(edge_ids: list[str]) -> list[str]:
    """
    Internal helper function that removes consecutive duplicate edge pairs from a list of edges.

    Parameters
    ----------
    edge_ids : list[str]
        A list of edge IDs representing the route.

    Returns
    -------
    list[str]
        A list of edge IDs with pairwise duplicates removed.
    """
    seen_pairs = set()
    cleaned_edges = []

    for i in range(0, len(edge_ids), 2):
        pair = (edge_ids[i], edge_ids[i + 1])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            cleaned_edges.extend(pair)

    return cleaned_edges


def repair_faulty_route_at_junctions(
    from_edge: str, to_edge: str, route: list[str], net: sumolib.net
) -> None | list[str]:
    """
    Repair a faulty route at junctions by identifying and replacing faulty edges.

    This function attempts to repair a route where a U-turn or other issues have resulted
    in faulty edges. It checks if the edges in question are connected via a junction and,
    if so, removes the faulty edges from the route.

    Parameters
    ----------
    from_edge : str
        The edge ID where the route starts to be faulty.
    to_edge : str
        The edge ID where the route ends being faulty.
    route : list[str]
        The list of edge IDs representing the route.
    net : sumolib.net
        The SUMO network object.

    Returns
    -------
    list[str] | None
        The repaired route without the faulty edges, or None if no repair is possible.

    Notes
    -----
    - The function considers U-turns and other types of junction issues.
    - The route is checked for connectivity between edges at the junction to determine
      if removing the faulty edges is a valid solution.
    """
    # Flag to indicate if the issue is a U-turn
    u_turn = False

    # If from_edge is not part of route, return None
    if not from_edge in route:
        return None

    # If to_edge is not part of route, return None
    if not to_edge in route:
        return None

    # Get all u need
    from_edge_from_node = net.getEdge(from_edge).getFromNode()
    from_edge_to_node = net.getEdge(from_edge).getToNode()
    to_edge_from_node = net.getEdge(to_edge).getFromNode()
    to_edge_to_node = net.getEdge(to_edge).getToNode()

    # Set u-turn flag to true if the from_edge goes to the junction, and the to_edge goes to the junction.
    if from_edge_from_node is to_edge_to_node:
        u_turn = True

    # Case for correct route but also regular u-turn (not at a junction per se). Both edges point to/come from the next junction but not the junction of interest.
    elif from_edge_to_node is to_edge_from_node:
        return None

    # Catch these case
    # if (from_edge_to_node is to_edge_to_node) or (from_edge_from_node is to_edge_from_node):
    route_indices = _find_indices_consecutive_edge_pair(
        from_edge, to_edge, route
    )

    if len(route_indices) > 2 or len(route_indices) == 0:
        raise ValueError

    before_edge_index = route_indices[0]
    after_edge_index = route_indices[1]

    # print(f"From: {from_edge} [{before_edge_index}] -> To: {to_edge} [{after_edge_index}]")

    # print(before_edge_index)
    # print(after_edge_index)

    # If both edges are coming from or going to the same junction (U-turn)
    # Remember that those two edges are given by Routechecker. Between them, the route breaks.
    if u_turn:
        # Get the junction
        junction = net.getEdge(from_edge).getFromNode()
        # TODO
        if before_edge_index == None or after_edge_index == None:
            return None
        # Set indices for edges before and after breaking point in route
        before_edge_index = before_edge_index - 1
        after_edge_index = after_edge_index + 1

        # print(f'Before_edge_index: {before_edge_index}')
        # print(f'After_edge_index: {after_edge_index}')

        # Check if index is within boundaries
        before_edge_index, after_edge_index = _check_boundaries(
            before_edge_index, after_edge_index, route
        )

        # print(f'After check: {before_edge_index} and {after_edge_index}')
        # Get edges (str)
        before_edge = route[before_edge_index]
        after_edge = route[after_edge_index]

        # print(f'Before_edge: {before_edge}')
        # print(f'After_edge: {after_edge}')

        # Check whether before_edge and after_edge have got a connection via junction. This obviously implies that both are connected to the junction.
        # We don't need to check whether they are in the sets of incoming and outgoing edges of the junction.
        # Highly unlikely more than one connection between two edges given the junction exist.
        connection = junction.getConnections(
            net.getEdge(before_edge), net.getEdge(after_edge)
        )
        # print(connection)
        # If connection exist, return route without faulty edges
        if connection:
            # print("Case 1")
            # print(f"{before_edge} - {after_edge}")
            # print(f"{from_edge} and {to_edge} removed.")
            return route[: before_edge_index + 1] + route[after_edge_index:]

    else:
        junction_candidates = []

        junction_alt_1 = net.getEdge(from_edge).getFromNode()
        junction_alt_2 = net.getEdge(from_edge).getToNode()

        if junction_alt_1 is junction_alt_2:
            junction_candidates.append(junction_alt_1)

        else:
            junction_candidates.append(junction_alt_1)
            junction_candidates.append(junction_alt_2)

        # Get edges before and after breaking point in edge list (route)
        # from_edge - 1
        before_edge_index = before_edge_index
        # If no U-turn we know 3 edges are involved
        after_edge_index = after_edge_index + 1

        # Check if index is within boundaries
        before_edge_index, after_edge_index = _check_boundaries(
            before_edge_index, after_edge_index, route
        )

        before_edge = route[before_edge_index]
        after_edge = route[after_edge_index]

        # Check whether before_edge and after_edge have got a connection via junction. This obviously implies that both are connected to the junction.
        # We don't need to check whether they are in the sets of incoming and outgoing edges of the junction.
        # Highly unlikely more than one connection between two edges given  the junction exist.
        for candidate in junction_candidates:
            connection = candidate.getConnections(
                net.getEdge(before_edge), net.getEdge(after_edge)
            )
            # print(connection)
            # If connection exist, return route without faulty edges
            if connection:
                # print("Case 2")
                # print(f"{before_edge} - {after_edge}")
                # print(f"{to_edge} removed.")
                return route[: before_edge_index + 1] + route[after_edge_index:]

        # Set
        before_edge_index = before_edge_index - 1
        after_edge_index = after_edge_index - 1

        # print(f'Before_edge_index: {before_edge_index}')
        # print(f'After_edge_index: {after_edge_index}')

        # Check if index is within boundaries
        before_edge_index, after_edge_index = _check_boundaries(
            before_edge_index, after_edge_index, route
        )

        # print(f'After check: {before_edge_index} and {after_edge_index}')

        before_edge = route[before_edge_index]
        after_edge = route[after_edge_index]

        # print(f'Before_edge: {before_edge}')
        # print(f'After_edge: {after_edge}')

        for candidate in junction_candidates:
            connection = candidate.getConnections(
                net.getEdge(before_edge), net.getEdge(after_edge)
            )
            # print(connection)
            # If connection exist, return route without faulty edges
            if connection:
                # print("Case 3")
                # print(f"{before_edge} - {after_edge}")
                # print(f"{from_edge} removed.")
                return route[: before_edge_index + 1] + route[after_edge_index:]

    return None


def _search_connection(
    to_edge_id: str, next_edge_id: int, net: sumolib.net
) -> Any | None:
    """
    Internal helper function that searches for a direct connection between two edges in the given network.

    Parameters
    ----------
    to_edge_id : str
        The ID of the edge from which the connection is being searched.
    next_edge_id : str
        The ID of the edge to which a connection is being searched.
    net : sumolib.net
        The SUMO network object.

    Returns
    -------
    sumolib.net.Edge | None
        The next edge if a direct connection is found, otherwise None.

    Notes
    -----
    This function checks if the `next_edge_id` is directly reachable from `to_edge_id`
    within the network.
    """
    # Get the edges from the network using their IDs
    to_edge = net.getEdge(to_edge_id)
    next_edge = net.getEdge(next_edge_id)
    # Check if next_edge is in the outgoing edges of to_edge
    if next_edge in to_edge.getOutgoing():
        # print(f"Connection found: {to_edge.getID()} -> {next_edge.getID()}")
        return next_edge

    # No direct connection found
    return None


# ===========================================================
# SECTION 4: Visualization
# ===========================================================


def _generate_colors(num_colors: int) -> list[str]:
    """
    Internal helper function that generates a list of distinct colors in hex format.

    Parameters
    ----------
    num_colors : int
        The number of distinct colors to generate.

    Returns
    -------
    list
        A list of hex color codes representing distinct colors.

    Notes
    -----
    - Uses the 'husl' color palette from seaborn, which generates evenly spaced colors around the color wheel.
    - The matplotlib library is used to convert the RGB color values into hex format.
    """
    # Generate a color palette with 'husl' color space and the specified number of colors
    palette = sns.color_palette("husl", num_colors)

    # Convert the RGB color values in the palette to hex format and return as a list
    return [mcolors.to_hex(color) for color in palette]


def generate_completeness_histogram(
    shape_ids: list[str],
    completeness_dict: dict[str, float],
    display_flag: bool = True,
    path: str = "./routes/completeness_histogram.pdf",
    plot_title: str = None,
) -> None:
    """
    Generates and saves a histogram of bus route completeness values, with options to display it.

    The function takes a list of shape IDs and filters the provided completeness dictionary
    to include only the relevant routes. It then generates a histogram showing the distribution
    of completeness values (between 0 and 1) and optionally displays the histogram or saves
    it as a PDF file.

    Parameters
    ----------
    shape_ids : list[str]
        A list of shape IDs corresponding to the bus routes for which completeness
        values are provided.
    completeness_dict : dict[str, float]
        A dictionary where keys are shape IDs and values are float completeness scores
        between 0 and 1 representing the percentage of each route's completeness.
    display_flag : bool, optional
        If True, the histogram will be displayed after it is generated. Default is True.
    path : str, optional
        The file path where the histogram will be saved.
        Default is './routes/completeness_histogram.pdf'.

    Returns
    -------
    None
    """
    filtered_completeness_dict = filter_dict_by_keys(
        completeness_dict, shape_ids
    )
    # Extract float values (completeness) from the dictionary
    completeness_values = list(filtered_completeness_dict.values())

    # Create a histogram and set interval size
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(
        completeness_values,
        bins=np.arange(0.5, 1.05, 0.05),
        edgecolor="black",
        color="tab:blue",
    )

    # Add the number for each bin on top of each bar
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.text(
            patch.get_x() + patch.get_width() / 2,
            height + 1,
            int(count),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Set titles and labels
    plt.title(plot_title, fontsize=16)
    plt.xlabel("Completeness in [0;1]", fontsize=16)
    plt.ylabel("Number of Routes", fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.grid(True)
    # Save histogram
    plt.savefig(path)
    # Display the histogram with counts if flag is set to True
    if display_flag:
        plt.show()


def generate_html_main_shape_stop_sequences(
    main_shape_stop_sequences: dict[str, pd.DataFrame],
    name: str,
    dir_path: str = "./folium_maps/main_shape_stop_sequences/",
) -> None:
    """
    Generates and saves HTML files from the DataFrames of shape stop sequences.

    Parameters
    ----------
    main_shape_stop_sequences : dict[str, pd.DataFrame]
        A dictionary where the keys are shape IDs (as strings) and the values are pd.DataFrames containing
        stop sequences data for the corresponding shapes.
    name : str
        A suffix to be appended to each HTML file's name.
    dir_path : str, optional
        The directory path where the HTML files will be saved, by default "./folium_maps/main_shape_stop_sequences/".

    Returns
    -------
    None
        The function generates and saves an HTML file for each shape ID's stop sequence DataFrame.
    """
    # Iterate through each shape ID and its corresponding DataFrame
    for shape_id, df in main_shape_stop_sequences.items():
        # Convert the DataFrame to HTML and save it with the naming convention: <shape_id>_<name>.html
        df.to_html(os.path.join(dir_path, f"{shape_id}_{name}.html"))


def generate_shape_map(
    main_shapes_collection: dict,
    map_name: str,
    bbox_lon_lat: list[tuple[float, float]],
    path: str = "./folium_maps/",
    show_flag: bool = False,
) -> None:
    """
    Generates a folium map with shapes and corresponding markers for each point of the shape. The map is saved as an HTML file.

    Parameters
    ----------
    main_shapes_collection : dict
        A dictionary where the keys are shape IDs and the values are lists of geo-coordinates (longitude, latitude)
        representing the shape's route.
    map_name : str
        The name of the HTML map file to be saved (without the .html extension).
    bbox_lon_lat : list[tuple[float,float]]
        The bounding box coordinates for the map in the format ((minLon, minLat), (maxLon, maxLat)).
    path : str, optional
        The directory path where the HTML map will be saved, by default "folium_maps/".
    show_flag : bool, optional
        A flag indicating whether the feature group for each shape should be initially visible on the map, by default False.

    Returns
    -------
    None
        The function saves the generated map as an HTML file.
    """
    # Get the center of the bounding box to initialize the map
    bbox_center = _get_center_bbox(bbox_lon_lat)

    # Initialize the folium map with the bounding box center and OpenStreetMap tiles
    map = folium.Map(location=bbox_center, tiles="OpenStreetMap", zoom_start=11)

    # Generate a list of colors for the shapes, then shuffle the list for randomness
    colors = _generate_colors(len(main_shapes_collection.keys()))
    random.shuffle(colors)

    # Iterate through each shape in the collection
    for i, shape_id in enumerate(main_shapes_collection.keys()):
        # Initialize a feature group for the current shape
        group = folium.FeatureGroup(name=f"Shape {shape_id}", show=show_flag)
        color = colors[i]
        counter = 0

        # Add markers for each point in the shape
        for lon, lat in main_shapes_collection[shape_id]:
            folium.CircleMarker(
                location=[
                    lat,
                    lon,
                ],  # Set the marker at the (latitude, longitude) position
                radius=8,  # Set the radius of the marker
                fill=True,  # Ensure the marker is filled
                color=color,  # Assign the shape's color
                weight=10,  # Set the weight (thickness) of the marker's border
                popup=folium.Popup(
                    f"Counter: {counter} Lat: {lat} Lon: {lon}"
                ),  # Popup with counter, lat, and lon
            ).add_to(group)
            counter += 1

        # Prepare coordinates for the PolyLine by swapping (lat, lon) for (lon, lat)
        locationLatLon = [
            coords[::-1] for coords in main_shapes_collection[shape_id]
        ]

        # Add a PolyLine connecting the points for the current shape
        folium.PolyLine(
            locations=locationLatLon,  # List of (lat, lon) tuples
            color=color,  # Use the assigned color
            tooltip=f"Shape: {shape_id}",  # Tooltip that shows when hovered
            weight=5,  # Set the width of the line
            opacity=0.8,  # Set the opacity of the line
        ).add_to(group)

        # Add the feature group to the map
        group.add_to(map)

    # Add a layer control to allow toggling of the different shapes
    folium.LayerControl().add_to(map)

    # Add a search bar (geocoder) to the map for easier navigation
    Geocoder().add_to(map)

    # Save the generated map as an HTML file at the specified path
    map.save(os.path.join(path, f"{map_name}.html"))


def map_routes(
    main_shapes_collection: dict,
    map_name: str,
    bbox_lon_lat: list[tuple[float, float]],
    dir_path: str = "folium_maps/",
    show_flag: bool = False,
) -> None:
    """
    Generates a folium map with routes (represented by PolyLines) and saves it as an HTML file.

    Parameters
    ----------
    main_shapes_collection : dict
        A dictionary where the keys are shape IDs and the values are lists of geo-coordinates (longitude, latitude)
        representing the routes.
    map_name : str
        The name of the HTML map file to be saved (without the .html extension).
    bbox_lon_lat : list[tuple[float,float]]
        The bounding box coordinates for the map in the format ((minLon, minLat), (maxLon, maxLat)).
    path : str, optional
        The directory path where the HTML map will be saved, by default "folium_maps/".
    show_flag : bool, optional
        A flag indicating whether the feature group for each shape should be initially visible on the map, by default False.

    Returns
    -------
    None
        The function saves the generated map as an HTML file.
    """
    # Get the center of the bounding box to initialize the map
    bbox_center = _get_center_bbox(bbox_lon_lat)

    # Initialize the folium map with the bounding box center and OpenStreetMap tiles
    map = folium.Map(location=bbox_center, tiles="OpenStreetMap", zoom_start=11)

    # Generate a list of colors for the shapes, then shuffle the list for randomness
    colors = _generate_colors(len(main_shapes_collection.keys()))
    random.shuffle(colors)

    # Create a feature group for each shape (route)
    for i, shape_id in enumerate(main_shapes_collection.keys()):
        # Initialize a feature group for the current shape
        group = folium.FeatureGroup(name=f"Shape {shape_id}", show=show_flag)
        color = colors[i]

        # Swap the latitude and longitude in the shape's geo-coordinate list to match (lat, lon) format
        location_lat_lon = [
            coords[::-1] for coords in main_shapes_collection[shape_id]
        ]

        # Add a PolyLine representing the route for the current shape
        folium.PolyLine(
            locations=location_lat_lon,
            color=color,
            tooltip=f"Shape: {shape_id}",
            weight=3,
            opacity=1,
        ).add_to(group)

        # Add the feature group to the map
        group.add_to(map)

    # Add a layer control to toggle feature groups on the map
    folium.LayerControl().add_to(map)

    # Add a search bar (geocoder) to the map for easier navigation
    Geocoder().add_to(map)

    # Save the generated map as an HTML file at the specified path
    map.save(os.path.join(dir_path, f"{map_name}.html"))


def map_routes_and_stops(
    main_shapes_collection: dict[str, list[tuple[float, float]]],
    main_shape_stop_sequences: dict[str, pd.DataFrame],
    map_name: str,
    bbox_lon_lat: list[tuple[float, float]],
    path: str = "folium_maps/",
    show_flag: bool = False,
) -> None:
    """
    Generates a folium map with routes and stops and saves it as an HTML file.

    Parameters
    ----------
    main_shapes_collection : dict[str,list[tuple[float,float]]]
        A dictionary where the keys are shape IDs and the values are lists of geo-coordinates (longitude, latitude)
        representing the routes.
    main_shape_stop_sequences : dict[str, pandas.DataFrame]
        A dictionary where the keys are shape IDs and the values are DataFrames containing stop sequences.
        Each DataFrame should have at least the following columns:
        - 'stop_lat': Latitude of the stop.
        - 'stop_lon': Longitude of the stop.
        - 'stop_sequence': The order of the stop along the shape.
    map_name : str
        The name of the HTML map file to be saved (without the .html extension).
    bbox_lon_lat : list[tuple[float,float]]
        The bounding box coordinates for the map in the format ((minLon, minLat), (maxLon, maxLat)).
    path : str, optional
        The directory path where the HTML map will be saved, by default "folium_maps/".
    show_flag : bool, optional
        A flag indicating whether the feature group for each shape should be initially visible on the map, by default False.

    Returns
    -------
    None
        The function saves the generated map as an HTML file.
    """
    # Get the center of the bounding box to initialize the map
    bbox_center = _get_center_bbox(bbox_lon_lat)

    # Initialize the folium map with the bounding box center and OpenStreetMap tiles
    map = folium.Map(location=bbox_center, tiles="OpenStreetMap", zoom_start=11)

    # Generate a list of colors for the shapes, then shuffle the list for randomness
    colors = _generate_colors(len(main_shapes_collection.keys()))
    random.shuffle(colors)

    # Create a feature group for each shape (route)
    for i, shape_id in enumerate(main_shapes_collection.keys()):
        # Initialize a feature group for the current shape
        group = folium.FeatureGroup(name=f"Shape {shape_id}", show=show_flag)
        color = colors[i]

        # Add a marker for each stop in the stop sequence of the current shape
        for _, row in main_shape_stop_sequences[shape_id].iterrows():
            folium.CircleMarker(
                location=[row.stop_lat, row.stop_lon],
                radius=8,
                fill=True,
                color=color,
                weight=10,
                popup=folium.Popup(
                    f"Shape ID: {shape_id}, Stop Sequence: {row.stop_sequence}"
                ),
            ).add_to(group)

        # Swap the latitude and longitude in the shape's geo-coordinate list to match (lat, lon) format
        location_lat_lon = [
            coords[::-1] for coords in main_shapes_collection[shape_id]
        ]

        # Add a PolyLine representing the route for the current shape
        folium.PolyLine(
            locations=location_lat_lon,
            color=color,
            tooltip=f"Shape: {shape_id}",
            weight=3,
            opacity=1,
        ).add_to(group)

        # Add the feature group to the map
        group.add_to(map)

    # Add a layer control to toggle feature groups on the map
    folium.LayerControl().add_to(map)

    # Add a search bar (geocoder) to the map for easier navigation
    Geocoder().add_to(map)

    # Save the generated map as an HTML file at the specified path
    map.save(os.path.join(path, f"{map_name}.html"))


def map_stops(
    bus_stops,
    bboxLonLat: list[tuple[float, float]],
    path: str = "./folium_maps/bbox_bus_stops",
) -> None:
    """
    Creates a folium map of bus stops, categorizing them based on whether they are outside the bounding box,
    mappable onto the SUMO network, or non-mappable. The map is saved as an HTML file.

    Parameters
    ----------
    bus_stops : pandas.DataFrame
        A DataFrame containing bus stop data. The DataFrame should have the following columns:
        - stop_id: The ID of the bus stop.
        - stop_name: The name of the bus stop.
        - stop_lat: The latitude of the bus stop.
        - stop_lon: The longitude of the bus stop.
        - outside_bbox: A boolean indicating if the stop is outside the given bounding box.
        - edge_id: The ID of the corresponding SUMO network edge (can be None for non-mappable stops).
    bboxLonLat : list[tuple[float,float]]
        A tuple representing the bounding box coordinates in the format ((minLon, minLat), (maxLon, maxLat)).
    path : str, optional
        The file path to save the generated HTML map, by default "./folium_maps/bbox_bus_stops".

    Returns
    -------
    None
        The map is saved as an HTML file at the specified path.
    """
    # Get the center of the bounding box to initialize the map
    bbox_center = _get_center_bbox(bboxLonLat)

    # Initialize the folium map with the bounding box center and a default zoom level
    map = folium.Map(location=bbox_center, tiles="OpenStreetMap", zoom_start=11)

    # Feature groups for different categories of bus stops
    bbox_group = folium.FeatureGroup(name="Outside bounding box", show=True)
    not_mapped_group = folium.FeatureGroup(
        name="Non-mappable bus stops", show=True
    )
    mapped_group = folium.FeatureGroup(name="Mapped bus stops", show=True)

    # Lists to collect data for heat maps
    heat_data_bbox = []
    heat_data_non_mappable_stops = []
    heat_data_mappable_stops = []

    # Iterate through each bus stop and categorize it based on the given conditions
    for index, row in bus_stops.iterrows():
        if row.outside_bbox:
            # Bus stops outside the bounding box
            color = "#66c2a5"
            group = bbox_group
            heat_data_bbox.append([row.stop_lat, row.stop_lon])
        elif not row.edge_id:
            # Non-mappable bus stops (no corresponding edge ID)
            color = "#fc8d62"
            group = not_mapped_group
            heat_data_non_mappable_stops.append([row.stop_lat, row.stop_lon])
        else:
            # Mappable bus stops (associated with a SUMO network edge)
            color = "#8da0cb"
            group = mapped_group
            heat_data_mappable_stops.append([row.stop_lat, row.stop_lon])

        # Add a circle marker for each bus stop with a popup containing the stop information
        folium.CircleMarker(
            location=[row.stop_lat, row.stop_lon],
            radius=8,
            fill=True,
            color=color,
            weight=10,
            popup=folium.Popup(
                f"""
                <div style="padding: 10px; text-align: left;">
                    <h6>Bus Stop Info</h6>
                    <p>Stop ID: {row.stop_id}</p>
                    <p>Stop Name: {row.stop_name}</p>
                    <p>Stop Location: {row.stop_lat}, {row.stop_lon}</p>
                </div>
                """
            ),
        ).add_to(group)

    # Draw the bounding box on the map as a rectangle
    bounds_rect = folium.Rectangle(
        bounds=(
            (bboxLonLat[0][1], bboxLonLat[0][0]),
            (bboxLonLat[1][1], bboxLonLat[1][0]),
        ),
        color="black",
        fill=False,
        fill_opacity=0.1,
    )
    map.add_child(bounds_rect)

    # Add the feature groups to the map
    bbox_group.add_to(map)
    not_mapped_group.add_to(map)
    mapped_group.add_to(map)

    # Add heat maps for non-mappable and mappable bus stops
    HeatMap(heat_data_non_mappable_stops).add_to(
        folium.FeatureGroup(
            name="Heat Map: Non-mappable Stops", show=False
        ).add_to(map)
    )
    HeatMap(heat_data_mappable_stops).add_to(
        folium.FeatureGroup(name="Heat Map: Mappable Stops", show=False).add_to(
            map
        )
    )

    # Add a search bar (geocoder) to the map for easier navigation
    Geocoder().add_to(map)

    # Add layer control for toggling between different feature groups and heat maps
    folium.LayerControl().add_to(map)

    # # Print a summary of bus stop counts by category
    # print("Bus stops outside the given bbox:", len(heat_data_bbox))
    # print(
    #     "Bus stops that cannot be mapped onto the SUMO network:",
    #     len(heat_data_non_mappable_stops),
    # )
    # print(
    #     "Bus stops mapped onto the SUMO network:", len(heat_data_mappable_stops)
    # )
    # print(
    #     "Bus stops in total:",
    #     len(heat_data_bbox)
    #     + len(heat_data_non_mappable_stops)
    #     + len(heat_data_mappable_stops),
    # )

    # Save the generated map as an HTML file at the specified path
    map.save(f"{path}.html")


# ===========================================================
# SECTION 5: Write SUMO Files
# ===========================================================


def stops_to_poi_XML(
    bus_stops: pd.DataFrame,
    net: sumolib.net,
    path: str = "./routes/plain_bus_stops.add.xml",
    color: str = "red",
    type: str = "non-mappable",
) -> None:
    """
    Convert non-mappable bus stops to an (additional) XML format suitable for SUMO.

    Parameters
    ----------
    bus_stops : pd.DataFrame
        DataFrame containing information about bus stops. Must include columns:
        - 'stop_id': Unique identifier for each bus stop.
        - 'stop_lon': Longitude of the bus stop.
        - 'stop_lat': Latitude of the bus stop.

    net : sumolib.net
        SUMO network object used to convert geographic coordinates to Cartesian coordinates.

    path : str, optional
        Path to the output XML file. Default is "./routes/plain_bus_stops.add.xml".

    color : str
        color used for poi

    type : str
        Type used for identification

    Returns
    -------
    None
        The function writes the XML data directly to the specified file.
    """
    counter = 0  # Initialize counter for the number of bus stops processed

    # Open the specified XML file in write mode
    with sumolib.openz(path, mode="w") as output_file:
        # Write the XML header
        sumolib.xml.writeHeader(output_file, root="additional")

        # Iterate over each row in the bus_stops DataFrame
        for i, row in bus_stops.iterrows():
            # Convert longitude and latitude to Cartesian coordinates
            x, y = net.convertLonLat2XY(row.stop_lon, row.stop_lat)
            counter += 1  # Increment the counter for each bus stop

            # Extract bus stop information
            poi_id = row.stop_id
            stop_name = row.stop_name
            layer = "10"

            # Write the bus stop information as an XML line
            output_file.write(
                f'    <poi id="{poi_id}" type="{type}" color="{color}" layer="{layer}" x="{x:.2f}" y="{y:.2f}">\n        <param key="stop_name" value="{stop_name}"/>\n    </poi>\n'
            )

        # Write the closing tag for the XML root element
        output_file.write("</additional>\n")

    # # Print the number of non-mappable bus stops processed
    # print(f"Number of non-mappable bus stops: {counter}")


def routes_to_XML(
    sumo_routes: dict[str, str],
    stop_sequences: dict[str, pd.DataFrame],
    stop_duration: float = 20.0,
    path: str = "./routes/plain_routes.rou.xml",
    ignore_bus_stops: bool = False,
    include_vehicles: bool = True,
    start_at_first_stop: bool = False,
    depart: int = 500,
) -> None:
    """
    Generate an XML file describing the routes and vehicles for SUMO simulation.

    Parameters
    ----------
    sumo_routes : dict[str,str]
        Dictionary mapping shape identifiers to a string of SUMO edge IDs.
        Keys are shape IDs, and values are strings representing edges.

    stop_sequences : dict[str,pd.DataFrame]
        Dictionary mapping shape identifiers to DataFrames containing bus stop sequences.
        Keys are shape IDs, and values are DataFrames with bus stop information.

    stop_duration : float, optional
        Duration (in seconds) each bus stop should be held. Default is 20.0.

    path : str, optional
        Path to the output XML file. Default is "./routes/plain_routes.rou.xml".

    ignore_bus_stops : bool, optional
        Flag to indicate whether bus stops should be included in the XML output.
        If True, bus stops are not included. Default is False.

    include_vehicles : bool, optional
        Flag to indicate whether vehicles should be included in the XML output.
        If True, vehicles are included. Default is True.

    start_at_first_stop:
        If true routes will start at the edge of the first bus stop to avoid having to adjust departure times
    depart : int, optional
        Departure time for vehicles (in seconds). Default is 500.

    Returns
    -------
    None
        The function writes the XML data directly to the specified file.
    """
    with sumolib.openz(path, mode="w") as output_file:
        # Write XML header with root element "routes"
        sumolib.xml.writeHeader(output_file, root="routes")

        # Define vehicle type and write it to XML
        output_file.write('    <vType id="bus" vClass="bus"/>\n')

        # Write route definitions
        for shape_id in sumo_routes:
            typ_route = "route"
            route_id = shape_id
            edges = sumo_routes[shape_id]

            if start_at_first_stop:
                edge_id_first_stop = stop_sequences[shape_id].iloc[0]["edge_id"]
                edge_list = edges.split(" ")
                if edge_id_first_stop in edge_list:
                    edge_list = edge_list[edge_list.index(edge_id_first_stop) :]
                    edges = " ".join(edge_list)

            # Write route element with edges
            output_file.write(
                '    <%s id="%s" edges="%s">\n' % (typ_route, route_id, edges)
            )  # noqa

            # Optionally write bus stop information
            if not ignore_bus_stops:
                for i, row in stop_sequences[shape_id].iterrows():
                    output_file.write(
                        '        <stop busStop="%s" duration="%s"/><!-- %s -->\n'
                        % (row.stop_id, stop_duration, row.stop_name)
                    )  # noqa

            output_file.write("    </route>\n")

        # Write vehicle definitions if requested
        if include_vehicles:
            for shape_id in sumo_routes:
                typ_vehicle = "vehicle"
                vehicle_id = f"veh_{shape_id}"
                v_type = "bus"
                route = shape_id
                depart = depart

                # Write vehicle element with route and departure time
                output_file.write(
                    '    <%s id="%s" type="%s" route="%s" departPos="stop" depart="%s"/>\n'
                    % (typ_vehicle, vehicle_id, v_type, route, depart)
                )  # noqa

        # Close the root "routes" element
        output_file.write("</routes>")


def routes_to_XML_flow(
    sumo_routes: dict[str, str],
    stop_sequences: dict[str, pd.DataFrame],
    interval_dict: dict[str, dict[str, Union[float, str]]],
    stop_duration: float = 20.0,
    path: str = "./routes/plain_routes.rou.xml",
    human_readable_info: dict[str, dict[str, str, str, str, float]] = None,
    with_parking: bool = False,
) -> None:
    """
    Generate an XML file describing routes and flows for SUMO simulation.

    Parameters
    ----------
    sumo_routes : dict[str,str]
        Dictionary mapping shape identifiers to a string of SUMO edge IDs.
        Keys are shape IDs, and values are strings representing edges.

    stop_sequences : dict[str,pd.DataFrame]
        Dictionary mapping shape identifiers to DataFrames containing bus stop sequences.
        Keys are shape IDs, and values are DataFrames with bus stop information.

    interval_dict : dict[str, dict[str, Union[float, str]]]
        Dictionary mapping shape identifiers to interval information.
        Includes keys like 'earliest_departure', 'last_departure', and 'median_interval_minutes'.

    stop_duration : int
        Duration (in seconds) each bus stop should be held.

    path : str, optional
        Path to the output XML file. Default is "./routes/plain_routes.rou.xml".

    with_parking : bool, optional
        If true the parking attribute will be added.
    Returns
    -------
    None
        The function writes the XML data directly to the specified file.
    """
    with sumolib.openz(path, mode="w") as output_file:
        # Write XML header with root element "routes"
        sumolib.xml.writeHeader(output_file, root="routes")

        # Define vehicle type and write it to XML
        output_file.write('    <vType id="bus" vClass="bus"/>\n')

        # Write route definitions
        for shape_id in sumo_routes.keys():
            route_id = (
                human_readable_info[shape_id]["route_short_name"]
                + "_"
                + shape_id
            )
            color = "240,215,34"  # https://color-hex.org/color/f0d722
            edges = sumo_routes[shape_id]

            # Write route element with edges
            output_file.write(
                '    <route id="%s" color="%s" edges="%s">\n'
                % (route_id, color, edges)
            )  # noqa

            stop_string = (
                '        <stop busStop="%s" duration="%s" until="%s" parking="true"/><!-- %s -->\n'
                if with_parking
                else '        <stop busStop="%s" duration="%s" until="%s"/><!-- %s -->\n'
            )
            # Write bus stops for the route
            for i, row in stop_sequences[shape_id].iterrows():
                # Write stop information with departure time and duration
                output_file.write(
                    stop_string
                    % (
                        row.stop_id,
                        stop_duration,
                        row.departure_time_sec,
                        row.stop_name,
                    )
                )  # noqa

            output_file.write("    </route>\n")

        # Write flow definitions
        for shape_id in interval_dict:
            if shape_id in sumo_routes.keys():
                route_id = (
                    human_readable_info[shape_id]["route_short_name"]
                    + "_"
                    + shape_id
                )
                v_type = "bus"
                # Calculate the begin and end times for the flow
                begin = interval_dict[shape_id]["adjusted_earliest_departure"]
                end = interval_dict[shape_id]["adjusted_last_departure"]
                period = interval_dict[shape_id]["median_interval_minutes"] * 60
                line = human_readable_info[shape_id]["route_short_name"]

                headsign = human_readable_info[shape_id]["name_str"]
                short_name = human_readable_info[shape_id]["route_short_name"]
                completeness = human_readable_info[shape_id]["completeness"]
                # Write flow element with timing and period
                # output_file.write('    <flow id="%s" type="%s" route="%s" begin="%s" end="%s" period="%s"/>\n' % (id, v_type, route, begin, end, period))  # noqa
                output_file.write(
                    '    <flow id="%s" type="%s" route="%s" begin="%s" end="%s" period="%s" line="%s">\n'
                    % (route_id, v_type, route_id, begin, end, period, line)
                )
                output_file.write(
                    '        <param key="name" value="%s"/>\n' % headsign
                )
                output_file.write(
                    '        <param key="short_name" value="%s"/>\n'
                    % short_name
                )
                output_file.write(
                    '        <param key="completeness" value="%.2f"/>\n'
                    % completeness
                )
                output_file.write("    </flow>\n")

        # Close the root "routes" element
        output_file.write("</routes>")


def stops_to_XML(
    bus_stops: pd.DataFrame, path: str = "./routes/plain_bus_stops.add.xml"
) -> None:
    """
    Generate an XML file describing bus stops for SUMO simulation.

    Parameters
    ----------
    bus_stops : pd.DataFrame
        DataFrame containing bus stop information. Expected columns:
        - 'stop_id': Unique identifier for the bus stop.
        - 'lane_id': ID of the lane where the bus stop is located.
        - 'start_pos': Starting position of the bus stop on the lane (in meters).
        - 'end_pos': Ending position of the bus stop on the lane (in meters).
        - 'stop_name': Name of the bus stop.

    path : str, optional
        Path to the output XML file. Default is "./routes/plain_bus_stops.add.xml".

    Returns
    -------
    None
        The function writes the XML data directly to the specified file.
    """
    counter = 0
    with sumolib.openz(path, mode="w") as output_file:
        # Write XML header with root element "additional"
        sumolib.xml.writeHeader(output_file, root="additional")

        # Iterate over the bus stops DataFrame
        for i, row in bus_stops.iterrows():
            # Only include bus stops with a valid 'lane_id'
            if row.lane_id:
                counter += 1
                id = row.stop_id
                lane = row.lane_id
                start_pos = row.start_pos
                end_pos = row.end_pos
                name = row.stop_name
                typ = "busStop"
                parking_length = (
                    row.parking_length
                    if row.parking_length
                    else max(1, end_pos - start_pos)
                )
                # Write bus stop element with attributes
                output_file.write(
                    '    <%s id="%s" lane="%s" startPos="%.2f" endPos="%.2f" name="%s" friendlyPos="true" parkingLength="%.2f"/>\n'
                    % (typ, id, lane, start_pos, end_pos, name, parking_length)
                )  # noqa

        # Close the root "additional" element
        output_file.write("</additional>\n")

    # Print the number of mapped bus stops
    # print(f"Number of mapped bus stops: {counter}")


# ===========================================================
# SECTION 6: SUMO Tools
# ===========================================================


def _calculate_median_depart_times(
    vehicle_dict: dict[str, list[float]], insertion_times: dict[str, float]
) -> dict[int, float]:
    """
    Internal helper function that calculates the median adjusted departure times for each stop index.

    Parameters
    ----------
    vehicle_dict : dict[str, list[float]]
        A dictionary where keys are vehicle IDs and values are lists of departure times.
    insertion_times : dict[str, float]
        A dictionary where keys are vehicle IDs and values are insertion times.

    Returns
    -------
    dict[int, float]
        A dictionary where keys are stop indices and values are the median adjusted departure times for each stop.
    """
    median_times = {}

    for vehicle_id, depart_times in vehicle_dict.items():
        insert_time = insertion_times[vehicle_id]

        for i, time in enumerate(depart_times):
            adjusted_time = time - insert_time
            if i not in median_times:
                median_times[i] = []
            median_times[i].append(adjusted_time)

    for stop_index, times in median_times.items():
        # Sort adjusted departure times for the current stop
        sorted_times = sorted(times)

        # Calculate median
        median_value = statistics.median(sorted_times)

        # Store the median value for the current stop
        median_times[stop_index] = median_value

    return median_times


def get_median_stop_times(
    num_vehicles: int,
    route_ids_num_stops: dict[str, int],
    rou_file: str,
    sumo_config: str,
    sumo_binary: str,
) -> dict[str, dict[int, float]]:
    """
    Start a SUMO simulation, gather stop times for a specified number of vehicles across different routes,
    calculate median stop times, and return the results.

    Parameters
    ----------
    num_vehicles : int
        Number of vehicles to simulate.
    route_ids_num_stops : dict[str, int]
        Dictionary where keys are route IDs and values are the number of stops in each route.
    rou_file : str
        Path to the SUMO route file.
    sumo_config : str
        Path to the SUMO configuration file.
    sumo_binary : str
        Path to the SUMO executable binary.

    Returns
    -------
    dict[str, dict[int, float]]
        A dictionary where keys are route IDs and values are dictionaries with stop indices as keys and
        median stop times as values.
    """
    # Start simulation
    _start_sumo(sumo_binary, sumo_config)
    # Store stop times
    all_stop_times = {}

    for route, num_stops in route_ids_num_stops.items():

        tmp_dict_stop_times = {}
        tmp_dict_insert_times = {}

        for i in range(0, num_vehicles):
            vehicle_id: str = f"veh_{i}"
            insert_time, stop_times = _get_stop_times_per_route(
                route, vehicle_id, stop_duration=20.0
            )

            if num_stops == len(stop_times):
                tmp_dict_stop_times[vehicle_id] = stop_times
                tmp_dict_insert_times[vehicle_id] = insert_time
            else:
                raise Exception(
                    f"Number of stops is not equal for: {route, num_stops, len(stop_times)}"
                )

        median_departures = _calculate_median_depart_times(
            tmp_dict_stop_times, tmp_dict_insert_times
        )
        # print(f"Route: {route}, Median Stop Times: {median_departures}")
        all_stop_times[route] = median_departures

    return all_stop_times


def get_median_travel_time_until_first_stop(
    num_vehicles: int, route_ids: list[str], sumo_config: str, sumo_binary: str
) -> dict[str, float]:
    """
    Start a SUMO simulation, gather travel times until the first stop for a specified number of vehicles
    across different routes, calculate the median travel times, and return the results.

    Parameters
    ----------
    num_vehicles : int
        Number of vehicles to simulate per route.
    route_ids : list[str]
        List of route IDs to simulate.
    sumo_config : str
        Path to the SUMO configuration file.
    sumo_binary : str
        Path to the SUMO executable binary.

    Returns
    -------
    dict[str, float]
        A dictionary where keys are route IDs and values are the median travel times until the first stop.
    """
    _start_sumo(sumo_binary, sumo_config)
    all_travel_times = {}

    for route in route_ids:
        tmp_travel_times = []
        # Run 'num_vehicles' vehicles per route
        for i in range(0, num_vehicles):
            vehicle_id: str = f"veh_{route}"
            travel_time = _get_travel_time_until_first_stop(route, vehicle_id)
            tmp_travel_times.append(travel_time)
        # Sort list of travel times
        sorted_times = sorted(tmp_travel_times)
        # Get median
        median_value = statistics.median(sorted_times)
        print(f"Route: {route}, Travel time: {median_value}")
        all_travel_times[route] = median_value

    traci.close()
    return all_travel_times


def _get_stop_times_per_route(
    route_id: str, vehicle_id: str, stop_duration: float = 20.0
) -> tuple[float, list[float]]:
    """
    Internal helper functions that simulates a vehicle along a route and get the times at which it stops using TRACI.

    Parameters
    ----------
    route_id : str
        ID of the route to simulate.
    vehicle_id : str
        ID of the vehicle to be added to the simulation.
    stop_duration : float
        Duration that the vehicle stays at each stop (in seconds).

    Returns
    -------
    Tuple[float, List[float]]
        - The simulation time at which the vehicle was inserted.
        - A list of times at which the vehicle stops.
    """
    traci.vehicle.add(vehicle_id, route_id, typeID="bus")
    num_bus_stops = len(traci.vehicle.getStops(vehicle_id))
    # Important to calculate until value
    insert_time = traci.simulation.getTime()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if traci.vehicle.getIDCount() > 0:
            # Get current stop info. Assumes bus doesn't despawn after stop duration.
            stop_info = traci.vehicle.getStops(vehicle_id)
            """ 
            if len(stop_info) == 0:
                prev_stops = traci.vehicle.getStops(vehicle_id, -num_bus_stops)
                for stop in prev_stops:
                    stop_times[stop.stoppingPlaceID] = stop.depart
            """
            # You could try to retrieve the data in the simulation step after the bus left the last stop.
            # But if the stop is at the end of the last edge and the bus despawns after the stop duration, you would miss the depart/arrival time of the last stop.
            # If only one stop remains
            if len(stop_info) == 1 and traci.vehicle.isAtBusStop(vehicle_id):
                stop_times = []
                # print('Only execute once.')
                prev_stops = traci.vehicle.getStops(vehicle_id, -num_bus_stops)
                for stop in prev_stops:
                    stop_times.append(stop.depart)
                # Current stop
                stop_times.append(stop_info[0].arrival + stop_duration)

    return insert_time, stop_times


def _get_travel_time_until_first_stop(route_id: str, vehicle_id: str) -> float:
    """
    Internal helper function that simulates a vehicle along a route and calculate the time taken to reach the first stop.

    Parameters
    ----------
    route_id : str
        ID of the route to simulate.
    vehicle_id : str
        ID of the vehicle to be added to the simulation.

    Returns
    -------
    float
        The travel time from the moment the vehicle is inserted until it reaches the first stop.
    """
    # Insert vehicle
    traci.vehicle.add(vehicle_id, route_id, typeID="bus", departPos="stop")
    # Important to calculate until value
    insert_time = traci.simulation.getTime()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if traci.vehicle.getIDCount() > 0:
            # When the bus is at the first bus stop
            if traci.vehicle.isAtBusStop(vehicle_id):
                # Get current stop info. Assumes bus doesn't despawn after stop duration.
                stop_info = traci.vehicle.getStops(vehicle_id)
                # Current stop, get arrival time, calculate travel time from inserted to arrived
                travel_time = stop_info[0].arrival - insert_time
                # Abort trip
                traci.vehicle.remove(vehicle_id)
                # Exit loop
                break

    return travel_time


def _start_sumo(sumo_binary: str, sumo_config: str) -> None:
    """
    Internal helper function that starts a SUMO simulation using TraCI.

    Parameters
    ----------
    sumo_binary : str
        Path to the SUMO binary executable (e.g., "sumo" or "sumo-gui").
    sumo_config : str
        Path to the SUMO configuration file (e.g., "my_config.sumocfg").

    Returns
    -------
    None
    """
    if traci.isLoaded():
        traci.close()
    traci.start([sumo_binary, "-c", sumo_config])


# ===========================================================
# SECTION 7: Output Analysis
# ===========================================================


def from_lane_to_edge_junction(lane_id: str) -> str:
    """
    Converts a lane ID to its corresponding edge or junction ID.

    Parameters
    ----------
    lane_id : str
        The lane ID to be converted.

    Returns
    -------
    str
        The corresponding edge or junction ID.
    """
    # Internal cluster junction lanes
    if lane_id.startswith(":cluster"):
        # Example: :cluster_1243852524_2779282810_2779282814_29689184_530175_530176_4_1
        # Split the string by ':' and '_'. Remove the last two '_'.
        parts = lane_id.split(":")[1].split("_")[:-2]
        # Join the parts together.
        junction_id = "_".join(parts)
        return junction_id
    # Internal non-cluster junctions
    elif lane_id.startswith(":"):
        junction_id = lane_id.rsplit("_", 2)[0].split(":")[1]
        return junction_id
    # Normal lanes
    else:
        edge_id = lane_id.split("_")[0]
        return edge_id


def lanes_to_poi(
    teleport_dict: dict, path="./routes/teleport_edges.add.xml"
) -> None:
    """
    Converts a DataFrame of teleport data into a POI (additional) XML file for SUMO.

    Parameters
    ----------
    teleport_dict : pd.DataFrame
        DataFrame containing teleportation data with 'id' and 'coordinates' columns.
    path : str, optional
        Path to the output XML file (default is "./routes/teleport_edges.add.xml").

    Returns
    -------
    None
    """
    norm = plt.Normalize(
        vmin=np.min(teleport_dict["counts"]),
        vmax=np.percentile(teleport_dict["counts"], 80),
    )
    cmap = cm.get_cmap("bwr")
    counter = 0
    with sumolib.openz(path, mode="w") as output_file:
        sumolib.xml.writeHeader(output_file, root="additional")
        for i, row in teleport_dict.iterrows():
            x, y = row.coordinates
            counter += 1
            id = row.id
            color_rgba = cmap(norm(row.counts))
            color = f"{color_rgba[0]}, {color_rgba[1]}, {color_rgba[2]}, {color_rgba[3]}"
            type = "teleports"
            layer = "10"
            teleports = row.counts
            output_file.write(
                '    <poi id="%s" type="%s" color="%s" layer="%s" x="%.2f" y="%.2f">\n        <param key="teleports" value="%s"/>\n    </poi>\n'
                % (id, type, color, layer, x, y, teleports)
            )  # noqa

        output_file.write("</additional>\n")
    print(f"Number of mapped POIs: {counter}")


def parse_teleport_warnings(file_path: str) -> pd.DataFrame:
    """
    Parses the console.log file to identify problem lanes and flows where vehicles were teleported.

    Parameters
    ----------
    file_path : str
        Path to the console log file containing teleport warnings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed warnings with columns for vehicle ID, reason, lane ID, and time.
    """

    # Define the regular expression pattern for matching the log entries
    pattern = r"Warning: Teleporting vehicle '([^']+)'; (.*?), lane='([^']+)', .*?[, ]*time=([\d\.]+).*\."

    # Warning: Teleporting vehicle '9908_flow.4'; waited too long (blocked), lane='26338221_0', time=21102.00.

    # Initialize a list to store the parsed log entries
    data = []

    # Read the file line by line
    with open(file_path, "r") as file:
        for line in file:
            # print(line)
            # Use regex to match the pattern
            match = re.search(pattern, line)
            # print(match)
            if match:
                vehicle_id, reason, lane_id, time = match.groups()
                # Append the matched groups as a tuple to the data list
                data.append((vehicle_id, reason, lane_id, float(time)))

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=["vehicle_id", "reason", "lane_id", "time"])

    return df


def parse_emergency_warnings(file_path: str) -> pd.DataFrame:
    """
    Parse emergency stop warnings from a SUMO log file into a DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the SUMO log file to be processed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'vehicle_id': str, the ID of the vehicle that performed the emergency stop.
        - 'lane_id': str, the ID of the lane where the stop occurred.
        - 'time': float, the simulation time when the stop occurred.
    """
    # Regular expression pattern to match the relevant log information
    pattern = re.compile(
        r"Vehicle '([^']*)' performs emergency stop at the end of lane '([^']*)' for unknown reasons \(decel=[0-9.]+, offset=[0-9.]+\), time=([0-9.]+)\."
    )

    vehicles = []
    lanes = []
    times = []

    # Read the log file and process each line
    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                vehicles.append(match.group(1))
                lanes.append(match.group(2))
                times.append(float(match.group(3)))

    df = pd.DataFrame({"vehicle_id": vehicles, "lane_id": lanes, "time": times})

    return df


def plot_teleports_over_time(
    teleports_df: pd.DataFrame,
    dir_path: str,
    file_name="teleports_over_time.pdf",
) -> None:
    """
    Plots the cumulative number of teleports over time and saves the plot to file.

    Parameters
    ----------
    teleports_df : pd.DataFrame
        DataFrame containing teleport warnings with a 'time' column.
    dir_path : str
        Directory path where the plot PDF will be saved.
    file_name : str, optional
        The name of the file to save the plot as (default is "teleports_over_time.pdf").

    Returns
    -------
    None
    """
    # Sort by timestamp (optional but recommended for cumulative plotting)
    teleports_df = teleports_df.sort_values(by="time")

    # Calculate the cumulative number of teleports
    teleports_df["cumulative_teleports"] = (
        teleports_df["time"].rank(method="first").astype(int)
    )
    # fig, ax = plt.subplots()
    # Plot the cumulative number of teleports over time
    plt.figure(figsize=(10, 6))
    plt.step(
        teleports_df["time"],
        teleports_df["cumulative_teleports"],
        where="post",
        label="Cumulative Teleports",
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Cumulative Number of Teleports")
    plt.title("Cumulative Number of Teleports Over Time")

    plt.grid(True)
    plt.legend()

    plt.savefig(f"{dir_path}teleports_over_time.pdf")
    plt.show()


def plot_until_values(
    route_id: str,
    gtfs_dict: dict[str, dict[str, float]],
    traci_dict: dict[str, dict[str, float]],
    dir_path: str,
) -> None:
    """
    Plots the 'until' values for a given route from GTFS and TRACI sources.

    Parameters
    ----------
    route_id : str
        The ID of the route being plotted.
    gtfs_dict : dict[str, dict[str, float]]
        Dictionary containing GTFS 'until' values with bus stop IDs as keys.
    traci_dict : dict[str, dict[str, float]]
        Dictionary containing TRACI 'until' values with bus stop IDs as keys.
    dir_path : str
        Directory path where the plot PDF will be saved.

    Returns
    -------
    None
    """
    # Ensure stops are in the same order for both dictionaries
    stops = list(gtfs_dict.keys())
    # Extract 'until' values for GTFS and TRACI
    until_values_dict1 = [gtfs_dict[stop] for stop in stops]
    until_values_dict2 = [traci_dict[stop] for stop in stops]

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.plot(stops, until_values_dict1, label="GTFS", marker="o")
    plt.plot(stops, until_values_dict2, label="TRACI", marker="o")

    plt.title(f"Until Values for Route: {route_id}")
    plt.xlabel("Bus Stops")
    plt.ylabel("Until Time (seconds)")

    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(f"{dir_path}route_{route_id}.pdf")
    plt.close()


# ===========================================================
# SECTION 8: XML Manipulation
# ===========================================================


def add_parking_attribute(file_path: str) -> None:
    """
    Adds a 'parking="True"' attribute to all 'stop' elements in the XML file.

    Parameters
    ----------
    file_path : str
        The path to the XML file to be modified.

    Returns
    -------
    None
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Iterate over all busStop elements and add the parking="True" attribute
    for bus_stop in root.findall(".//stop"):
        bus_stop.set("parking", "true")

    # Write the modified XML back to the file
    tree.write(file_path, encoding="utf-8", xml_declaration=True)


def add_until_attribute_to_rou(
    file_path: str, until_values_dict: dict[str, dict[str, float]]
) -> None:
    """
    Adds an 'until' attribute to 'stop' elements in 'route' elements based on the provided dictionary.

    Parameters
    ----------
    file_path : str
        The path to the XML file to be modified.
    until_values_dict : dict[str, dict[str, float]]
        A dictionary where the key is the route ID and the value is another dictionary mapping stop indices to 'until' values.

    Returns
    -------
    None
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Iterate through each route in the XML
    for route in root.findall("route"):
        route_id = route.get("id")
        shape_id = route_id.split("_")[1]
        if shape_id in until_values_dict:
            stops = route.findall("stop")
            for index, stop in enumerate(stops):
                if str(index) in until_values_dict[shape_id]:
                    # print('yep')
                    stop.set(
                        "until", str(until_values_dict[shape_id][str(index)])
                    )

    # Write the updated tree back to the file
    tree.write(file_path, xml_declaration=True, encoding="utf-8", method="xml")


def extract_route_ids(file_path: str) -> set[str]:
    """Extracts unique route IDs from a SUMO rou.xml file.

    Parameters
    ----------
    file_path : str
        file (str): Path to the .rou.xml file.

    Returns
    -------
    set[str]
        A set of unique route IDs in the file.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract all the route ids from the file
    route_ids = set()
    for route in root.findall("route"):
        route_id = route.get("id")
        if route_id:
            route_ids.add(route_id)

    return route_ids


def find_bus_lane_edges(net_file: str) -> dict[str, tuple[int, int]]:
    """
    Finds edges with bus lanes from a SUMO net XML file.

    Parameters
    ----------
    net_file : str
        The path to the SUMO net XML file.

    Returns
    -------
    dict[str, tuple[int, int]]
        A dictionary where the key is the edge ID and the value is a tuple containing:
        - The index of the bus lane in the edge's lane list.
        - The total number of lanes in that edge.
    """
    # Parse the XML file
    tree = ET.parse(net_file)
    root = tree.getroot()
    bus_lane_edges = {}
    # Iterate over all edges in the XML
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if edge_id.startswith(":"):
            continue
        lanes = edge.findall("lane")
        total_lanes = len(lanes)
        for index, lane in enumerate(lanes):
            if lane.get("allow") == "bus coach":
                bus_lane_edges[edge_id] = (index, total_lanes)
                break
    return bus_lane_edges


def find_route_id_difference(file1: str, file2: str) -> dict[str, set[str]]:
    """Compares two SUMO .rou.xml files and returns the difference in unique route IDs.

    Parameters
    ----------
    file1 : str
        Path to the first .rou.xml file.
    file2 : str
        Path to the second .rou.xml file.

    Returns
    -------
    dict[str, set[str]]
        A dictionary containing the route IDs only in file1 and only in file2.
    """
    # Extract route IDs from both files
    route_ids_file1 = extract_route_ids(file1)
    route_ids_file2 = extract_route_ids(file2)

    # Find route IDs that are unique to each file
    unique_to_file1 = route_ids_file1 - route_ids_file2
    unique_to_file2 = route_ids_file2 - route_ids_file1

    # Return the difference in route IDs as a dictionary
    return {
        "unique_to_rou_file_1": unique_to_file1,
        "unique_to_rou_file_2": unique_to_file2,
    }


def get_routes_from_rou_file(rou_file: str, route_ids: list) -> dict[str, str]:
    """Extracts the corresponding route strings for a given list of route IDs from a SUMO .rou.xml file.


    Parameters
    ----------
    rou_file : str
        Path to the SUMO .rou.xml file.
    route_ids : list
        List of route IDs to look up.

    Returns
    -------
    dict[str,str]
        A dictionary where keys are route IDs and values are the corresponding route strings.
    """

    # Parse the SUMO .rou.xml file
    tree = ET.parse(rou_file)
    root = tree.getroot()

    # Create a set from the route_ids for faster lookups
    route_ids_set = set(route_ids)

    # Initialize a dictionary to store the route strings
    routes_dict = {}

    # Iterate through all the route elements in the file
    for route in root.findall("route"):
        route_id = route.get("id")

        # If the route_id is in the target list
        if route_id in route_ids_set:
            # Get the edges string from the route element
            route_edges = route.get("edges")

            # Store the route_id and corresponding edges in the dictionary
            routes_dict[route_id] = route_edges

    return routes_dict


def find_vehicles_on_bus_lane_edges_and_update(
    bus_lane_edges: dict[str, tuple[int, int]],
    rou_file: str,
    target_rou_file: str,
    sumo_net: sumolib.net,
) -> tuple[list[str], list[str]]:
    """
    Finds vehicles starting on bus lane edges, updates their departLane and departPos attributes, and saves the updated file.
    Important: Only relevant for vehicle definitions which specify 'departLane'.

    Parameters
    ----------
    bus_lane_edges : dict[str, tuple[int, int]]
        A dictionary with edge IDs as keys and tuples (bus_lane_index, total_lanes) as values.
    rou_file : str
        The path to the input SUMO rou XML file.
    target_rou_file : str
        The path to the output SUMO rou XML file with updated attributes.
    sumo_net : sumolib.net
        The SUMO network object to get edge lengths.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple where:
        - The first element is a list of vehicle IDs that start on bus lanes.
        - The second element is a list of edges where the departPos was adjusted due to being larger than edge length.
    """
    tree = ET.parse(rou_file)
    root = tree.getroot()

    vehicles_on_bus_lanes = []
    depart_pos2do = []

    for vehicle in root.findall("vehicle"):
        route = vehicle.find("route")
        if route is not None:
            edges = route.get("edges")
            if edges is not None:
                first_edge = edges.split()[0]
                edge_length = sumo_net.getEdge(first_edge).getLength()
                depart_pos = float(vehicle.get("departPos"))
                if depart_pos > edge_length:
                    # print('Change departPos. Edge length is smaller.', vehicle.get('id'), edge_length, first_edge)
                    vehicle.set("departPos", str(edge_length))
                    depart_pos2do.append(first_edge)
                if first_edge in bus_lane_edges:
                    bus_lane_index, total_lanes = bus_lane_edges[first_edge]
                    current_depart_lane = vehicle.get(
                        "departLane", "0"
                    )  # Default to '0' if not set
                    if int(current_depart_lane) == bus_lane_index:
                        vehicles_on_bus_lanes.append(vehicle.get("id"))
                        vehicle.set("departLane", "1")

    # Save the updated rou.xml file
    tree.write(target_rou_file, encoding="utf-8", xml_declaration=True)

    # Manually fix route tags in the output file
    with open(target_rou_file, "r") as file:
        lines = file.readlines()

    with open(target_rou_file, "w") as file:
        for line in lines:
            line = line.replace('" />', '"/>')
            if '<route edges="' in line:
                # Ensure the route tag is properly closed
                line = line.replace('"/>', '"></route>')
            file.write(line)

    return vehicles_on_bus_lanes, depart_pos2do


def _indent(elem, level=0) -> None:
    """
    Internal helper function that recursively adds indentation to an XML element tree for pretty-printing.

    Parameters
    ----------
    elem : xml.etree.ElementTree.Element
        The XML element to indent.
    level : int, optional
        The current indentation level (default is 0).

    Returns
    -------
    None
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def parse_routes_and_num_stops(file_path: str) -> dict[str, int]:
    """
    Parse an XML file to extract route IDs and count the number of stops for each route.

    Parameters
    ----------
    file_path : str
        The path to the XML file containing route information.

    Returns
    -------
    dict[str, int]
        A dictionary where keys are route IDs and values are the number of stops in each route.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    route_info = {}

    for route in root.findall("route"):
        route_id = route.get("id")
        stops = route.findall("stop")
        route_info[route_id] = len(stops)

    return route_info


def parse_sumo_routes(file_path: str) -> dict[str, dict[str, float]]:
    """
    Parse a SUMO routes XML file to extract route IDs and associated stop information.

    Parameters
    ----------
    file_path : str
        The path to the XML file containing route information.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary where keys are route IDs and values are dictionaries with stop IDs and their associated 'until' values.
    """
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionary to store the output
    route_dict = {}

    # Iterate through each route
    for route in root.findall("route"):
        route_id = route.get("id")
        stops_dict = {}

        # Iterate through each stop within the route
        for stop in route.findall("stop"):
            bus_stop_id = stop.get("busStop")
            until_value = float(stop.get("until", "0.0"))
            stops_dict[bus_stop_id] = until_value

        # Store the stops information for the current route
        route_dict[route_id] = stops_dict

    return route_dict


def remove_until_attribute(rou_file_path: str, to_file_path: str) -> None:
    """
    Remove the 'until' attribute from all <stop> elements in a SUMO rou.xml file.

    Parameters
    ----------
    rou_file_path : str
        The path to the input SUMO rou.xml file.
    to_file_path : str
        The path where the updated SUMO rou.xml file will be saved.

    Returns
    -------
    None
    """
    tree = ET.parse(rou_file_path)
    root = tree.getroot()

    for stop in root.findall(".//stop"):
        if "until" in stop.attrib:
            del stop.attrib["until"]

    # Save the updated rou.xml file
    tree.write(to_file_path, encoding="utf-8", xml_declaration=True)


def replace_edges_in_route_file(
    rou_file_path: str,
    edge_replacements: dict[str, str],
    output_xml_file_path: str,
) -> None:
    """
    Replace edge IDs in a SUMO route file based on the provided edge replacement mapping.

    Parameters
    ----------
    rou_file_path : str
        The path to the input SUMO route XML file.
    edge_replacements : dict[str, str]
        A dictionary where keys are original edge IDs and values are space-separated lists of new edge IDs.
    output_xml_file_path : str
        The path where the modified SUMO route XML file will be saved.

    Returns
    -------
    None
    """
    # Parse the XML file
    tree = ET.parse(rou_file_path)
    root = tree.getroot()

    # Traverse routes and replace edge_ids
    for vehicle in root.findall(".//vehicle"):
        route = vehicle.find("route")
        if route is not None:
            edges = route.get("edges").split()
            new_edges = []
            for edge in edges:
                if edge in edge_replacements:
                    new_edges.extend(edge_replacements[edge].split())
                else:
                    new_edges.append(edge)
            route.set("edges", " ".join(new_edges))

    # Write the modified XML content back to a new file
    tree.write(output_xml_file_path, encoding="utf-8", xml_declaration=True)


# ===========================================================
# Universal Functions
# ===========================================================


def filter_dict_by_keys(original_dict: dict, target_keys: Iterable) -> dict:
    """
    Filters the dictionary to include only the entries with keys in the target_keys list.

    Parameters
    ----------
    original_dict : dict[str, Any]
        The original dictionary to filter.
    target_keys : Iterable
        A list of keys to include in the filtered dictionary.

    Returns
    -------
    dict[str, Any]
        A dictionary containing only the entries with keys in the target_keys list.
        If any key in target_keys is not found in original_dict, an empty dictionary is returned.
    """
    return_dict = {}
    for key in target_keys:
        if key in original_dict:
            return_dict[key] = original_dict[key]
        else:
            return {}  # Return an empty dictionary if the key is not found

    return return_dict
