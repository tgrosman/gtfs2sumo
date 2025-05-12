import copy
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime

import pandas as pd
import sumolib
from shapely.geometry.point import Point
from tqdm import tqdm

from lib import gtfs2sumo

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))


def create_directory_if_not_exist(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def log_time(task_description, func, *args, **kwargs):
    """Logs the task description, starts a timer, calls the function, and logs the duration."""
    print(task_description, end="", flush=True)
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    print(f" took {end_time - start_time:.2f} seconds.")


def get_routechecker():
    if not "SUMO_HOME" in os.environ:
        print("SUMO_HOME environment variable not set")
        exit(-1)
    routechecker_path = os.path.join(
        os.environ["SUMO_HOME"], "tools", "route", "routecheck.py"
    )
    return routechecker_path


class GTFS2SUMO:
    traffic_extracted = False
    _MAP_DIRECTORY = "maps"

    def __init__(
        self,
        net_file_path: str,
        gtfs_data_path: str,
        gtfs_date: str,
        create_maps: bool = False,
        verbose: bool = False,
        out_directory: str = "./out",
        log_directory: str = "./log",
        output_prefix: str = "extracted",
        default_stop_length: float = 14,
        default_stop_duration: float = 20.0,
        stop_search_distance: int = 30,
        stop_merging_distance: float = 10.0,
        correction_search_depth: int = 4,
        number_of_duplicate_ids: int = 6,
        detour_length_threshold: float = 35.0,
    ) -> None:
        """
        This constructor takes the required input files (a SUMO Network and a GTFS zip) as well as adjustable parameters, changing the conversion output.
        The result will be two files: An additional SUMO file containing the extracted bus stops and a SUMO route file containing the extracted routes and schedules as flows. # TODO integrate per bus scheduling
        Parameters
        ----------
        net_file_path: str
            Path to the SUMO network file.
        gtfs_data_path: str
            Path to the GTFS zip file.
        gtfs_date: str
            Format: "yyyymmdd"; The date to be used (must be present in GTFS). # TODO maybe prepend busiest day extraction as option
        create_maps: bool
            If set to `True` a collection of folium maps will be created, which can help with analyzing and debugging conversion results.
        verbose: bool
            If set to `True` more detailed summaries on sub-processes will be printed.
        out_directory: str
            The directory where the output will be placed.
        log_directory: str
            The directory where created maps and intermediate files will be placed. # TODO print stdout to file, including parameters?
        output_prefix: str
            The prefix that will be used for the output files.
        default_stop_length: float
            Unit: [m]; The default length that will be used for bus stops.
        default_stop_duration: float
            Unit: [s]; The default duration that buses will stop at each halt.
        stop_search_distance: int
            Unit: [m]; The maximal distance that will be applied for the search of an closest edge. (30m has empirically be found to be a good trade-off)
        stop_merging_distance: float
            Unit: [m]; The maximum distance to stops on the same edge can be apart from each other to be handled as the same stop
        correction_search_depth:
            # TODO I don't quite get this
        number_of_duplicate_ids: int
            Number of edge IDs to check for duplicate pairs in remove_duplicate_pairs_in_routes.
        detour_length_threshold: float
            Unit: [m]; Tracemapper will sometimes generate invalid detours due to fault GTFS shapes, we try to filter them by the length of a detour.
            This parameter sets the threshold for detours that will be discarded.
        """
        # Constants
        self._NET_FILE_PATH = net_file_path
        self._CREATE_MAPS = create_maps
        self._VERBOSE = verbose
        self._OUT_DIRECTORY = out_directory
        self._LOG_DIRECTORY = log_directory
        self._OUTPUT_PREFIX = output_prefix
        self._DEFAULT_STOP_LENGTH = default_stop_length
        self._DEFAULT_STOP_DURATION = default_stop_duration
        self._STOP_SEARCH_DISTANCE = stop_search_distance
        self._STOP_MERGING_DISTANCE = stop_merging_distance
        self._DETOUR_LENGTH_THRESHOLD = detour_length_threshold
        self._CORRECTION_SEARCH_DEPTH = correction_search_depth
        self._NUMBER_OF_DUPLICATE_IDS = number_of_duplicate_ids
        # read files
        log_time("Reading Net File...", self.__read_net_file)
        log_time(
            "Reading GTFS Data...",
            self.__read_gtfs_data,
            gtfs_data_path,
            gtfs_date,
        )

    def __read_net_file(self) -> sumolib.net.Net:
        net = sumolib.net.readNet(self._NET_FILE_PATH)
        # Get bbox from net
        bbox = net.getBBoxXY()
        # Transform x/y to lon/lat
        self._bbox_lon_lat = gtfs2sumo.bbox_XY_2_lon_lat(bbox, net)
        self._net = net

    def __read_gtfs_data(self, gtfs_data_path, gtfs_date) -> None:
        # Read GTFS
        gtfs_zip = zipfile.ZipFile(
            sumolib.openz(gtfs_data_path, mode="rb", tryGZip=False)
        )
        # Split data set for specific date
        (
            self._routes,
            self._trips_on_day,
            self._shapes,
            stops,
            self._stop_times,
        ) = gtfs2sumo.import_gtfs(gtfs_date, gtfs_zip)
        # Extend stops data frame with info from network
        self._stops = self.__extend_stops(stops)

        gtfs_zip.fp.close()

    def __extend_stops(self, stops: pd.DataFrame):
        """
        Adds additional info to stops dataframe  based on network and the bounding box
        Parameters
        ----------
        stops
        the dataframe containing the stops

        Returns
        -------
        The extended stops dataframe

        """
        # Add important information to stops DataFrame
        # Boolean column whether the stop is outside the boundaries of the SUMO network
        stops["outside_bbox"] = stops.apply(
            lambda row: gtfs2sumo.outside_bbox(
                (float(row["stop_lat"]), float(row["stop_lon"])),
                self._bbox_lon_lat,
            ),
            axis=1,
        )
        # Get closest lane_id based on stop location within the given radius
        stops["lane_id"] = stops.apply(
            lambda row: gtfs2sumo.get_lane_from_lon_lat(
                row["stop_lon"],
                row["stop_lat"],
                self._net,
                self._STOP_SEARCH_DISTANCE,
                return_candidates=False,
            ),
            axis=1,
        )
        stops["edge_id"] = stops.apply(
            lambda row: (
                self._net.getLane(row.lane_id).getEdge().getID()
                if row.lane_id is not None
                else None
            ),
            axis=1,
        )
        # Boolean column whether the stop can naively be mapped onto the SUMO network given the radius.
        stops["mapped"] = stops["lane_id"].notna().astype(int)
        return stops

    def __filter_gtfs_data(self) -> None:
        (
            self._gtfs_data,
            self._trip_list,
            self._filtered_stops,
            self._shapes,
            self._shapes_dict,
        ) = gtfs2sumo.filter_gtfs(
            self._routes,
            self._trips_on_day,
            self._shapes,
            self._stops,
            self._stop_times,
            self._net,
        )

    def __extract_main_shapes(self) -> None:
        # Create list of main_shapes
        self._main_shape_ids = list(set(self._shapes_dict.values()))

        # Generate for each main shape a list of shape points
        self._main_shape_dict = gtfs2sumo.get_shape_dict(
            self._main_shape_ids, self._shapes
        )

    def __create_plain_main_shape_map(self) -> None:
        directory = os.path.join(self._LOG_DIRECTORY, self._MAP_DIRECTORY)
        create_directory_if_not_exist(directory)
        gtfs2sumo.map_routes(
            self._main_shape_dict,
            "plain_main_shapes",
            bbox_lon_lat=self._bbox_lon_lat,
            dir_path=directory,
            show_flag=True,
        )

    def __trip_list_add_main_shape(self):
        # Map main_shape_id to each shape_id
        self._trip_list["main_shape_id"] = self._trip_list["shape_id"].apply(
            lambda shape_id: self._shapes_dict[shape_id]
        )
        self._trip_list["departure_fixed"] = pd.to_timedelta(
            self._trip_list["departure_fixed"]
        )

    def __generate_main_shape_data(self):
        # Generate interval, earliest and last departure per main_shape
        self._main_shape_interval_dict = gtfs2sumo.compute_interval_dict(
            self._trip_list, generate_csv=False
        )

        # Get the bus stop sequence for each main shape
        self._main_shape_stop_sequences = gtfs2sumo.get_shape_stop_sequences(
            self._main_shape_ids, self._gtfs_data
        )

        for shape_id in self._main_shape_stop_sequences.keys():
            # Add column with shapely.geometry.Point for later for each stop
            self._main_shape_stop_sequences[shape_id][
                "geometry"
            ] = self._main_shape_stop_sequences[shape_id].apply(
                lambda row: Point(
                    self._net.convertLonLat2XY(row["stop_lon"], row["stop_lat"])
                ),
                axis=1,
            )

    def __summarize_gtfs_data(self) -> None:
        gtfs_trips_on_day = len(self._gtfs_data["trip_id"].unique())
        gtfs_active_main_shapes = len(self._main_shape_stop_sequences.keys())
        gtfs_active_bus_lines = len(
            self._trip_list["route_short_name"].unique()
        )
        # Quick summary of GTFS data set
        print("============ GTFS DATA SUMMARY ============")
        print(
            "    Active trips on given day (translates to buses in SUMO):",
            gtfs_trips_on_day,
        )
        print("    Active main shapes on given day:", gtfs_active_main_shapes)
        print("    Active bus lines on given day:", gtfs_active_bus_lines)
        print("===========================================")

    def __create_all_stops_map(self):
        directory = os.path.join(self._LOG_DIRECTORY, self._MAP_DIRECTORY)
        create_directory_if_not_exist(directory)
        # Generate OSM map for all stops in the GTFS data set.
        gtfs2sumo.map_stops(
            self._stops,
            self._bbox_lon_lat,
            path=os.path.join(directory, "bbox_bus_stops"),
        )

    def __create_grouped_shape_maps(self):
        grouped_shapes_path = os.path.join(
            self._LOG_DIRECTORY, self._MAP_DIRECTORY, "grouped_shapes"
        )
        create_directory_if_not_exist(grouped_shapes_path)
        # Group all shapes by main_shape and generate OSM maps
        grouped_shapes_by_main_shape = defaultdict(list)
        for shape_id, main_shape_id in self._shapes_dict.items():
            grouped_shapes_by_main_shape[main_shape_id].append(shape_id)

        # Convert the defaultdict back to a regular dictionary
        grouped_shapes_by_main_shape = dict(grouped_shapes_by_main_shape)

        for k, v in grouped_shapes_by_main_shape.items():
            # Get shape collection for each main shape
            shape_collection = gtfs2sumo.get_shape_dict(v, self._shapes)
            # Add str 'main_shape' to main_shape_id in shape_collection for transparency
            main_shape_key = f"{k}_main_shape"
            shape_collection = {
                (main_shape_key if shape_id == k else shape_id): value
                for shape_id, value in shape_collection.items()
            }
            # Map shapes + corresponding main_shape
            gtfs2sumo.map_routes(
                shape_collection,
                f"{k}_shapes",
                self._bbox_lon_lat,
                grouped_shapes_path,
                show_flag=True,
            )

    def __create_non_mappable_stops(self):
        # Get all non-mappable stops by filtering the 'stops' DataFrame where the 'mapped' column is 0
        non_mappable_stops = self._stops[self._stops["mapped"] == 0]
        mappable_stops = self._stops[self._stops["mapped"] == 1]
        # Generate an additional XML file for SUMO containing Points of Interest (POIs) for non-mappable stops
        # This can be used for visualization and verification on the SUMO network
        directory = os.path.join(self._LOG_DIRECTORY)
        create_directory_if_not_exist(directory)
        gtfs2sumo.stops_to_poi_XML(
            non_mappable_stops,
            self._net,
            os.path.join(directory, "non_mappable_stops.add.xml"),
        )
        gtfs2sumo.stops_to_poi_XML(
            mappable_stops,
            self._net,
            os.path.join(directory, "mappable_stops.add.xml"),
            color="blue",
            type="mappable",
        )

    def __find_shapes_with_non_mappable_stops(self):
        """
        For each non-mappable stop, find all main shapes that the stop is part of the stop sequence.
        Find vital stops that are part of many routes. Good starting point if you want to efficiently expand your network.
        """
        self._main_shapes_non_mappable_stops = {}
        # Iterate over each main shape's stop sequence
        for (
            shape_id,
            stop_sequence_df,
        ) in self._main_shape_stop_sequences.items():
            # Find the stop_ids in the shape's sequence that cannot be mapped
            stop_ids = stop_sequence_df[stop_sequence_df["mapped"] == 0][
                "stop_id"
            ].values.tolist()
            # If there are non-mappable stops, add the shape_id and corresponding stop_ids to the dictionary
            if len(stop_ids) > 0:
                self._main_shapes_non_mappable_stops[shape_id] = stop_ids

    def __save_shapes_with_non_mappable_stops(self):
        stop_to_shapes_dict = {}
        # Iterate through each shape and its list of non-mappable stops
        for shape_id, stop_ids in self._main_shapes_non_mappable_stops.items():
            for stop_id in stop_ids:
                # If the stop_id is not already in the dictionary, initialize it with an empty list
                if stop_id not in stop_to_shapes_dict:
                    stop_to_shapes_dict[stop_id] = []
                # Append the shape_id to the list of shapes that contain this stop_id
                stop_to_shapes_dict[stop_id].append(shape_id)
        # Sort the dictionary by the number of shapes that have the stop_id (in descending order)
        stop_to_shapes_dict = dict(
            sorted(
                stop_to_shapes_dict.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
        )
        # Convert the dictionary to a DataFrame for easy visualization
        stop_to_shapes_df = pd.DataFrame(
            list(stop_to_shapes_dict.items()), columns=["stop_id", "shape_ids"]
        )
        # Optional: Save the DataFrame as a CSV file
        create_directory_if_not_exist(self._LOG_DIRECTORY)
        stop_to_shapes_df.to_csv(
            os.path.join(
                self._LOG_DIRECTORY, "non_mappable_stops_main_shapes.csv"
            ),
            index=False,
        )

    def __summarize_gtfs_extraction(self):
        print("======== GTFS EXTRACTION SUMMARY ==========")
        print(
            "    Non-mappable stops:",
            len(self._stops[self._stops["mapped"] == 0]),
        )
        print("    Total number of main shapes:", len(self._main_shape_ids))
        print(
            "    Clean main shapes (all stops of main shape can be mapped naively):",
            len(self._clean_main_shapes),
        )
        print("    Incomplete main shapes:", len(self._incomplete_main_shapes))
        print("===========================================")

    def __create_main_shape_stop_sequences_map(self, sub_directory):
        directory = os.path.join(
            self._LOG_DIRECTORY, self._MAP_DIRECTORY, sub_directory
        )
        create_directory_if_not_exist(directory)
        # Generate and save all main_shape_stop_sequences to html. Recommended if this is your first time dealing with GTFS.
        gtfs2sumo.generate_html_main_shape_stop_sequences(
            self._main_shape_stop_sequences,
            "stop_sequence",
            dir_path=directory,
        )

    def __find_clean_shapes(self):
        # Determine for each main shape if all stops are mappable
        self._clean_main_shapes = []
        self._incomplete_main_shapes = []
        for shape_id, df in self._main_shape_stop_sequences.items():
            if len(df[df["mapped"] == 0]) == 0:
                self._clean_main_shapes.append(shape_id)
            else:
                self._incomplete_main_shapes.append(shape_id)

    def __reset_stop_sequences(self):
        # Reset stop sequence for all main shapes. (All sequences start with 0)
        gtfs2sumo.reset_stop_sequence(
            self._main_shape_ids, self._main_shape_stop_sequences
        )

    def __include_start_end_cut_off_shapes(self):
        """
        Expand the pool of main shapes that are put into Tracemapper to find potential bus routes using the 'mappable'-column and regular expressions.
        For main shapes with non-mappable stops, find those main shapes that start/end with a block of non-mappable stops but no switch in between.
        Only include main shapes with a given min. length after cutoff.
        Although we only naively determine whether a stop can be mapped onto the network, it's already a good indicator if we can retrieve a functioning SUMO route.
        Test different radii and rerun the notebook.
        """
        self._start_end_cutoff_shapes = []
        # Initialize a dictionary to store the completeness of each shape after cutoff
        self._completeness_dict = {
            shape_id: 1.0 for shape_id in self._main_shape_ids
        }
        self._remaining_incomplete_main_shapes = []
        self._start_end_cutoff_shapes_lack_completeness = []
        # Loop through each main shape ID from the incomplete list
        for shape_id in self._incomplete_main_shapes:
            # Generate a string representation of the "mapped" column (e.g. '0001111111')
            mapped_string = "".join(
                str(x)
                for x in self._main_shape_stop_sequences[shape_id][
                    "mapped"
                ].tolist()
            )
            # - A block of 0s followed by a block of 1s (e.g., '000111')
            # - Or a block of 1s followed by a block of 0s (e.g., '111000')
            # The pattern ensures that there is only one switch from 0 to 1 or 1 to 0, without alternating back and forth.
            pattern = r"0+1+$|1+0+$"
            # Apply regex to string
            start_flag, last_one_index = gtfs2sumo.get_change_index(
                pattern, mapped_string
            )
            # If the regex pattern is matched and a valid index is found (not None)
            if last_one_index is not None:
                # Get main shape stop sequence
                cut_sequence = self._main_shape_stop_sequences[shape_id]
                if start_flag:  # Cutoff at start
                    # Adjust the string by removing the initial non-mappable stops
                    adjusted_str = mapped_string[last_one_index + 1 :]
                    # Update the stop sequence to only keep stops after the cutoff point
                    self._main_shape_stop_sequences[shape_id] = cut_sequence[
                        cut_sequence["stop_sequence"] > float(last_one_index)
                    ]
                    cut_off_index = 0
                else:  # Cutoff at end
                    adjusted_str = mapped_string[: last_one_index + 1]
                    self._main_shape_stop_sequences[shape_id] = cut_sequence[
                        cut_sequence["stop_sequence"] <= float(last_one_index)
                    ]
                    cut_off_index = -1
                # Calculate completeness: how much of the original shape remains after the cutoff
                completeness = round(len(adjusted_str) / len(mapped_string), 2)
                # Cut the shape's geometry based on the remaining stops
                directory = os.path.join(
                    self._LOG_DIRECTORY, self._MAP_DIRECTORY, "cut_shapes"
                )
                create_directory_if_not_exist(directory)
                cut_shape = gtfs2sumo.cut_shape_points(
                    shape_id,
                    self._main_shape_stop_sequences[shape_id]
                    .iloc[cut_off_index]
                    .geometry,
                    self._main_shape_dict[shape_id],
                    self._bbox_lon_lat,
                    self._net,
                    start_flag,
                    self._CREATE_MAPS,
                    directory=directory,
                )
                self._main_shape_dict[shape_id] = cut_shape

                # If the completeness of the shape after cutoff is less than 0.5, we exclude it
                # Change factor if required
                if completeness < 0.5:
                    self._start_end_cutoff_shapes_lack_completeness.append(
                        shape_id
                    )
                    self._completeness_dict[shape_id] = completeness
                    continue  # Skip to the next shape
                # Otherwise, if the completeness is sufficient, we add the shape to the list of eligible main shapes
                else:
                    self._start_end_cutoff_shapes.append(shape_id)
                    self._completeness_dict[shape_id] = completeness
            else:
                # If the pattern doesn't match, the shape is left incomplete, and we add it to the remaining list
                self._remaining_incomplete_main_shapes.append(shape_id)

    def __summarize_start_end_cut_off(self):
        print("============= CUT OFF SUMMARY =============")
        print(
            "    Number of cut main shapes:", len(self._start_end_cutoff_shapes)
        )
        print(
            "    Removed main shapes due to incompleteness:",
            len(self._start_end_cutoff_shapes_lack_completeness),
        )
        print(
            "    Remaining incomplete main shapes:",
            len(self._remaining_incomplete_main_shapes),
        )
        print("===========================================")

    def __prepare_shapes_for_tracemapper(self):
        # Combine 'clean' main shapes with those that passed the checks (start/end cutoff shapes and only-one-stop-missing shapes)
        self._checked_main_shapes = (
            self._clean_main_shapes + self._start_end_cutoff_shapes
        )

        # Combine all shapes into a dictionary and retrieve the shape points (list of geo coordinates)
        # This dictionary includes shapes from 'clean_main_shapes', 'start_end_cutoff_shapes', and 'only_one_stop_missing_main_shapes'
        self._checked_main_shapes_dict = {
            shape_id: self._main_shape_dict[shape_id]
            for shape_id in self._checked_main_shapes
        }

    def __start_tracemapper(self):
        """
        Find a route through the SUMO network based on the shape points of each main shape
        """
        self._tracemapper_routes_dict = {}
        # Tracemapper can sometimes include repeated pairs of edges at the beginning or end of a route (e.g., "A, B, A, B, A, B, C, D, E...").
        # We remove these duplicates to prevent faulty routes.
        for shape_id in tqdm(
            self._checked_main_shapes
        ):  # TODO this can potentially be parallelized
            self.__execute_tracemapper(shape_id)

    def __execute_tracemapper(self, shape_id):
        # Convert the longitude and latitude coordinates of the shape points to SUMO's 2D XY coordinates
        coord_2d = [
            self._net.convertLonLat2XY(coord[0], coord[1])
            for coord in self._checked_main_shapes_dict[shape_id]
        ]
        # Convert tuples to lists (since SUMO expects lists)
        coord_2d = [list(t) for t in coord_2d]
        # Map the trace of shape coordinates to a route on the SUMO network using Tracemapper (or mapTrace in sumolib)
        mapped_route = sumolib.route.mapTrace(
            coord_2d,
            self._net,
            self._STOP_SEARCH_DISTANCE,
            verbose=False,
            fillGaps=5000,
            gapPenalty=500,
            debug=False,
        )
        # If a 'valid' mapped route is found (not really valid, see below)
        if mapped_route:
            # Extract the edge IDs of the mapped route
            route = [e.getID() for e in mapped_route]
            # Remove consecutive duplicate edge IDs from the route. This can happen with fuzzy shape points (e.g., "A, A, A, B, B, C" -> "A, B, C")
            cleaned_route = gtfs2sumo.remove_consecutive_edge_duplicates(route)
            # If the route is longer than 2 * number_of_ids, we remove duplicate pairs in routes
            if len(route) >= (2 * self._NUMBER_OF_DUPLICATE_IDS):
                cleaned_route = gtfs2sumo.filter_duplicate_pairs_at_start_end(
                    cleaned_route, self._NUMBER_OF_DUPLICATE_IDS
                )
            # Convert the cleaned route to a space-separated string of edge IDs
            route_string = " ".join(cleaned_route)
            # Update the Tracemapper dictionary with the shape ID and its corresponding route
            self._tracemapper_routes_dict[shape_id] = route_string

    def __create_route_file_for_clean_routes(self):
        create_directory_if_not_exist(
            os.path.join(self._LOG_DIRECTORY, "routes")
        )
        routes_path = os.path.join(
            self._LOG_DIRECTORY, "routes", "clean_routes_test.rou.xml"
        )
        # Optional: Generate a rou file for clean routes and test it. You can ignore 'main_shape_stop_sequences because you are only testing the routes yet.
        # Set 'include_vehicles' to True and set the depart parameter.
        gtfs2sumo.routes_to_XML(
            self._clean_routes_dict,
            self._main_shape_stop_sequences,
            stop_duration=self._DEFAULT_STOP_DURATION,
            path=routes_path,
            ignore_bus_stops=True,
            include_vehicles=True,
            depart=500,
        )

    def __attempt_to_correct_faulty_routes_at_junctions(self):
        """
        Due to the imperfect accuracy of shape points, Tracemapper can sometimes include extra edges at junctions that are unnecessary and even harmful to the route.
        This can happen, for example, when roundabouts are simplified into regular junctions during network generation.
        As a result, Tracemapper might have difficulty identifying the correct route, even though it exists in the network.
        The following method detects such problematic routes and attempts to correct them.
        """
        # Iterate over each row in the 'routechecker_df' DataFrame
        for i, row in self._routechecker_df_after_tracemapper.iterrows():
            # Retrieve the route associated with the current 'shape_id' and split it into a list of edges
            route = self._faulty_routes_dict[str(row.shape_id)].split(" ")
            # Attempt to fix the route by specifying the problematic section (from 'from_edge' to 'to_edge')
            fixed_route = gtfs2sumo.repair_faulty_route_at_junctions(
                row.from_edge, row.to_edge, route, self._net
            )
            # If the route is successfully fixed
            if fixed_route:
                # Convert the fixed route back into a single string of edge IDs
                edge_string = " ".join(fixed_route)
                # Update the route in the dictionary with the fixed route
                self._faulty_routes_dict[str(row.shape_id)] = edge_string
                # Remove the row from 'routechecker_df' since the error has been fixed
                self._routechecker_df_after_tracemapper.drop(i, inplace=True)

    def __attempt_to_correct_faulty_routes_at_beginning(self):
        """
        Buses often execute U-turns at the start or end of their routes on small roads or private streets that are not represented in the network.
        Instead of discarding the entire route, we attempt to preserve as much of it as possible by trimming the disconnected portions, guided by the Routechecker errors.
        Here we focus on the start...
        """
        # Filter the DataFrame 'routechecker_df_after_junction_correction' to only consider routes where an error occurs at the start of the route.
        starting_df = self._routechecker_df_after_junction_correction[
            self._routechecker_df_after_junction_correction["to_index"]
            <= self._CORRECTION_SEARCH_DEPTH
        ]
        # Make a deep copy of the faulty routes dictionary
        self._start_cutoff_routes_dict = copy.deepcopy(self._faulty_routes_dict)
        # Save trimmed edges for stop-mapping
        self._shape_cutoff_beginning_edges = {}
        # Iterate over each row in the filtered DataFrame 'starting_df'
        for i, row in starting_df.iterrows():
            # Get the route and split it
            route = self._start_cutoff_routes_dict[str(row.shape_id)].split(" ")
            # If the route has already been adjusted (i.e., 'from_edge' or 'to_edge' are no longer in the route), skip the current shape. Happens if multiple errors per route exist
            if not row.from_edge in route or not row.to_edge in route:
                continue
            # Attempt to correct the route by removing unnecessary edges from the start of the route
            fixed_route, cutoff_edges = gtfs2sumo.correct_start(
                str(row.shape_id),
                row.from_edge,
                row.to_edge,
                row.to_index,
                route,
                5,  # TODO: why is this 5?
                self._net,
            )
            # If a valid fixed route is found
            if fixed_route:
                # Store the edges that were cutoff during the correction process in 'shape_cutoff_edges'
                self._shape_cutoff_beginning_edges[
                    f"{str(row.shape_id)}_start"
                ] = cutoff_edges
                # Update the 'start_end_cutoff_routes' dictionary with the fixed route
                self._start_cutoff_routes_dict[str(row.shape_id)] = " ".join(
                    fixed_route
                )

    def __check_routes_after_tracemapper(self):
        # Create DataFrame for easy visualization
        self._routechecker_df_after_tracemapper = self.__check_routes(
            "routes_after_tracemapper.rou.xml", self._tracemapper_routes_dict
        )
        # # Extract the unique shape IDs from the 'routechecker_df' that correspond to faulty routes, and convert to string
        self._remaining_faulty_sumo_route_ids = list(
            map(
                str,
                self._routechecker_df_after_tracemapper["shape_id"]
                .unique()
                .tolist(),
            )
        )
        # Get the list of valid route IDs
        self._clean_route_ids = list(
            set(self._tracemapper_routes_dict.keys())
            - set(self._remaining_faulty_sumo_route_ids)
        )
        # Create a dictionary of valid SUMO routes ('shape_id': SUMO_route)
        self._clean_routes_dict = gtfs2sumo.filter_dict_by_keys(
            self._tracemapper_routes_dict, self._clean_route_ids
        )
        # Create a dictionary of faulty SUMO routes
        self._faulty_routes_dict = gtfs2sumo.filter_dict_by_keys(
            self._tracemapper_routes_dict, self._remaining_faulty_sumo_route_ids
        )

    def __check_routes_after_junction_correction(self):
        self._routechecker_df_after_junction_correction = self.__check_routes(
            "routes_after_junction_corrections.rou.xml",
            self._faulty_routes_dict,
        )

    def __check_routes_after_beginning_corrections(self):
        self._routechecker_df_after_beginning_correction = self.__check_routes(
            "routes_after_beginning_corrections.rou.xml",
            self._start_cutoff_routes_dict,
        )

    def __attempt_to_correct_faulty_routes_at_ending(self):
        """
        Similarly, we apply the same approach to the end of faulty routes, trimming unconnected sections. TODO can probably be unified with __attempt_to_correct_faulty_routes_at_beginning()
        """
        # Filter the DataFrame 'routechecker_df_after_junction_correction' to only consider routes where an error occurs at the end of the route.
        ending_df = self._routechecker_df_after_beginning_correction[
            self._routechecker_df_after_beginning_correction["to_index"]
            >= self._routechecker_df_after_beginning_correction["route_length"]
            - 1
            - self._CORRECTION_SEARCH_DEPTH
        ]
        # Make a deep copy of the faulty routes dictionary
        self._end_cutoff_routes_dict = copy.deepcopy(
            self._start_cutoff_routes_dict
        )
        # Save trimmed edges for stop-mapping
        self._shape_cutoff_ending_edges = {}
        # Iterate over each row in the filtered DataFrame 'ending_df'
        for i, row in ending_df.iterrows():
            # Get the route and split it
            route = self._end_cutoff_routes_dict[str(row.shape_id)].split(" ")
            # If the route has already been adjusted (i.e., 'from_edge' or 'to_edge' are no longer in the route), skip the current shape. Happens if multiple errors per route exist
            if not row.from_edge in route or not row.to_edge in route:
                continue
            # Attempt to correct the route by removing unnecessary edges from the end of the route
            fixed_route, cutoff_edges = gtfs2sumo.correct_end(
                str(row.shape_id),
                row.from_edge,
                row.from_index,
                row.to_edge,
                route,
                5,  # TODO why is this 5? Magic Number
                self._net,
            )
            # If a valid fixed route is found
            if fixed_route:
                # Store the edges that were cutoff during the correction process in 'shape_cutoff_edges'
                self._shape_cutoff_ending_edges[f"{str(row.shape_id)}_end"] = (
                    cutoff_edges
                )
                # Update the 'start_end_cutoff_routes' dictionary with the fixed route
                self._end_cutoff_routes_dict[str(row.shape_id)] = " ".join(
                    fixed_route
                )

    def __check_routes_after_ending_correction(self):
        self._rou_df_after_ending_corrections = self.__check_routes(
            "routes_after_ending_corrections.rou.xml",
            self._end_cutoff_routes_dict,
        )
        # Obtain number of remaining faulty routes and create list with unique shape_ids
        remaining_faulty_routes = list(
            self._rou_df_after_ending_corrections["shape_id"].unique()
        )
        remaining_faulty_routes = [
            str(entry) for entry in remaining_faulty_routes
        ]
        # Sort them. Easier for visualization and selection.
        self._remaining_faulty_routes = [
            str(x) for x in sorted(map(int, remaining_faulty_routes))
        ]
        # Create dict for remaining faulty routes and optionally generate rou file for verification (netedit). Are the remaining errors valid or do we have to update the methods?
        self._remaining_faulty_routes_dict = {
            shape_id: self._end_cutoff_routes_dict[shape_id]
            for shape_id in self._remaining_faulty_routes
        }

    def __check_routes(self, route_file_name, routes_df):
        create_directory_if_not_exist(
            os.path.join(self._LOG_DIRECTORY, "routes")
        )
        routes_path = os.path.join(
            self._LOG_DIRECTORY, "routes", route_file_name
        )
        # To check whether the found routes are valid, we use SUMO Routechecker. Generate a SUMO rou file as input
        # Routechecker does not handle nested stop elements
        gtfs2sumo.routes_to_XML(
            routes_df,
            self._main_shape_stop_sequences,
            stop_duration=self._DEFAULT_STOP_DURATION,
            path=routes_path,
            ignore_bus_stops=True,
            include_vehicles=False,
            depart=500,
        )
        # Execute routechecker from within the notebook
        routechecker_output = gtfs2sumo.execute_routechecker(
            "/usr/local/Cellar/sumo/1.20.0/share/sumo/tools/route/routecheck.py",
            self._NET_FILE_PATH,
            routes_path,
        )
        # Create DataFrame for easy visualization
        return gtfs2sumo.create_routechecker_df(routechecker_output, routes_df)

    def __summarize_route_checks(self):
        print("========== ROUTECHECKER SUMMARY ==========")
        print("After Initial Tracemapper Call:")
        print("    Number of valid routes: ", len(self._clean_route_ids))
        print(
            "    Number of faulty routes found: ",
            len(self._remaining_faulty_sumo_route_ids),
        )
        print("After Fixes at Junctions:")
        faulty_routes_after_junction_fixes = list(
            map(
                str,
                self._routechecker_df_after_junction_correction["shape_id"]
                .unique()
                .tolist(),
            )
        )
        clean_routes_after_junction_fixes = list(
            set(self._tracemapper_routes_dict.keys())
            - set(faulty_routes_after_junction_fixes)
        )
        print(
            "    Number of valid routes: ",
            len(clean_routes_after_junction_fixes),
        )
        print(
            "    Number of unique faulty routes found: ",
            len(faulty_routes_after_junction_fixes),
        )
        print("After Beginning Fixes:")
        faulty_routes_after_beginning_fixes = list(
            map(
                str,
                self._routechecker_df_after_beginning_correction["shape_id"]
                .unique()
                .tolist(),
            )
        )
        clean_routes_after_beginning_fixes = list(
            set(self._tracemapper_routes_dict.keys())
            - set(faulty_routes_after_beginning_fixes)
        )
        print(
            "    Number of valid routes: ",
            len(clean_routes_after_beginning_fixes),
        )
        print(
            "    Number of unique faulty routes found: ",
            len(faulty_routes_after_beginning_fixes),
        )
        print("After Ending Fixes:")
        faulty_routes_after_ending_fixes = list(
            map(
                str,
                self._rou_df_after_ending_corrections["shape_id"]
                .unique()
                .tolist(),
            )
        )
        clean_routes_after_ending_fixes = list(
            set(self._tracemapper_routes_dict.keys())
            - set(faulty_routes_after_ending_fixes)
        )
        print(
            "    Number of valid routes: ", len(clean_routes_after_ending_fixes)
        )
        print(
            "    Number of unique faulty routes found: ",
            len(faulty_routes_after_ending_fixes),
        )
        print("===========================================")

    def __prepare_routes_for_stop_mapping(self):
        self._final_routes_before_stop_mapping = {}
        # Create dicts for adjusted, now clean routes
        for shape_id in self._end_cutoff_routes_dict.keys():
            if not shape_id in self._remaining_faulty_routes:
                self._final_routes_before_stop_mapping[shape_id] = (
                    self._end_cutoff_routes_dict[shape_id]
                )
        # Add clean found routes to final routes, before stop-mapping
        self._final_routes_before_stop_mapping.update(self._clean_routes_dict)
        self._shapes_with_duplicates = []
        # Tracemapper might find valid routes that contain unnecessary loops. Detect and remove them.
        for (
            shape_id,
            edge_str,
        ) in self._final_routes_before_stop_mapping.items():
            # Split the route string into a list
            edge_list = edge_str.split(" ")
            # Detect any PAIRWISE duplicate edge IDs
            duplicates = gtfs2sumo.detect_pairwise_duplicates(edge_list)
            if duplicates:  # TODO what happens with multiple duplicates?
                # print(shape_id, duplicates)
                # Remove the loop between duplicate pairs
                cleaned_route = gtfs2sumo.remove_detour(
                    edge_list, self._net, self._DETOUR_LENGTH_THRESHOLD
                )
                if len(cleaned_route.split(" ")) < len(edge_list):
                    # Update the 'final_routes_before_stop_mapping' dictionary with the cleaned route
                    self._final_routes_before_stop_mapping[shape_id] = (
                        cleaned_route
                    )
                    self._shapes_with_duplicates.append(shape_id)

    def __map_stops_to_routes(self):
        # TODO is this actually adding any routes?
        """
        For all valid routes, we map the sequence of stops onto the identified SUMO route within an expanded radius of 200 meters.
        To accommodate stops located on previously cut edges, we significantly increase the search radius.
        The method determines the optimal lane, start, and end positions for each stop based on the geo coordinates, explicitly allowing duplicates.
        Some stops serve as the final stop for one route and the starting point for another. If a stop is located on an unmapped street, it is likely that we cut the start of one and end of the other route.
        Since these routes are polar opposites, a single stop cannot be used to service both.
        """
        # Initialize dictionaries and lists to store results
        self._update_stops = {}
        self._shape_ids_non_mappable = []

        # Iterate through each shape and its corresponding route string in 'final_routes'
        for (
            shape_id,
            route_str,
        ) in self._final_routes_before_stop_mapping.items():
            cutoff_edges = []
            # Flag to determine if cutoff occurred at the start of the route
            cutoff_at_start = False
            # Split the route string into a list of edges
            edge_list = route_str.split(" ")
            # Retrieve the stop sequence for the current shape
            main_shape_stop_sequence = self._main_shape_stop_sequences[shape_id]
            all_cutoff_edges = {
                **self._shape_cutoff_beginning_edges,
                **self._shape_cutoff_ending_edges,
            }
            cutoff_str_list = [f"{shape_id}_start", f"{shape_id}_end"]
            for shape_str in cutoff_str_list:
                if shape_str in all_cutoff_edges.keys():
                    # print("cutoff_at_start set to True", shape_str)
                    cutoff_at_start = True
                    cutoff_edges += all_cutoff_edges[shape_str]
            stops_per_shape = gtfs2sumo.map_stops_to_route(
                shape_id,
                edge_list,
                cutoff_edges,
                main_shape_stop_sequence,
                self._net,
                200,  # TODO magic number
                13,  # self._DEFAULT_STOP_LENGTH,
                cutoff_at_start,
                4,  # TODO magic number. Constant?
            )
            if stops_per_shape:
                self._update_stops.update(stops_per_shape)
            else:
                self._shape_ids_non_mappable.append(shape_id)

        # Sort them. Easier for visualization and selection
        self._shape_ids_non_mappable = [
            str(x) for x in sorted(map(int, self._shape_ids_non_mappable))
        ]

    def __summarize_route_stop_mapping(self):
        print("========== ROUTE STOP MAPPING SUMMARY ==========")
        print(
            "Number of Routes were all stops could be mapped: ",
            len(self._final_routes_before_stop_mapping)
            - len(self._shape_ids_non_mappable),
        )
        print(
            "Number of routes with non-mappable stops:",
            len(self._shape_ids_non_mappable),
        )
        print("================================================")

    def __process_extracted_routes(self):
        # Get all routes (clean and faulty) to generate OSM maps
        self._total_routes_dict = copy.deepcopy(
            self._final_routes_before_stop_mapping
        )
        self._total_routes_dict.update(self._remaining_faulty_routes_dict)

        # Remove shape IDs for routes that have non-mappable stops
        target_keys = list(
            set(self._final_routes_before_stop_mapping.keys())
            - set(self._shape_ids_non_mappable)
        )
        # Sort them. Easier for visualization and selection
        target_keys = [str(x) for x in sorted(map(int, target_keys))]
        # Filter 'final_routes' to include only the routes with a valid stop sequence
        self._final_routes = gtfs2sumo.filter_dict_by_keys(
            self._final_routes_before_stop_mapping, target_keys
        )

    def __create_maps_for_extracted_routes(self):
        """
        For each main shape and found SUMO route, create an OSM map to compare them. Especially helpful for debugging.
        """
        # Transform edge list into geo coordinates and create map for the underlying shape and found SUMO route
        # Be aware, takes a couple of minutes.
        directory_faulty_stops_shapes_sumo = os.path.join(
            self._LOG_DIRECTORY, self._MAP_DIRECTORY, "faulty_stops_shapes_sumo"
        )
        create_directory_if_not_exist(directory_faulty_stops_shapes_sumo)
        directory_faulty_shapes_sumo = os.path.join(
            self._LOG_DIRECTORY, self._MAP_DIRECTORY, "faulty_shapes_sumo"
        )
        create_directory_if_not_exist(directory_faulty_shapes_sumo)
        directory_shapes_sumo = os.path.join(
            self._LOG_DIRECTORY, self._MAP_DIRECTORY, "shapes_sumo"
        )
        create_directory_if_not_exist(directory_shapes_sumo)
        for shape_id in self._total_routes_dict.keys():
            tmp_dict = {shape_id: self._checked_main_shapes_dict[shape_id]}
            # Get shape route
            # Transform SUMO route to geo coordinates
            geo_sumo_route = gtfs2sumo.transform_edge_to_geo_points(
                self._total_routes_dict[shape_id], self._net
            )
            # Add 'sumo' to SUMO to distinguish between both
            tmp_dict[f"{shape_id}_sumo"] = geo_sumo_route
            if shape_id in self._shape_ids_non_mappable:
                gtfs2sumo.generate_shape_map(
                    tmp_dict,
                    shape_id,
                    self._bbox_lon_lat,
                    directory_faulty_stops_shapes_sumo,
                    show_flag=True,
                )
            elif shape_id in self._remaining_faulty_routes:
                gtfs2sumo.generate_shape_map(
                    tmp_dict,
                    shape_id,
                    self._bbox_lon_lat,
                    directory_faulty_shapes_sumo,
                    show_flag=True,
                )
            else:
                gtfs2sumo.generate_shape_map(
                    tmp_dict,
                    shape_id,
                    self._bbox_lon_lat,
                    directory_shapes_sumo,
                    show_flag=True,
                )

    def __create_overview_maps_for_routes(self):
        directory = os.path.join(self._LOG_DIRECTORY, self._MAP_DIRECTORY)
        create_directory_if_not_exist(directory)
        # Transform SUMO routes to geo coordinates
        final_routes_geo = gtfs2sumo.transform_edge_dict_to_geo_dict(
            self._final_routes, self._net
        )
        # Map all valid SUMO routes on OSM map
        gtfs2sumo.map_routes(
            final_routes_geo,
            "overview_final_routes",
            self._bbox_lon_lat,
            dir_path=directory,
            show_flag=True,
        )
        # Transform SUMO routes to geo coordinates
        remaining_faulty_routes_geo = gtfs2sumo.transform_edge_dict_to_geo_dict(
            self._remaining_faulty_routes_dict, self._net
        )
        # Map all remaining faulty SUMO routes on OSM map
        gtfs2sumo.map_routes(
            remaining_faulty_routes_geo,
            "overview_remaining_faulty_routes",
            self._bbox_lon_lat,
            dir_path=directory,
            show_flag=True,
        )
        # Get SUMO routes with non-mappable stops
        shape_ids_non_mappable_dict = gtfs2sumo.filter_dict_by_keys(
            self._final_routes_before_stop_mapping, self._shape_ids_non_mappable
        )
        # Get stop sequences for those routes
        stop_sequences_non_mappable = gtfs2sumo.filter_dict_by_keys(
            self._main_shape_stop_sequences, self._shape_ids_non_mappable
        )
        # Transform SUMO routes to geo coordinates
        shape_ids_non_mappable_geo = gtfs2sumo.transform_edge_dict_to_geo_dict(
            shape_ids_non_mappable_dict, self._net
        )
        # Map all routes and their stop sequences to OSM map.
        gtfs2sumo.map_routes_and_stops(
            shape_ids_non_mappable_geo,
            stop_sequences_non_mappable,
            "overview_non_mappable_stops",
            self._bbox_lon_lat,
            path=directory,
            show_flag=True,
        )

    def __transform_stops_for_sequence_mapping(self):
        """
        Transform the found stops to a DataFrame. Retrieve the stop sequence from the main shape stop sequence
        """
        # List to collect new rows
        new_rows = []
        # Update stops according to found lanes for the given final SUMO routes
        for key, value in self._update_stops.items():
            shape_id_index = key.split("_")
            orig_stop_name = self._main_shape_stop_sequences[
                shape_id_index[0]
            ].iloc[int(shape_id_index[1])]["stop_name"]
            new_row = {
                "stop_id": key,
                "stop_name": orig_stop_name,
                "lane_id": value[0],
                "edge_id": value[0].split("_")[0],
                "outside_bbox": False,
                "mapped": 1,
                "start_pos": value[1],
                "end_pos": value[2],
                "parking_length": self._DEFAULT_STOP_LENGTH,
            }
            new_rows.append(new_row)

        # Create a DataFrame from the list of new rows
        self._final_stops_df = pd.DataFrame(new_rows)

    def __consolidate_stops(self):
        # TODO: Maybe run this twice, so that closeby aggregated stops will be merged (see agg_10621_28_11845_31_9629_17_11943_12 agg_10740_2_11649_2)
        # Find unique lanes
        lanes = self._final_stops_df["lane_id"].unique()
        # Create a list to hold groups of close stops
        close_stop_groups = []
        # Check each lane for stops with a delta less than the threshold
        for lane in lanes:
            stops_on_lane = (
                self._final_stops_df[self._final_stops_df["lane_id"] == lane]
                .sort_values(by="start_pos")
                .reset_index(drop=True)
            )
            if (
                len(stops_on_lane) > 1
            ):  # Only consider lanes with more than one stop
                group = []  # Temporary list to hold stops in the current group
                for i in range(len(stops_on_lane) - 1):
                    # Check if the current stop and the next stop are close
                    if stops_on_lane.iloc[i]["stop_name"] == stops_on_lane.iloc[
                        i + 1
                    ]["stop_name"] and (
                        abs(
                            stops_on_lane.iloc[i]["start_pos"]
                            - stops_on_lane.iloc[i + 1]["start_pos"]
                        )
                        < self._STOP_MERGING_DISTANCE
                        or abs(
                            stops_on_lane.iloc[i]["end_pos"]
                            - stops_on_lane.iloc[i + 1]["start_pos"]
                        )
                        < self._STOP_MERGING_DISTANCE
                        or abs(
                            stops_on_lane.iloc[i]["end_pos"]
                            - stops_on_lane.iloc[i + 1]["end_pos"]
                        )
                        < self._STOP_MERGING_DISTANCE
                    ):
                        # Add both stops to the group if they are not already in it
                        if stops_on_lane.iloc[i]["stop_id"] not in [
                            stop["stop_id"] for stop in group
                        ]:
                            group.append(stops_on_lane.iloc[i].to_dict())
                        if stops_on_lane.iloc[i + 1]["stop_id"] not in [
                            stop["stop_id"] for stop in group
                        ]:
                            group.append(stops_on_lane.iloc[i + 1].to_dict())
                    else:
                        # If the next stop is not close, finalize the current group
                        if group:
                            close_stop_groups.append(group)
                            group = (
                                []
                            )  # Reset the group for the next set of close stops
                # Finalize the last group if it exists
                if group:
                    close_stop_groups.append(group)

        # get all stops that will be aggregated
        removable_stops = []
        for group in close_stop_groups:
            for stop in group:
                removable_stops.append(stop["stop_id"])

        # Create a list to hold merged stops
        merged_stops = []

        my_stop_id_mapping = {}
        # Iterate through each group of close stops
        for group in close_stop_groups:
            max_end_pos = max(stop["end_pos"] for stop in group)
            # Ensure the aggregated stop length is able to handle enough buses
            parking_length = len(group) * self._DEFAULT_STOP_LENGTH
            min_start_pos = max(
                max_end_pos - self._DEFAULT_STOP_LENGTH * min(3, len(group)), 0
            )  # Extend the end_pos to meet the minimum length

            # Concatenate stop_id values with the prefix "agg_"
            merged_stop_id = "agg_" + "_".join(
                stop["stop_id"] for stop in group
            )
            my_stop_id_mapping[merged_stop_id] = [s["stop_id"] for s in group]
            stop_name = group[0]["stop_name"]
            # Use the lane_id and description from the first stop in the group
            lane_id = group[0]["lane_id"]

            # Add the merged stop to the list
            merged_stops.append(
                {
                    "stop_id": merged_stop_id,
                    "lane_id": lane_id,
                    "start_pos": min_start_pos,
                    "end_pos": max_end_pos,
                    "outside_bbox": False,
                    "mapped": 1,
                    "stop_name": stop_name,
                    "parking_length": parking_length,
                }
            )
        # Create a new column 'aggregated_id' in stop DataFrame and fill it based on stop_id_mapping
        self._final_stops_df["aggregated_id"] = self._final_stops_df[
            "stop_id"
        ].apply(
            lambda x: next(
                (
                    stop_id
                    for stop_id, value in my_stop_id_mapping.items()
                    if x in value
                ),
                x,
            )
        )
        # Create a new dataframe with the merged stops
        self._merged_stops_df = pd.DataFrame(merged_stops)
        self._not_removed_stops_df = self._final_stops_df[
            ~self._final_stops_df["stop_id"].isin(removable_stops)
        ]

        self._remaining_stops_df = pd.concat(
            [self._not_removed_stops_df, self._merged_stops_df], axis=0
        ).drop(["edge_id", "outside_bbox", "mapped"], axis=1)

    def __write_stops_additional(self):
        directory = os.path.join(self._OUT_DIRECTORY, "routes")
        create_directory_if_not_exist(directory)
        gtfs2sumo.stops_to_XML(
            self._remaining_stops_df,
            os.path.join(directory, self._OUTPUT_PREFIX + "_stops.add.xml"),
        )

    def __summarize_stops(self):
        print("========== GENERATED STOP ADDITIONAL SUMMARY ==========")
        print("Number of Created Stops: ", len(self._remaining_stops_df))
        print("Number of Merged Stops: ", len(self._merged_stops_df))
        print(
            "Number of originally converted Stops: ",
            len(self._not_removed_stops_df),
        )
        print("================================================")

    def __update_stop_sequences(self):
        """
        Update the stop_id, lane_id, edge_id for each stop in main_shape_stop_sequence that is part of 'final_routes'
        """
        self._updated_stop_sequences = copy.deepcopy(
            self._main_shape_stop_sequences
        )
        for shape_id in self._final_routes.keys():
            # Get all stops per route
            filtered_dict = self._final_stops_df[
                self._final_stops_df["stop_id"].str.startswith(shape_id)
            ]
            # Adjust stop_id, lane_id and edge_id
            for i, row in filtered_dict.iterrows():
                orig_shape_id, stop_sequence = row.stop_id.split("_")
                shape_df = self._updated_stop_sequences[orig_shape_id]
                shape_df.iloc[int(stop_sequence), [1, 5, 6]] = (
                    row.aggregated_id,
                    row.lane_id,
                    row.edge_id,
                )

    def __extract_bus_schedule(self):
        # Get main_shape_interval_dict for final routes
        main_shape_interval_dict_final_routes = gtfs2sumo.filter_dict_by_keys(
            self._main_shape_interval_dict, self._final_routes.keys()
        )

        for shape_id in main_shape_interval_dict_final_routes.keys():
            main_shape_interval_dict_final_routes[shape_id][
                "adjusted_earliest_departure"
            ] = main_shape_interval_dict_final_routes[shape_id][
                "earliest_departure"
            ]
            main_shape_interval_dict_final_routes[shape_id][
                "adjusted_last_departure"
            ] = main_shape_interval_dict_final_routes[shape_id][
                "last_departure"
            ]
        # Sort the dict by 'adjusted_earliest_departure' to maintain time sequence (SUMO requirement for vehicles/flows)
        self._sorted_main_shape_interval_dict_final_routes = dict(
            sorted(
                main_shape_interval_dict_final_routes.items(),
                key=lambda item: datetime.strptime(
                    item[1]["adjusted_earliest_departure"], "%H:%M:%S"
                ).time(),
            )
        )

    def __collect_route_sequences(self):
        # Get stop sequences for all 'final_routes'
        self._final_route_stop_sequences = gtfs2sumo.filter_dict_by_keys(
            self._updated_stop_sequences, self._final_routes.keys()
        )
        # For each stop sequence compute the median travel time between stops and add to 'final_route_stop_sequences'
        for shape_id in self._final_route_stop_sequences.keys():
            # TODO this should be simplified
            self._final_route_stop_sequences[shape_id] = (
                gtfs2sumo.compute_median_travel_times(
                    shape_id,
                    self._gtfs_data,
                    self._final_route_stop_sequences[shape_id],
                    self._main_shape_interval_dict[shape_id],
                    self._DEFAULT_STOP_DURATION,
                    0,  # TODO I just use 0 here because we use departPos="stop"
                )
            )

            # Remove consecutive duplicates in 'stop_id' and keep the latter
            self._final_route_stop_sequences[
                shape_id
            ] = self._final_route_stop_sequences[shape_id][
                self._final_route_stop_sequences[shape_id]["stop_id"].shift()
                != self._final_route_stop_sequences[shape_id]["stop_id"]
            ]
            # Reset the index after filtering
            self._final_route_stop_sequences[shape_id] = (
                self._final_route_stop_sequences[shape_id].reset_index(
                    drop=True
                )
            )

        # Create human-readable bus line names for each route for better readability of the SUMO route file
        self._bus_line_info_dict = (
            gtfs2sumo.create_human_readable_bus_line_info(
                self._trip_list, self._main_shape_ids, self._completeness_dict
            )
        )

    def __write_final_route_file(self):
        directory = os.path.join(self._OUT_DIRECTORY, "routes")
        create_directory_if_not_exist(directory)
        # Generate rou file with routes and flows
        gtfs2sumo.routes_to_XML_flow(
            self._final_routes,
            self._final_route_stop_sequences,
            self._sorted_main_shape_interval_dict_final_routes,
            self._DEFAULT_STOP_DURATION,
            os.path.join(directory, self._OUTPUT_PREFIX + "_routes.rou.xml"),
            self._bus_line_info_dict,
            with_parking=True,
        )

    def __gtfs2sumo_summary(self):
        # Quick summary and comparison
        sumo_trips_on_day = len(
            self._trip_list[
                self._trip_list["main_shape_id"].isin(self._final_routes.keys())
            ]
        )
        sumo_active_routes = len(self._final_routes.keys())
        sumo_active_bus_lines = len(
            self._trip_list[
                self._trip_list["main_shape_id"].isin(self._final_routes.keys())
            ]["route_short_name"].unique()
        )
        gtfs_trips_on_day = len(self._gtfs_data["trip_id"].unique())
        gtfs_active_main_shapes = len(self._main_shape_stop_sequences.keys())
        gtfs_active_bus_lines = len(
            self._trip_list["route_short_name"].unique()
        )
        print("=========== SUMMARY ===============")
        print(
            f"Transformed active trips on day: {sumo_trips_on_day} (GTFS: {gtfs_trips_on_day}) {(sumo_trips_on_day / gtfs_trips_on_day) * 100:.2f}%"
        )
        print(
            f"Transformed routes: {sumo_active_routes} (GTFS: {gtfs_active_main_shapes}) {(sumo_active_routes / gtfs_active_main_shapes) * 100:.2f}%"
        )
        print(
            f"Transformed bus lines: {sumo_active_bus_lines} (GTFS: {gtfs_active_bus_lines}) {(sumo_active_bus_lines / gtfs_active_bus_lines) * 100:.2f}%"
        )

    def extract_sumo_traffic(self):
        print("=== Start GTFS2SUMO ===")
        print("=> Preparing GTFS")
        log_time("Filtering GTFS...", self.__filter_gtfs_data)
        log_time("Extracting Main Shapes...", self.__extract_main_shapes)
        if self._CREATE_MAPS:
            log_time(
                "Creating Plain Shape Map...",
                self.__create_plain_main_shape_map,
            )
        log_time("Adjusting Trip List...", self.__trip_list_add_main_shape)
        log_time(
            "Generating Main Shape Data...", self.__generate_main_shape_data
        )
        if self._VERBOSE:
            self.__summarize_gtfs_data()
        if self._CREATE_MAPS:
            log_time("Creating Stops Map...", self.__create_all_stops_map)
        if self._CREATE_MAPS:
            log_time(
                "Creating Grouped Shapes Maps...",
                self.__create_grouped_shape_maps,
            )
        log_time(
            "Generating (Non-)Mappable Stops Additional...",
            self.__create_non_mappable_stops,
        )
        log_time(
            "Finding Shapes with Non-Mappable Stops...",
            self.__find_shapes_with_non_mappable_stops,
        )
        if self._CREATE_MAPS:
            log_time(
                "Storing Shapes with Non-Mappable Stops TO CSV...",
                self.__save_shapes_with_non_mappable_stops,
            )
        if self._CREATE_MAPS:
            log_time(
                "Generating Main Shape Stop Sequences Maps...",
                self.__create_main_shape_stop_sequences_map,
                "main_shape_stop_sequences",
            )
        log_time("Finding Clean Shapes", self.__find_clean_shapes)
        if self._VERBOSE:
            self.__summarize_gtfs_extraction()
        log_time("Resetting Stop Sequences...", self.__reset_stop_sequences)
        log_time(
            "Including Shapes that are cut off at start/end...",
            self.__include_start_end_cut_off_shapes,
        )
        if self._VERBOSE:
            self.__summarize_start_end_cut_off()
        if self._CREATE_MAPS:
            log_time(
                "Generating Cut Main Shape Stop Sequences Maps...",
                self.__create_main_shape_stop_sequences_map,
                "cut_main_shape_stop_sequences",
            )
        print("=> Starting Tracemapper")
        log_time(
            "Preparing Shapes for Tracemapper...",
            self.__prepare_shapes_for_tracemapper,
        )
        log_time("Mapping Routes with Tracemapper...", self.__start_tracemapper)
        log_time(
            "Validating Routes with Routechecker...",
            self.__check_routes_after_tracemapper,
        )
        if self._CREATE_MAPS:
            log_time(
                "Generating Maps for Clean Routes...",
                self.__create_route_file_for_clean_routes,
            )
        log_time(
            "Attempting to fix invalid Routes at junctions...",
            self.__attempt_to_correct_faulty_routes_at_junctions,
        )
        log_time(
            "Validating Routes after junction fix...",
            self.__check_routes_after_junction_correction,
        )
        log_time(
            "Attempting to fix invalid Routes at beginning...",
            self.__attempt_to_correct_faulty_routes_at_beginning,
        )
        log_time(
            "Validating Routes after beginning fixes...",
            self.__check_routes_after_beginning_corrections,
        )
        log_time(
            "Attempting tof fix invalid Routes at ending...",
            self.__attempt_to_correct_faulty_routes_at_ending,
        )
        log_time(
            "Validating Routes after ending fixes...",
            self.__check_routes_after_ending_correction,
        )
        if self._VERBOSE:
            self.__summarize_route_checks()
        print("=> Start Mapping Stops to Routes...")
        log_time(
            "Preparing routes for stop mapping...",
            self.__prepare_routes_for_stop_mapping,
        )
        log_time("Mapping Stops to Routes...", self.__map_stops_to_routes)
        if self._VERBOSE:
            self.__summarize_route_stop_mapping()
        log_time(
            "Preparing Stops and Routes for final processing...",
            self.__process_extracted_routes,
        )
        if self._CREATE_MAPS:
            log_time(
                "Creating Maps for all Routes...",
                self.__create_maps_for_extracted_routes,
            )
        if self._CREATE_MAPS:
            log_time(
                "Creating Overview Maps for Routes...",
                self.__create_overview_maps_for_routes,
            )
        log_time(
            "Prepare Stops for merging...",
            self.__transform_stops_for_sequence_mapping,
        )
        log_time("Merging close-by Stops...", self.__consolidate_stops)
        log_time(
            "Create Stops Additional File...", self.__write_stops_additional
        )
        if self._VERBOSE:
            self.__summarize_stops()
        log_time(
            "Update Stop Sequences with consolidated Stops...",
            self.__update_stop_sequences,
        )
        log_time("Extracting Bus Schedules...", self.__extract_bus_schedule)
        log_time(
            "Collecting final Route Sequences...",
            self.__collect_route_sequences,
        )
        log_time("Writing Final Route File...", self.__write_final_route_file)
        self.__gtfs2sumo_summary()
        self.traffic_extracted = True

    def get_final_routes(self):
        if self.traffic_extracted:
            return self._final_routes

    def get_completeness_dict(self):
        if self.traffic_extracted:
            return self._completeness_dict
