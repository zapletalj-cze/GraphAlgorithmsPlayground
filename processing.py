import functools
import warnings
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import split


# For testing purposes only
def create_temporal_check_file(to_test):
    path = r'C:\Computation\CUNI\01_ZS1\GINFO\cviceni\cv5_GINFO\checks\TEMP_check.gpkg'
    to_test.to_file(path)


def flatten_reduce_lambda(matrix):
    return list(functools.reduce(lambda x, y: x + y, matrix, []))


def clip_data_to_regions(gdf_roads, gdf_cities, gdf_selected_regions, regions_attribute, regions_values):
    """
    :param gdf_roads: GeoDataFrame to be clipped.
                      Accepts either a file path (str) or a GeoDataFrame containing roads data.

    :param gdf_cities: str or GeoDataFrame to be clipped.
                       Accepts either a file path (str) or a GeoDataFrame containing cities data.

    :param gdf_selected_regions: Geometry used for clipping.
                                Represents the geometry used for clipping gdf_roads and gdf_cities.

    :param regions_attribute: Name of the attribute in gdf_selected_regions used for region selection. Specifies the
    attribute name from gdf_selected_regions used to select regions for clipping (e.g., region_name).

    :param regions_values: List of values from the attribute in gdf_selected_regions. Represents a list of values
    from the attribute specified by regions_attribute. Regions with these values will be selected for clipping (e.g.,
    ['Czechia', 'Slovakia', 'Poland']).

    :return: GeoDataframe - categorized road network

    """

    if regions_attribute is not None and regions_values is not None:
        gdf_selected_regions_subset = gdf_selected_regions[gdf_selected_regions[regions_attribute].isin(regions_values)]
        gdf_selected_regions = gdf_selected_regions_subset.dissolve()
        warnings.filterwarnings("ignore")
        gdf_selected_regions['geometry'] = gdf_selected_regions['geometry'].buffer(0.0001)
        warnings.resetwarnings()

    gdf_roads = gpd.clip(gdf_roads, gdf_selected_regions, keep_geom_type=True)
    gdf_cities = gpd.clip(gdf_cities, gdf_selected_regions, keep_geom_type=True)
    gdf_roads = gdf_roads.reset_index(drop=True)
    gdf_cities = gdf_cities.reset_index(drop=True)
    return gdf_roads, gdf_cities


def categorize_osm(gdf_osm_data, road_types):
    """
    :param gdf_osm_data: str, path object or file-like object
    :param road_types: list, list of selected road types, supports [
    'roadway', 'primary', 'secondary', ('tertiary'- not recommended)]
    :return: GeoDataframe - categorized road network

    # the main aim is to select road types based on predefined categories
    # implement a method for weighted graph (use the nested method defined above this one)
    # create nodes as edge endpoints
    """
    gdf_osm_data = gdf_osm_data[['full_id', 'highway', 'maxspeed', 'junction', 'oneway', 'geometry']]
    roads_categories = {'roadway': ['motorway', 'trunk', 'motorway_link', 'trunk_link'],
                        'primary': ['primary', 'primary_link'],
                        'secondary': ['secondary', 'secondary_link'],
                        'tertiary': ['tertiary', 'tertiary_link', 'unclassified', 'residential', '	living_street',
                                     'service', 'pedestrian', 'track', 'raceway', 'escape', 'road', 'busway']}

    roads_categories_subset = {key: roads_categories[key] for key in road_types}
    roads_categories_subset = flatten_reduce_lambda(list(roads_categories_subset.values()))
    gdf_osm_data = gdf_osm_data[gdf_osm_data['highway'].isin(roads_categories_subset)]
    gdf_osm_data['highway'] = gdf_osm_data['highway'].fillna('NODATA')
    gdf_osm_data['maxspeed'] = gdf_osm_data['maxspeed'].fillna(50)
    gdf_osm_data['oneway'] = gdf_osm_data['oneway'].fillna('no')
    gdf_osm_data.loc[gdf_osm_data['junction'] == 'roundabout', 'oneway'] = 'yes'
    return gdf_osm_data


def extract_nodes(gdf_roads):
    """
    :param gdf_roads: GeoDataFrame (geometry type: line)
    """
    gdf_roads = gdf_roads.explode()
    first_coord = gdf_roads['geometry'].apply(lambda g: g.coords[0])
    last_coord = gdf_roads['geometry'].apply(lambda g: g.coords[-1])
    start_points = gpd.GeoDataFrame(geometry=[Point(coords) for coords in first_coord])
    end_points = gpd.GeoDataFrame(geometry=[Point(coords) for coords in last_coord])
    gdf_nodes = pd.concat([start_points, end_points], ignore_index=True)
    gdf_nodes = gdf_nodes.groupby('geometry').size().reset_index(name='point_count')
    gdf_nodes = gpd.GeoDataFrame(gdf_nodes, geometry='geometry').set_crs(gdf_roads.crs)
    gdf_endpoints = gdf_nodes[gdf_nodes['point_count'] == 1]
    gdf_nodes_cont = gdf_nodes[gdf_nodes['point_count'] == 2]
    gdf_intersections = gdf_nodes[gdf_nodes['point_count'] > 2]
    warnings.resetwarnings()
    return gdf_endpoints, gdf_nodes_cont, gdf_intersections


def reproject_to_local_utm(gdf_input):
    estimated = gdf_input.estimate_utm_crs(datum_name='WGS 84')
    print(f'Reprojecting geodataframe from {gdf_input.crs} to {estimated}')
    gdf_input = gdf_input.to_crs(estimated)
    return gdf_input


def reproject_to_wgs84(gdf_input):
    gdf_input = gdf_input.to_crs(4326)
    return gdf_input


def refine_road_network(gdf_roads, primary_key):
    """
    :param primary_key:
    :param gdf_roads: GeoDataFrame (geometry type: line)
    """
    gdf_roads['unique_key'] = gdf_roads['highway'] + gdf_roads['maxspeed'].astype(str) + gdf_roads['oneway']
    gdf_roads_dissolved = gdf_roads.dissolve(by=['unique_key'], aggfunc='first')
    gdf_roads_dissolved = gdf_roads_dissolved[[primary_key, 'highway', 'maxspeed', 'oneway', 'geometry']]
    gdf_roads_dissolved['fid'] = range(1, len(gdf_roads_dissolved) + 1)
    gdf_roads_dissolved = gdf_roads_dissolved.to_crs(4326)
    return gdf_roads_dissolved


# method currently in test mode
def merge_lines_at_continous_nodes(gdf_lines, gdf_points, primary_key):
    warnings.filterwarnings("ignore")
    gdf_lines = gdf_lines.explode()
    gdf_lines[['first_x', 'first_y']] = gdf_lines['geometry'].apply(lambda g: pd.Series(g.coords[0]))
    gdf_lines[['last_x', 'last_y']] = gdf_lines['geometry'].apply(lambda g: pd.Series(g.coords[-1]))
    for index, row in gdf_points.iterrows():
        point_geometry = row['geometry']
        gdf_line_intersect = gdf_lines[gdf_lines['geometry'].intersects(point_geometry)]
        gdf_line01 = None
        gdf_line02 = None
        for index_line, _ in gdf_line_intersect.iterrows():
            if gdf_line01 is None:
                gdf_line01 = gdf_line_intersect[
                    gdf_line_intersect[['last_x', 'last_y']].eq([point_geometry.x, point_geometry.y]).all(axis=1)]
            if gdf_line02 is None:
                gdf_line02 = gdf_line_intersect[
                    gdf_line_intersect[['first_x', 'first_y']].eq([point_geometry.x, point_geometry.y]).all(axis=1)]
        if len(gdf_line01) == 1 and len(gdf_line02) == 1:
            same_values = (
                    gdf_line01.iloc[0][['maxspeed', 'oneway']] == gdf_line02.iloc[0][['maxspeed', 'oneway']]).all()
            if same_values:
                gdf_lines = gdf_lines[~gdf_lines[primary_key].isin(gdf_line01[primary_key])]
                gdf_lines = gdf_lines[~gdf_lines[primary_key].isin(gdf_line02[primary_key])]
                connected_line = LineString(
                    list(gdf_line01['geometry'].iloc[0].coords) + list(gdf_line02['geometry'].iloc[0].coords))
                first_x_update, first_y_update = connected_line.xy[0][0], connected_line.xy[1][0]
                last_x_update, last_y_update = connected_line.xy[0][-1], connected_line.xy[1][-1]
                # update geometries
                gdf_line01.at[gdf_line01.index[0], 'geometry'] = connected_line

                # update attributes
                gdf_line01.at[gdf_line01.index[0], 'first_x'] = first_x_update
                gdf_line01.at[gdf_line01.index[0], 'first_y'] = first_y_update
                gdf_line01.at[gdf_line01.index[0], 'last_x'] = last_x_update
                gdf_line01.at[gdf_line01.index[0], 'last_y'] = last_y_update
                gdf_lines = pd.concat([gdf_lines, gdf_line01], ignore_index=True)
                gdf_lines = gdf_lines.reset_index(drop=True)
    return gdf_lines


def split_lines_at_intersections(gdf_lines, gdf_intersections, primary_key):
    coord_sys = gdf_lines.crs
    key_index = 900000
    gdf_lines[['first_x', 'first_y']] = gdf_lines['geometry'].apply(lambda g: pd.Series(g.coords[0]))
    gdf_lines[['last_x', 'last_y']] = gdf_lines['geometry'].apply(lambda g: pd.Series(g.coords[-1]))
    for index_point, row_point in gdf_intersections.iterrows():
        point_geometry = row_point['geometry']
        gdf_line_intersect = gdf_lines[gdf_lines['geometry'].intersects(point_geometry)]
        condition1 = gdf_line_intersect[
            gdf_line_intersect[['last_x', 'last_y']].eq([point_geometry.x, point_geometry.y]).all(axis=1)]
        condition2 = gdf_line_intersect[
            gdf_line_intersect[['first_x', 'first_y']].eq([point_geometry.x, point_geometry.y]).all(axis=1)]
        gdf_line_intersect = gdf_line_intersect[~gdf_line_intersect[primary_key].isin(condition1[primary_key])]
        gdf_line_intersect = gdf_line_intersect[~gdf_line_intersect[primary_key].isin(condition2[primary_key])]
        if len(gdf_line_intersect) > 0:
            for index_line, row_line in gdf_line_intersect.iterrows():
                primary_key_single_line = row_line[primary_key]
                linestring = row_line['geometry']
                warnings.filterwarnings("ignore")
                line_segments = list((split(linestring, point_geometry)).geoms)
                gdf_lines = gdf_lines.query(f"{primary_key} != '{primary_key_single_line}'")
                warnings.resetwarnings()
                for geometry in line_segments:
                    new_line_add = row_line.copy()
                    new_line_add['geometry'] = geometry
                    new_line_add[primary_key] = 'w' + str(key_index)
                    key_index += 1
                    gdf_new_line_add = gpd.GeoDataFrame(data=[new_line_add], geometry='geometry', crs=coord_sys)
                    gdf_lines = pd.concat([gdf_lines, gdf_new_line_add])
    gdf_lines = gdf_lines.reset_index(drop=True)
    gdf_lines = gdf_lines[[primary_key, 'highway', 'maxspeed', 'oneway', 'geometry']]
    return gdf_lines


def get_start_end_point_from_user(gdf_cities, gdf_points):
    gdf_points.sindex
    print(f'List of cities from provided file:')
    gdf_cities = gdf_cities.drop_duplicates(subset='NAMN1').to_crs(gdf_points.crs)
    for i in range(0, len(gdf_cities['NAMN1'].tolist()), 7):
        names = gdf_cities['NAMN1'].tolist()[i:i + 7]
        print(names)
    start_name = input("Insert the start point name: ")
    end_name = input("Insert the end point name: ")
    geometry_start = gdf_cities.loc[gdf_cities['NAMN1'] == start_name, 'geometry'].values[0]
    geometry_end = gdf_cities.loc[gdf_cities['NAMN1'] == end_name, 'geometry'].values[0]

    distances = gdf_points['geometry'].distance(geometry_start)
    start_point_index = distances.idxmin()
    distances = gdf_points['geometry'].distance(geometry_end)
    end_point_index = distances.idxmin()

    return start_point_index, end_point_index


def get_start_end_point_from_user_coordinates(coordinates_list, gdf_points):
    distances = gdf_points['geometry'].distance(Point(coordinates_list[0], coordinates_list[1]))
    start_point_index = distances.idxmin()
    distances = gdf_points['geometry'].distance(Point(coordinates_list[2], coordinates_list[3]))
    end_point_index = distances.idxmin()
    return start_point_index, end_point_index


def calculate_edge_weights(gdf_lines, weighting_method):
    """
    :param gdf_lines:
    :param weighting_method: str, supports distance (Euclidean),
     time (based on max-speed/distance), time_max_speed_curvature
    """

    if 'distance' in gdf_lines.columns:
        pass
    else:
        gdf_lines['length'] = gdf_lines['geometry'].length

    if weighting_method == 'distance':
        gdf_lines.rename(columns={'length': 'weight'}, inplace=True)

    if weighting_method == 'time_max_speed':
        if 'maxspeed' in gdf_lines.columns:
            gdf_lines['time'] = (gdf_lines['length']/1000) / gdf_lines['maxspeed']
            gdf_lines.rename(columns={'time': 'weight'}, inplace=True)
        else:
            warnings.warn("Max speed attribute not found in GeoDataFrame, use different weighting method.",
                          category=UserWarning)

    if weighting_method == 'time_max_speed_curvature':
        gdf_lines[['start_x', 'start_y', 'end_x', 'end_y']] = gdf_lines['geometry'].apply(
            lambda line: pd.Series([line.xy[0][0], line.xy[1][0], line.xy[0][-1], line.xy[1][-1]]))
        gdf_lines['distance_direct'] = (gdf_lines.apply
                                        (lambda row: Point(row['start_x'], row['start_y']).distance
                                        (Point(row['end_x'], row['end_y'])), axis=1))
        gdf_lines = gdf_lines.drop(['start_x', 'start_y', 'end_x', 'end_y'], axis=1)
        gdf_lines['curvature_c'] = gdf_lines['distance_direct'] / gdf_lines['length']
        gdf_lines['curvature_c'] = gdf_lines['curvature_c'].where(gdf_lines['curvature_c'] >= 0.5, 0.5).where(
            gdf_lines['curvature_c'] <= 0.90, 1)
        gdf_lines['time'] = (gdf_lines['length']/1000) / (gdf_lines['maxspeed'] * gdf_lines['curvature_c'])
        gdf_lines.rename(columns={'time': 'weight'}, inplace=True)
    return gdf_lines


def build_dict_graph_from_geodataframe(gdf_lines, orientation_attribute, weights_attribute, line_primary_key):
    graph_road = {}
    gdf_lines[['start_x', 'start_y', 'end_x', 'end_y']] = gdf_lines['geometry'].apply(
        lambda line: pd.Series([line.xy[0][0], line.xy[1][0], line.xy[0][-1], line.xy[1][-1]]))
    gdf_points = gpd.GeoDataFrame(columns=['ID', 'start_to', 'end_to', 'geometry'], crs=gdf_lines.crs)
    gdf_points['start_to'] = []
    gdf_points['end_to'] = []
    gdf_lines['from'] = 'nodata'
    gdf_lines['to'] = 'nodata'
    gdf_lines[['from', 'to']] = gdf_lines[['from', 'to']].astype(str)
    gdf_lines[weights_attribute] = gdf_lines[weights_attribute].astype('float')
    gdf_lines[weights_attribute] = gdf_lines[weights_attribute].round(2)

    point_id = 10000
    for index_line, row_line in gdf_lines.iterrows():
        line_id = row_line[line_primary_key]
        point1 = Point(row_line['start_x'], row_line['start_y'])
        point2 = Point(row_line['end_x'], row_line['end_y'])
        if any(gdf_points['geometry'].eq(point1)):
            match_id = gdf_points.loc[gdf_points['geometry'] == point1, 'ID'].values[0]
            gdf_points.loc[gdf_points['ID'] == match_id, 'start_to'] = gdf_points.loc[
                gdf_points['ID'] == match_id, 'start_to'].apply(lambda x: str(x) + ',' + line_id)
            gdf_lines.at[index_line, 'from'] = match_id

        else:
            data_start = {
                'ID': ['P' + str(point_id)],
                'start_to': line_id,
                'geometry': [point1]
            }
            gdf_new_point = gpd.GeoDataFrame(data_start, geometry='geometry', crs=gdf_lines.crs)
            gdf_points = pd.concat([gdf_points, gdf_new_point], ignore_index=True)
            gdf_lines.at[index_line, 'from'] = 'P' + str(point_id)
            point_id += 1

        if any(gdf_points['geometry'].eq(point2)):
            match_id = gdf_points.loc[gdf_points['geometry'] == point2, 'ID'].values[0]
            gdf_points.loc[gdf_points['ID'] == match_id, 'end_to'] = gdf_points.loc[
                gdf_points['ID'] == match_id, 'end_to'].apply(lambda x: str(x) + ',' + line_id)
            gdf_lines.at[index_line, 'to'] = match_id
        else:
            data_end = {
                'ID': ['P' + str(point_id)],
                'end_to': line_id,
                'geometry': [point2]
            }
            gdf_lines.at[index_line, 'to'] = 'P' + str(point_id)
            gdf_new_point = gpd.GeoDataFrame(data_end, geometry='geometry', crs=gdf_lines.crs)
            gdf_points = pd.concat([gdf_points, gdf_new_point], ignore_index=True)
            point_id += 1

    # Point ID to index table
    gdf_points['index'] = gdf_points.index
    gdf_points['index'] = gdf_points['index'].astype(int)
    gdf_indices = gdf_points[['index', 'ID', 'geometry']]

    # Build oriented graph
    for index_point, row_point in gdf_points.iterrows():
        point_index = row_point['index']
        line_ends = set(str(row_point['end_to']).split(','))
        line_starts = set(str(row_point['start_to']).split(','))
        line_ends.discard('nan')
        line_starts.discard('nan')
        point_simple_dict = {}

        for line in line_starts:
            weight = gdf_lines.loc[gdf_lines[line_primary_key] == line, weights_attribute].values[0]
            dest_id = gdf_lines.loc[gdf_lines[line_primary_key] == line, 'to'].values[0]
            dest_index = gdf_points.loc[gdf_points['ID'] == dest_id, 'index'].iloc[0]
            if dest_index == point_index:
                dest_id = gdf_lines.loc[gdf_lines[line_primary_key] == line, 'from'].values[0]
                dest_index = gdf_points.loc[gdf_points['ID'] == dest_id, 'index'].iloc[0]

            point_simple_dict[dest_index] = weight
        for line in line_ends:
            weight = gdf_lines.loc[gdf_lines[line_primary_key] == line, weights_attribute].values[0]
            dest_id = gdf_lines.loc[gdf_lines[line_primary_key] == line, 'to'].values[0]
            dest_index = gdf_points.loc[gdf_points['ID'] == dest_id, 'index'].iloc[0]
            if dest_index == point_index:
                dest_id = gdf_lines.loc[gdf_lines[line_primary_key] == line, 'from'].values[0]
                dest_index = gdf_points.loc[gdf_points['ID'] == dest_id, 'index'].iloc[0]
            if gdf_lines.loc[gdf_lines[line_primary_key] == line, orientation_attribute].values[0] == 'no':
                point_simple_dict[dest_index] = weight
        graph_road[point_index] = point_simple_dict
    return graph_road, gdf_indices


def translate_path_index_to_pk(path: list, relation_table):
    id_list = []
    for index in path:
        if index > 0:
            pk_id = relation_table[relation_table['index'] == index]['ID'].values
            id_list.append(pk_id)
    return id_list


def reverse_list(list_input):
    new_list = list_input[::-1]
    return new_list


def convert_path_to_line(path_reindexed, gdf_lines):
    gdf_path = gpd.GeoDataFrame(columns=gdf_lines.columns, geometry='geometry')
    cols = ['full_id', 'highway', 'maxspeed', 'oneway', 'weight', 'from', 'to', 'geometry']
    for i in range(len(path_reindexed) - 1):
        value1 = str(path_reindexed[i][0])
        value2 = str(path_reindexed[i + 1][0])
        select = (gdf_lines['from'] == value1) & (gdf_lines['to'] == value2) | (gdf_lines['from'] == value2) & (
                    gdf_lines['to'] == value1)
        result_row = gdf_lines.loc[select]
        new_gdf = gpd.GeoDataFrame(result_row, geometry='geometry')
        if len(new_gdf) == 1:
            warnings.filterwarnings("ignore")
            gdf_path = pd.concat([gdf_path, new_gdf], ignore_index=True)
            warnings.resetwarnings()

    gdf_path = gdf_path[cols]
    return gdf_path
