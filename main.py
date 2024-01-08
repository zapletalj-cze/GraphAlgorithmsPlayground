import os.path
import geopandas as gpd
import processing
import dijktra
import floyd_warshall
import sys
import pandas as pd

workspace = r'C:\Computation\CUNI\01_ZS1\GINFO\cviceni\cv5_GINFO\\'
results_folder = r'C:\Computation\CUNI\01_ZS1\GINFO\cviceni\cv5_GINFO\results\\'

# input data
roads_osm_file = r'data\OSM\drop.gpkg'
cities_file = r'data\okresy_ARCCR\Obce_Data200\MSK_Obce_Data200.gpkg'
regions_file = r'data\okresy_ARCCR\Okresy_ArcCR500\MSK_Okresy.gpkg'

# parameters
# supports 'distance' (Euclidean), 'time' (based on max-speed/distance), 'time_max_speed_curvature'
weighting_method = 'distance'

# selection based on: https://wiki.openstreetmap.org/wiki/Key:highway
# roadway = 'motorway', 'trunk', 'motorway_link', 'trunk_link'
road_types = ['roadway', 'primary']

regions_attribute = 'NAZ_LAU1'  # attribute in regions_file used for region selection
selected_districts = ['Frýdek-Místek']  # list of values from the regions_attribute in regions_file

# OPTIONAL, can be defined later on
# format [x_start, y_start, x_end, y_end]
# list of x,y pairs to find the nearest node in a graph to build a path between
start_end_coordinates = ['308196', '5505752', '309066', '5506684']  # or empty list


# ***************** END // INPUTS /// USER PARAMETERS ***************#
gdf_roads = gpd.read_file(os.path.join(workspace, roads_osm_file))
gdf_cities = gpd.read_file(os.path.join(workspace, cities_file))
gdf_regions = gpd.read_file(os.path.join(workspace, regions_file))

# Set all geometries to road network crs
if gdf_roads.crs == 'EPSG:4326':
    gdf_cities, gdf_regions = [processing.reproject_to_wgs84(item) for item in [gdf_cities, gdf_regions]]
else:
    gdf_cities, gdf_regions = [item.to_crs(gdf_roads.crs)for item in [gdf_cities, gdf_regions]]

# Clip all data with selected regions
gdf_roads, gdf_cities = processing.clip_data_to_regions(gdf_roads=gdf_roads,
                                                        gdf_cities=gdf_cities,
                                                        gdf_selected_regions=gdf_regions,
                                                        regions_attribute=regions_attribute,
                                                        regions_values=selected_districts)

# OSM Road Network - categories and extract selected road types
# https://wiki.openstreetmap.org/wiki/Key:highway
gdf_roads = processing.categorize_osm(gdf_osm_data=gdf_roads,
                                      road_types=road_types)

# Extract 2nd degree nodes
gdf_nodes_cont = processing.extract_nodes(gdf_roads)[1]

# # OSM Road Network - simplify road network
# gdf_roads = processing.refine_road_network(gdf_roads)

# OSM Road Network - road network merge at selected nodes
gdf_roads = processing.merge_lines_at_continous_nodes(gdf_lines=gdf_roads,
                                                      gdf_points=gdf_nodes_cont,
                                                      primary_key='full_id')

#  Extract all nodal types based on degree of a vertex
gdf_endpoints, gdf_nodes_cont, gdf_intersections = processing.extract_nodes(gdf_roads)

# Split lines at intersections
gdf_roads = processing.split_lines_at_intersections(gdf_lines=gdf_roads,
                                                    gdf_intersections=gdf_endpoints,
                                                    primary_key='full_id')

gdf_roads = gdf_roads.drop_duplicates(subset='geometry')

# reproject to metric system
gdf_roads = processing.reproject_to_local_utm(gdf_roads)
gdf_roads['maxspeed'] = gdf_roads['maxspeed'].astype('int')
gdf_roads['length'] = gdf_roads['geometry'].length

# Calculate weights for edges
gdf_roads = processing.calculate_edge_weights(gdf_lines=gdf_roads, weighting_method=weighting_method)

# Generate oriented graph
graph, relation_table = processing.build_dict_graph_from_geodataframe(gdf_lines=gdf_roads,
                                                                      orientation_attribute='oneway',
                                                                      weights_attribute='weight',
                                                                      line_primary_key='full_id')
processing.create_temporal_check_file(relation_table)
# Start and end point as user input
if not start_end_coordinates:
    start_index, end_index = processing.get_start_end_point_from_user(gdf_cities, gdf_points=relation_table)
else:
    start_index, end_index = processing.get_start_end_point_from_user_coordinates(start_end_coordinates,
                                                                                  gdf_points=relation_table)

p = dijktra.dijkstra_algorithm(graph, start_index)
path = dijktra.pathRec(p, start_index, end_index)

# path (node indicies to its primary key)
path = processing.reverse_list(path)
if len(path) > 2:
    id_list = processing.translate_path_index_to_pk(path, relation_table)
else:
    print(f'Path between selected points was not found.')
    sys.exit()

# convert path to line Geodataframe
gdf_line = processing.convert_path_to_line(id_list, gdf_roads)

# export results
gdf_line.to_file(os.path.join(results_folder, 'final_path_dijkstra.gpkg'))

# All pairs shortest path matrix
distance_matrix = floyd_warshall.floyd_warshall_algorithm(graph)
for row in distance_matrix:
    print(row)
df = pd.DataFrame(distance_matrix)
df.to_csv(os.path.join(results_folder, 'OD_matrix_floyd.csv'))

