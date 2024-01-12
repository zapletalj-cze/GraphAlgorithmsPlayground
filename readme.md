# GraphAlgorithmsPlayground

This is school's project repo! This project focuses on simplifying and analyzing road networks from OpenStreetMap using a script that transforms the intricate road system into a graph. The script also utilizes a table for converting point IDs to indices allowing interoperability between linear features, points, and graph indices in both directions. Testing has been conducted exclusively on the datasets described below to ensure the reliability of the results.


# Test data
The preprocessing script and algorithms have been tested on the following datasets:

-   Subset of the road network from OpenStreetMap for the Moravian-Silesian Region, data from Geofabrik are not recommended due to limited  attributes
-   Vector dataset ARCCR 500 containing districts in the Moravian-Silesian Region.
-   Dataset DATA200 comprising municipalities in the Moravian-Silesian Region.


# Data preprocessing
## Projection Adjustment
All datasets are transformed to the projection of the road network to ensure consistent spatial alignment.

## Spatial Subset Creation
The script includes functionality for generating a spatial subset of input data based on selected regions from the polygon defined in the regions_file.

## Road Category Selection
Road categories are selected based on the highway tag values. The categories "trunk" and "motorway" have been unified into the "roadway" category. And categories 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street', 'service', 'pedestrian', 'track', 'raceway', 'escape', 'road', 'busway' have been unified into the "tertiary" category.

## Road Network Simplification
For graph simplification, nodes of the second degree (coinciding with exactly two edges) were extracted. On these nodes, line merging took place under the assumption that maximum speed and one-way attributes are identical. 

## Intersection Extraction
Roads were split at first degree nodes, assuming that the extracted first degree node (end point) intersects with continous line. By dividing, the first degree node was removed, and a third degree (intersection) node was created.

## Weights Computation for All Edges in the Graph
Euclidean distance over the network converted to local UTM.
Time - calculated as distance divided by maximum speed (from OSM attribute).
Time influenced by road curvature - computed as the ratio of the distance between endpoints and the road length. This coefficient is multiplied by the maximum speed. For coefficients (k) < 0.5, a value of 0.5 is used, and for coefficients (k) >= 0.9, a value of 1 is applied (unchanged).

## Conversion of GeoDataframe to oriented graph
The function obtains generated nodes and identifies accessible nodes for each point based on attributes from the road network. For evaluating the orientation of edges, it uses the "oneway" attribute and the direction of the linestring within the line. All IDs are converted to indices for the proper functioning of graph algorithms.

## Start and end Points Preparation
There are several methods available for creating start/end points:
### Directly by coordinates:
Using the attribute start_end_coordinates, specified as a list: [x_start, y_start, x_end, y_end]
### From list of cities 
If the start_end_coordinates attribute is left as an empty list, the user is prompted to provide two city names from the displayed list of cities within the defined region before running the Dijkstra algorithm.
In both cases, the specified points are joined to the nearest points in the road network before running the Dijkstra algorithm.

# Dijkstra Algorithm Implementation
To find the shortest path between points, the Dijkstra algorithm is implemented. The input is the index of the starting point, and the algorithm returns a list of predecessors. Subsequently, this list of predecessors is converted into the final path.

## Conversion of the Path as a List of Indices to Geodataframe
In the last phase, the list of points is transformed using a table back into the original nodes with its Node_ID. By utilizing attributes in the lines (start, end), which identify specific points, the path is reconstructed as a Geodataframe and exported to a Geopackage.


# Solution for discontinous road network
The automatically generated network is discontinuous, and most points are not reachable from every point. A function has been implemented to find the nearest other point and apply the Dijkstra algorithm. The start and end points are beign replaced in this process.

# Comparison with nav services
This section presents selected results from testing the final version of the script with various city combinations. For the shortest distance, the Dijkstra algorithm achieved less favorable results due to a less dense road network. However, the results for the shortest time show consistency between the script and navigation services.

## Shortest distance comparison
![Shortest distance comparison, Ostravice-Mosty u Jablunkova](images/Ostravice_Mosty_distance_comparison.png)
![Shortest distance comparison, Hnojník-Paskov](images/Hnojník_Paskov_distance_comparison.png)

## Fastest route comparison
![Fastest route comparison, Ostravice-Mosty u Jablunkova](images/Ostravice_Mosty_fastest_comparison.png)
![Fastest route comparison, Hnojník-Paskov](images/Hnojník_Paskov_fastest_comparison.png)


