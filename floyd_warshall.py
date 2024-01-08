def floyd_warshall_algorithm(graph):
    # Get the number of vertices in the graph
    vertices = list(graph.keys())
    num_vertices = len(vertices)

    # Initialize the distance matrix with infinite values
    dist_matrix = [[float('inf') for _ in range(num_vertices)] for _ in range(num_vertices)]

    # Set the diagonal elements to 0 as the distance from a vertex to itself is 0
    for i in range(num_vertices):
        dist_matrix[i][i] = 0

    # Populate the distance matrix with the initial weights from the graph
    for i in range(num_vertices):
        for neighbor, weight in graph[vertices[i]].items():
            j = vertices.index(neighbor)
            dist_matrix[i][j] = weight.round(2)

    # Apply the Floyd-Warshall algorithm
    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])

    return dist_matrix


