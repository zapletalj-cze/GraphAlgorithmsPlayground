import math
import queue


def convert_graph_to_positive(graph):
    min_value = float('inf')
    # find min value
    for outer_key, inner_dict in graph.items():
        for value in inner_dict.values():
            min_value = min(min_value, value)

    # calculate necessary values shift
    if min_value <= 0:
        print(f'Shifting values in graph to be positively weighted.')
        print(f'Min value in values: {min_value}')
        adjustment = abs(min_value) + 1
        # shift values by adjustment
        for outer_key, inner_dict in graph.items():
            for inner_key in inner_dict:
                graph[outer_key][inner_key] += adjustment

    return graph


def dijkstra_algorithm(G, u):
    G = convert_graph_to_positive(G)
    n = len(G)
    # set infinite distance
    d = [math.inf] * (n + 1)
    # no predecessors
    P = [-1] * (n + 1)
    # create priority queue
    PQ = queue.PriorityQueue()
    # u = start
    d[u] = 0
    # add start vertex
    PQ.put((0, u))

    while not PQ.empty():
        du, u = PQ.get()
        # relaxation
        for v, wuv in G[u].items():
            if d[v] > d[u] + wuv:
                # found shorter way - update distance
                d[v] = d[u] + wuv
                # update predecessor
                P[v] = u
                # add to priority queue
                PQ.put((d[v], v))
    return P


def pathRec(P, u, v):
    path = []
    while v != u and v != -1:
        path.append(v)
        v = P[v]
    path.append(v)
    return path
