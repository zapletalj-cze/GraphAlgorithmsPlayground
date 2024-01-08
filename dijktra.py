import math
import queue


def dijkstra_algorithm(G, u):
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
