import math


def calc_distance(start, end):
    dx = start[0] - end[0]
    dy = start[1] - end[1]
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return round(dist, 2)


def dijkstra(dist, vertices, customer):
    # dijkstra algorythm to find the shortest path from each costumer/depot to the other customers/depots
    # initialize
    distance = {v: math.inf for v in vertices}
    predecessor = {v: False for v in vertices}
    distance[customer] = 0
    unvisitid_vertices = vertices.copy()
    # algorythm
    while(len(unvisitid_vertices)) > 0:
        help_dist = {v: distance[v] for v in unvisitid_vertices}
        i = min(help_dist, key=help_dist.get)  # node with minimal distance as current node
        unvisitid_vertices.remove(i)  # delete node from unvisited nodes
        for j in vertices:
            if (i, j) in dist:  # all neighbors
                if j in unvisitid_vertices:
                    alt_dist = distance[i] + dist[i, j]
                    if alt_dist < distance[j]:
                        distance[j] = round(alt_dist, 2)
                        predecessor[j] = i
    return distance, predecessor


def calc_angle(coord_depot, coord_customer):
    vector = [coord_customer[i] - coord_depot[i] for i in range(len(coord_depot))]
    amount = math.sqrt(math.sum([vector[i]**2 for i in range(len(vector))]))
    angle = math.arccos(vector[0]/amount)
    return angle

