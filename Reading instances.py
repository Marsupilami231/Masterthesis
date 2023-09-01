import math

import pandas as pd
import openrouteservice as ors
from openrouteservice.directions import directions
from openrouteservice.distance_matrix import distance_matrix
import numpy as np
import folium
import osmnx as ox
import geopy
from math import radians, cos, sin, asin, sqrt
from itertools import product
from Classes import Node
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

import random
import time
import os
from datetime import datetime

client = ors.Client(key='5b3ce3597851110001cf62482ac01804d2d6434a8cdd87785bfb541a')  # Specify your personal API key
max_num_matrix = 3500  # max number of elements in the distance matrix
locator = geopy.Nominatim(user_agent='TUM_TA_ph')


# class Node:
#
#     def __init__(self, ID, long, lat, q=0, st=0):
#         self.ID = ID
#         self.long = long
#         self.lat = lat
#         self.q = q
#         self.st = st
#
#     def __str__(self):
#         return f'(Node: ID:{self.ID}, long:{self.long}, lat:{self.lat}, demand:{self.q}, service time:{self.st})'
#
#     def __repr__(self):
#         return f'(Node: ID:{self.ID}, long:{self.long}, lat:{self.lat}, demand:{self.q}, service time:{self.st})'


class visit:

    def __init__(self, node, finish_time):
        self.node = node
        self.time = finish_time

    def __str__(self):
        return f'leaves node {self.node.ID} at time {self.time}.'

    def __repr__(self):
        return f'leaves node {self.node.ID} at time {self.time}.'


def haversine(start, end):
    long1 = radians(start.long)
    long2 = radians(end.long)
    lat1 = radians(start.lat)
    lat2 = radians(end.lat)

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlong / 2) ** 2
    r = 6371
    dist = 2 * r * asin(sqrt(a))

    return dist


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / math.pi


def find_near_arcs(nodes):
    distance = {s: {e: haversine(nodes[s], nodes[e]) for e in nodes if e != s if haversine(nodes[s], nodes[e]) < 30} for s in nodes}
    arcs = dict()
    for start, distance_from_start in distance.items():
        arcs[start] = []
        min_dist = min(distance_from_start,
                       key=distance_from_start.get)  # find the node with the shortest distance to start node
        arcs[start].append(min_dist)  # add the arc with the shortest distance to the dict of all arcs
        del distance_from_start[min_dist]  # delete the distance just added
        while len(distance_from_start) > 0:
            add = True
            cutoff = 100
            min_dist = min(distance_from_start,
                           key=distance_from_start.get)  # find node with the next shortest distance
            new_vec = (nodes[min_dist].long - nodes[start].long,
                       nodes[min_dist].lat - nodes[start].lat)  # vector for new possible arc
            for end in arcs[start]:  # cycle through existing arcs
                old_vec = (nodes[end].long - nodes[start].long, nodes[end].lat - nodes[start].lat)
                angle = angle_between(new_vec, old_vec)
                if angle * math.sqrt(distance[start][min_dist]) < cutoff:  # check if the new arc is too close to exisiting arc
                    add = False  # if yes, dont add it
                    break  # one failure is enough to not add it
            if add:
                arcs[start].append(min_dist)  # add only if no other similar arcs
            del distance_from_start[min_dist]  # delete the checked arc

    return arcs

    # min_distance = {n: min(list(distance[n].values())) for n in nodes}
    # max_distance = {n: max(list(distance[n].values())) for n in nodes}
    # max_min_distance = max(list(min_distance.values()))
    # min_max_distance = min(list(max_distance.values()))
    # arcs = []
    # for start in distance:
    #     for end in distance[start]:
    #         v1 = (nodes[end].long - nodes[start].long, nodes[end].lat - nodes[start].lat)
    #         for end2 in distance[start]:
    #             v2 = (nodes[end2].long - nodes[start].long, nodes[end2].lat - nodes[start].lat)
    #             angle = angle_between(v1, v2)
    #             if angle * distance[start][end2] / 100:
    #                 arcs.append(())
    # return arcs


df = pd.read_csv('standorte_7.csv', index_col=0)

nodes_dict = df.to_dict()
nodes = {i: Node(i, nodes_dict['Longitude'][i], nodes_dict['Latitude'][i]) for i in nodes_dict['Type']}

max_long = max(list(nodes_dict['Longitude'].values()))
min_long = min(list(nodes_dict['Longitude'].values()))
max_lat = max(list(nodes_dict['Latitude'].values()))
min_lat = min(list(nodes_dict['Latitude'].values()))

mid_long = (max_long + min_long) / 2
mid_lat = (max_lat + min_lat) / 2
loc = (mid_lat, mid_long)
fmap = folium.Map(location=loc, tiles='cartodbpositron', zoom_start=8)

for n in nodes.values():
    if n.ID in [1, 5111]:
        col = 'red'
    else:
        col = 'green'

    folium.Marker(location=(n.lat, n.long),
                  tooltip=folium.map.Tooltip(
                      f'<p><b>Customer: {n.ID}</b><br>long: {round(n.long, 3)}<br>lat: {round(n.lat, 3)}<br>demand: {n.q}<br>serv time: {n.st}</p>'),
                  icon=folium.Icon(color=col)
                  ).add_to(fmap)

arcs = find_near_arcs(nodes)

for start in arcs:
    for end in arcs[start]:
        folium.PolyLine([tuple([nodes[start].lat, nodes[start].long]), tuple([nodes[end].lat, nodes[end].long])], color='blue').add_to(fmap)

sum_edges = sum(list(range(1, len(nodes))))
edges = list(product(nodes, nodes))
num_edges = len(edges)

coordinates = []
for n in nodes.values():
    coordinates.append([n.long, n.lat])

coordinates_test = coordinates[0:20]

distances = dict()
nodes_list = list(nodes.keys())
step = math.floor(max_num_matrix / len(coordinates))
for i in range(0, len(coordinates), step):
    if i + step >= len(coordinates):
        end = len(coordinates) - 1
    else:
        end = i + step
    dest = list(range(i, end+1))
    driving_distance = distance_matrix(client, profile='driving-hgv', metrics=['distance'], units='km', locations=coordinates, destinations=dest)
    new_distances = {(nodes_list[start], nodes_list[end]): driving_distance['distances'][start][end] for start in range(len(nodes_list)) for end in dest}
    distances.update(new_distances)

new_df = pd.DataFrame(data=distances, columns=['Start', 'End', 'Distance'])

print(driving_distance)
fmap.save("map.html")
print('The End!')
