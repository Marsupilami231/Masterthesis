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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

import random
import time
import os
from datetime import datetime

client = ors.Client(key='5b3ce3597851110001cf62482ac01804d2d6434a8cdd87785bfb541a')  # Specify your personal API key
locator = geopy.Nominatim(user_agent='TUM_TA_ph')


class Node:

    def __init__(self, ID, long, lat, q=0, st=0):
        self.ID = ID
        self.long = long
        self.lat = lat
        self.q = q
        self.st = st

    def __str__(self):
        return f'(Node: ID:{self.ID}, long:{self.long}, lat:{self.lat}, demand:{self.q}, service time:{self.st})'

    def __repr__(self):
        return f'(Node: ID:{self.ID}, long:{self.long}, lat:{self.lat}, demand:{self.q}, service time:{self.st})'


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
    lat1 = radians(start.long)
    lat2 = radians(end.long)

    dlong = long2 - long1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlong / 2) ** 2
    r = 6371
    dist = 2 * r * asin(sqrt(a))

    return dist


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
m = folium.Map(location=loc, tiles='cartodbpositron', zoom_start=8)

for n in nodes.values():
    folium.Marker(location=(n.lat, n.long),
                  tooltip=folium.map.Tooltip(f'<p><b>Customer: {n.ID}</b><br>long: {round(n.long,3)}<br>lat: {round(n.lat,3)}<br>demand: {n.q}<br>serv time: {n.st}</p>'),
                  icon=folium.Icon(color='green')
                  ).add_to(m)


sum_edges = sum(list(range(1, len(nodes))))
edges = list(product(nodes, nodes))
num_edges = len(edges)

distance = {s: {e: haversine(nodes[s], nodes[e]) for e in nodes if e != s} for s in nodes}
min_distance = {n: min(list(distance[n].values())) for n in nodes}
max_distance = {n: max(list(distance[n].values())) for n in nodes}
max_min_distance = max(list(min_distance.values()))
min_max_distance = min(list(max_distance.values()))

coordinates = []
for n in nodes.values():
    coordinates.append([n.long, n.lat])

driving_time = distance_matrix(client, coordinates)

m.save("map.html")
print('The End!')
