import math

import pandas as pd
import openrouteservice as ors
from openrouteservice.distance_matrix import distance_matrix
import geopy
import time
from Classes import Node

client = ors.Client(key='5b3ce3597851110001cf62482ac01804d2d6434a8cdd87785bfb541a')  # Specify your personal API key
max_num_matrix = 3500  # max number of elements in the distance matrix
locator = geopy.Nominatim(user_agent='TUM_TA_ph')


# read all nodes
df = pd.read_csv('standorte_7.csv', index_col=0)
nodes_dict = df.to_dict()
nodes = {i: Node(i, nodes_dict['Longitude'][i], nodes_dict['Latitude'][i]) for i in nodes_dict['Type']}

# create list of coordinates
coordinates = []
for n in nodes.values():
    coordinates.append([n.long, n.lat])


distances = []
nodes_list = list(nodes.keys())
step = math.floor(max_num_matrix / len(coordinates))
for i in range(0, len(coordinates), step):
    if i + step >= len(coordinates):
        end = len(coordinates)
    else:
        end = i + step
    sour = list(range(i, end))
    driving_distance = distance_matrix(client, profile='driving-hgv', metrics=['distance'], units='km', locations=coordinates, sources=sour)
    time.sleep(1.6)
    new_distances = [[nodes_list[start], nodes_list[end], driving_distance['distances'][start-sour[0]][end]] for start in sour for end in range(len(nodes_list))]
    distances += new_distances


new_df = pd.DataFrame(data=distances, columns=['Start', 'End', 'Distance'])
new_df.to_csv('arcs.csv', index=False)
print(f'Found {len(new_df)} arcs.')


