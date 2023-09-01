import pandas as pd
from Classes import Node
from pyrosm import OSM, get_data

# read all nodes
nodes_df = pd.read_csv('standorte_7.csv', index_col=0)
nodes_dict = nodes_df.to_dict()
nodes = {i: Node(i, nodes_dict['Longitude'][i], nodes_dict['Latitude'][i]) for i in nodes_dict['Type']}

# read all arcs
arcs_df = pd.read_csv('arcs.csv')
arcs_dict = arcs_df.to_dict()
arcs = {(arcs_dict['Start'][i], arcs_dict['End'][i]): arcs_dict['Distance'][i] for i in arcs_dict['Start']}

osm = OSM(get_data("Nordrhein-Westfalen"))
n_drive,  e_drive = osm.get_network(nodes=True, network_type="driving")


print('End!')
