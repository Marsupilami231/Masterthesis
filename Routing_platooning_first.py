import openpyxl
import methods_old as met
import time
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math


# import file
file = 'Test_Daten_Discrete.xlsx'

wb = openpyxl.load_workbook(file)

# Customers
C = wb['Customers']['A']
coord_cus = {C[i].value: [wb['Customers']['B'][i].value, wb['Customers']['C'][i].value] for i in range(1, len(C))}
q = {C[i].value: wb['Customers']['D'][i].value for i in range(1, len(C))}
C = [C[i].value for i in range(1, len(C))]

# aux nodes
N = wb['Nodes']['A']
coord_nod = {N[i].value: [wb['Nodes']['B'][i].value, wb['Nodes']['C'][i].value] for i in range(1, len(N))}
N = [N[i].value for i in range(1, len(N))]

# depots
D = wb['Depots']['A']
coord_dep = {D[i].value: [wb['Depots']['B'][i].value, wb['Depots']['C'][i].value] for i in range(1, len(D))}
D = [D[i].value for i in range(1, len(D))]

# all vertices combined
V = C + N + D
CN = C + N

# Edges
sh = wb['Arcs_new']
E1 = [(sh['A'][r].value, sh['A'][c].value) for r in range(1, len(sh['A']))
      for c in range(1, len(sh['A'])) if sh[r + 1][c].value == 1]
# if used with 'A' as column identifier column comes first; if numbers used, the row comes first
E2 = [(i, j) for (j, i) in E1]
E = E1 + E2
del E1, E2

# simple sets
T_max = wb['Others']['B'][6].value  # number of time steps
T = list(range(T_max))
K_max = wb['Others']['B'][0].value  # max number of trucks
K = {f'K{i}' for i in range(K_max)}

# ---Parameters--- #
Tw_max = wb['Others']['B'][1].value  # max working time
Td_max = wb['Others']['B'][2].value  # max driving time
n_f = wb['Others']['B'][3].value  # fuel saving factor
n_d = wb['Others']['B'][4].value  # driving relief factor
c_l = wb['Others']['B'][5].value  # labor cost
ts = wb['Others']['B'][7].value  # service time at customers
velocity = wb['Others']['B'][8].value  # velocity of trucks
c_d = wb['Others']['B'][9].value  # cost of depreciation if truck used
c_f = wb['Others']['B'][10].value  # fuel cost for trucks
Q = wb['Others']['B'][11].value  # capacity of trucks
len_step = wb['Others']['B'][12].value  # length of time steps in h
# coordinates for all vertices
coord = dict()
coord.update(coord_cus)
coord.update(coord_dep)
coord.update(coord_nod)

# find the closest node for customers and depots and add arcs to and from this node to E
for c in C + D:
    dist_cus = {n: met.calc_distance(coord[c], coord[n]) for n in N}
    dist_sorted = sorted(dist_cus.items(), key=lambda c: c[1])
    n = dist_sorted[0][0]
    E.append((c, n))
    E.append((n, c))

d = {(i, j): met.calc_distance(coord[i], coord[j]) for (i, j) in E}


# find the shortest path between all customers/depots to reduce the size of the problem
shortest_distance = dict()
predecessor_shortest_path = dict()
for i in C + D:
    shortest_distance[i], predecessor_shortest_path[i] = met.dijkstra(d, V, i)


# allocate the customers to the clostest depot
allocation = {d: [] for d in D}
for c in C:
    dist_depot = {d: shortest_distance[c][d] for d in D}
    d = min(dist_depot, key=dist_depot.get)
    allocation[d].append(c)
print(allocation)

# create first route
demand = {d: 0 for d in D}
angle = dict()
for d in D:
    # calculate demand of depot
    for c in allocation[d]:
        demand[d] += q[c]
        angle[c] = met.calc_angle(coord[d], coord[c])



