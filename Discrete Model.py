import gurobipy as gp
from gurobipy import GRB
import openpyxl
import time
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math


def calc_distance(start, end):
    dx = start[0] - end[0]
    dy = start[1] - end[1]
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return round(dist, 2)


# import file
file = 'Test_Daten_Discrete.xlsx'
platoon = False

wb = openpyxl.load_workbook(file)

# --- Sets --- #


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
v = wb['Others']['B'][8].value  # velocity of trucks
c_d = wb['Others']['B'][9].value  # cost of depreciation if truck used
c_f = wb['Others']['B'][10].value  # fuel cost for trucks
Q = wb['Others']['B'][11].value  # capacity of trucks
len_step = wb['Others']['B'][12].value  # length of time steps in h
# coordinates for all vertices
coord = dict()
coord.update(coord_cus)
coord.update(coord_dep)
coord.update(coord_nod)

# ---Show graph--- #
if False:
    for i in D:
        plt.scatter(coord[i][0], coord[i][1], c='b')
        plt.text(coord[i][0], coord[i][1], i, c='b')

    for i in C:
        plt.scatter(coord[i][0], coord[i][1], c='g')
        plt.text(coord[i][0], coord[i][1], i, c='g')

    for i in N:
        plt.scatter(coord[i][0], coord[i][1], c='r')
        plt.text(coord[i][0], coord[i][1], i, c='r')

    for (i, j) in E:
        plt.plot([coord[i][0], coord[j][0]], [coord[i][1], coord[j][1]], color='black')
    plt.show()

# ---Coying of nodes--- #


# find the closest node for customers and depots and add arcs to and from this node to E
for c in C + D:
    dist_cus = {n: calc_distance(coord[c], coord[n]) for n in N}
    dist_sorted = sorted(dist_cus.items(), key=lambda c: c[1])
    n = dist_sorted[0][0]
    E.append((c, n))
    E.append((n, c))

# dictionary with end nodes as keys and dictionaries (key: name of copied node, value: start node) of start nodes as value, only needed to create A
copies = {j: {f'{j}_{i}': i for i in V if (i, j) in E} for j in N}
copies.update({j: {j: i for i in V if (i, j) in E} for j in (
        C + D)})  # customers and depots have only one connection to the aux network, so they don´t need any copies of them

# create list of all arcs from the copies to other copies
A = []
for k, r in copies.items():  # end nodes
    for j in r:
        A.extend([(i, j) for i in copies[r[j]].keys()])

R = {j: list(copies[j].keys()) for j in copies}  # lists of all copies for each node
# create a dict for each edge that includes all arcs that represent this edge
M = {(i, j): [(cop, j) if (cop, j) in A else (cop, f'{j}_{i}')
              for cop in copies[i] if (cop, f'{j}_{i}') in A or (cop, j) in A] for (i, j) in E}

V = []  # update V with all the copies
for r in R.values():
    V = V + list(r)

o = dict()  # dict to find the Original node of a copy
for k, orig in R.items():
    o.update({copy: k for copy in orig})
d = {(i, j): 3 * calc_distance(coord[o[i]], coord[o[j]]) for (i, j) in A}  # calculating distance for all arcs

# updating service time for all copies
t_s = {c: ts for c in C}
t_s.update({i: 0 for i in V if i not in C})
q.update({i: 0 for i in V if i not in C})  # all non-customer nodes have a demand of 0
# calculate driving time for all arcs in time steps (dist in km devided by km/h devided by length of timesteps)
t_d = {(i, j): math.ceil(d[(i, j)] / v / len_step) for (i, j) in d}

# ---Model--- #


mod = gp.Model('VRP_TPMDSD')

# Definition of Variables
x = mod.addVars(A, K, T, vtype=GRB.BINARY, name='x')
y = mod.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, name='y')
tw = mod.addVars(D, K, vtype=GRB.INTEGER, lb=0, name='tw')
z = mod.addVars(K, vtype=GRB.BINARY, name='z')
s = mod.addVars(V, K, vtype=GRB.BINARY, name='s')
pl = mod.addVars(E, K, T, vtype=GRB.BINARY, name='pl')
pf = mod.addVars(E, K, T, vtype=GRB.BINARY, name='pf')

# ---constraints--- #

mod.addConstrs((gp.quicksum(x[i, j, k, t] for j in V if (i, j) in A for t in T) ==
                gp.quicksum(x[j, i, k, t] for j in V if (j, i) in A for t in T)
                for k in K for i in V), name='flow conservation')

mod.addConstrs((gp.quicksum(x[i, j, k, t] for t in T for (i, j) in M[(m, n)]) <= 1
                for k in K for (m, n) in E), name='each edge only once')

mod.addConstrs((tw[d, k] + t_d[(d, j)] * gp.quicksum(x[d, j, k, t] for t in T) + t_s[j] * s[j, k] <=
                gp.quicksum(t * x[j, m, k, t] for t in T for m in V if (j, m) in A) +
                T_max * (1 - gp.quicksum(x[d, j, k, t] for t in T)) for k in K for d in D for j in V if (d, j) in A),
               name='time propagation depot')

mod.addConstrs((gp.quicksum(t * x[i, j, k, t] for t in T) + t_d[(i, j)] * gp.quicksum(x[i, j, k, t] for t in T) + t_s[
    j] * s[j, k] <=
                gp.quicksum(t * x[j, m, k, t] for t in T for m in V if (j, m) in A) +
                T_max * (1 - gp.quicksum(x[i, j, k, t] for t in T)) for k in K for i in V if i not in D for j in V if
                (i, j) in A), name='time propagation nodes')

mod.addConstrs((gp.quicksum(x[d, j, k, t] for t in T for d in D for j in V if (d, j) in A) == z[k] for k in K),
               name='only one depot')

# Restrictions

mod.addConstrs((gp.quicksum(y[i, k] for i in V) <= Q for k in K), name='capacity of trucks')

mod.addConstrs((gp.quicksum(y[i, k] for k in K) >= q[i] for i in V), name='demand fulfillment of customers')

mod.addConstrs((y[i, k] <= Q * s[i, k] for i in V for k in K), name='only leave products if served')

mod.addConstrs((s[j, k] <= gp.quicksum(x[i, j, k, t] for t in T for i in V if (i, j) in A) for j in C for k in K),
               name='only serve if visiting')

mod.addConstrs((s[i, k] == 0 for k in K for i in V if i not in C), name='servicing of non costumers')

mod.addConstrs((gp.quicksum(t_d[(i, j)] *
                            (x[i, j, k, t] - n_f * pf[m, n, k, t]) for (m, n) in E
                            for (i, j) in M[m, n] for t in T) <=
                Td_max for k in K), name='max driving time')

mod.addConstrs((gp.quicksum(t * x[i, d, k, t] + t_d[i, d] - tw[d, k] for t in T for i in V if (i, d) in A)
                <= Tw_max for d in D for k in K), name='max working time')

# Platoon formation

mod.addConstrs((gp.quicksum(x[i, j, k, t] for (i, j) in M[m, n]) +
                (gp.quicksum(x[i, j, l, t] for (i, j) in M[m, n] for l in K)) / K_max >=
                pl[m, n, k, t] + pf[m, n, k, t] + (gp.quicksum(x[i, j, k, t] for (i, j) in M[m, n])) / K_max
                for k in K for (m, n) in E for t in T), name='two trucks on the same edge at the same time')

mod.addConstrs((gp.quicksum(x[i, j, k, t] for k in K for (i, j) in M[m, n]) - 1 <=
                K_max * gp.quicksum(pl[m, n, k, t] for k in K) for t in T for (m, n) in E),
               name='need for a leader in a platoon')

# objective function
fuel_cost = gp.quicksum(
    c_f * (gp.quicksum(d[i, j] * x[i, j, k, t] for (i, j) in M[m, n]) - n_f * pf[m, n, k, t]) for (m, n) in E for k in K
    for t in T)

labor_cost = gp.quicksum(
    c_l * (gp.quicksum(t * x[i, d, k, t] for t in T for i in V if (i, d) in A) - tw[i, k]) for i in D for k in K)

depreciation_cost = gp.quicksum(c_d * z[k] for k in K)

mod.setObjective(fuel_cost + labor_cost + depreciation_cost, GRB.MINIMIZE)
mod.Params.MIPGap = 0.05
print('Model finished, optimizing next')
mod.optimize()

if mod.Status == GRB.OPTIMAL or mod.Status == GRB.INTERRUPTED:
    edges = {key: x[key].x for key in x if x[key].x > 0.5}
    delivered = {key: y[key].x for key in y if y[key].x > 0.5}
    trucks = [key for key in z if z[key].x > 0.5]

    print(f'fuel cost: {fuel_cost.getValue()}')
    print(f'labor cost: {labor_cost.getValue()}')
    print(f'depecreation cost: {depreciation_cost.getValue()}')

    print('Used Arcs:')
    for k in K:
        for (i, j) in A:
            if x[i, j, k].x >= 0.5:
                print(f'{(i, j, k)}: {x[i, j, k].x}')

    # print routes
    edges_truck = {k: dict() for k in K}
    for (i, j, k) in edges.keys():
        edges_truck[k].update({i: j})

    routes = dict()
    for k, i in edges_truck.items():
        if len(i) > 0:
            s = list(set(D).intersection(list(i.keys())))
            if len(s) > 0:
                s = s[0]
            else:
                s = list(i.keys())[0]
            route = [s, i[s]]
            while i[s] != route[0]:
                s = i[s]
                route.append(i[s])
            routes[k] = route

    for k, v in routes.items():
        print(f'Route {k}: {v}')

        for i in D:
            plt.scatter(coord[i][0], coord[i][1], c='b')
            plt.text(coord[i][0], coord[i][1], i, c='b')

    for i in C:
        plt.scatter(coord[i][0], coord[i][1], c='g')
        plt.text(coord[i][0], coord[i][1], i, c='g')

    for i in N:
        plt.scatter(coord[i][0], coord[i][1], c='r')
        plt.text(coord[i][0], coord[i][1], i, c='r')

    colors = {K[k]: list(mcolors.TABLEAU_COLORS.values())[k] for k in range(len(K))}
    width = {K[k]: (len(K) - k) / 2 for k in range(len(K))}
    for k in K:
        for (i, j) in A:
            if (i, j, k) in edges:
                plt.plot([coord[i][0], coord[j][0]], [coord[i][1], coord[j][1]], colors[k], linewidth=width[k])

    plt.show()
else:
    print('no feasible solution found.')
