import gurobipy as gp
from gurobipy import GRB
import openpyxl
import Classes
import time
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# import file
file = 'Test_Daten_2.xlsx'
platoon = False


W, D, C, N, V, CN, K, S, A, c_d, Q, c_f, v, d, a, q, K_max, Tw_max, Td_max, n_f, n_d, c_l, t_s, coord, sets, parameter = Classes.read_data(file)

demand = {w: sum([q[i][w] for i in C]) for w in W}
total_demand = sum([demand[w] * a[w] for w in W])
# definition of variables
m = gp.Model('VRP_TPMDSD')
print(A)
x = m.addVars(A, K, vtype=GRB.BINARY, name='x')
y = m.addVars(C, K, W, vtype=GRB.CONTINUOUS, lb=0, name='y')
t = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, name='t')
tw = m.addVars(D, K, vtype=GRB.CONTINUOUS, lb=0, name='tw')
# u = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0)
z = m.addVars(K, vtype=GRB.BINARY, name='z')
s = m.addVars(V, K, vtype=GRB.BINARY, name='s')
p = m.addVars(A, K, K, S, vtype=GRB.BINARY, name='p')
pl = m.addVars(A, K, S, vtype=GRB.BINARY, name='pl')
pf = m.addVars(A, K, vtype=GRB.BINARY, name='pf')
ps = m.addVars(A, S, vtype=GRB.CONTINUOUS, lb=0, name='ps')
o = m.addVars(A, K, S, vtype=GRB.BINARY, name='o')

var_list = [x, y, t, tw, z, s, p, pl, pf, ps, o]

# subject to constraints
# network flow
m.addConstrs((gp.quicksum(x[i, j, k] for i in V if (i, j) in A for k in K) <= 1 for j in C),
             name='only one visit per customer')
m.addConstrs(
    (gp.quicksum(x[i, j, k] for j in V if (i, j) in A) == gp.quicksum(x[j, i, k] for j in V if (i, j) in A) for k in K
     for i in V), name='flow conservation')
time_propagation = m.addConstrs(
    (t[j, k] >= t[i, k] + x[i, j, k] * d[i, j] / v[k] + s[j, k] * t_s[j] - Tw_max * (1 - x[i, j, k]) for i in CN for j
     in V if (i, j) in A for k in K), name='time propagation')
m.addConstrs(
    (t[j, k] >= tw[i, k] + x[i, j, k] * d[i, j] / v[k] + s[j, k] * t_s[j] - Tw_max * (1 - x[i, j, k]) for i in D for j
     in V if (i, j) in A for k in K), name='time propagation depot')
m.addConstrs((tw[i, k] <= t[i, k] for i in D for k in K), name='limit for waiting time')
m.addConstrs((t[i, k] <= Tw_max * gp.quicksum(x[i, j, k] for j in V if (i, j) in A) for i in V for k in K),
             name='time to 0')
m.addConstrs((tw[i, k] <= Tw_max * gp.quicksum(x[i, j, k] for j in V if (i, j) in A) for i in D for k in K),
             name='wait time to 0')
m.addConstrs((gp.quicksum(x[i, j, k] for i in D for j in V if (i, j) in A) == z[k] for k in K),
             name='exactly one depot')

# restrictions
m.addConstrs((gp.quicksum(y[i, k, w] for k in K) == q[i][w] for i in C for w in W), name='demand fulfillment')
m.addConstrs((gp.quicksum(a[w] * y[i, k, w] for w in W for i in C) <= Q[k] for k in K), name='truck capacity')
m.addConstrs((y[i, k, w] <= Q[k] * gp.quicksum(x[i, j, k] for j in V if (i, j) in A) for k in K for i in C for w in W),
             name='only serve when visit')
m.addConstrs((gp.quicksum(y[i, k, w] for w in W) <= Q[k] * s[i, k] for i in C for k in K), name='customer serving')
m.addConstrs((gp.quicksum(s[i, k] for i in C) <= Q[k] * z[k] for k in K), name='truck usage')
m.addConstrs((gp.quicksum(d[i, j] / v[k] * (x[i, j, k] - n_d * pf[i, j, k]) for (i, j) in A) <= Td_max for k in K),
             name='max driving time')
m.addConstrs((t[i, k] - tw[i, k] <= Tw_max for i in D for k in K), name='max working time')

# --- platoon formation ---
if platoon:
    m.addConstrs(
        (x[i, j, k] + x[i, j, l] >= 2 * gp.quicksum(p[i, j, k, l, s] for s in S) for (i, j) in A for k in K for l in K if
         l != k), name='platoon formation')
    m.addConstrs(
        (t[i, k] - t[i, l] <= Tw_max * (1 - gp.quicksum(p[i, j, k, l, s] for s in S)) for i in CN for j in V if (i, j) in A
         for k in K for l in K if k != l), name='start time of platoon')
    m.addConstrs(
        (tw[i, k] - tw[i, l] <= Tw_max * (1 - gp.quicksum(p[i, j, k, l, s] for s in S)) for i in D for j in V if (i, j) in A
         for k in K for l in K if k != l), name='start time of platoon')
    m.addConstrs((p[i, j, k, l, s] == p[i, j, l, k, s] for (i, j) in A for k in K for l in K if k != l for s in
                  S), name='pairwise platoons')
    m.addConstrs(
        ((s - 1) * o[i, j, k, s] == gp.quicksum(p[i, j, k, l, s] for l in K if k != l) for (i, j) in A for k in K for s in
         S), name='size of platoon')
    m.addConstrs((gp.quicksum(o[i, j, k, s] for s in S) <= 1 for (i, j) in A for k in K), name='only one platoon per arc')
    m.addConstrs((s * ps[i, j, s] == gp.quicksum(o[i, j, k, s] for k in K) for (i, j) in A for s in S),
                 name='size of platoon')
    m.addConstrs((gp.quicksum(pl[i, j, k, s] for k in K) == ps[i, j, s] for (i, j) in A for s in S),
                 name='number platoon leader')
    m.addConstrs((pl[i, j, k, s] <= o[i, j, k, s] for (i, j) in A for k in K for s in S), name='platoon leader')
    m.addConstrs((pf[i, j, k] <= gp.quicksum(o[i, j, k, s] for s in S) for (i, j) in A for k in K), name='platoon follower')
    m.addConstrs((pf[i, j, k] + gp.quicksum(pl[i, j, k, s] for s in S) <= 1 for (i, j) in A for k in K),
                 name='position in platoon')
    m.addConstrs((tw[g, k] <= Tw_max * gp.quicksum(o[i, j, k, s] for (i, j) in A for s in S) for g in D for k in K),
                 name='wait time at depot')

# objective function
fuel_cost = gp.quicksum(c_f[k] * d[i, j] * (x[i, j, k] - n_f * pf[i, j, k]) for (i, j) in A for k in K)
labor_cost = gp.quicksum(c_l * (t[i, k] - tw[i, k]) for i in D for k in K)
depreciation_cost = gp.quicksum(c_d[k] * z[k] for k in K)
m.setObjective(fuel_cost + labor_cost + depreciation_cost, GRB.MINIMIZE)

# lazy constraints
for k in K:
    for i in CN:
        for j in V:
            if (i, j) in A:
                time_propagation[i, j, k].Lazy = 1

m._countLazy = 0
m.Params.MIPGap = 0.05
# m.Params.LazyConstraints = 1
m._var_list = var_list
m._sets = sets
m._para = parameter
m.optimize()

print(f'{m._countLazy} lazy contraints were added')

if m.Status == GRB.OPTIMAL or m.Status == GRB.INTERRUPTED:
    edges = {key: x[key].x for key in x if x[key].x > 0.5}
    delivered = {key: y[key].x for key in y if y[key].x > 0.5}
    trucks = [key for key in z if z[key].x > 0.5]

    print(fuel_cost.getValue())
    print(labor_cost.getValue())
    print(depreciation_cost.getValue())

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

    print(delivered)
    print(trucks)

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
