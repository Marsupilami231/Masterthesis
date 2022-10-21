import gurobipy as gp
from gurobipy import GRB
import openpyxl
import Classes
import time
from itertools import product

# import file
file = 'Test_Daten.xlsx'
wb = openpyxl.load_workbook(file)

# definition of sets and parameters

# products
W = wb['Products']['A']
a = {W[i].value: wb['Products']['B'][i] for i in range(1, len(W))}
W = [W[i].value for i in range(1, len(W))]
# customers
C = wb['Customers']['A']
coord_cus = {C[i].value: [wb['Customers']['B'][i].value, wb['Customers']['C'][i].value] for i in range(1, len(C))}
cus_y = {C[i].value: wb['Customers']['C'][i].value for i in range(1, len(C))}
q = {C[i].value: {W[w]: float(wb['Customers']['D'][i].value.split(';')[w]) for w in range(len(W))} for i in range(1, len(C))}
s_t = {C[i].value: wb['Customers']['E'][i].value for i in range(1, len(C))}
C = [C[i].value for i in range(1, len(C))]


# auxiliary nodes
N = wb['Nodes']['A']
coord_nod = {N[i].value: [wb['Nodes']['B'][i].value, wb['Nodes']['C'][i].value] for i in range(1, len(N))}
nod_y = {N[i].value: wb['Nodes']['C'][i].value for i in range(1, len(N))}
N = [N[i].value for i in range(1, len(N))]
# depots
D = wb['Depots']['A']
coord_dep = {D[i].value: [wb['Depots']['B'][i].value, wb['Depots']['C'][i].value] for i in range(1, len(D))}
dep_y = {D[i].value: wb['Depots']['C'][i].value for i in range(1, len(D))}
D = [D[i].value for i in range(1, len(D))]
# all vertices combined
V = C + N + D
# Trucks
K = wb['Trucks']['A']  # ID
v = {K[i].value: wb['Trucks']['B'][i].value for i in range(1, len(K))}  #velocity
d = {K[i].value: wb['Trucks']['B'][i].value for i in range(1, len(K))}  #depreciation cost
Q = {K[i].value: wb['Trucks']['B'][i].value for i in range(1, len(K))}  # capacity
c_f = {K[i].value: wb['Trucks']['B'][i].value for i in range(1, len(K))}  # fuel cost
K = [K[i].value for i in range(1, len(K))]

# sizes
S = wb['Sizes']['A']
S = [S[i].value for i in range(1, len(S))]  # platoon sizes

# right now we take all arcs
CvN = C+N
customers = list(product(CvN, CvN))
D_leave = list(product(D, CvN))
D_arrive = list(product(CvN, D))
arcs = customers + D_arrive + D_leave
coord = coord_cus.copy()
coord.update(coord_nod)
coord.update(coord_dep)
A = {(i, j): Classes.calc_distance(coord[i], coord[j]) for (i, j) in arcs if i != j}



#del customers, D_arrive, D_leave, arcs

# definition of parameters
K_max = wb['Others']['B'][0].value
WT_max = wb['Others']['B'][1].value
DT_max = wb['Others']['B'][2].value
n_f = wb['Others']['B'][3].value  # fuel saving factor
n_d = wb['Others']['B'][4].value  # driving relief factor
c_l = wb['Others']['B'][5].value  # labor cost

# definition of variables
m = gp.Model('VRP_TPMDSD')

x = m.addVars(A, K, vtype=GRB.BINARY)
y = m.addVars(C, K, W, vtype=GRB.CONTINUOUS, lb=0)
t = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0)
tw = m.addVars(D, K, vtype=GRB.CONTINUOUS, lb=0)
u = m.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0)
z = m.addVars(K, vtype=GRB.BINARY)
s = m.addVars(V, K, vtype=GRB.BINARY)
p = m.addVars(A, K, K, S, vtype=GRB.BINARY)
pl = m.addVars(A, K, S, vtype=GRB.BINARY)
pf = m.addVars(A, K, vtype=GRB.BINARY)
ps = m.addVars(A, K, vtype=GRB.BINARY)


# # subject to constraints
# # m.addConstrs((gp.quicksum(x[i, j, k] for i in V if (i, j) in A for k in K) == 1 for j in C))
# m.addConstrs((gp.quicksum(y[i, k, w] for k in K) == q[i][w] for i in C for w in W), name='demand fullfilment')
# m.addConstrs((gp.quicksum(a[w] * y[i, k, w] for w in W for i in C) <= Q[k] for k in K), name='truck capacity')
# m
#
#
#
# # objective function
# fuel_cost = gp.quicksum(K[k].cd * A[i, j] * (x[i, j, k] - n_f * pf[i, j, k]) for (i, j) in A for k in K)
# labor_cost = gp.quicksum(c_l * (t[i, k] - tw[i, k]) for i in D for k in K)
# depreciation_cost = gp.quicksum(K[k].cd * z[k] for k in K)
# m.setObjective(fuel_cost + labor_cost + depreciation_cost, GRB.MINIMIZE)
#
# m.optimize()


