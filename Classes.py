import math
import gurobipy as gp
from gurobipy import GRB
import openpyxl


class Node:

    def __init__(self, ID, x, y, q=0, st=0):
        self.ID = ID
        self.x = x
        self.y = y
        self.q = q
        self.st = st

    def __str__(self):
        return f'(Node: ID:{self.ID}, x:{self.x}, y:{self.y}, demand:{self.q}, service time:{self.st})'

    def __repr__(self):
        return f'(Node: ID:{self.ID}, x:{self.x}, y:{self.y}, demand:{self.q}, service time:{self.st})'


class Truck:

    def __init__(self, ID, v, cd, Q, cf):
        self.ID = ID
        self.Q = Q
        self.v = v
        self.cd = cd
        self.cf = cf

    def __str__(self):
        return f'(Truck: ID:{self.ID}, capacity:{self.Q}, velocity:{self.v}, depreciation cost:{self.cd}, fuel cost:{self.cf})'

    def __repr__(self):
        return f'(Truck: ID:{self.ID}, capacity:{self.Q}, velocity:{self.v}, depreciation cost:{self.cd}, fuel cost:{self.cf})'


def calc_distance(start, end):
    dx = start[0] - end[0]
    dy = start[1] - end[1]
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return round(dist, 2)


def callback(m, where):
    if where == GRB.Callback.MIPSOL:
        ind = {var[list(var.keys())[0]].VarName.split('[')[0]: {k: var[k].index for k in var} for var in
               m._var_list}
        vars = m.getVars()
        sol = m.cbGetSolution(vars)
        for k in m._sets['K']:
            for i in m._sets['CN']:
                for j in m._sets['V']:
                    if (i, j) in m._sets['A']:
                        # check constraint for time propagation
                        lhs = sol[ind['t'][j, k]]
                        rhs = sol[ind['t'][i, k]] + sol[ind['x'][i, j, k]] * m._para['v'][k] * m._para['d'][
                            i, j] + sol[ind['s'][j, k]] * m._para['t_s'][j] - m._para['Tw_max'] * (
                                      1 - sol[ind['x'][i, j, k]])
                        if lhs < rhs:
                            m.cbLazy(vars[ind['t'][j, k]] >= vars[ind['t'][i, k]] + vars[ind['x'][i, j, k]] *
                                     m._para['v'][k] * m._para['d'][i, j] + vars[ind['s'][j, k]] * m._para['t_s'][j] -
                                     m._para['Tw_max'] * (1 - vars[ind['x'][i, j, k]]))
                            print(f'Constraint {i}, {j}, {k} added.')
                            m._countLazy += 1

    else:
        pass


def read_data(path):
    wb = openpyxl.load_workbook(path)

    # products
    W = wb['Products']['A']
    a = {W[i].value: wb['Products']['B'][i].value for i in range(1, len(W))}
    W = [W[i].value for i in range(1, len(W))]

    # customers
    C = wb['Customers']['A']
    coord_cus = {C[i].value: [wb['Customers']['B'][i].value, wb['Customers']['C'][i].value] for i in range(1, len(C))}
    q = {C[i].value: {W[w]: float(str(wb['Customers']['D'][i].value).split(';')[w]) for w in range(len(W))} for i in
         range(1, len(C))}
    t_s = {C[i].value: wb['Customers']['E'][i].value for i in range(1, len(C))}
    C = [C[i].value for i in range(1, len(C))]  # customers

    # auxiliary nodes
    N = wb['Nodes']['A']
    coord_nod = {N[i].value: [wb['Nodes']['B'][i].value, wb['Nodes']['C'][i].value] for i in range(1, len(N))}
    N = [N[i].value for i in range(1, len(N))]


    # depots
    D = wb['Depots']['A']
    coord_dep = {D[i].value: [wb['Depots']['B'][i].value, wb['Depots']['C'][i].value] for i in range(1, len(D))}
    D = [D[i].value for i in range(1, len(D))]
    t_s.update({i: 0 for i in D})
    # all vertices combined
    V = C + N + D
    CN = C + N

    # Trucks
    K = wb['Trucks']['A']  # ID
    v = {K[i].value: wb['Trucks']['B'][i].value for i in range(1, len(K))}  # velocity
    c_d = {K[i].value: wb['Trucks']['C'][i].value for i in range(1, len(K))}  # depreciation cost
    Q = {K[i].value: wb['Trucks']['D'][i].value for i in range(1, len(K))}  # capacity
    c_f = {K[i].value: wb['Trucks']['E'][i].value for i in range(1, len(K))}  # fuel cost
    K = [K[i].value for i in range(1, len(K))]

    # sizes
    S = wb['Sizes']['A']
    S = [S[i].value for i in range(1, len(S))]  # platoon sizes

    # all coordinates in one dict
    coord = coord_cus.copy()
    coord.update(coord_dep)

    # Arcs
    sh = wb['Arcs']
    A = [(sh['A'][r].value, sh['A'][c].value) for r in range(1, len(sh['A'])) for c in
         range(1, len(sh['A'])) if sh[r + 1][c].value == 1]
    # if used with 'A' as column identifier column comes first; if numbers used, the row comes first

    # copy auxiliary nodes
    N_copy = []
    counters = []
    for n in N:
        counter = 0
        for (i, j) in A:
            if j == n:
                counter += 1
                N_copy.append(f'{n}_{counter}')
        counters.append(counter)
    N = N_copy
    # update parameter of nodes
    t_s.update({i: 0 for i in N})
    coord_node_copied = {n: coord_nod[n[:3]] for n in N}
    coord.update(coord_node_copied)  # all coordinates
    # update Sets
    V = C + N + D
    CN = C + N
    A = [(i, j) for i in V for j in V if (i[:3], j[:3]) in A]

    # calculate distances
    d = {(i, j): calc_distance(coord[i], coord[j]) for (i, j) in A}

    # single parameters
    # definition of parameters
    K_max = wb['Others']['B'][0].value
    Tw_max = wb['Others']['B'][1].value
    Td_max = wb['Others']['B'][2].value
    n_f = wb['Others']['B'][3].value  # fuel saving factor
    n_d = wb['Others']['B'][4].value  # driving relief factor
    c_l = wb['Others']['B'][5].value  # labor cost

    sets = dict()
    sets['W'] = W
    sets['D'] = D
    sets['C'] = C
    sets['N'] = N
    sets['V'] = V
    sets['CN'] = CN
    sets['K'] = K
    sets['S'] = S
    sets['A'] = A

    parameter = dict()
    parameter['c_d'] = c_d
    parameter['Q'] = Q
    parameter['c_f'] = c_f
    parameter['v'] = v
    parameter['d'] = d
    parameter['a'] = a
    parameter['q'] = q
    parameter['K_max'] = K_max
    parameter['Tw_max'] = Tw_max
    parameter['Td_max'] = Td_max
    parameter['n_f'] = n_f
    parameter['n_d'] = n_d
    parameter['c_l'] = c_l
    parameter['t_s'] = t_s

    return W, D, C, N, V, CN, K, S, A, c_d, Q, c_f, v, d, a, q, K_max, Tw_max, Td_max, n_f, n_d, c_l, t_s, coord, sets, parameter
