import math
import gurobipy as gp
from gurobipy import GRB


class Node:

    # this class is used for
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
        # m.addConstrs(
        #     (t[j, k] >= t[i, k] + x[i, j, k] * d[i, j] / v[k] + s[j, k] * t_s[j] - Tw_max * (1 - x[i, j, k]) for i in
        #      CvN for j
        #      in V if (i, j) in A for k in K), name='time propagation')
        for k in m._sets['K']:
            for i in m._sets['CN']:
                for j in m._sets['V']:
                    if (i, j) in m._sets['A']:
                        # check constraint for time propagation
                        lhs = sol[ind['t'][j, k]]
                        rhs = sol[ind['t'][i, k]] + sol[ind['x'][i, j, k]] * m._param['v'][k] * m._param['d'][
                            i, j] + sol[ind['s'][j, k]] * m._param['t_s'][j] - m._param['Tw_max'] * (
                                      1 - sol[ind['x'][i, j, k]])
                        if lhs < rhs:
                            m.cbLazy()
                            print('Constraints not working)')

    else:
        pass
