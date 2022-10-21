import math


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
    dist = math.sqrt(dx**2 + dy**2)
    return round(dist, 2)
