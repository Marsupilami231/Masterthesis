import math
import numpy as np
import openpyxl
import time
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import operator
import numpy as np
import csv
from openrouteservice.directions import directions
import openrouteservice as ors

client = ors.Client(key='5b3ce3597851110001cf62482ac01804d2d6434a8cdd87785bfb541a')  # Specify your personal API key

new_network = False
file = r'C:\Git\Masterthesis\Test_Daten_Heuristik.xlsx'
file_edges = 'edges.csv'


# --- definition of functions --- #
def pythagoras(start, end):
    # calculate the pythagoras distance between two nodes, both inputs should be Coord
    diff_lon = start.lon - end.lon
    diff_lat = start.lat - end.lat
    dist = math.sqrt(diff_lon ** 2 + diff_lat ** 2)
    return round(dist, 2)


def dijkstra(_distance, _starting_node):
    # dijkstra algorythm to find the shortest path from one starting node to all other nodes
    # initialize
    new_distance = {v: math.inf for v in V}
    predecessor = {v: None for v in V}
    new_distance[_starting_node] = 0
    unvisited_vertices = V.copy()
    # algorythm
    while (len(unvisited_vertices)) > 0:
        help_dist = {v: new_distance[v] for v in unvisited_vertices}
        i = min(help_dist, key=help_dist.get)  # node with minimal distance as current node
        del unvisited_vertices[i]  # delete node from unvisited nodes
        for j in V:
            if (i, j) in E:  # all neighbors
                if j in unvisited_vertices:
                    alt_dist = new_distance[i] + _distance[i][j]
                    if alt_dist < new_distance[j]:
                        new_distance[j] = round(alt_dist, 2)
                        predecessor[j] = i
    return new_distance, predecessor


def calc_angle(vector1, vector2):
    # vector needs to be a np.array
    vector1_norm = vector1 / np.linalg.norm(vector1)  # normalizes vector
    vector2_norm = vector2 / np.linalg.norm(vector2)
    # np.dot: dot product of two vectors; np.clip: minimal -1, maximal, 1
    return 180 / np.pi * np.arccos(np.clip(np.dot(vector1_norm, vector2_norm), -1.0, 1.0))


def calc_driving_time(_distance):
    # calculates the driving time from node i to node j in time steps
    driving_time = {i: {j: math.ceil(_distance[i][j] / Velocity / Len_time_step)
                        for j in _distance[i]} for i in _distance}
    return driving_time


def find_exact_path(_start, _end, _predecessor):
    _path = [_end]
    while _end != _start:
        _end = _predecessor[_start][_end]
        _path.insert(0, _end)
    return _path


def find_start_index_of_arc(route, start, end):
    ind = R[route].exact_tour.index(start)
    while end is not R[route].exact_tour[ind + 1]:
        ind = R[route].exact_tour.index(start, ind + 1)
    return ind


def include_path_into_tour(_route, _start, _end):
    # check if any arc is not yet in exact_tour
    # customers can only be visited once by a tour! otherwise we would have a problem here
    if _start not in _route.exact_tour and _end not in _route.exact_tour:
        return f'ERROR: arc {_start, _end} is not on tour {_route.ID}!'
    elif _start in _route.exact_tour and _end in _route.exact_tour:
        ind_start = _route.exact_tour.index(_start)
        ind_end = _route.exact_tour.index(_end)
        del _route.exact_tour[ind_start + 1, ind_end]  # remove all nodes that are in between start and end
        ind_end = ind_start + 1
    elif _start not in _route.exact_tour and _end in _route.exact_tour:
        ind_start = _route.exact_tour.index(_end)  # index of start is the current index of end
        # because end is shifted once to the right
        _route.exact_tour.insert(ind_start, _start)
        ind_end = ind_start + 1  # probably faster than finding the index again
    elif _start in _route.exact_tour and _end not in _route.exact_tour:
        ind_start = _route.exact_tour.index(_start)
        ind_end = ind_start + 1
        _route.exact_tour.insert(ind_end, _end)
    else:
        return f'no idea how i reached this point'

    # now all three cases have the same situation: start and end are next to each other
    path = Path_fwa[_start][_end][1:-1]  # shortest path excluded the start and end
    new_tour = _route.exact_tour
    for n in path:
        new_tour.insert(ind_end, n)
    return new_tour


def create_platoons():
    for (i, j) in E:  # reset possible platoons
        Platoon_arcs[i][j] = []
    # find possible platoons
    for r in R.values():
        for i in range(len(r.exact_tour) - 1):
            Platoon_arcs[r.exact_tour[i]][r.exact_tour[i + 1]].append(r.ID)

    for r in R.values():
        r.platoon = [0] * (len(r.exact_tour) - 1)
        r.update_times()

    for start, end_val in Platoon_arcs.items():
        for end, route_ids in end_val.items():
            # for (start,end), route_ids in platoon_arcs.items():
            if len(route_ids) > 1:
                start_index = {r: find_start_index_of_arc(r, start, end) for r in route_ids}
                leaving_times = {r: R[r].leaving_times[start_index[r]] for r in route_ids}
                latest_route = max(leaving_times, key=leaving_times.get)
                for other_route in route_ids:
                    if other_route is not latest_route:
                        time_diff = leaving_times[latest_route] - leaving_times[other_route]
                        if time_diff >= 0:  # if time difference is equal 0, both tours can already form a platoon
                            R[latest_route].platoon[start_index[latest_route]] = 1
                            R[other_route].platoon[start_index[other_route]] = 1
                            if time_diff > 0:
                                # if time diff is greater 0, the leaving times of the earlier tour have to be adjusted
                                # rest_route = R[other_route].exact_tour[start_index[other_route]:]
                                for index_later_nodes in \
                                        range(len(R[other_route].exact_tour[start_index[other_route]:])):
                                    # iterate over the nodes, that are later than the start node of the platooning arc
                                    # the first node to be updated is the start node (index_late_nodes = 0)
                                    R[other_route].leaving_times[
                                        start_index[other_route] + index_later_nodes] += time_diff
                                # routes[latest_route].driving_time = routes[latest_route].leaving_times[-1]
                                # routes[other_route].driving_time = routes[other_route].leaving_times[-1]
                        else:
                            print(
                                'Here is a problem: Time difference between the latest route '
                                'and another should be greater or equal 0!')
                del other_route
        del end, route_ids
    del start, end_val


def create_gene_from_tours():
    _gene_code = []
    _depots = []
    for r in R.values():
        for i in r.main_tour[1:-1]:
            _gene_code.append(i)
            _depots.append(r.main_tour[0])
        del i
    del r
    _gene = Genecode(gene=_gene_code, depot_allocation=_depots)
    return _gene


def distance_ors(start_lon, start_lat, end_lon, end_lat):

    coordinates = ((start_lon, start_lat), (end_lon, end_lat))
    direction_params = {'coordinates': coordinates,  # lon, lat
                        'profile': 'driving-hgv',
                        'format_out': 'geojson',
                        'preference': 'shortest',
                        'geometry': 'true'}
    route = directions(client=client, **direction_params)
    time.sleep(1.6)  # wait for a short time because the server only allows 40 accesses per minute
    distance, duration = route['features'][0]['properties']['summary'].values()
    return route, distance


# --- Definition of Classes --- #


class Coord:

    def __init__(self, lon, lat):
        self.lon = float(lon)  # lon = x
        self.lat = float(lat)  # lat = y

    def __str__(self):
        return f'lon:{self.lon} , lat:{self.lat}'

    def __repr__(self):
        return f'lon:{self.lon} , lat:{self.lat}'


class Route:

    def __init__(self, _id, _tour):
        self.ID = _id
        self.main_tour = _tour
        self.exact_tour = None
        self.index_main_on_exact = None
        self.demand = None
        self.length = None
        self.leaving_times = [0] * len(_tour)
        self.driving_time = None
        self.cost = 0
        self.labor_cost = 0
        self.fuel_cost = 0
        self.platoon = None

    def __str__(self):
        return f'Route {self.ID}: cost: {self.cost}, demand: {self.demand} \n ' \
               f'{self.main_tour} \n' \
               f'{self.exact_tour} \n'

    def __repr__(self):
        return f'Route {self.ID}: cost: {self.cost}, demand: {self.demand} \n ' \
               f'{self.main_tour} \n' \
               f'{self.exact_tour} \n'

    def get_id_as_num(self):
        return int(self.ID.replace('R', ''))

    def add_node(self, _node, _position):
        # position on the main tour
        self.main_tour.insert(_position, _node)  # insert node on main tour
        node_before = self.main_tour[_position - 1]
        node_after = self.main_tour[_position + 1]
        self.demand += V[_node].demand  # increase demand of the route by the demand of the newly added customer
        self.length = self.length - Dist_fwa[node_before][node_after] + Dist_fwa[node_before][_node] \
                      + Dist_fwa[_node][node_after]

        # add path from before to new
        self.exact_tour = include_path_into_tour(self, node_before, _node)
        # add path from new to after
        self.exact_tour = include_path_into_tour(self, _node, node_after)

        # replaced by the function above, but saved just in case. spent too much time on it to just delete it
        # # remove the old path from the node before to the node after the newly added one
        # ind_before = self.index_main_on_exact[_position - 1]  # index of the main node before the newly added one
        # ind_after = self.index_main_on_exact[_position + 1]  # index of the main node after the newly added one
        # del self.exact_tour[ind_before + 1:ind_after]  # del aux nodes from before node to after node
        # ind_insert = ind_before + 1  # index on the main tour, where new nodes of
        # # include new paths, assuming the shortest path is taken
        # path_from_new_to_after = path_fwa[_node, self.main_tour[_position + 1]][1:-1]
        # path_from_new_to_after.reverse()
        # for n in path_from_new_to_after:
        #     self.main_tour.insert(ind_insert, n)
        # self.main_tour.insert(_position, _node)
        # path_from_before_to_new = path_fwa[self.main_tour[_position - 1], _node][1:-1]
        # path_from_before_to_new.reverse()
        # for n in path_from_before_to_new:
        #     self.main_tour.insert(ind_insert, n)

        # self.platoon.insert(_position, 0)
        # ToDo: finish this function,
        # toDo: how can i calculate the leaving times when im already in a platoon?
        #  Recursive function to check the route that

    def remove_node(self, _node):
        ind_node = self.main_tour.index(_node)
        node_before = self.main_tour[ind_node - 1]
        node_after = self.main_tour[ind_node + 1]
        self.main_tour.remove(_node)
        self.demand += V[_node].demand
        self.length = self.length + Dist_fwa[node_before][node_after] - Dist_fwa[node_before][_node] \
                      - Dist_fwa[_node][node_after]
        self.exact_tour = include_path_into_tour(self, node_before, node_after)

    def update_visits(self, drive_time=None):
        # update times first to make sure they are correct
        self.update_times(drive_time)
        # clear all leaving times of this route
        for n in self.main_tour:
            Visits[n][self.ID] = []
        # add all leaving times again
        for n in range(len(self.main_tour)):
            Visits[self.main_tour[n]][self.ID].append(self.leaving_times[n])

    def update_times(self, drive_time_func=None):
        if drive_time_func is None:
            drive_time_func = Drive_time_graph
        # labor cost
        if sum(self.platoon) == 0:  # if there are no platoons, it's easy to calculate the driving and leaving times
            self.driving_time = V[self.exact_tour[0]].service_time  # leaving time of first node (usually depot, so 0)
            self.leaving_times = [0] * len(self.exact_tour)
            for i in range(1, len(self.exact_tour)):
                self.driving_time += drive_time_func[self.exact_tour[i - 1]][self.exact_tour[i]] \
                                     + V[self.exact_tour[i]].service_time / Len_time_step
                self.leaving_times[i] = self.driving_time
        else:
            self.driving_time = self.leaving_times[-1]
        self.labor_cost = Cost_factor_labor * self.driving_time

    def update_demand(self):
        self.demand = 0
        for n in self.main_tour:
            self.demand += V[n].demand

    def update_route(self, distance_func=None, drive_time_func=None):
        if distance_func is None:
            distance_func = Dist_graph
        # fuel cost
        self.length = 0
        for i in range(1, len(self.exact_tour)):
            self.length += round(distance_func[self.exact_tour[i - 1]][self.exact_tour[i]], 2)
        self.fuel_cost = Cost_factor_fuel * self.length
        # demand
        self.demand = 0
        for n in self.exact_tour:
            self.demand += V[n].demand
        # update leaving times
        self.update_times(drive_time_func=drive_time_func)
        # update visits
        # update total cost
        self.cost = self.fuel_cost + self.labor_cost
        self.update_visits()

    def find_exact_tour(self):
        new_path = [self.main_tour[0]]
        for end in range(1, len(self.main_tour)):
            # path = find_exact_path(self.main_tour[start], self.main_tour[start + 1], predecessor)
            path = Path_fwa[self.main_tour[end - 1]][self.main_tour[end]]
            new_path = new_path + path[1:]
        self.exact_tour = new_path
        self.update_index()
        self.update_length()

    def update_length(self, distance_func=None):
        if distance_func is None:
            distance_func = Dist_graph
        # fuel cost
        self.length = 0
        for i in range(1, len(self.exact_tour)):
            self.length += round(distance_func[self.exact_tour[i - 1]][self.exact_tour[i]], 2)
        self.fuel_cost = Cost_factor_fuel * self.length

    def update_index(self):
        self.index_main_on_exact = [None] * len(self.main_tour)
        for n in range(len(self.main_tour)):
            self.index_main_on_exact[n] = self.exact_tour.index(self.main_tour[n])
        self.index_main_on_exact[-1] = len(self.exact_tour) - 1
        # depot is twice in the list, always first index is found, so second time has to be corrected


class Node:

    def __init__(self, node_id, lon, lat, category, q=0, st=0):
        self.ID = node_id
        self.coord = Coord(lon=lon, lat=lat)
        self.category = category
        self.demand = q
        self.service_time = st

    def __str__(self):
        return f'(Node: ID:{self.ID}, type:{self.category}, coord:{self.coord}, demand:{self.demand}, ' \
               f'service time:{self.service_time}) \n'

    def __repr__(self):
        return f'(Node: ID:{self.ID}, type:{self.category}, coord:{self.coord}, demand:{self.demand}, ' \
               f'service time:{self.service_time}) \n'


class Visit:

    def __init__(self, route, node, leave_time, successor=None, predecessor=None, earliest_time=0, latest_time=None):
        if latest_time is None:
            self.latest_time = T_max
        else:
            self.latest_time = latest_time
        self.route = route
        self.node = node
        self.leave_time = leave_time
        self.predecessor = predecessor
        self.successor = successor
        self.earliest_time = earliest_time

    def __str__(self):
        return f'Node {self.node} on route {self.route} leaving at {self.leave_time} to {self.successor}.'

    def __repr__(self):
        return f'Node {self.node} on route {self.route} leaving at {self.leave_time} to {self.successor}.'


class Genecode:

    def __init__(self, gene, depot_allocation, travel_cost=None):
        self.gene = gene
        self.depot_allocation = depot_allocation
        if travel_cost is None:
            self.travel_cost = Dist_fwa
        else:
            self.travel_cost = travel_cost

    def __str__(self):
        return f'Gene: {self.gene} with depots {self.depot_allocation}.'

    def __repr__(self):
        return f'Gene: {self.gene} with depots {self.depot_allocation}.'

    def recreate_tour(self):
        for d in D:
            # create list of all visited customers for each depot
            _gene_depot = []
            for c_ind in range(len(self.gene)):
                if self.depot_allocation[c_ind] == d:
                    _gene_depot.append(self.gene[c_ind])
            del c_ind

            # create network
            _network = {i: dict() for i in [d] + _gene_depot}  # depot is possible starting node

            for c1_ind in range(len(_gene_depot)):
                c1 = _gene_depot[c1_ind]
                # it´s easier to find c with the index than to find the index with c,
                # since c could be twice or more in the list
                _demand = 0
                for c2_ind in range(c1_ind, len(_gene_depot)):
                    c2 = _gene_depot[c2_ind]
                    if _demand + V[c2].demand <= Capacity_truck:
                        _demand += V[c2].demand
                        # calculate the costs of the subtour
                        _cost = 0

                        if c2_ind - c1_ind == 0:
                            # only one node in subtour
                            _cost += self.travel_cost[d][c1]
                            _cost += self.travel_cost[c1][d]

                        for n_ind in range(c1_ind, c2_ind):
                            n = _gene_depot[n_ind]
                            if n == c1:
                                # first node of subtour
                                _cost += self.travel_cost[d][n]

                            if c2_ind - c1_ind > 0:
                                # middle node
                                _cost += self.travel_cost[n][_gene_depot[n_ind + 1]]

                            if n == _gene_depot[c2_ind]:
                                # last node of subtour
                                _cost += self.travel_cost[n][d]
                        del n_ind

                    else:
                        # as soon as the demand is too high once, it will be too high for all the following nodes too
                        break

                    _network[c1][c2] = _cost
                del c2_ind
            del c1_ind
            print(_network)
        del d


# %% ---Importing data --- %% #

wb = openpyxl.load_workbook(file)

# ---Parameters--- #
Max_working_time = wb['Others']['B'][1].value  # max working time
Max_driving_time = wb['Others']['B'][2].value  # max driving time
Fuel_saving_factor = wb['Others']['B'][3].value  # fuel saving factor
Driving_relief_factor = wb['Others']['B'][4].value  # driving relief factor
Cost_factor_labor = wb['Others']['B'][5].value  # labor cost
Service_time = wb['Others']['B'][7].value  # service time at customers in h
Velocity = wb['Others']['B'][8].value  # velocity of trucks
Cost_factor_depreciation = wb['Others']['B'][9].value  # cost of depreciation if truck used
Cost_factor_fuel = wb['Others']['B'][10].value  # fuel cost for trucks
Capacity_truck = wb['Others']['B'][11].value  # capacity of trucks
Len_time_step = wb['Others']['B'][12].value  # length of time steps in h

# simple sets
T_max = wb['Others']['B'][6].value  # number of time steps
T = list(range(T_max))
K_max = wb['Others']['B'][0].value  # max number of trucks
K = {f'K{i}' for i in range(K_max)}

# Customers
C = wb['Customers']['A']
coord_cus = {C[i].value: [wb['Customers']['B'][i].value, wb['Customers']['C'][i].value] for i in range(1, len(C))}
demand = {C[i].value: wb['Customers']['D'][i].value for i in range(1, len(C))}
C = [C[i].value for i in range(1, len(C))]
C = {i: Node(i, coord_cus[i][0], coord_cus[i][1], 'Customer', demand[i], Service_time / Len_time_step) for i in C}
del demand, coord_cus

# aux nodes
N = wb['Nodes']['A']
coord_aux = {N[i].value: [wb['Nodes']['B'][i].value, wb['Nodes']['C'][i].value] for i in range(1, len(N))}
N = [N[i].value for i in range(1, len(N))]
N = {i: Node(i, coord_aux[i][0], coord_aux[i][1], 'Auxiliary') for i in N}
del coord_aux

# depots
D = wb['Depots']['A']
coord_dep = {D[i].value: [wb['Depots']['B'][i].value, wb['Depots']['C'][i].value] for i in range(1, len(D))}
D = [D[i].value for i in range(1, len(D))]
D = {i: Node(i, coord_dep[i][0], coord_dep[i][1], 'Depot') for i in D}
del coord_dep
del wb

# Combined dictionaries
V = dict()
V.update(C)
V.update(N)
V.update(D)
CN = dict()
CN.update(C)
CN.update(N)
CD = dict()
CD.update(C)
CD.update(D)

# coordinates for all vertices
# _coord = dict()
# _coord.update(coord_cus)
# _coord.update(coord_dep)
# _coord.update(coord_aux)
# del coord_dep, coord_cus, coord_aux

# %% Calculate or read edges %% #

if new_network:
    # create set with all edges, not fitting ones are removed later
    E_temp = set([(i, j) for i in N for j in N if j != i])
    distance_temp = {(i, j): pythagoras(V[i].coord, V[j].coord) for (i, j) in E_temp}
    # for each node check all edges to other nodes, from furthest to closest, if they are too similar to other edges
    for i in N:
        dist_to_j = {j: distance_temp[i, j] for j in N if j != i}

        while len(dist_to_j) > 1:
            furthest_node = max(dist_to_j, key=dist_to_j.get)  # node that is the furthest away from current node
            del dist_to_j[furthest_node]  # remove from list of edges to check
            # vector to the furthest remaining node
            vector_furthest = np.array([V[i].coord.lon - V[furthest_node].coord.lon,
                                        V[i].coord.lat - V[furthest_node].coord.lat])
            # angles to all the other nodes
            angles = {j: calc_angle(vector_furthest, np.array([V[i].coord.lon - V[j].coord.lon,
                                                               V[i].coord.lat - V[j].coord.lat])) for j in dist_to_j}

            for j in angles:
                if abs(angles[j]) < 20:  # angle to small
                    E_temp.remove((i, furthest_node))
                    break
                elif abs(angles[j]) < 60:  # angle in a medium size, check if the detour would be too big to keep it
                    short_dist_j_to_all, temp = dijkstra(distance_temp, j)
                    dist_i_j_furthest = distance_temp[i, j] + short_dist_j_to_all[furthest_node]
                    if distance_temp[i, furthest_node] * 1.5 > dist_i_j_furthest:
                        E_temp.remove((i, furthest_node))
                        break

    del angles, dist_i_j_furthest, dist_to_j, vector_furthest, furthest_node, distance_temp

    # create set with final edges
    E = set()
    for (i, j) in E_temp:  # add all the reverse arcs
        E.add((j, i))
        E.add((i, j))
    del E_temp

    # create cvs with the arcs to reduce loading time of network
    with open(file_edges, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(E)
    print('new edges created and written to csv-file-')
    del csvfile, csvwriter

else:
    # load csv with edges
    E = set()
    # opening the CSV file
    with open(file_edges, mode='r') as file_temp:

        # reading the CSV file
        csvFile = csv.reader(file_temp)

        # displaying the contents of the CSV file
        for lines in csvFile:
            if len(lines) > 0:  # for some reason every other line is just empty
                i = lines[0]
                j = lines[1]
                E.add((i, j))
    print('edges read from csv-file.')
    del csvFile, lines, file_temp

del file_edges, file, new_network

# find the closest node for customers and depots and add arcs to and from this node to E
for c in CD:
    dist_cus = {n: pythagoras(V[c].coord, V[n].coord) for n in N}
    closest_node = min(dist_cus, key=dist_cus.get)
    min_dist = dist_cus[closest_node]
    v1 = np.array([V[c].coord.lon - V[closest_node].coord.lon,
                   V[c].coord.lat - V[closest_node].coord.lat])
    n = closest_node
    while (dist_cus[n] < 2 * min_dist):
        v2 = np.array([V[c].coord.lon - V[n].coord.lon,
                       V[c].coord.lat - V[n].coord.lat])
        if calc_angle(vector1=v1, vector2=v2) > 60:
            E.add((c, n))
            E.add((n, c))
        del dist_cus[n]
        n = min(dist_cus, key=dist_cus.get)
del c, dist_cus, n

# change E from set to list
E = list(E)
E.sort()

# %% Distance Calculations % ##

# recalculate all the distances
Dist_graph = {i: {j: pythagoras(V[i].coord, V[j].coord) for j in V if (i, j) in E} for i in V}
for i in V:
    Dist_graph[i][i] = 0
del i
Drive_time_graph = calc_driving_time(Dist_graph)

# find fastest and alternative tours from any customer/depot to all other customer/depot
Detour_factor = (Cost_factor_fuel + Cost_factor_labor / Velocity) / ((1 - Fuel_saving_factor) * Cost_factor_fuel
                                                                     + Cost_factor_labor / Velocity)

# floyd warshall algorithm, incl. the path and also alternatives
Dist_fwa = {i: {j: Dist_graph[i][j] if (i, j) in E else np.inf for j in V} for i in V}
Pred_fwa = {i: {j: i if (i, j) in E else None for j in V} for i in V}
Path_fwa = {i: {j: [i, j] if (i, j) in E else None for j in V} for i in V}
Path_alt = {i: {j: [] for j in V} for i in V}  # list of alternative path

for i in V:
    Dist_fwa[i][i] = 0
    Pred_fwa[i][i] = i
    Path_fwa[i][i] = [i]
del i

for k in V:
    for i in V:
        for j in V:
            if Dist_fwa[i][j] > Dist_fwa[i][k] + Dist_fwa[k][j]:
                Dist_fwa[i][j] = round(Dist_fwa[i][k] + Dist_fwa[k][j], 2)
                Pred_fwa[i][j] = Pred_fwa[k][j]
                Path_fwa[i][j] = Path_fwa[i][k] + Path_fwa[k][j][1:]
            if Dist_fwa[i][j] * Detour_factor > Dist_fwa[i][k] + Dist_fwa[k][j]:
                Path_alt[i][j].append((Path_fwa[i][k] + Path_fwa[k][j][1:], Dist_fwa[i][k] + Dist_fwa[k][j]))
                # store in a tuple to easier access the distance of the path
        del j
    del i
del k
# remove all alternative paths, that are now too long by creating a new list
for i in V:
    for j in V:
        Path_alt[i][j] = [alt_path for alt_path in Path_alt[i][j]
                          if alt_path[1] < Dist_fwa[i][j] * Detour_factor]
del i, j
# visits as dict
# created here, so it is usable in the main file as well as here
# each route visited adds a new element to the inner dict
# with the route as key and a list of the leaving times of this node as value

Visits = {v: dict() for v in V}
Platoon_arcs = {i: {j: [] for j in Dist_graph[i]} for i in Dist_graph}
R = dict()
Visits_dict = {r_ind: [] for r_ind in R}
R_at_depot = {d: [] for d in D}
