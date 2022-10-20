import gurobipy
import openpyxl
import time

#import file
file = 'Test_Daten.xlsx'
wb = openpyxl.load_workbook(file)



# definition of sets and parameters
customer = wb['Customers']['A']
C = [customer[i].value for i in range(1, len(customer))]  # customers
nodes = wb['Nodes']['A']
N = [nodes[i].value for i in range(1, len(nodes))]  # auxiliary nodes
depots = wb['Depots']['A']
D = [depots[i].value for i in range(1, len(depots))]  # depots
V = C + N + D  # all vertices
trucks = wb['Trucks']['A']
K = [trucks[i].value for i in range(1, len(trucks))]  # trucks
products = wb['Products']['A']
W = [products[i].value for i in range(1, len(products))]  # products
sizes = wb['Sizes']['A']
S = [sizes[i].value for i in range(1, len(sizes))]  # platoon sizes
A = [(i, j) for i in D for j in C+N]

del customer, nodes, depots, trucks, products, sizes

# definition of parameters



# definition of variables
