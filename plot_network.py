import Classes
import matplotlib.pyplot as plt

# import file
file = 'Test_Daten_2.xlsx'

W, D, C, N, V, CN, K, S, A, c_d, Q, c_f, v, d, a, q, K_max, Tw_max, Td_max, n_f, n_d, c_l, t_s, coord, sets, parameter= Classes.read_data(file)

print(coord)
# plot network
for i in D:
    plt.scatter(coord[i][0], coord[i][1], c='b')
    plt.text(coord[i][0], coord[i][1], i, c='b')

for i in C:
    plt.scatter(coord[i][0], coord[i][1], c='g')
    plt.text(coord[i][0], coord[i][1], i, c='g')

for i in N:
    plt.scatter(coord[i][0], coord[i][1], c='r')
    plt.text(coord[i][0], coord[i][1], i, c='r')

for (i, j) in A:
    plt.plot([coord[i][0], coord[j][0]], [coord[i][1], coord[j][1]], 'y')

plt.show()
