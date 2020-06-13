import pylab as plt
import networkx as nx
import  random_network  as rn

g = rn.Random_Network(10, 2)
A = g.adj_matrix

plt.imshow(A)