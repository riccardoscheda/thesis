import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn

N = 20
K = 1

number_of_active_nodes = []
g = rn.Random_Network(N, K)
A = g.adj_matrix


#plt.imshow(A)


def evolution(time):
    
    for t in range(time):
        up = np.zeros((N,1))
        for i in range(N):
            somma = 0 
            for j in range(N):
                somma = somma + g.adj_matrix[i][j]*g.nodes[j]
            up[i] = somma
            
        for i in range(N):
            if up[i] != 0:
                g.nodes[i] = 1
            else:
                g.nodes[i] = 0 

def shuffle():
    for i in range(N):
        for j in range(N):
            g.nodes[j] = 0
        g.nodes[i] = 1
        evolution(20)
        number_of_active_nodes.append(np.sum(g.nodes))

shuffle()
print(number_of_active_nodes)

graph = nx.from_numpy_matrix(A.T,create_using = nx.DiGraph)
pos = nx.layout.spring_layout(graph)
plt.figure() 
plt.plot(number_of_active_nodes, label = "number of active nodes")
plt.legend()
#nx.draw_networkx(graph, pos = pos, with_labels= True)

