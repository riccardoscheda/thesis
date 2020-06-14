import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn

#N = 20
K = 2



#plt.imshow(A)


def evolution(N,time):
    
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
        

def shuffle(N):
    for i in range(N):
        for j in range(N):
            g.nodes[j] = 0
        g.nodes[0] = 1
    
        evolution(i,i)
        number_of_active_nodes.append(np.mean(g.nodes))
        
    means.append(np.mean(number_of_active_nodes))
        

#print(number_of_active_nodes)

# graph = nx.from_numpy_matrix(A.T,create_using = nx.DiGraph)
# pos = nx.layout.spring_layout(graph)


means = []
for i in range(2,30):
    
    g = rn.Random_Network(i, K)
    #A = g.adj_matrix

    number_of_active_nodes = []
    shuffle(i)
    
plt.figure()
plt.plot(means)
plt.figure()
graph = nx.from_numpy_matrix(g.adj_matrix,create_using = nx.DiGraph)
active_nodes = []
        
for i in range(len(g.nodes)):
    if g.nodes[i] == 1 :
        active_nodes.append(i)
        
pos = nx.layout.spring_layout(graph)
nx.draw_networkx(graph,pos = pos,with_labels=True)
#nx.draw_networkx(graph, pos = pos, with_labels= True)

nx.draw_networkx_nodes(g,pos,
                        nodelist=active_nodes,
                        node_color='y')


K = 2
n = 30
means = []
for N in range(1,n):
    g = rn.Random_Network(N, K)
    g.nodes[0] = 1
    #evolution
    for t in range(N):
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
    means.append(np.mean(g.nodes.T))
plt.plot(means)
