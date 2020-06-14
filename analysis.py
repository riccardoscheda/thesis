import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn


n = 30

for K in range(1,3):
    means = []
    totmeans = []
    for N in range(1,n):
        
        for i in range(30):
            g = rn.Random_Network(N, K)
            g.nodes[np.random.randint(N)] = 1
            
            #evolution
            for t in range(N):
                up = np.zeros((N,1))
                for k in range(N):
                    somma = 0 
                    for j in range(N):
                        somma = somma + g.adj_matrix[k][j]*g.nodes[j]
                    up[k] = somma
                    
                for k in range(N):
                    if up[k] != 0:
                        g.nodes[k] = 1
                    else:
                        g.nodes[k] = 0 
                        
            means.append(np.mean(g.nodes.T))
            #g.nodes = np.zeros((N,1))
        totmeans.append(np.mean(means))
       
    
    plt.ylim(0,1.2)
    plt.plot(totmeans)
    
    
    
    
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

