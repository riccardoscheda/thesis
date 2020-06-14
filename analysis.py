import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn


n = 40

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
                #NOISE
                # for i in range(N):
                #  p = 1
                #  if np.random.uniform(0,1)>p:
                #      g.nodes[i] = 1
             
                
                up = g.adj_matrix.dot(g.nodes)
                g.nodes = (up >0).astype(int)
                   
            means.append(np.mean(g.nodes.T))
            #g.nodes = np.zeros((N,1))
        totmeans.append(np.mean(means))
       
    
    plt.ylim(0,1.2)
    plt.plot(totmeans)
    

# plt.figure()
# graph = nx.from_numpy_matrix(g.adj_matrix,create_using = nx.DiGraph)
# active_nodes = []
        
# for i in range(len(g.nodes)):
#     if g.nodes[i] == 1 :
#         active_nodes.append(i)
        
# pos = nx.layout.spring_layout(graph)
# nx.draw_networkx(graph,pos = pos,with_labels=True)
# #nx.draw_networkx(graph, pos = pos, with_labels= True)

# nx.draw_networkx_nodes(g,pos,
#                         nodelist=active_nodes,
#                         node_color='y')


