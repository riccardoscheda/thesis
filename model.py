
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd
import networkx as nx

import  random_network  as rn

#iterations
frames = 100
#particles
N = 5
M = 5
K = 2
#time step
dt = 0.1
###################################
fig, ax = plt.subplots(1)

labels = {}
  


g = rn.Random_Network(N,K)
h = rn.Random_Network(M, K)

#hinibitory links
neg1 = np.zeros((N, M))
neg1[np.random.randint(N)][np.random.randint(M)] = -10
neg2 = np.zeros((M, N))
neg2[np.random.randint(M)][np.random.randint(N)] = 0


tot = np.block([[g.adj_matrix,       neg1        ],
    [neg2, h.adj_matrix              ]])

#plt.imshow(tot)
Net = rn.Network(tot)
graph = nx.from_numpy_matrix(tot, create_using=nx.DiGraph)
npos = nx.layout.spring_layout(graph)

def init():


    ax = nx.draw(graph,pos = npos)
    active_nodes = []
    non_active_nodes = []
    
    Net.nodes[0] = 1
    Net.nodes[-1] = 1
    
    for i in range(len(Net.nodes)):
        if Net.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
           
    ax = nx.draw_networkx(graph,npos, with_labels= True)
    
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
    
    return ax,



def evo(frames):
    plt.ion()
    plt.cla()
    
    
 
    up = Net.adj_matrix.dot(Net.nodes)
    Net.nodes = (up >0).astype(int)
       
    ax = nx.draw(graph,pos = npos)

    active_nodes = []
    non_active_nodes = []
    
    for i in range(len(Net.nodes)):
        if Net.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
    ax = nx.draw_networkx(graph,npos, with_labels= True)
    
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
    
    return  ax, 


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")
