
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
N = 30
K = 2
#time step
dt = 0.1
###################################
fig, ax = plt.subplots(1)

labels = {}
  


g = rn.Random_Network(N,K)
graph = nx.from_numpy_matrix(g.adj_matrix,create_using= nx.DiGraph)
npos = nx.layout.spring_layout(graph)

 
h = rn.Random_Network(N, K)
hgraph = nx.from_numpy_matrix(h.adj_matrix,create_using= nx.DiGraph)
hpos = nx.layout.spring_layout(hgraph)

for el in hpos:
    hpos[el][0] = hpos[el][0] + 5
    
  
#g = nx.DiGraph(g)



def init():


    ax = nx.draw(graph,pos = npos)
    active_nodes = []
    non_active_nodes = []
    
    g.nodes[0] = 1
    for i in range(len(g.nodes)):
        if g.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
           
    ax = nx.draw_networkx(graph,npos, with_labels= True)
    
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
    ax = nx.draw_networkx(hgraph,hpos, with_labels= True)
    
    return ax,



def evo(frames):
    plt.ion()
    plt.cla()
    
    
 
    up = g.adj_matrix.dot(g.nodes)
    g.nodes = (up >0).astype(int)
       
    ax = nx.draw(graph,pos = npos)

    active_nodes = []
    non_active_nodes = []
    
    for i in range(len(g.nodes)):
        if g.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
    ax = nx.draw_networkx(graph,npos, with_labels= True)
    
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
    ax = nx.draw_networkx(hgraph,hpos, with_labels= True)
    
    
    return  ax, 


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")
