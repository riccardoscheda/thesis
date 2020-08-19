
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functools import reduce

import matplotlib.animation as animation # animation plot
import pandas as pd
import networkx as nx

import  random_network  as rn

#iterations
frames = 100
#particles
N = 10
M = 5
K = 2
#time step
dt = 0.1
###################################
fig, ax = plt.subplots(1,1)

labels = {}
  
num = 1
gr = [rn.Random_Network(N,K) for i in range(num)]

tot = gr[0].adj_matrix
negedges = []
if num>1:
    for i in range(num-1):
        neg1 = np.zeros((N*(i+1),N))
        a = [np.random.randint(N*(i+1))]
        a.append(np.random.randint(N))
        neg1[a[0]][a[1]] = -1
        neg2 = np.zeros((N,N*(i+1)))
        c = [np.random.randint(N*(i+1))]
        c.append(np.random.randint(N))
        
        neg2[c[1]][c[0]] = -1

        #print(negedges)
        
        tot = np.block([[tot,       neg1        ],
                        [neg2, gr[i].adj_matrix              ]])
    

negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))

#print(tot)
print("outgoing links: " + str(sum(tot)))

Net = rn.Network(tot)
graph = nx.from_numpy_matrix(tot.T, create_using=nx.DiGraph)
npos = nx.layout.spring_layout(graph)
cycles = nx.cycle_basis(graph.to_undirected())
print("cycles: " + str(cycles))
driver_node = list(reduce(lambda x,y: set(x)&set(y),cycles))

z = list(reduce(lambda x,y: x+y,cycles))
print("driver node: "+ str(driver_node))
for i in range(N):
    print("node " + str(i), z.count(i))
def init():


    ax = nx.draw(graph,pos = npos)
    active_nodes = []
    non_active_nodes = []
    
    for i in range(num):
        if len(driver_node)>0:
            Net.nodes[driver_node[0]] = 1
        else:
            Net.nodes[i*M] = 1
    
    
    for i in range(len(Net.nodes)):
        if Net.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
           
    ax = nx.draw_networkx(graph,npos, with_labels= False)
    
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
    
    #plt.imshow(tot)
    return ax,


mean_activity = []


def noise(node):
    ##### ATTENZIONE NOISE A ZERO #####
    p = 0.8
    if np.random.uniform(0,1)>p:
        #print("ok")
        return 0
    else: 
        return node
    
    
def evo(frames):
    plt.ion()
    plt.cla()
    
    
 
    up = Net.adj_matrix.dot(Net.nodes)
    Net.nodes = (up >0).astype(int)
       
    for i in range(len(Net.nodes)):
       Net.nodes[i] = noise(Net.nodes[i])
        
        
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
    ax = nx.draw_networkx_edges(graph, npos,
                       edgelist=negedges,
                       width=3, alpha=0.4, edge_color='r')
    #mean_activity.append(np.mean(Net.nodes))
    #print(mean_activity)
    return  ax


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")
