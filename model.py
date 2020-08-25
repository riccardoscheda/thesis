
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import choice
from functools import reduce

import matplotlib.animation as animation # animation plot
import pandas as pd
import networkx as nx

import  random_network  as rn

#iterations
frames = 100

N = 5
M = 5
K = 2
num = 2

######################################
fig, ax = plt.subplots(1,1)

labels = {}
  

gr = [rn.Random_Network(N,K) for i in range(num)]

def find_control_nodes(gr):
    graph = nx.from_numpy_matrix(gr.adj_matrix.T, create_using=nx.DiGraph)
    npos = nx.layout.spring_layout(graph)
    cycles = nx.cycle_basis(graph.to_undirected())
    print("cycles: " + str(cycles))
    driver_node = list(reduce(lambda x,y: set(x)&set(y),cycles))
    
    final = []
    z = list(reduce(lambda x,y: x+y,cycles))
    print("driver node: "+ str(driver_node))
    for i in range(N):
        final.append(z.count(i))
         
    control_node = np.argmax(final)
    print(control_node)
    return control_node
    

single_cluster_control_nodes = [find_control_nodes(gr[i]) for i in range(num)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(num)]
tot = gr[0].adj_matrix
negedges = []
################### AGGIUNGE I LINK NEGATIVI TRA DUE CLUSTERS ###################
if num>1:
    for i in range(num-1):
        neg1 = np.zeros((N*(i+1),N))
        neg2 = np.zeros((N,N*(i+1)))
        
        tot = np.block([[tot,       neg1        ],
                        [neg2, gr[i].adj_matrix              ]])
    

for j in range(num):
    
    #tot[control_nodes[num-1-j]][choice([i for i in range(N*j,N*(j+1)) if i not in control_nodes])] = -1
     tot[control_nodes[-j]][control_nodes[-j-1]] = -1
############################################################################

negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))

#print(tot)
print("outgoing links: " + str(sum(tot)))

Net = rn.Network(tot)
graph = nx.from_numpy_matrix(tot.T, create_using=nx.DiGraph)
npos = nx.layout.spring_layout(graph)
cycles = nx.cycle_basis(graph.to_undirected())
print("cycles: " + str(cycles))
driver_node = list(reduce(lambda x,y: set(x)&set(y),cycles))

final = []
z = list(reduce(lambda x,y: x+y,cycles))
print("driver node: "+ str(driver_node))
for i in range(N):
    final.append(z.count(i))
     
control_node = np.argmax(final)
print(control_node)


def init():


    ax = nx.draw(graph,pos = npos)
    active_nodes = []
    non_active_nodes = []
    
    for i in range(num):
        Net.nodes[control_nodes[i]] = 1
    
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
                        nodelist=control_nodes,
                        node_size=800)
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
  
    ax = nx.draw_networkx_edges(graph, npos,
                       edgelist=negedges,
                       width=3, alpha=0.4, edge_color='r')
    
    #print(Net.activity())
    #plt.imshow(tot)
    return  ax


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")

np.savetxt("network.txt", tot, fmt="%d",delimiter=" ")
