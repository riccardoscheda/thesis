import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import choice
from functools import reduce
import matplotlib.animation as animation # animation plot
import pandas as pd
import networkx as nx
import  random_network  as rn

fig, ax = plt.subplots(1,1)

#iterations
frames = 500

N = 10
K = 2
number_of_clusters = 1

######################################

#creation of the subnetworks
gr = [rn.Random_Network(N,K) for i in range(number_of_clusters)]

single_cluster_control_nodes = [rn.outgoing_links(gr[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(gr, control_nodes, N,number_of_clusters)
negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))

Net = rn.Network(tot)
graph = nx.from_numpy_matrix(tot.T, create_using=nx.DiGraph)
npos = nx.spring_layout(graph)
cycles = nx.cycle_basis(graph.to_undirected())

################## ONLY FOR VISUALIZATION #######################
abs_tot = abs(tot)
graph1 = nx.from_numpy_matrix(abs_tot.T, create_using=nx.DiGraph)
npos = nx.kamada_kawai_layout(graph1)
#################################################################




def init():
    ax = nx.draw(graph,pos = npos)
    active_nodes = []
    non_active_nodes = []
    for i in range(number_of_clusters):
        Net.nodes[control_nodes[i]] = 1
    
    for i in range(N):
        Net.nodes[i] = 1
        
        
    for i in range(len(Net.nodes)):
        if Net.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
    ax = nx.draw_networkx(graph,npos, with_labels= False)
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')

    return ax,


def evo(frames):
    plt.ion()
    plt.cla()
    
    up = Net.adj_matrix.dot(Net.nodes)
    Net.nodes = (up >0).astype(int)
    rn.noise(Net,p=0.2)
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
    plt.title("frame " +str(frames))
    return  ax

print(rn.outgoing_links(gr[0], N))
print(rn.outgoing_links(Net, N))
ani = FuncAnimation(fig, evo, frames = np.arange(0,500), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")


#%% 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import choice
from functools import reduce
import matplotlib.animation as animation # animation plot
import pandas as pd
import networkx as nx
import  random_network  as rn

fig, ax = plt.subplots(1,1)

#iterations
frames = 500

N = 10
K = 2
number_of_clusters = 2

######################################

#creation of the subnetworks
gr = [rn.Random_Network(N,K)]
gr.append(rn.Random_Network(1, 1))
gr[1].adj_matrix
single_cluster_control_nodes = [rn.outgoing_links(gr[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(gr, control_nodes, N,number_of_clusters)
negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))

Net = rn.Network(tot)
graph = nx.from_numpy_matrix(tot.T, create_using=nx.DiGraph)
npos = nx.spring_layout(graph)
cycles = nx.cycle_basis(graph.to_undirected())

################## ONLY FOR VISUALIZATION #######################
abs_tot = abs(tot)
graph1 = nx.from_numpy_matrix(abs_tot.T, create_using=nx.DiGraph)
npos = nx.kamada_kawai_layout(graph1)
#################################################################




def init():
    ax = nx.draw(graph,pos = npos)
    active_nodes = []
    non_active_nodes = []
    for i in range(number_of_clusters):
        Net.nodes[control_nodes[i]] = 1
    
    for i in range(N):
        Net.nodes[i] = 1
        
        
    for i in range(len(Net.nodes)):
        if Net.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
    ax = nx.draw_networkx(graph,npos, with_labels= False)
    ax = nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')

    return ax,


def evo(frames):
    plt.ion()
    plt.cla()
    
    up = Net.adj_matrix.dot(Net.nodes)
    Net.nodes = (up >0).astype(int)
    rn.noise(Net,p=0.2)
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
    plt.title("frame " +str(frames))
    return  ax

print(rn.outgoing_links(gr[0], N))
print(rn.outgoing_links(Net, N))
ani = FuncAnimation(fig, evo, frames = np.arange(0,500), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")

