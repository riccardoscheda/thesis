"""
===========
Slider Demo
===========

Using the slider widget to control visual properties of your plot.

In this example, a slider is used to choose the frequency of a sine
wave. You can control many continuously-varying properties of your plot in
this way.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import networkx as nx
import random_network as rn
###########################################################################################################
N = 5
K = 2
number_of_clusters = 3


#creation of the subnetworks
gr = [rn.Random_Network(N,K) for i in range(number_of_clusters)]

single_cluster_control_nodes = [rn.find_control_nodes(gr[i],N) for i in range(number_of_clusters)]
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

##########################################################################################################
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

f0 = 1
delta_f = 1

g = nx.random_k_out_graph(n=f0,k=2,alpha=0.5)

axcolor = 'lightgray'
axfreq = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'noise', 1, 20, valinit=f0, valstep=delta_f,valfmt='%0.0f')

active_nodes = []
non_active_nodes = []

for i in range(number_of_clusters):
    Net.nodes[control_nodes[i]] = 1
for i in range(len(Net.nodes)):
    if Net.nodes[i] == 1 :
        active_nodes.append(i)
    else:
        non_active_nodes.append(i)

def update(val):
    plt.cla()
    freq = sfreq.val

    rn.evolution(Net,iterations=10)
    for i in range(len(Net.nodes)):
        if Net.nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
    
    nx.draw_networkx(graph,npos, with_labels= True)
    nx.draw_networkx_nodes(graph,npos,
                        nodelist=control_nodes,
                        node_size=800)
    nx.draw_networkx_nodes(graph,npos,
                        nodelist=active_nodes,
                        node_color='y')
    nx.draw_networkx_edges(graph, npos,
                       edgelist=negedges,
                       width=3, alpha=0.4, edge_color='r')
    nx.draw_networkx(graph,with_labels=True)
    fig.canvas.draw_idle()


sfreq.on_changed(update)
resetax = plt.axes([0.15, 0.15, 0.8, 0.8])
button = Button(resetax, '', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    plt.cla()
button.on_clicked(reset)



plt.show()
