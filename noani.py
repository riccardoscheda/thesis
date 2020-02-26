


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd

import networkx as nx

#iterations
frames = 100
#particles
n = 100
#time step

###################################

#matrix =  np.random.randint(2, size=(n,n))
# matrix = np.array([[0,1,0,0,0],
#           [0,0,1,0,0],
#           [1,0,0,0,0],
#           [0,1,0,0,0],
#           [0,0,1,0,0]])


nodes = np.zeros((n,1))
up = np.zeros((n,1))
#initial state
# nodes[0] = 1
#nodes[-3] = 1
#nodes = np.random.randint(2,size= (n,1))

labels = {}
g = nx.Graph()
pos = nx.layout.spring_layout(g)
time = []

def decimal(u):
    b = '\n'.join(''.join('%d' %x for x in y) for y in u)
    return int(b,2)
    
def random_adjancency_matrix(n):
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][np.random.randint(n)] = 1

    return matrix

matrix = random_adjancency_matrix(n).T
print(matrix)

g1 = nx.from_numpy_matrix(matrix)



initial_states = []

def evo(nodes,t):
    
    up = np.zeros((n,1))
    up = matrix.dot(nodes)
    previous_state = decimal(nodes.T)
    if t == 1:
        initial_states.append(previous_state)
        
    for i in range(n):
        nodes[i] = up[i]
    state = decimal(nodes.T)
    g.add_edge(previous_state,state)
    labels[state] = state
    
def shuffle(nodes):
    for i in range(n):
        nodes = np.zeros((n,1))
        nodes[i] = 1
        for t in range(100):
            evo(nodes,t)
            
            
shuffle(nodes) 
       

pos = nx.layout.kamada_kawai_layout(g)

nx.draw(g,pos = pos)
nx.draw_networkx_nodes(g,pos,
                       nodelist=initial_states,node_color='r')
#nx.draw(g1)
nx.draw_networkx_labels(g,pos,labels,font_size=1)

