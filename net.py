
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd

import networkx as nx

#iterations
frames = 100
#particles
n = 5
#time step
dt = 0.1
###################################
fig, ax = plt.subplots(2,2)

#matrix =  np.random.randint(2, size=(n,n))
matrix = np.array([[0,1,0,0,0],
          [0,0,1,0,0],
          [0,0,0,1,0],
          [0,0,0,0,1],
          [0,0,1,0,0]])


matrix = matrix.T
nodes = np.zeros((n,1))
up = np.zeros((n,1))
#initial state
# nodes[0] = 1
nodes[0] = 1
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

# matrix = random_adjancency_matrix(n).T
# print(matrix)

g1 = nx.from_numpy_matrix(matrix)



def init():

    
    ax[0,0].imshow(nodes.T)
    ax[1,0] = nx.draw(g)
    ax[0,1].imshow(matrix.T)
    
    return ax[0,0],


def change():
    nodes = np.zeros((n,1))
    up = np.zeros((n,1))


def evo(frames):
    plt.ion()
    plt.cla()
    
    
    
    up = np.zeros((n,1))
    up = matrix.dot(nodes)
    previous_state = decimal(nodes.T)
    
    for i in range(n):
        nodes[i] = up[i]
    state = decimal(nodes.T)
    g.add_edge(previous_state,state)

    # for i in range(n):
    #     somma = 0
    #     for j in range(n):
    #         somma  = somma + matrix[i][j] * nodes[i]
    #    # up[i]= somma
    
    
    

    ax[0,0].imshow(nodes.T)
    pos = nx.layout.circular_layout(g)
    ax[1,0] = nx.draw(g,pos = pos)
    ax[1,0] = nx.draw_networkx_nodes(g,pos,
                       nodelist=[state],
                       node_color='g')
    ax[1,0] = nx.draw_networkx_nodes(g,pos,
                       nodelist=[previous_state],
                       node_color='y')
    labels[state] = state
    ax[1,0] = nx.draw_networkx_labels(g,pos,labels,font_size=5)

    
    ### changin initial conditions after a time t
      
    return  ax[0,0], ax[1,0] ,ax[1,1], [0,1]


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)

