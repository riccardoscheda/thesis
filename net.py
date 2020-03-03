
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd

import networkx as nx

#iterations
frames = 100
#particles
n = 6
#time step
dt = 0.1
###################################
fig, ax = plt.subplots(2,2)

#matrix =  np.random.randint(2, size=(n,n))
matrix = np.array([[0,1,0,0,0,0],
          [0,0,1,0,0,0],
          [0,0,0,1,0,0],
          [0,0,0,0,1,0],
          [0,0,1,0,0,0],
          [0,0,0,0,0,0]])


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
    #return int(b,2)
    return b
    
def random_adjancency_matrix(n):
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][np.random.randint(n)] = 1

    return matrix

# matrix = random_adjancency_matrix(n).T
# print(matrix)

#g1 = nx.from_numpy_matrix(matrix)



def init():

    
    ax[0,0].imshow(nodes.T)
    #ax[1,0] = nx.draw(g)
    ax[0,1].imshow(matrix.T)
    ax[1,0] = nx.draw(g)
    
    state = decimal(nodes.T)
    labels[state] = state
    ax[1,0] = nx.draw_networkx_labels(g,labels,font_size=5)
    return ax[0,0],


def change():
    nodes = np.zeros((n,1))
    up = np.zeros((n,1))


def evo(frames):
    plt.ion()
    plt.cla()
    
    
    
    up = np.zeros((n,1))
    #up = matrix.dot(nodes)
    previous_state = decimal(nodes.T)
    
    for i in range(n):
        somma = 0 
        for j in range(n):
            somma = somma + matrix[i][j]*nodes[j]
        up[i] = somma
        
    for i in range(n):
        if up[i] != 0:
            nodes[i] = 1
        else:
            nodes[i] = 0 
    # for i in range(n):
    #     somma = 0
    #     for j in range(n):
    #         somma  = somma + matrix[i][j] * nodes[i]
    #    # up[i]= somma
    

    time.append(1)
    ### changin initial conditions after a time t
    if len(time)>10:
        nodes[-1] = 1
        matrix[0][5] = 0
        
    state = decimal(nodes.T)
    g.add_edge(previous_state,state)

    ax[0,0].imshow(nodes.T)
    npos = nx.layout.shell_layout(g)
    
    ax[1,0] = nx.draw(g,pos = npos)
    ax[1,0] = nx.draw_networkx_nodes(g,npos,
                       nodelist=[state],
                       node_color='g')
    ax[1,0] = nx.draw_networkx_nodes(g,npos,
                       nodelist=[previous_state],
                       node_color='y')
    labels[state] = state
    ax[1,0] = nx.draw_networkx_labels(g,npos,labels,font_size=5)
    ax[0,1].imshow(matrix.T)
    
    return  ax[0,0], ax[1,0] ,ax[1,1], ax[0,1]


ani = FuncAnimation(fig, evo, frames = np.arange(0,1000), interval = 200,init_func = init, blit = False)

