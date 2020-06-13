
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd

import networkx as nx

#iterations
frames = 100
#particles
n = 10
#time step
dt = 0.1
###################################
fig, ax = plt.subplots(2,2)

matrix =  np.random.randint(2, size=(n,n))

matrix = matrix.T
nodes = np.zeros((n,1))
up = np.zeros((n,1))
#initial state
# nodes[0] = 1
nodes[0] = 1
#nodes = np.random.randint(2,size= (n,1))

labels = {}

time = []

def decimal(u):
    b = '\n'.join(''.join('%d' %x for x in y) for y in u)
    #return int(b,2)
    return b
    
def random_adjancency_matrix(n):
    matrix = np.zeros((n,n))
    for i in range(n):
        matrix[i][np.random.randint(n//2)] = 1
        #matrix[n//2-i][np.random.randint(n//2)] = 1
        

    return matrix

matrix = random_adjancency_matrix(n).T
g = nx.from_numpy_matrix(matrix,create_using= nx.DiGraph)
npos = nx.layout.spring_layout(g)
    
#g = nx.DiGraph(g)



def init():

    
    #ax[0,0].imshow(nodes.T)

    ax[0,1].imshow(matrix.T)

    ax[1,0] = nx.draw(g,pos = npos)
    active_nodes = []
    non_active_nodes = []
    
    for i in range(len(nodes)):
        if nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
        
    ax[1,0] = nx.draw_networkx_nodes(g,npos,
                        nodelist=active_nodes,
                        node_color='y')
    
    ax[1,0] = nx.draw_networkx_nodes(g,npos,
                        nodelist=non_active_nodes,
                        node_color='black')
    
    return ax[1,0],


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

    for i in range(n):
            p = 0.5
            if np.random.uniform(0,1)>p:
                nodes[i] = 1
        
    #ax[0,0].imshow(nodes.T)
    ax[1,0] = nx.draw(g,pos = npos)

    active_nodes = []
    non_active_nodes = []
    
    for i in range(len(nodes)):
        if nodes[i] == 1 :
            active_nodes.append(i)
        else:
            non_active_nodes.append(i)
        
        
    ax[1,0] = nx.draw_networkx_nodes(g,npos,
                        nodelist=active_nodes,
                        node_color='y')
    
    ax[1,0] = nx.draw_networkx_nodes(g,npos,
                        nodelist=non_active_nodes,
                        node_color='black')

    
    ax[0,1].imshow(matrix)
    
    return  ax[0,0], ax[1,0] ,ax[1,1], ax[0,1]


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)
#ani.save('network.gif',dpi = 100,writer = "imagemagick")

