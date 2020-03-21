


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
g = nx.DiGraph()
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
                
    matrix[np.random.randint(n)][np.random.randint(n)] = 1
    matrix[np.random.randint(n)][np.random.randint(n)] = 1
    
    return matrix

matrix = random_adjancency_matrix(n)
#matrix = np.random.randint(2,size= (n,n))

# matrix = np.array([[0,1,0,0,0],
#           [0,0,1,0,0],
#           [0,0,0,1,0],
#           [0,0,0,0,1],
#           [0,0,1,0,0]])

#network con ambiente
matrix = np.array([[0,1,0,0,0,0],
          [0,0,1,0,0,0],
          [0,0,0,1,0,0],
          [0,0,0,0,1,0],
          [0,0,1,0,0,0],
          [0,0,0,0,1,0]])


#print(matrix)

#g1 = nx.from_numpy_matrix(matrix)

#matrix = matrix.T
print(matrix)


initial_states = []
final_states = []


def evo(nodes,t):
    
    up = np.zeros((n,1))
    up[-1] = 1
    #up = matrix.dot(nodes)
    previous_state = decimal(nodes.T)
    if t == 0:
        initial_states.append(previous_state)
        
    # for i in range(n):
    #     nodes[i] = up[i]
        
    for i in range(n):
        somma = 0 
        for j in range(n):
            somma = somma + matrix[j][i]*nodes[j]
        up[i] = somma
    
    for i in range(n):
        if up[i] != 0:
            nodes[i] = 1
        else:
            nodes[i] = 0 
        
    nodes[-1] = 1
    state = decimal(nodes.T)
    g.add_edge(previous_state,state)
    labels[state] = state
    if previous_state == state:
        final_states.append(previous_state)
    
def shuffle(nodes):
    for i in range(n):
        nodes = np.zeros((n,1))
        nodes[i] = 1
        for t in range(25):
            evo(nodes,t)
            
state = decimal(nodes.T)
labels[state] = state         
shuffle(nodes) 


g1 = nx.from_numpy_matrix(matrix)



pos = nx.layout.spring_layout(g)

nx.draw_networkx(g,pos = pos,node_size = 15,with_labels= True)
nx.draw_networkx_nodes(g,pos,
                       nodelist=initial_states,node_color='y',node_size = 35)
nx.draw_networkx_nodes(g,pos,
                       nodelist=final_states,node_color='r',node_size = 35)


#nx.draw(g1)
#plt.show
#nx.draw_networkx_labels(g,pos,labels,font_size=20)
plt.show()
plt.savefig("confenv.png")
#nx.write_gexf(g,"net.gexf")