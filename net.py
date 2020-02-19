
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd



#iterations
frames = 100
#particles
n = 4
#time step
dt = 0.1
###################################
fig = plt.figure()

#matrix =  np.random.randint(2, size=(n,n))
matrix = np.array([[0,1,0,0],
          [0,0,1,1],
          [1,0,0,0],
          [0,0,1,0]])
nodes = np.zeros((n,1))
up = np.zeros((n,1))
#initial state
nodes[0] = 1

def init():

    
    im = plt.imshow(nodes)
    
    return im,



def evo(frames):
    plt.clf()
    

    for i in range(n):
        somma = 0
        for j in range(n):
            somma  = somma + matrix[i][j] * nodes[i]
        up[i]= somma
    
    
    
    im = plt.imshow(up)
    
    for i in range(n):
        nodes[i] = up[i]
    print(nodes)
    return im

ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)

