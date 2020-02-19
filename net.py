
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd



#iterations
frames = 100
#particles
n = 10
#time step
dt = 0.1
###################################
fig = plt.figure()

matrix =  np.random.randint(2, size=(10,10))

nodes = np.ones((10,10))
def init():

    
    im = plt.imshow(nodes)
    
    return im,


def evo(frames):
    plt.clf()
    for i in range(n):
        for j in range(n):
            nodes[i][j] = matrix[i][j] + nodes[i][j]
    
    im = plt.imshow(nodes)
    #print(nodes)
    return im

ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)

