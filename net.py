
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd



#iterations
frames = 100
#particles
N = 5000
#time step
dt = 0.1
###################################
fig = plt.figure()

matrix =  np.random.randint(2, size=(10,10))

def init():

    
    im = plt.imshow(matrix)
    
    return im,


def evo(frames):
    plt.clf()
    
    for i in range(10):
        for j in range(10):    
            matrix[i,j] = matrix[j,i]*matrix[i,j]
    im = plt.imshow(matrix)
    print(matrix)
    return im

ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)

