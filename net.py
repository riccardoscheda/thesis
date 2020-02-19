
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

matrix = np.ones((10,10))

def init():

    im = plt.imshow(matrix)
    return im,


def evo(frames):

    im = plt.imshow(matrix)
    return im

ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 50,init_func = init, blit = True)

