#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:32:00 2020

@author: riccardo
"""



import numpy as np
import pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd

import integration as inte


#number of genes
n = 30

vmax = 4
nodes = np.zeros((n,n))
nodes[0] = np.random.randint(0,2,size=n)

#W = np.random.randint(0,2,size=(n,n))
W = np.zeros((n,n))
W[0][10] = 1
W[10][0] = 1
W[1][29] = 1
W[29][12] = 1
W[12][25] = 1
W[25][1] = 1
W[25][22] = 1
W[25][23] = 1

fig, ax = plt.subplots(2,2)

upper, = ax[0,0].plot([],[], c="black",linestyle = "--",label = "p")
lower, = ax[0,0].plot([],[], c="black",linestyle = "--",label = "p")
upper2, = ax[1,0].plot([],[], c="black",linestyle = "--",label = "p")
lower2, = ax[1,0].plot([],[], c="black",linestyle = "--",label = "p")
trajectory = []
mean = []

for i in range(n):
    t, = ax[0,0].plot([],[])
    field, = ax[1,0].plot([],[])
    mean.append(field)
    trajectory.append(t)
    
#maxwelldist, = ax[1,0].step([],[],label = 'distribution')
#maxwellfit, = ax[1,0].plot([],[],label = "fit")
# ax[0,0].legend()
# ax[1,0].legend()
#phasespace, = ax[0,1].step([],[])
#entr, = ax[1,1].plot([],[], label = "S")
#shan, = ax[1,1].plot([],[], linestyle = "--", label = "S_infty")
#ax[1,1].legend()


t = {}
field = {}

for i in range(n):
    t[i] = []
    field[i] = []
    
def init():

  ax[0,0].set_xlim(-0.1,200)
  ax[0,0].set_ylim(-1.5,1.5)
  ax[0,0].set_title("Equazione stocastica")
  ax[0,0].set_xlabel("time")
  ax[0,0].set_ylabel("p")

  ax[0,1].imshow(nodes)
  # ax[0,1].set_xlim(-5,5)
  # ax[0,1].set_ylim(0,0.1)
  # ax[0,1].set_xlabel("x")
  # ax[0,1].set_ylabel("rho_x")
  # ax[0,1].set_title("Distribution of positions")
  
  ax[1,0].set_xlim(-0.1,200)
  ax[1,0].set_ylim(-1.5,1.5)
 # ax[1,0].set_xlabel("time")
  ax[1,0].set_ylabel("p")
  ax[1,0].set_title("Equazione di campo medio")

  ax[1,1].set_xlim(-0.1,200)
  ax[1,1].set_ylim(-1.5,1.5)
  ax[1,1].set_xlabel("time")
  # ax[1,1].set_ylabel("entropy")
  #ax[1,1].set_title("Media delle realizzazioni")

  x = np.linspace(0,200, num = 200)
  y = np.ones(200)
  upper.set_data(x,y)
  y = np.zeros(200)
  lower.set_data(x,y)
  x = np.linspace(0,200, num = 200)
  y = np.ones(200)
  upper2.set_data(x,y)
  y = np.zeros(200)
  lower2.set_data(x,y)
  
  return trajectory[0],


def evo(frames):
    
    #nodes[0] = nodes[1] + realization(nodes[1],d(nodes[1],nodes[0]))
    for i in range(n):
        somma = 0 
        for j in range(n):
            somma = somma + W[i][j]*nodes[frames][j]
        nodes[frames+1][i] = somma
        
    for i in range(n):
        if nodes[frames+1][i] != 0:
            nodes[frames+1][i] = 1
        else:
            nodes[frames+1][i] = 0 

    ax[0,1].imshow(nodes)
 
    return  trajectory[0]#,trajectory[1],trajectory[2]#,#ax[0,1]

#print((L.shape))
#print(np.linalg.eig(L)[0])



ani = FuncAnimation(fig, evo, frames = np.arange(0,200), interval = 50,init_func = init, blit = False)
#plt.tight_layout()
#ani.save('biblio/transition.gif', dpi=120, writer='imagemagick')
