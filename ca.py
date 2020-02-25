#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:29:12 2020

@author: riccardo
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd



#iterations
frames = 100
#particles
n = 100
#time step
dt = 0.1
###################################
fig = plt.figure()

#matrix =  np.random.randint(2, size=(n,n))
matrix = np.array([[0,1,0,0],
          [0,0,1,0],
          [1,0,0,0],
          [0,0,1,0]])
ca = np.random.random(size=(n,n))
up = np.zeros((n,n))
#initial state
#nodes[0] = 1

def init():

    
    im = plt.imshow(ca)
    
    return im,



def integration(u):
    ut = np.ones((n,n))
    b = 0.5
    c = 0.5
    a = [-1,0.5,0.5,0.5]
    for i in range(n):
        for j in range(n):
            for k in range(-1,1):
                for l in range(-1,1):
                    ut[i][j] = a[k]*u[i+k][j+l] + b*u[i][j]*(1-u[i][j]) + c*u[i-20][j]
    #ut = np.array([a[k]*u[i+k][j+l] + b*u[i][j]*(1-u[i][j]) + c*u[i-20][j] for l in range(-1,2) for k in range(-1,2) for j in range(n) for i in range(n)]).reshape(n,n)
    #print(ut.shape)
    return ut

def evo(frames):
    plt.clf()
    up = np.zeros((n,n))


    up = integration(ca)
    
    im = plt.imshow(up)
    
    for i in range(n):
        ca[i] = up[i]
    #print(ca.T)
    
    return im

ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 200,init_func = init, blit = False)

