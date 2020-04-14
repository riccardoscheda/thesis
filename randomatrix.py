#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:09:13 2020

@author: riccardo
"""


import numpy as np
import pylab as plt 

#number of genes
n = 1000
#transition matrix
W = np.random.normal(0,1,size = (n,n))

#mean lifetime of ecited states
gamma = np.ones(n)*2.
gammahat = sum(W.T)
gammahat
deltagamma = gamma - gammahat
deltagamma
#deltagamma = np.ones(n)*2
#laplacian matrix
L = W - gammahat*np.identity(n)

Ws = (W + W.T) / 2
eigenvalues , _ = np.linalg.eig(L)

plt.hist(eigenvalues,bins = 30,density= True)
