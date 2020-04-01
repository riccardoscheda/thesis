
import numpy as np
import pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd

import integration as inte


#number of genes
n = 2
#transition matrix
W = np.random.uniform(-1,1,size = (n,n))
W = np.array([[-10,0],
              [0,0]])
#mean lifetime of ecited states
gamma = np.random.uniform(0,1,size = (n))

#probability tha the node is in the state 1
p = np.ones(n)


#laplacian matrix
L = W - gamma*np.identity(n)


############PARAMETERS######################################
gamma = 1.
eps = 0.6
#######################################################


#iterations
frames = 100
#particles
N = n
#time step
dt = 0.1

##############################################################
qx, px = np.ones(N), -np.ones(N)
qx[0] = 1
########################################################
fig, ax = plt.subplots(2,2)

particle, = ax[0,0].plot([],[], label = "p")
trajectory = []

for i in range(n):
    t, = ax[0,0].plot([],[], label = "fit")
    trajectory.append(t)
    
maxwelldist, = ax[1,0].step([],[],label = 'distribution')
maxwellfit, = ax[1,0].plot([],[],label = "fit")
# ax[0,0].legend()
# ax[1,0].legend()
phasespace, = ax[0,1].step([],[])
entr, = ax[1,1].plot([],[], label = "S")
shan, = ax[1,1].plot([],[], linestyle = "--", label = "S_infty")
#ax[1,1].legend()


tx = []
tp = []
entropy = []
shannon = []


def init():

  ax[0,0].set_xlim(0,100)
  ax[0,0].set_ylim(-10,10)
  # ax[0,0].set_title("Distribution of momenta")
  # ax[0,0].set_xlabel("p")
  # ax[0,0].set_ylabel("rho_p")

  # ax[0,1].set_xlim(-5,5)
  # ax[0,1].set_ylim(0,0.1)
  # ax[0,1].set_xlabel("x")
  # ax[0,1].set_ylabel("rho_x")
  # ax[0,1].set_title("Distribution of positions")

  # ax[1,0].set_xlim(-0.1, 5)
  # ax[1,0].set_ylim(-0.001, 0.1)
  # ax[1,0].set_xlabel("v")
  # ax[1,0].set_ylabel("rho(v)")
  # ax[1,0].set_title("velocity Distribution")

  # ax[1,1].set_xlim(0, 300)
  # ax[1,1].set_ylim(-5, 5)
  # ax[1,1].set_xlabel("time")
  # ax[1,1].set_ylabel("entropy")
  # ax[1,1].set_title("Entropy")

  return trajectory[0],

def evo(frames):
    for i in range(N):
      s = np.sum(W[i].dot(qx))
      qx[i], px[i] = inte.simplettic(qx[i], px[i],dt,eps  ,gamma,s,i)
      
      
    tx.append(qx[0])
    tp.append(qx[1])  
    
    #particle.set_data(np.arange(0,len(tx)),tx)
    trajectory[0].set_data(np.arange(0,len(tx)),tx)
    trajectory[1].set_data(np.arange(0,len(tp)),tp)
        

    return trajectory[0],trajectory[1], phasespace, maxwelldist, maxwellfit, entr, shan


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 50,init_func = init, blit = True)
#plt.tight_layout()
#ani.save('Doublewell.gif', dpi=140, writer='imagemagick')

  

