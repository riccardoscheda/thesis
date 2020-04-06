
import numpy as np
import pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd

import integration as inte


#number of genes
n = 5
#transition matrix
W = np.random.uniform(-1,1,size = (n,n))
# W = np.array([[-0.5,0.3,0.1],
#               [0.5,-0.7,0.2],
#               [0.1,-0.1,0.2]])
#mean lifetime of ecited states
gamma = np.ones(n)*2
gammahat = sum(W)

deltagamma = gamma - gammahat
#laplacian matrix
L = W - gammahat*np.identity(n)

############PARAMETERS#################################
eps = 0.1
#######################################################


#iterations
frames = 100
#particles
N = n
#time step
dt = 0.1

##############################################################
qx, px = np.ones(N)*0.5, np.zeros(N)*0.5

########################################################
fig, ax = plt.subplots(2,2)

particle, = ax[0,0].plot([],[], label = "p")
trajectory = []
nodes = np.ones((1,n))

for i in range(n):
    t, = ax[0,0].plot([],[], label = "fit")
    trajectory.append(t)
    
maxwelldist, = ax[1,0].step([],[],label = 'distribution')
maxwellfit, = ax[1,0].plot([],[],label = "fit")
# ax[0,0].legend()
# ax[1,0].legend()
#phasespace, = ax[0,1].step([],[])
entr, = ax[1,1].plot([],[], label = "S")
shan, = ax[1,1].plot([],[], linestyle = "--", label = "S_infty")
#ax[1,1].legend()


tx = []
tp = []
t3 = []

entropy = []
shannon = []

t = {}

for i in range(n):
    t[i] = []

def init():

  ax[0,0].set_xlim(-0.1,100)
  ax[0,0].set_ylim(-1.5,1.5)
  # ax[0,0].set_title("Distribution of momenta")
  # ax[0,0].set_xlabel("p")
  # ax[0,0].set_ylabel("rho_p")

  #ax[0,1].imshow(nodes)
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

def realization(p,nodes,i):
    if np.random.uniform(0,1)<p:
        nodes[0][i] = 1
    else:
        nodes[0][i] = 0 
        
def evo(frames):
    for i in range(N):
      s = np.sum(L [i].dot(qx))
      s = 0 
      qx[i]= inte.simplettic(qx[i],dt,eps  ,deltagamma[i],s,i)
      #realization(qx[i],nodes,i)
      t[i].append(qx[i])
      trajectory[i].set_data(np.arange(0,len(t[i])),t[i])
    
    
   # ax[0,1].imshow(nodes)
    
    return trajectory[0],trajectory[1],trajectory[2],trajectory[3],trajectory[4], maxwelldist, maxwellfit, entr, shan


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 50,init_func = init, blit = True)
#plt.tight_layout()
#ani.save('Doublewell.gif', dpi=140, writer='imagemagick')

  

