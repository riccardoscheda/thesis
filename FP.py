
import numpy as np
import pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd

import integration as inte


#number of genes
n = 100
#transition matrix
W = inte.create_transition_matrix(n)

# W = np.zeros((n,n))
# W[0][1] = 1
# W[1][0] = 1

#mean lifetime of ecited states
#gamma = np.random.uniform(1,1.5,size= n)

#W = np.zeros((n,n))
# for i in range(n-1):
#     W[i+1][i] = 1

#W[0][-90] = 1

gamma = np.ones(n)

gammahat = sum(W.T)
gammahat
deltagamma = gamma - gammahat

##########################        SETTANDO TUTTO A ZERO, HO SOLO LA MATRICE W    #################
# deltagamma = np.zeros(n)
# gammahat = deltagamma
######################

#laplacian matrix
L = W - gammahat*np.identity(n)

############PARAMETERS#################################
eps = 0.1
#######################################################


#iterations
frames = 200
#particles
N = n
#time step
dt = 0.1

################################### CONDIZIONI INIZIALI ########################

p= np.zeros(N)
p[0] = 1
#p[1] = 0.5

m = np.zeros(N)
m[0] = 1
#m[1] = 0.5
########################################################
fig, ax = plt.subplots(2,2)

upper, = ax[0,0].plot([],[], c="black",linestyle = "--",label = "p")
lower, = ax[0,0].plot([],[], c="black",linestyle = "--",label = "p")
upper2, = ax[1,0].plot([],[], c="black",linestyle = "--",label = "p")
lower2, = ax[1,0].plot([],[], c="black",linestyle = "--",label = "p")
trajectory = []
mean = []
nodes = np.zeros((1,n))

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

  #ax[0,1].imshow(nodes.reshape(10,10))
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

def realization(p,n):
    if np.random.uniform(0,1)<p:
        n = 1
    else:
        n = 0 
    return n
        

up = np.zeros((1,n))
ripetitions = 50
def evo(frames):
    
    #local state
    for i in range(N):
        up[0][i] = nodes[0][i]
        
    for i in range(N):
        
      
      #s = 0 
      #print(s)
      media = 0
      for j in range(ripetitions):
          #laplacian matrix
          s = np.sum(L[i].dot(nodes[0]))
          ist = inte.simplettic(p[i],up[0][i],dt,eps,deltagamma[i],s,i)
          media = media + ist
          up[0][i] = realization(ist,nodes[0][i])
          
      p[i] = media/ripetitions    
      t[i].append(p[i])
      
      
      
      #### mean field ##################
      s = np.sum(L[i].dot(m))
      #s = 0 
      m[i] = inte.mean_field(m[i], dt, eps, deltagamma[i], s, i)
      field[i].append(m[i])
      ###############################
      
      
      trajectory[i].set_data(np.arange(0,len(t[i])),t[i])
      mean[i].set_data(np.arange(0,len(t[i])),field[i])
      
    for i in range(n):
        nodes[0][i] = realization(p[i],up[0][i])

    #ax[0,1].imshow(nodes.reshape(10,10))
    #print(sum)

    return  trajectory[0]#,trajectory[1],trajectory[2]#,#ax[0,1]


ani = FuncAnimation(fig, evo, frames = np.arange(0,200), interval = 50,init_func = init, blit = False)
#plt.tight_layout()
#ani.save('biblio/transition.gif', dpi=120, writer='imagemagick')
