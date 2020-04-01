
import numpy as np
import pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd

import integration as inte


#number of genes
n = 100
#transition matrix
W = np.random.uniform(-1,1,size = (n,n))
#mean lifetime of ecited states
gamma = np.random.uniform(0,1,size = (n))


#probability tha the node is in the state 1
p = np.zeros(n)
p[25] = 1


#laplacian matrix
L = W - gamma*np.identity(n)


for i in range(10):
    p = sum(L)*p - gamma*p
    #plt.plot(np.linspace(0,n,num = n),p)
    
    
    
############PARAMETERS######################################
gamma = 1.
eps = 0.6
KT = eps**2/(2*gamma)
m = 1
################## HARMONIC OSCILLATOR ######################
#eps = 0.
#gamma = .0
############################################################


#iterations
frames = 100
#particles
N = n
#time step
dt = 0.1

##############################################################
qx, px = np.ones(N), -np.ones(N)*10

rhox, binx = np.histogram(qx,density = 1,bins=50)
rhop, binp = np.histogram(px,density = 1,bins=50)

varx = rhox.var()
varp = rhop.var()

var = varx + varp
########################################################
fig, ax = plt.subplots(2,2)

particle, = ax[0,0].step([],[], label = "rho_p")
trajectory, = ax[0,0].step([],[], label = "fit")
maxwelldist, = ax[1,0].step([],[],label = 'distribution')
maxwellfit, = ax[1,0].plot([],[],label = "fit")
ax[0,0].legend()
ax[1,0].legend()
phasespace, = ax[0,1].step([],[])
entr, = ax[1,1].plot([],[], label = "S")
shan, = ax[1,1].plot([],[], linestyle = "--", label = "S_infty")
ax[1,1].legend()


tx = []
tp = []
entropy = []
shannon = []



def init():

  ax[0,0].set_xlim(-2,2)
  ax[0,0].set_ylim(0,0.05)
  ax[0,0].set_title("Distribution of momenta")
  ax[0,0].set_xlabel("p")
  ax[0,0].set_ylabel("rho_p")

  ax[0,1].set_xlim(-5,5)
  ax[0,1].set_ylim(0,0.1)
  ax[0,1].set_xlabel("x")
  ax[0,1].set_ylabel("rho_x")
  ax[0,1].set_title("Distribution of positions")

  ax[1,0].set_xlim(-0.1, 5)
  ax[1,0].set_ylim(-0.001, 0.1)
  ax[1,0].set_xlabel("v")
  ax[1,0].set_ylabel("rho(v)")
  ax[1,0].set_title("velocity Distribution")

  ax[1,1].set_xlim(0, 300)
  ax[1,1].set_ylim(-5, 5)
  ax[1,1].set_xlabel("time")
  ax[1,1].set_ylabel("entropy")
  ax[1,1].set_title("Entropy")

  return particle,

def evo(frames):
    for i in range(N):
      s = np.sum(W[i])
      qx[i], px[i] = inte.simplettic(qx[i], px[i],dt,eps  ,gamma,s,i)

    vel = np.sqrt(px**2)/m

    bins = np.arange(np.floor(qx.min()),np.ceil(qx.max()))
    n = 100

    rhox, binx = np.histogram(qx,density = 1, bins = 100)
    rhop, binp = np.histogram(px,density = 1,bins = 50)


    hist = np.histogram(vel,density = 1, bins=50)



    #tx.append(qx[0])
    #tp.append(px[0]/m)

    rhox += 1e-35
    rhop += 1e-35
    #entropy.append(-np.sum(rhox*rhop*np.log(rhox*rhop)*np.diff(binx)*np.diff(binp)))

    x = np.linspace(-10, 10, num = 1000)
    maxwellpdf = np.sqrt(m/(2*np.pi*KT))*np.exp(-x**2*m/(2*KT))/50
    trajectory.set_data(x, maxwellpdf)
    particle.set_data(binp[1:], rhop/50)
    phasespace.set_data(binx[1:],rhox/50)

    ####################################################
    space = np.array(qx)
    space = np.vstack((space,px))

    hist_space, bins = np.histogramdd(np.array(space.T),bins =50,density = 1)#,range = (interval,interval,interval,interval,interval,interval))
    hist_space += 1e-35

    entropy.append(-np.sum(hist_space*np.log(hist_space)*np.diff(bins[0])*np.diff(bins[1])))



    ###########################################################
    #fit
    x = np.linspace(0, 10, num = 1000)
    maxwellpdf = 2*np.sqrt(m/(2*np.pi*KT))*np.exp(-x**2*m/(2*KT))/50

    maxwelldist.set_data(hist[1][1:],hist[0]/50)

    maxwellfit.set_data(x, maxwellpdf)
    entr.set_data(np.arange(0,len(entropy)),entropy)



    x = np.linspace(0, 1000, num = 1000)
    y = np.ones(1000)*(np.log(KT) + 0.5*np.log(var))
    shan.set_data(x,y)

    return particle, trajectory, phasespace, maxwelldist, maxwellfit, entr, shan


ani = FuncAnimation(fig, evo, frames = np.arange(0,100), interval = 50,init_func = init, blit = True)
#plt.tight_layout()
#ani.save('Doublewell.gif', dpi=140, writer='imagemagick')

  

