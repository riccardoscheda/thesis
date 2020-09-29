
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation # animation plot
import pandas as pd

import random_network as rn

############PARAMETERS######################################

################## HARMONIC OSCILLATOR ######################
#eps = 0.
#gamma = .0
############################################################


#iterations
frames = 1000
#particles

##############################################################
qx = []
K = 2
K1 = 2
N = 10
M = 30
noise = 0.2
env_noise = 0.1
realizations = 500
env_control_nodes = []
control_nodes = []
number_of_clusters = 2
for i in range(realizations):
    a = True
    while a:
        try:
            graphs = [rn.Random_Network(N, K)]
            graphs.append(rn.Random_Network(M,K1))
            control_nodes.append([graphs[i].control_nodes[0]+i*N for i in range(number_of_clusters) ])
            env_control_nodes.append([graphs[i].control_nodes[1]+i*N for i in range(number_of_clusters)])
            tot = rn.create_net(graphs, control_nodes[i],env_control_nodes[i], N,M)
            net = rn.Network(tot,number_of_clusters)
            qx.append(net)
            a = False
        except:
            pass
            
        
######################### INITIAL CONDITIONS ################################
for i in range(realizations):
    #qx[i].nodes[np.random.randint(N+M)] = 1
    for j in range(N//2):
        qx[i].nodes[j] = 1
#############################################################################    
act = np.zeros(realizations)
t = np.zeros(realizations)
rhox, binx = np.histogram(act,density = 1,bins=20)
#rhop, binp = np.histogram(qx,density = 1,bins=50)

########################################################
fig, ax = plt.subplots(2,2)

particle, = ax[0,0].step([],[], label = "rho_p")
trajectory, = ax[0,0].step([],[], label = "fit")
maxwelldist, = ax[1,0].step([],[],label = 'distribution')
#maxwellfit, = ax[1,0].plot([],[],label = "fit")

ax[0,0].legend()
ax[1,0].legend()
phasespace, = ax[0,1].step([],[])
potential, = ax[0,1].plot([],[],label = "x")
entr, = ax[1,1].plot([],[], label = "S")
shan, = ax[1,1].plot([],[], linestyle = "--", label = "S_infty")
#ax[1,1].legend()


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

  ax[0,1].set_xlim(-2,2)
  ax[0,1].set_ylim(0,0.2)
  ax[0,1].set_xlabel("x")
  ax[0,1].set_ylabel("rho_x")
  ax[0,1].set_title("Distribution of activities")

  ax[1,0].set_xlim(-0.1, 10)
  ax[1,0].set_ylim(-0.001, 0.05)
  ax[1,0].set_xlabel("time (log)")
  ax[1,0].set_ylabel("frequency")
  ax[1,0].set_title("Distribution of transition times")

  #ax[1,1].set_xlim(0, 300)
  # ax[1,1].set_ylim(-5, 5)
  # ax[1,1].set_xlabel("time")
  # ax[1,1].set_ylabel("entropy")
  # ax[1,1].set_title("Entropy")

  return particle,

times = []

def evo(frames):
    for i in range(realizations):
      prima = rn.activity(qx[i],N,M,number_of_clusters=2)
      prima[0] = - prima[0]
      rn.evolution(qx[i],iterations=1,p=noise)
      rn.env(qx[i],env_control_nodes[i], p=env_noise)
      dopo = rn.activity(qx[i],N,M,number_of_clusters=2)
      dopo[0] = -dopo[0]
      act[i] = dopo[0] + dopo[1]
      # if  frames>2 and (prima[0]+prima[1])*act[i]<0:
      #     times.append(np.log(frames-t[i]))
      #     t[i] = frames
          
    bins = np.arange(np.floor(act.min()),np.ceil(act.max()))
    

    rhox, binx = np.histogram(act,density = 1, bins = 20)


    #hist = np.histogram(times,density = 1, bins=20)



    #tx.append(qx[0])
    #tp.append(px[0]/m)

    rhox += 1e-35
    #rhop += 1e-35
    #entropy.append(-np.sum(rhox*rhop*np.log(rhox*rhop)*np.diff(binx)*np.diff(binp)))

    #x = np.linspace(-10, 10, num = 1000)
   # maxwellpdf = np.sqrt(m/(2*np.pi*KT))*np.exp(-x**2*m/(2*KT))/50
    #trajectory.set_data(x, maxwellpdf)
    #particle.set_data(binp[1:], rhop/50)
    phasespace.set_data(binx[1:],rhox/20)

    ####################################################
    #space = np.array(qx)
    #space = np.vstack((space,px))

    #hist_space, bins = np.histogramdd(np.array(space.T),bins =50,density = 1)#,range = (interval,interval,interval,interval,interval,interval))
    #hist_space += 1e-35

    #entropy.append(-np.sum(hist_space*np.log(hist_space)*np.diff(bins[0])*np.diff(bins[1])))



    ###########################################################
    #fit
    #x = np.linspace(0, 10, num = 1000)
    #maxwellpdf = 2*np.sqrt(m/(2*np.pi*KT))*np.exp(-x**2*m/(2*KT))/50

    
    #maxwelldist.set_data(hist[1][1:],hist[0]/25)

#    maxwellfit.set_data(x, maxwellpdf)
   # entr.set_data(np.arange(0,len(entropy)),entropy)


    
   # x = np.linspace(-5, 5, num = 1000)
    
   # shan.set_data(x,y)
    
    return phasespace, #maxwelldist,  #entr, shan


ani = FuncAnimation(fig, evo, frames = np.arange(0,1000), interval = 50,init_func = init, blit = False)
#plt.tight_layout()
#ani.save('Doublewell.gif', dpi=140, writer='imagemagick')