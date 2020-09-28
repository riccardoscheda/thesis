import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd
import random_network as rn

frames = 1000
qx = []
K = 2
K1 = 1
N = 15
M = 15
noise = 0.2
env_noise = 0.1
realizations = 1000
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
# for i in range(realizations):
#     #qx[i].nodes[np.random.randint(N+M)] = 1
    for j in range(N):
        qx[i].nodes[j] = 1
#############################################################################    
act = np.zeros(realizations)
t = np.zeros(realizations)

times = []
times.append(0)
def evo(frames):
    for i in range(realizations):
      prima = rn.activity(qx[i],N,M,number_of_clusters=2)
      prima[0] = - prima[0]
      rn.evolution(qx[i],iterations=1,p=noise)
      rn.env(qx[i],env_control_nodes[i], p=env_noise)
      dopo = rn.activity(qx[i],N,M,number_of_clusters=2)
      dopo[0] = -dopo[0]
      act[i] = dopo[0] + dopo[1]
      if  frames>2 and (prima[0]+prima[1])*act[i]<0:
        times.append(np.log(frames-t[i]))
        t[i] = frames
          
    bins = np.arange(np.floor(act.min()),np.ceil(act.max()))
    rhox, binx = np.histogram(act,density = 1, bins = 30)

    return rhox, binx

for i in range(frames):
    a,b = evo(i)
   
    
   
plt.hist(times,density=True,stacked=True)
plt.figure()
plt.step(b[1:],a/30)
plt.xlim(-2,2)
plt.ylim(0,1)