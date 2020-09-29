import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation # animation plot
import pandas as pd
import random_network as rn

from scipy.stats import expon
from scipy.optimize import curve_fit

frames = 1000
qx = []
K = 2
K1 = 2
N = 10
M = 20
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
    for j in range(N//2):
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
    
fig, ax = plt.subplots(2,1)
ax[0].set_title("noise"+str(noise)+"N="+str(N)+"M="+str(M)+"K="+str(K)+"K1="+str(K1))
bins = np.linspace(0, 7, 20)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
data,bin1 = np.histogram(times, bins=bins)

def fit_function(x, A,B):
    return (A * np.exp(-(x)) + B * np.exp( -(x)**2))


# 5.) Fit the function to the histogram data.
popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data)
xspace = np.linspace(0, 7, 1000)



ax[1].bar(binscenters, data, width=bins[1] - bins[0])
ax[1].plot(xspace, fit_function(xspace, *popt) , color='darkorange')
         
ax[0].step(b[1:],a/30)
ax[0].set_xlim(-2,2)
ax[0].set_ylim(0,0.2)

plt.savefig("noise"+str(noise)+"N="+str(N)+"M="+str(M)+"K="+str(K)+"K1="+str(K1)+".png")


df = pd.DataFrame()
df[0] = bin1[1:]
df[1] = data/sum(data)
df.to_csv("tesi/data/histotimes.dat",sep = " ",decimal=".",index=False,header=False)



