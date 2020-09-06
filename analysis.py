import pylab as plt
import numpy as np
import networkx as nx

import  random_network  as rn
import pandas as pd

PATH = "tesi"
#%%  ################################ MEAN ACTIVITY VERSUS NOISE #####################À


realizations = 20
steps = 100
N = 10
K = 2
for k in [False,True]:
    mean_activities = []
    probabilities = [i*0.01 for i in range(steps)]
    #label = "N =" + str(N)
        
    for i in range(steps):
        activities = []
        graphs = [rn.Random_Network(N, K) for i in range(realizations)]
    
        for j in range(realizations):
            #initial conditions
            for s in range(realizations):
                graphs[s].nodes[np.random.randint(N)] = 1
            
            
            #rn.initial_conditions(graphs[j], N)
            if k:
                rn.evolution(graphs[j],iterations=N+1,p=0.01*i)
            else:
                for m in range(N+1):
                    graphs[j].nodes[graphs[j].control_node] = 0  
                    rn.evolution(graphs[j],iterations=1,p=0.01*i)

            activities.append(rn.activity(graphs[j], N))
        mean_activities.append(np.mean((np.array(activities))))
    
    if k :
        label = "with control nodes"
    else:
        label = "without control nodes"
    plt.plot(probabilities,mean_activities,label=label)
    plt.legend()
plt.xlabel("noise")
plt.ylabel("average activity")
#plt.savefig("activity.png")
# a = [i*0.01 for i in range(steps)]
# s = pd.Series(a)
# df = pd.DataFrame()
# df[0] = a
# df[1] = pd.DataFrame(np.array(mean_activities))
# df.to_csv(PATH+"/data/"+label+".dat",sep = " ",decimal=".",index=False,header=False)


#%% ########################## MEAN ACTIVITY VERSUS K-INCOMING LINKS #############################

realizations = 100

for K in range(1,3):
    mean_activities = []
    for N in range(2,50):
        activity = []
        graphs = [rn.Random_Network(N, K) for i in range(realizations)]
        for i in range(realizations):
            graphs[i].nodes = np.zeros((N,1))
            try:
                graphs[i].nodes[rn.find_control_nodes(graphs[i], N)] = 1
            except:
                graphs[i].nodes[np.random.randint(N)] = 1
            rn.evolution(graphs[i],p=0)
            activity.append(rn.activity(graphs[i],N))
        mean_activities.append(np.mean(np.array(activity)))
    
    
    plt.ylim(0,1.1)
    plt.plot(np.array(mean_activities))
    # df = pd.DataFrame(np.array(mean_activities))
    # df.to_csv(PATH+"/data/"+label+".dat",sep = " ")

#%%   
# plt.figure()
# graph = nx.from_numpy_matrix(g.adj_matrix.T,create_using = nx.DiGraph)
# active_nodes = []
        
# for i in range(len(g.nodes)):
#     if g.nodes[i] == 1 :
#         active_nodes.append(i)
        
# pos = nx.layout.spring_layout(graph)
# nx.draw_networkx(graph,pos = pos,with_labels=True)
# #nx.draw_networkx(graph, pos = pos, with_labels= True)

# nx.draw_networkx_nodes(g,pos,
#                         nodelist=active_nodes,
#                         node_color='y')
# # plt.savefig("net.png")
# #%%
K = 2
max_nodes = 50
iterations = 10

outgoing_links = []
number_of_loops = []
mean_number_of_loops = []
mean_outgoing_links = []
for i in range(2,max_nodes):
    for j in range(iterations): 
        g = rn.Random_Network(i, K)
        outgoing_links.append(np.mean(sum(g.adj_matrix)))
        graph = nx.from_numpy_matrix(g.adj_matrix.T, create_using=nx.DiGraph)
        cycles = nx.cycle_basis(graph.to_undirected())
        number_of_loops.append(len(cycles))
        
    mean_outgoing_links.append(np.mean(outgoing_links))
    mean_number_of_loops.append(np.mean(number_of_loops))


df = pd.DataFrame()
df[0] = mean_outgoing_links
df.to_csv(PATH+"/data/mean-outgoing-links.dat",sep=" ",header=False)
df[0] = mean_number_of_loops
df.to_csv(PATH+"/data/mean-number-of-loops.dat",sep=" ",header=False)

# plt.suptitle("incoming links K =" + str(K))
# ax[0].plot(mean_outgoing_links)
# ax[0].plot(x,y,"--")
# ax[0].set_ylim(0.5,2.5)
# ax[0].set_xlabel("nodes")
# ax[0].set_ylabel("mean of outgoing links")

# ax[1].plot(mean_number_of_loops)
# ax[1].set_xlabel("nodes")
# ax[1].set_ylabel("mean number of loops")

# plt.savefig("k="+ str(K) +".png")
# plt.figure()
# nx.draw(nx.from_numpy_array(g.adj_matrix,create_using= nx.DiGraph))

# if num>1:
#     for i in range(num-1):
#         neg1 = np.zeros((N*(i+1),N))
#         a = [np.random.randint(N*(i+1))]
#         a.append(np.random.randint(N))
#         neg1[a[0]][a[1]] = -1
#         neg2 = np.zeros((N,N*(i+1)))
#         c = [np.random.randint(N*(i+1))]
#         c.append(np.random.randint(N))
        
#         neg2[c[1]][c[0]] = -1

#         #print(negedges)
        
#         tot = np.block([[tot,       neg1        ],
#                         [neg2, gr[i].adj_matrix              ]])

#%%
import  random_network  as rn

import pylab as plt
import numpy as np
import networkx as nx

import  random_network  as rn
import pandas as pd


noise = 100
number_of_clusters = 3
N = 10
K = 2
graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
Net = rn.Network(tot)
############# INITIAL CONDITIONS #####################
for i in range(number_of_clusters):
    #Net.nodes[control_nodes[i]] = 1
    Net.nodes[np.random.randint(N*i,N*(i+1))] = 1
    
    activities = []
    for j in range(noise):
        
    
        for i in control_nodes:
        ##### PROBABILITÀ DI AZZERARE IL CONTROL NODE AD OGNI STEP
           Net.nodes[control_nodes] = 0
        
        rn.evolution(Net,iterations=1,p = 0.1)
        activities.append(rn.activity(Net, N,number_of_clusters=number_of_clusters))
    
    #UCCIDO UN LINK AD OGNI STEP ####################
    #a =  np.random.randint(N*number_of_clusters-N)
    #Net.adj_matrix[a + np.random.randint(-N,N)][a + np.random.randint(-N,N)] = 0    
        
    
plt.plot(activities,label="cluster")
plt.legend()
plt.xlabel("t")
plt.ylabel("Average activity")
plt.ylim(0,2)
#plt.savefig("3clusters.png")

#%%%% MEAN ACTIVITY VERSUS CONTROL NODES
    
import  random_network  as rn

import pylab as plt
import numpy as np
import networkx as nx

import  random_network  as rn
import pandas as pd

realizations = 40
steps = 100
N = 15
K = 2

probabilities = [i*0.01 for i in range(steps)]
#label = "N =" + str(N)

for A in [False, True]:
    label = str(A)
    mean_activities = []
    for i in range(steps):
        activities = []
        graphs = [rn.Random_Network(N, K) for i in range(realizations)]
    
        for j in range(realizations):
            #initial conditions
            for s in range(realizations):
                graphs[s].nodes[np.random.randint(N)] = 1
     
            for m in range(N+1):
                if A:
                    graphs[j].nodes[graphs[j].control_node] = 0  
                rn.evolution(graphs[j],iterations=1,p=0.01*i)
    
            activities.append(rn.activity(graphs[j], N))
        mean_activities.append(np.mean((np.array(activities))))
    
    plt.plot(probabilities,mean_activities,label=label)
plt.legend()
plt.xlabel("noise")
plt.ylabel("average activity")
plt.savefig("doubleclustersactivity.png")
# a = [i*0.01 for i in range(steps)]
# s = pd.Series(a)
# df = pd.DataFrame()
# df[0] = a
# df[1] = pd.DataFrame(np.array(mean_activities))
# df.to_csv(PATH+"/data/"+label+".dat",sep = " ",decimal=".",index=False,header=False)

#%%
import random_network as rn
import pylab as plt
import networkx as nx
N = 5
K = 2
number_of_clusters = 3
graphs = [rn.Random_Network(N, K) for i in range(number_of_clusters)]
control_nodes = [graphs[i].control_node for i in range(number_of_clusters) ]
tot = rn.create_clusters(graphs, control_nodes, N)
Net = rn.Network(tot)
gr = nx.from_numpy_matrix(Net.adj_matrix,create_using=nx.DiGraph)
nx.draw_networkx(gr, with_labels=True)
#%%
import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn
import pandas as pd


time = 1000
number_of_clusters = 2
N = 50
K = 2
graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
single_cluster_control_nodes = [rn.outgoing_links(graphs[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
Net = rn.Network(tot)
############# INITIAL CONDITIONS #####################
#Net.nodes[np.random.randint(N)] = 1
#for i in range(number_of_clusters):
    
activities = []
for j in range(time):
    
    
    activities.append(rn.activity(Net, N,number_of_clusters=number_of_clusters))
    rn.evolution(Net,iterations=1,p = 0.)
    if np.random.uniform(0,1)<0.1:
        Net.nodes[control_nodes[np.random.randint(2)]] = 1
    
    #UCCIDO UN LINK AD OGNI STEP ####################
    #a =  np.random.randint(N*number_of_clusters-N)
    #Net.adj_matrix[a + np.random.randint(-N,N)][a + np.random.randint(-N,N)] = 0    
        
    
plt.plot(activities,label="cluster")
plt.legend()
plt.xlabel("t")
plt.ylabel("Average activity")
plt.ylim(0,2)
plt.xlim(-2,time)
#plt.savefig("3clusters.png")
#%%%

import random_network as rn
import numpy as np
import pylab as plt
N = 20

out = []
for i in range(1000):
    g = rn.Random_Network(N, 2)
    out.append(np.sum(g.adj_matrix)/N)
    
plt.hist(out,density=True)
plt.xlim(0,5)

#%%
import pylab as plt
import pandas as pd
import numpy as np

p = np.linspace(0.001,0.999,num=1000)
k = 1/(2*p*(1-p)) 

a = [i*0.001 for i in range(len(k))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(k))
df.to_csv("phase-transition.dat",sep = " ",decimal=".",index=False,header=False)