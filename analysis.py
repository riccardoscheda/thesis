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

realizations = 20

for K in range(1,3):
    mean_activities = []
    for N in range(10,45):
        activity = []
        graphs = [rn.Random_Network(N, K) for i in range(realizations)]
        for i in range(realizations):
            graphs[i].nodes = np.zeros((N,1))
            try:
                graphs[i].nodes[rn.find_control_nodes(graphs[i], N)[0]] = 1
            except:
                graphs[i].nodes[np.random.randint(N)] = 1
            rn.evolution(graphs[i],iterations = N,p=0)
            activity.append(rn.activity(graphs[i],N))
        mean_activities.append(np.mean(np.array(activity)))
    
    
    plt.ylim(0,1.1)
    plt.plot(np.array(mean_activities))
    # df = pd.DataFrame(np.array(mean_activities))
    # df.to_csv(PATH+"/data/"+label+".dat",sep = " ")

#%%   ####### OUTGOIN LINKS AND NUMBER OF LOOPS ##########
K = 2
max_nodes = 50
iterations = 20

outgoing_links = []
number_of_loops = []
mean_number_of_loops = []
mean_outgoing_links = []
for i in range(10,max_nodes):
    for j in range(iterations): 
        g = rn.Random_Network(i, K)
        outgoing_links.append(np.mean(g.adj_matrix.sum(axis=0)))
        #graph = nx.from_numpy_matrix(g.adj_matrix.T, create_using=nx.DiGraph)
        #cycles = nx.cycle_basis(graph.to_undirected())
        #number_of_loops.append(len(cycles))
        
    mean_outgoing_links.append(np.mean(outgoing_links))
    #mean_number_of_loops.append(np.mean(number_of_loops))

plt.plot(mean_outgoing_links)
df = pd.DataFrame()
# df[0] = mean_outgoing_links
# df.to_csv(PATH+"/data/mean-outgoing-links.dat",sep=" ",header=False)
# df[0] = mean_number_of_loops
# df.to_csv(PATH+"/data/mean-number-of-loops.dat",sep=" ",header=False)


#%%
import  random_network  as rn

import pylab as plt
import numpy as np
import networkx as nx

import  random_network  as rn
import pandas as pd



time = 300
noise = 5
number_of_clusters = 1
N = 30
K = 2
graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
Net = rn.Network(tot)
############# INITIAL CONDITIONS ########################
# for i in range(number_of_clusters):
#     Net.nodes[control_nodes[i]] = 1
#     Net.nodes[np.random.randint(N*i,N*(i+1))] = 1
    
Net.nodes = np.ones((N*number_of_clusters,1))

for i in range(noise):
    activities = []
    Net = rn.Random_Network(N,K)
    Net.nodes = np.ones((N*number_of_clusters,1))
    for j in range(time):
        
        #for i in control_nodes:
        ##### PROBABILITÀ DI AZZERARE IL CONTROL NODE AD OGNI STEP
            #Net.nodes[np.random.choice(control_nodes)] = 1
        
        rn.evolution(Net,iterations=1,p = i*0.1)
        activities.append(rn.activity(Net, N,number_of_clusters=number_of_clusters))
    
        #UCCIDO UN LINK AD OGNI STEP ####################
        #a =  np.random.randint(N*number_of_clusters-N)
        #Net.adj_matrix[a + np.random.randint(-N,N)][a + np.random.randint(-N,N)] = 0    
            
        
    plt.plot(activities,label=" noise = " + str(i*0.1))
plt.legend()
plt.xlabel("t")
plt.title("1 cluster, N = 30 ")
plt.ylabel("Average activity")
plt.ylim(0,2)
plt.savefig("1cluster-tempevolution.png")

#%%%% MEAN ACTIVITY VERSUS CONTROL NODES
    
import  random_network  as rn

import pylab as plt
import numpy as np
import networkx as nx

import  random_network  as rn
import pandas as pd

realizations = 40
steps = 200
noise = 2
N = 30
K = 2
number_of_clusters = 1
probabilities = [i*0.01 for i in range(steps)]
#label = "N =" + str(N)
for j in range(noise):
    for A in [False, True]:
        if A:
            label = "With control node, noise = " +str(j*0.1)
        else:
            label = "Without control node, noise = " +str(j*0.1)
        activity = []
        Net = rn.Random_Network(N, K)
        ####### INITIAL CONDITIONS ###########
        Net.nodes = np.ones((N*number_of_clusters,1))
        #Net.nodes[np.random.randint(N)] = 1
        ########################################
       
        for i in range(time):
            
            rn.evolution(Net,iterations=1, p = 0.1*j)
            if not A:
                Net.nodes[Net.control_nodes] = 0
            activity.append(rn.activity(Net, N,number_of_clusters=number_of_clusters))
        
        if A:
            plt.plot(activity,label=label,c=(1-j*0.3,0,j*0.3))
        else:
            plt.plot(activity,linestyle = "--",label=label,c=(1-j*0.3,0,j*0.3))
plt.legend()
plt.xlabel("t")
plt.title("Average activity with/without control nodes. 1 cluster")
plt.ylabel("average activity")
plt.ylim(0,2)
plt.savefig("1clustervscontrolnodes.png")
# a = [i*0.01 for i in range(steps)]
# s = pd.Series(a)
# df = pd.DataFrame()
# df[0] = a
# df[1] = pd.DataFrame(np.array(mean_activities))
# df.to_csv(PATH+"/data/"+label+".dat",sep = " ",decimal=".",index=False,header=False)

#%%  ################################### 3 CLUSTERS ################################
import random_network as rn
import pylab as plt
import networkx as nx
N = 20
K = 2
number_of_clusters = 3
time = 200
graphs = [rn.Random_Network(N, K) for i in range(number_of_clusters)]
control_nodes = [graphs[i].control_node for i in range(number_of_clusters) ]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters=number_of_clusters)
Net = rn.Network(tot,number_of_clusters)
######### INITIAL CONDITIONS ################
#Net.nodes = np.ones((N*number_of_clusters,1))

##### INITIAL ACTIVE NODES PER CLUSTER###############
# for i in range(2):
#     nod = [np.random.randint(N*i,N*(i+1))  for i in range(number_of_clusters)]
#     Net.nodes[nod] = 1
for i in range(N*(number_of_clusters-2)):
    Net.nodes[i] = 1
#############################################

activity = []
for i in range(time):
    rn.evolution(Net,iterations=1,p=0.05)
    activity.append(rn.activity(Net,N,number_of_clusters=3))
plt.plot(np.array(activity))

plt.xlabel("t")
plt.ylabel("average activity")
plt.ylim(0,2)
plt.title("3 clusters temporal evolution")
#plt.savefig("3clusters.png")
################################################# 3 CLUSTERS ##############################################
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
N = 30
number_of_clusters = 3
graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
Net = rn.Network(tot)

out = []

for i in range(200):
    graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
    single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
    control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
    tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
    Net = rn.Network(tot)

    Net.nodes = np.ones((N*number_of_clusters,1)) 

    rn.evolution(Net,iterations = 50,p=0.2)
    out.append(rn.activity(Net,N,number_of_clusters=number_of_clusters))
   # print(rn.activity(Net,N))
        
    
plt.hist(np.array(out))
#plt.xlim(0,1)

#%% ###########LATEX GRAPHSSSSSSSSSSSSSSSSS #################################S
import pylab as plt
import pandas as pd
import numpy as np

p = np.linspace(-5,5,num=1000)
a = 1.6
b = 0.29
deltap = 0.1
V = 2 + (p-deltap)**4 + b*(p-deltap)**3 - a*(p-deltap)**3 - a*b*(p-deltap)**2

k = np.exp(-V)*100
plt.plot(p,k)
a = [i*0.001 for i in range(len(k))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(k))
df.to_csv("tesi/data/histo.dat",sep = " ",decimal=".",index=False,header=False)
#plt.ylim(0,40)
#np.linspace(0,100,num=1000)
#%%
import random_network as rn
import pylab as plt
import networkx as nx
N = 5
K = 2
number_of_clusters = 2

######################################

#creation of the subnetworks
gr = [rn.Random_Network(N,K) for i in range(number_of_clusters)]

control_nodes = [gr[i].control_node+i*N for i in range(number_of_clusters)]
print(control_nodes)
tot = rn.create_clusters(gr, control_nodes, N,number_of_clusters,visual=False)
negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))
#print(negedges)
Net = rn.Network(tot,number_of_clusters)
graph = nx.from_numpy_matrix(tot.T, create_using=nx.DiGraph)
npos = nx.spring_layout(graph)
#cycles = nx.cycle_basis(graph.to_undirected())

################## ONLY FOR VISUALIZATION #######################
tot1 = rn.create_clusters(gr, control_nodes, N,number_of_clusters,visual=True)
abs_tot = abs(tot1)
graph1 = nx.from_numpy_matrix(abs_tot.T, create_using=nx.DiGraph)
npos = nx.kamada_kawai_layout(graph1)
#################################################################

nx.draw_networkx(graph,npos, with_labels= True)
nx.draw_networkx_nodes(graph,npos,
                    nodelist=control_nodes,
                    node_size=800,node_color="green")
nx.draw_networkx_edges(graph, npos,
                   edgelist=negedges,
                   width=3, alpha=0.4, edge_color='r')
control_nodes = [graphs[i].control_node+i*N for i in range(number_of_clusters)]
negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))
print(negedges)
print(control_nodes)
#plt.savefig("network.png")
#%% ############################### NUMBER OF LOOPS ##########################################

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt

mean_loops = []

for N in range(5,50):
    loops = []
    for i in range(10):
        g = rn.Random_Network(N,2)
        graph = nx.from_numpy_matrix(g.adj_matrix, create_using=nx.DiGraph)
        cycles = nx.simple_cycles(graph)
        #nx.draw_networkx(graph,with_labels=True)
        loops.append(len(list(cycles)))
    mean_loops.append(np.mean(np.array(loops)))
plt.plot(mean_loops)
a = [i for i in range(len(mean_loops))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(mean_loops))
df.to_csv("tesi/data/loops.dat",sep = " ",decimal=".",index=False,header=False)
#%%  ######################## NUMBER OF LOOPS PER NUMBER OF NODES ###################################################

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd

#N = 10
K = 2
mean_loops = []
for N in range(10,60):
    loops = []
    for i in range(10):
        g = rn.Random_Network(N,K)
        loops.append(g.loops)
    mean_loops.append(np.log(np.mean(np.array(loops))))
    
plt.plot(mean_loops)
plt.title("Number of loops per number of nodes")
plt.savefig(".png")

a = [i for i in range(len(mean_loops))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(mean_loops))
df.to_csv("tesi/data/numberofloops.dat",sep = " ",decimal=".",index=False,header=False)

#%% ################### MEAN ACTIVITY WITH AND WITHOUT CONTROL NODE #####################################À

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd


K = 2
N = 20
steps = 500
realizations = 100
activities = []
mean_activities0 = []
mean_activities1 = []
noise = 0.3

g = [rn.Random_Network(N,K) for i in range(realizations)]
for i in range(realizations):
    g[i].nodes = np.ones((N,1))
    
    
for i in range(steps):
    for j in range(realizations):
        activities.append(rn.activity(g[j],N))
        rn.evolution(g[j],iterations= 1, p=noise)
    mean_activities0.append(np.mean(np.array(activities)))
        
  
g = [rn.Random_Network(N,K) for i in range(realizations)]
activities = []
for i in range(realizations):
    g[i].nodes = np.ones((N,1))
    g[i].nodes[g[i].control_nodes[0]] = 0
    
for i in range(steps):
    for j in range(realizations):
        activities.append(rn.activity(g[j],N))
        g[j].nodes[g[j].control_nodes[0]] = 0 
        
        rn.evolution(g[j],iterations= 1, p=noise)
    mean_activities1.append(np.mean(np.array(activities)))
    
    
plt.plot(mean_activities0)
plt.plot(mean_activities1)
plt.ylim(0,1)
plt.xlim(0,steps)
#plt.savefig("averageactivitywithandwithoutcontrolnodes1.png")
a = [i for i in range(steps)]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(mean_activities0))
df.to_csv("tesi/data/meanactivity0"+str(noise)+".dat",sep = " ",decimal=".",index=False,header=False)
df2 = pd.DataFrame()
df2[0] = a
df2[1] = pd.DataFrame(np.array(mean_activities1))
df2.to_csv("tesi/data/meanactivity1"+str(noise)+".dat",sep = " ",decimal=".",index=False,header=False)
#%%
import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd


K = 2
N = 20

steps = 300
realizations = 20
mean_activities = pd.DataFrame()

for i in range(realizations):
    graphs = [rn.Random_Network(N, K) for i in range(2)]
    control_nodes = [graphs[k].control_nodes[0]+k*N for k in range(number_of_clusters) ]
    env_control_nodes = [graphs[k].control_nodes[1]+k*N for k in range(number_of_clusters)]
    
    tot = rn.create_net(graphs, control_nodes,env_control_nodes, N,M)
    
    Net = rn.Network(tot,number_of_clusters=2)
    for k in range(N):
        Net.nodes[k] = 1
    
    activity = []
    
    for j in range(1,steps):
        rn.evolution(Net,iterations=1,p=0.2)
        rn.env(Net,env_control_nodes,p=0.2)    
        act = rn.activity(Net,N,N,number_of_clusters=2)
        activity.append(act)
        
    mean_activities[i] = activity
    plt.plot(np.array(activity))
    
#%% ########################  FREQUENCY OF LOOPS FOR THE SINGLE CONTROL NODE ############################

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd


K = 2
a = 10
b = 40
mean_percentage = []
for N in range(a,b):
    percentage = []
    for i in range(40):
        g = rn.Random_Network(N,K)
        loopspercn = np.array(g.loops_per_cn)
        frequency = loopspercn/g.loops
        percentage.append(frequency[g.control_nodes[0]])  
    mean_percentage.append(np.mean(np.array(percentage)))
#nx.draw_networkx(nx.from_numpy_matrix(g.adj_matrix.T,create_using=nx.DiGraph), with_labels=True)
plt.plot(range(a,b),mean_percentage)
plt.ylim(0,1)
plt.title("frequency of loops per control nodes")
#plt.savefig("frequency of loops per control nodes.png")

a = [i for i in range(a,b)]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(mean_percentage))
df.to_csv("tesi/data/loopspercn.dat",sep = " ",decimal=".",index=False,header=False)


#%%  ###################################  3 CLUSTERS HISTOGRAM TIMES ################################
import random_network as rn
import pylab as plt
import networkx as nx
import pandas as pd

realizations = 100
noise = 0.1
N = 10
M=10
K = 2
number_of_clusters = 3
time = 2000

mean_times = []

for k in range(20):
    

    graphs = [rn.Random_Network(N, K) for i in range(number_of_clusters)]
    
    #graphs = [rn.Random_Network(N,K) for i in range(number_of_clusters)]
    control_nodes = [graphs[i].control_nodes[0]+i*N for i in range(number_of_clusters) ]
    env_control_nodes = [graphs[i].control_nodes[1]+i*N for i in range(number_of_clusters)]
    
    #tot = rn.create_net(graphs, control_nodes,env_control_nodes, N,M)
    tot = rn.create_clusters(graphs,control_nodes,env_control_nodes, N,number_of_clusters,visual=True)
    #print(tot)
    Net = rn.Network(tot,number_of_clusters)
    
    for j in range(realizations):
        times = []

        activity = []
        mean_activities = []
        # for i in range(N*(number_of_clusters-1)):
        #     Net.nodes[i] = 1
        Net.nodes[np.random.randint(N+M+N)] = 1
            
        for i in range(1,time):
            rn.evolution(Net,iterations=1,p=0.2)
            rn.env(Net,env_control_nodes,p=0.01)    
            act = rn.activity(Net,N,0,number_of_clusters=3)
        
            if abs(act[0]-act[2])<0.001:
                times.append(np.log(i))
            
                break
        
            #activity.append(act)
        #mean_activities.append(np.mean(activity))
        mean_times.append(np.mean(np.array(times)))
plt.hist(mean_times,bins=20)
plt.title("sdistribution of log of time transitions-N="+str(N)+" M="+str(M))
plt.savefig("histo-N="+str(N)+" M="+str(M)+".png")
   # plt.figure()

    # negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))
    # print(negedges)
    # print(control_nodes)
    # plt.plot(np.array(activity))
    # plt.xlabel("t")
    # plt.ylabel("average activity")
    # plt.ylim(0,2)
    # plt.title("N="+str(N)+" M="+str(M))
    # plt.savefig("N="+str(N)+" M="+str(M)+".png")
#print(act)


#################################### TIMES HISTOGRAM ##########################################
# df = pd.DataFrame()
# df[0] = np.histogram(times)[1][1:]
# df[1] = np.histogram(times)[0]
# df.to_csv("tesi/data/times.dat",sep = " ",decimal=".",index=False,header=False)
###############################################################################################
#%%  ##########################                2 CLUSTERS ############################
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
                graphs[i].nodes[rn.find_control_nodes(graphs[i], N)[0]] = 1
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



time = 300
noise = 5
number_of_clusters = 1
N = 30
K = 2
graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
Net = rn.Network(tot)
############# INITIAL CONDITIONS ########################
# for i in range(number_of_clusters):
#     Net.nodes[control_nodes[i]] = 1
#     Net.nodes[np.random.randint(N*i,N*(i+1))] = 1
    
Net.nodes = np.ones((N*number_of_clusters,1))

for i in range(noise):
    activities = []
    Net = rn.Random_Network(N,K)
    Net.nodes = np.ones((N*number_of_clusters,1))
    for j in range(time):
        
        #for i in control_nodes:
        ##### PROBABILITÀ DI AZZERARE IL CONTROL NODE AD OGNI STEP
            #Net.nodes[np.random.choice(control_nodes)] = 1
        
        rn.evolution(Net,iterations=1,p = i*0.1)
        activities.append(rn.activity(Net, N,number_of_clusters=number_of_clusters))
    
        #UCCIDO UN LINK AD OGNI STEP ####################
        #a =  np.random.randint(N*number_of_clusters-N)
        #Net.adj_matrix[a + np.random.randint(-N,N)][a + np.random.randint(-N,N)] = 0    
            
        
    plt.plot(activities,label=" noise = " + str(i*0.1))
plt.legend()
plt.xlabel("t")
plt.title("1 cluster, N = 30 ")
plt.ylabel("Average activity")
plt.ylim(0,2)
plt.savefig("1cluster-tempevolution.png")

#%%%% MEAN ACTIVITY VERSUS CONTROL NODES
    
import  random_network  as rn

import pylab as plt
import numpy as np
import networkx as nx

import  random_network  as rn
import pandas as pd

realizations = 40
time = 200
noise = 2
N = 30
K = 2
number_of_clusters = 1
probabilities = [i*0.01 for i in range(steps)]
#label = "N =" + str(N)
for j in range(noise):
    for A in [False, True]:
        if A:
            label = "With control node, noise = " +str(j*0.1)
        else:
            label = "Without control node, noise = " +str(j*0.1)
        activity = []
        Net = rn.Random_Network(N, K)
        ####### INITIAL CONDITIONS ###########
        Net.nodes = np.ones((N*number_of_clusters,1))
        #Net.nodes[np.random.randint(N)] = 1
        ########################################
       
        for i in range(time):
            
            rn.evolution(Net,iterations=1, p = 0.1*j)
            if not A:
                Net.nodes[Net.control_node] = 0
            activity.append(rn.activity(Net, N,number_of_clusters=number_of_clusters))
        
        if A:
            plt.plot(activity,label=label,c=(1-j*0.3,0,j*0.3))
        else:
            plt.plot(activity,linestyle = "--",label=label,c=(1-j*0.3,0,j*0.3))
plt.legend()
plt.xlabel("t")
plt.title("Average activity with/without control nodes. 1 cluster")
plt.ylabel("average activity")
plt.ylim(0,2)
plt.savefig("1clustervscontrolnodes.png")
# a = [i*0.01 for i in range(steps)]
# s = pd.Series(a)
# df = pd.DataFrame()
# df[0] = a
# df[1] = pd.DataFrame(np.array(mean_activities))
# df.to_csv(PATH+"/data/"+label+".dat",sep = " ",decimal=".",index=False,header=False)

#%%  ################################### 3 CLUSTERS ################################
import random_network as rn
import pylab as plt
import networkx as nx
N = 20
K = 2
number_of_clusters = 2
time = 200
graphs = [rn.Random_Network(N, K) for i in range(number_of_clusters)]
control_nodes = [graphs[i].control_nodes[0] for i in range(number_of_clusters) ]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters=number_of_clusters)
Net = rn.Network(tot,number_of_clusters)
######### INITIAL CONDITIONS ################
#Net.nodes = np.ones((N*number_of_clusters,1))

##### INITIAL ACTIVE NODES PER CLUSTER###############
# for i in range(2):
#     nod = [np.random.randint(N*i,N*(i+1))  for i in range(number_of_clusters)]
#     Net.nodes[nod] = 1
for i in range(N*(number_of_clusters-2)):
    Net.nodes[i] = 1
#############################################

activity = []
for i in range(time):
    rn.evolution(Net,iterations=1,p=0.05)
    activity.append(rn.activity(Net,N,number_of_clusters=3))
plt.plot(np.array(activity))

plt.xlabel("t")
plt.ylabel("average activity")
plt.ylim(0,2)
plt.title("3 clusters temporal evolution")
#plt.savefig("3clusters.png")
################################################# 3 CLUSTERS ##############################################
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
N = 30
number_of_clusters = 3
graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
Net = rn.Network(tot)

out = []

for i in range(200):
    graphs = [rn.Random_Network(N, 2) for i in range(number_of_clusters)]
    single_cluster_control_nodes = [rn.find_control_nodes(graphs[i],N) for i in range(number_of_clusters)]
    control_nodes = [single_cluster_control_nodes[i] + i*N for i in range(number_of_clusters)]
    tot = rn.create_clusters(graphs, control_nodes, N,number_of_clusters)
    Net = rn.Network(tot)

    Net.nodes = np.ones((N*number_of_clusters,1)) 

    rn.evolution(Net,iterations = 50,p=0.2)
    out.append(rn.activity(Net,N,number_of_clusters=number_of_clusters))
   # print(rn.activity(Net,N))
        
    
plt.hist(np.array(out))
#plt.xlim(0,1)

#%% ###########LATEX GRAPHSSSSSSSSSSSSSSSSS #################################S
import pylab as plt
import pandas as pd
import numpy as np

p = np.linspace(-5,5,num=1000)
a = 1.6
b = 0.29
deltap = 0.1
V = 2 + (p-deltap)**4 + b*(p-deltap)**3 - a*(p-deltap)**3 - a*b*(p-deltap)**2

k = np.exp(-V)*100
plt.plot(p,k)
a = [i*0.001 for i in range(len(k))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(k))
df.to_csv("tesi/data/histo.dat",sep = " ",decimal=".",index=False,header=False)
#plt.ylim(0,40)
#np.linspace(0,100,num=1000)
#%%
import random_network as rn
import pylab as plt
import networkx as nx
N = 5
K = 2
number_of_clusters = 2

######################################

#creation of the subnetworks
gr = [rn.Random_Network(N,K) for i in range(number_of_clusters)]

control_nodes = [gr[i].control_node+i*N for i in range(number_of_clusters)]
print(control_nodes)
tot = rn.create_clusters(gr, control_nodes, N,number_of_clusters,visual=False)
negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))
#print(negedges)
Net = rn.Network(tot,number_of_clusters)
graph = nx.from_numpy_matrix(tot.T, create_using=nx.DiGraph)
npos = nx.spring_layout(graph)
#cycles = nx.cycle_basis(graph.to_undirected())

################## ONLY FOR VISUALIZATION #######################
tot1 = rn.create_clusters(gr, control_nodes, N,number_of_clusters,visual=True)
abs_tot = abs(tot1)
graph1 = nx.from_numpy_matrix(abs_tot.T, create_using=nx.DiGraph)
npos = nx.kamada_kawai_layout(graph1)
#################################################################

nx.draw_networkx(graph,npos, with_labels= True)
nx.draw_networkx_nodes(graph,npos,
                    nodelist=control_nodes,
                    node_size=800,node_color="green")
nx.draw_networkx_edges(graph, npos,
                   edgelist=negedges,
                   width=3, alpha=0.4, edge_color='r')
control_nodes = [graphs[i].control_node+i*N for i in range(number_of_clusters)]
negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))
print(negedges)
print(control_nodes)
#plt.savefig("network.png")
#%% ############################### NUMBER OF LOOPS ##########################################

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt

mean_loops = []

for N in range(5,50):
    loops = []
    for i in range(10):
        g = rn.Random_Network(N,2)
        graph = nx.from_numpy_matrix(g.adj_matrix, create_using=nx.DiGraph)
        cycles = nx.simple_cycles(graph)
        #nx.draw_networkx(graph,with_labels=True)
        loops.append(len(list(cycles)))
    mean_loops.append(np.mean(np.array(loops)))
plt.plot(mean_loops)
a = [i for i in range(len(mean_loops))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(mean_loops))
df.to_csv("tesi/data/loops.dat",sep = " ",decimal=".",index=False,header=False)
#%%  ######################## NUMBER OF LOOPS PER NUMBER OF NODES ###################################################

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd

#N = 10
K = 2
mean_loops = []
for N in range(10,60):
    loops = []
    for i in range(10):
        g = rn.Random_Network(N,K)
        loops.append(g.loops)
    mean_loops.append(np.log(np.mean(np.array(loops))))
    
plt.plot(mean_loops)
plt.title("Number of loops per number of nodes")
plt.savefig(".png")

a = [i for i in range(len(mean_loops))]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame(np.array(mean_loops))
df.to_csv("tesi/data/numberofloops.dat",sep = " ",decimal=".",index=False,header=False)

#%% ################### MEAN ACTIVITY WITH AND WITHOUT CONTROL NODE #####################################À

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd


K = 2
N = 20
steps = 300
realizations = 50
activities = []
mean_activities0 = []
mean_activities1 = []

g = [rn.Random_Network(N,K) for i in range(realizations)]
for i in range(realizations):
    g[i].nodes = np.ones((N,1))
    
    
for i in range(steps):
    for j in range(realizations):
        activities.append(rn.activity(g[j],N))
        rn.evolution(g[j],iterations= 1, p=0.3)
    mean_activities0.append(np.mean(np.array(activities)))
        
  
g = [rn.Random_Network(N,K) for i in range(realizations)]
activities = []
for i in range(realizations):
    g[i].nodes = np.ones((N,1))
    g[i].nodes[g[i].control_nodes[0]] = 0
    
for i in range(steps):
    for j in range(realizations):
        activities.append(rn.activity(g[j],N))
        g[j].nodes[g[j].control_nodes[0]] = 0 
        
        rn.evolution(g[j],iterations= 1, p=0.3)
    mean_activities1.append(np.mean(np.array(activities)))
    
    
plt.plot(mean_activities0)
plt.plot(mean_activities1)
plt.ylim(0,1)
plt.xlim(0,steps)
plt.savefig("averageactivitywithandwithoutcontrolnodes1.png")
#%%
import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd


K = 2
N = 20

steps = 300
realizations = 20
mean_activities = pd.DataFrame()

for i in range(realizations):
    graphs = [rn.Random_Network(N, K) for i in range(2)]
    control_nodes = [graphs[k].control_nodes[0]+k*N for k in range(number_of_clusters) ]
    env_control_nodes = [graphs[k].control_nodes[1]+k*N for k in range(number_of_clusters)]
    
    tot = rn.create_net(graphs, control_nodes,env_control_nodes, N,M)
    
    Net = rn.Network(tot,number_of_clusters=2)
    for k in range(N):
        Net.nodes[k] = 1
    
    activity = []
    
    for j in range(1,steps):
        rn.evolution(Net,iterations=1,p=0.2)
        rn.env(Net,env_control_nodes,p=0.2)    
        act = rn.activity(Net,N,N,number_of_clusters=2)
        activity.append(act)
        
    mean_activities[i] = activity
    plt.plot(np.array(activity))
    
#%% ########################  FREQUENCY OF LOOPS FOR THE SINGLE CONTROL NODE ############################

import numpy as np
import networkx as nx
import random_network as rn
import pylab as plt
import pandas as pd


K = 2
mean_percentage = []
for N in range(10,80):
    percentage = []
    for i in range(20):
        g = rn.Random_Network(N,K)
        loopspercn = np.array(g.loops_per_cn)
        frequency = loopspercn/g.loops
        percentage.append(frequency[g.control_nodes[0]])  
    mean_percentage.append(np.mean(np.array(percentage)))
#nx.draw_networkx(nx.from_numpy_matrix(g.adj_matrix.T,create_using=nx.DiGraph), with_labels=True)
plt.plot(range(10,80),mean_percentage)
plt.ylim(0,1)
plt.title("frequency of loops per control nodes")
#plt.savefig("frequency of loops per control nodes.png")

# a = [i for i in range(len(mean_loops))]
# df = pd.DataFrame()
# df[0] = a
# df[1] = pd.DataFrame(np.array(mean_loops))
# df.to_csv("tesi/data/loopspercn.dat",sep = " ",decimal=".",index=False,header=False)


#%%  ###################################   HISTOGRAM TIMES ################################
import random_network as rn
import pylab as plt
import networkx as nx
import pandas as pd

realizations = 10
noise = 0.25
N = 10
M = 10
K = 2
number_of_clusters = 2
time = 4000

mean_times = []
mean_times1 = []

for k in range(100):
    
    try:
        graphs = [rn.Random_Network(N, K)]
        graphs.append(rn.Random_Network(M,K))
        #graphs = [rn.Random_Network(N,K) for i in range(number_of_clusters)]
        control_nodes = [graphs[i].control_nodes[0]+i*N for i in range(number_of_clusters) ]
        env_control_nodes = [graphs[i].control_nodes[1]+i*N for i in range(number_of_clusters)]
        
        tot = rn.create_net(graphs, control_nodes,env_control_nodes, N,M)
        #tot = rn.create_clusters(graphs,control_nodes,env_control_nodes, N,number_of_clusters,visual=True)
        #print(tot)
        Net = rn.Network(tot,number_of_clusters)
    except: 
        print("noooooooooooooo")
        pass
    
    for j in range(realizations):
        t = 0
        times = []
        times1 = []
        activity = []
        mean_activities = []
        Net.nodes = np.zeros((N+M,1))
        # for i in range(N):
        #     Net.nodes[i] = 1
        
        Net.nodes[np.random.randint(M + N)] = 1
            
        for i in range(1,time):
            rn.evolution(Net,iterations=1,p=noise)
            rn.env(Net,env_control_nodes,p=0.01)    
            act = rn.activity(Net,N,M,number_of_clusters=number_of_clusters)
            activity.append(act)
            if act[0]>0.8 and act[1] < 0.2:
                times.append(np.log(i-t))
                t = i
            if act[1]> 0.8 and act[0] <0.2:
                times.append(np.log(i-t))
                t = i
            
        #mean_activities.append(np.mean(activity))
        mean_times.append(np.mean(np.array(times)))
        #mean_times1.append(np.mean(np.array(times1)))
plt.hist(mean_times,bins=25)
#plt.hist(mean_times1,bins=25,alpha=0.5)
plt.title("sdistribution of log of time transitions-N="+str(N)+" M="+str(M)+"-noise="+str(noise))
plt.savefig("sshisto-N="+str(N)+" M="+str(M)+"-noise="+str(noise)+".png")
   # plt.figure()

    # negedges = list(zip(list(np.where(tot.T<0)[0]),list(np.where(tot.T<0)[1])))
    # print(negedges)
    # print(control_nodes)
    # plt.plot(np.array(activity))
    # plt.xlabel("t")
    # plt.ylabel("average activity")
    # plt.ylim(0,2)
    # plt.title("N="+str(N)+" M="+str(M))
    # plt.savefig("N="+str(N)+" M="+str(M)+".png")
#print(act)


#################################### TIMES HISTOGRAM ##########################################
# df = pd.DataFrame()
# df[0] = np.histogram(times)[1][1:]
# df[1] = np.histogram(times)[0]
# df.to_csv("tesi/data/times.dat",sep = " ",decimal=".",index=False,header=False)
###############################################################################################
#%% ##############  BETWEENNESS CENTRALITY ####################################################

import random_network as rn
import networkx as nx
import numpy as np
import pylab as plt
from functools import reduce


K = 2
realizations = 100
mean_bc0 = []
mean_bc1 = []
mean_max = []
#mean_percentage = []
for N in range(10,30):
    bc0 = []
    bc1 = []
    maxx = []
    #percentage = []
    for j in range(realizations):
        g = rn.Random_Network(N,K)
        net = nx.from_numpy_matrix(g.adj_matrix,create_using=nx.DiGraph)
        #nx.draw_networkx(net,with_labels=True)
        if np.argmax(list(nx.betweenness_centrality(net))) == g.control_nodes[0]:
            bc0.append(1)
        
        if np.argmax(list(nx.betweenness_centrality(net))) == g.control_nodes[1]:
            bc1.append(1)
    
        #maxx.append(max([nx.betweenness_centrality(net)[i] for i in range(N)]))
        # loopspercn = np.array(g.loops_per_cn)
        # frequency = loopspercn/g.loops
        # percentage.append(frequency[g.control_nodes[0]])
    mean_bc0.append((len(bc1)+len(bc0))/realizations)

    #mean_max.append(np.mean(np.array(maxx)))
    #mean_percentage.append(np.mean(np.array(percentage)))

plt.step(range(10,30),mean_bc0)

#plt.ylim(0,1)
plt.title("histogram of control_nodes")
plt.savefig("histogram-betweenness centrality-control_node.png")
#%%  ######################### PERIFERICAL NODES ################################################

import random_network as rn
import networkx as nx
import numpy as np
import pylab as plt
from functools import reduce


K = 3
realizations = 30
mean_bc0 = []
mean_bc1 = []
mean_max = []
periferical = []
A = 5
B = 20
for N in range(A,B):
    bc0 = []
    bc1 = []
    maxx = []
    periferical_nodes = []

    for j in range(realizations):
        g = rn.Random_Network(N,K)
        net = nx.from_numpy_matrix(g.adj_matrix,create_using=nx.DiGraph)
        
        cycles = nx.simple_cycles(net)
        final = []
        z = list(reduce(lambda x,y: x+y,cycles))
        
        for i in range(N):
            final.append(z.count(i))
             

        periferical_nodes.append((len(np.where(np.array(final) == 0)[0])/N))
        
    periferical.append(np.mean(np.array(periferical_nodes)))
    

plt.plot(range(A,B),periferical)
plt.ylim(0,1)
plt.title("PERIFERICAL NODES ")
plt.savefig("periferical nodes.png")

#%% SWITCHING TIMES ##################################


import random_network as rn
import networkx as nx
import numpy as np
import pylab as plt
from functools import reduce


steps = 4000
N = 10
M = 20 
K = 2
realizations = 100
number_of_clusters = 2

mean_switch = []
mean_no_switch = []
for i in range(realizations):
        graphs = [rn.Random_Network(N, K)]
        graphs.append(rn.Random_Network(M,K))
        control_nodes = [graphs[i].control_nodes[0]+i*N for i in range(number_of_clusters) ]
        env_control_nodes = [graphs[i].control_nodes[1]+i*N for i in range(number_of_clusters)]
        tot = rn.create_net(graphs, control_nodes,env_control_nodes, N,M)
        net = rn.Network(tot,number_of_clusters)
        
        net.nodes[np.random.randint(N+M)] = 1
        switch = 0
        no_switch = 0
        for i in range(1,steps):
            rn.evolution(net,iterations=1,p=0.2)
            rn.env(net,env_control_nodes,p=0.1)
            act = rn.activity(net,N,M,2)
            
            if (act[0] > 0.8 and act[1] < 0.2) or (act[1] > 0.8 and act[0] < 0.2):
                switch += 1
            elif (act[0] < 0.2 and act[1]< 0.2) or (act[0]>0.8 and act[1]> 0.8):
                no_switch +=1
            
        mean_switch.append(switch)
        mean_no_switch.append(no_switch)
 
plt.plot(mean_switch)
plt.plot(mean_no_switch)
#%% ##################### TEMPORAL EVOLUTION ##################
import random_network as rn
import networkx as nx
import numpy as np
import pylab as plt
from functools import reduce
import pandas as pd

steps = 500
N = 20
M = 40 
K = 2
realizations = 100
number_of_clusters = 2

mean_switch = []
mean_no_switch = []

graphs = [rn.Random_Network(N, K)]
graphs.append(rn.Random_Network(M,K))
control_nodes = [graphs[i].control_nodes[0]+i*N for i in range(number_of_clusters) ]
env_control_nodes = [graphs[i].control_nodes[1]+i*N for i in range(number_of_clusters)]
tot = rn.create_net(graphs, control_nodes,env_control_nodes, N,M)
Net = rn.Network(tot,number_of_clusters)

Net.nodes[np.random.randint(N)] = 1
    
activity = []
for i in range(1,steps):
    rn.evolution(Net,iterations=1,p=0.2)
    rn.env(Net,env_control_nodes,p=0.1)    
    act = rn.activity(Net,N,M,number_of_clusters=number_of_clusters)
    activity.append(act)

plt.plot(activity)
plt.ylim(0,2)

a = [i for i in range(steps)]
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame([activity[i][0] for i in range(steps-1)])
#df.to_csv("tesi/data/temps.dat",sep = " ",decimal=".",index=False,header=False)
df = pd.DataFrame()
df[0] = a
df[1] = pd.DataFrame([activity[i][1] for i in range(steps-1)])
#df.to_csv("tesi/data/temps2.dat",sep = " ",decimal=".",index=False,header=False)

