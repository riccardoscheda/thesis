import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn
import pandas as pd

PATH = "tesi"
#%%  ################################ MEAN ACTIVITY VERSUS NOISE #####################À


realizations = 100
steps = 100
N = 40
K = 2

for N in [20,40]:
    for s in [True]:
        mean_activities = []
        probabilities = [i*0.01 for i in range(steps)]
        if s:
            label = "noisy links"
        else:
            label = "noisy nodes"
            
            
        for i in range(steps):
            activities = []
            graphs = [rn.Random_Network(N, K) for i in range(realizations)]
        
            for j in range(realizations):
                #graphs = [rn.Random_Network(N, 2) for i in range(realizations)]
                rn.initial_conditions(graphs[j], N)
                rn.evolution(graphs[j],iterations=N*2,p=(0.01*i),p_noise=s)
                activities.append(rn.activity(graphs[j], N))
            mean_activities.append(np.mean((np.array(activities))))
        
        
        plt.plot(probabilities,mean_activities,label=label)
        plt.legend()
        plt.xlabel("noise")
        plt.ylabel("mean activity")
        plt.savefig(PATH + "activity.png")
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