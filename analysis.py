import pylab as plt
import numpy as np
import networkx as nx
import  random_network  as rn


realizations = 100
steps = 100
N = 10
mean_activities = []

######################### PARAMETRIC NOISE ######################
for i in range(steps):
    activities = []
    graphs = [rn.Random_Network(N, 2) for i in range(realizations)]

    for j in range(realizations):
        rn.initial_conditions(graphs[j], N)
        rn.evolution(graphs[j],iterations=20,p=(0.01*i),p_noise=True)
        activities.append(rn.activity(graphs[j], N))
    mean_activities.append(np.mean((np.array(activities))))

probabilities = [i*0.01 for i in range(steps)]

plt.plot(probabilities,mean_activities)
plt.xlabel("noise")
plt.ylabel("mean activity")
# plt.savefig("noise-activity.png")















#%%
# n = 10

# for K in range(1,3):
#     means = []
#     totmeans = []
#     for N in range(1,n):
        
#         for i in range(30):
#             g = rn.Random_Network(N, K)
#             g.nodes[np.random.randint(N)] = 1
            
#             #evolution
#             for t in range(N):
#                 up = np.zeros((N,1))
#                 #NOISE
#                 # for i in range(N):
#                 #  p = 1
#                 #  if np.random.uniform(0,1)>p:
#                 #      g.nodes[i] = 1
             
                
#                 up = g.adj_matrix.dot(g.nodes)
#                 g.nodes = (up >0).astype(int)
                   
#             means.append(np.mean(g.nodes.T))
#             #g.nodes = np.zeros((N,1))
#         totmeans.append(np.mean(means))
       
    
#     # plt.ylim(0,1.2)
#     # plt.plot(totmeans, label = "mean of incoming links K = " + str(K) )
#     # plt.ylabel("active links")
#     # plt.xlabel("number of nodes N")
#     # plt.legend()    
#     # plt.savefig("active_links.png")
# # print(g.nodes.T)
# #plt.imshow(g.adj_matrix)
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
# K = 1
# max_nodes = 100
# iterations = 10

# fig, ax = plt.subplots(1,2)
# outgoing_links = []
# number_of_loops = []
# mean_number_of_loops = []
# mean_outgoing_links = []
# for i in range(2,max_nodes):
#     for j in range(iterations): 
#         g = rn.Random_Network(i, K)
#         outgoing_links.append(np.mean(sum(g.adj_matrix)))
#         graph = nx.from_numpy_matrix(g.adj_matrix.T, create_using=nx.DiGraph)
#         cycles = nx.cycle_basis(graph.to_undirected())
#         number_of_loops.append(len(cycles))
        
#     mean_outgoing_links.append(np.mean(outgoing_links))
#     mean_number_of_loops.append(np.mean(number_of_loops))


# x = np.linspace(0,max_nodes,num = max_nodes)
# y = np.ones(max_nodes)*K

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