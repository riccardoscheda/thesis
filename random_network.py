import numpy as np
import random
import networkx as nx
from functools import reduce



class Random_Network:
    def __init__(self, N, K):
        self.n = N
        self.k = K
        self.nodes = np.zeros((self.n,1))
        self.activity = np.sum(self.nodes)
    
        self.adj_matrix = np.zeros((self.n,self.n))
        if self.k == 1:
            while True:
                self.adj_matrix = np.identity(self.n)
                np.random.shuffle(self.adj_matrix)
                self.adj_matrix[np.random.randint(self.n)][np.random.randint(self.n)] = 1
                if np.trace(self.adj_matrix) == 0:
                    break
        else:
        
            for i in range(self.n):
                for j in range(self.k):
                    numbers = list(range(0,i)) + list(range(i+1,self.n))
                    r = random.choice(numbers)
                    self.adj_matrix[i][r] = 1
                    
        def activity(self):
            return np.mean(self.nodes)
                    
    
            
class Network:
    def __init__(self, matrix):

        self.adj_matrix = matrix
        self.nodes = np.zeros((len(self.adj_matrix),1))

    def activity(self):
            return np.mean(self.nodes)
                    
    
def find_control_nodes(gr,N):
    """
    Finds the nodes with max connectivity in a graph
    ---------------------------
    Parameters:
        gr: a Random Network graph
    ---------------------------
    Returns:
        control_node: int which is the index of the control node
    """
    graph = nx.from_numpy_matrix(gr.adj_matrix.T, create_using=nx.DiGraph)
    npos = nx.layout.spring_layout(graph)
    cycles = nx.cycle_basis(graph.to_undirected())
   
    driver_node = list(reduce(lambda x,y: set(x)&set(y),cycles))
    
    final = []
    z = list(reduce(lambda x,y: x+y,cycles))

    for i in range(N):
        final.append(z.count(i))
         
    control_node = np.argmax(final)
     
    # print("cycles: " + str(cycles))
    # print("driver node: "+ str(driver_node))
    # print(control_node)
    
    return control_node



def activity(graph,N,number_of_clusters=1):
    """
    Measures the activity of each cluster in the network
    
    --------------------------------
    Parameters:
        graph: a Random Network graph
        N: int, number_of_clustersber of nodes for each cluster
        number_of_clusters: the number_of_clustersber of clusters in the network
    ---------------------------------
    Returns:
        list with the mean activity of the clusters
    """
    activity = []

    
    for j in range(number_of_clusters):
        cluster = [graph.nodes[k] for k in range(N*j,N*(j+1)) ]
        activity.append(np.mean(cluster))
        
    return activity
    
 
def noise(graph, p = 1.):
    ##### ATTENZIONE NOISE A ZERO #####
    for i in range(len(graph.nodes)):
        if np.random.uniform(0,1)>p:
            #print("ok")
            graph.nodes[i] = 0
        else: 
            pass
    
def evolution(graph,iterations = 10,p=1):
    """
    Dynamical evolution of the network.
    -------------------------------------------------
    
    Parameters:
        graph: Random Network graph
        iterations: default = 10 int of iterations for which the evolution of the network
        
    """
    for i in range(iterations):
        next_state = graph.adj_matrix.dot(graph.nodes)
        graph.nodes = (next_state >0).astype(int)
        noise(graph,p)