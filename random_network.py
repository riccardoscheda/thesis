import numpy as np
import pandas as pd
import random
import networkx as nx
from functools import reduce
import os


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
       
        self.edges = [(str(a),str(b)) for a,b in zip(np.where(self.adj_matrix == 1)[0],np.where(self.adj_matrix == 1)[1])]
        self.control_node = find_control_nodes(self, self.n)
        #self.control_node = outgoing_links(self, self.n)
        
class Network:
    def __init__(self, matrix):

        self.adj_matrix = matrix
        self.nodes = np.zeros((len(self.adj_matrix),1))
        self.edges = [(str(a),str(b)) for a,b in zip(np.where(self.adj_matrix == 1)[0],np.where(self.adj_matrix == 1)[1])]
        #self.control_node = find_control_nodes(self, len(self.adj_matrix))
        #self.control_node = outgoing_links(self, len(self.adj_matrix))

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
    cycles = nx.simple_cycles(graph)
    #print("cycles: " + str(list(cycles)))
    #driver_node = list(reduce(lambda x,y: set(x)&set(y),cycles))
    cycles = nx.simple_cycles(graph)
    final = []
    z = list(reduce(lambda x,y: x+y,cycles))

    for i in range(N):
        final.append(z.count(i))
         
    #print(final)
    control_node = np.argmax(final)
    # print("driver node: "+ str(driver_node))
    # print(control_node)
    return control_node
def outgoing_links(gr,N):
    """
    
    Finds the node with the max number of outgoing links.
    Parameters
    ----------
    gr : Random Network graph
        
    N : int
        number of nodes of the network

    Returns
    -------
    the index of the node with max number of ougoing links.
    """
    outgoing_links = []
    for i in range(N):
        outgoing_links.append((sum(gr.adj_matrix.T)))
        
    return np.argmax(outgoing_links)


def create_clusters(graphs,control_nodes, N,number_of_clusters=1):
    """
    Builds a network made of different clusters. These Clusters are linked with negative weights (-1).
    The link are between the control nodes of each cluster.
    -----------------------------------------------------
    Parameters:
        graphs: list of Random Network graphs.
        control_nodes: list of indeces of the control nodes.
        N: number of nodes for each cluster.
        number_of_clusters: int, default 1, number of clusters of the network.
    -------------------------------------------------------
    Returns:
        numpy matrix which is the connectivity matrix of the network.
    """

    tot = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tot[i][j] = graphs[0].adj_matrix[i][j]
    
    if number_of_clusters>1:
        for i in range(number_of_clusters-1):
            neg1 = np.zeros((N*(i+1),N))
            neg2 = np.zeros((N,N*(i+1)))
            
            tot = np.block([[tot,       neg1        ],
                            [neg2, graphs[i].adj_matrix              ]])
    
        for j in range(number_of_clusters):
             tot[control_nodes[-j]][control_nodes[-j-1]] = -100
             tot[control_nodes[-j-1]][control_nodes[-j]] = -100
             
    return tot

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
    
 
def noise(graph,p = 0.):
    """
    Gives a probability for a node to be turned off.
    -----------------------------------------------
    Parameters:
        graph: Random Network graph
        p: float, default 0, gives the probability for the node to be turned off.
        
    """
    
    for i in range(len(graph.nodes)):
        if np.random.uniform(0,1)<p:
            #print("ok")
            graph.nodes[i] = 0


def parametric_noise(graph, p=1):
    """
    Gives a probability for a link to be turned off.
    -----------------------------------------------
    Parameters:
        graph: Random Network graph
        p: float, default 0, gives the probability for the node to be turned off.
    -----------------------------------------------    
    Returns:
        noisy_adj_matrix: numpy matrix with noise
        
    """
    noisy_adj_matrix = np.zeros((len(graph.nodes),len(graph.nodes)))
    for i in range(len(graph.nodes)):
        for j in range(len(graph.nodes)):
            noisy_adj_matrix[i][j] = graph.adj_matrix[i][j]
            
            
    for i,j in zip(np.where(graph.adj_matrix==1)[0],np.where(graph.adj_matrix==1)[1]):
        if np.random.uniform(0,1)<p:
            noisy_adj_matrix[i][j] = 0 
       
    return noisy_adj_matrix



def initial_conditions(graph,N):
    """
    Initialize the graph turning on the control node of the graph.
    -------------------------------------------------
    Parameters:
        graph: Random Network graph
        N: int, number of nodes
    """
    control_nodes = find_control_nodes(graph, N)
    graph.nodes = np.zeros((N,1))
    #graph.nodes = np.ones((N,1))
    graph.nodes[control_nodes] = 1
 
    
    
def evolution(graph,iterations = 10,p=0,p_noise=False):
    """
    Dynamical evolution of the network.
    -------------------------------------------------
    
    Parameters:
        graph: Random Network graph
        iterations: default = 10 int of iterations for which the evolution of the network
        p: float, probability to turn off
        p_noise: Bool, default False, to choose the type of noise
        
    """
    if p_noise:
        for i in range(iterations):
            noisy_adj_matrix = parametric_noise(graph,p)
            next_state = noisy_adj_matrix.dot(graph.nodes)
            graph.nodes = (next_state >0).astype(int)
                
        
    else:
        for i in range(iterations):
            next_state = graph.adj_matrix.dot(graph.nodes)
            graph.nodes = (next_state >0).astype(int)
            noise(graph,p)
            
            
def to_latex(data,file ="data.dat",axis_factor=1., xmin=0,xmax=100,ymin=0,ymax=1,xlabel="N",ylabel="Activity",path="./"):
    """
    Makes directly the pdf of a plot of a list of data.
    
    """
    a = [i*axis_factor for i in range(len(data))]
    df = pd.DataFrame()
    df[0] = a
    df[1] = pd.DataFrame(np.array(data))
    df.to_csv(file,sep = " ",decimal=".",index=False,header=False)
    latex = r"""\documentclass{standalone}
\usepackage[utf8x]{inputenc}
\usepackage{pgfplots}
\usepackage{tikz}
\usepackage{pdfpages}
\usepackage{standalone}
\usepackage{placeins}
\usepackage{float}
\usepackage{subfigure}
\usepackage{graphicx}
\begin{document}
\centering
\begin{tikzpicture}[scale=0.7]
\centering
\begin{axis}\addplot[thick,blue]
file {"""+file+"""};
\end{axis}
\end{tikzpicture}
\end{document}
"""

    out_file = open("graph.tex","w+")
    out_file.write(latex)
    out_file.close()
    
    os.system('pdflatex -output-directory="tesi/images" graph.tex')

    
#[thick,blue,xmin="""+str(xmin)+""",xmax="""+str(xmax)+""",xlabel=$"""+str(xlabel)+"""$ ,ylabel=$"""+str(ylabel)+"""$,ymin="""+str(ymin)+""",ymax="""+str(ymax)+""",grid=major]
