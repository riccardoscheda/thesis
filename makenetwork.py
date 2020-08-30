
import networkx as nx
import random_network as rn
import numpy as np
from network2tikz import plot
import network2tikz

PATH = "tesi/"
N = 10
K = 1
graph = rn.Random_Network(N, K)

nodes = [str(i) for i in range(N)]
graph.edges
edges = graph.edges
gender = ['w' for i in range(N)]
colors = {'g': 'gray', 'w': 'white'}
g = nx.DiGraph(edges)
style = {}
style["vertex_shape"] = "circle"
style['vertex_size'] = .4
style["edge_width"] = 0.1
style['node_label'] = nodes
style['node_label_size'] = 6
style['node_color'] = [colors[g] for g in gender]
#style['node_opacity'] = .5
style['edge_curved'] = .0


#pos = nx.layout.spring_layout(g,k=5/np.sqrt(g.order()))
pos = nx.kamada_kawai_layout(g)
layout = nx.layout.spring_layout(g,pos = pos,iterations = 0)

plot((g),PATH + 'prova.tex',layout= layout,**style)

import os

os.system('pdflatex -output-directory="tesi" tesi/prova.tex')
