
import networkx as nx
import random_network as rn
import numpy as np
from network2tikz import plot
import network2tikz

PATH = "tesi/"
N = 4
K = 1

nodes = ['1','2','3','4','5','6']
edges = [('1','2'),('2','3'),('2','4'),('4','5'),('4','6')]
gender = ['w' for i in range(N)]
colors = {'g': 'gray', 'w': 'white'}
g = nx.DiGraph(edges)
style = {}
style["vertex_shape"] = "circle"
style['vertex_size'] = .9
style["edge_width"] = 0.8
#style["edge_label"] = ["+1","+1","+1","+1","+1"]
#style['node_label'] = nodes
style['node_label_size'] = 6
#style['node_color'] = [colors[g] for g in gender]
#style['node_opacity'] = .5
style['edge_curved'] = .0


#pos = nx.layout.spring_layout(g,k=5/np.sqrt(g.order()))
pos = nx.kamada_kawai_layout(g)
layout = nx.layout.spring_layout(g,pos = pos,iterations = 2)

plot((g),PATH + 'cells.tex',layout= layout,**style)

import os

os.system('pdflatex -output-directory="tesi" tesi/prova.tex')

