
import networkx as nx
import random_network as rn
import numpy as np
from network2tikz import plot
import network2tikz

PATH = "tesi/"
N = 5
K = 2

nodes = ['1','2','3','4','5','6','7','8','9']
edges = [('6','1'),('1','6'),('1','2'),('1','3'),('1','0'),('4','1'),('0','2'),('1','0'),('0','3'),('3','4'),('6','7'),('6','5'),('6','8'),('8','9'),('9','6'),('5','7'),('5','8'),('8','9')]
# gender = ['w' for i in range(N)]
# colors = {'g': 'gray', 'w': 'white'}

g = nx.DiGraph(edges)
style = {}
style["vertex_shape"] = "circle"
style['vertex_size'] = .9
style["edge_width"] = 0.9
#style["edge_label"] = ["+1","+1","+1","+1","+1"]
#style['node_label'] = nodes
style['node_label_size'] = 6
style['node_color'] = ["orange","orange","white","white","white","white","white","white","white","white"]
style['edge_color'] = ["red","","","","red"]
#style['node_opacity'] = .5
style['edge_curved'] = .0


#pos = nx.layout.spring_layout(g,k=5/np.sqrt(g.order()))
pos = nx.kamada_kawai_layout(g)
layout = nx.layout.spring_layout(g,pos = pos,iterations = 2)

plot((g),PATH + 'doublecluster.tex',layout= layout,**style)

import os

os.system('pdflatex -output-directory="tesi" tesi/prova.tex')

