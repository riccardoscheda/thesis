
import networkx as nx

nodes = ['1','2','3','4']
edges = [('1','2'),('1','1'),('2','1'),('2','2'),('2','4'),('1','3'),('3','1'),('3','3'),('4','2'),('4','4')]
colors = {'g': 'gray', 'w': 'white'}
g = nx.Graph(edges)
style = {}
#style["vertex_shape"] = "triangle"
style['vertex_size'] = .4
style["edge_width"] = 0.5
style['node_label'] = ['A1','A2','A3','A4']
style['node_label_size'] = 6
#style['node_opacity'] = .5
style['edge_curved'] = .0

from network2tikz import plot
import network2tikz


layout = nx.layout.circular_layout(g)

plot((g),'fg5.tex',layout= layout,**style)
