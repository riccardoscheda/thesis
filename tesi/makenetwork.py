
import networkx as nx
import random_network as rn
N = 10 
K = 1
graph = rn.Random_Network(N, K)

nodes = [str(i) for i in range(N)]


edges = [('0','2')]
gender = ['w','w','w','w','w','w','w','w','w','w']
colors = {'g': 'gray', 'w': 'white'}
g = nx.DiGraph(edges)
style = {}
style["vertex_shape"] = "circle"
style['vertex_size'] = .4
style["edge_width"] = 0.1
style['node_label'] = ['10','0','12','11','13','1','2','4','15','3','14','5','9','8','6','7']
style['node_label_size'] = 6
style['node_color'] = [colors[g] for g in gender]
#style['node_opacity'] = .5
style['edge_curved'] = .0

from network2tikz import plot
import network2tikz

pos = {'10':(-0.9,0.9),'0':(-0.9,0.6),'11':(-1.1,0.4),'12':(-0.7,0.4),'1':(-0.6,0.1),'13':(-1.2,0.1),'2':(0.7,0.9),'4':(0.7,0.6),'15':(0.9,0.4),'3':(0.5,0.4),'14':(0.4,0.1),'5':(1,0.1),'9':(-0.8,-0.3),'8':(-0.4,-0.3),'6':(0,-0.3),'7':(0.4,-0.3)}
layout = nx.layout.spring_layout(g,pos = pos,iterations = 0)

plot((g),'fg5.tex',layout= layout,**style)
