import networkx as nx
import pylab as plt

g = nx.DiGraph()
pos = nx.layout.spring_layout(g)

g.add_edge(1,2)
g.add_edge(2,3)
g.add_edge(3,4)
g.add_edge(4,5)
g.add_edge(5,3)
g.add_edge(6,5)

pos = nx.layout.spring_layout(g)

nx.draw(g,pos = pos)
nx.draw_networkx_labels(g,pos = pos)
nx.draw_networkx_nodes(g,pos,
                       nodelist=[6],node_color='y')

plt.savefig("env.png")
