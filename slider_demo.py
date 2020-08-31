"""
===========
Slider Demo
===========

Using the slider widget to control visual properties of your plot.

In this example, a slider is used to choose the frequency of a sine
wave. You can control many continuously-varying properties of your plot in
this way.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import networkx as nx

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

f0 = 1
delta_f = 1

g = nx.random_k_out_graph(n=f0,k=2,alpha=0.5)

axcolor = 'lightgray'
axfreq = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'noise', 1, 20, valinit=f0, valstep=delta_f,valfmt='%0.0f')


def update(val):
    plt.cla()
 
    freq = sfreq.val
    g = nx.random_k_out_graph(n=int(freq),k=2,alpha=0.5)
    nx.draw_networkx(g,with_labels=True)
    ax.margins(x=100)


    fig.canvas.draw_idle()


sfreq.on_changed(update)

resetax = plt.axes([0.15, 0.15, 0.8, 0.8])
button = Button(resetax, '', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
 
    plt.cla()
button.on_clicked(reset)



plt.show()
