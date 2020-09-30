import numpy as np
import pandas as pd
import scipy
import pylab as plt
import os


file = "tesi/data/gif/dist0.dat"

def to_latex(file,i):
    """
    Makes directly the pdf of a plot of a list of data.
    
    """

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
\begin{tikzpicture}
\centering
\begin{axis}[xlabel=$Activity$,xmin=-2,xmax=2,ylabel=\emph{Distribution} ,ymin=0,ymax=0.8]
\addplot[ybar,fill,orange] table [x, y] {"""+file+"""};

\end{axis}
\end{tikzpicture}
\end{document}
"""

    out_file = open("tesi/data/gif/graph"+str(i)+".tex","w+")
    out_file.write(latex)
    out_file.close()
    
    os.system('pdflatex -output-directory="tesi/data/gif" graph'+str(i)+'.tex')

for i in range(100):
    file = "tesi/data/gif/dist"+str(i)+".dat"
    to_latex(file,i)
    #%%
for i in range(100):
    os.system('mv graph'+str(i)+'-1.png graph'+str(i)+'.png')