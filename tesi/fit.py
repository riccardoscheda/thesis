import numpy as np
import pandas as pd
import scipy
import pylab as plt
from scipy.optimize import curve_fit


df = pd.DataFrame(pd.read_csv("data/meanactivity10.3.dat",sep=" ",names=["x","y"]))

plt.plot(df["x"],df["y"])
plt.ylim(0,1.5)

def func(x,a,b,c):
    return  a*np.exp(-b*x) + c

popt, pcov = curve_fit(func, df["x"], df["y"])

plt.plot(df["x"], func(df["x"], *popt), '--', label="Fitted Curve")

df["y"] =  func(df["x"], *popt)
df.to_csv("data/fit3.dat",sep = " ",decimal=".",index=False,header=False)

