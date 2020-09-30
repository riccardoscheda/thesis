import numpy as np
import pandas as pd
import scipy
import pylab as plt
from scipy.optimize import curve_fit


df = pd.DataFrame(pd.read_csv("data/histotimes.dat",sep=" ",names=["x","y"]))

plt.plot(df["x"],df["y"])
#plt.ylim(0,1.5)


#popt, pcov = curve_fit(func, df["x"], df["y"],p0=[0.001,0.001,2,2],bounds=(-10,10))
p = np.poly1d(np.polyfit(df["x"], df["y"], 4))
plt.plot(df["x"],p(df["x"]))
#plt.plot(df["x"], func(df["x"], *popt), '--', label="Fitted Curve")


df["y"] = p(df["x"])
df.to_csv("data/fittimes0.dat",sep = " ",decimal=".",index=False,header=False)

