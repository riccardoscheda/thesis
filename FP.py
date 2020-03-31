import numpy as np
import pylab as plt


#number of genes
n = 50
#transition matrix
W = np.random.uniform(-1,1,size = (n,n))
#mean lifetime of ecited states
gamma = np.random.uniform(0,1,size = (n))

#probability tha the node is in the state 1
p = np.zeros(n)
p[25] = 1

plt.plot(p)
#laplacian matrix
L = W - gamma*np.identity(n)


for i in range(10):
    p = sum(L)*p - gamma*p
    plt.plot(np.linspace(0,n,num = n),p)
    
  

