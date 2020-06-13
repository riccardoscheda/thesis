import numpy as np

class Random_Network:
    def __init__(self, N, K):
        self.n = N
        self.k = K
        self.nodes = np.zeros((self.n,1))

        self.adj_matrix = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.k):
                self.adj_matrix[i][np.random.randint(self.n)] = 1
        
