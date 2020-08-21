import numpy as np
import random


class Random_Network:
    def __init__(self, N, K):
        self.n = N
        self.k = K
        self.nodes = np.zeros((self.n,1))
        self.activity = np.sum(self.nodes)
    
        self.adj_matrix = np.zeros((self.n,self.n))
        if self.k == 1:
            while True:
                self.adj_matrix = np.identity(self.n)
                np.random.shuffle(self.adj_matrix)
                self.adj_matrix[np.random.randint(self.n)][np.random.randint(self.n)] = 1
                if np.trace(self.adj_matrix) == 0:
                    break
        else:
        
            for i in range(self.n):
                for j in range(self.k):
                    numbers = list(range(0,i)) + list(range(i+1,self.n))
                    r = random.choice(numbers)
                    self.adj_matrix[i][r] = 1
                    
        def activity(self):
            return np.mean(self.nodes)
                    
    
            
class Network:
    def __init__(self, matrix):

        self.adj_matrix = matrix
        self.nodes = np.zeros((len(self.adj_matrix),1))

    def activity(self):
            return np.mean(self.nodes)
                    
    