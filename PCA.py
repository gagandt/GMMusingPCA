import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA

class PCA:
    
    def __init__(self, l, d, x_train, total_size):
        
        self.l = l # reduced dimension
        self.d = d # original dimension
        self.x_train = x_train # dataset for dimensionality reduction
        self.mean = np.mean(x_train, axis = 0) # mean of the dataset
        self.q_matrix = np.zeros((l, d)) # optimal projection directions (transformation matrix)
        self.total_size = total_size # total size of the dataset
     
        
    # returns the cov matrix of the dataset
    def cov_mat(self):
        
        d = self.d
        total_size = self.total_size
        x_train = self.x_train
        mean = self.mean
        
        cov_matrix = np.zeros((d, d))
        for i in range(total_size):
            cov_matrix = cov_matrix + np.outer(x_train[i] - mean, x_train[i] - mean)
        
        cov_matrix = cov_matrix / total_size
        
        return cov_matrix
       
        
    # returns the reduced dimension dataset
    def fit(self):
        
        l = self.l
        d = self.d
        
        cov_matrix = self.cov_mat()
        eig_values, eig_vectors = LA.eig(cov_matrix)
        
        eig_mat = np.concatenate((eig_vectors, eig_values.reshape((d, 1))), axis = 1)
        eig_mat = eig_mat[eig_mat[:,d].argsort()]
        
        y_value = np.flip(eig_mat[:, d], axis = 0)
        print(y_value)
        x_value = np.arange(d)
        print(x_value)
        
        q_matrix = []
        for i in range(0, l):
            index = d - i - 1
            vec = eig_mat[index, :d]
            vec = vec.tolist()
            q_matrix.append(vec)
        
        # set of l significant vectors to which the dataset would be projected
        q_matrix = np.array(q_matrix)
        self.q_matrix = q_matrix
        plt.scatter(x_value, y_value)
        plt.show()


    # returns reduce dimensional vector
    def reduce_vector(self, x_vec):
        y_vec = x_vec - self.mean
        q_matrix = self.q_matrix
        return q_matrix.dot(y_vec)
    
    
    # returns reduced dataset
    def transform(self, X):
        red_X = []
        for x_vec in X:
            reduced_vec = self.reduce_vector(x_vec)
            reduced_vec = reduced_vec.tolist()
            red_X.append(reduced_vec)
        
        red_X = np.array(red_X)
        
        return red_X