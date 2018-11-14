import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv
from numpy import linalg as LA

class Fisher:
    
    def __init__(self, d, x1_train, x2_train):
        
        self.d = d # dimension of data points
        self.x1_train = x1_train # dataset of the classes
        self.x2_train = x2_train
        self.mean_1 = np.mean(x1_train, axis = 0) # means of the classes
        self.mean_2 = np.mean(x2_train, axis = 0)
        self.fisher_vector = np.zeros(d) # fisher vector direction
        self.N_1 = int(np.size(x1_train) / np.size(x1_train[0])) # number of data points in the classes
        self.N_2 = int(np.size(x2_train) / np.size(x2_train[0]))
        
        
    # returns the scatter matrix of the class    
    def scatter_mat(self, Class):
        
        if Class == 1:
            x_train = self.x1_train
        else:
            x_train = self.x2_train
            
        d = self.d
        total_size = int(np.size(x_train) / np.size(x_train[0]))
        mean = np.mean(x_train, axis = 0)
        
        scatter_mat = np.zeros((d, d))
        for i in range(total_size):
            scatter_mat = scatter_mat + np.outer(x_train[i] - mean, x_train[i] - mean)
               
        return scatter_mat
    
    
    # finds the optimal fisher direction
    def fit(self):
        
        scatter_1 = self.scatter_mat(1)
        scatter_2 = self.scatter_mat(2)
        
        S_w = scatter_1 + scatter_2
        
        vec = inv(S_w).dot(self.mean_1 - self.mean_2)
        vec = vec / LA.norm(vec) # normalized fisher vector direction
        self.fisher_vector = vec
    
    
    # gives fisher representation of the data point
    def F_vec(self, x_vec):
        return np.dot(self.fisher_vector, x_vec)
    
    
    # gives the fisher dataset
    def F_transform(self, X):
        
        F_data = []
        for x_vec in X:
            vec = self.F_vec(x_vec)
            F_data.append(vec)
        
        F_data = np.array(F_data)
        
        return F_data
    
    
        