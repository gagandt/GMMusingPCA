import numpy as np
import random
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import decimal

class GMM:
    k = 0 # number of clusters
    x_train = np.array([]) # training dataset
    mean_vec = np.array([]) # k means
    cov_mat = np.array([]) # k covariance matrix
    mixture_coeff = np.array([]) # mixture coefficients of k clusters
    N_eff = np.array([]) # effective number of data points in k clusters
    vec_dim = 0 # dimension of data points
    total_size = 0 # total size of dataset
    
    
    # initialization of parameter is taken from k-means clustering
    def __init__ (self, k, x_train, mean_vec, cov_mat, mixture_coeff):
        
        self.k = k
        self.x_train = x_train
        self.mean_vec = mean_vec
        self.cov_mat = cov_mat
        self.mixture_coeff = mixture_coeff
        self.N_eff = np.zeros(self.k)
        self.vec_dim = np.size(x_train[0])
        self.total_size = int(np.size(x_train) / self.vec_dim)
    
    # gaussian with parameters as arguments of the function
    def N(self, x_vec, mean_vec, cov_mat):
        denominator = math.sqrt(2*3.1417*abs(np.linalg.det(cov_mat)))
        exponent_term = np.dot(x_vec - mean_vec, inv(cov_mat).dot(x_vec - mean_vec)) / 2
        return math.exp(-exponent_term) / denominator
        #return decimal.Decimal(-exponent_term).exp() / denominator
    
    # finds max of the three posterior probability
    def max(self, a, b, c):
        if (a > b):
           val = a
           Class = 1
        else:
           val = b
           Class = 2
        if (val < c):
            val = c
            Class = 3
            
        return Class
    
    
    # function to fit the dataset on the k gaussian clusters to get maximum value of log likelihood
    def fit(self, precision):
        
        iter_num = 0
        cost_func_f = 1000000
        cost_func_i = 0

        x = []
        y = []
        while iter_num < 3 or abs(cost_func_f - cost_func_i) > precision:
            
            cost_func_i = cost_func_f
            cost_func_f = 0
            new_mean_vec = np.zeros(shape = (self.k, self.vec_dim))
            new_cov_mat = np.zeros((self.k, self.vec_dim, self.vec_dim))
            new_mixture_coeff = np.zeros(self.k)
            self.N_eff = np.zeros(self.k)
                
            for i in range(0, self.k):
                
                cov_term1 = np.zeros(shape = (self.vec_dim, self.vec_dim))
                
                for j in range(0, self.total_size):
                    total_prob = 0
                    x_vec = self.x_train[j]
                    for k in range(0, self.k):
                        total_prob = total_prob + self.mixture_coeff[k] * self.N(x_vec, self.mean_vec[k], self.cov_mat[k])
                    
                    gamma = (self.mixture_coeff[i] * self.N(x_vec, self.mean_vec[i], self.cov_mat[i])) / total_prob
                    new_mean_vec[i] = new_mean_vec[i] + gamma * x_vec
                    cov_term1 = cov_term1 + gamma * np.outer(x_vec, x_vec)
                    self.N_eff[i] = self.N_eff[i] + gamma
                    cost_func_f = cost_func_f + math.log(total_prob)

                
                new_mean_vec[i] = new_mean_vec[i] / self.N_eff[i]
                new_cov_mat[i] = cov_term1 / self.N_eff[i] - np.outer(new_mean_vec[i], new_mean_vec[i])
                new_mixture_coeff[i] = self.N_eff[i]
            
            cost_func_f = cost_func_f / int(self.k)
            y.append(cost_func_f)
            total_N_eff = np.sum(self.N_eff)
            for i in range(0, self.k):
                self.mixture_coeff[i] = new_mixture_coeff[i] / total_N_eff
                self.mean_vec[i] = new_mean_vec[i]
                self.cov_mat[i] = new_cov_mat[i]

            iter_num = iter_num + 1
            x.append(iter_num)
            print("iteration no: %d diff: %f cost function: %f" % (iter_num, abs(cost_func_f - cost_func_i), cost_func_f))

        plt.scatter(x, y)
        plt.show()
        
    # predicts the cluster of the data point
    def ClusterPredict(self, X):
        
        pred_arr = np.array([])
        for x_vec in X:
            max_prob = -1000000
            for i in range(0, self.k):
                ln_gamma = math.log(self.mixture_coeff[i]) - np.dot(x_vec - self.mean_vec[i], inv(self.cov_mat[i]).dot(x_vec - self.mean_vec[i])) / 2 - (1 / 2) * math.log(np.linalg.det(self.cov_mat[i]))
                if (ln_gamma > max_prob):
                    max_prob = ln_gamma
                    cluster_num = i

            pred_arr = np.append(pred_arr, cluster_num)

        return pred_arr
    
    # calculate the posterior probability of the class
    def PostProb(self, x_vec):
        prob = 0
        for i in range(0, self.k):
            mean = self.mean_vec[i]
            cov_mat = self.cov_mat[i]
            mixture_coeff = self.mixture_coeff[i]
            prob = prob + mixture_coeff * self.N(x_vec, mean, cov_mat)
        
        return math.log(prob) + math.log(self.total_size)
            
    # plot the cluster plotted
    def PlotCluster(self):
        y_pred = self.ClusterPredict(self.x_train)
        plt.scatter(self.x_train[y_pred == 0, 0], self.x_train[y_pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
        plt.scatter(self.x_train[y_pred == 1, 0], self.x_train[y_pred == 1, 1], s = 20, c = 'green', label = 'Cluster 2')
        plt.scatter(self.x_train[y_pred == 2, 0], self.x_train[y_pred == 2, 1], s = 20, c = 'blue', label = 'Cluster 3')
        plt.scatter(self.mean_vec[:, 0], self.mean_vec[:, 1], s = 50, c = 'yellow', label = 'Centroids')
        plt.show()
                    
    