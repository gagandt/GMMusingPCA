import numpy as np
import random
import math
import matplotlib.pyplot as plt
import datetime

class Kmeans:
    k = 0
    x_train = np.array([])
    mean_vec = np.array([])
    z = np.array([])
    N_z = np.array([])
    vec_dim = 0
    total_size = 0
    
    def __init__ (self, k, x_train):
        
        self.k = k
        self.x_train = x_train
        self.mean_vec = np.zeros(shape = (k, np.size(x_train[0])))
        self.z = np.zeros(np.size(self.x_train))
        self.N_z = np.zeros(self.k)
        self.vec_dim = np.size(x_train[0])
        self.total_size = int(np.size(x_train) / self.vec_dim)
        
        
    # euclidean distance measure
    def DistMeasure(self, x1, x2):
        
        return math.sqrt(np.dot(x1 - x2, x1 - x2))
    
    # function returns cluster number of the data point accor. to Euc. Distance
    def MinDistCluster(self, x_vec, mean_vec):
        
        min_dist = self.DistMeasure(x_vec, mean_vec[0])
        cluster_num = 0
        for i in range(0, self.k):
            value = self.DistMeasure(x_vec, mean_vec[i])
            if (value <= min_dist):
                min_dist = value
                cluster_num = i
                
        return cluster_num
    
    # fit the model on the training dataset and return optimal k means distribution
    """def fit(self, iterations, precision):
        
        dataset = self.x_train
        np.random.shuffle(dataset)
        num = int(self.total_size / self.k)
        for i in range(0, self.k):
            self.mean_vec[i] = dataset[i * num + int(num / 2)]
        
        iter_num = 0
        cost_func_i = 1000000
        cost_func_f = 0
        while cost_func_i - cost_func_f > precision:
            a = datetime.datetime.now()
            self.N_z = np.zeros(self.k)
            new_mean_vec = np.zeros(shape = (self.k, self.vec_dim))
            cost_func_i = cost_func_f
            
            for i in range(0, self.total_size):
                cluster_num = self.MinDistCluster(self.x_train[i], self.mean_vec)
                new_mean_vec[cluster_num] = new_mean_vec[cluster_num] + self.x_train[i]
                self.z[i] = cluster_num
                self.N_z[cluster_num] = self.N_z[cluster_num] + 1
                
            cost_term1_f = 0
            cost_term2_f = 0
                
            for i in range(0, self.k):
                if (self.N_z[i] == 0):
                    new_mean_vec[i] = new_mean_vec[i]
                else:
                    new_mean_vec[i] = (1 / self.N_z[i]) * new_mean_vec[i]
                cost_term1_f = cost_term1_f + np.dot(new_mean_vec[i], new_mean_vec[i]) * self.N_z[i]

            for i in range(0, self.total_size):
                cost_term2_f = cost_term2_f + np.dot(self.x_train[i], new_mean_vec[int(self.z[i])])

            cost_func_f = cost_term1_f - 2 * cost_term2_f
            
            self.mean_vec = new_mean_vec
            iter_num = iter_num + 1
            print("iternation no: %d diff: %d" % (iter_num, cost_func_i - cost_func_f))
            b = datetime.datetime.now()
            #print(b - a)
            
        print(self.N_z)
        print(self.mean_vec)
        print(cost_func_i - cost_func_f)
    """
    
    def fit(self, iterations, precision):
        x = []
        y = []
        dataset = self.x_train
        np.random.shuffle(dataset)
        num = int(self.total_size / self.k)
        for i in range(0, self.k):
            self.mean_vec[i] = dataset[i * num + int(num / 2)]
        
        cost_func_i = 0
        for i in range(0, self.total_size):
            cluster_num = self.MinDistCluster(self.x_train[i], self.mean_vec)
            self.z[i] = cluster_num
            self.N_z[cluster_num] = self.N_z[cluster_num] + 1
            cost_func_i = cost_func_i + np.dot(self.x_train[i] - self.mean_vec[cluster_num], self.x_train[i] - self.mean_vec[cluster_num])

        iter_num = 0
        cost_func_f = 0
        while iter_num < 3 or abs(cost_func_i - cost_func_f) > precision:
            cost_func_i = cost_func_f
            self.N_z = np.zeros(self.k)
            new_mean_vec = np.zeros(shape = (self.k, self.vec_dim))
            
            cost_func_f = 0
            for i in range(0, self.total_size):
                cost_func_f = cost_func_f + np.dot(self.x_train[i] - self.mean_vec[int(self.z[i])], self.x_train[i] - self.mean_vec[int(self.z[i])])
                cluster_num = self.MinDistCluster(self.x_train[i], self.mean_vec)
                new_mean_vec[cluster_num] = new_mean_vec[cluster_num] + self.x_train[i]
                self.z[i] = cluster_num
                self.N_z[cluster_num] = self.N_z[cluster_num] + 1
                
                
            for i in range(0, self.k):
                if (self.N_z[i] == 0):
                    new_mean_vec[i] = new_mean_vec[i]
                else:
                    new_mean_vec[i] = (1 / self.N_z[i]) * new_mean_vec[i]
                
            y.append(cost_func_f)
            self.mean_vec = new_mean_vec
            iter_num = iter_num + 1
            x.append(iter_num)
            print("iternation no: %d diff: %d %f" % (iter_num, cost_func_i - cost_func_f, cost_func_f))
            
        plt.scatter(x, y)
        plt.show()
    
    # returns the covariance matrix of k clusters
    def CovMatrix(self):
        cov_mat_array = np.zeros((self.k, self.vec_dim, self.vec_dim))
        for i in range(0, self.total_size):
            cov_mat_array[int(self.z[i])] = cov_mat_array[int(self.z[i])] + np.outer(self.x_train[i] - self.mean_vec[int(self.z[i])], self.x_train[i] - self.mean_vec[int(self.z[i])])
        
        for i in range(0, self.k):
            cov_mat_array[i] = (1 / self.N_z[i]) * cov_mat_array[i]
        
        return cov_mat_array
    
    # returns initial mixture coefficient for GMM model
    def MixtureCoeff(self):
        
        total_num = np.sum(self.N_z)
        return self.N_z / total_num
        
    # predict the cluster of data points after training has occured
    def ClusterPredict(self, X):
        pred_arr = np.array([])
        for x_vec in X:
            pred_arr = np.append(pred_arr, self.MinDistCluster(x_vec, self.mean_vec))

        return pred_arr
    
    # returns the BoVW representation of image patches
    def BoVW(self, mean_vec, img_vec):
        #bag = np.zeros(self.k)
        bag = []
        for i in range(0, self.k):
            bag.append(0)
            
        for vec in img_vec:
            cluster_num = self.MinDistCluster(vec, mean_vec)
            bag[cluster_num] = bag[cluster_num] + 1
            
        return bag
        
    # plot the cluster for 2-D dimensional data points
    def PlotCluster(self):
        y_pred = self.ClusterPredict(self.x_train)
        plt.scatter(self.x_train[y_pred == 0, 0], self.x_train[y_pred == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
        plt.scatter(self.x_train[y_pred == 1, 0], self.x_train[y_pred == 1, 1], s = 20, c = 'green', label = 'Cluster 2')
        plt.scatter(self.x_train[y_pred == 2, 0], self.x_train[y_pred == 2, 1], s = 20, c = 'blue', label = 'Cluster 3')
        plt.scatter(self.mean_vec[:, 0], self.mean_vec[:, 1], s = 50, c = 'yellow', label = 'Centroids')
        plt.show()