import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as ListedColormap
from numpy.linalg import inv
from KMeans import Kmeans
from GMM import GMM

"""
This class deals with all the calculations required for all the three classes 
combined (i.e calculating decision boundary, confusion matrix, and other performance
parameters)

"""

class BayesianGM(Kmeans, GMM):
    
    def __init__(self, k, precision, xi_train, xj_train, xk_train, yi_train, yj_train, yk_train, total_size):
        self.k = k
        self.xi_train = xi_train
        self.yi_train = yi_train
        self.kmeans_obj_i = Kmeans(k, xi_train)
        self.kmeans_obj_i.fit(100, precision)
        
        means = self.kmeans_obj_i.mean_vec
        cov_mat_list = self.kmeans_obj_i.CovMatrix()
        mixture_coeff = self.kmeans_obj_i.MixtureCoeff()
        
        self.GMM_obj_i = GMM(k, xi_train, means, cov_mat_list, mixture_coeff)
        
        self.xj_train = xj_train
        self.yj_train = yj_train
        self.kmeans_obj_j = Kmeans(k, xj_train)
        self.kmeans_obj_j.fit(100, precision)
        
        means = self.kmeans_obj_j.mean_vec
        cov_mat_list = self.kmeans_obj_j.CovMatrix()
        mixture_coeff = self.kmeans_obj_j.MixtureCoeff()
        
        self.GMM_obj_j = GMM(k, xj_train, means, cov_mat_list, mixture_coeff)
        
        self.xk_train = xk_train
        self.yk_train = yk_train
        self.kmeans_obj_k = Kmeans(k, xk_train)
        self.kmeans_obj_k.fit(100, precision)
        
        means = self.kmeans_obj_k.mean_vec
        cov_mat_list = self.kmeans_obj_k.CovMatrix()
        mixture_coeff = self.kmeans_obj_k.MixtureCoeff()
        
        self.GMM_obj_k = GMM(k, xk_train, means, cov_mat_list, mixture_coeff)
        
        self.P_ci = np.size(yi_train) / total_size
        self.P_cj = np.size(yj_train) / total_size
        self.P_ck = np.size(yk_train) / total_size
        self.total_size = total_size
        
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
    
    
    def N(self, x_vec, mean_vec, cov_mat):
        denominator = math.sqrt(2*3.1417*abs(np.linalg.det(cov_mat)))
        exponent_term = np.dot(x_vec - mean_vec, inv(cov_mat).dot(x_vec - mean_vec)) / 2
        return math.exp(-exponent_term) / denominator


    def fit(self, precision):
        
        self.GMM_obj_i.fit(precision)
        self.GMM_obj_j.fit(precision)
        self.GMM_obj_k.fit(precision)
    
    
    def ClassPredict(self, X):
        
        class_list = []
        
        for x_vec in X:
            prob_i = 0
            prob_j = 0
            prob_k = 0
            
            for i in range(0, self.k):
                mean_i = self.GMM_obj_i.mean_vec[i]
                cov_mat_i = self.GMM_obj_i.cov_mat[i]
                mixture_coeff_i = self.GMM_obj_i.mixture_coeff[i]
                
                prob_i = prob_i + mixture_coeff_i * self.N(x_vec, mean_i, cov_mat_i)
                print(prob_i)
                
                mean_j = self.GMM_obj_j.mean_vec[i]
                cov_mat_j = self.GMM_obj_j.cov_mat[i]
                mixture_coeff_j = self.GMM_obj_j.mixture_coeff[i]
                
                prob_j = prob_j + mixture_coeff_j * self.N(x_vec, mean_j, cov_mat_j)
                print(prob_j)
                
                mean_k = self.GMM_obj_k.mean_vec[i]
                cov_mat_k = self.GMM_obj_k.cov_mat[i]
                mixture_coeff_k = self.GMM_obj_k.mixture_coeff[i]
                
                prob_k = prob_k + mixture_coeff_k * self.N(x_vec, mean_k, cov_mat_k)
                print(prob_k)
                
            ln_i = math.log(prob_i) + math.log(self.GMM_obj_i.total_size)
            ln_j = math.log(prob_j) + math.log(self.GMM_obj_j.total_size)
            ln_k = math.log(prob_k) + math.log(self.GMM_obj_k.total_size)
            
            Class = self.max(ln_i, ln_j, ln_k)
            class_list.append(Class)
     
        return np.array(class_list)

        
        
        
        
     
        
        


       