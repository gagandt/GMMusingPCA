import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as ListedColormap
from numpy.linalg import inv
from KMeans import Kmeans
from GMM import GMM
from sklearn.mixture import GaussianMixture

class Fbayes(GaussianMixture):
    
    def __init__ (self, k, class_1, class_2, x1_train, y1_train, x2_train, y2_train, total_size):
        self.k = k
        self.class_1 = class_1
        self.class_2 = class_2
        self.x1_train = x1_train
        self.x2_train = x2_train
        self.classfier_1 = GaussianMixture(k, tol = 0.002, covariance_type = 'full')
        self.classfier_2 = GaussianMixture(k, tol = 0.002, covariance_type = 'full')
        self.P_c1 = np.size(y1_train) / total_size
        self.P_c2 = np.size(y2_train) / total_size
        self.total_size = total_size
    
    
    def max(self, a, b):
        if (a > b):
            return self.class_1
        else:
            return self.class_2

    def fit(self):
        self.classfier_1.fit(self.x1_train)
        self.classfier_2.fit(self.x2_train)
    
    def ClassPredict(self, X):
        class_list = []
        for x_vec in X:
            ln_1 = self.classfier_1.score(x_vec.reshape(1, -1)) + math.log(self.P_c1)
            ln_2 = self.classfier_2.score(x_vec.reshape(1, -1)) + math.log(self.P_c2)
            
            Class = self.max(ln_1, ln_2)
            class_list.append(Class)
        
        return np.array(class_list)
            
