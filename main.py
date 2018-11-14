import pandas
import numpy as np
import matplotlib.pyplot as plt
import math
from PCA import PCA
from FisherLDA import Fisher
from GMM import GMM
from KMeans import Kmeans
from BayesianGM import BayesianGM
from fisher_bayes import Fbayes


x1_train = np.loadtxt('BoVW/x1_train.txt')
x2_train = np.loadtxt('BoVW/x2_train.txt')
x3_train = np.loadtxt('BoVW/x3_train.txt')

x1_test = np.loadtxt('BoVW/x1_test.txt')
x2_test = np.loadtxt('BoVW/x2_test.txt')
x3_test = np.loadtxt('BoVW/x3_test.txt')

total_size = 150

y1_train = np.full(50, 1)
y2_train = np.full(50, 2)
y3_train = np.full(50, 3)

y1_test = np.full(50, 1)
y2_test = np.full(50, 2)
y3_test = np.full(50, 3)


dataset = np.concatenate((x1_train, x2_train, x3_train), axis = 0)

# number of reduced dimensions 
l = 20
d = 32
pca = PCA(l, d, dataset, total_size)
pca.fit()

k = 1

total_size_12 = 100
total_size_23 = 100
total_size_31 = 100

flda_1 = Fisher(d, x1_train, x2_train)
flda_1.fit()

# fisher representation of the class 1 and class 2
x1_train_f = flda_1.F_transform(x1_train)
x2_train_f = flda_1.F_transform(x2_train)

x1_test_f = flda_1.F_transform(x1_test)
x2_test_f = flda_1.F_transform(x2_test)

fbayes = Fbayes(k, 1, 2, x1_train_f, x2_train_f, y1_train, y2_train, total_size_12)
fbayes.fit()
x_pred = np.concatenate((x1_test_f, x2_test_f), axis = 0)
y_pred_12 = fbayes.ClassPredict(x_pred)


flda_2 = Fisher(d, x2_train, x3_train)
flda_2.fit()
x2_train_f = flda_2.F_transform(x2_train)
x3_train_f = flda_2.F_transform(x3_train)

x2_test_f = flda_2.F_transform(x2_test)
x3_test_f = flda_2.F_transform(x3_test)

fbayes = Fbayes(k, 2, 3, x2_train_f, x3_train_f, y2_train, y3_train, total_size_23)
fbayes.fit()
x_pred = np.concatenate((x2_test_f, x3_test_f), axis = 0)
y_pred_23 = fbayes.ClassPredict(x_pred)

flda_3 = Fisher(d, x3_train, x1_train)
flda_3.fit()
x3_train_f = flda_3.F_transform(x3_train)
x1_train_f = flda_3.F_transform(x1_train)

x3_test_f = flda_3.F_transform(x3_test)
x1_test_f = flda_1.F_transform(x1_test)

fbayes = Fbayes(k, 3, 1, x3_train_f, x1_train_f, y3_train, y1_train, total_size_31)
fbayes.fit()
x_pred = np.concatenate((x3_test_f, x1_test_f), axis = 0)
y_pred_31 = fbayes.ClassPredict(x_pred)

x1_train_r = pca.transform(x1_train)
x2_train_r = pca.transform(x2_train)
x3_train_r = pca.transform(x3_train)

x1_test_r = pca.transform(x1_test)
x2_test_r = pca.transform(x2_test)
x3_test_r = pca.transform(x3_test)

# only for 2-d dataset-1
plt.scatter(x1_train_r[:, 0], x1_train_r[:, 1], color = 'r')
plt.scatter(x2_train_r[:, 0], x2_train_r[:, 1], color = 'g')
plt.scatter(x3_train_r[:, 0], x3_train_r[:, 1], color = 'b')
plt.show()

#gmm = BayesianGM(2, 0.02, x1_train_r, x2_train_r, x3_train_r, y1_train, y2_train, y3_train, total_size)
#gmm.fit(0.02)
from Bayes import Bayes
k = 8
bayes = Bayes(k, x1_train_r, y1_train, x2_train_r, y2_train, x3_train_r, y3_train, total_size)
bayes.fit()

from sklearn.metrics import confusion_matrix

y_true = np.concatenate((y1_test, y2_test, y3_test), axis = 0)
y_pred = np.concatenate((bayes.ClassPredict(x1_test_r), bayes.ClassPredict(x2_test_r), bayes.ClassPredict(x3_test_r)), axis = 0)

M = confusion_matrix(y_true, y_pred)
print(M)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))

# apply now the classification from this point