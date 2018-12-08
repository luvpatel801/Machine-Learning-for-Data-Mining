import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import os
import scipy.io as io

from sklearn.decomposition import PCA
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, concatenate
from keras.utils import np_utils
from keras.initializers import RandomNormal

# Data Loading
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("X_train dimensions: ", X_train.shape)
print("y_train dimensions: ", y_train.shape)
print("X_test dimensions: ", X_test.shape)
print("y_test dimensions: ", y_test.shape)

# Data Reshaping
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print("Train matrix dimensions: ", X_train.shape)
print("Test matrix dimensions: ", X_test.shape)

# Data Normalization
X_train /= 255
X_test /= 255

# Converting Y into one-hot vector
n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Y train dimensions: ", Y_train.shape)
print("Y test dimensions: ", Y_test.shape)

from math import *

n_components = 0.95

pca = PCA(n_components=n_components)
X_pca_train = pca.fit_transform(X_train)
X_pca_test = pca.transform(X_test)

pca.fit(X_pca_train)

pi = np.ones((n_classes,))
covM = pca.get_covariance()
mu = np.ones((n_classes,X_pca_train.shape[1]))
print("Pi Shape: ", pi.shape)
print("Mu Shape: ", mu.shape)
print("Covariance Matrix Shape: ", covM.shape)

def calPi(X, Y):
  total = np.zeros((n_classes,))
  for i in range(n_classes):
    for j in range(Y.shape[0]):
      if(Y[j][i]==1):
        total[i]+=1
    p = total/Y.shape[0]
  return p

def calMu(X, Y):
  count = np.zeros((n_classes))
  total = np.zeros((n_classes,X.shape[1]))
  for i in range(n_classes):
    for j in range(Y.shape[0]):
      if(Y[j][i]==1):
        total[i]+=X[j]
        count[i]+=1
    total[i] = total[i]/count[i]
  m = total
  return m

def deltaLDA(x, k):
  x = np.transpose(x)
  p = log(pi[k]) - (0.5)*(np.matmul(np.matmul(np.transpose(x),np.linalg.inv(covM)),x) + np.matmul(np.matmul(np.transpose(mu[k]),np.linalg.inv(covM)),mu[k])) + np.matmul(np.matmul(np.transpose(x),np.linalg.inv(covM)),mu[k])
  return p

def classifyLDA(x):
  clas = 0
  delLDA = deltaLDA(x,0)
  for i in range(n_classes):
    if delLDA < deltaLDA(x,i):
      clas = i
  delLDA = deltaLDA(x,i)
  return clas

def accuracyCal(x, y):
  accCount = 0
  for i in range(x.shape[0]):
    if(y[i][classifyLDA(x[i])]==1):
      accCount+=1
  return (accCount/(x.shape[0]))

pi = calPi(X_pca_train, Y_train)
mu = calMu(X_pca_train, Y_train)
#print(mu)
print("accuracy" , accuracyCal(X_pca_test,Y_test))

