
# coding: utf-8

# In[1]:


from sklearn import tree
from sklearn.svm import SVC 
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

import numpy as np


# In[2]:


# Data and labels
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']



# In[3]:


# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier() #Decision Tree
#clf_svm = SVC() 
clf_svm = SVC(kernel="linear") # Linear SVM 
clf_svm_RBF = SVC(gamma=2, C=1) # RBF SVM
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier() # Nearest Neighbors
clf_GPC =  GaussianProcessClassifier() #Gaussian Process
clf_RFC =  RandomForestClassifier() # Random Forest
clf_NN = MLPClassifier(alpha=1) # Neural Net
clf_ADA = AdaBoostClassifier() # AdaBoost
clf_GNB = GaussianNB() # Naive Bayes
clf_QDA = QuadraticDiscriminantAnalysis() # QDA


# In[4]:


# CHALLENGE - ...and train them on our data
# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)
clf_svm_RBF.fit(X, Y)
clf_GPC.fit(X, Y)
clf_RFC.fit(X, Y)
clf_NN.fit(X, Y)
clf_ADA.fit(X, Y)
clf_GNB.fit(X, Y)
clf_QDA.fit(X, Y)

#prediction = clf.predict([[190, 70, 43]])


# In[5]:


# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))


# In[6]:


# Linear SVM 
pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))


# In[7]:


# perceptron
pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))


# In[8]:


# Nearest Neighbors
pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))


# In[9]:


# RBF SVM
pred_svm_RBF = clf_svm_RBF.predict(X)
acc_svm_RBF = accuracy_score(Y, pred_svm_RBF) * 100
print('Accuracy for RBF SVM: {}'.format(acc_svm_RBF))


# In[10]:


# Gaussian Process
pred_GPC = clf_GPC.predict(X)
acc_GPC = accuracy_score(Y, pred_GPC) * 100
print('Accuracy for GPC: {}'.format(acc_GPC))


# In[11]:


# Random Forest
pred_RFC = clf_RFC.predict(X)
acc_RFC = accuracy_score(Y, pred_RFC) * 100
print('Accuracy for RFC: {}'.format(acc_RFC))


# In[12]:


# Neural Net
pred_NN = clf_NN.predict(X)
acc_NN = accuracy_score(Y, pred_NN) * 100
print('Accuracy for NN: {}'.format(acc_NN))


# In[13]:


# AdaBoost
pred_ADA = clf_ADA.predict(X)
acc_ADA = accuracy_score(Y, pred_ADA) * 100
print('Accuracy for ADA: {}'.format(acc_ADA))


# In[14]:


# Naive Bayes
pred_GNB = clf_GNB.predict(X)
acc_GNB = accuracy_score(Y, pred_GNB) * 100
print('Accuracy for GNB: {}'.format(acc_GNB))


# In[15]:


# QDA
pred_QDA = clf_QDA.predict(X)
acc_QDA = accuracy_score(Y, pred_QDA) * 100
print('Accuracy for QDA: {}'.format(acc_QDA))


# In[16]:


# CHALLENGE compare their reusults and print the best one!
#print(prediction)

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN]) # only compared for 3 classifier
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))


# In[ ]:




