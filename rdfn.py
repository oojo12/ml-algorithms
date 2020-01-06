""" Implementation of radial basis neural network. It is programmed to have 3
layers and a sigmoid activation function with batch gradient descent"""

import numpy as np
import pandas as pd
from math import log,floor

__author__ = "Femi"
__version__ = "1"
__status__ = "starting"

class rbf():
    def __init__(self,x_train,y_train,x_test,eta,epochs,num_prototypes):
        ## initialize parameters
        self.features = x_train
        centroids, assignment = k_means(x_train,num_prototypes)
        self.cetroids = centroids
        self.spread = centroids.std(axis = 1)
        self.mean = centroids.mean()
        self.features.reset_index(inplace = True, drop = True)
        self.targets = y_train
        self.targets.reset_index(inplace = True, drop = True)
        self.eta = eta
        self.epochs = epochs
        self.test = x_test
        radial = []
        for row in self.spread.index:
            radial.append(np.exp(np.linalg.norm(self.features - self.mean,axis =1)**2/(-2*(self.spread[row])**2)))

        ## add bias layer
        self.radial = pd.DataFrame(radial).T
        self.radial['bias'] = 1
        self.weights = np.random.uniform(low = -.01, high = .01,size = (num_prototypes+1,len(y_train.columns)))

    ## train with batch gradient descent
    def fit(self):
        for epoch in range(self.epochs):
            output = self.radial.dot(self.weights)
            output = pd.DataFrame(output)

            self.output = output
            self.error = self.targets - self.output.values
            self.adjust = self.eta*self.error.T.dot(self.radial[self.radial.columns[0:-1]])/len(self.error)
            self.weights[0:-1]+=self.adjust.T
            self.weights[-1] += self.eta*(self.error.sum())/len(self.error)


    def pred_sig(self,outputs):
        temp = outputs -outputs.max()
        pos = np.exp(temp)/(1+np.exp(temp))
        neg = 1/(1+np.exp(temp))
        pos = pd.DataFrame(pos)
        pos.columns = ['1']
        neg = pd.DataFrame(neg)
        return pos.join(neg)

    ## activation function
    def sigmoid(self,outputs):
        temp = outputs -outputs.max()
        if len(self.targets.columns) >1:
            return (np.exp(temp).T.divide(np.exp(temp).sum(axis = 1))).T
        else:
            pos = np.exp(temp)/(1+np.exp(temp))
            neg = 1/(1+np.exp(temp))
            return pd.DataFrame(pos)

    ## make predictions
    def predict(self):
        radial = []
        ## rebuild radial basis values
        for row in self.spread.index:
            radial.append(np.exp(np.linalg.norm(self.test - self.mean,axis =1)**2/(-2*(self.spread[row])**2)))

        ## add bias layer
        self.radial = pd.DataFrame(radial).T
        self.radial['bias'] = 1
        output = self.radial.dot(self.weights)
        if len(self.targets.columns) >1:
            output = self.sigmoid(output)
            output.columns = self.targets.columns
            self.predictions = output.idxmax(axis=1)
        else:
            self.output = self.pred_sig(output)
            self.output.columns = [0,1]
            predictions = self.output.idxmax(axis=1)
            self.predictions = predictions
        return self.predictions
