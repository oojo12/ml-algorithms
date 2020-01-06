import numpy as np

"""This is an implementation of the k nearest neighbors algorithm for a classification task

    Parameters:
    train_data - numeric pandas data frame of training data
    targets - numpy array, pandas series, or pandas dataframe
    test_data - numeric pandas data frame of testing data
    k - number of nearest neighbors to consider

    output:
    pandas dataframe of class predictions
"""
__author__ = "Femi"
__version__ = "1"
__status__ = "Developing"

class knn:


    def __init__(self,train_data,targets,k,test_data):
        self.data = train_data.copy()
        self.targets = targets.copy()
        self.k = k
        self.assigned_class = []
        self.test_data = test_data.copy()


    def find_distance(self,point_1):
        distance = np.linalg.norm(self.test_data.loc[point_1] - self.data, ord = 2, axis =1)
        return distance

    def find_nearest_neighbors(self,distances):
        ### Flatten distances so argmin corresponds to precise index location
        distances = distances.flatten()
        count = 0
        class_votes = []
        while count < self.k:
            min_point = np.argmin(distances)
            distances = np.delete(distances, min_point)
            ## Record votes
            class_votes.append(self.targets.loc[min_point])
            count+=1
        self.assigned_class.append(pd.Series(class_votes).value_counts().index[0])

    def predict(self):
        for point_1 in self.test_data.index:
            distance = self.find_distance(point_1)
            self.find_nearest_neighbors(distance)
        return pd.DataFrame(self.assigned_class)
