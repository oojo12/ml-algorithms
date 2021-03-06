import pandas as pd
import numpy as np

"""Implementation of the classic k-means algorithm"""

__author__ = "Femi"
__version__ = "1"
__status__ = "Developing"

class k_means():
    def __init__(self,num_clusters,max_iter,train_data,random_state,test_data=None):
        assert (type(np.array(1)) ==  type(train_data)) or (type(pd.DataFrame()) == type(train_data)),\
        "not a pandas data frame or a numpy array"

        if isinstance(data,type(pd.DataFrame())):
            df = pd.DataFrame({'float': [1.0],
                   'int': [1],
                   'datetime': [pd.Timestamp('20180310')],
                   'string': ['foo']})
            assert (data.dtypes != df.dtypes[2]).all() or (data.dtypes != df.dtypes[3]).all(),\
            """
                dataframe contains datetime or string types. This implementation only supports
                float or int data types.
            """
        else:
            """fill with appropriate test for numpy array."""
            pass
        self.data = train_data
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.seed = random_state
        self.init_centroids()
        self.test_data = self.test_data

    # Initialize centroids
    def init_centroids(self):
        if isinstance(self.data,type(pd.DataFrame())):
            self.centroids = self.data.sample(n=self.num_clusters, random_state=self.seed)
            self.centroids.reset_index(drop=True, inplace=True)
        else:
            np.random.RandomState(seed=self.seed)
            np.random.shuffle(self.centroids)
            self.centroids = self.data[0:num_clusters]

    # Calculate distances
    def calc_dist(self):
        if isinstance(self.data, type(pd.DataFrame())):
            ##### Find clusters for each data point within feature
            for centroid in self.centroids.index:
                if centroid == self.centroids.index[0]:
                    distance = np.linalg.norm(self.data - self.centroids.loc[centroid], axis = 1, ord = 2)
                else:
                    distance = np.vstack((distance,np.linalg.norm(self.data - self.centroids.loc[centroid], axis = 1, ord = 2)))
            self.distance = distance

    def update_centroids(self):
        if isinstance(self.data,type(pd.DataFrame())):
            count = -1
            for label in self.data['ASSIGNMENT'].unique():
                count+=1
                self.centroids.loc[count] = self.data[self.data['ASSIGNMENT'] == label].mean()
        else:
            """Update this block to allow functionallity with numpy arrays"""

    # Assign centroids to data
    def assign(self,semi_predict=False):
        if isinstance(self.data,type(pd.DataFrame())):
            if not semi_predict:
                self.data['ASSIGNMENT'] =  np.argmin(self.distance, axis = 0)
            else:
                self.test_data['ASSIGNMENT'] =  np.argmin(self.distance, axis = 0)
        else:
            """Update this block to allow functionallity with numpy arrays"""

    def fit(self):
        for iteration in range(self.max_iter):
            self.calc_dist()
            self.assign()
            self.update_centroids()
            if iteration != 0:
                if np.equal(self.prev_labels, self.data['ASSIGNMENT']).all():
                    print("Finished in %s iterations"%iteration)
                    break

            self.prev_labels = self.data['ASSIGNMENT'].copy()


    def predict(self):
        self.assign()
        return self.data['ASSIGNMENT']

    def semi_predict(self):
        assert (self.test_date != None),\
        "No test data passed as argument"
        self.calc_dist()
        self.assign()
        return self.test_data['ASSIGNMENT']
