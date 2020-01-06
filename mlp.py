import pandas as pd
import numpy as np

__author__ = "Femi"
__version__ = "1"
__status__ = "starting"

class mlp():
    """Implementation of the autonomous neural network algorithm"""
    import pandas as pd
    import numpy as np
    def __init__(self,data,layers):
        self.data = data
        self.layers = layers

    def softmax(self):
        np.exp()

    def relu(self):


     def sigmoid(self,outputs):
        temp = outputs -outputs.max()
        if len(self.targets.columns) >1:
            return (np.exp(temp).T.divide(np.exp(temp).sum(axis = 1))).T
        else:
            pos = np.exp(temp)/(1+np.exp(temp))
            neg = 1/(1+np.exp(temp))
            return pd.DataFrame(pos)
